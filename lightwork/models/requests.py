"""Pure request builder (D2: requests are derived). The reducer calls it to emit ``model_request`` intents; the live
client sends exactly what it returns, with each ``{"$frame": sha}`` swapped for a data URL of those bytes.

Meta backend: Responses API for Spark, Wonder and SAM. Chat backend: Chat Completions for Glimmer on llama.cpp and the
D15 dev stand-ins; the JSON schema goes into the system prompt (no ``response_format``: SP2 test G3 showed llama.cpp
does not enforce it for Glimmer), and the reply is validated by ``parse`` with one repair retry.

``ctx`` per role:
- verdict: ``{step_id, instruction, question, expected?}`` (C9 ``VerdictCtx``) + a keyframe.
- ground: ``{nouns: [str], shape: point|box|count}`` + a keyframe.
- why: ``{step_id, instruction, detail, last_reason?}`` (C9 ``WhyCtx``), no frame.
- segment: ``{concept}`` or ``{object_prompts: [{id, box?: [x0,y0,x1,y1] half-open, points?: [[x,y]],
  negative?: [[x,y]]}], w, h}`` (sent-image px) + a keyframe; Meta backend only.
- author: ``{text, notes?}``, or ``{filename, notes?}`` with ``frame_sha`` = sha of the PDF (Meta backend only).
- any role but segment, for its one retry: ``ctx["retry"] = {"reason": "invalid", "previous": <reply text>,
  "error": <parse error>}`` (adds the reply and a repair turn) or ``{"reason": "length"}`` (doubles the output budget).
"""

from __future__ import annotations

import json
from typing import Any

from meta_sam_parser import FramePrompt, PromptBox, PromptObject, PromptPoint, build_object_prompt
from pydantic import ValidationError

from lightwork.contracts.backend import VerdictCtx, WhyCtx
from lightwork.contracts.calls import frame_ref

from .prompts import REPAIR, author, ground, segment, verdict, why
from .wire import CHAT, RESPONSES

ROLES = ("verdict", "ground", "why", "segment", "author")
BACKENDS = ("meta", "chat")

PROMPT_VERSIONS: dict[str, str] = {
    "verdict": verdict.VERSION,
    "ground": ground.VERSION,
    "why": why.VERSION,
    "segment": segment.VERSION,
    "author": author.VERSION,
}

# Meta backend models per role (D4).
META_MODELS: dict[str, str] = {
    "verdict": "muse-spark-1.3-contributor",
    "ground": "muse-spark-1.3-contributor",
    "why": "muse-spark-1.3-contributor",
    "author": "muse-wonder-1.0",
    "segment": "sam-3.1",
}
META_EFFORT = {"verdict": "minimal", "ground": "minimal", "why": "minimal", "author": "low"}
META_MAX_OUTPUT = {"verdict": 4000, "ground": 4000, "why": 4000, "author": 32000}
CHAT_MAX_TOKENS = {"verdict": 2048, "ground": 4096, "why": 2048, "author": 12288}

# Glimmer on llama.cpp (briefs/glimmer.md §1): the model card's sampling and the lowest reasoning strength. Stand-ins
# get no sampling fields (briefs/standins.md: temperature 0 hits omniroute's cache, some models reject it).
GLIMMER_FIELDS: dict[str, Any] = {
    "chat_template_kwargs": {"reasoning_strength": "low"},
    "temperature": 1.0,
    "top_p": 0.95,
    "top_k": 64,
    "min_p": 0.0,
}


def is_glimmer(model: str) -> bool:
    return "glimmer" in model.lower()


def backend_models(lw_backend: str, chat_model: str | None) -> tuple[str, dict[str, str]]:
    """``LW_BACKEND`` (meta | glimmer) -> (C9 backend name, role -> model id). The chat backend has no SAM."""
    if lw_backend == "meta":
        return "meta", dict(META_MODELS)
    if lw_backend == "glimmer":
        if not chat_model:
            raise ValueError("LW_BACKEND=glimmer needs LW_CHAT_MODEL")
        return "chat", {role: chat_model for role in ("verdict", "ground", "why", "author")}
    raise ValueError(f"unknown LW_BACKEND {lw_backend!r} (meta | glimmer)")


def build_request(role: str, *, model: str, backend: str, frame_sha: str | None, ctx: dict) -> tuple[str, dict]:
    """(endpoint, canonical request with media as ``{"$frame": sha}``). Raises ``ValueError`` on bad input."""
    if role not in ROLES:
        raise ValueError(f"unknown role {role!r}")
    if backend not in BACKENDS:
        raise ValueError(f"unknown backend {backend!r} (meta | chat)")
    if role == "segment":
        if backend != "meta":
            raise ValueError("segment needs the meta backend (SAM)")
        if "retry" in ctx:
            raise ValueError("segment requests have no retry variant")
        return RESPONSES, _segment(model, _need_frame(role, frame_sha), ctx)
    system, parts, schema = _task(role, backend, frame_sha, ctx)
    retry = ctx.get("retry")
    turns: list[tuple[str, Any]] = [("user", parts)]
    factor = 1
    if retry is not None:
        reason = retry.get("reason") if isinstance(retry, dict) else None
        if reason == "invalid":
            previous, error = retry.get("previous"), retry.get("error")
            if not isinstance(previous, str) or not isinstance(error, str):
                raise ValueError('retry "invalid" needs str "previous" and "error"')
            turns += [("assistant", previous), ("user", [("text", REPAIR.format(error=error))])]
        elif reason == "length":
            factor = 2
        else:
            raise ValueError(f"unknown retry {retry!r}")
    if backend == "meta":
        return RESPONSES, _responses(model, role, system, turns, schema, factor)
    return CHAT, _chat(model, role, system, turns, schema, factor)


# ---------------------------------------------------------------- tasks (backend-neutral)

# A part is ("text", str) | ("image", sha) | ("file", sha, filename).


def _task(role: str, backend: str, frame_sha: str | None, ctx: dict) -> tuple[str | None, list[tuple], tuple | None]:
    """(system text, user parts, (schema name, schema) or None)."""
    if role == "verdict":
        vctx = _model(VerdictCtx, ctx, ("step_id", "instruction", "question", "expected"))
        parts = [("image", _need_frame(role, frame_sha)), ("text", verdict.user_text(vctx))]
        return verdict.INSTRUCTIONS, parts, (verdict.NAME, verdict.SCHEMA)
    if role == "ground":
        nouns, shape = ctx.get("nouns"), ctx.get("shape")
        if not isinstance(nouns, list) or not nouns or not all(isinstance(n, str) and n.strip() for n in nouns):
            raise ValueError("ground needs a non-empty list of nouns")
        if shape not in ground.SHAPES:
            raise ValueError(f"ground shape must be one of {ground.SHAPES}")
        parts = [("image", _need_frame(role, frame_sha)), ("text", ground.user_text([n.strip() for n in nouns], shape))]
        return None, parts, ground.SCHEMAS[shape]
    if role == "why":
        if frame_sha is not None:
            raise ValueError("why takes no frame")
        wctx = _model(WhyCtx, ctx, ("step_id", "instruction", "detail", "last_reason"))
        return why.INSTRUCTIONS, [("text", why.user_text(wctx))], None
    notes = ctx.get("notes", "")
    if not isinstance(notes, str):
        raise ValueError("author notes must be a string")
    if frame_sha is None:
        text = ctx.get("text")
        if not isinstance(text, str) or not text.strip():
            raise ValueError("author needs a source text or a PDF")
        return author.INSTRUCTIONS, [("text", author.user_text(text, notes))], (author.NAME, author.SCHEMA)
    if backend != "meta":
        raise ValueError("PDF sources need the meta backend; pass extracted text to the chat backend")
    filename = ctx.get("filename")
    if not isinstance(filename, str) or not filename:
        raise ValueError("author from a PDF needs a filename")
    parts = [("file", frame_sha, filename), ("text", author.user_text(None, notes))]
    return author.INSTRUCTIONS, parts, (author.NAME, author.SCHEMA)


def _model(cls, ctx: dict, keys: tuple[str, ...]):
    try:
        return cls(**{k: ctx[k] for k in keys if k in ctx})
    except ValidationError as e:
        raise ValueError(f"bad {cls.__name__}: {e.errors()[0]['loc']} {e.errors()[0]['msg']}") from None


def _need_frame(role: str, frame_sha: str | None) -> str:
    if not frame_sha:
        raise ValueError(f"{role} needs a frame")
    return frame_sha


# ---------------------------------------------------------------- wire shapes


def _responses(model: str, role: str, system: str | None, turns, schema, factor: int) -> dict[str, Any]:
    req: dict[str, Any] = {
        "model": model,
        "input": [_responses_turn(r, c) for r, c in turns],
        "reasoning": {"effort": META_EFFORT[role]},
        "max_output_tokens": META_MAX_OUTPUT[role] * factor,
        "store": False,
        "stream": True,
    }
    if system is not None:
        req["instructions"] = system
    if schema is not None:
        name, sch = schema
        req["text"] = {"format": {"type": "json_schema", "name": name, "schema": sch, "strict": True}}
    return req


def _responses_turn(role: str, content) -> dict[str, Any]:
    if role == "assistant":
        return {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": content}]}
    out = []
    for p in content:
        if p[0] == "text":
            out.append({"type": "input_text", "text": p[1]})
        elif p[0] == "image":
            out.append({"type": "input_image", "image_url": frame_ref(p[1])})
        else:
            out.append({"type": "input_file", "filename": p[2], "file_data": frame_ref(p[1])})
    return {"type": "message", "role": "user", "content": out}


def _chat(model: str, role: str, system: str | None, turns, schema, factor: int) -> dict[str, Any]:
    messages: list[dict[str, Any]] = []
    if system is not None:
        if schema is not None:
            system += "\n\nThe reply must be valid against this JSON Schema:\n" + json.dumps(
                schema[1], sort_keys=True, separators=(",", ":"), ensure_ascii=False
            )
        messages.append({"role": "system", "content": system})
    for r, c in turns:
        messages.append({"role": "assistant", "content": c} if r == "assistant" else {"role": "user", "content": [
            {"type": "text", "text": p[1]} if p[0] == "text" else {"type": "image_url", "image_url": {"url": frame_ref(p[1])}}
            for p in c
        ]})
    req: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "max_tokens": CHAT_MAX_TOKENS[role] * factor,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    if is_glimmer(model):
        req.update(GLIMMER_FIELDS)
    return req


def _segment(model: str, frame_sha: str, ctx: dict) -> dict[str, Any]:
    if "concept" in ctx and "object_prompts" in ctx:
        raise ValueError("segment takes a concept OR object prompts, not both")
    if "concept" in ctx:
        concept = ctx["concept"]
        if not isinstance(concept, str) or not concept.strip():
            raise ValueError("segment concept must be a non-empty noun phrase")
        if any(sep in concept for sep in segment.CONCEPT_SEPARATORS):
            raise ValueError(f"segment concept {concept!r} must be one noun phrase (no lists)")
        text = concept.strip()
    else:
        text = _object_prompt(ctx)
    return {
        "model": model,
        "input": [
            {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": text}, {"type": "input_image", "image_url": frame_ref(frame_sha)}],
            }
        ],
        "metadata": dict(segment.METADATA),
        "stream": True,
    }


def _object_prompt(ctx: dict) -> str:
    prompts, w, h = ctx.get("object_prompts"), ctx.get("w"), ctx.get("h")
    if not isinstance(prompts, list) or not prompts:
        raise ValueError("segment needs a concept or a non-empty object_prompts list")
    if len(prompts) > segment.MAX_OBJECTS:
        raise ValueError(f"segment takes at most {segment.MAX_OBJECTS} prompted objects")
    objects = []
    for p in prompts:
        if not isinstance(p, dict) or set(p) - {"id", "box", "points", "negative"}:
            raise ValueError(f"bad object prompt {p!r}")
        box = None
        if p.get("box") is not None:
            x0, y0, x1, y1 = _ints(p["box"], 4)
            box = PromptBox(x0, y0, x1, y1)
        points = [PromptPoint(*_ints(xy, 2)) for xy in p.get("points") or []]
        points += [PromptPoint(*_ints(xy, 2), label="negative") for xy in p.get("negative") or []]
        objects.append(PromptObject(id=p.get("id"), prompts=[FramePrompt(frame=0, box=box, points=points)]))
    return build_object_prompt(size=(w, h), objects=objects)


def _ints(v, n: int) -> list[int]:
    if not isinstance(v, (list, tuple)) or len(v) != n or not all(isinstance(x, int) and not isinstance(x, bool) for x in v):
        raise ValueError(f"expected {n} integers, got {v!r}")
    return list(v)
