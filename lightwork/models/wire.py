"""Own SSE parser and readers for recorded response bodies (D2). Pure: bytes in, ``Reading`` out.

Two wire formats: the Responses API (Meta: Spark, Wonder, SAM) and Chat Completions (llama.cpp Glimmer, dev
stand-ins). Each comes as an SSE stream (``kind="sse"``) or one JSON body (``kind="json"``, also every HTTP error
body). SSE lines are split on CRLF/LF/CR only (JSON text may hold U+2028); only ``data:`` fields carry events, plus
llama.cpp's ``error:`` field for mid-stream errors; comments, pings, ``event:``/``id:``/``retry:`` lines are skipped.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Literal

RESPONSES = "/v1/responses"
CHAT = "/v1/chat/completions"

_LINE_BREAK = re.compile(r"\r\n|\r|\n")
# Error codes after which the whole request is retried (error-handling.md: 503 family, 500 parse failure).
RETRYABLE_CODES = frozenset(
    {"server_shutting_down", "service_overloaded", "backend_unavailable", "rate_limit_exceeded", "server_error"}
)
STREAM_CUT = "stream_cut"  # error_code of a stream that ended without a terminal event


@dataclass(frozen=True)
class SSEEvent:
    field: Literal["data", "error"]
    data: str


def iter_sse(raw: bytes) -> list[SSEEvent]:
    """Events of an SSE body; multi-line ``data:`` fields are joined with ``\\n`` (SSE spec)."""
    events: list[SSEEvent] = []
    lines: list[str] = []
    field: Literal["data", "error"] = "data"
    for line in _LINE_BREAK.split(raw.decode("utf-8", errors="replace")):
        if not line:
            if lines:
                events.append(SSEEvent(field, "\n".join(lines)))
            lines, field = [], "data"
            continue
        name, _, value = line.partition(":")
        if name not in ("data", "error"):
            continue
        if value.startswith(" "):
            value = value[1:]
        if name == "error":
            field = "error"
        lines.append(value)
    if lines:
        events.append(SSEEvent(field, "\n".join(lines)))
    return events


@dataclass(frozen=True)
class Reading:
    """What a body says, independent of role. ``status`` is ``ok`` | ``incomplete`` (output budget ran out) | ``error``."""

    status: Literal["ok", "incomplete", "error"]
    text: str = ""
    reasoning: str = ""
    error: str | None = None
    error_code: str | None = None
    usage: dict[str, Any] | None = None
    response_id: str | None = None


def read_body(raw: bytes, kind: str, endpoint: str) -> Reading:
    if kind not in ("sse", "json"):
        raise ValueError(f"unknown response kind {kind!r}")
    if endpoint not in (RESPONSES, CHAT):
        raise ValueError(f"unknown endpoint {endpoint!r}")
    if not raw.strip():
        return Reading("error", error="empty response body", error_code="empty")
    if kind == "json":
        obj = _loads(raw.decode("utf-8", errors="replace"))
        if not isinstance(obj, dict):
            return Reading("error", error=f"body is not a JSON object: {_excerpt(raw)}", error_code="malformed")
        return _responses_json(obj) if endpoint == RESPONSES else _chat_json(obj)
    events: list[dict[str, Any]] = []
    for ev in iter_sse(raw):
        if ev.data.strip() == "[DONE]":
            continue
        obj = _loads(ev.data)
        if not isinstance(obj, dict):
            return Reading("error", error=f"malformed SSE data: {ev.data[:200]!r}", error_code="malformed")
        events.append({"type": "error", "error": obj} if ev.field == "error" else obj)
    return _responses_sse(events) if endpoint == RESPONSES else _chat_sse(events)


# ---------------------------------------------------------------- Responses API


def _responses_sse(events: list[dict[str, Any]]) -> Reading:
    deltas: list[str] = []
    done_texts: list[str] = []
    refusal_deltas: list[str] = []
    refusal_done = ""
    final: dict[str, Any] | None = None
    terminal: str | None = None
    err: Any = None
    response_id: str | None = None
    for obj in events:
        resp = obj.get("response") if isinstance(obj.get("response"), dict) else None
        if resp is not None and isinstance(resp.get("id"), str):
            response_id = resp["id"]
        t = obj.get("type")
        if t == "response.output_text.delta":
            deltas.append(_str(obj.get("delta")))
        elif t == "response.output_text.done":
            done_texts.append(_str(obj.get("text")))
        elif t == "response.refusal.delta":
            refusal_deltas.append(_str(obj.get("delta")))
        elif t == "response.refusal.done":
            refusal_done = _str(obj.get("refusal"))
        elif t in ("response.completed", "response.incomplete", "response.failed"):
            final, terminal = resp or {}, t
            if t == "response.failed":
                err = final.get("error")
        elif t == "error":
            terminal, err = t, obj.get("error") if isinstance(obj.get("error"), dict) else obj
    usage = _usage(final)
    if terminal is None:
        return Reading("error", error="stream ended without a terminal event", error_code=STREAM_CUT,
                       response_id=response_id)
    if terminal in ("response.failed", "error"):
        return _error_reading(err, usage, response_id)
    text = "".join(deltas) or "".join(done_texts) or _output_text(final)
    refusal = "".join(refusal_deltas) or refusal_done or _output_refusal(final)
    if refusal:
        return Reading("error", text=text, error=f"refused: {refusal[:200]}", error_code="refusal", usage=usage,
                       response_id=response_id)
    if terminal == "response.incomplete":
        return _incomplete(final, text, usage, response_id)
    return Reading("ok", text=text, usage=usage, response_id=response_id)


def _responses_json(obj: dict[str, Any]) -> Reading:
    response_id = obj.get("id") if isinstance(obj.get("id"), str) else None
    usage = _usage(obj)
    status = obj.get("status")
    if status == "failed" or (obj.get("error") and status is None):
        return _error_reading(obj.get("error"), usage, response_id)
    text = _output_text(obj)
    if status == "incomplete":
        return _incomplete(obj, text, usage, response_id)
    if status != "completed":
        return Reading("error", error=f"response status {status!r}", error_code="malformed", usage=usage,
                       response_id=response_id)
    refusal = _output_refusal(obj)
    if refusal:
        return Reading("error", text=text, error=f"refused: {refusal[:200]}", error_code="refusal", usage=usage,
                       response_id=response_id)
    return Reading("ok", text=text, usage=usage, response_id=response_id)


def _incomplete(resp: dict[str, Any] | None, text: str, usage, response_id) -> Reading:
    details = (resp or {}).get("incomplete_details")
    reason = details.get("reason") if isinstance(details, dict) else None
    return Reading("incomplete", text=text, error=_str(reason) or "incomplete", error_code="incomplete",
                   usage=usage, response_id=response_id)


def _output_parts(resp: dict[str, Any] | None):
    for item in (resp or {}).get("output") or []:
        if isinstance(item, dict) and item.get("type") == "message":
            for part in item.get("content") or []:
                if isinstance(part, dict):
                    yield part


def _output_text(resp: dict[str, Any] | None) -> str:
    return "".join(_str(p.get("text")) for p in _output_parts(resp) if p.get("type") == "output_text")


def _output_refusal(resp: dict[str, Any] | None) -> str:
    return "".join(_str(p.get("refusal")) for p in _output_parts(resp) if p.get("type") == "refusal")


# ---------------------------------------------------------------- Chat Completions


def _chat_sse(events: list[dict[str, Any]]) -> Reading:
    content: list[str] = []
    reasoning: list[str] = []
    finish: str | None = None
    usage: dict[str, Any] | None = None
    response_id: str | None = None
    for obj in events:
        if obj.get("type") == "error" or obj.get("error"):
            err = obj.get("error")
            return _error_reading(err if err is not None else obj, usage, response_id)
        if isinstance(obj.get("id"), str):
            response_id = obj["id"]
        if isinstance(obj.get("usage"), dict):
            usage = obj["usage"]
        for choice in obj.get("choices") or []:
            if not isinstance(choice, dict):
                continue
            delta = choice.get("delta") if isinstance(choice.get("delta"), dict) else {}
            content.append(_str(delta.get("content")))
            reasoning.append(_str(delta.get("reasoning_content")) or _str(delta.get("reasoning")))
            if choice.get("finish_reason"):
                finish = str(choice["finish_reason"])
    if finish is None:
        return Reading("error", text="".join(content), reasoning="".join(reasoning),
                       error="stream ended without a finish_reason", error_code=STREAM_CUT, usage=usage,
                       response_id=response_id)
    return _chat_finish(finish, "".join(content), "".join(reasoning), usage, response_id)


def _chat_json(obj: dict[str, Any]) -> Reading:
    if obj.get("error"):
        return _error_reading(obj["error"], None, None)
    response_id = obj.get("id") if isinstance(obj.get("id"), str) else None
    usage = obj.get("usage") if isinstance(obj.get("usage"), dict) else None
    choices = obj.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
        return Reading("error", error="no choices in body", error_code="malformed", usage=usage,
                       response_id=response_id)
    message = choices[0].get("message") if isinstance(choices[0].get("message"), dict) else {}
    reasoning = _str(message.get("reasoning_content")) or _str(message.get("reasoning"))
    finish = choices[0].get("finish_reason")
    if not finish:
        return Reading("error", text=_str(message.get("content")), reasoning=reasoning,
                       error="no finish_reason in body", error_code="malformed", usage=usage, response_id=response_id)
    return _chat_finish(str(finish), _str(message.get("content")), reasoning, usage, response_id)


def _chat_finish(finish: str, text: str, reasoning: str, usage, response_id) -> Reading:
    if finish == "stop":
        return Reading("ok", text=text, reasoning=reasoning, usage=usage, response_id=response_id)
    if finish == "length":
        return Reading("incomplete", text=text, reasoning=reasoning, error="length", error_code="incomplete",
                       usage=usage, response_id=response_id)
    return Reading("error", text=text, reasoning=reasoning, error=f"finish_reason {finish}", error_code=finish,
                   usage=usage, response_id=response_id)


# ---------------------------------------------------------------- helpers


def _error_reading(err: Any, usage, response_id) -> Reading:
    if not isinstance(err, dict):
        return Reading("error", error=_str(err)[:300] or "unspecified error", usage=usage, response_id=response_id)
    code = err.get("code")
    if not (isinstance(code, str) and code):
        code = _str(err.get("type")) or None  # llama.cpp: numeric code, string type
    message = _str(err.get("message")) or json.dumps(err, sort_keys=True)[:200]
    return Reading("error", error=(f"{code}: {message}" if code else message)[:300], error_code=code, usage=usage,
                   response_id=response_id)


def _usage(resp: dict[str, Any] | None) -> dict[str, Any] | None:
    u = (resp or {}).get("usage")
    return u if isinstance(u, dict) else None


def _loads(text: str) -> Any:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def _str(v: Any) -> str:
    return v if isinstance(v, str) else ""


def _excerpt(raw: bytes) -> str:
    return repr(raw[:200].decode("utf-8", errors="replace"))
