"""Deterministic synthetic response bodies in the exact wire formats ``parse`` reads (Responses SSE, Chat SSE, SAM
records with ``one_bit`` masks). For tests, scenario builders and fixture runs; never used by the live client.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

# Private encoder of the pinned meta-sam-parser (uv.lock); the package exposes decoders only.
from meta_sam_parser._mask_codec import _encode_raster

_CHUNK = 24  # chars per synthetic text delta, so readers must join deltas


def _sse(events: list[tuple[str | None, Any]], *, done: bool = True) -> bytes:
    out: list[str] = []
    for name, data in events:
        if name is not None:
            out.append(f"event: {name}\n")
        out.append("data: " + json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n\n")
    if done:
        out.append("data: [DONE]\n\n")
    return "".join(out).encode("utf-8")


def _chunks(text: str) -> list[str]:
    return [text[i : i + _CHUNK] for i in range(0, len(text), _CHUNK)] or [""]


def default_usage(text: str) -> dict[str, Any]:
    out_tokens = max(1, len(text) // 4)
    return {
        "input_tokens": 1000,
        "input_tokens_details": {"cached_tokens": 0},
        "output_tokens": out_tokens,
        "output_tokens_details": {"reasoning_tokens": 0},
        "total_tokens": 1000 + out_tokens,
    }


def responses_sse(
    text: str,
    *,
    model: str,
    response_id: str = "resp_synthetic",
    usage: dict[str, Any] | None = None,
    terminal: Literal["completed", "incomplete", "failed"] = "completed",
) -> bytes:
    """A Responses API stream that writes ``text`` and ends with ``response.<terminal>``."""
    usage = default_usage(text) if usage is None else usage
    msg_id = f"msg_{response_id}"
    part = {"type": "output_text", "text": text, "annotations": []}
    final: dict[str, Any] = {
        "id": response_id,
        "object": "response",
        "model": model,
        "status": terminal,
        "output": [{"type": "message", "id": msg_id, "role": "assistant", "status": "completed", "content": [part]}],
        "usage": usage,
    }
    if terminal == "incomplete":
        final["incomplete_details"] = {"reason": "max_output_tokens"}
    if terminal == "failed":
        final["error"] = {"code": "server_error", "message": "The server had an error processing the request."}
    start = {"id": response_id, "object": "response", "model": model, "status": "in_progress"}
    events: list[tuple[str | None, Any]] = [
        ("response.created", {"type": "response.created", "response": start}),
        ("response.in_progress", {"type": "response.in_progress", "response": start}),
        ("response.output_item.added", {"type": "response.output_item.added", "output_index": 0,
                                        "item": {"type": "message", "id": msg_id, "role": "assistant", "content": []}}),
        ("response.content_part.added", {"type": "response.content_part.added", "output_index": 0, "content_index": 0,
                                         "part": {"type": "output_text", "text": ""}}),
    ]
    for d in _chunks(text):
        events.append(("response.output_text.delta", {"type": "response.output_text.delta", "output_index": 0,
                                                      "content_index": 0, "delta": d}))
    events += [
        ("response.output_text.done", {"type": "response.output_text.done", "output_index": 0, "content_index": 0,
                                       "text": text}),
        ("response.content_part.done", {"type": "response.content_part.done", "output_index": 0, "content_index": 0,
                                        "part": part}),
        (f"response.{terminal}", {"type": f"response.{terminal}", "response": final}),
    ]
    for seq, (_, data) in enumerate(events, start=1):
        data["sequence_number"] = seq
    return _sse(events)


def responses_error_sse(code: str, message: str, *, model: str, response_id: str = "resp_synthetic_error") -> bytes:
    """A Responses API stream cut by a terminal ``error`` event (e.g. ``server_shutting_down``)."""
    start = {"id": response_id, "object": "response", "model": model, "status": "in_progress"}
    return _sse(
        [
            ("response.created", {"type": "response.created", "response": start, "sequence_number": 1}),
            ("error", {"type": "error", "code": code, "message": message, "param": None, "sequence_number": 2}),
        ],
        done=False,
    )


def chat_sse(
    content: str,
    *,
    model: str,
    reasoning: str = "",
    finish: str = "stop",
    usage: dict[str, Any] | None = None,
    response_id: str = "chatcmpl-synthetic",
) -> bytes:
    """A Chat Completions stream (llama.cpp style): reasoning deltas, content deltas, finish chunk, usage chunk."""

    def chunk(delta: dict[str, Any], finish_reason: str | None = None) -> tuple[None, dict[str, Any]]:
        return None, {"id": response_id, "object": "chat.completion.chunk", "model": model,
                      "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}]}

    events = [chunk({"role": "assistant", "content": None})]
    if reasoning:
        events += [chunk({"reasoning_content": d}) for d in _chunks(reasoning)]
    if content:
        events += [chunk({"content": d}) for d in _chunks(content)]
    events.append(chunk({}, finish))
    usage = usage if usage is not None else {
        "prompt_tokens": 1000,
        "completion_tokens": max(1, (len(content) + len(reasoning)) // 4),
        "total_tokens": 1000 + max(1, (len(content) + len(reasoning)) // 4),
    }
    events.append((None, {"id": response_id, "object": "chat.completion.chunk", "model": model, "choices": [],
                          "usage": usage}))
    return _sse(events)


@dataclass(frozen=True)
class SamObject:
    """One SAM match: half-open box ``(x0, y0, x1, y1)`` in sent-image px; ``mask`` is a bool array of shape
    ``(y1 - y0, x1 - x0)`` (``None`` = the filled box)."""

    object_id: int
    box: tuple[int, int, int, int]
    mask: np.ndarray | None = None
    confidence: float | None = None


def sam_record(obj: SamObject, size: tuple[int, int]) -> str:
    x0, y0, x1, y1 = obj.box
    w, h = x1 - x0, y1 - y0
    mask = np.ones((h, w), dtype=bool) if obj.mask is None else np.asarray(obj.mask, dtype=bool)
    if mask.shape != (h, w):
        raise ValueError(f"mask shape {mask.shape} does not match box {w}x{h}")
    payload = _encode_raster(mask.astype(np.uint8).ravel().tolist(), w, h)
    c = "" if obj.confidence is None else f";c={obj.confidence!r}"
    return (f"{obj.object_id}<|box;x1={x0};y1={y0};x2={x1 - 1};y2={y1 - 1};w={size[0]};h={size[1]}{c}|>"
            f"<|mask;x=0;y=0{c};data={h},{w},{payload}|>")


def sam_sse(objects: list[SamObject], size: tuple[int, int], *, response_id: str = "resp_synthetic_sam") -> bytes:
    """A SAM 3.1 image stream: one ``<0f>`` line with every object (no line at all for zero matches)."""
    text = "<0f>" + ",".join(sam_record(o, size) for o in objects) if objects else ""
    usage = {"images_processed": 1, "input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    return responses_sse(text, model="sam-3.1", response_id=response_id, usage=usage)
