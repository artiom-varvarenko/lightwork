"""Writer of one C5 call directory; the live client, fixture builders and scenario recordings all use it."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any, Literal

from lightwork.contracts.calls import CallMeta, call_dirname, request_fingerprint
from lightwork.contracts.canonical import canonical_bytes


def write_call(
    run_dir: Path,
    *,
    index: int,
    role: str,
    request: dict[str, Any],
    raw: bytes,
    kind: Literal["sse", "json"],
    meta: CallMeta,
    attempt_responses: list[tuple[bytes, str]] | None = None,
) -> str:
    """Write ``calls/<NNNN>-<role>-<model>/`` (request.json, response.sse|json, meta.json); return that relative path.

    ``request`` is the canonical request (media as ``{"$frame": sha}``, no key); ``raw`` the response bytes as
    received. Refuses an existing call dir and a ``meta`` whose model or fingerprint does not match ``request``.
    """
    if kind not in ("sse", "json"):
        raise ValueError(f"unknown response kind {kind!r}")
    if meta.model != request.get("model"):
        raise ValueError(f"meta.model {meta.model!r} != request model {request.get('model')!r}")
    if meta.fingerprint != request_fingerprint(request):
        raise ValueError("meta.fingerprint does not match the request")
    if _has_inline_media(request):
        raise ValueError('request.json must reference media as {"$frame": sha}, not inline data')
    name = call_dirname(index, role, request["model"])
    calls = Path(run_dir) / "calls"
    final = calls / name
    if final.exists():
        raise FileExistsError(f"call dir exists: {final}")
    calls.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{name}.", dir=calls) as temp:
        partial = Path(temp)
        (partial / "request.json").write_bytes(canonical_bytes(request) + b"\n")
        (partial / f"response.{kind}").write_bytes(raw)
        for n, (body, body_kind) in enumerate(attempt_responses or [], start=1):
            if body_kind not in ("sse", "json"):
                raise ValueError(f"unknown attempt response kind {body_kind!r}")
            (partial / "attempts").mkdir(exist_ok=True)
            (partial / "attempts" / f"{n:04d}.{body_kind}").write_bytes(body)
        (partial / "meta.json").write_bytes(canonical_bytes(meta) + b"\n")
        os.rename(partial, final)
    return f"calls/{name}"


def _has_inline_media(node: Any) -> bool:
    if isinstance(node, str):
        return node.startswith("data:")
    if isinstance(node, dict):
        return any(_has_inline_media(v) for v in node.values())
    if isinstance(node, list):
        return any(_has_inline_media(v) for v in node)
    return False
