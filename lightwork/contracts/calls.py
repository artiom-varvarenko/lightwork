"""C5 Model call record: ``calls/<NNNN>-<role>-<model>/``.

- ``request.json``: canonical request, key redacted, media replaced by ``{"$frame": sha}``.
- ``response.sse`` (raw bytes) or ``response.json``.
- ``meta.json``: :class:`CallMeta`.

``fingerprint = sha256(canonical(request without transport-only fields))``. The reducer computes it from the pure
request builder when it emits a ``model_request`` intent; verify recomputes both sides.
"""

from __future__ import annotations

import re
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict

from .canonical import canonical_sha

# Fields that change how bytes travel, not what is asked.
TRANSPORT_FIELDS = frozenset({"stream", "stream_options"})
FRAME_REF_KEY = "$frame"


class _M(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class Attempt(_M):
    n: int
    status: int | None  # HTTP status; None on transport error/timeout
    error: str | None
    wait_s: float  # backoff slept before this attempt
    ms: int


class ServerProps(_M):
    build_info: str | None = None
    model_path: str | None = None
    modalities: dict[str, bool] | None = None


class CallMeta(_M):
    backend: Literal["meta", "chat"]
    base_host: str
    endpoint: str  # "/v1/responses" | "/v1/chat/completions" | ...
    model: str
    status: Literal["ok", "error", "timeout"]
    http_status: int | None
    attempts: list[Attempt]
    headers: dict[str, str]  # x-ratelimit-*, retry-after only
    response_id: str | None
    ttfb_ms: int | None
    total_ms: int
    usage: dict[str, Any] | None
    cost_usd: float
    fingerprint: str
    server_props: ServerProps | None = None
    provenance: Literal["meta", "dev-standin", "synthetic"]


def strip_transport(request: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in request.items() if k not in TRANSPORT_FIELDS}


def request_fingerprint(request: dict[str, Any]) -> str:
    """Fingerprint of a canonical (media-redacted) request."""
    return canonical_sha(strip_transport(request))


def frame_ref(sha: str) -> dict[str, str]:
    return {FRAME_REF_KEY: sha}


_SAFE = re.compile(r"[^A-Za-z0-9._\-]+")


def call_dirname(index: int, role: str, model: str) -> str:
    return f"{index:04d}-{role}-{_SAFE.sub('_', model)}"
