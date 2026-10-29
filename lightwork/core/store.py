"""The reducer's only I/O: recorded keyframes and recorded model responses."""

from __future__ import annotations

from typing import Protocol

from ..contracts.canonical import sha256_hex


class RunStore(Protocol):
    """Implemented by ``lightwork.runs.DirStore`` (a run directory) and :class:`MemoryStore`."""

    def frame_bytes(self, sha: str) -> bytes: ...

    def call_response(self, call_dir: str) -> tuple[bytes, str]:  # (raw, "sse" | "json")
        ...


class MemoryStore:
    """In-memory :class:`RunStore` for tests, scenarios and the live orchestrator's cache."""

    def __init__(self, frames: dict[str, bytes] | None = None, calls: dict[str, tuple[bytes, str]] | None = None):
        self.frames: dict[str, bytes] = dict(frames or {})
        self.calls: dict[str, tuple[bytes, str]] = dict(calls or {})

    def put_frame(self, jpeg: bytes) -> str:
        sha = sha256_hex(jpeg)
        self.frames[sha] = jpeg
        return sha

    def put_call(self, call_dir: str, raw: bytes, kind: str) -> None:
        self.calls[call_dir] = (raw, kind)

    def frame_bytes(self, sha: str) -> bytes:
        try:
            return self.frames[sha]
        except KeyError:
            raise FileNotFoundError(f"frames/{sha}.jpg") from None

    def call_response(self, call_dir: str) -> tuple[bytes, str]:
        try:
            return self.calls[call_dir]
        except KeyError:
            raise FileNotFoundError(f"{call_dir}/response.*") from None
