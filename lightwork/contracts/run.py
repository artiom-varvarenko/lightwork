"""C6 Run directory ``runs/<YYYYMMDDTHHMMSSZ>_<pack>/``.

Layout: ``manifest.json``, ``events.jsonl``, ``stepgraph.json``, ``layout.json``, ``calib.json``,
``frames/<sha>.jpg``, ``preview/<seq>.jpg`` (<= 1 fps, <= 480 px tall, work-area crop), ``calls/``, ``verify.json``.
``manifest.files`` covers every file except ``manifest.json`` and ``verify.json``.
Size target <= 10 MB, hard limit 25 MB.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict

RUN_SIZE_TARGET = 10 * 1024 * 1024
RUN_SIZE_LIMIT = 25 * 1024 * 1024
UNHASHED_FILES = frozenset({"manifest.json", "verify.json"})
Provenance = Literal["meta", "dev-standin"]


class _M(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class Totals(_M):
    passed: int
    failed: int
    unverified: int
    calls: int
    cost_usd: float
    duration_s: float


class Manifest(_M):
    run_id: str
    created_utc: str
    code_commit: str
    dirty: bool
    version: str
    platform: str
    provenance: Provenance
    backend: Literal["meta", "chat"]
    models: dict[str, str]
    files: dict[str, str]  # relative posix path -> sha256
    size_bytes: int
    totals: Totals


class Diff(_M):
    cls: Literal["chain", "manifest", "derived", "fingerprint", "schema", "missing", "extra"]
    seq: int | None = None
    path: str | None = None
    detail: str


class VerifyReport(_M):
    run_id: str
    ok: bool
    n_events: int
    n_inputs: int
    n_derived: int
    n_calls: int
    diffs: list[Diff]
    code_version: str
