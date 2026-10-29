"""Manifest (C6): file hashes, totals and size class. The recorder writes it at close; verify and audit check it."""

from __future__ import annotations

import re
from collections.abc import Sequence
from pathlib import Path

from ..contracts.canonical import file_sha
from ..contracts.events import Event
from ..contracts.run import RUN_SIZE_LIMIT, RUN_SIZE_TARGET, UNHASHED_FILES, Manifest, Provenance, Totals

RUN_ID_RE = re.compile(r"^(\d{4})(\d{2})(\d{2})T(\d{2})(\d{2})(\d{2})Z_([a-z0-9][a-z0-9_\-]*)$")


def created_utc(run_id: str) -> str:
    """ISO-8601 UTC creation time encoded in a ``<YYYYMMDDTHHMMSSZ>_<pack>`` run id."""
    m = RUN_ID_RE.match(run_id)
    if not m:
        raise ValueError(f"run id {run_id!r} is not <YYYYMMDDTHHMMSSZ>_<pack>")
    y, mo, d, h, mi, s, _pack = m.groups()
    return f"{y}-{mo}-{d}T{h}:{mi}:{s}Z"


def hashed_files(run_dir: Path) -> dict[str, str]:
    """``{relative posix path: sha256}`` of every file the manifest covers (all but manifest.json, verify.json)."""
    out = {}
    for p in sorted(run_dir.rglob("*")):
        if p.is_file():
            rel = p.relative_to(run_dir).as_posix()
            if rel not in UNHASHED_FILES:
                out[rel] = file_sha(p)
    return out


def files_size(run_dir: Path, files: dict[str, str]) -> int:
    return sum((run_dir / rel).stat().st_size for rel in files)


def size_class(size_bytes: int) -> str:
    if size_bytes > RUN_SIZE_LIMIT:
        return "over-limit"
    if size_bytes > RUN_SIZE_TARGET:
        return "over-target"
    return "ok"


def run_totals(events: Sequence[Event]) -> Totals:
    """``passed``/``unverified`` count steps by their last state, ``failed`` counts transitions to ``failed`` (caught
    mistakes), ``calls`` counts ``model_result`` events, ``cost_usd`` is the last ``cost`` event and ``duration_s``
    the last event's ``t``."""
    last: dict[str, str] = {}
    failed = calls = 0
    cost = 0.0
    for ev in events:
        if ev.type == "step":
            last[ev.payload["step"]] = ev.payload["state"]
            failed += ev.payload["state"] == "failed"
        elif ev.type == "model_result":
            calls += 1
        elif ev.type == "cost":
            cost = ev.payload["usd_total"]
    states = list(last.values())
    return Totals(
        passed=states.count("passed"),
        failed=failed,
        unverified=states.count("unverified"),
        calls=calls,
        cost_usd=cost,
        duration_s=events[-1].t if events else 0.0,
    )


def build_manifest(run_dir: Path, events: Sequence[Event], provenance: Provenance) -> Manifest:
    started = events[0].payload
    files = hashed_files(run_dir)
    return Manifest(
        run_id=started["run_id"],
        created_utc=created_utc(started["run_id"]),
        code_commit=started["code_commit"],
        dirty=started["dirty"],
        version=started["version"],
        platform=started["platform"],
        provenance=provenance,
        backend=started["backend"],
        models=started["models"],
        files=files,
        size_bytes=files_size(run_dir, files),
        totals=run_totals(events),
    )
