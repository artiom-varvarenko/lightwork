"""Read access to a run directory (C6). :class:`DirStore` is the reducer's ``RunStore`` for live and recorded runs."""

from __future__ import annotations

import os
import re
import threading
from pathlib import Path

from ..contracts.layout import Calibration, Layout
from ..contracts.stepgraph import StepGraph

EVENTS = "events.jsonl"
MANIFEST = "manifest.json"
VERIFY = "verify.json"
STEPGRAPH = "stepgraph.json"
LAYOUT = "layout.json"
CALIB = "calib.json"
RUN_INPUT_FILES = {STEPGRAPH: StepGraph, LAYOUT: Layout, CALIB: Calibration}
RESPONSE_FILES = (("response.sse", "sse"), ("response.json", "json"))

SHA_RE = re.compile(r"^[0-9a-f]{64}$")
CALL_DIR_RE = re.compile(r"^calls/\d{4,}-[A-Za-z0-9._\-]+$")  # "calls/" + contracts.calls.call_dirname(...)
PREVIEW_RE = re.compile(r"^(\d+)\.jpg$")


def write_atomic(path: Path, data: bytes) -> None:
    """Write ``data`` to ``path`` through an fsynced temp file and a rename (no torn files after a crash)."""
    tmp = path.with_name(f".{path.name}.{threading.get_ident()}.tmp")
    with open(tmp, "wb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def is_run_dir(path: Path) -> bool:
    return (path / EVENTS).is_file() or (path / MANIFEST).is_file()


def find_runs(path: Path) -> list[Path]:
    """``path`` itself if it is a run directory, else its run subdirectories in name order."""
    if is_run_dir(path):
        return [path]
    if not path.is_dir():
        return []
    return sorted(p for p in path.iterdir() if p.is_dir() and is_run_dir(p))


class DirStore:
    """Frames and call responses of one run directory. Paths taken from events are validated, so a crafted log
    cannot make the reducer read outside the run."""

    def __init__(self, run_dir: Path) -> None:
        self.run_dir = Path(run_dir)

    def _path(self, *parts: str) -> Path:
        path = self.run_dir.joinpath(*parts)
        if not path.resolve().is_relative_to(self.run_dir.resolve()):
            raise ValueError(f"path escapes the run directory: {path}")
        return path

    def frame_path(self, sha: str) -> Path:
        if not SHA_RE.match(sha):
            raise ValueError(f"not a sha256 hex digest: {sha!r}")
        return self._path("frames", f"{sha}.jpg")

    def frame_bytes(self, sha: str) -> bytes:
        return self.frame_path(sha).read_bytes()

    def call_path(self, call_dir: str) -> Path:
        if not CALL_DIR_RE.match(call_dir):
            raise ValueError(f"not a call directory: {call_dir!r}")
        return self._path(call_dir)

    def call_response(self, call_dir: str) -> tuple[bytes, str]:
        """Raw response bytes and their kind ("sse" | "json"); FileNotFoundError when the call has none."""
        self.call_path(call_dir)
        for name, kind in RESPONSE_FILES:
            p = self._path(call_dir, name)
            if p.is_file():
                return p.read_bytes(), kind
        raise FileNotFoundError(f"{call_dir}: no response file")

    def previews(self) -> list[tuple[int, Path]]:
        """``preview/<seq>.jpg`` files ordered by seq."""
        d = self._path("preview")
        if not d.is_dir():
            return []
        found = ((PREVIEW_RE.match(p.name), p) for p in d.iterdir() if p.is_file())
        return sorted((int(m.group(1)), self._path("preview", p.name)) for m, p in found if m)
