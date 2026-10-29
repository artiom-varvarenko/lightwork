"""C8 run files: ``GET /api/runs``, ``/api/runs/{id}/{path}``, ``/frames/{sha}``.

``app.state.runs_root`` (default ``./runs``) is the only directory these routes read from, plus the optional
``app.state.frames_dirs`` (list of directories holding ``<sha>.jpg``, e.g. the live run's ``frames/``) searched first by
``/frames``. Paths are resolved (symlinks included) and must stay inside the run directory.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse

router = APIRouter()

_RUN_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_SHA = re.compile(r"^[0-9a-f]{64}$")
_MANIFEST_KEYS = ("created_utc", "provenance", "backend", "totals", "version")


def runs_root(request: Request) -> Path:
    return Path(getattr(request.app.state, "runs_root", None) or Path.cwd() / "runs").resolve()


def _run_dir(root: Path, run_id: str) -> Path:
    if not _RUN_ID.match(run_id) or run_id in (".", ".."):
        raise HTTPException(404, "no such run")
    d = (root / run_id).resolve()
    if d.parent != root or not d.is_dir():
        raise HTTPException(404, "no such run")
    return d


@router.get("/api/runs")
def list_runs(request: Request) -> list[dict]:
    root = runs_root(request)
    if not root.is_dir():
        return []
    out = []
    for d in sorted(p for p in root.iterdir() if p.is_dir() and _RUN_ID.match(p.name)):
        if not (d / "events.jsonl").is_file():
            continue
        row: dict = {"id": d.name}
        try:
            manifest = json.loads((d / "manifest.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            manifest = None
        row["complete"] = isinstance(manifest, dict)
        if isinstance(manifest, dict):
            row.update({k: manifest[k] for k in _MANIFEST_KEYS if k in manifest})
        out.append(row)
    return out


@router.get("/api/runs/{run_id}/{path:path}")
def run_file(request: Request, run_id: str, path: str) -> FileResponse:
    d = _run_dir(runs_root(request), run_id)
    f = (d / path).resolve()
    if not f.is_relative_to(d) or not f.is_file():
        raise HTTPException(404, "no such file")
    return FileResponse(f)


@router.get("/frames/{sha}")
def frame(request: Request, sha: str) -> FileResponse:
    if not _SHA.match(sha):
        raise HTTPException(404, "no such frame")
    dirs = [Path(p) for p in getattr(request.app.state, "frames_dirs", None) or []]
    root = runs_root(request)
    if root.is_dir():
        dirs += sorted(p / "frames" for p in root.iterdir() if p.is_dir())
    for d in dirs:
        f = d / f"{sha}.jpg"
        if f.is_file():
            return FileResponse(f, media_type="image/jpeg", headers={"Cache-Control": "max-age=31536000, immutable"})
    raise HTTPException(404, "no such frame")
