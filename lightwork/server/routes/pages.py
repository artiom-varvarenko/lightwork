"""C8 pages: ``GET /projector``, ``/screen``, ``/viewer`` (``viewer.html`` is Q's; 404 until it exists).

Served with ``Cache-Control: no-cache`` so the phone revalidates after code changes. Each page's own script URL also
carries the script file's modification time: Chrome may keep an old module script "fresh" without asking (heuristic
caching; seen 2 Nov on the projector phone), and a changed URL is always fetched.
"""

from __future__ import annotations

import re

from fastapi import APIRouter, HTTPException
from fastapi.responses import HTMLResponse

from ..app import WEB_DIR

router = APIRouter()

NO_CACHE = {"Cache-Control": "no-cache"}
_SCRIPT = re.compile(r'(src="\.?/static/)([^"?]+)"')


def _stamp(m: re.Match) -> str:
    f = WEB_DIR / m.group(2)
    return f'{m.group(1)}{m.group(2)}?v={f.stat().st_mtime_ns if f.is_file() else 0}"'


def _page(name: str) -> HTMLResponse:
    path = WEB_DIR / f"{name}.html"
    if not path.is_file():
        raise HTTPException(404, f"{name}.html not found")
    return HTMLResponse(_SCRIPT.sub(_stamp, path.read_text(encoding="utf-8")), headers=NO_CACHE)


@router.get("/projector", include_in_schema=False)
def projector() -> HTMLResponse:
    return _page("projector")


@router.get("/screen", include_in_schema=False)
def screen() -> HTMLResponse:
    return _page("screen")


@router.get("/viewer", include_in_schema=False)
def viewer() -> HTMLResponse:
    return _page("viewer")
