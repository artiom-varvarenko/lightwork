"""C8 pages: ``GET /projector``, ``/screen``, ``/viewer`` (``viewer.html`` is Q's; 404 until it exists).

Served with ``Cache-Control: no-cache`` so the phone revalidates after code changes.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from ..app import WEB_DIR

router = APIRouter()

NO_CACHE = {"Cache-Control": "no-cache"}


def _page(name: str) -> FileResponse:
    path = WEB_DIR / f"{name}.html"
    if not path.is_file():
        raise HTTPException(404, f"{name}.html not found")
    return FileResponse(path, media_type="text/html", headers=NO_CACHE)


@router.get("/projector", include_in_schema=False)
def projector() -> FileResponse:
    return _page("projector")


@router.get("/screen", include_in_schema=False)
def screen() -> FileResponse:
    return _page("screen")


@router.get("/viewer", include_in_schema=False)
def viewer() -> FileResponse:
    return _page("viewer")
