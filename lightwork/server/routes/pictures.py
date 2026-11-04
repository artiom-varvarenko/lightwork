"""``GET /pictures/{name}``: pictures for the C2 ``picture`` primitive (Muse Image's, read from disk).

A scene goes out up to 30 times a second, so it carries only this URL, never the bytes. Names are the picture's content
hash, so a page may keep one for good. ``app.state.pictures_dir`` is the only directory read.
"""

from __future__ import annotations

import re
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse

router = APIRouter()

_NAME = re.compile(r"^[0-9a-f]{16,64}\.(png|jpg)$")


@router.get("/pictures/{name}", include_in_schema=False)
def picture(name: str, request: Request) -> FileResponse:
    root = getattr(request.app.state, "pictures_dir", None)
    if root is None or not _NAME.match(name):
        raise HTTPException(404, "no such picture")
    path = Path(root) / name
    if not path.is_file():
        raise HTTPException(404, "no such picture")
    return FileResponse(path, headers={"Cache-Control": "max-age=86400, immutable"})
