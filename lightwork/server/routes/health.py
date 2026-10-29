from __future__ import annotations

from fastapi import APIRouter, Request

from ... import __version__

router = APIRouter()


@router.get("/api/health")
def health(request: Request) -> dict:
    return {"ok": True, "version": __version__, "mode": getattr(request.app.state, "mode", "idle")}
