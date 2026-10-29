"""FastAPI app (C8). Every ``lightwork/server/routes/*.py`` module that exposes ``router`` is included, so
workstreams add routes without editing this file. Shared runtime objects (orchestrator, replay engine, hubs) live on
``app.state`` and are set by the command that starts the server (``live``, ``replay``).
"""

from __future__ import annotations

import importlib
import pkgutil
from pathlib import Path
from typing import Any

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from .. import __version__
from . import routes

WEB_DIR = Path(__file__).resolve().parent.parent / "web"


def discover_routers():
    out = []
    for info in sorted(pkgutil.iter_modules(routes.__path__), key=lambda i: i.name):
        if info.name.startswith("_"):
            continue
        mod = importlib.import_module(f"{routes.__name__}.{info.name}")
        router = getattr(mod, "router", None)
        if router is not None:
            out.append((info.name, router))
    return out


def create_app(**state: Any) -> FastAPI:
    app = FastAPI(title="Lightwork", version=__version__, docs_url=None, redoc_url=None)
    for key, value in state.items():
        setattr(app.state, key, value)
    for _name, router in discover_routers():
        app.include_router(router)
    app.mount("/static", StaticFiles(directory=WEB_DIR, check_dir=True), name="static")
    return app


def serve(app: FastAPI, host: str = "127.0.0.1", port: int = 8000) -> None:
    import uvicorn

    uvicorn.run(app, host=host, port=port, log_level="warning", ws_ping_interval=10, ws_ping_timeout=10)
