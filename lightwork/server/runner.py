"""Run the FastAPI app in a background thread (commands that drive the hubs from their own main thread)."""

from __future__ import annotations

import threading
import time

from fastapi import FastAPI

from ..contracts.server import HOST, PORT


class ServerHandle:
    def __init__(self, server, thread: threading.Thread):
        self._server = server
        self._thread = thread

    def stop(self, timeout: float = 5.0) -> None:
        self._server.should_exit = True
        self._thread.join(timeout)


def start_server(app: FastAPI, host: str = HOST, port: int = PORT, *, timeout: float = 10.0) -> ServerHandle:
    """Start uvicorn on ``host:port`` in a daemon thread and return once it listens.
    Raises ``RuntimeError`` if it does not start (e.g. the port is taken)."""
    import uvicorn

    config = uvicorn.Config(app, host=host, port=port, log_level="warning", ws_ping_interval=10, ws_ping_timeout=10)
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, name="lightwork-server", daemon=True)
    thread.start()
    deadline = time.monotonic() + timeout
    while not server.started:
        if not thread.is_alive() or time.monotonic() > deadline:
            server.should_exit = True
            raise RuntimeError(f"server did not start on {host}:{port}")
        time.sleep(0.02)
    return ServerHandle(server, thread)
