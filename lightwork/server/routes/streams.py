"""C8 websockets ``/ws/render``, ``/ws/camera``, ``/ws/control`` and ``GET /api/pages`` (connected page status).

The hubs come from ``app.state.render_hub`` / ``preview_hub`` / ``control_hub`` (``hub.make_hubs()``); without
them the socket is accepted and closed with 1011 so the page shows its link-lost state instead of retrying blindly.
"""

from __future__ import annotations

from fastapi import APIRouter, Request, WebSocket, WebSocketDisconnect

from ...contracts.server import WS_CAMERA, WS_CONTROL, WS_RENDER

router = APIRouter()


async def _serve(websocket: WebSocket, attr: str) -> None:
    hub = getattr(websocket.app.state, attr, None)
    await websocket.accept()
    if hub is None:
        await websocket.close(code=1011, reason=f"no {attr}")
        return
    try:
        await hub.serve(websocket)
    except WebSocketDisconnect:
        pass


@router.websocket(WS_RENDER)
async def ws_render(websocket: WebSocket) -> None:
    await _serve(websocket, "render_hub")


@router.websocket(WS_CAMERA)
async def ws_camera(websocket: WebSocket) -> None:
    await _serve(websocket, "preview_hub")


@router.websocket(WS_CONTROL)
async def ws_control(websocket: WebSocket) -> None:
    await _serve(websocket, "control_hub")


@router.get("/api/pages")
def pages(request: Request) -> dict:
    render_hub = getattr(request.app.state, "render_hub", None)
    preview_hub = getattr(request.app.state, "preview_hub", None)
    return {
        "render": render_hub.stats() if render_hub else [],
        "camera": preview_hub.stats() if preview_hub else [],
    }
