"""``/ws/mic`` (C8): PCM16 LE mono 24 kHz binary frames from the browser into ``app.state.voice.feed_pcm``.

The newest connection wins: frames from an older tab are ignored, so two open pages never interleave audio.
Without a voice sink (replay, or voice disabled) the socket is closed with ``VOICE_OFF_CLOSE`` and mic.js retries
slowly.
"""

from __future__ import annotations

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

router = APIRouter()

VOICE_OFF_CLOSE = 4003


@router.websocket("/ws/mic")
async def mic(ws: WebSocket) -> None:
    state = ws.app.state
    sink = getattr(state, "voice", None)
    token = object()
    if sink is not None:
        state.mic_owner = token  # before accept: once the client sees the socket open, older tabs are ignored
    await ws.accept()
    if sink is None:
        await ws.close(code=VOICE_OFF_CLOSE, reason="voice not running")
        return
    try:
        while True:
            msg = await ws.receive()
            if msg["type"] == "websocket.disconnect":
                break
            data = msg.get("bytes")
            if data and state.mic_owner is token:
                sink.feed_pcm(data)
    except WebSocketDisconnect:
        pass
