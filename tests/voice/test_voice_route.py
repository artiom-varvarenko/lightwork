from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from lightwork.server.app import create_app, discover_routers
from lightwork.server.routes.voice import VOICE_OFF_CLOSE


class Sink:
    def __init__(self):
        self.frames: list[bytes] = []

    def feed_pcm(self, frame: bytes) -> None:
        self.frames.append(frame)


def test_mic_route_feeds_voice_sink(no_network):
    assert "voice" in {name for name, _ in discover_routers()}
    sink = Sink()
    client = TestClient(create_app(voice=sink))
    with client.websocket_connect("/ws/mic") as ws:
        for i in range(3):
            ws.send_bytes(bytes([i]) * 3840)
        ws.send_text("ignored")
        ws.send_bytes(bytes([9]) * 3840)
    assert [f[0] for f in sink.frames] == [0, 1, 2, 9]
    assert all(len(f) == 3840 for f in sink.frames)


def test_mic_route_closes_without_voice(no_network):
    client = TestClient(create_app())
    with client.websocket_connect("/ws/mic") as ws:
        with pytest.raises(WebSocketDisconnect) as exc:
            ws.receive_bytes()
    assert exc.value.code == VOICE_OFF_CLOSE


def test_newest_mic_connection_wins(no_network):
    sink = Sink()
    client = TestClient(create_app(voice=sink))
    with client.websocket_connect("/ws/mic") as old:
        with client.websocket_connect("/ws/mic") as new:
            old.send_bytes(b"\x03\x00")  # superseded tab: ignored
            new.send_bytes(b"\x04\x00")
    assert sink.frames == [b"\x04\x00"]
