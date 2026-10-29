"""C8 Server constants. Bind 127.0.0.1:8000.

HTTP: ``GET /projector``, ``/screen``, ``/viewer?run=``, ``/api/runs``, ``/api/runs/{id}/{path}``, ``/frames/{sha}``.
WebSockets: ``/ws/render`` (C2), ``/ws/camera``, ``/ws/control``, ``/ws/mic``.
``lightwork/server/app.py`` includes every ``lightwork/server/routes/*.py`` module exposing ``router``.
"""

from __future__ import annotations

import struct

HOST = "127.0.0.1"
PORT = 8000

# /ws/camera: binary frames = 12-byte header (u32 frame_id, f64 t_mono, little-endian) + JPEG (960x540, q~70).
CAMERA_HEADER = struct.Struct("<Id")
CAMERA_PREVIEW_SIZE = (960, 540)
CAMERA_PREVIEW_QUALITY = 70

# /ws/mic: PCM16 LE mono 24 kHz, 80 ms frames.
MIC_RATE = 24_000
MIC_FRAME_MS = 80
MIC_FRAME_SAMPLES = MIC_RATE * MIC_FRAME_MS // 1000  # 1920
MIC_FRAME_BYTES = MIC_FRAME_SAMPLES * 2  # 3840

WS_RENDER = "/ws/render"
WS_CAMERA = "/ws/camera"
WS_CONTROL = "/ws/control"
WS_MIC = "/ws/mic"


def pack_camera_frame(frame_id: int, t_mono: float, jpeg: bytes) -> bytes:
    return CAMERA_HEADER.pack(frame_id & 0xFFFFFFFF, t_mono) + jpeg


def unpack_camera_frame(data: bytes) -> tuple[int, float, bytes]:
    frame_id, t = CAMERA_HEADER.unpack_from(data)
    return frame_id, t, data[CAMERA_HEADER.size :]
