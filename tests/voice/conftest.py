"""Voice test doubles, exposed as fixtures: a local Muse Voice Transcribe realtime server and a real-time PCM feeder.

``FakeUpstream`` enforces the documented session rules (meta-docs/speech-to-text.md "Keep the session alive",
cookbook__voice-api-fundamentals.md "Cutoff behaviour"): handshake first within the deadline with the right fields
and key, then binary audio only; >backlog_limit_s ahead of real time → 1008; below real time by more than
below_rt_limit_s → 1008; session older than max_session_s → 1011 "Max session duration reached"; ``endStream`` →
flush and close 1000. Limits are configurable so tests run in seconds. It emulates ENDPOINTING on the audio itself:
a run of loud frames is a turn; after ``silence_frames_end`` quiet frames it emits speechEnd and speechComplete with
the next scripted transcript.
"""

from __future__ import annotations

import asyncio
import json
import math
import threading
import time
import uuid
from dataclasses import dataclass, field
from types import SimpleNamespace
from urllib.parse import parse_qs, urlsplit

import numpy as np
import pytest
from websockets.asyncio.server import ServerConnection, serve
from websockets.exceptions import ConnectionClosed

from lightwork.config import Settings
from lightwork.voice import VoiceService

BYTES_PER_S = 48_000
FRAME_BYTES = 3840
LOUD = 2000


def tone_frame(amplitude: int = 8000, freq: float = 440.0) -> bytes:
    t = np.arange(FRAME_BYTES // 2) / 24_000
    return (amplitude * np.sin(2 * math.pi * freq * t)).astype("<i2").tobytes()


SILENCE = bytes(FRAME_BYTES)
TONE = tone_frame()


@dataclass
class FakeSession:
    id: str
    handshake: dict | None = None
    bytes: int = 0
    frames: list[tuple[float, int]] = field(default_factory=list)
    close_code: int | None = None
    close_reason: str = ""
    ended_by_client: bool = False
    turns: int = 0
    ws: ServerConnection | None = None


class FakeUpstream:
    def __init__(self, *, key: str = "LLM_TEST", backlog_limit_s: float = 5.0, below_rt_limit_s: float = 10.0,
                 handshake_timeout_s: float = 10.0, max_session_s: float | None = None,
                 reject_status: int | None = None, silence_frames_end: int = 3, duplicate_complete: bool = False):
        self.key = key
        self.backlog_limit_s = backlog_limit_s
        self.below_rt_limit_s = below_rt_limit_s
        self.handshake_timeout_s = handshake_timeout_s
        self.max_session_s = max_session_s
        self.reject_status = reject_status
        self.silence_frames_end = silence_frames_end
        self.duplicate_complete = duplicate_complete
        self.sessions: list[FakeSession] = []
        self.connections = 0
        self._script: list[str] = []
        self._lock = threading.Lock()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._stop: asyncio.Event | None = None
        self._ready = threading.Event()
        self._thread: threading.Thread | None = None
        self.port = 0

    # ---------------------------------------------------------------- control (test thread)

    def start(self) -> str:
        self._thread = threading.Thread(target=lambda: asyncio.run(self._main()), daemon=True)
        self._thread.start()
        assert self._ready.wait(5)
        return f"http://127.0.0.1:{self.port}/v1"

    def stop(self) -> None:
        if self._loop is not None:
            self._loop.call_soon_threadsafe(self._stop.set)
        if self._thread is not None:
            self._thread.join(5)

    def say(self, *texts: str) -> None:
        with self._lock:
            self._script.extend(texts)

    def close_active(self, code: int, reason: str = "") -> None:
        sess = next(s for s in reversed(self.sessions) if s.ws is not None and s.close_code is None)
        fut = asyncio.run_coroutine_threadsafe(self._close(sess.ws, sess, code, reason), self._loop)
        fut.result(5)

    def open_sessions(self) -> list[FakeSession]:
        return [s for s in self.sessions if s.handshake is not None and s.close_code is None]

    # ---------------------------------------------------------------- server (fake thread)

    async def _main(self) -> None:
        self._loop = asyncio.get_running_loop()
        self._stop = asyncio.Event()

        def process_request(connection, request):
            self.connections += 1
            if self.reject_status is not None:
                return connection.respond(self.reject_status, "denied\n")
            if not request.path.startswith("/v1/asr/realtime"):
                return connection.respond(404, "not found\n")
            return None

        async with serve(self._handler, "127.0.0.1", 0, process_request=process_request) as server:
            self.port = server.sockets[0].getsockname()[1]
            self._ready.set()
            await self._stop.wait()

    async def _close(self, ws: ServerConnection, sess: FakeSession, code: int, reason: str) -> None:
        if sess.close_code is None:
            sess.close_code, sess.close_reason = code, reason
        await ws.close(code, reason)

    async def _send(self, ws: ServerConnection, ev: dict) -> None:
        await ws.send(json.dumps(ev))

    async def _handler(self, ws: ServerConnection) -> None:
        query = parse_qs(urlsplit(ws.request.path).query)
        sess = FakeSession(id=query.get("sessionId", [f"fake-{uuid.uuid4()}"])[0])
        self.sessions.append(sess)
        try:
            msg = await asyncio.wait_for(ws.recv(), self.handshake_timeout_s)
        except TimeoutError:
            await self._close(ws, sess, 1008, "handshake timeout")
            return
        except ConnectionClosed:
            return
        problem = self._validate(msg)
        if problem:
            await self._send(ws, {"type": "error", "message": problem, "sessionId": sess.id})
            await self._close(ws, sess, 1008, problem)
            return
        sess.handshake = json.loads(msg)
        sess.ws = ws
        await self._send(ws, {"sessionId": sess.id})
        t_ack = time.monotonic()
        police = asyncio.create_task(self._police(ws, sess, t_ack))
        in_speech, quiet = False, 0
        try:
            async for msg in ws:
                if isinstance(msg, str):
                    ev = json.loads(msg)
                    if ev.get("type") == "endStream":
                        sess.ended_by_client = True
                        if in_speech:
                            await self._finish_turn(ws, sess)
                        await self._close(ws, sess, 1000, "")
                        return
                    await self._close(ws, sess, 1008, "unexpected text frame")
                    return
                if sess.ended_by_client:
                    await self._close(ws, sess, 1008, "audio after endStream")
                    return
                sess.bytes += len(msg)
                sess.frames.append((time.monotonic(), len(msg)))
                loud = int(np.abs(np.frombuffer(msg, dtype="<i2").astype(np.int32)).max(initial=0)) > LOUD
                if loud and not in_speech:
                    in_speech, quiet = True, 0
                    sess.turns += 1
                    await self._send(ws, {"type": "speechStart", "turnId": sess.turns})
                    await self._send(ws, {"type": "transcript", "transcript": "…", "final": False})
                elif in_speech and not loud:
                    quiet += 1
                    if quiet >= self.silence_frames_end:
                        in_speech = False
                        await self._finish_turn(ws, sess)
                elif loud:
                    quiet = 0
        except ConnectionClosed:
            pass
        finally:
            police.cancel()
            if sess.close_code is None:
                sess.close_code, sess.close_reason = ws.close_code, ws.close_reason or ""

    async def _finish_turn(self, ws: ServerConnection, sess: FakeSession) -> None:
        with self._lock:
            text = self._script.pop(0) if self._script else ""
        await self._send(ws, {"type": "speechEnd", "turnId": sess.turns})
        ev = {"type": "speechComplete", "turnId": sess.turns, "transcript": text}
        await self._send(ws, ev)
        if self.duplicate_complete:
            await self._send(ws, ev)

    def _validate(self, msg) -> str | None:
        if not isinstance(msg, str):
            return "first frame must be the JSON handshake"
        try:
            hs = json.loads(msg)
        except ValueError:
            return "handshake is not JSON"
        token = (hs.get("authorization") or {}).get("accessToken")
        if token != f"Bearer {self.key}":
            return "invalid credentials"
        if hs.get("audioEncoding") not in ("PCM_24KHZ", "PCM_16KHZ"):
            return "audioEncoding required"
        if hs.get("model") != "muse-voice-transcribe-1.0":
            return "model not found"
        if hs.get("mode") not in (None, "PUSH_TO_TALK", "ENDPOINTING", "DIARIZATION"):
            return "invalid mode"
        return None

    async def _police(self, ws: ServerConnection, sess: FakeSession, t_ack: float) -> None:
        while True:
            await asyncio.sleep(0.05)
            elapsed = time.monotonic() - t_ack
            audio_s = sess.bytes / BYTES_PER_S
            if audio_s - elapsed > self.backlog_limit_s:
                await self._close(ws, sess, 1008, "audio backlog exceeded")
                return
            if elapsed - audio_s > self.below_rt_limit_s and not sess.ended_by_client:
                await self._close(ws, sess, 1008, "ingress below real time")
                return
            if self.max_session_s is not None and elapsed > self.max_session_s:
                await self._close(ws, sess, 1011, "Max session duration reached")
                return


class Feeder:
    """Pushes 80 ms frames into ``sink.feed_pcm`` on an absolute real-time schedule, like the browser does.

    ``plan`` items: ``("tone"|"silence", seconds)`` feed frames, ``("gap", seconds)`` feeds nothing (tab closed),
    ``("burst", seconds)`` feeds that much audio at once (browser catching up after a stall).
    """

    def __init__(self, sink, plan: list[tuple[str, float]]):
        self.sink = sink
        self.plan = plan
        self.thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> Feeder:
        self.thread.start()
        return self

    def join(self, timeout: float = 30) -> None:
        self.thread.join(timeout)

    def _run(self) -> None:
        t = time.monotonic()
        for kind, seconds in self.plan:
            n = round(seconds / 0.08)
            if kind == "gap":
                t += n * 0.08
                continue
            if kind == "burst":
                for _ in range(n):
                    self.sink.feed_pcm(TONE)
                continue
            frame = TONE if kind == "tone" else SILENCE
            for _ in range(n):
                t += 0.08
                time.sleep(max(0.0, t - time.monotonic()))
                self.sink.feed_pcm(frame)


def wait_until(pred, timeout: float = 5.0, interval: float = 0.02) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if pred():
            return True
        time.sleep(interval)
    return pred()


@pytest.fixture
def voice_kit():
    return SimpleNamespace(TONE=TONE, SILENCE=SILENCE, FRAME_BYTES=FRAME_BYTES, Feeder=Feeder, wait_until=wait_until)


@pytest.fixture
def fake_upstream(no_network):
    """Factory: ``fake_upstream(**limits) -> (FakeUpstream, api_base)``; servers stop at teardown."""
    servers: list[FakeUpstream] = []

    def make(**kw) -> tuple[FakeUpstream, str]:
        fake = FakeUpstream(**kw)
        servers.append(fake)
        return fake, fake.start()

    yield make
    for fake in servers:
        fake.stop()


@pytest.fixture
def voice_service():
    """Factory: ``voice_service(api_base, **kw) -> (VoiceService, events, huds)``, started; closed at teardown."""
    services: list[VoiceService] = []

    def make(api_base: str, *, key: str = "LLM_TEST", keywords=("pipette",), **kw):
        events: list[tuple[str, dict]] = []
        huds: list[dict] = []
        svc = VoiceService(Settings(model_api_key=key, meta_api_base=api_base), list(keywords),
                           lambda t, p: events.append((t, p)), on_hud=huds.append, **kw)
        services.append(svc)
        svc.start()
        return svc, events, huds

    yield make
    for svc in services:
        svc.close()
