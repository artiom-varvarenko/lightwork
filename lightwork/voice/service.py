"""Voice relay: browser PCM → paced upstream Muse Voice Transcribe session → C1 ``voice`` / ``voice_rejected`` events.

Threading: the service runs its own asyncio loop on a daemon thread ("lw-voice"), so nothing on the server's loop
(or any CV worker) can starve the pacer; a starved pacer means below-real-time ingress and close code 1008.
``feed_pcm``, ``set_muted``, ``on_tts``, ``set_keywords`` and ``step_changed`` are thread-safe and cheap.
``on_event`` and ``on_hud`` are called from the voice thread (``on_hud`` also from callers of the mute methods);
callers make their sinks thread-safe.

Pipeline (briefs/webvoice.md §1.2):
- ``feed_pcm`` appends to a ring that keeps only the newest 0.5 s; while TTS plays (plus a 300 ms tail) incoming
  audio is discarded, so the pacer sends silence instead.
- The pacer ticks on an absolute 80 ms schedule and always sends exactly 3840 B, zero-padded on underflow.
- One long-lived ENDPOINTING session; rotated at 55 min and on ``set_keywords`` (new session first, then
  ``endStream`` to the old one, which drains until it closes). Failures follow ``policy.close_action``.
- Only ``speechComplete`` can produce a command, once per (sessionId, turnId); a turn that started before the last
  ``step_changed``/``set_keywords`` is stale and logged as rejected. Exact grammar (``grammar.parse_command``).
- D14 fallback (``mode="file"`` or realtime access denied): RMS VAD on the paced stream cuts 0.3–4 s utterances that
  go to ``POST /asr/transcribe`` with ``PUSH_TO_TALK``; same grammar.
- ``dry_run``: full pipeline and HUD, but ``on_event`` is never called (the planner receives nothing).
"""

from __future__ import annotations

import asyncio
import json
import logging
import threading
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Literal

import httpx
from websockets.asyncio.client import ClientConnection, connect
from websockets.exceptions import ConnectionClosed, InvalidHandshake, InvalidStatus

from ..config import Settings
from ..contracts.events import make_payload
from .grammar import build_keywords, parse_command
from .pcm import FRAME_BYTES, FRAME_S, PcmRing, RmsVad, Utterance, wav_bytes
from .policy import REPEAT_1008_WINDOW_S, Action, Backoff, close_action, status_action

log = logging.getLogger(__name__)

MODEL = "muse-voice-transcribe-1.0"
ROTATE_AFTER_S = 55 * 60
HANDSHAKE_TIMEOUT_S = 10.0
TTS_TAIL_S = 0.3
TTS_WATCHDOG_S = 15.0  # a lost tts "end" must not mute the mic forever (utterances are <= 12 s)
SEND_STALL_FRAMES = 32  # 2.56 s queued for one socket: the network is gone, abort instead of building backlog
MAX_CATCHUP_S = 2.0
STABLE_SESSION_S = 30.0  # a session that lived this long resets the backoff
DRAIN_TIMEOUT_S = 10.0
STOP_DRAIN_TIMEOUT_S = 3.0
LATENCY_WINDOW_S = 10.0

Mode = Literal["realtime", "file"]


def realtime_url(api_base: str) -> str:
    """``https://host/v1`` → ``wss://host/v1/asr/realtime`` (``http`` → ``ws`` for local fakes)."""
    scheme, rest = api_base.split("://", 1)
    ws_scheme = "wss" if scheme == "https" else "ws"
    return f"{ws_scheme}://{rest}/asr/realtime"


@dataclass(eq=False)
class _Session:
    ws: ClientConnection
    session_id: str
    acked_at: float
    queue: asyncio.Queue = field(default_factory=asyncio.Queue)
    closed: asyncio.Event = field(default_factory=asyncio.Event)
    ending: bool = False
    aborted: bool = False
    error: str = ""
    close_code: int | None = None
    close_reason: str = ""
    turns: dict[Any, float] = field(default_factory=dict)  # turnId -> monotonic time of speechStart
    tasks: list[asyncio.Task] = field(default_factory=list)


class _ConnectError(Exception):
    def __init__(self, action: Action):
        super().__init__(action.why)
        self.action = action


class _VoiceOff(Exception):
    pass


class VoiceService:
    def __init__(
        self,
        settings: Settings,
        keywords: list[str],
        on_event: Callable[[str, dict], None],
        *,
        dry_run: bool = False,
        on_hud: Callable[[dict], None] | None = None,
        mode: Mode = "realtime",
        rotate_after_s: float = ROTATE_AFTER_S,
        language_bias: tuple[str, ...] = ("English",),
        http_transport: httpx.AsyncBaseTransport | None = None,
    ):
        self._settings = settings
        self._keywords = build_keywords(keywords)
        self._on_event = on_event
        self._on_hud = on_hud
        self.dry_run = dry_run
        self._mode: Mode = mode
        self._rotate_after_s = rotate_after_s
        self._language_bias = list(language_bias)
        self._http_transport = http_transport

        self._ring = PcmRing()
        self._vad = RmsVad()  # voice thread only
        self._lock = threading.Lock()
        self._muted = False
        self._tts: dict[str, float] = {}
        self._unmute_at = 0.0
        self._last_change = float("-inf")
        self._hud_state: dict[str, Any] = {
            "voice": "off", "turn": "idle", "heard": "", "command": None, "accepted": False,
            "dry_run": dry_run, "muted": False, "error": None, "session_id": None,
        }
        self._counters = {"bytes_in": 0, "bytes_muted": 0, "bytes_out": 0, "frames_paced": 0, "sessions": 0}
        self._closes: list[tuple[int | None, str]] = []
        self._latencies_ms: list[int] = []

        self._thread: threading.Thread | None = None
        self._started = threading.Event()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._stop = asyncio.Event()
        self._rekey = asyncio.Event()
        self._active: _Session | None = None
        self._retiring: set[asyncio.Task] = set()
        self._done_turns: set[tuple[str, str]] = set()
        self._utterances: asyncio.Queue[Utterance] = asyncio.Queue()
        self._pacer_reset = False
        self._last_1008 = float("-inf")
        self._http: httpx.AsyncClient | None = None

    # ------------------------------------------------------------------ public, thread-safe

    def start(self) -> None:
        if self._thread is not None:
            return
        if not self._settings.model_api_key:
            self._hud(voice="off", error="MODEL_API_KEY not set: voice input disabled")
            return
        self._thread = threading.Thread(target=lambda: asyncio.run(self._main()), name="lw-voice", daemon=True)
        self._thread.start()
        self._started.wait(5.0)

    def close(self, timeout: float = 5.0) -> None:
        if self._thread is None:
            return
        self._call_in_loop(self._stop.set)
        self._thread.join(timeout)

    def feed_browser_pcm(self, frame: bytes) -> None:
        """A frame from a browser page's mic (``/ws/mic``): it takes over from a server mic while it streams."""
        self._browser_t = time.monotonic()
        self.feed_pcm(frame)

    def browser_active(self, within_s: float = 1.0) -> bool:
        return time.monotonic() - getattr(self, "_browser_t", -1e9) < within_s

    def feed_pcm(self, frame: bytes) -> None:
        now = time.monotonic()
        with self._lock:
            self._counters["bytes_in"] += len(frame)
            muted = self._is_muted(now)
            if muted:
                self._counters["bytes_muted"] += len(frame)
        if not muted:
            self._ring.push(frame)

    def set_muted(self, muted: bool) -> None:
        with self._lock:
            if self._muted and not muted:
                self._unmute_at = time.monotonic() + TTS_TAIL_S
            self._muted = muted
        self._hud(muted=self._mute_flag())

    def on_tts(self, tts) -> None:
        """``render.Tts`` from the speaking page (RenderHub.on_tts callback): start mutes, end/error unmutes."""
        now = time.monotonic()
        with self._lock:
            self._tts = {k: t for k, t in self._tts.items() if now - t < TTS_WATCHDOG_S}
            if tts.state == "start":
                self._tts[tts.id] = now
            elif self._tts.pop(tts.id, None) is not None:
                self._unmute_at = now + TTS_TAIL_S
        self._hud(muted=self._mute_flag())

    def set_keywords(self, keywords: list[str]) -> None:
        """New procedure: new keywords need a new upstream session; turns already open become stale."""
        kws = build_keywords(keywords)
        with self._lock:
            self._keywords = kws
            self._last_change = time.monotonic()
        self._call_in_loop(self._rekey.set)

    def step_changed(self) -> None:
        """Called on every step transition: turns that started before now can no longer authorise a command."""
        with self._lock:
            self._last_change = time.monotonic()

    @property
    def keywords(self) -> list[str]:
        return list(self._keywords)

    @property
    def mode(self) -> Mode:
        return self._mode

    @property
    def hud(self) -> dict[str, Any]:
        with self._lock:
            return dict(self._hud_state)

    def stats(self) -> dict[str, Any]:
        with self._lock:
            return {
                **self._counters,
                "dropped_backlog_bytes": self._ring.dropped_bytes,
                "closes": list(self._closes),
                "latencies_ms": list(self._latencies_ms),
            }

    # ------------------------------------------------------------------ mute / HUD helpers

    def _is_muted(self, now: float) -> bool:
        return (
            self._muted
            or now < self._unmute_at
            or any(now - t < TTS_WATCHDOG_S for t in self._tts.values())
        )

    def _mute_flag(self) -> bool:
        """HUD value: muted by flag or an active TTS utterance (the 300 ms tail is not shown)."""
        now = time.monotonic()
        with self._lock:
            return self._muted or any(now - t < TTS_WATCHDOG_S for t in self._tts.values())

    def _hud(self, **kw: Any) -> None:
        with self._lock:
            new = {**self._hud_state, **kw}
            if new == self._hud_state:
                return
            self._hud_state = new
            snapshot = dict(new)
        if self._on_hud is not None:
            try:
                self._on_hud(snapshot)
            except Exception:
                log.exception("voice HUD callback failed")

    def _call_in_loop(self, fn: Callable[[], Any]) -> None:
        loop = self._loop
        if loop is None:
            return
        try:
            loop.call_soon_threadsafe(fn)
        except RuntimeError:  # loop already closed
            pass

    def _emit(self, type_: str, payload: dict) -> None:
        payload = make_payload(type_, payload)
        try:
            self._on_event(type_, payload)
        except Exception:
            log.exception("voice event sink failed")

    # ------------------------------------------------------------------ voice thread

    async def _main(self) -> None:
        self._loop = asyncio.get_running_loop()
        self._started.set()
        pacer = asyncio.create_task(self._pacer())
        try:
            async with httpx.AsyncClient(transport=self._http_transport, timeout=15.0) as http:
                self._http = http
                if self._mode == "realtime" and await self._run_realtime():
                    self._mode = "file"
                if self._mode == "file" and not self._stop.is_set():
                    await self._run_file()
        except _VoiceOff:
            pass
        except Exception as e:
            log.exception("voice service crashed")
            self._hud(voice="error", error=f"voice crashed: {e}")
        finally:
            pacer.cancel()
            await asyncio.gather(pacer, return_exceptions=True)
            if self.hud["voice"] != "error":
                self._hud(voice="off")

    async def _pacer(self) -> None:
        loop = asyncio.get_running_loop()
        t0, k = loop.time(), 0
        while True:
            k += 1
            delay = t0 + k * FRAME_S - loop.time()
            if delay > 0:
                await asyncio.sleep(delay)
            elif -delay > MAX_CATCHUP_S:
                log.warning("voice pacer %.2f s late; resyncing", -delay)
                t0, k = loop.time(), 0
            if self._pacer_reset:
                self._pacer_reset = False
                self._ring.clear()
                t0, k = loop.time(), 0
            self._tick(self._ring.pop(FRAME_BYTES), time.monotonic())

    def _tick(self, chunk: bytes, now: float) -> None:
        self._counters["frames_paced"] += 1
        utt = self._vad.push(chunk, now)
        if self._mode == "file":
            if utt is not None:
                self._utterances.put_nowait(utt)
            return
        sess = self._active
        if sess is None or sess.ending or sess.aborted:
            return
        if sess.queue.qsize() >= SEND_STALL_FRAMES:
            sess.aborted = True
            sess.error = "upstream send stalled"
            log.warning("voice: upstream send stalled, aborting session %s", sess.session_id)
            sess.ws.transport.abort()
            return
        sess.queue.put_nowait(chunk)

    # ------------------------------------------------------------------ realtime sessions

    async def _run_realtime(self) -> bool:
        """Run sessions until stop or voice off. Returns True when the file fallback should take over."""
        backoff = Backoff()
        current: _Session | None = None
        want_new = True
        while not self._stop.is_set():
            if want_new:
                if current is None:
                    self._hud(voice="connecting" if self._counters["sessions"] == 0 else "reconnecting")
                try:
                    new = await self._connect_or_stop()
                except _ConnectError as e:
                    if current is not None:
                        self._retire(current)
                        current = self._active = None
                    verdict = await self._handle(e.action, backoff)
                    if verdict != "retry":
                        return verdict == "fallback"
                    continue
                if new is None:  # stopped while connecting
                    break
                old, current = current, new
                self._active = new
                if old is not None:
                    self._retire(old)
                want_new = False
                self._hud(voice="live", session_id=new.session_id, error=None)
            why = await self._wait(current)
            if why == "stop":
                break
            if why in ("rotate", "rekey"):
                want_new = True
                continue
            self._active = None
            now = time.monotonic()
            if now - current.acked_at >= STABLE_SESSION_S:
                backoff.reset()
            reason = current.error or current.close_reason
            action = close_action(current.close_code, reason, acked=True, ended_by_us=False,
                                  repeat_1008=now - self._last_1008 < REPEAT_1008_WINDOW_S)
            if current.close_code == 1008:
                log.error("voice: upstream closed with 1008 mid-stream (pacing bug?): %s", reason)
                self._last_1008 = now
                self._pacer_reset = True
            if action.kind == "reconnect" and now - current.acked_at < STABLE_SESSION_S:
                action = Action("backoff", action.why, 0.5)
            current = None
            want_new = True
            verdict = await self._handle(action, backoff)
            if verdict != "retry":
                return verdict == "fallback"
        if current is not None:
            self._retire(current)
            self._active = None
        if self._retiring:
            done, pending = await asyncio.wait(self._retiring, timeout=STOP_DRAIN_TIMEOUT_S)
            for t in pending:
                t.cancel()
        return False

    async def _handle(self, action: Action, backoff: Backoff) -> str:
        if action.kind in ("reconnect", "done"):
            self._hud(voice="reconnecting", error=action.why)
            return "retry"
        if action.kind == "off":
            log.error("voice off: %s", action.why)
            self._hud(voice="error", error=action.why)
            return "off"
        if action.kind == "fallback":
            log.warning("voice: %s; using the file endpoint", action.why)
            self._hud(voice="fallback", error=action.why, session_id=None)
            return "fallback"
        delay = backoff.next(action)
        if delay is None:
            why = f"gave up after {backoff.tries} tries: {action.why}"
            log.error("voice off: %s", why)
            self._hud(voice="error", error=why)
            return "off"
        self._hud(voice="reconnecting", error=action.why)
        try:
            await asyncio.wait_for(self._stop.wait(), delay)
        except TimeoutError:
            pass
        return "retry"

    async def _connect_or_stop(self) -> _Session | None:
        conn = asyncio.create_task(self._connect())
        stop = asyncio.create_task(self._stop.wait())
        await asyncio.wait({conn, stop}, return_when=asyncio.FIRST_COMPLETED)
        stop.cancel()
        if not conn.done():
            conn.cancel()
            await asyncio.gather(conn, return_exceptions=True)
            return None
        return conn.result()

    async def _connect(self) -> _Session:
        self._rekey.clear()
        keywords = self._keywords
        sid = f"lw-{uuid.uuid4()}"
        url = f"{realtime_url(self._settings.meta_api_base)}?sessionId={sid}"
        try:
            ws = await connect(url, open_timeout=HANDSHAKE_TIMEOUT_S, ping_interval=10, ping_timeout=10,
                               close_timeout=3, compression=None, max_size=2**20)
        except InvalidStatus as e:
            raise _ConnectError(status_action(e.response.status_code)) from e
        except (OSError, TimeoutError, InvalidHandshake) as e:
            raise _ConnectError(Action("backoff", f"connect failed: {e}", 0.5, limited=False)) from e
        handshake = {
            "authorization": {"accessToken": f"Bearer {self._settings.model_api_key}"},
            "audioEncoding": "PCM_24KHZ",
            "model": MODEL,
            "mode": "ENDPOINTING",
            "partialMode": "CUMULATIVE",
            "emitAudioProgress": False,
            "keywords": keywords,
            "languageBias": self._language_bias,
        }
        error = ""
        try:
            await ws.send(json.dumps(handshake))
            async with asyncio.timeout(HANDSHAKE_TIMEOUT_S):
                while True:
                    ev = _json_obj(await ws.recv())
                    if ev is None:
                        continue
                    if "type" not in ev:
                        session_id = str(ev.get("sessionId") or sid)
                        break
                    if ev["type"] == "error":
                        error = str(ev.get("message") or "")
        except TimeoutError as e:
            await ws.close()
            raise _ConnectError(Action("backoff", "no handshake acknowledgement within 10 s", 0.5)) from e
        except ConnectionClosed as e:
            self._record_close(ws.close_code, error or ws.close_reason or "")
            raise _ConnectError(close_action(ws.close_code, error or ws.close_reason or "", acked=False,
                                             ended_by_us=False, repeat_1008=False)) from e
        sess = _Session(ws=ws, session_id=session_id, acked_at=time.monotonic())
        sess.tasks = [asyncio.create_task(self._receiver(sess)), asyncio.create_task(self._sender(sess))]
        with self._lock:
            self._counters["sessions"] += 1
        log.info("voice: session %s open (keywords=%s)", session_id, keywords)
        return sess

    async def _wait(self, sess: _Session) -> str:
        waits = {
            asyncio.create_task(self._stop.wait()): "stop",
            asyncio.create_task(sess.closed.wait()): "closed",
            asyncio.create_task(self._rekey.wait()): "rekey",
        }
        timeout = max(0.0, sess.acked_at + self._rotate_after_s - time.monotonic())
        done, pending = await asyncio.wait(waits, timeout=timeout, return_when=asyncio.FIRST_COMPLETED)
        for t in pending:
            t.cancel()
        names = {waits[t] for t in done}
        for name in ("stop", "closed", "rekey"):
            if name in names:
                return name
        return "rotate"

    def _retire(self, sess: _Session) -> None:
        """Send ``endStream`` after the queued audio and drain the session until it closes."""
        if sess.ending:
            return
        sess.ending = True
        sess.queue.put_nowait(None)

        async def drain() -> None:
            try:
                await asyncio.wait_for(sess.closed.wait(), DRAIN_TIMEOUT_S)
            except TimeoutError:
                sess.ws.transport.abort()
            except asyncio.CancelledError:
                sess.ws.transport.abort()
                raise

        task = asyncio.create_task(drain())
        self._retiring.add(task)
        task.add_done_callback(self._retiring.discard)

    async def _sender(self, sess: _Session) -> None:
        try:
            while True:
                chunk = await sess.queue.get()
                if chunk is None:
                    await sess.ws.send(json.dumps({"type": "endStream"}))
                    return
                await sess.ws.send(chunk)
                self._counters["bytes_out"] += len(chunk)
        except ConnectionClosed:
            pass

    async def _receiver(self, sess: _Session) -> None:
        try:
            async for msg in sess.ws:
                ev = _json_obj(msg)
                if ev is not None:
                    self._on_server_event(sess, ev)
        except ConnectionClosed:
            pass
        finally:
            sess.close_code = sess.ws.close_code
            sess.close_reason = sess.ws.close_reason or ""
            self._record_close(sess.close_code, sess.error or sess.close_reason)
            for t in sess.tasks:
                if t is not asyncio.current_task():
                    t.cancel()
            sess.closed.set()

    def _record_close(self, code: int | None, reason: str) -> None:
        with self._lock:
            self._closes.append((code, reason))

    def _on_server_event(self, sess: _Session, ev: dict) -> None:
        typ = ev.get("type")
        if typ == "speechStart":
            sess.turns[ev.get("turnId")] = time.monotonic()
            self._hud(turn="speech")
        elif typ == "transcript":
            if not ev.get("final"):
                self._hud(heard=str(ev.get("transcript") or ""))
        elif typ == "speechEnd":
            self._hud(turn="idle")
        elif typ == "speechComplete":
            turn = ev.get("turnId")
            started = sess.turns.pop(turn, sess.acked_at)
            self._complete(sess.session_id, str(turn), str(ev.get("transcript") or ""), started)
        elif typ == "error":
            sess.error = str(ev.get("message") or "")
            log.warning("voice: upstream error on %s: %s", sess.session_id, sess.error)
            self._hud(error=sess.error)

    # ------------------------------------------------------------------ completion → events

    def _complete(self, session_id: str, turn_id: str, transcript: str, started: float) -> None:
        key = (session_id, turn_id)
        if key in self._done_turns:
            return
        self._done_turns.add(key)
        now = time.monotonic()
        text = transcript.strip()
        if not text:
            self._hud(turn="idle")
            return
        end_t = self._vad.last_end_t
        if end_t is not None and 0.0 <= now - end_t < LATENCY_WINDOW_S:
            with self._lock:
                self._latencies_ms.append(round((now - end_t) * 1000))
        with self._lock:
            stale = started < self._last_change
        command = None if stale else parse_command(text)
        if stale:
            log.info("voice: stale turn %s/%s rejected: %r", session_id, turn_id, text)
        self._hud(turn="idle", heard=text, command=command, accepted=command is not None)
        if self.dry_run:
            log.info("voice dry-run: %r -> %s", text, command)
            return
        if command is not None:
            self._emit("voice", {"command": command, "transcript": text, "session_id": session_id,
                                 "turn_id": turn_id})
        else:
            self._emit("voice_rejected", {"transcript": text})

    # ------------------------------------------------------------------ D14 file fallback

    async def _run_file(self) -> None:
        self._hud(voice="fallback", session_id=None)
        base_sid = f"lw-file-{uuid.uuid4()}"
        n = 0
        while True:
            get = asyncio.create_task(self._utterances.get())
            stop = asyncio.create_task(self._stop.wait())
            done, _ = await asyncio.wait({get, stop}, return_when=asyncio.FIRST_COMPLETED)
            if stop in done:
                get.cancel()
                return
            stop.cancel()
            n += 1
            await self._transcribe(get.result(), f"{base_sid}-{n}", str(n))

    async def _transcribe(self, utt: Utterance, session_id: str, turn_id: str) -> None:
        assert self._http is not None
        request = {
            "mode": "PUSH_TO_TALK",
            "model": MODEL,
            "audioEncoding": "WAV",
            "keywords": self._keywords,
            "languageBias": self._language_bias,
        }
        files = {
            "request": (None, json.dumps(request), "application/json"),
            "audio": ("utterance.wav", wav_bytes(utt.pcm), "audio/wav"),
        }
        url = f"{self._settings.meta_api_base}/asr/transcribe"
        headers = {"Authorization": f"Bearer {self._settings.model_api_key}"}
        for attempt in range(2):
            try:
                r = await self._http.post(url, params={"sessionId": session_id}, headers=headers, files=files)
            except httpx.HTTPError as e:
                problem, status = f"transcribe failed: {e}", None
            else:
                if r.status_code == 200:
                    break
                problem, status = f"transcribe HTTP {r.status_code}: {r.text[:200]}", r.status_code
            if status in (401, 403, 404):
                self._hud(voice="error", error=problem)
                raise _VoiceOff(problem)
            if attempt or status not in (None, 429, 500, 503):
                log.warning("voice: %s", problem)
                self._hud(error=problem)
                return
            await asyncio.sleep(2.0 if status == 429 else 0.5)
        body = r.json()
        self._complete(str(body.get("sessionId") or session_id), turn_id, str(body.get("transcript") or ""),
                       utt.t_start)


def _json_obj(msg: str | bytes) -> dict | None:
    if not isinstance(msg, str):
        return None
    try:
        ev = json.loads(msg)
    except ValueError:
        return None
    return ev if isinstance(ev, dict) else None
