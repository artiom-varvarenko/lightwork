"""Hubs between worker threads and the pages (C2, C8). One instance of each lives on ``app.state``.

Every public method that worker threads call is thread-safe: state changes happen under a ``threading`` lock and the
per-connection asyncio senders are woken with ``loop.call_soon_threadsafe``. Page callbacks (``on_tts``,
``on_action``) run on one dispatcher thread, never on the event loop, so they may block briefly.

``RenderHub`` (``/ws/render``)
    Full scene snapshots, latest wins, at most ``MAX_UNACKED`` scenes unacknowledged per page. The wire ``seq`` of a
    scene is a per-page counter assigned at send time (the reducer's own ``seq`` is not sent); a page ACKs the seq it
    drew after ``requestAnimationFrame`` and an ACK covers every older seq. The current scene is re-sent as a
    heartbeat after ``HEARTBEAT_S`` without a scene, which keeps the page's "LINK LOST" timer quiet.
    ``show_mode`` replaces the scene with a bare measurement/pattern scene until ``show_mode("normal")`` and waits for
    the link page (newest projector, else newest screen) to ACK it. ``set_view`` turns the projector page (C2
    ``view``); every page gets the current view on connect, before its first scene. ``say`` and ``sfx`` queue on the
    link page only and go out before the next scene.
``PreviewHub`` (``/ws/camera``)
    Binary frames ``CAMERA_HEADER`` + JPEG, latest wins, at most ``MAX_UNACKED`` frames in flight per viewer; the page
    ACKs with the text message ``{"ack": frame_id}`` after drawing.
``ControlHub`` (``/ws/control``)
    Server -> page: ``{"type": "state", "state": {...}}``, always the full merged state (latest wins).
    ``push_state(patch)`` merges top-level keys. Keys the screen page renders:
      ``run`` {id, pack, mode: live|replay|layout|idle, backend, provenance}; ``banner`` str|null;
      ``camera`` {w, h} (camera px of the scene prims, default 1920x1080);
      ``preview_xywh`` [x, y, w, h] camera-px rect the preview frames show (default the whole camera);
      ``steps`` [{id, say, state}]; ``active`` step id; ``chips`` [{step, check, status, reason}];
      ``trace`` [{intent_id, role, model, status, latency_ms, cost_usd}]; ``cost_usd`` float;
      ``hud`` {voice, turn, heard, ...}; ``last_say`` str;
      ``layout`` {objects: [{id, kind, noun?, corners_cam_px?, polygon_cam_px?}], work_area_xywh?}.
    ``collection`` {session_id, seed, index (zero-based), total, card, phase, frame_sha?, error?};
    trial capture/review controls emit ``trial_capture``, ``trial_accept``, ``trial_reject``, ``trial_next``.
    ``layout.message`` is operator guidance; layout_capture/layout_suggest/layout_noun actions carry object id.
    Page -> server: ``{"type": "ui", "action": str, "args": {...}}``. Actions sent by ``screen.html``:
      ``start`` {tts, mic}; ``confirm``, ``repeat``, ``why``, ``back``, ``stop`` {};
      ``layout_corners`` {object, corners_cam_px: [TL, TR, BR, BL]}; ``layout_rect`` {object, xywh};
      ``layout_work_area`` {xywh}; ``layout_accept`` {}.  All coordinates are camera px.
"""

from __future__ import annotations

import asyncio
import json
import logging
import queue
import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from pydantic import TypeAdapter, ValidationError

from ..contracts import render
from ..contracts.server import CAMERA_PREVIEW_QUALITY, CAMERA_PREVIEW_SIZE, pack_camera_frame

log = logging.getLogger(__name__)

MAX_UNACKED = 2
HEARTBEAT_S = 1.0
LATENCY_WINDOW = 2000

_page_msg = TypeAdapter(render.PageToServer)


class _Dispatcher:
    """Runs callbacks in order on one daemon thread."""

    def __init__(self, name: str):
        self._name = name
        self._q: queue.SimpleQueue[tuple[Callable[..., Any], tuple] | None] = queue.SimpleQueue()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._closed = False

    def submit(self, fn: Callable[..., Any], *args: Any) -> None:
        with self._lock:
            if self._closed:
                raise RuntimeError(f"{self._name} is closed")
            if self._thread is None:
                self._thread = threading.Thread(target=self._loop, name=self._name, daemon=True)
                self._thread.start()
            self._q.put((fn, args))

    def close(self, timeout: float = 5.0) -> None:
        """Drain accepted callbacks and join the worker; call after websocket producers stop."""
        with self._lock:
            if not self._closed:
                self._closed = True
                self._q.put(None)
            thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout)
            if thread.is_alive():
                raise TimeoutError(f"{self._name} callback did not stop within {timeout}s")

    def _loop(self) -> None:
        while True:
            item = self._q.get()
            if item is None:
                return
            fn, args = item
            try:
                fn(*args)
            except Exception:  # a broken callback must not stop later ones
                log.exception("%s callback failed", self._name)


class _Conn:
    """One websocket connection: an asyncio wake event that worker threads can set."""

    def __init__(self) -> None:
        self.loop = asyncio.get_running_loop()
        self.wake = asyncio.Event()

    def poke(self) -> None:
        try:
            self.loop.call_soon_threadsafe(self.wake.set)
        except RuntimeError:  # loop already closed (server shutting down)
            pass

    async def wait(self, timeout: float | None) -> None:
        try:
            await asyncio.wait_for(self.wake.wait(), timeout)
        except TimeoutError:
            pass
        self.wake.clear()


async def _stop(task: asyncio.Task) -> None:
    task.cancel()
    try:
        await task
    except BaseException:  # the sender's own error or cancellation; the connection is over either way
        pass


def _percentile(values: list[float], p: float) -> float | None:
    if not values:
        return None
    s = sorted(values)
    return s[min(len(s) - 1, int(round(p / 100 * (len(s) - 1))))]


# ------------------------------------------------------------------------------------------------ render


@dataclass(eq=False)
class _Page(_Conn):
    hello: render.Hello
    connected_at: float = field(default_factory=time.monotonic)
    next_seq: int = 1
    outstanding: dict[int, tuple[int, float]] = field(default_factory=dict)  # wire seq -> (version, t_sent)
    sent_version: int = 0
    last_scene_t: float = 0.0
    calib_version: int = 0
    view_version: int = 0
    says: deque[render.SayMsg] = field(default_factory=deque)
    sfx: deque[render.Sfx] = field(default_factory=deque)
    acked_version: int = 0
    acked_versions: deque[tuple[int, float]] = field(default_factory=lambda: deque(maxlen=64))  # (version, first ACK t)
    last_ack: render.Ack | None = None
    last_ack_t: float | None = None
    latencies_ms: deque[float] = field(default_factory=lambda: deque(maxlen=LATENCY_WINDOW))
    max_unacked: int = 0
    scenes_sent: int = 0

    def __post_init__(self) -> None:
        _Conn.__init__(self)


class RenderHub:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._cond = threading.Condition(self._lock)
        self._scene = render.SceneMsg(seq=0, mode="normal", prims=[])
        self._mode_scene: render.SceneMsg | None = None
        self._version = 1
        self._effective = self._dump_effective()
        self._calib: render.Calib | None = None
        self._calib_version = 0
        self._view: render.View | None = None
        self._view_version = 0
        self._pages: list[_Page] = []
        self._say_n = 0
        self._tts_cbs: list[Callable[[render.Tts], None]] = []
        self._dispatch = _Dispatcher("render-hub-callbacks")

    # ---- worker-thread API

    def publish_scene(self, scene: render.SceneMsg) -> int:
        """Make ``scene`` the current one; returns its version, for :meth:`acked_at`."""
        with self._lock:
            self._scene = scene
            if self._mode_scene is None:
                self._bump()
            return self._version

    def version(self) -> int:
        """Version of the scene pages are being sent now."""
        with self._lock:
            return self._version

    def acked_at(self, version: int) -> float | None:
        """Laptop ``time.monotonic()`` of the link page's first ACK covering ``version`` (or a newer one), or None while
        it has not drawn it. Non-blocking, for frame loops that must know when a scene is on the surface."""
        with self._lock:
            page = self._link_page()
            if page is None or page.acked_version < version:
                return None
            for v, t in page.acked_versions:
                if v >= version:
                    return t
            return page.last_ack_t

    def set_calib(self, calib: render.Calib) -> None:
        with self._lock:
            self._calib = calib
            self._calib_version += 1
            self._poke_all()

    def set_view(self, view: render.View) -> None:
        """Turn everything the projector page draws (C2 ``view``). Set it before a pattern flash: a mapping solved
        from an unturned pattern does not hold for a turned page."""
        with self._lock:
            self._view = view
            self._view_version += 1
            self._poke_all()

    def say(self, text: str, priority: str = "normal") -> str:
        """Queue speech on the speaking page (newest projector, else newest screen). Dropped when neither is
        connected; the returned id then never appears in a ``tts`` report."""
        with self._lock:
            self._say_n += 1
            msg = render.SayMsg(id=f"say-{self._say_n}", text=text, priority=priority)
            page = self._link_page()
            if page is not None:
                page.says.append(msg)
                page.poke()
            return msg.id

    def sfx(self, name: str, gain: float = 1.0) -> None:
        """Queue a short UI sound (C2 ``sfx``) on the link page, like :meth:`say`: dropped when neither page is
        connected. ``gain`` is clamped to 0..1. The projector page stays silent until its tap to start."""
        with self._lock:
            msg = render.Sfx(name=name, gain=min(1.0, max(0.0, float(gain))))
            page = self._link_page()
            if page is not None:
                page.sfx.append(msg)
                page.poke()

    def show_mode(self, mode: str, pattern: render.Pattern | None = None, timeout: float = 1.0) -> float | None:
        """PageLink (B, D): switch every page to ``mode`` and wait until the link page ACKs a draw of it.
        Returns the laptop ``time.monotonic()`` at which the ACK arrived, or None on timeout / no page.
        Must not be called from the event loop thread (it blocks)."""
        if mode not in ("normal", "pattern", "dark", "grey", "white"):
            raise ValueError(f"unknown mode {mode!r}")
        if (mode == "pattern") != (pattern is not None):
            raise ValueError("pattern is required for mode 'pattern' and only for it")
        if self._on_loop_thread():
            raise RuntimeError("RenderHub.show_mode blocks; call it from a worker thread")
        deadline = time.monotonic() + timeout
        with self._cond:
            self._mode_scene = None if mode == "normal" else render.SceneMsg(seq=0, mode=mode, prims=[], pattern=pattern)
            self._bump()
            target = self._version
            while True:
                page = self._link_page()
                if page is not None and page.acked_version >= target:
                    return page.last_ack_t
                left = deadline - time.monotonic()
                if left <= 0:
                    return None
                self._cond.wait(left)

    def hello(self, role: str) -> render.Hello | None:
        with self._lock:
            page = self._newest(role)
            return page.hello if page else None

    def last_ack(self) -> render.Ack | None:
        """Newest ACK from the link page (after ``show_mode("pattern")`` it carries the drawn marker rects)."""
        with self._lock:
            page = self._link_page()
            return page.last_ack if page else None

    def on_tts(self, cb: Callable[[render.Tts], None]) -> None:
        with self._lock:
            self._tts_cbs.append(cb)

    def close(self, timeout: float = 5.0) -> None:
        """Drain TTS callbacks after the server stops. Idempotent."""
        self._dispatch.close(timeout)

    def stats(self) -> list[dict]:
        """Per connected page: role, send->ACK latency p50/p95 (ms, laptop clock), max unacked seen, scenes sent."""
        with self._lock:
            return [
                {
                    "role": p.hello.role,
                    "hello": p.hello.model_dump(mode="json"),
                    "connected_s": round(time.monotonic() - p.connected_at, 3),
                    "scenes_sent": p.scenes_sent,
                    "acks": len(p.latencies_ms),
                    "ack_p50_ms": _percentile(list(p.latencies_ms), 50),
                    "ack_p95_ms": _percentile(list(p.latencies_ms), 95),
                    "max_unacked": p.max_unacked,
                }
                for p in self._pages
            ]

    # ---- internals (lock held)

    def _dump_effective(self) -> dict:
        scene = self._mode_scene or self._scene
        return scene.model_dump(mode="json", exclude_none=True)

    def _bump(self) -> None:
        self._version += 1
        self._effective = self._dump_effective()
        self._poke_all()

    def _poke_all(self) -> None:
        for p in self._pages:
            p.poke()

    def _newest(self, role: str) -> _Page | None:
        for p in reversed(self._pages):
            if p.hello.role == role:
                return p
        return None

    def _link_page(self) -> _Page | None:
        return self._newest("projector") or self._newest("screen")

    def _on_loop_thread(self) -> bool:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return False
        with self._lock:
            return any(p.loop is loop for p in self._pages)

    def _outgoing(self, page: _Page, now: float) -> list[str]:
        out: list[str] = []
        if self._view is not None and page.view_version != self._view_version:  # first: it applies to what follows
            page.view_version = self._view_version
            out.append(self._view.model_dump_json())
        if self._calib is not None and page.calib_version != self._calib_version:
            page.calib_version = self._calib_version
            out.append(self._calib.model_dump_json())
        while page.says:
            out.append(page.says.popleft().model_dump_json())
        while page.sfx:  # before the scene: a sound lands with the frame that shows its change
            out.append(page.sfx.popleft().model_dump_json())
        due = page.sent_version != self._version or now - page.last_scene_t >= HEARTBEAT_S
        if due and len(page.outstanding) < MAX_UNACKED:
            seq = page.next_seq
            page.next_seq += 1
            page.outstanding[seq] = (self._version, now)
            page.max_unacked = max(page.max_unacked, len(page.outstanding))
            page.sent_version = self._version
            page.last_scene_t = now
            page.scenes_sent += 1
            out.append(json.dumps({**self._effective, "seq": seq}, separators=(",", ":")))
        return out

    def _on_ack(self, page: _Page, ack: render.Ack, now: float) -> None:
        sent = page.outstanding.get(ack.seq)
        if sent is None:
            return
        page.latencies_ms.append((now - sent[1]) * 1000.0)
        for seq in [s for s in page.outstanding if s <= ack.seq]:
            del page.outstanding[seq]
        if sent[0] > page.acked_version:
            page.acked_versions.append((sent[0], now))
        page.acked_version = max(page.acked_version, sent[0])
        page.last_ack = ack
        page.last_ack_t = now
        self._cond.notify_all()

    # ---- websocket side (event loop)

    async def serve(self, ws) -> None:
        """Run one ``/ws/render`` connection; the page must send ``hello`` first."""
        try:
            first = _page_msg.validate_json(await ws.receive_text())
        except ValidationError:
            first = None
        if not isinstance(first, render.Hello):
            await ws.close(code=1008, reason="hello first")
            return
        page = _Page(hello=first)
        with self._lock:
            self._pages.append(page)
            page.poke()
        sender = asyncio.create_task(self._send_loop(ws, page))
        try:
            while True:
                try:
                    msg = _page_msg.validate_json(await ws.receive_text())
                except ValidationError as e:
                    log.warning("render page sent an invalid message: %s", e.errors()[:1])
                    continue
                now = time.monotonic()
                with self._lock:
                    if isinstance(msg, render.Hello):
                        page.hello = msg
                        self._cond.notify_all()
                    elif isinstance(msg, render.Ack):
                        self._on_ack(page, msg, now)
                        page.poke()
                    else:
                        for cb in self._tts_cbs:
                            self._dispatch.submit(cb, msg)
        finally:
            await _stop(sender)
            with self._lock:
                self._pages.remove(page)
                self._cond.notify_all()

    async def _send_loop(self, ws, page: _Page) -> None:
        while True:
            with self._lock:
                out = self._outgoing(page, time.monotonic())
                wait_s = max(0.0, page.last_scene_t + HEARTBEAT_S - time.monotonic())
            for text in out:
                await ws.send_text(text)
            await page.wait(wait_s if wait_s > 0 else HEARTBEAT_S)


# ------------------------------------------------------------------------------------------------ preview


class _Viewer(_Conn):
    def __init__(self, pushed_at_connect: int) -> None:
        super().__init__()
        self.in_flight: set[int] = set()
        self.last_sent_id: int | None = None
        self.sent = 0
        self.pushed_at_connect = pushed_at_connect


def encode_preview(bgr) -> bytes:
    """Downscale a camera frame to the C8 preview size and encode it as JPEG (q~70)."""
    import cv2

    small = cv2.resize(bgr, CAMERA_PREVIEW_SIZE, interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, CAMERA_PREVIEW_QUALITY])
    if not ok:
        raise RuntimeError("preview JPEG encode failed")
    return buf.tobytes()


class PreviewHub:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._latest: tuple[int, bytes] | None = None  # (frame_id, packed message)
        self._pushed = 0
        self._viewers: list[_Viewer] = []

    @property
    def has_viewers(self) -> bool:
        """Lets the capture loop skip preview encoding when nobody watches."""
        with self._lock:
            return bool(self._viewers)

    def push(self, frame_id: int, t_mono: float, jpeg: bytes) -> None:
        msg = pack_camera_frame(frame_id, t_mono, jpeg)
        with self._lock:
            self._latest = (frame_id & 0xFFFFFFFF, msg)
            self._pushed += 1
            for v in self._viewers:
                v.poke()

    def stats(self) -> list[dict]:
        """Per viewer: frames pushed since it connected, frames sent, skipped fraction."""
        with self._lock:
            out = []
            for v in self._viewers:
                pushed = self._pushed - v.pushed_at_connect
                out.append({"pushed": pushed, "sent": v.sent, "skipped": (1 - v.sent / pushed) if pushed else 0.0})
            return out

    async def serve(self, ws) -> None:
        with self._lock:
            viewer = _Viewer(self._pushed - (self._latest is not None))  # the frame waiting at connect counts
            self._viewers.append(viewer)
            viewer.poke()
        sender = asyncio.create_task(self._send_loop(ws, viewer))
        try:
            while True:
                try:
                    ack = int(json.loads(await ws.receive_text())["ack"])
                except (ValueError, KeyError, TypeError):
                    continue
                with self._lock:
                    # frames are drawn in order: an ACK also covers older frames whose own ACK never came
                    viewer.in_flight = {f for f in viewer.in_flight if f > ack}
                    viewer.poke()
        finally:
            await _stop(sender)
            with self._lock:
                self._viewers.remove(viewer)

    async def _send_loop(self, ws, viewer: _Viewer) -> None:
        while True:
            await viewer.wait(None)
            with self._lock:
                latest = self._latest
                if latest is None or latest[0] == viewer.last_sent_id or len(viewer.in_flight) >= MAX_UNACKED:
                    continue
                viewer.last_sent_id = latest[0]
                viewer.in_flight.add(latest[0])
                viewer.sent += 1
            await ws.send_bytes(latest[1])


# ------------------------------------------------------------------------------------------------ control


class ControlHub:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._state: dict[str, Any] = {}
        self._version = 0
        self._clients: list[_Conn] = []
        self._cbs: list[Callable[[str, dict], None]] = []
        self._dispatch = _Dispatcher("control-hub-callbacks")

    def on_action(self, cb: Callable[[str, dict], None]) -> None:
        with self._lock:
            self._cbs.append(cb)

    def close(self, timeout: float = 5.0) -> None:
        """Drain UI callbacks after the server stops. Idempotent."""
        self._dispatch.close(timeout)

    def push_state(self, patch: dict[str, Any]) -> None:
        """Merge top-level keys into the screen state (thread-safe; latest wins per page)."""
        with self._lock:
            self._state.update(patch)
            self._version += 1
            for c in self._clients:
                c.poke()

    def state(self) -> dict[str, Any]:
        with self._lock:
            return dict(self._state)

    async def serve(self, ws) -> None:
        conn = _Conn()
        with self._lock:
            self._clients.append(conn)
            conn.poke()
        sender = asyncio.create_task(self._send_loop(ws, conn))
        try:
            while True:
                try:
                    msg = json.loads(await ws.receive_text())
                except ValueError:
                    continue
                if not isinstance(msg, dict) or msg.get("type") != "ui" or not isinstance(msg.get("action"), str):
                    log.warning("control page sent an unknown message: %.200r", msg)
                    continue
                args = msg.get("args") or {}
                if not isinstance(args, dict):
                    continue
                with self._lock:
                    cbs = list(self._cbs)
                for cb in cbs:
                    self._dispatch.submit(cb, msg["action"], args)
        finally:
            await _stop(sender)
            with self._lock:
                self._clients.remove(conn)

    async def _send_loop(self, ws, conn: _Conn) -> None:
        sent = -1
        while True:
            await conn.wait(None)
            with self._lock:
                if self._version == sent:
                    continue
                sent = self._version
                text = json.dumps({"type": "state", "state": self._state}, separators=(",", ":"))
            await ws.send_text(text)


def make_hubs() -> dict[str, Any]:
    """``create_app(**make_hubs())`` puts the three hubs on ``app.state``."""
    return {"render_hub": RenderHub(), "preview_hub": PreviewHub(), "control_hub": ControlHub()}
