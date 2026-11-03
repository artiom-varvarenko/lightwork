"""One live run, with no CV or model I/O on the ASGI loop.

``LiveSession(settings, graph, layout, calib, source=...)`` owns its camera,
optional HandTracker/VoiceService, recorder, and model client. ``start()`` starts
one serialized event consumer, a vision worker, a measurement worker, three SAM
workers and one language-model worker. Camera acquisition and voice pacing keep
their existing dedicated threads. ``stop()`` stops producers, drains accepted
inputs and all outstanding calls (including late replies), then seals the run.
``join()`` waits for that seal; it raises the original runtime error, if any.

Collection uses ``subscribe(callback(events))``, ``capture_now() -> Future[Event]``,
``on_ui(action, args)`` and ``before_close(callback(session))``. Callbacks run on
the serialized consumer: never wait on a capture Future there. Before-close
callbacks may write manifest-covered labels, but must not submit new work.
``set_scene_transform`` changes presentation only, including the projection mask;
it never changes reducer scenes or their recorded hashes.

Every stored/sent image is a validated work-area crop. B verifies the measurement
pair and encodes JPEG before H reads the saved bytes. Screen-only capture uses
B's normal mode; once a projector has connected, loss of that page cannot silently
switch to normal capture. A screen-only calibration (C7 ``measurement_mode``
"normal") never projects: /projector pages get no mapping and are ignored.
No cache, synthetic response or model substitution is
used in production. ``scripted=True`` is restricted to synthetic recorders and is
only for the offline replay-as-live diagnostic (see ``diagnostic.py``).
"""

from __future__ import annotations

import json
import logging
import platform
import queue
import threading
import time
from collections import Counter
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .. import __version__
from ..contracts.calls import request_fingerprint
from ..contracts.events import Event, make_payload, read_events
from ..contracts.layout import Calibration, Layout
from ..contracts.render import Calib, SceneMsg
from ..contracts.stepgraph import SAM_CHECKS, StepGraph
from ..core import Reducer
from ..core.graph import iter_steps, validate
from ..geometry import Regions
from ..models.budget import Budget
from ..models.client import ModelClient, run_provenance, timeout_for
from ..models.errors import ModelAccessError, ModelConfigError, ProvenanceError
from ..models.requests import backend_models
from ..runs.recorder import Recorder
from ..server.hub import encode_preview, make_hubs
from ..server.screen_state import ScreenState
from ..vision.capture import open_camera
from ..vision.gate import Gate
from ..vision.hands import HandEvents, HandsUnavailable, HandTracker
from ..vision.measure_frames import capture_keyframe
from ..vision.overlay_mask import OverlayMask
from ..voice.service import VoiceService

log = logging.getLogger(__name__)


class CaptureUnavailable(RuntimeError):
    """No ACK, no camera pair, or failed B content verification; no evidence was accepted."""


def validate_live_graph(graph: StepGraph, layout: Layout, backend: str) -> None:
    """Reject incompatible graphs rather than silently replacing SAM or extra judges."""
    errors = validate(graph, layout)
    if graph.id != layout.pack:
        errors.append(f"layout pack {layout.pack!r} differs from graph {graph.id!r}")
    for step in iter_steps(graph.steps):
        sam = {c.object for c in step.checks if c.kind in SAM_CHECKS}
        if backend != "meta" and sam:
            errors.append(f"{step.id}: SAM checks need --backend meta; author an explicit Glimmer graph")
        if len(sam) > 3:
            errors.append(f"{step.id}: more than three SAM objects on one keyframe")
        if sum(c.kind == "judge" for c in step.checks) > 1:
            errors.append(f"{step.id}: only one verdict per step attempt is allowed")
    x, y, w, h = layout.work_area_xywh
    if min(x, y) < 0 or min(w, h) <= 0:
        errors.append("work area must have nonnegative origin and positive size")
    if errors:
        raise ValueError("live graph/layout rejected: " + "; ".join(errors))


@dataclass
class _Message:
    kind: str
    value: Any = None
    future: Future | None = None
    t: float | None = None


class LiveSession:
    def __init__(
        self, settings, graph: StepGraph, layout: Layout, calib: Calibration, *,
        source=None, hubs=None, model_client=None, recorder: Recorder | None = None,
        backend: str | None = None, hands=None, voice=None, quiet: bool = False,
        voice_dry_run: bool = False, clock: Callable[[], float] = time.monotonic,
        scripted: bool = False, flat_field: bytes | None = None,
        code_commit: str = "unknown", dirty: bool = True,
    ):
        self.settings, self.graph, self.layout, self.calib = settings, graph, layout, calib
        self.backend, self.models = backend_models(backend or settings.backend, settings.chat_model)
        if recorder is not None:
            self.backend = recorder.started.payload["backend"]
            self.models = dict(recorder.started.payload["models"])
        validate_live_graph(graph, layout, self.backend)
        if scripted and (recorder is None or recorder.provenance != "synthetic"):
            raise ValueError("scripted live input requires an explicitly synthetic recorder")
        if isinstance(model_client, ModelClient) and model_client.cache is not None:
            raise ValueError("live recordings must not use a model cache")
        self.source, self.hubs = source, hubs if hubs is not None else make_hubs()
        self.render = self.hubs["render_hub"]
        self.preview = self.hubs["preview_hub"]
        self.control = self.hubs["control_hub"]
        self.client, self.recorder = model_client, recorder
        self.hands, self.voice = hands, voice
        self.quiet, self.voice_dry_run = quiet, voice_dry_run
        self.clock, self.scripted, self.flat_field = clock, scripted, flat_field
        self.code_commit, self.dirty = code_commit, dirty
        self.error: BaseException | None = None
        self.run_dir: Path | None = None
        self.reducer: Reducer | None = None
        self.screen = ScreenState(graph)
        self.regions = Regions(layout, graph)
        self._queue: queue.Queue[_Message] = queue.Queue()
        self._lock = threading.RLock()
        self._subscribers: list[Callable[[list[Event]], None]] = []
        self._before_close: list[Callable[[LiveSession], None]] = []
        self._transform: Callable[[SceneMsg], SceneMsg] | None = None
        self._last_scene = SceneMsg(seq=0, mode="normal", prims=[])
        self._projection_seen = False
        self._mask_projection = False
        self._started = False
        self._accepting = False
        self._stopping = False
        self._stop_reason = "operator"
        self._producer_stop = threading.Event()
        self._ready = threading.Event()
        self._done = threading.Event()
        self._thread: threading.Thread | None = None
        self._vision: threading.Thread | None = None
        self._executors: dict[str, ThreadPoolExecutor] = {}
        self._calls: set[Future] = set()
        self._dispatched: set[str] = set()
        self._frame_seq: dict[str, int] = {}
        self._sam_counts: Counter[int] = Counter()
        self._verdicts: set[tuple[str, str]] = set()
        self._capture_job: Future | None = None
        self._capture_waiters: list[Future] = []
        self._last_seq = 0

    def subscribe(self, callback: Callable[[list[Event]], None]) -> None:
        with self._lock:
            self._subscribers.append(callback)

    def before_close(self, callback: Callable[[LiveSession], None]) -> None:
        with self._lock:
            self._before_close.append(callback)

    def set_scene_transform(self, callback: Callable[[SceneMsg], SceneMsg] | None) -> None:
        with self._lock:
            if not self._started:
                self._transform = callback
                return
        self._submit(_Message("transform", callback))

    def _submit(self, message: _Message) -> Future:
        future = message.future if message.future is not None else Future()
        message.future = future
        with self._lock:
            if not self._accepting:
                future.set_exception(RuntimeError("live session is not accepting input"))
            else:
                self._queue.put(message)
        return future

    def on_ui(self, action: str, args: dict | None = None) -> Future:
        """Thread-safe C1 UI input. Unknown actions remain visible to collection subscribers."""
        action = "done" if action == "confirm" else action
        return self._submit(_Message("input", ("ui", "ui", make_payload("ui", {"action": action, "args": args or {}}))))

    def on_key(self, key: str) -> Future:
        return self._submit(_Message("input", ("key", "key", make_payload("key", {"key": key}))))

    def on_voice(self, type_: str, payload: dict) -> None:
        if self.voice_dry_run or self.quiet:
            return
        if type_ not in ("voice", "voice_rejected"):
            raise ValueError("voice sink only accepts final voice or voice_rejected events")
        self._submit(_Message("input", (type_, "voice", make_payload(type_, payload))))

    def capture_now(self) -> Future[Event]:
        """Schedule a B capture; coalesced callers receive the same verified C1 frame.

        An unverified frame is still logged, but its Future raises CaptureUnavailable.
        Never call ``.result()`` from a subscriber or UI callback on this session.
        """
        return self._submit(_Message("capture"))

    def feed_recorded_input(self, event: Event) -> Future:
        """Offline diagnostic only: feed a non-image/non-model recorded input through the queue."""
        if not self.scripted or event.derived or event.type in {
            "run_started", "procedure_loaded", "layout_loaded", "calibration_loaded", "keyframe", "model_result", "run_stopped"
        }:
            raise ValueError("not an admissible scripted input")
        return self._submit(_Message("input", (event.type, event.src, event.payload), t=event.t))

    def start(self) -> LiveSession:
        with self._lock:
            if self._started:
                raise RuntimeError("a LiveSession can only be started once")
            self._started = True
        try:
            self._prepare()
            self._accepting = True
            self._thread = threading.Thread(target=self._loop, name="lightwork-live")
            self._thread.start()
            self._ready.wait()
            if self.error is not None:
                self.join()
            return self
        except BaseException as exc:
            self.error = self.error or exc
            if self._thread is None:
                self._dispose()
            raise

    def _prepare(self) -> None:
        synthetic = self.recorder is not None and self.recorder.provenance == "synthetic"
        if not synthetic:
            if self.backend == "meta" and not self.settings.model_api_key:
                raise ModelConfigError("MODEL_API_KEY is not set; live Meta mode cannot run")
            prov, _ = run_provenance(self.settings, self.backend, self.models)
            if prov != "meta" and not self.settings.dev_allow_non_meta:
                raise ProvenanceError("non-Meta live calls require LW_DEV_ALLOW_NON_META=1 and runs-dev provenance")
        else:
            prov = "synthetic"
        if self.source is None or isinstance(self.source, (str, Path)):
            self.source = open_camera(None if self.source is None else str(self.source), state_dir=self.settings.state_dir)
        first = self.source.latest()
        if first is None:
            raise CaptureUnavailable("camera has no initial frame")
        self._crop(first.image)
        info = self.source.describe()
        if not synthetic:
            self._check_camera(info)
        # A screen-only calibration has no projector mapping: nothing is projected, so nothing is masked.
        self.overlay = None if self.calib.screen_only else OverlayMask(
            (first.image.shape[1], first.image.shape[0]), self.calib.H_cam_to_page, (self.calib.page.w, self.calib.page.h))
        self.gate = Gate(self.regions, self.layout.work_area_xywh)
        self.hand_events = HandEvents(self.regions)
        if self.recorder is None:
            self.recorder = Recorder(self.settings.root, self.graph.id, {
                "backend": self.backend, "models": self.models, "code_commit": self.code_commit,
                "dirty": self.dirty, "platform": platform.platform(), "version": __version__,
            }, provenance=prov, clock=self.clock)
        if self.recorder.provenance != prov:
            raise ProvenanceError("supplied recorder provenance differs from authenticated backend provenance")
        self.run_dir = self.recorder.run_dir
        if self.calib.flat_field_sha:
            x, y, w, h = self.layout.work_area_xywh
            fx, fy, fw, fh = self.calib.flat_field_crop_xywh or (0, 0, 0, 0)
            if fw <= 0 or fh <= 0 or not (x <= fx and y <= fy and fx+fw <= x+w and fy+fh <= y+h):
                raise ValueError("calibration flat field must fit wholly inside the accepted work area")
            if self.flat_field is None or self.recorder.put_frame(self.flat_field) != self.calib.flat_field_sha:
                raise ValueError("calibration flat field is missing or hashes differently")
        if self.client is None:
            self.client = ModelClient(self.settings, run_dir=self.run_dir, provenance=prov,
                                      budget=Budget.from_settings(self.settings), cache=None, recording=True)
        elif Path(self.client.run_dir) != self.run_dir:
            raise ValueError("model client must record into this session's run directory")
        self.reducer = Reducer(self.graph, self.layout, self.calib, self.recorder.store,
                               backend=self.backend, models=self.models)
        if self.hands is None and not self.scripted:
            try:
                self.hands = HandTracker(self.settings.state_dir / "models" / "hand_landmarker.task")
            except (HandsUnavailable, ImportError, OSError, RuntimeError) as exc:
                log.warning("MOTION ONLY: hands unavailable: %s", exc)
                self.hands = False
        if self.voice is None and not self.scripted:
            self.voice = VoiceService(self.settings, [o.noun for o in self.graph.objects][:20],
                                      self.on_voice, dry_run=self.voice_dry_run,
                                      on_hud=lambda hud: self.control.push_state({"hud": hud}))
        self._executors = {
            "capture": ThreadPoolExecutor(max_workers=1, thread_name_prefix="lightwork-measure"),
            "segment": ThreadPoolExecutor(max_workers=3, thread_name_prefix="lightwork-sam"),
            "language": ThreadPoolExecutor(max_workers=1, thread_name_prefix="lightwork-model"),
        }
        self.control.on_action(self.on_ui)
        self.render.on_tts(self._on_tts)
        if not self.calib.screen_only:
            self.render.set_calib(Calib(H_cam_to_page=self.calib.H_cam_to_page,
                                        page_size=[self.calib.page.w, self.calib.page.h]))
        self.control.push_state({
            **self.screen.snapshot(), "run": {"id": self.recorder.run_id, "pack": self.graph.id,
                "mode": "live", "backend": self.backend, "provenance": prov},
            "banner": "SYNTHETIC OFFLINE DIAGNOSTIC — not Meta evidence" if prov == "synthetic" else None,
            "camera": {"w": first.image.shape[1], "h": first.image.shape[0], **info,
                       "controlled": bool(getattr(self.source, "controlled", False))},
            "preview_xywh": self.layout.work_area_xywh,
            "hands": "tracked (advisory)" if self.hands else "motion-only advisory; no fingertip evidence",
            "quiet": self.quiet,
        })

    def stop(self, reason: str = "operator") -> None:
        """Nonblocking; already accepted calls complete and are recorded before sealing."""
        with self._lock:
            if self._accepting:
                self._accepting = False
                self._queue.put(_Message("stop", reason))

    def join(self, timeout: float | None = None) -> Path:
        if self._thread is None:
            raise RuntimeError("session has not started")
        if threading.current_thread() is self._thread:
            raise RuntimeError("cannot join the serialized consumer from its callback")
        self._thread.join(timeout)
        if self._thread.is_alive():
            raise TimeoutError("live session is still draining outstanding work")
        if self.error is not None:
            raise self.error
        return self.run_dir

    @property
    def finished(self) -> bool:
        return self._done.is_set()

    def _ingest(self, type_: str, src: str, payload: dict, t: float | None = None) -> list[Event]:
        events = self.recorder.ingest(type_, src, payload, self.recorder.now() if t is None else t)
        self._publish(events)
        return events

    def _publish(self, events: list[Event]) -> None:
        for ev in events:
            self._last_seq = ev.seq
            self.control.push_state(self.screen.feed(ev.type, ev.payload))
            if ev.type == "keyframe":
                self._frame_seq[ev.payload["frame_sha"]] = ev.seq
            elif ev.type == "scene":
                self._publish_scene()
            elif ev.type == "say" and not self.quiet and not self._stopping:
                self.render.say(ev.payload["text"], ev.payload["priority"])
            elif ev.type == "step" and self.voice:
                self.voice.step_changed()
            elif ev.type == "model_request" and not self._stopping:
                self._dispatch(ev.payload)
            elif ev.type == "model_result":
                meta = json.loads((self.recorder.store.call_path(ev.payload["call_dir"]) / "meta.json").read_bytes())
                self.control.push_state(self.screen.set_call_cost(ev.payload["intent_id"], meta["cost_usd"]))
        with self._lock:
            callbacks = tuple(self._subscribers)
        for callback in callbacks:
            try:
                callback(events)
            except Exception:
                # A failed collection sink is fatal, but must not fail recursively
                # when we record its error and drain outstanding model replies.
                with self._lock:
                    self._subscribers.remove(callback)
                raise

    def _publish_scene(self) -> None:
        scene = self.reducer.scene
        if self._transform is not None:
            scene = self._transform(scene.model_copy(deep=True))
        self._last_scene = scene
        self.render.publish_scene(scene)
        self._sync_projection(force=True)

    def _sync_projection(self, *, force: bool = False) -> bool:
        if self.calib.screen_only:  # a /projector page gets no mapping and never lights the bench
            return False
        projecting = self.render.hello("projector") is not None
        if projecting:
            self._projection_seen = True
        if force or projecting != self._mask_projection:
            self._mask_projection = projecting
            # A screen overlay is not light on the physical bench. Keep the SAME mask
            # for gate and capture, but do not pretend screen-only primitives are projected.
            masked = self._last_scene if self._projection_seen else SceneMsg(seq=0, mode="normal", prims=[])
            self.overlay.set_scene(masked, self.clock())
        return projecting

    def _on_tts(self, tts) -> None:
        if self.voice and not self._done.is_set():
            self.voice.on_tts(tts)

    def _dispatch(self, payload: dict) -> None:
        iid, role = payload["intent_id"], payload["role"]
        if iid in self._dispatched:
            raise RuntimeError(f"duplicate derived model intent {iid}")
        endpoint, request = self.reducer.request_for(iid)
        if request_fingerprint(request) != payload["fingerprint"]:
            raise RuntimeError(f"derived request fingerprint mismatch: {iid}")
        repair = iid.endswith(("/retry", "/fallback"))
        if role == "segment":
            frame_seq = self._frame_seq[payload["frame_sha"]]
            if self._sam_counts[frame_seq] >= 3:
                raise RuntimeError(f"SAM keyframe budget exceeded for frame event {frame_seq}; no call sent")
            self._sam_counts[frame_seq] += 1
        elif role == "verdict" and not repair:
            parts = iid.split("/")
            attempt = (parts[0], parts[-1])
            if attempt in self._verdicts:
                raise RuntimeError(f"more than one verdict for step attempt {attempt}; no call sent")
            self._verdicts.add(attempt)
        self._dispatched.add(iid)
        frames = ({payload["frame_sha"]: self.recorder.store.frame_bytes(payload["frame_sha"])}
                  if payload["frame_sha"] else {})
        index, _ = self.recorder.alloc_call(role, payload["model"])
        executor = self._executors["segment" if role == "segment" else "language"]
        future = executor.submit(self.client.execute, intent_id=iid, role=role, endpoint=endpoint,
                                 request=request, frames=frames, index=index,
                                 timeout_s=timeout_for(role, self.backend))
        self._calls.add(future)
        future.add_done_callback(lambda f: self._queue.put(_Message("model", f, t=self.recorder.now())))

    def _begin_capture(self, future: Future) -> None:
        if self._stopping:
            future.set_exception(CaptureUnavailable("session is stopping"))
            return
        self._capture_waiters.append(future)
        if self._capture_job is None:
            self._capture_job = self._executors["capture"].submit(self._capture)
            self._capture_job.add_done_callback(lambda f: self._queue.put(_Message("captured", f)))

    def _check_camera(self, info: dict) -> None:
        expected = self.calib.camera
        if info["device_id"] != expected.device_id or info["mode"] != expected.mode:
            raise ValueError("camera device/mode differs from calibration; recalibrate this source")
        if info["controls"] != expected.controls:
            raise ValueError("camera control readback differs from calibration; restore controls or recalibrate")
        report = getattr(self.source, "report", None)
        if report is not None and not report.ok:
            raise CaptureUnavailable("camera control lock is incomplete: " + self.source.status)


    def _capture(self):
        if self.recorder.provenance != "synthetic":
            self._check_camera(self.source.describe())
        projecting = self._sync_projection()
        if self._projection_seen and not projecting:
            raise CaptureUnavailable("projector disconnected; cannot establish measurement illumination")
        mode = self.calib.measurement_mode if projecting else "normal"
        # Derive the exact recorder origin without reaching into Recorder internals.
        # run_time() is quantized, so reconstruct with a quantized clock sample.
        sample = self.clock()
        t0 = sample - self.recorder.run_time(sample)
        result = capture_keyframe(self.source, self.render, mode, self.layout.work_area_xywh,
                                  overlay=self.overlay, t0=t0, clock=self.clock)
        if projecting and self.render.hello("projector") is None:
            raise CaptureUnavailable("projector disconnected during measurement")
        if result is None:
            raise CaptureUnavailable("measurement page did not ACK or camera gave no fresh pair")
        return result

    def _finish_capture(self, future: Future) -> None:
        waiters, self._capture_waiters = self._capture_waiters, []
        self._capture_job = None
        try:
            jpeg, payload = future.result()
            sha = self.recorder.put_frame(jpeg)
            events = self._ingest("keyframe", "camera", {"frame_sha": sha, **payload},
                                  max(self.recorder.now(), payload["t_capture"]))
            frame = events[0]
            if not payload["verified"]:
                raise CaptureUnavailable("measurement pair failed B content verification")
        except Exception as exc:
            self._ingest("ui", "orchestrator", {"action": "capture_failed", "args": {"reason": str(exc)}})
            self.control.push_state({"capture_error": str(exc)})
            for waiter in waiters:
                if not waiter.done():
                    waiter.set_exception(exc)
            if not isinstance(exc, CaptureUnavailable):
                raise
        else:
            self.control.push_state({"capture_error": None})
            for waiter in waiters:
                if not waiter.done():
                    waiter.set_result(frame)

    def _crop(self, image):
        x, y, w, h = self.layout.work_area_xywh
        if x + w > image.shape[1] or y + h > image.shape[0]:
            raise ValueError("work area does not fit camera frame; refusing an expanded or clamped crop")
        return image[y:y+h, x:x+w]

    def _vision_loop(self) -> None:
        after = float("-inf")
        capture_needed = False
        last_preview = float("-inf")
        hand_observed = False
        try:
            while not self._producer_stop.is_set():
                frame = self.source.wait_frame(after, .25)
                if frame is None:
                    latest = self.source.latest()
                    if latest is None or self.clock() - latest.t_mono > 2:
                        raise CaptureUnavailable("camera stopped delivering frames (ended or disconnected)")
                    continue
                after = frame.t_mono
                crop = self._crop(frame.image)
                self._sync_projection()
                hand_over = False
                if self.hands:
                    obs = self.hands.process(frame)
                    x, y, w, h = self.layout.work_area_xywh
                    hand_over = any(x <= a < x+w and y <= b < y+h for a, b in (obs.tips or (() if obs.tip_xy is None else (obs.tip_xy,))))
                    observations = self.hand_events.update(obs, frame.t_mono)
                    if not hand_observed and not observations:
                        # An actual successful empty observation establishes tracking;
                        # do not let "no hand yet" masquerade as motion-only mode.
                        observations = [("hands", {"over_regions": [], "tip_xy": None})]
                    hand_observed = True
                    for kind, payload in observations:
                        self._submit(_Message("input", (kind, "hands", payload), t=self.recorder.run_time(frame.t_mono)))
                mask = self.overlay.mask_at(frame.t_mono) if self.overlay is not None else None
                for kind, payload in self.gate.update(frame, mask):
                    self._submit(_Message("input", (kind, "gate", payload), t=self.recorder.run_time(frame.t_mono)))
                    capture_needed = payload["state"] == "settled"
                if capture_needed and not hand_over:
                    self.capture_now()
                    capture_needed = False
                if frame.t_mono - last_preview >= 1 / 15:
                    if self.preview.has_viewers:
                        self.preview.push(frame.idx, frame.t_mono, encode_preview(crop))
                    self.recorder.put_preview(crop, self.recorder.run_time(frame.t_mono))
                    last_preview = frame.t_mono
        except BaseException as exc:
            self._queue.put(_Message("failure", exc))

    def _begin_stop(self, reason: str) -> None:
        if self._stopping:
            return
        self._stopping = True
        self._stop_reason = reason
        with self._lock:
            self._accepting = False
        self._producer_stop.set()
        if self.voice:
            self.voice.set_muted(True)
            self.voice.close()
        self.source.close()
        if self._vision is not None:
            self._vision.join()

    def _fail(self, exc: BaseException) -> None:
        if self.error is None:
            self.error = exc
            log.error("live session failed: %s", exc)
            self.control.push_state({"runtime_error": str(exc)})
            self._ingest("ui", "orchestrator", {"action": "runtime_error", "args": {"reason": str(exc)}})
        self._begin_stop("runtime error: " + str(exc))

    def _loop(self) -> None:
        try:
            self._publish([self.recorder.started, *self.recorder.attach(self.reducer)])
            for name, model, kind, field in (
                ("stepgraph.json", self.graph, "procedure_loaded", "stepgraph_sha"),
                ("calib.json", self.calib, "calibration_loaded", "calib_sha"),
                ("layout.json", self.layout, "layout_loaded", "layout_sha"),
            ):
                sha = self.recorder.put_run_file(name, model)
                self._ingest(kind, "orchestrator", {field: sha}, 0.)
            if not self.scripted:
                self._ingest("ui", "orchestrator", {"action": "runtime_mode", "args": {
                    "hands": "tracked-advisory" if self.hands else "motion-only-advisory",
                    "quiet": self.quiet, "voice_dry_run": self.voice_dry_run,
                }})
                if self.voice:
                    self.voice.set_muted(self.quiet)
                    self.voice.start()
                self._vision = threading.Thread(target=self._vision_loop, name="lightwork-vision")
                self._vision.start()
            self._ready.set()
            next_tick = self.clock() + .25
            while True:
                try:
                    msg = self._queue.get(timeout=.05)
                except queue.Empty:
                    if self._stopping and not self._calls and self._capture_job is None:
                        break
                    if not self.scripted and not self._stopping and self.clock() >= next_tick:
                        self._ingest("tick", "orchestrator", {})
                        next_tick = self.clock() + .25
                    continue
                try:
                    if msg.kind == "stop":
                        self._begin_stop(msg.value)
                    elif msg.kind == "failure":
                        self._fail(msg.value)
                    elif msg.kind == "transform":
                        self._transform = msg.value
                        self._publish_scene()
                        msg.future.set_result(None)
                    elif msg.kind == "capture":
                        self._begin_capture(msg.future)
                    elif msg.kind == "captured":
                        self._finish_capture(msg.value)
                    elif msg.kind == "model":
                        self._calls.discard(msg.value)
                        try:
                            payload = msg.value.result()
                        except ModelAccessError as exc:
                            self._ingest("model_result", "model", exc.result, msg.t)
                            raise
                        self._ingest("model_result", "model", payload, msg.t)
                    elif msg.kind == "input":
                        kind, src, payload = msg.value
                        events = self._ingest(kind, src, payload, msg.t)
                        if kind == "ui" and payload["action"] == "start" and not self.quiet:
                            # The initial say happened before the browser's user gesture.
                            self._ingest("ui", "orchestrator", {"action": "repeat", "args": {}})
                        if (kind == "ui" and payload["action"] == "stop") or (kind == "key" and payload["key"] == "q"):
                            self._begin_stop("operator")
                        if msg.future is not None and not msg.future.done():
                            msg.future.set_result(events)
                except BaseException as exc:
                    if msg.future is not None and not msg.future.done():
                        msg.future.set_exception(exc)
                    self._fail(exc)
            for callback in tuple(self._before_close):
                callback(self)
            if self.voice:
                from ..contracts.canonical import canonical_bytes
                from ..runs.store import write_atomic
                write_atomic(self.recorder.run_dir / "voice-stats.json", canonical_bytes(self.voice.stats()) + b"\n")
            self.run_dir = self.recorder.close(self._stop_reason)
            # close() owns the terminal input and derivation. Publish its actual tail,
            # never synthesize another run_stopped or reduce it twice.
            self._publish([e for e in read_events(self.run_dir / "events.jsonl") if e.seq > self._last_seq])
        except BaseException as exc:
            self.error = self.error or exc
            self._begin_stop("runtime error: " + str(exc))
        finally:
            self._ready.set()
            self._dispose()
            self._done.set()

    def _dispose(self) -> None:
        self._producer_stop.set()
        for executor in self._executors.values():
            executor.shutdown(wait=True, cancel_futures=False)
        if self.voice:
            self.voice.close()
        if self.hands:
            self.hands.close()
        if self.source is not None and not isinstance(self.source, (str, Path)):
            self.source.close()
        if self.client is not None:
            self.client.close()
        if self.recorder is not None and self.recorder.manifest is None:
            try:
                self.run_dir = self.recorder.close("runtime error: " + str(self.error))
            except Exception as exc:
                self.error = self.error or exc
        for future in self._capture_waiters:
            if not future.done():
                future.set_exception(RuntimeError("session closed before capture completed"))
        with self._lock:
            self._accepting = False
        while True:
            try:
                msg = self._queue.get_nowait()
            except queue.Empty:
                break
            if msg.future is not None and not msg.future.done():
                msg.future.set_exception(RuntimeError("session closed"))
