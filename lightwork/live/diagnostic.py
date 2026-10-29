"""Explicitly synthetic, offline replay-as-live acceptance.

This is NOT the ordinary replay engine and never calls an API. A camera adapter
feeds decoded fixture crops through the real B measurement worker; the live queue,
Recorder, Reducer, intent dispatch workers and call recording all run. Only raw
responses already present in an explicitly synthetic run are allowed. Requests
must equal their recorded request after mapping the newly encoded crop hash back
to its source hash. No fabricated inference or fallback is possible.

Recorded motion, hands and user input facts are enqueued at their recorded times;
recorded model-result order releases the real worker calls. JPEG re-encoding changes
media/request hashes, so the optional source-derived comparison normalizes ONLY
those hashes. Separately verify re-derives the new run byte-exactly, with no such
normalization. Output always goes to runs-synthetic/ and audit must reject it.
"""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from ..config import Settings
from ..contracts.calls import CallMeta, request_fingerprint
from ..contracts.canonical import canonical
from ..contracts.events import Event, read_events
from ..contracts.layout import Calibration, Layout
from ..contracts.run import Manifest
from ..contracts.stepgraph import StepGraph
from ..models.record import write_call
from ..runs.recorder import Recorder
from ..runs.store import DirStore
from ..runs.verify import verify_run
from ..vision.source import Frame
from .session import LiveSession


class FixtureClock:
    def __init__(self):
        self.t = 0.

    def __call__(self) -> float:
        return self.t


class FixtureCamera:
    """Finite timestamped pairs from fixture crops, not live hardware or inference."""
    controlled = False

    def __init__(self, crop_xywh: list[int], camera_info: dict):
        self.crop_xywh = crop_xywh
        self.info = camera_info
        self.image = None
        self.t = 0.
        self.idx = 0
        self.closed = False

    def set_frame(self, jpeg: bytes, t_capture: float) -> None:
        crop = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
        x, y, w, h = self.crop_xywh
        if crop is None or crop.shape[:2] != (h, w):
            raise ValueError("fixture keyframe dimensions differ from its work-area crop")
        self.image = np.zeros((y+h, x+w, 3), np.uint8)
        self.image[y:y+h, x:x+w] = crop
        self.t = t_capture

    def latest(self) -> Frame | None:
        return None if self.closed or self.image is None else Frame(self.image, self.t - .1, self.idx)

    def wait_frame(self, after_t: float, timeout: float) -> Frame | None:
        if self.closed:
            return None
        for t in (self.t - .033, self.t):
            if t > after_t:
                self.idx += 1
                return Frame(self.image, t, self.idx)
        return None

    def describe(self) -> dict:
        return dict(self.info)

    def close(self) -> None:
        self.closed = True


def _replace(value, replacements: dict[str, str]):
    if isinstance(value, dict):
        return {k: _replace(v, replacements) for k, v in value.items()}
    if isinstance(value, list):
        return [_replace(v, replacements) for v in value]
    if isinstance(value, str):
        if value.startswith("frame:"):
            return "frame:" + replacements.get(value[6:], value[6:])
        return replacements.get(value, value)
    return value


class RecordedCalls:
    """Threaded F-compatible transport reading only the supplied synthetic run."""
    def __init__(self, source: Path, run_dir: Path, events: list[Event]):
        self.store, self.run_dir = DirStore(source), run_dir
        self.results = {e.payload["intent_id"]: e.payload for e in events if e.type == "model_result"}
        self.released = {iid: threading.Event() for iid in self.results}
        self.hashes: dict[str, str] = {}
        self.closed = False

    def execute(self, *, intent_id, role, endpoint, request, frames, index, timeout_s):
        if intent_id not in self.results:
            raise ValueError(f"fixture has no recorded reply for intent {intent_id}; no synthetic fallback")
        if not self.released[intent_id].wait(15):
            raise TimeoutError(f"fixture did not release {intent_id}")
        payload = self.results[intent_id]
        source_dir = self.store.call_path(payload["call_dir"])
        original_request = json.loads((source_dir / "request.json").read_bytes())
        if canonical(_replace(request, self.hashes)) != canonical(original_request):
            raise ValueError(f"request differs from synthetic recording: {intent_id}")
        for sha, data in frames.items():
            from ..contracts.canonical import sha256_hex
            if sha256_hex(data) != sha or sha not in self.hashes:
                raise ValueError("live request did not use a captured fixture crop")
        meta = CallMeta.model_validate_json((source_dir / "meta.json").read_bytes())
        if meta.provenance != "synthetic" or meta.endpoint != endpoint:
            raise ValueError("diagnostic refuses non-synthetic or incompatible call evidence")
        raw, kind = self.store.call_response(payload["call_dir"])
        meta = meta.model_copy(update={"fingerprint": request_fingerprint(request)})
        attempts = [(p.read_bytes(), p.suffix[1:]) for p in sorted((source_dir / "attempts").glob("*"))]
        relative = write_call(self.run_dir, index=index, role=role, request=request, raw=raw,
                              kind=kind, meta=meta, attempt_responses=attempts)
        return {**payload, "call_dir": relative}

    def close(self) -> None:
        self.closed = True
        for release in self.released.values():
            release.set()


@dataclass(frozen=True)
class DiagnosticResult:
    run_dir: Path
    verify: object
    source_derived_equal: bool


def replay_as_live(source: Path, root: Path, *, on_started=None) -> DiagnosticResult:
    source = Path(source)
    manifest = Manifest.model_validate_json((source / "manifest.json").read_bytes())
    if manifest.provenance != "synthetic":
        raise ValueError("--replay-as-live accepts only explicit synthetic fixtures, never real inference runs")
    events = read_events(source / "events.jsonl")
    graph = StepGraph.model_validate_json((source / "stepgraph.json").read_bytes())
    layout = Layout.model_validate_json((source / "layout.json").read_bytes())
    calib = Calibration.model_validate_json((source / "calib.json").read_bytes())
    store = DirStore(source)
    clock = FixtureClock()
    first = next(e for e in events if e.type == "keyframe")
    camera = FixtureCamera(layout.work_area_xywh, calib.camera.model_dump())
    camera.set_frame(store.frame_bytes(first.payload["frame_sha"]), first.payload["t_capture"])
    recorder = Recorder(root, graph.id, events[0].payload, provenance="synthetic", clock=clock)
    client = RecordedCalls(source, recorder.run_dir, events)
    settings = Settings(root=Path(root), chat_model=events[0].payload["models"].get("verdict"))
    session = LiveSession(settings, graph, layout, calib, source=camera, recorder=recorder,
                          model_client=client, clock=clock, scripted=True, quiet=True, hands=False, voice=False,
                          flat_field=store.frame_bytes(calib.flat_field_sha) if calib.flat_field_sha else None)
    received: dict[str, threading.Event] = {iid: threading.Event() for iid in client.results}
    captured: list[Event] = []

    def observe(batch):
        captured.extend(batch)
        for event in batch:
            if event.type == "model_result":
                received[event.payload["intent_id"]].set()

    session.subscribe(observe)
    try:
        session.start()
        if on_started is not None:
            on_started(session)
        for event in events:
            if event.derived or event.type in {"run_started", "procedure_loaded", "calibration_loaded", "layout_loaded"}:
                continue
            clock.t = event.t
            if event.type == "keyframe":
                camera.set_frame(store.frame_bytes(event.payload["frame_sha"]), event.payload["t_capture"])
                frame = session.capture_now().result(10)
                client.hashes[frame.payload["frame_sha"]] = event.payload["frame_sha"]
            elif event.type == "model_result":
                iid = event.payload["intent_id"]
                client.released[iid].set()
                if not received[iid].wait(10):
                    raise session.error or TimeoutError(f"live worker did not record {iid}")
            elif event.type == "run_stopped":
                session.stop(event.payload["reason"])
            else:
                session.feed_recorded_input(event).result(10)
        session.stop("synthetic replay-as-live complete")
        path = session.join(20)
    finally:
        for release in client.released.values():
            release.set()
        session.stop("diagnostic exit")
        if session._thread is not None and not session.finished:
            try:
                session.join(20)
            except BaseException:
                pass
        session.render.close()
        session.control.close()
    originals = {e.payload["intent_id"]: e.payload["fingerprint"] for e in events if e.type == "model_request"}
    replacements = dict(client.hashes)
    for event in captured:
        if event.type == "model_request" and event.payload["intent_id"] in originals:
            replacements[event.payload["fingerprint"]] = originals[event.payload["intent_id"]]
    got = [(e.t, e.type, _replace(e.payload, replacements)) for e in captured if e.derived]
    expected = [(e.t, e.type, e.payload) for e in events if e.derived]
    report = verify_run(path)
    return DiagnosticResult(path, report, canonical(got) == canonical(expected))
