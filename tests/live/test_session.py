"""Consumer-visible races, evidence boundaries and lifecycle of the real live queue."""

from __future__ import annotations

import json
import threading
import time
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
import pytest

from lightwork.config import Settings
from lightwork.contracts.calls import CallMeta, request_fingerprint
from lightwork.contracts.canonical import sha256_hex
from lightwork.contracts.events import Event, read_events
from lightwork.contracts.layout import Layout
from lightwork.contracts.render import Tts
from lightwork.contracts.stepgraph import StepGraph
from lightwork.core.scenarios import calibration
from lightwork.live import CaptureUnavailable, LiveSession, validate_live_graph
from lightwork.live.diagnostic import FixtureCamera, FixtureClock, replay_as_live
from lightwork.models.errors import ModelAccessError
from lightwork.models.pricing import cost_usd
from lightwork.models.record import write_call
from lightwork.models.synthetic import SamObject, responses_sse, sam_sse
from lightwork.models.wire import read_body
from lightwork.runs import Recorder, audit_run, verify_run
from lightwork.server.hub import RenderHub, make_hubs
from lightwork.vision.capture import open_camera
from lightwork.vision.measure_frames import encode_jpeg

pytestmark = pytest.mark.usefixtures("no_network")


def artefacts():
    graph = StepGraph.model_validate({
        "id": "synthetic-live", "title": "Synthetic live concurrency",
        "source": {"kind": "text", "ref": "tests/live", "licence": "Apache-2.0", "attribution": "Synthetic test only"},
        "objects": [{"id": obj, "noun": obj, "kind": "item"} for obj in ("a", "b", "c")]
                   + [{"id": "slot", "noun": "slot", "kind": "zone"}],
        "steps": [{"id": f"s{i}", "say": "Move the items into the slot.", "targets": ["slot"], "checks": [
            *[{"id": obj, "kind": "present", "object": obj, "region": "slot"} for obj in ("a", "b", "c")],
            {"id": "judge", "kind": "judge", "question": "Are the items correctly seated?"},
        ]} for i in (1, 2)],
    })
    layout = Layout.model_validate({
        "pack": graph.id, "created": "2026-10-29T12:00:00Z", "work_area_xywh": [10, 20, 80, 50],
        "objects": {obj: {"kind": "zone" if obj == "slot" else "item",
            "corners_cam_px": [[20, 25], [80, 25], [80, 65], [20, 65]],
            "sam": {"mode": "noun", "noun": obj}} for obj in ("a", "b", "c", "slot")},
    })
    calib = calibration()
    return graph, layout, calib


class Calls:
    """Synthetic raw responses, gated independently to expose completion-order races."""
    def __init__(self, run_dir):
        self.run_dir = run_dir
        self.permits = defaultdict(threading.Event)
        self.entered = defaultdict(threading.Event)
        self.active = Counter()
        self.peak = Counter()
        self.requests = []
        self.lock = threading.Lock()
        self.closed = False
        self.access_error = None

    def execute(self, *, intent_id, role, endpoint, request, frames, index, timeout_s):
        with self.lock:
            self.active[role] += 1
            self.peak[role] = max(self.peak[role], self.active[role])
            self.requests.append((intent_id, request, frames))
        self.entered[intent_id].set()
        try:
            if not self.permits[intent_id].wait(8):
                raise TimeoutError(f"test did not release {intent_id}")
            raw = (sam_sse([SamObject(1, (15, 10, 30, 25))], size=(80, 50)) if role == "segment"
                   else responses_sse('{"done":"yes","contradiction":null,"evidence":"Synthetic test."}', model=request["model"]))
            reading = read_body(raw, "sse", endpoint)
            status = "error" if intent_id == self.access_error else "ok"
            meta = CallMeta(backend="meta", base_host="api.meta.ai", endpoint=endpoint, model=request["model"],
                status=status, http_status=401 if status == "error" else 200, attempts=[], headers={},
                response_id=reading.response_id, ttfb_ms=1, total_ms=5, usage=reading.usage,
                cost_usd=cost_usd(request["model"], reading.usage), fingerprint=request_fingerprint(request),
                provenance="synthetic")
            relative = write_call(self.run_dir, index=index, role=role, request=request, raw=raw, kind="sse", meta=meta)
            result = {"intent_id": intent_id, "call_dir": relative, "status": status, "latency_ms": 5,
                      "http_status": meta.http_status, "usage": meta.usage}
            if status == "error":
                raise ModelAccessError("test entitlement error", result)
            return result
        finally:
            with self.lock:
                self.active[role] -= 1

    def release_all(self):
        for step in ("s1", "s2"):
            for suffix in ("segment/a/1", "segment/b/1", "segment/c/1", "judge/1"):
                self.permits[f"{step}/{suffix}"].set()

    def close(self):
        self.closed = True


class Events:
    def __init__(self):
        self.events = []
        self.condition = threading.Condition()

    def __call__(self, batch):
        with self.condition:
            self.events.extend(batch)
            self.condition.notify_all()

    def until(self, pred, timeout=5):
        with self.condition:
            if not self.condition.wait_for(lambda: pred(self.events), timeout):
                raise AssertionError("expected runtime event did not arrive")
            return list(self.events)


def event(type_, payload, t):
    return Event(seq=1, t=t, type=type_, src="test", derived=False, payload=payload)


@pytest.fixture
def session_factory(tmp_path):
    sessions = []

    def create(*, quiet=False, voice=None, voice_dry_run=False, hubs=None, measurement_mode="grey"):
        graph, layout, calib = artefacts()
        calib = calib.model_copy(update={"measurement_mode": measurement_mode})
        clock = FixtureClock()
        recorder = Recorder(tmp_path / str(len(sessions)), graph.id, {
            "run_id": "20261029T120000Z_" + graph.id, "backend": "meta",
            "models": {"verdict": "muse-spark-1.3-contributor", "segment": "sam-3.1"},
            "code_commit": "0000000", "dirty": False, "platform": "synthetic", "version": "0.1.0",
        }, provenance="synthetic", clock=clock)
        calls = Calls(recorder.run_dir)
        camera = FixtureCamera(layout.work_area_xywh, calib.camera.model_dump())
        camera.set_frame(encode_jpeg(np.full((50, 80, 3), 100, np.uint8)), .1)
        session = LiveSession(Settings(root=tmp_path), graph, layout, calib, source=camera, recorder=recorder,
                              model_client=calls, hands=False, voice=voice or False, quiet=quiet,
                              voice_dry_run=voice_dry_run, scripted=True, clock=clock, hubs=hubs)
        observer = Events()
        session.subscribe(observer)
        sessions.append((session, calls))
        session.start()
        return session, calls, camera, clock, observer

    yield create
    for session, calls in sessions:
        calls.release_all()
        session.stop("test teardown")
        try:
            session.join(10)
        except BaseException:
            pass
        session.render.close()
        session.control.close()


def capture_action(session, camera, clock, value, t):
    clock.t = t
    session.feed_recorded_input(event("motion", {"state": "moving", "active_regions": ["slot"]}, t)).result(2)
    clock.t = t + .7
    session.feed_recorded_input(event("motion", {"state": "settled", "active_regions": ["slot"]}, t+.7)).result(2)
    clock.t = t + 1
    camera.set_frame(encode_jpeg(np.full((50, 80, 3), value, np.uint8)), t+.95)
    return session.capture_now().result(3)


def test_fixture_replay_as_live_is_exact_offline(tmp_path, synthetic_kit_run):
    result = replay_as_live(synthetic_kit_run, tmp_path)
    assert result.verify.ok, result.verify.diffs
    assert result.source_derived_equal
    assert "runs-synthetic" in result.run_dir.parts
    assert audit_run(result.run_dir)  # deliberately not submission evidence


def test_concurrent_results_one_provisional_and_draining(session_factory):
    session, calls, camera, clock, observed = session_factory()
    clock.t = .2
    session.capture_now().result(3)
    first = capture_action(session, camera, clock, 130, 1)
    for suffix in ("segment/a/1", "segment/b/1", "segment/c/1", "judge/1"):
        assert calls.entered[f"s1/{suffix}"].wait(3)
    second = capture_action(session, camera, clock, 160, 3)
    assert first.payload["frame_sha"] != second.payload["frame_sha"]
    assert calls.peak["segment"] == 3
    assert all(iid.startswith("s1/") for iid, *_ in calls.requests)
    # Reverse completion order must not reorder recorded input/reducer batches.
    for iid in ("s1/judge/1", "s1/segment/c/1", "s1/segment/a/1", "s1/segment/b/1"):
        calls.permits[iid].set()
        observed.until(lambda events: any(e.type == "model_result" and e.payload["intent_id"] == iid for e in events))
    observed.until(lambda events: any(e.type == "step" and e.payload["step"] == "s2" and e.payload["state"] == "provisional" for e in events))
    assert calls.entered["s2/judge/1"].wait(3)
    session.stop("operator with calls outstanding")
    with pytest.raises(TimeoutError):
        session.join(.03)
    calls.release_all()
    path = session.join(5)
    logged = read_events(path / "events.jsonl")
    assert [e.payload["intent_id"] for e in logged if e.type == "model_result"][:4] == [
        "s1/judge/1", "s1/segment/c/1", "s1/segment/a/1", "s1/segment/b/1"]
    assert logged[-1].type == "run_stopped"
    assert sum(e.type == "model_result" for e in logged) == 8
    assert [e.t for e in logged] == sorted(e.t for e in logged)
    assert calls.peak["segment"] <= 3 and calls.peak["verdict"] == 1
    assert calls.closed and camera.closed
    assert verify_run(path).ok
    assert not any(t.name.startswith(("lightwork-sam", "lightwork-model", "lightwork-measure", "lightwork-live")) for t in threading.enumerate())


def test_late_reply_cannot_pass_confirmed_timeout(session_factory):
    session, calls, camera, clock, observed = session_factory()
    clock.t = .2
    session.capture_now().result(3)
    capture_action(session, camera, clock, 130, 1)
    clock.t = 30
    session.feed_recorded_input(event("tick", {}, 30)).result(2)
    session.on_key("space").result(2)
    calls.release_all()
    observed.until(lambda events: sum(e.type == "model_result" for e in events) == 4)
    session.stop()
    path = session.join(5)
    states = [e.payload["state"] for e in read_events(path / "events.jsonl") if e.type == "step" and e.payload["step"] == "s1"]
    assert "unverified" in states and "passed" not in states
    assert verify_run(path).ok


def test_access_failure_is_recorded_and_drains_other_calls(session_factory):
    session, calls, camera, clock, observed = session_factory()
    calls.access_error = "s1/judge/1"
    clock.t = .2
    session.capture_now().result(3)
    capture_action(session, camera, clock, 130, 1)
    calls.release_all()
    with pytest.raises(ModelAccessError, match="entitlement"):
        session.join(5)
    events = read_events(session.run_dir / "events.jsonl")
    assert sum(e.type == "model_result" for e in events) == 4
    assert not any(e.type == "step" and e.payload["state"] == "passed" for e in events)
    assert events[-1].type == "run_stopped"
    assert verify_run(session.run_dir).ok


class Voice:
    def __init__(self):
        self.tts = []
        self.transitions = 0
        self.muted = False

    def step_changed(self):
        self.transitions += 1

    def on_tts(self, tts):
        self.tts.append(tts.state)
        self.muted = tts.state == "start"

    def set_muted(self, value):
        self.muted = value

    def close(self):
        pass

    def stats(self):
        return {"closes": [], "latencies_ms": []}


class Render(RenderHub):
    def __init__(self):
        super().__init__()
        self.spoken = []
        self.projector = False
        self.ack = None
        self.modes = []

    def say(self, text, priority="normal"):
        self.spoken.append(text)
        return super().say(text, priority)

    def hello(self, role):
        return object() if role == "projector" and self.projector else None

    def show_mode(self, mode, pattern=None, timeout=1):
        self.modes.append(mode)
        return self.ack() if callable(self.ack) else self.ack


@pytest.mark.parametrize("quiet,dry", [(True, False), (False, True), (False, False)])
def test_speech_and_voice_callbacks_obey_modes(session_factory, quiet, dry):
    voice, renderer = Voice(), Render()
    hubs = make_hubs()
    hubs["render_hub"].close()
    hubs["render_hub"] = renderer
    session, _, _, _, observed = session_factory(quiet=quiet, voice=voice, voice_dry_run=dry, hubs=hubs)
    session.on_ui("start", {"tts": True, "mic": True}).result(2)
    session.on_voice("voice", {"command": "repeat", "transcript": "repeat", "session_id": "test", "turn_id": "one"})
    session.on_key("r").result(2)  # queue barrier, not a sleep
    assert bool(renderer.spoken) is not quiet
    assert sum(e.type == "voice" for e in observed.events) == int(not quiet and not dry)
    session._on_tts(Tts(id="x", state="start"))
    assert voice.muted
    session._on_tts(Tts(id="x", state="end"))
    assert not voice.muted and voice.tts == ["start", "end"]
    assert voice.transitions > 0


def test_projected_capture_waits_past_ack_and_verifies_crop(session_factory, monkeypatch):
    renderer = Render()
    hubs = make_hubs()
    hubs["render_hub"].close()
    hubs["render_hub"] = renderer
    session, _, camera, clock, observed = session_factory(hubs=hubs, measurement_mode="dark")
    renderer.projector = True
    clock.t = 1
    renderer.ack = clock
    camera.set_frame(encode_jpeg(np.full((50, 80, 3), 100, np.uint8)), 1.3)
    original = camera.wait_frame
    waits = []

    def wait_frame(after_t, timeout):
        waits.append(after_t)
        frame = original(after_t, timeout)
        if frame is not None:
            clock.t = max(clock.t, frame.t_mono)
        return frame

    monkeypatch.setattr(camera, "wait_frame", wait_frame)
    frame = session.capture_now().result(3)
    assert waits[0] == pytest.approx(1.12)
    assert renderer.modes == ["dark", "normal"]
    assert frame.payload["mode"] == "dark" and frame.payload["verified"]
    assert frame.payload["t_capture"] <= frame.t
    jpeg = session.recorder.store.frame_bytes(frame.payload["frame_sha"])
    assert cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR).shape[:2] == (50, 80)


def test_capture_failure_and_projector_disconnect_never_accept_evidence(session_factory):
    renderer = Render()
    hubs = make_hubs()
    hubs["render_hub"].close()
    hubs["render_hub"] = renderer
    session, calls, camera, clock, observed = session_factory(hubs=hubs)
    renderer.projector = True
    clock.t = 1
    camera.set_frame(encode_jpeg(np.full((50, 80, 3), 100, np.uint8)), 1.3)
    with pytest.raises(CaptureUnavailable, match="ACK"):
        session.capture_now().result(3)
    assert renderer.modes == ["grey", "normal"]
    renderer.projector = False
    with pytest.raises(CaptureUnavailable, match="disconnected"):
        session.capture_now().result(3)
    assert not calls.requests
    assert not any(e.type == "keyframe" for e in observed.events)


def test_unverified_crop_is_logged_but_never_model_evidence(session_factory):
    session, calls, camera, clock, observed = session_factory()
    clock.t = 1
    camera.set_frame(encode_jpeg(np.full((50, 80, 3), 255, np.uint8)), .95)
    with pytest.raises(CaptureUnavailable, match="content"):
        session.capture_now().result(3)
    frames = [e for e in observed.events if e.type == "keyframe"]
    assert len(frames) == 1 and not frames[0].payload["verified"]
    saved = session.recorder.store.frame_bytes(frames[0].payload["frame_sha"])
    assert cv2.imdecode(np.frombuffer(saved, np.uint8), cv2.IMREAD_COLOR).shape[:2] == (50, 80)
    assert not calls.requests


def test_compatibility_and_privacy_refuse_before_running():
    graph, layout, _ = artefacts()
    with pytest.raises(ValueError, match="SAM checks"):
        validate_live_graph(graph, layout, "chat")
    changed = graph.model_dump()
    changed["steps"][0]["checks"].append({"id": "second", "kind": "judge", "question": "Another question?"})
    with pytest.raises(ValueError, match="one verdict"):
        validate_live_graph(StepGraph.model_validate(changed), layout, "meta")


def test_collection_hooks_write_before_manifest_and_inputs_serialize(session_factory):
    session, _, camera, clock, observed = session_factory()
    session.before_close(lambda live: (live.recorder.run_dir / "labels.jsonl").write_text("", encoding="utf-8"))
    session.set_scene_transform(lambda scene: scene.model_copy(update={"banner": "trial card"}))
    session.on_ui("trial_review", {"reviewed": True}).result(2)
    clock.t = .3
    frame = session.capture_now().result(3)
    assert frame.type == "keyframe" and frame.payload["verified"]
    session.stop()
    path = session.join(5)
    manifest = json.loads((path / "manifest.json").read_bytes())
    assert manifest["files"]["labels.jsonl"] == sha256_hex(b"")
    assert any(e.type == "ui" and e.payload["action"] == "trial_review" for e in observed.events)
    assert observed.events[-1].type == "run_stopped"
    with pytest.raises(RuntimeError, match="not accepting"):
        session.capture_now().result()
    assert verify_run(path).ok


def test_actual_capture_gate_pipeline_and_crop_preview(tmp_path):
    graph, layout, calib = artefacts()
    images = tmp_path / "images"
    images.mkdir()
    # Camera source contains bright out-of-work-area pixels. None may reach disk/network.
    for i in range(50):
        image = np.full((80, 100, 3), 255, np.uint8)
        image[20:70, 10:90] = 100
        if i >= 25:
            image[30:50, 30:55] = 30
        (images / f"{i:03}.jpg").write_bytes(encode_jpeg(image))
    camera = open_camera(str(images), loop=True)
    recorder = Recorder(tmp_path, graph.id, {"backend": "meta", "models": {"verdict": "muse-spark-1.3-contributor", "segment": "sam-3.1"},
        "code_commit": "0000000", "dirty": False, "platform": "synthetic", "version": "0.1.0"}, provenance="synthetic")
    calls = Calls(recorder.run_dir)
    calls.release_all()
    session = LiveSession(Settings(root=tmp_path), graph, layout, calib, source=camera,
                          recorder=recorder, model_client=calls, hands=False, voice=False, quiet=True)
    observed = Events()
    session.subscribe(observed)
    try:
        session.start()
        observed.until(lambda events: any(e.type == "motion" and e.payload["state"] == "moving" for e in events), timeout=5)
        observed.until(lambda events: any(e.type == "model_result" for e in events), timeout=5)
        session.stop()
        path = session.join(5)
        assert camera.stats()["frames"] > 30
        assert any(e.type == "ui" and e.payload["action"] == "runtime_mode" and e.payload["args"]["hands"] == "motion-only-advisory" for e in observed.events)
        assert not any(e.type == "hands" for e in observed.events)
        for jpg in list((path / "frames").glob("*.jpg")) + list((path / "preview").glob("*.jpg")):
            crop = cv2.imread(str(jpg))
            assert crop.shape[:2] == (50, 80)
            assert crop.max() < 180
        assert calls.requests
        for _, _, frames in calls.requests:
            for jpeg in frames.values():
                assert cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR).shape[:2] == (50, 80)
        assert verify_run(path).ok
    finally:
        session.stop()
        try:
            session.join(5)
        finally:
            session.render.close()
            session.control.close()
