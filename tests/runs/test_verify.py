"""Recorder, verify, audit and replay mechanics with a deterministic double and the production reducer."""

from __future__ import annotations

import json
import math
from pathlib import Path

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from lightwork.cli import main
from lightwork.contracts.calls import CallMeta, frame_ref, request_fingerprint
from lightwork.contracts.canonical import canonical, canonical_sha, q_od, q_s
from lightwork.contracts.events import read_events
from lightwork.contracts.layout import Calibration, Layout
from lightwork.contracts.render import Ring, SceneMsg
from lightwork.contracts.run import VerifyReport
from lightwork.contracts.server import unpack_camera_frame
from lightwork.contracts.stepgraph import StepGraph
from lightwork.models.parse import parse_response
from lightwork.models.pricing import aggregate_usage, cost_usd
from lightwork.models.record import write_call as write_model_call
from lightwork.models.synthetic import responses_sse
from lightwork.models.wire import read_body
from lightwork.runs import Recorder, ReplayEngine, audit_run, verify_run
from lightwork.runs.manifest import build_manifest
from lightwork.server.app import create_app
from lightwork.server.hub import make_hubs
from lightwork.testing.synth import Item, encode_jpeg, kit_scene

pytestmark = pytest.mark.usefixtures("no_network")

SPARK = "muse-spark-1.3-contributor"
PACK = "fake-kit"
RUN_ID = f"20261030T120000Z_{PACK}"
STARTED = {
    "run_id": RUN_ID,
    "backend": "meta",
    "models": {"verdict": SPARK, "segment": "sam-3.1"},
    "code_commit": "0000000",
    "dirty": False,
    "platform": "test",
    "version": "0.1.0",
}
STEPGRAPH = StepGraph.model_validate(
    {
        "id": PACK,
        "title": "Fake kit",
        "source": {"kind": "text", "ref": "tests/runs", "licence": "Apache-2.0", "attribution": "Lightwork authors"},
        "objects": [{"id": "remote", "noun": "remote control", "kind": "item"}, {"id": "slot", "noun": "slot", "kind": "zone"}],
        "steps": [
            {
                "id": "remote-in",
                "say": "Put the remote in its slot.",
                "targets": ["slot"],
                "checks": [
                    {"id": "hand-at-slot", "kind": "activity_at", "region": "slot"},
                    {"id": "remote-judge", "kind": "judge", "question": "Is the remote in the slot?"},
                ],
            },
            {"id": "close-case", "say": "Close the case.", "checks": [{"id": "case-judge", "kind": "judge", "question": "Is it closed?"}]},
        ],
    }
)
LAYOUT = Layout.model_validate(
    {
        "pack": PACK,
        "created": "2026-10-30T12:00:00Z",
        "work_area_xywh": [0, 0, 640, 360],
        "objects": {
            "remote": {"kind": "item", "polygon_cam_px": [[80, 250], [120, 250], [120, 350], [80, 350]], "sam": {"mode": "noun", "noun": "remote control"}},
            "slot": {"kind": "zone", "corners_cam_px": [[300, 100], [400, 100], [400, 200], [300, 200]], "sam": {"mode": "noun", "noun": "slot"}},
        },
    }
)
CALIB = Calibration.model_validate(
    {
        "H_cam_to_page": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        "page": {"w": 854.0, "h": 480.0, "dpr": 1.0},
        "camera": {"device_id": "fake", "mode": "MJPG 640x360@30", "controls": {}},
        "residuals": {"rms_px": 0.5, "p95_px": 0.9, "max_px": 1.2, "n_markers": 15, "n_inliers": 60},
        "mm": {"source": "nominal"},
        "measurement_mode": "grey",
        "created": "2026-10-30T11:55:00Z",
    }
)
VERDICT = {"done": "yes", "contradiction": None, "evidence": "The remote lies in the slot."}
USAGE = {"input_tokens": 1400, "output_tokens": 30}
CALL_COST = cost_usd(SPARK, USAGE)


def scene_hash(scene: SceneMsg) -> str:
    return canonical_sha(scene.model_dump(mode="json", exclude={"seq"}))


class FakeReducer:
    """A hand over the slot, then a keyframe whose red-channel OD reaches ``threshold``, makes step 1 provisional and
    asks for a verdict (prompt text and version from ``prompt_version``); an ok "yes" passes it."""

    def __init__(self, stepgraph, layout, calib, store, *, backend, models, threshold=0.0, prompt_version="verdict-v1"):
        self.steps = stepgraph.steps
        self.store = store
        self.model = models["verdict"]
        self.threshold = threshold
        self.prompt_version = prompt_version
        self.hand = False
        self.cost = 0.0
        self.requests: dict[str, tuple[str, dict]] = {}
        self._scene = SceneMsg(seq=0, mode="normal", prims=[])

    @property
    def scene(self) -> SceneMsg:
        return self._scene

    def request_for(self, intent_id: str) -> tuple[str, dict]:
        return self.requests[intent_id]

    def feed(self, ev):
        assert not ev.derived, "verify must feed inputs only"
        handler = getattr(self, f"on_{ev.type}", None)
        return handler(ev) if handler else []

    def activate(self, i: int) -> list:
        step = self.steps[i]
        self._scene = SceneMsg(seq=0, mode="normal", prims=[Ring(id=f"t-{step.id}", c=[350.0, 150.0 + i], r=60.0, role="target")])
        return [
            ("step", {"step": step.id, "state": "active", "reason": ""}),
            ("scene", {"hash": scene_hash(self._scene)}),
            ("say", {"text": step.say, "priority": "normal"}),
        ]

    def on_layout_loaded(self, ev):
        return self.activate(0)

    def on_hands(self, ev):
        self.hand = bool(ev.payload["over_regions"])
        return []

    def on_keyframe(self, ev):
        sha = ev.payload["frame_sha"]
        img = cv2.imdecode(np.frombuffer(self.store.frame_bytes(sha), np.uint8), cv2.IMREAD_COLOR)
        od = q_od(-math.log10(max(float(img[:, :, 2].mean()), 1.0) / 255.0))
        cell = {"od": od, "lab": [0.0, 0.0, 0.0], "n": int(img.shape[0] * img.shape[1]), "sd": 0.0}
        out = [("measurement", {"frame_sha": sha, "channel": "r", "cells": {"work": cell}})]
        if not self.hand:
            return out
        self.hand = False
        step = self.steps[0].id
        if od < self.threshold:
            return out + [
                ("check", {"step": step, "check": "hand-at-slot", "status": "fail", "reason": f"od {od}", "evidence": [f"frame:{sha}"]}),
                ("step", {"step": step, "state": "failed", "reason": "no change at the slot"}),
            ]
        intent = f"{step}/remote-judge/1"
        content = [{"type": "input_text", "text": f"[{self.prompt_version}] {self.steps[0].say}"}, {"type": "input_image", "image_url": frame_ref(sha)}]
        request = {"model": self.model, "input": [{"type": "message", "role": "user", "content": content}], "store": False, "stream": True}
        self.requests[intent] = ("/v1/responses", request)
        mr = {"intent_id": intent, "role": "verdict", "model": self.model, "fingerprint": request_fingerprint(request), "frame_sha": sha, "prompt_version": self.prompt_version}
        return out + [
            ("check", {"step": step, "check": "hand-at-slot", "status": "pass", "reason": f"od {od}", "evidence": [f"frame:{sha}"]}),
            ("step", {"step": step, "state": "provisional", "reason": "local checks passed"}),
            ("model_request", mr),
            *self.activate(1),
        ]

    def on_model_result(self, ev):
        intent = ev.payload["intent_id"]
        step = intent.split("/")[0]
        raw, kind = self.store.call_response(ev.payload["call_dir"])
        verdict = parse_response("verdict", raw, kind, "/v1/responses")["result"]
        self.cost = round(self.cost + cost_usd(self.model, ev.payload.get("usage") or USAGE), 6)
        ok = verdict["done"] == "yes"
        return [
            ("parsed", {"intent_id": intent, "kind": "verdict", "result": verdict}),
            ("check", {"step": step, "check": "remote-judge", "status": "pass" if ok else "fail", "reason": f"verdict {verdict['done']}", "evidence": [f"intent:{intent}"]}),
            ("step", {"step": step, "state": "passed" if ok else "failed", "reason": verdict["evidence"]}),
            ("cost", {"usd_total": self.cost}),
        ]


def fake(**params):
    return lambda *a, **k: FakeReducer(*a, **k, **params)


def write_call(rec: Recorder, request: dict, raw: bytes, provenance: str, prior_usage: dict | None) -> str:
    index, call_dir = rec.alloc_call("verdict", request["model"])
    usage = aggregate_usage([prior_usage, USAGE])
    previous = [] if prior_usage is None else [
        (responses_sse("partial", model=SPARK, response_id="resp_synthetic_retry",
                       usage=prior_usage, terminal="incomplete"), "sse")
    ]
    meta = CallMeta(
        backend="meta",
        base_host="api.meta.ai",
        endpoint="/v1/responses",
        model=request["model"],
        status="ok",
        http_status=200,
        attempts=[{"n": n + 1, "status": 200, "error": None, "wait_s": 0.0, "ms": 2412}
                  for n in range(len(previous) + 1)],
        headers={},
        response_id=read_body(raw, "sse", "/v1/responses").response_id,
        ttfb_ms=1880,
        total_ms=2412,
        usage=usage,
        cost_usd=cost_usd(request["model"], usage),
        fingerprint=request_fingerprint(request),
        provenance=provenance,
    )
    assert write_model_call(
        rec.run_dir, index=index, role="verdict", request=request, raw=raw, kind="sse", meta=meta,
        attempt_responses=previous,
    ) == call_dir
    return call_dir


def kit_frames() -> tuple[np.ndarray, np.ndarray]:
    case, slots = (280, 80, 160, 160), {"slot": (300, 100, 100, 100)}
    before = kit_scene([Item("remote", "rect", (40, 40, 40), (100, 300), (40, 100))], size=(640, 360), case_rect=case, slots=slots)
    after = kit_scene([Item("remote", "rect", (40, 40, 40), (350, 150), (80, 30))], size=(640, 360), case_rect=case, slots=slots)
    return before, after


def record(root: Path, provenance: str = "synthetic", *, factory=FakeReducer, prior_usage: dict | None = None) -> Path:
    """Synthetic kit; explicit non-synthetic labels are metadata doubles used only by release-audit unit tests."""
    now = [100.0]
    started = {**STARTED, "platform": "synthetic" if provenance == "synthetic" else "test"}
    rec = Recorder(root, PACK, started, provenance=provenance, clock=lambda: now[0])
    stepgraph_sha = rec.put_run_file("stepgraph.json", STEPGRAPH)
    calib_sha = rec.put_run_file("calib.json", CALIB)
    layout_sha = rec.put_run_file("layout.json", LAYOUT)
    reducer = factory(STEPGRAPH, LAYOUT, CALIB, rec.store, backend="meta", models=STARTED["models"])
    rec.attach(reducer)
    rec.ingest("procedure_loaded", "orchestrator", {"stepgraph_sha": stepgraph_sha}, 0.0)
    rec.ingest("calibration_loaded", "orchestrator", {"calib_sha": calib_sha}, 0.0)
    rec.ingest("layout_loaded", "orchestrator", {"layout_sha": layout_sha}, 0.0)
    before, after = kit_frames()
    for t, img in ((0.5, before), (4.3, after)):
        sha = rec.put_frame(encode_jpeg(img))
        keyframe = {"frame_sha": sha, "crop_xywh": [0, 0, 640, 360], "mode": "grey", "verified": True, "t_capture": q_s(t - 0.02)}
        out = rec.ingest("keyframe", "camera", keyframe, t)
        rec.put_preview(img, t)
        if t < 1:
            rec.ingest("motion", "gate", {"state": "moving", "active_regions": ["slot"]}, 2.1)
            rec.ingest("hands", "hands", {"over_regions": ["slot"], "tip_xy": [350.0, 150.0]}, 2.2)
            rec.ingest("motion", "gate", {"state": "settled", "active_regions": []}, 4.1)
    intent = next(e.payload["intent_id"] for e in out if e.type == "model_request")
    response_id = "resp_synthetic_kit" if provenance == "synthetic" else "resp_metadata_double"
    raw = responses_sse(canonical(VERDICT), model=SPARK, response_id=response_id, usage=USAGE)
    call_dir = write_call(rec, reducer.request_for(intent)[1], raw, provenance, prior_usage)
    rec.ingest("tick", "orchestrator", {}, 5.5)
    rec.ingest("model_result", "model", {
        "intent_id": intent, "call_dir": call_dir, "status": "ok", "latency_ms": 2412,
        "usage": aggregate_usage([prior_usage, USAGE]),
        "http_status": 200,
    }, 6.712)
    now[0] = 109.0
    return rec.close("test end")


@pytest.fixture
def run_dir(tmp_path) -> Path:
    return record(tmp_path)


def diff_keys(report: VerifyReport) -> list[tuple[str, str | None]]:
    return [(d.cls, d.path) for d in report.diffs]


def seq_of(run_dir: Path, type_: str) -> int:
    return next(e.seq for e in read_events(run_dir / "events.jsonl") if e.type == type_)


def edit_event(run_dir: Path, seq: int, **payload) -> None:
    path = run_dir / "events.jsonl"
    lines = path.read_text(encoding="utf-8").splitlines()
    ev = json.loads(lines[seq - 1])
    ev["payload"].update(payload)
    lines[seq - 1] = canonical(ev)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ---------------------------------------------------------------- verify


def test_round_trip_has_zero_diffs_and_writes_verify_json(run_dir):
    report = verify_run(run_dir, reducer_factory=FakeReducer)
    assert report.ok and report.diffs == []
    assert (report.run_id, report.n_calls, report.code_version) == (RUN_ID, 1, "0.1.0")
    assert report.n_events == report.n_inputs + report.n_derived and report.n_derived > report.n_inputs // 2
    first = (run_dir / "verify.json").read_bytes()
    assert VerifyReport.model_validate_json(first) == report
    assert verify_run(run_dir, reducer_factory=FakeReducer).ok  # verify.json is outside the manifest
    assert (run_dir / "verify.json").read_bytes() == first
    totals = json.loads((run_dir / "manifest.json").read_text())["totals"]
    assert totals == {"passed": 1, "failed": 0, "unverified": 0, "calls": 1, "cost_usd": CALL_COST, "duration_s": 9.0}


def test_flipped_frame_byte_is_a_manifest_diff(run_dir):
    sha = [e.payload["frame_sha"] for e in read_events(run_dir / "events.jsonl") if e.type == "keyframe"][-1]
    path = run_dir / "frames" / f"{sha}.jpg"
    data = bytearray(path.read_bytes())
    data[len(data) // 2] ^= 0x01
    path.write_bytes(bytes(data))
    report = verify_run(run_dir, reducer_factory=FakeReducer, write=False)
    assert diff_keys(report)[0] == ("manifest", f"frames/{sha}.jpg")
    assert "differs from manifest" in report.diffs[0].detail and not report.ok


def test_edited_event_is_a_chain_diff(run_dir):
    seq = seq_of(run_dir, "motion")  # an input the reducer ignores: only the log itself can tell
    edit_event(run_dir, seq, active_regions=["case"])  # same length: the size still matches
    report = verify_run(run_dir, reducer_factory=FakeReducer, write=False)
    assert [(d.cls, d.seq, d.path) for d in report.diffs] == [("chain", seq, "events.jsonl"), ("manifest", None, "events.jsonl")]


def test_edited_derived_event_is_a_chain_and_derived_diff(run_dir):
    seq = seq_of(run_dir, "say")
    edit_event(run_dir, seq, text="Put the cable in.")
    report = verify_run(run_dir, reducer_factory=FakeReducer, write=False)
    assert report.diffs[0].cls == "chain" and report.diffs[0].seq == seq
    derived = [d for d in report.diffs if d.cls == "derived"]
    assert [d.seq for d in derived] == [seq]
    assert derived[0].detail == 'say payload.text: logged "Put the cable in.", re-derived "Put the remote in its slot."'


def test_changed_prompt_version_is_a_fingerprint_diff(run_dir):
    report = verify_run(run_dir, reducer_factory=fake(prompt_version="verdict-v2"), write=False)
    assert {d.cls for d in report.diffs} == {"fingerprint"}
    assert [d.path for d in report.diffs] == [None, "calls/0001-verdict-muse-spark-1.3-contributor/request.json"]
    assert report.diffs[0].seq == seq_of(run_dir, "model_request")


def test_changed_reducer_threshold_is_a_derived_diff(run_dir):
    report = verify_run(run_dir, reducer_factory=fake(threshold=10.0), write=False)
    first = report.diffs[0]
    assert first.cls == "derived" and first.seq == seq_of(run_dir, "check")
    assert first.detail.startswith('check payload.status: logged "pass", re-derived "fail"')
    assert {d.cls for d in report.diffs} == {"derived", "fingerprint"}


def test_changed_derived_stream_length_and_invalid_output(run_dir):
    class Chatty(FakeReducer):
        def on_tick(self, ev):
            return [("say", {"text": "still here", "priority": "normal"})]

    class Broken(FakeReducer):
        def on_tick(self, ev):
            return [("say", {"text": "no priority"})]

    tick = seq_of(run_dir, "tick")
    chatty = verify_run(run_dir, reducer_factory=Chatty, write=False)
    assert [(d.cls, d.seq) for d in chatty.diffs] == [("derived", tick)] and "still here" in chatty.diffs[0].detail
    broken = verify_run(run_dir, reducer_factory=Broken, write=False)
    assert [(d.cls, d.seq) for d in broken.diffs] == [("schema", tick), ("derived", tick)]


def test_files_outside_the_manifest_and_missing_files(run_dir):
    preview = sorted((run_dir / "preview").iterdir())[0]
    preview.unlink()
    (run_dir / "frames" / "stray.txt").write_text("x")
    report = verify_run(run_dir, reducer_factory=FakeReducer, write=False)
    keys = diff_keys(report)
    assert ("extra", "frames/stray.txt") in keys and ("missing", f"preview/{preview.name}") in keys
    assert {d.cls for d in report.diffs} == {"extra", "missing", "manifest"}  # + size_bytes


def test_edited_request_is_caught_by_manifest_and_fingerprint(run_dir):
    req = next(run_dir.glob("calls/*/request.json"))
    body = json.loads(req.read_text())
    body["store"] = True
    req.write_text(canonical(body) + "\n")
    report = verify_run(run_dir, reducer_factory=FakeReducer, write=False)
    assert {d.cls for d in report.diffs} == {"manifest", "fingerprint"}
    assert [d.path.rsplit("/", 1)[1] for d in report.diffs if d.cls == "fingerprint"] == ["meta.json", "request.json"]


def test_incomplete_run_is_reported_and_its_prefix_still_verifies(run_dir):
    (run_dir / "manifest.json").unlink()  # the recorder never reached close()
    report = verify_run(run_dir, reducer_factory=FakeReducer, write=False)
    assert diff_keys(report) == [("missing", "manifest.json")]
    assert "incomplete run" in report.diffs[0].detail
    with open(run_dir / "events.jsonl", "ab") as f:  # a crash in the middle of an append
        f.write(b'{"derived":false,"h":"ab')
    torn = verify_run(run_dir, reducer_factory=FakeReducer, write=False)
    assert diff_keys(torn) == [("schema", "events.jsonl"), ("missing", "manifest.json")]


def test_missing_run_input_skips_rederivation(run_dir):
    (run_dir / "layout.json").unlink()
    report = verify_run(run_dir, reducer_factory=FakeReducer, write=False)
    assert diff_keys(report).count(("missing", "layout.json")) == 1
    assert any(d.cls == "derived" and d.detail.startswith("not re-derived") for d in report.diffs)


def test_cli_verify_exit_codes(run_dir, tmp_path, monkeypatch, capsys):
    monkeypatch.setattr("lightwork.runs.verify.default_reducer_factory", lambda: FakeReducer)
    assert main(["verify", str(run_dir.parent)]) == 0
    assert f"{RUN_ID}: 0 diffs (" in capsys.readouterr().out
    edit_event(run_dir, seq_of(run_dir, "motion"), active_regions=["case"])
    assert main(["verify", str(run_dir)]) == 1
    out = capsys.readouterr().out
    assert f"{RUN_ID}: 2 diffs" in out and "chain" in out
    assert main(["verify", str(tmp_path / "nowhere")]) == 2


# ---------------------------------------------------------------- audit


def rewrite_call_meta(run_dir: Path, **fields) -> str:
    """Edit the run's call ``meta.json`` and re-seal the manifest (as a hand edit followed by a re-hash would)."""
    path = next(run_dir.glob("calls/*/meta.json"))
    path.write_text(canonical({**json.loads(path.read_text()), **fields}) + "\n", encoding="utf-8")
    provenance = json.loads((run_dir / "manifest.json").read_bytes())["provenance"]
    manifest = build_manifest(run_dir, read_events(run_dir / "events.jsonl"), provenance)
    (run_dir / "manifest.json").write_text(canonical(manifest) + "\n", encoding="utf-8")
    return path.relative_to(run_dir).as_posix()


def test_audit_accepts_valid_meta_metadata_double(tmp_path, capsys):
    run_dir = record(tmp_path, provenance="meta")
    assert audit_run(run_dir) == []
    assert main(["audit", str(run_dir.parent)]) == 0
    assert capsys.readouterr().out == f"{RUN_ID}: ok\n"


def test_audit_rejects_a_dev_standin_run(tmp_path, capsys):
    run = record(tmp_path, provenance="dev-standin")
    assert run == tmp_path / "runs-dev" / RUN_ID
    assert verify_run(run, reducer_factory=FakeReducer, write=False).ok  # intact, but not publishable
    assert audit_run(run) == ["provenance dev-standin: dev stand-in runs stay in runs-dev/ (D15)"]
    assert main(["audit", str(run.parent)]) == 1
    assert capsys.readouterr().out.startswith(f"{RUN_ID}: REJECTED\n")


def test_audit_rejects_a_stand_in_call_inside_a_meta_run(tmp_path):
    run_dir = record(tmp_path, provenance="meta")
    rel = rewrite_call_meta(run_dir, provenance="dev-standin")
    assert audit_run(run_dir) == [f"manifest    {rel}: provenance is meta but the call says 'dev-standin'"]


def test_audit_rejects_dev_cache_calls_and_incomplete_runs(tmp_path):
    run_dir = record(tmp_path, provenance="meta")
    rel = rewrite_call_meta(run_dir, attempts=[])
    assert audit_run(run_dir) == [f"{rel}: served from the dev cache, not a recorded call (D21)"]
    (run_dir / "manifest.json").unlink()
    assert audit_run(run_dir)[0].startswith("missing     manifest.json: incomplete run")


# ---------------------------------------------------------------- replay


class FakeClock:
    def __init__(self) -> None:
        self.t = 0.0
        self.sleeps: list[float] = []

    def now(self) -> float:
        return self.t

    def sleep(self, s: float) -> None:
        self.sleeps.append(s)
        self.t += s


class FakeHubs:
    def __init__(self, hellos_before_screen: int = 0) -> None:
        self.scenes: list[SceneMsg] = []
        self.previews: list[tuple[int, float, bytes]] = []
        self.state: dict = {}
        self.hello_calls = 0
        self.hellos_before_screen = hellos_before_screen

    def as_dict(self) -> dict:
        return {"render_hub": self, "preview_hub": self, "control_hub": self}

    def publish_scene(self, scene: SceneMsg) -> None:
        self.scenes.append(scene)

    def hello(self, role: str):
        self.hello_calls += 1
        return object() if self.hello_calls > self.hellos_before_screen else None

    def push(self, frame_id: int, t_mono: float, jpeg: bytes) -> None:
        self.previews.append((frame_id, t_mono, jpeg))

    def push_state(self, patch: dict) -> None:
        self.state.update(patch)


def test_replay_paces_by_recorded_time_and_renders(run_dir):
    clock, hubs, lines = FakeClock(), FakeHubs(hellos_before_screen=2), []
    engine = ReplayEngine(
        run_dir, speed=2.0, hubs=hubs.as_dict(), wait_for_screen=True, reducer_factory=FakeReducer,
        echo=lines.append, clock=clock.now, sleep=clock.sleep,
    )
    report = engine.run()
    assert report.ok
    assert clock.sleeps[:2] == [0.1, 0.1]  # waited for the screen page
    assert clock.t == pytest.approx(0.2 + 9.0 / 2)  # last input (run_stopped at 9.0) at 2x
    events = read_events(run_dir / "events.jsonl")
    scenes = [e.payload["hash"] for e in events if e.type == "scene"]
    assert [scene_hash(s.model_copy(update={"banner": None})) for s in hubs.scenes] == scenes
    previews = sorted(int(p.stem) for p in (run_dir / "preview").iterdir())
    assert [p[0] for p in hubs.previews] == previews
    assert hubs.previews[0][2] == (run_dir / "preview" / f"{previews[0]}.jpg").read_bytes()
    st = hubs.state
    assert st["run"] == {"id": RUN_ID, "pack": PACK, "mode": "replay", "backend": "meta", "provenance": "synthetic"}
    assert (st["preview_xywh"], st["camera"]) == ([0, 0, 640, 360], {"w": 640, "h": 360})
    assert [(s["id"], s["state"]) for s in st["steps"]] == [("remote-in", "passed"), ("close-case", "active")]
    assert st["trace"] == [
        {"intent_id": "remote-in/remote-judge/1", "role": "verdict", "model": SPARK, "status": "ok", "latency_ms": 2412, "cost_usd": CALL_COST}
    ]


def test_replay_reports_diffs_like_verify(run_dir):
    clock = FakeClock()
    engine = ReplayEngine(run_dir, speed=1.0, reducer_factory=fake(threshold=10.0), echo=lambda s: None, clock=clock.now, sleep=clock.sleep)
    report = engine.run()
    assert report == verify_run(run_dir, reducer_factory=fake(threshold=10.0), write=False)
    assert not (run_dir / "verify.json").exists()


def hello(role: str) -> dict:
    return {
        "type": "hello", "role": role, "innerWidth": 1280, "innerHeight": 720, "dpr": 1.0,
        "screen": {"width": 1280, "height": 720}, "orientation": "landscape-primary", "fullscreen": False,
        "secure": True, "wakelock": False, "ua": "pytest",
    }


def test_replay_through_the_server_reaches_the_screen_page(run_dir):
    hubs = make_hubs()
    report = ReplayEngine(run_dir, speed=1000.0, hubs=hubs, reducer_factory=FakeReducer, echo=lambda s: None).run()
    assert report.ok
    last_scene = [e.payload["hash"] for e in read_events(run_dir / "events.jsonl") if e.type == "scene"][-1]
    last_preview = max(int(p.stem) for p in (run_dir / "preview").iterdir())
    client = TestClient(create_app(mode="replay", runs_root=run_dir.parent, frames_dirs=[run_dir / "frames"], **hubs))
    with client:
        with client.websocket_connect("/ws/render") as ws:
            ws.send_json(hello("screen"))
            msg = ws.receive_json()
        assert msg["type"] == "scene"
        assert scene_hash(SceneMsg.model_validate({**msg, "banner": None})) == last_scene
        with client.websocket_connect("/ws/control") as ws:
            state = ws.receive_json()["state"]
        assert (state["run"]["mode"], state["run"]["provenance"]) == ("replay", "synthetic")
        assert [s["state"] for s in state["steps"]] == ["passed", "active"]
        with client.websocket_connect("/ws/camera") as ws:
            frame_id, _t, jpeg = unpack_camera_frame(ws.receive_bytes())
        assert (frame_id, jpeg) == (last_preview, (run_dir / "preview" / f"{last_preview}.jpg").read_bytes())
        assert client.get("/api/health").json()["mode"] == "replay"


def test_cli_replay_headless(run_dir, monkeypatch, capsys):
    monkeypatch.setattr("lightwork.runs.verify.default_reducer_factory", lambda: FakeReducer)
    assert main(["replay", str(run_dir), "--no-server", "--speed", "1000"]) == 0
    out = capsys.readouterr().out
    assert f"{RUN_ID}: 0 diffs" in out
    assert main(["replay", str(run_dir), "--no-server", "--speed", "0"]) == 2
    assert main(["replay", str(run_dir / "frames"), "--no-server"]) == 2


@pytest.fixture
def real_run(tmp_path):
    from lightwork.core import Reducer

    return record(tmp_path, factory=Reducer)


def test_production_reducer_record_verify_replay_round_trip(real_run, capsys):
    report = verify_run(real_run)
    assert report.ok, report.diffs
    assert main(["verify", str(real_run)]) == 0
    assert main(["replay", str(real_run), "--no-server", "--speed", "1000"]) == 0
    assert capsys.readouterr().out.count(f"{RUN_ID}: 0 diffs") == 2
    events = read_events(real_run / "events.jsonl")
    assert any(e.type == "step" and e.payload["state"] == "passed" for e in events)
    assert any("generated test data" in reason for reason in audit_run(real_run))


def test_production_prompt_change_is_fingerprint_diff(real_run, monkeypatch):
    from lightwork.models.prompts import verdict
    from lightwork.models.requests import PROMPT_VERSIONS

    monkeypatch.setattr(verdict, "INSTRUCTIONS", verdict.INSTRUCTIONS + "\nChanged prompt.")
    monkeypatch.setitem(PROMPT_VERSIONS, "verdict", "verdict-v2")
    report = verify_run(real_run, write=False)
    assert {d.cls for d in report.diffs} == {"fingerprint"}
    assert any(d.path and d.path.endswith("/request.json") for d in report.diffs)


def test_production_timeout_threshold_change_is_derived_diff(real_run):
    from dataclasses import replace
    from functools import partial

    from lightwork.core import Reducer
    from lightwork.core.params import DEFAULT

    factory = partial(Reducer, params=replace(DEFAULT, verify_timeout_meta_s=0.5))
    report = verify_run(real_run, reducer_factory=factory, write=False)
    assert not report.ok and {d.cls for d in report.diffs} == {"derived"}
    assert any(d.seq == seq_of(real_run, "tick") for d in report.diffs)


@pytest.mark.parametrize("tamper", ["frame", "event"])
def test_production_tamper_diff_classes(real_run, tamper):
    if tamper == "frame":
        sha = next(e.payload["frame_sha"] for e in read_events(real_run / "events.jsonl") if e.type == "keyframe")
        path = real_run / "frames" / f"{sha}.jpg"
        data = bytearray(path.read_bytes())
        data[len(data) // 2] ^= 1
        path.write_bytes(data)
        expected = "manifest"
    else:
        edit_event(real_run, seq_of(real_run, "say"), text="An edited instruction.")
        expected = "chain"
    report = verify_run(real_run, write=False)
    assert not report.ok and report.diffs[0].cls == expected


def test_audit_rejects_explicit_and_legacy_synthetic_evidence(run_dir, tmp_path, synthetic_kit_run):
    assert any("provenance synthetic" in p for p in audit_run(run_dir))
    assert any("platform synthetic" in p for p in audit_run(run_dir))
    assert any("synthetic response id" in p for p in audit_run(run_dir))
    assert audit_run(synthetic_kit_run)  # committed fixtures must never be publishable evidence
    legacy = record(tmp_path / "legacy", provenance="meta")
    rel = rewrite_call_meta(legacy, response_id="resp_synthetic_legacy")
    assert any(rel in p and "synthetic response id" in p for p in audit_run(legacy))


def test_audit_rechecks_claimed_meta_model_identity(tmp_path):
    run_dir = record(tmp_path, provenance="meta")
    rewrite_call_meta(run_dir, base_host="standin.example")
    assert any("model/server provenance is not Meta" in p for p in audit_run(run_dir))


def test_retry_usage_is_backed_by_each_raw_attempt(tmp_path):
    from lightwork.core import Reducer

    prior = {"input_tokens": 900, "output_tokens": 4000, "input_tokens_details": {"cached_tokens": 300}}
    run_dir = record(tmp_path, factory=Reducer, prior_usage=prior)
    report = verify_run(run_dir)
    assert report.ok, report.diffs
    total = aggregate_usage([prior, USAGE])
    manifest = json.loads((run_dir / "manifest.json").read_bytes())
    assert manifest["totals"]["cost_usd"] == cost_usd(SPARK, total)
    # Hashes all still match after this edit: the raw-attempt check, not just the manifest, must catch it.
    rewrite_call_meta(run_dir, usage={"input_tokens": 1, "output_tokens": 1})
    report = verify_run(run_dir, write=False)
    assert any(d.cls == "schema" and "model_result.usage differs" in d.detail for d in report.diffs)
    assert any(d.cls == "schema" and "aggregate raw response usage" in d.detail for d in report.diffs)


def test_retry_evidence_cannot_be_omitted_from_a_rehashed_run(tmp_path):
    from lightwork.core import Reducer

    run_dir = record(tmp_path, factory=Reducer, prior_usage={"input_tokens": 900, "output_tokens": 4000})
    next(run_dir.glob("calls/*/attempts/0001.sse")).unlink()
    rewrite_call_meta(run_dir)  # keep the edited manifest internally consistent
    report = verify_run(run_dir, write=False)
    assert any(d.cls == "missing" and "raw evidence of billed attempt 1" in d.detail for d in report.diffs)


def test_usage_event_must_match_call_metadata(real_run):
    edit_event(real_run, seq_of(real_run, "model_result"), usage={"input_tokens": 1})
    report = verify_run(real_run, write=False)
    assert any(d.cls == "schema" and "model_result.usage differs from meta.usage" in d.detail for d in report.diffs)


def test_audit_checks_synthetic_id_in_raw_response_not_just_metadata(tmp_path):
    run_dir = record(tmp_path, provenance="meta")
    response = next(run_dir.glob("calls/*/response.sse"))
    response.write_bytes(responses_sse(canonical(VERDICT), model=SPARK, usage=USAGE))
    rewrite_call_meta(run_dir)
    assert any("synthetic response id" in p for p in audit_run(run_dir))


def test_http_status_event_must_match_call_metadata(real_run):
    edit_event(real_run, seq_of(real_run, "model_result"), http_status=429)
    report = verify_run(real_run, write=False)
    assert any(d.cls == "schema" and "model_result.http_status differs" in d.detail for d in report.diffs)
