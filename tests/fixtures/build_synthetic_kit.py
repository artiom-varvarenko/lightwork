"""Build the explicitly synthetic C6 run and contract examples through Recorder and the production Reducer.

Run ``uv run python tests/fixtures/build_synthetic_kit.py``. Generated responses are test data, never Meta evidence:
all C5/C6 provenance fields say ``synthetic``, and release audit must reject the run even when verify has 0 diffs.
"""

from __future__ import annotations

import json
import shutil
from collections import deque
from pathlib import Path
from tempfile import TemporaryDirectory

import cv2
import numpy as np

from lightwork.contracts.calls import CallMeta, request_fingerprint
from lightwork.contracts.canonical import canonical
from lightwork.contracts.layout import Calibration, Layout
from lightwork.contracts.stepgraph import StepGraph
from lightwork.core.reducer import Reducer
from lightwork.models.pricing import cost_usd
from lightwork.models.record import write_call
from lightwork.models.requests import META_MODELS
from lightwork.models.synthetic import SamObject, responses_sse, sam_sse
from lightwork.models.wire import read_body
from lightwork.runs import Recorder, verify_run
from lightwork.testing.synth import Item, add_hand, encode_jpeg, kit_scene

HERE = Path(__file__).parent
RUN_ID = "20261030T120000Z_synthetic-kit"
OUT = HERE / "runs" / "synthetic-kit"
CONTRACTS = HERE / "contracts"
SPARK = "muse-spark-1.3-contributor"
CROP = [600, 80, 680, 560]  # work area x, y, w, h in camera px (1280x720 synthetic camera)

STEPGRAPH = {
    "schema_version": 1,
    "id": "synthetic-kit",
    "title": "Synthetic kit return check",
    "source": {"kind": "text", "ref": "tests/fixtures/build_synthetic_kit.py", "licence": "Apache-2.0", "attribution": "Lightwork authors"},
    "objects": [
        {"id": "case", "noun": "case", "kind": "container"},
        {"id": "remote", "noun": "remote control", "kind": "item"},
        {"id": "cable", "noun": "power cable", "kind": "item"},
        {"id": "slot-remote", "noun": "slot", "kind": "zone"},
        {"id": "slot-cable", "noun": "slot", "kind": "zone"},
    ],
    "steps": [
        {
            "id": "remote-in",
            "say": "Put the remote in its slot.",
            "detail": "The remote goes in the small slot on the right of the case.",
            "targets": ["slot-remote"],
            "arrow": {"from": "remote", "to": "slot-remote"},
            "checks": [
                {"id": "remote-present", "kind": "present", "object": "remote", "region": "slot-remote"},
                {"id": "hand-at-slot", "kind": "activity_at", "region": "slot-remote"},
                {"id": "remote-judge", "kind": "judge", "question": "Is the remote lying inside the right-hand slot of the case?"},
            ],
            "recovery": [{"on": "remote-present", "action": "redo_step", "say": "The remote is not in its slot yet."}],
        },
        {
            "id": "cable-in",
            "say": "Coil the power cable into the left slot.",
            "targets": ["slot-cable"],
            "checks": [
                {"id": "cable-present", "kind": "present", "object": "cable", "region": "slot-cable"},
                {"id": "done", "kind": "confirm", "required": False},
            ],
        },
    ],
    "vocabulary": ["remote", "cable", "case"],
}

SLOTS = {"slot-remote": (1000, 160, 180, 140), "slot-cable": (740, 160, 220, 380)}
LAYOUT = {
    "pack": "synthetic-kit",
    "created": "2026-10-30T12:00:00Z",
    "work_area_xywh": CROP,
    "objects": {
        "case": {"kind": "container", "corners_cam_px": [[700, 120], [1220, 120], [1220, 600], [700, 600]], "sam": {"mode": "noun", "noun": "case"}},
        "slot-remote": {"kind": "zone", "corners_cam_px": [[1000, 160], [1180, 160], [1180, 300], [1000, 300]], "sam": {"mode": "noun", "noun": "slot"}},
        "slot-cable": {"kind": "zone", "corners_cam_px": [[740, 160], [960, 160], [960, 540], [740, 540]], "sam": {"mode": "noun", "noun": "slot"}},
        "remote": {"kind": "item", "polygon_cam_px": [[300, 400], [340, 400], [340, 520], [300, 520]], "sam": {"mode": "grounded", "noun": "remote control"}},
        "cable": {"kind": "item", "polygon_cam_px": [[200, 200], [320, 200], [320, 300], [200, 300]], "sam": {"mode": "noun", "noun": "power cable"}},
    },
}

CALIB = {
    "H_cam_to_page": [[0.45, 0.0, -12.5], [0.0, 0.45, -8.0], [0.0, 0.0, 1.0]],
    "page": {"w": 854.0, "h": 480.0, "dpr": 1.0},
    "camera": {"device_id": "synthetic", "mode": "MJPG 1280x720@30", "controls": {"exposure_time_absolute": 156, "white_balance_temperature": 4600, "focus_absolute": 0}},
    "residuals": {"rms_px": 0.6, "p95_px": 1.1, "max_px": 1.4, "n_markers": 15, "n_inliers": 60},
    "mm": {"source": "nominal", "H_cam_to_mm": None},
    "flat_field_sha": None,
    "measurement_mode": "grey",
    "created": "2026-10-30T11:55:00Z",
}


def _crop(img):
    x, y, w, h = CROP
    return img[y : y + h, x : x + w]


def frames():
    remote_out = Item("remote", "rect", (40, 40, 40), (320, 460), (40, 120))
    remote_in = Item("remote", "rect", (40, 40, 40), (1090, 230), (120, 40))
    cable = Item("cable", "circle", (30, 30, 160), (260, 250), (100, 100))
    before = kit_scene([remote_out, cable], slots=SLOTS)
    during = add_hand(kit_scene([remote_in, cable], slots=SLOTS), (1090, 230), angle_deg=60)
    after = kit_scene([remote_in, cable], slots=SLOTS)
    return {"before": encode_jpeg(_crop(before)), "during": encode_jpeg(_crop(during), 70), "after": encode_jpeg(_crop(after))}


def build_run(root: Path) -> tuple[Path, dict]:
    """Record the first kit step, including Spark grounding, SAM and a verdict, entirely offline."""
    sg = StepGraph.model_validate(STEPGRAPH)
    layout, calib = Layout.model_validate(LAYOUT), Calibration.model_validate(CALIB)
    started = {
        "run_id": RUN_ID, "backend": "meta", "models": dict(META_MODELS), "code_commit": "0000000",
        "dirty": False, "platform": "synthetic", "version": "0.1.0",
    }
    now = [0.0]
    rec = Recorder(root, sg.id, started, provenance="synthetic", clock=lambda: now[0])
    stepgraph_sha = rec.put_run_file("stepgraph.json", sg)
    layout_sha = rec.put_run_file("layout.json", layout)
    calib_sha = rec.put_run_file("calib.json", calib)
    reducer = Reducer(sg, layout, calib, rec.store, backend="meta", models=started["models"])
    rec.attach(reducer)
    rec.ingest("procedure_loaded", "orchestrator", {"stepgraph_sha": stepgraph_sha}, 0.0)
    rec.ingest("calibration_loaded", "orchestrator", {"calib_sha": calib_sha}, 0.0)
    rec.ingest("layout_loaded", "orchestrator", {"layout_sha": layout_sha}, 0.0)
    initial_scene = reducer.scene.model_dump(mode="json")
    images = frames()
    shas = {name: rec.put_frame(data) for name, data in images.items()}

    def preview(name: str, t: float) -> None:
        rec.put_preview(cv2.imdecode(np.frombuffer(images[name], np.uint8), cv2.IMREAD_COLOR), t)

    rec.ingest("keyframe", "camera", {
        "frame_sha": shas["before"], "crop_xywh": CROP, "mode": "grey", "verified": True, "t_capture": 0.48,
    }, 0.5)
    preview("before", 0.5)
    rec.ingest("motion", "gate", {"state": "moving", "active_regions": ["slot-remote"]}, 2.1)
    rec.ingest("hands", "hands", {"over_regions": ["slot-remote"], "tip_xy": [1090.0, 230.0]}, 2.2)
    preview("during", 2.2)
    rec.ingest("hands", "hands", {"over_regions": [], "tip_xy": None}, 3.4)
    rec.ingest("motion", "gate", {"state": "settled", "active_regions": []}, 4.1)
    out = rec.ingest("keyframe", "camera", {
        "frame_sha": shas["after"], "crop_xywh": CROP, "mode": "grey", "verified": True, "t_capture": 4.28,
    }, 4.3)
    preview("after", 4.3)
    pending = deque(e.payload for e in out if e.type == "model_request")
    while pending:
        intent = pending.popleft()
        role, model = intent["role"], intent["model"]
        index, allocated = rec.alloc_call(role, model)
        endpoint, request = reducer.request_for(intent["intent_id"])
        response_id = f"resp_synthetic_{index:04d}"
        if role == "verdict":
            answer = {"done": "yes", "contradiction": None, "evidence": "A dark remote lies inside the right-hand slot."}
            raw = responses_sse(canonical(answer), model=model, response_id=response_id)
        elif role == "ground":
            # Camera box [1030,210,1150,251], normalised in the CROP sent to Spark (0-1000 grid).
            answer = {"items": [{"object_name": "remote control",
                                 "bbox": [{"x_min": 632, "y_min": 232, "x_max": 809, "y_max": 305}]}]}
            raw = responses_sse(canonical(answer), model=model, response_id=response_id)
        elif role == "segment":
            raw = sam_sse([SamObject(3, (430, 130, 550, 171), confidence=0.93)], (680, 560), response_id=response_id)
        else:
            raise AssertionError(f"unexpected fixture request: {role}")
        usage = read_body(raw, "sse", endpoint).usage
        meta = CallMeta(
            backend="meta", base_host="api.meta.ai", endpoint=endpoint, model=model, status="ok", http_status=200,
            attempts=[{"n": 1, "status": 200, "error": None, "wait_s": 0.0, "ms": 2412}], headers={},
            response_id=response_id, ttfb_ms=1880, total_ms=2412, usage=usage, cost_usd=cost_usd(model, usage),
            fingerprint=request_fingerprint(request), provenance="synthetic",
        )
        rel = write_call(rec.run_dir, index=index, role=role, request=request, raw=raw, kind="sse", meta=meta)
        assert rel == allocated
        out = rec.ingest("model_result", "model", {
            "intent_id": intent["intent_id"], "call_dir": rel, "status": "ok", "latency_ms": 2412, "usage": usage,
            "http_status": 200,
        }, 6.712)
        pending.extend(e.payload for e in out if e.type == "model_request")
    now[0] = 9.0
    return rec.close("synthetic fixture end"), initial_scene


def build() -> None:
    with TemporaryDirectory(prefix="lightwork-fixture-") as temp:
        run_dir, initial_scene = build_run(Path(temp))
        report = verify_run(run_dir)
        assert report.ok, report.model_dump()
        if OUT.exists():
            shutil.rmtree(OUT)
        shutil.copytree(run_dir, OUT)
    CONTRACTS.mkdir(parents=True, exist_ok=True)
    meta = json.loads(next(OUT.glob("calls/*-verdict-*/meta.json")).read_bytes())
    manifest = json.loads((OUT / "manifest.json").read_bytes())
    examples = {
        "C3-stepgraph": STEPGRAPH,
        "C4-layout": LAYOUT,
        "C5-call-meta": meta,
        "C6-manifest": manifest,
        "C7-calibration": CALIB,
        "C9-verdict": {"done": "yes", "contradiction": None, "evidence": "A dark remote lies inside the right-hand slot."},
        "C9-mask": {"object_id": 3, "box": [1030, 210, 1150, 251], "area_px": 4920, "centroid": [1090.0, 230.0], "confidence": 0.93, "rle": {"size": [41, 120], "counts": "0_1"}},
        "C6-verify": report.model_dump(mode="json"),
        "C2-server-to-page": initial_scene,
        "C2-page-to-server": {"type": "hello", "role": "projector", "innerWidth": 854, "innerHeight": 480, "dpr": 1.0, "screen": {"width": 854, "height": 480}, "orientation": "landscape-primary", "fullscreen": True, "secure": True, "wakelock": True, "ua": "synthetic"},
    }
    for name, obj in examples.items():
        (CONTRACTS / f"{name}.json").write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {OUT} ({manifest['size_bytes']} bytes, {report.n_events} events, provenance synthetic)")


if __name__ == "__main__":
    build()
