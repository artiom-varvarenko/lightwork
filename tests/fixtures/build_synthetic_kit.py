"""Build ``tests/fixtures/runs/synthetic-kit/`` (C6) and ``tests/fixtures/contracts/*.json`` deterministically.

Run: ``uv run python tests/fixtures/build_synthetic_kit.py``. The output is committed; tests check its hash chain,
schemas and manifest. Derived events here are hand-written to the contract; once the reducer (H) and recorder (I)
exist, this builder should produce the run through them instead, so `verify` can re-derive it.

Call files follow the Responses SSE shape documented in ``context/research/meta-api-ref.md`` §3.5.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from lightwork.contracts.calls import call_dirname, frame_ref, request_fingerprint
from lightwork.contracts.canonical import canonical, canonical_sha, file_sha, q_px, q_s, sha256_hex
from lightwork.contracts.events import genesis_hash, seal
from lightwork.contracts.layout import Calibration, Layout
from lightwork.contracts.render import Ring, SceneMsg
from lightwork.contracts.run import UNHASHED_FILES, Manifest
from lightwork.contracts.stepgraph import StepGraph, validate_semantics
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


def verdict_request(frame_sha: str) -> dict:
    schema = {
        "type": "object",
        "properties": {
            "done": {"type": "string", "enum": ["yes", "no", "unsure"]},
            "contradiction": {"type": ["string", "null"]},
            "evidence": {"type": "string"},
        },
        "required": ["done", "contradiction", "evidence"],
        "additionalProperties": False,
    }
    return {
        "model": SPARK,
        "input": [
            {
                "type": "message",
                "role": "user",
                "content": [
                    {"type": "input_text", "text": "Step: Put the remote in its slot.\nQuestion: Is the remote lying inside the right-hand slot of the case?"},
                    {"type": "input_image", "image_url": frame_ref(frame_sha)},
                ],
            }
        ],
        "reasoning": {"effort": "minimal"},
        "store": False,
        "stream": True,
        "max_output_tokens": 4000,
        "text": {"format": {"type": "json_schema", "name": "StepVerdict", "strict": True, "schema": schema}},
    }


def verdict_sse() -> bytes:
    text = '{"done":"yes","contradiction":null,"evidence":"A dark remote lies inside the right-hand slot."}'
    resp_id = "resp_synthetic_0001"
    events = [
        {"type": "response.created", "response": {"id": resp_id, "status": "in_progress", "model": SPARK}},
        {"type": "response.in_progress", "response": {"id": resp_id, "status": "in_progress"}},
        {"type": "response.output_item.added", "output_index": 0, "item": {"type": "message", "id": "msg_1", "role": "assistant", "content": []}},
        {"type": "response.content_part.added", "output_index": 0, "content_index": 0, "part": {"type": "output_text", "text": ""}},
    ]
    for i in range(0, len(text), 24):
        events.append({"type": "response.output_text.delta", "output_index": 0, "content_index": 0, "delta": text[i : i + 24]})
    events += [
        {"type": "response.output_text.done", "output_index": 0, "content_index": 0, "text": text},
        {
            "type": "response.completed",
            "response": {
                "id": resp_id,
                "status": "completed",
                "model": SPARK,
                "output": [{"type": "message", "id": "msg_1", "role": "assistant", "content": [{"type": "output_text", "text": text}]}],
                "usage": {"input_tokens": 1412, "output_tokens": 31, "total_tokens": 1443},
            },
        },
    ]
    out = "".join(f"event: {e['type']}\ndata: {json.dumps(e, separators=(',', ':'))}\n\n" for e in events) + "data: [DONE]\n\n"
    return out.encode()


def build() -> None:
    if OUT.exists():
        shutil.rmtree(OUT)
    (OUT / "frames").mkdir(parents=True)
    (OUT / "preview").mkdir()
    CONTRACTS.mkdir(parents=True, exist_ok=True)

    sg = StepGraph.model_validate(STEPGRAPH)
    errs = validate_semantics(sg)
    assert not errs, errs
    layout = Layout.model_validate(LAYOUT)
    calib = Calibration.model_validate(CALIB)
    for name, obj in (("stepgraph.json", sg), ("layout.json", layout), ("calib.json", calib)):
        (OUT / name).write_text(canonical(obj) + "\n", encoding="utf-8")

    shas = {}
    for name, data in frames().items():
        sha = sha256_hex(data)
        shas[name] = sha
        (OUT / "frames" / f"{sha}.jpg").write_bytes(data)

    req = verdict_request(shas["after"])
    fp = request_fingerprint(req)
    call_dir = OUT / "calls" / call_dirname(1, "verdict", SPARK)
    call_dir.mkdir(parents=True)
    (call_dir / "request.json").write_text(canonical(req) + "\n", encoding="utf-8")
    (call_dir / "response.sse").write_bytes(verdict_sse())
    meta = {
        "backend": "meta",
        "base_host": "api.meta.ai",
        "endpoint": "/v1/responses",
        "model": SPARK,
        "status": "ok",
        "http_status": 200,
        "attempts": [{"n": 1, "status": 200, "error": None, "wait_s": 0.0, "ms": 2412}],
        "headers": {"x-ratelimit-remaining-requests": "149"},
        "response_id": "resp_synthetic_0001",
        "ttfb_ms": 1880,
        "total_ms": 2412,
        "usage": {"input_tokens": 1412, "output_tokens": 31, "total_tokens": 1443},
        "cost_usd": 0.0,
        "fingerprint": fp,
        "server_props": None,
        "provenance": "meta",
    }
    (call_dir / "meta.json").write_text(canonical(meta) + "\n", encoding="utf-8")

    scene0 = SceneMsg(seq=1, mode="normal", prims=[Ring(id="t-slot-remote", c=[q_px(1090), q_px(230)], r=90.0, role="target")])
    scene1 = SceneMsg(seq=2, mode="normal", prims=[Ring(id="t-slot-cable", c=[q_px(850), q_px(350)], r=120.0, role="target")])
    intent = "remote-in/remote-judge/1"
    rel_call = call_dir.relative_to(OUT).as_posix()
    seq_events = [
        (0.0, "run_started", "orchestrator", {"run_id": RUN_ID, "pack": "synthetic-kit", "backend": "meta", "models": {"verdict": SPARK, "segment": "sam-3.1"}, "code_commit": "0000000", "dirty": False, "platform": "synthetic", "version": "0.1.0"}),
        (0.0, "procedure_loaded", "orchestrator", {"stepgraph_sha": canonical_sha(sg)}),
        (0.0, "calibration_loaded", "orchestrator", {"calib_sha": canonical_sha(calib)}),
        (0.0, "layout_loaded", "orchestrator", {"layout_sha": canonical_sha(layout)}),
        (0.001, "step", "reducer", {"step": "remote-in", "state": "active", "reason": ""}),
        (0.001, "scene", "reducer", {"hash": canonical_sha(scene0)}),
        (0.001, "say", "reducer", {"text": "Put the remote in its slot.", "priority": "normal"}),
        (0.5, "keyframe", "camera", {"frame_sha": shas["before"], "crop_xywh": CROP, "mode": "grey", "verified": True, "t_capture": 0.48}),
        (2.1, "motion", "gate", {"state": "moving", "active_regions": ["slot-remote"]}),
        (2.2, "hands", "hands", {"over_regions": ["slot-remote"], "tip_xy": [1090.0, 230.0]}),
        (3.4, "hands", "hands", {"over_regions": [], "tip_xy": None}),
        (4.1, "motion", "gate", {"state": "settled", "active_regions": []}),
        (4.3, "keyframe", "camera", {"frame_sha": shas["after"], "crop_xywh": CROP, "mode": "grey", "verified": True, "t_capture": 4.28}),
        (4.3, "check", "reducer", {"step": "remote-in", "check": "hand-at-slot", "status": "pass", "reason": "fingertip reached slot-remote", "evidence": ["hands:10"]}),
        (4.3, "step", "reducer", {"step": "remote-in", "state": "provisional", "reason": "local checks passed"}),
        (4.3, "model_request", "reducer", {"intent_id": intent, "role": "verdict", "model": SPARK, "fingerprint": fp, "frame_sha": shas["after"], "prompt_version": "verdict-v1"}),
        (4.3, "step", "reducer", {"step": "cable-in", "state": "active", "reason": ""}),
        (4.3, "scene", "reducer", {"hash": canonical_sha(scene1)}),
        (4.3, "say", "reducer", {"text": "Coil the power cable into the left slot.", "priority": "normal"}),
        (6.712, "model_result", "model", {"intent_id": intent, "call_dir": rel_call, "status": "ok", "latency_ms": 2412}),
        (6.712, "parsed", "reducer", {"intent_id": intent, "kind": "verdict", "result": {"done": "yes", "contradiction": None, "evidence": "A dark remote lies inside the right-hand slot."}}),
        (6.712, "check", "reducer", {"step": "remote-in", "check": "remote-judge", "status": "pass", "reason": "verdict yes", "evidence": [f"intent:{intent}", f"frame:{shas['after']}"]}),
        (6.712, "cost", "reducer", {"usd_total": 0.0019}),
        (9.0, "run_stopped", "orchestrator", {"reason": "fixture end"}),
    ]
    prev = genesis_hash(RUN_ID)
    lines = []
    for i, (t, typ, src, payload) in enumerate(seq_events, 1):
        ev = seal(prev, i, q_s(t), typ, src, payload)
        prev = ev.h
        lines.append(ev.line())
    (OUT / "events.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")

    files = {p.relative_to(OUT).as_posix(): file_sha(p) for p in sorted(OUT.rglob("*")) if p.is_file() and p.name not in UNHASHED_FILES}
    size = sum((OUT / f).stat().st_size for f in files)
    manifest = Manifest(
        run_id=RUN_ID,
        created_utc="2026-10-30T12:00:00Z",
        code_commit="0000000",
        dirty=False,
        version="0.1.0",
        platform="synthetic",
        provenance="meta",
        backend="meta",
        models={"verdict": SPARK, "segment": "sam-3.1"},
        files=files,
        size_bytes=size,
        totals={"passed": 0, "failed": 0, "unverified": 0, "calls": 1, "cost_usd": 0.0019, "duration_s": 9.0},
    )
    (OUT / "manifest.json").write_text(canonical(manifest) + "\n", encoding="utf-8")

    # Contract example fixtures (one per contract).
    examples = {
        "C3-stepgraph": STEPGRAPH,
        "C4-layout": LAYOUT,
        "C5-call-meta": meta,
        "C6-manifest": json.loads(canonical(manifest)),
        "C7-calibration": CALIB,
        "C9-verdict": {"done": "yes", "contradiction": None, "evidence": "A dark remote lies inside the right-hand slot."},
        "C9-mask": {"object_id": 3, "box": [1030, 210, 1150, 251], "area_px": 4920, "centroid": [1090.0, 230.0], "confidence": 0.93, "rle": {"size": [41, 120], "counts": "0_1"}},
        "C6-verify": {"run_id": RUN_ID, "ok": True, "n_events": len(lines), "n_inputs": 13, "n_derived": 11, "n_calls": 1, "diffs": [], "code_version": "0.1.0"},
        "C2-server-to-page": json.loads(canonical(scene0)),
        "C2-page-to-server": {"type": "hello", "role": "projector", "innerWidth": 854, "innerHeight": 480, "dpr": 1.0, "screen": {"width": 854, "height": 480}, "orientation": "landscape-primary", "fullscreen": True, "secure": True, "wakelock": True, "ua": "synthetic"},
    }
    for name, obj in examples.items():
        (CONTRACTS / f"{name}.json").write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {OUT} ({size} bytes, {len(lines)} events)")


if __name__ == "__main__":
    build()
