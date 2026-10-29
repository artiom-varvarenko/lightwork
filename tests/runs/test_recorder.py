from __future__ import annotations

import json

import cv2
import numpy as np
import pytest
from pydantic import ValidationError

from lightwork.contracts.canonical import canonical, canonical_sha, sha256_hex
from lightwork.contracts.events import check_chain, read_events
from lightwork.contracts.layout import Layout
from lightwork.contracts.run import Manifest
from lightwork.runs import Recorder
from lightwork.runs.manifest import RUN_ID_RE, created_utc, hashed_files, size_class

pytestmark = pytest.mark.usefixtures("no_network")

RUN_ID = "20261030T120000Z_rec-kit"
STARTED = {
    "run_id": RUN_ID,
    "backend": "meta",
    "models": {"verdict": "muse-spark-1.3-contributor"},
    "code_commit": "0000000",
    "dirty": False,
    "platform": "test",
    "version": "0.1.0",
}


class Echo:
    """Reducer double: one ``say`` naming each input it is fed."""

    def feed(self, ev):
        return [("say", {"text": ev.type, "priority": "normal"})]


def lines(rec: Recorder) -> list[str]:
    return (rec.run_dir / "events.jsonl").read_text(encoding="utf-8").splitlines()


def test_run_dir_by_provenance_and_hash_chain(tmp_path):
    rec = Recorder(tmp_path, "rec-kit", STARTED, provenance="meta")
    assert rec.run_dir == tmp_path / "runs" / RUN_ID
    assert all((rec.run_dir / d).is_dir() for d in ("frames", "preview", "calls"))
    rec.append("motion", "gate", {"state": "moving", "active_regions": ["slot"]}, 1.0)
    rec.append("say", "reducer", {"text": "hi", "priority": "normal"}, 1.0)
    events = read_events(rec.run_dir / "events.jsonl")
    check_chain(RUN_ID, events)
    assert [e.seq for e in events] == [1, 2, 3] and lines(rec) == [e.line() for e in events]
    assert events[0].type == "run_started" and events[0].payload["pack"] == "rec-kit" and events[0].t == 0.0

    dev = Recorder(tmp_path, "rec-kit", STARTED, provenance="dev-standin")
    assert dev.run_dir == tmp_path / "runs-dev" / RUN_ID
    synthetic = Recorder(tmp_path, "rec-kit", STARTED, provenance="synthetic")
    assert synthetic.run_dir == tmp_path / "runs-synthetic" / RUN_ID
    for recorder in (rec, dev, synthetic):
        recorder.close("test end")


def test_run_id_generated_validated_and_unique(tmp_path):
    started = {k: v for k, v in STARTED.items() if k != "run_id"}
    rec = Recorder(tmp_path, "rec-kit", started, provenance="meta")
    m = RUN_ID_RE.match(rec.run_id)
    assert m and m.group(7) == "rec-kit"
    with pytest.raises(ValueError):
        Recorder(tmp_path, "other-pack", STARTED, provenance="meta")
    fixed = Recorder(tmp_path, "rec-kit", STARTED, provenance="meta")
    with pytest.raises(FileExistsError):
        Recorder(tmp_path, "rec-kit", STARTED, provenance="meta")
    assert created_utc(RUN_ID) == "2026-10-30T12:00:00Z"
    with pytest.raises(ValueError):
        created_utc("synthetic-kit")
    rec.close("test end")
    fixed.close("test end")


def test_append_validates_and_keeps_time_monotonic(tmp_path):
    rec = Recorder(tmp_path, "rec-kit", STARTED, provenance="meta")
    with pytest.raises(ValueError):
        rec.append("say", "camera", {"text": "x", "priority": "normal"}, 1.0)  # derived from a non-reducer source
    with pytest.raises(ValueError):
        rec.append("tick", "reducer", {}, 1.0)  # input claimed by the reducer
    with pytest.raises(ValidationError):
        rec.append("motion", "gate", {"state": "flying", "active_regions": []}, 1.0)
    assert len(lines(rec)) == 1
    assert rec.append("tick", "orchestrator", {}, 5.12345).t == 5.123
    assert rec.append("tick", "orchestrator", {}, 3.0).t == 5.123  # late arrival takes the previous t
    rec.close("test end")


def test_attach_and_ingest_place_derived_events_after_their_input(tmp_path):
    rec = Recorder(tmp_path, "rec-kit", STARTED, provenance="meta")
    with pytest.raises(RuntimeError):
        rec.ingest("tick", "orchestrator", {}, 1.0)
    derived = rec.attach(Echo())
    assert [(e.seq, e.type, e.src, e.t, e.payload["text"]) for e in derived] == [(2, "say", "reducer", 0.0, "run_started")]
    out = rec.ingest("tick", "orchestrator", {}, 1.5)
    assert [(e.seq, e.type, e.t) for e in out] == [(3, "tick", 1.5), (4, "say", 1.5)]
    with pytest.raises(RuntimeError):
        rec.attach(Echo())
    with pytest.raises(ValueError):
        rec.ingest("say", "reducer", {"text": "x", "priority": "normal"}, 2.0)
    late = Recorder(tmp_path / "late", "rec-kit", STARTED, provenance="meta")
    late.append("tick", "orchestrator", {}, 0.5)
    with pytest.raises(RuntimeError):
        late.attach(Echo())  # run_started would no longer be followed by its derived events
    rec.close("test end")
    late.close("test end")


def test_files_frames_calls_and_previews(tmp_path):
    rec = Recorder(tmp_path, "rec-kit", STARTED, provenance="meta")
    jpeg = cv2.imencode(".jpg", np.full((40, 60, 3), 200, np.uint8))[1].tobytes()
    sha = rec.put_frame(jpeg)
    assert sha == sha256_hex(jpeg) and rec.put_frame(jpeg) == sha
    assert (rec.run_dir / "frames" / f"{sha}.jpg").read_bytes() == jpeg

    layout = Layout.model_validate(
        {"pack": "rec-kit", "created": "2026-10-30T12:00:00Z", "work_area_xywh": [0, 0, 640, 360], "objects": {}}
    )
    assert rec.put_run_file("layout.json", layout) == canonical_sha(layout)
    assert (rec.run_dir / "layout.json").read_text(encoding="utf-8") == canonical(layout) + "\n"
    with pytest.raises(TypeError):
        rec.put_run_file("stepgraph.json", layout)

    assert rec.alloc_call("verdict", "muse-spark-1.3-contributor") == (1, "calls/0001-verdict-muse-spark-1.3-contributor")
    assert rec.alloc_call("segment", "sam-3.1") == (2, "calls/0002-segment-sam-3.1")

    crop = np.full((720, 1000, 3), 120, np.uint8)
    assert rec.put_preview(crop, 1.0) == "preview/1.jpg"
    stored = cv2.imdecode(np.frombuffer((rec.run_dir / "preview" / "1.jpg").read_bytes(), np.uint8), cv2.IMREAD_COLOR)
    assert stored.shape[:2] == (480, 667)
    assert rec.put_preview(crop, 3.0) is None  # same seq
    rec.append("tick", "orchestrator", {}, 1.2)
    assert rec.put_preview(crop, 1.5) is None  # under a second after the last preview
    assert rec.put_preview(crop[:100], 2.0) == "preview/2.jpg"
    rec.close("test end")


def test_close_writes_manifest_once(tmp_path):
    rec = Recorder(tmp_path, "rec-kit", STARTED, provenance="meta", clock=iter([100.0, 109.25]).__next__)
    rec.attach(Echo())
    rec.ingest("tick", "orchestrator", {}, 4.0)
    run_dir = rec.close("user stop")
    events = read_events(run_dir / "events.jsonl")
    assert [(e.type, e.t) for e in events[-2:]] == [("run_stopped", 9.25), ("say", 9.25)]
    assert events[-2].payload == {"reason": "user stop"}
    m = Manifest.model_validate_json((run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert m == rec.manifest and m.provenance == "meta" and m.created_utc == "2026-10-30T12:00:00Z"
    assert m.files == hashed_files(run_dir) and "events.jsonl" in m.files
    assert m.size_bytes == sum((run_dir / f).stat().st_size for f in m.files)
    assert (m.totals.calls, m.totals.duration_s) == (0, 9.25)
    assert rec.close("again") == run_dir and len(read_events(run_dir / "events.jsonl")) == len(events)
    with pytest.raises(RuntimeError):
        rec.append("tick", "orchestrator", {}, 10.0)


def test_close_moves_a_run_with_a_non_meta_call_to_runs_dev(tmp_path):
    rec = Recorder(tmp_path, "rec-kit", STARTED, provenance="meta")
    _, call_dir = rec.alloc_call("verdict", "cx/gpt-5.6-luna-low")
    d = rec.run_dir / call_dir
    d.mkdir()
    (d / "meta.json").write_text(json.dumps({"provenance": "dev-standin"}), encoding="utf-8")
    run_dir = rec.close("end")
    assert run_dir == tmp_path / "runs-dev" / RUN_ID and not (tmp_path / "runs" / RUN_ID).exists()
    assert Manifest.model_validate_json((run_dir / "manifest.json").read_text()).provenance == "dev-standin"


def test_size_classes():
    assert size_class(10 * 2**20) == "ok"
    assert size_class(10 * 2**20 + 1) == "over-target"
    assert size_class(25 * 2**20 + 1) == "over-limit"


def test_closing_with_reducer_error_leaves_an_incomplete_run(tmp_path):
    class BadStop(Echo):
        def feed(self, ev):
            if ev.type == "run_stopped":
                raise ValueError("reducer failed at stop")
            return super().feed(ev)

    rec = Recorder(tmp_path, "rec-kit", STARTED, provenance="synthetic")
    rec.attach(BadStop())
    with pytest.raises(ValueError, match="reducer failed at stop"):
        rec.close("test end")
    assert not (rec.run_dir / "manifest.json").exists()


def test_sealed_run_cannot_be_modified_through_recorder(tmp_path):
    rec = Recorder(tmp_path, "rec-kit", STARTED, provenance="synthetic")
    layout = Layout(pack="rec-kit", created="2026-10-30T12:00:00Z", work_area_xywh=[0, 0, 10, 10], objects={})
    rec.put_run_file("layout.json", layout)
    rec.close("test end")
    mutations = [
        lambda: rec.put_run_file("layout.json", layout),
        lambda: rec.put_frame(b"late bytes"),
        lambda: rec.put_preview(np.zeros((10, 10, 3), np.uint8), 1.0),
        lambda: rec.alloc_call("verdict", "muse-spark-1.3-contributor"),
    ]
    for mutate in mutations:
        with pytest.raises(RuntimeError, match="closed"):
            mutate()


def test_generated_call_quarantines_previously_meta_run(tmp_path):
    rec = Recorder(tmp_path, "rec-kit", STARTED, provenance="meta")
    _, rel = rec.alloc_call("verdict", "muse-spark-1.3-contributor")
    call = rec.run_dir / rel
    call.mkdir()
    (call / "meta.json").write_text(json.dumps({"provenance": "synthetic"}), encoding="utf-8")
    assert rec.close("test end") == tmp_path / "runs-synthetic" / RUN_ID
    assert rec.manifest.provenance == "synthetic"


def test_store_reads_bytes_and_rejects_escaping_paths(tmp_path):
    from lightwork.runs import DirStore

    run = tmp_path / "run"
    frames = run / "frames"
    call = run / "calls" / "0001-verdict-muse-spark-1.3"
    frames.mkdir(parents=True)
    call.mkdir(parents=True)
    jpeg = b"opaque recorded frame bytes"
    sha = sha256_hex(jpeg)
    (frames / f"{sha}.jpg").write_bytes(jpeg)
    (call / "response.sse").write_bytes(b"recorded response")
    store = DirStore(run)
    assert store.frame_bytes(sha) == jpeg
    assert store.call_response("calls/0001-verdict-muse-spark-1.3") == (b"recorded response", "sse")
    with pytest.raises(ValueError):
        store.frame_bytes("../../private")
    with pytest.raises(ValueError):
        store.call_response("../private")
    outside = tmp_path / "private"
    outside.write_bytes(b"must not be read")
    (frames / f"{'0' * 64}.jpg").symlink_to(outside)
    with pytest.raises(ValueError, match="escapes"):
        store.frame_bytes("0" * 64)
    (call / "response.sse").unlink()
    (call / "response.sse").symlink_to(outside)
    with pytest.raises(ValueError, match="escapes"):
        store.call_response("calls/0001-verdict-muse-spark-1.3")
