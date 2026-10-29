"""Consumer boundaries on explicitly synthetic, temporary recordings; never real inference evidence."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

from lightwork.contracts.canonical import canonical
from lightwork.contracts.events import read_events
from lightwork.core.scenarios import build
from lightwork.eval.dataset import audit_dataset, build_dataset
from lightwork.runs.manifest import build_manifest

pytestmark = pytest.mark.usefixtures("no_network")


def _replace(value, mapping):
    if isinstance(value, dict):
        return {k: _replace(v, mapping) for k, v in value.items()}
    if isinstance(value, list):
        return [_replace(v, mapping) for v in value]
    return mapping.get(value, value) if isinstance(value, str) else value


def make_runs(root):
    # Reuse the real recorder/C5/reducer fixture harness, not a mocked verifier or a fabricated production run.
    spec = importlib.util.spec_from_file_location("record_scenarios", Path(__file__).parents[1] / "runs/test_scenarios.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for index in range(9):
        scenario = build("kit-clean")
        scenario.run_started["run_id"] = f"20261029T1200{index:02d}Z_{scenario.pack}"
        mapping, frames = {}, {}
        for old, image in scenario.frames.items():
            # JPEG trailing test marker preserves decoded pixels while making fixture variants content-addressable.
            data = image + f"\nSYNTHETIC TEST VARIANT {index}".encode()
            new = hashlib.sha256(data).hexdigest()
            mapping[old], frames[new] = new, data
        scenario.frames = frames
        scenario.timeline = [(t, kind, src, _replace(payload, mapping)) for t, kind, src, payload in scenario.timeline]
        original = scenario.respond
        reverse = {new: old for old, new in mapping.items()}
        scenario.respond = lambda iid, role, model, endpoint, request, original=original, reverse=reverse: original(
            iid, role, model, endpoint, _replace(request, reverse))
        run = module.record_scenario(root, scenario)
        events = read_events(run / "events.jsonl")
        requests = {e.payload["intent_id"]: e.payload for e in events if e.type == "model_request" and e.payload["role"] == "verdict"}
        labels = []
        for iid, request in requests.items():
            sid, check, *_ = iid.split("/")
            step = next(s for s in scenario.stepgraph.steps if s.id == sid)
            question = next(c.question for c in step.checks if c.id == check)
            labels.append({"session_id": "holdout" if index == 8 else "train", "run_id": scenario.run_started["run_id"],
                           "frame_sha": request["frame_sha"], "trial_id": f"{index}-{iid}", "step_id": sid,
                           "question": question, "expected": {"done": "yes", "contradiction": None,
                           "evidence": "Operator-authored synthetic test label, not model ground truth."},
                           "source": "trial-script", "reviewed": True})
        if index == 0:
            labels[0]["reviewed"] = False
            labels[1]["expected"]["done"] = "no"
        # Tests alone deliberately amend synthetic fixtures. Production collection writes labels BEFORE close.
        (run / "labels.jsonl").write_text("".join(canonical(l) + "\n" for l in labels))
        (run / "manifest.json").write_text(canonical(build_manifest(run, events, "synthetic")) + "\n")
    return root / "runs-synthetic"


@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    return make_runs(tmp_path_factory.mktemp("dataset-synthetic"))


def build_test(runs, output, **kwargs):
    return build_dataset(runs, output, holdout_sessions=["holdout"], mode="smoke", licence="Apache-2.0",
                         attribution="Synthetic unit tests; not real training data", allow_synthetic_for_tests=True, **kwargs)


def test_reproducible_session_split_and_review_filter(runs, tmp_path):
    a = build_test(runs, tmp_path / "a", seed=7)
    b = build_test(runs, tmp_path / "b", seed=7)
    assert a == b
    assert a["counts"] == {"train": 16, "holdout": 3}
    assert a["sessions"] == {"train": ["train"], "holdout": ["holdout"]}
    assert a["dropped"] == {"disagreement_or_missing_evidence": 1, "unreviewed": 1}
    assert audit_dataset(tmp_path / "a", release=False) == a
    with pytest.raises(ValueError, match="test-only"):
        audit_dataset(tmp_path / "a")
    assert (tmp_path / "a/manifest.json").read_bytes() == (tmp_path / "b/manifest.json").read_bytes()
    rows = [json.loads(l) for l in (tmp_path / "a/train.jsonl").read_text().splitlines()]
    assert all(set(json.loads(r["messages"][-1]["content"])) == {"done", "contradiction", "evidence"} for r in rows)


def test_production_rejects_synthetic_and_full_rejects_small_data(runs, tmp_path):
    with pytest.raises(ValueError, match="provenance"):
        build_dataset(runs, tmp_path / "real", holdout_sessions=["holdout"], mode="smoke", licence="Apache-2.0", attribution="test")
    with pytest.raises(ValueError, match=">=500 train and >=60 holdout"):
        build_dataset(runs, tmp_path / "small", holdout_sessions=["holdout"], licence="Apache-2.0", attribution="test",
                      allow_synthetic_for_tests=True)
    assert not (tmp_path / "small").exists()


def _copy_one(runs, tmp_path):
    source = sorted(runs.iterdir())[1]
    return Path(shutil.copytree(source, tmp_path / source.name))


def _change_labels(run, fn):
    labels = [json.loads(l) for l in (run / "labels.jsonl").read_text().splitlines()]
    fn(labels)
    (run / "labels.jsonl").write_text("".join(canonical(l) + "\n" for l in labels))
    (run / "manifest.json").write_text(canonical(build_manifest(run, read_events(run / "events.jsonl"), "synthetic")) + "\n")


@pytest.mark.parametrize("kind", ["trial", "example", "session"])
def test_duplicate_and_session_leakage_rejected(runs, tmp_path, kind):
    run = _copy_one(runs, tmp_path)
    def mutate(labels):
        duplicate = copy.deepcopy(labels[0])
        if kind != "trial":
            duplicate["trial_id"] += "-duplicate"
        if kind == "session":
            duplicate["session_id"] = "holdout"
        labels.append(duplicate)
    _change_labels(run, mutate)
    with pytest.raises(ValueError, match="duplicate"):
        build_test(run, tmp_path / "out")


def test_source_manifest_tampering_rejected(runs, tmp_path):
    run = _copy_one(runs, tmp_path)
    with (run / "labels.jsonl").open("a") as stream:
        stream.write("{}\n")
    with pytest.raises(ValueError, match="verification failed"):
        build_test(run, tmp_path / "out")


def test_dataset_tampering_rejected(runs, tmp_path):
    build_test(runs, tmp_path / "out")
    image = next((tmp_path / "out/images").iterdir())
    image.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="hash mismatch"):
        audit_dataset(tmp_path / "out", release=False)


def test_unknown_holdout_rejected(runs, tmp_path):
    with pytest.raises(ValueError, match="holdout ids"):
        build_dataset(runs, tmp_path / "out", holdout_sessions=["missing"], mode="smoke", licence="Apache-2.0",
                      attribution="test", allow_synthetic_for_tests=True)


def test_actual_cli_synthetic_smoke_and_release_refusal(runs, tmp_path):
    cmd = [sys.executable, "-m", "lightwork.cli", "dataset", "build", str(runs), "--output", str(tmp_path / "out"),
           "--holdout-session", "holdout", "--mode", "smoke", "--licence", "Apache-2.0", "--attribution", "test-only",
           "--allow-synthetic-for-tests"]
    result = subprocess.run(cmd, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["counts"]["train"] == 16
    result = subprocess.run([sys.executable, "-m", "lightwork.cli", "dataset", "audit", str(tmp_path / "out")], capture_output=True, text=True)
    assert result.returncode == 1 and "test-only" in result.stderr


def test_forged_release_flags_still_authenticate_sources(runs, tmp_path):
    build_test(runs, tmp_path / "out")
    path = tmp_path / "out/manifest.json"
    manifest = json.loads(path.read_text())
    manifest["release_eligible"], manifest["provenance"] = True, "meta"
    for source in manifest["source_runs"].values():
        source["provenance"] = "meta"
    path.write_text(canonical(manifest) + "\n")
    with pytest.raises(ValueError, match="source run authentication"):
        audit_dataset(tmp_path / "out", source_root=runs)


def test_sam_disagreement_drops_positive_verdict(runs):
    from lightwork.contracts.evaluation import TrialLabel
    from lightwork.eval.dataset import _evidence
    run = sorted(runs.iterdir())[1]
    label = TrialLabel.model_validate_json((run / "labels.jsonl").read_text().splitlines()[0])
    events = read_events(run / "events.jsonl")
    graph = json.loads((run / "stepgraph.json").read_text())
    assert _evidence(run, label, events, graph)
    for event in events:
        if event.type == "check" and event.payload["step"] == label.step_id and event.payload["check"] == "present":
            event.payload["status"] = "fail"
    assert _evidence(run, label, events, graph) is None
