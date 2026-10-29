"""Reviewed, session-split verdict datasets. Synthetic opt-in can never produce a release dataset."""
from __future__ import annotations

import base64
import hashlib
import json
from collections import Counter
from pathlib import Path

from ..contracts.backend import Verdict
from ..contracts.canonical import canonical
from ..contracts.evaluation import TrialLabel
from ..contracts.events import read_events
from ..models.prompts.verdict import INSTRUCTIONS, VERSION
from ..models.wire import read_body
from ..runs.audit import audit_run
from ..runs.store import DirStore, find_runs
from ..runs.verify import verify_run

BASE_MODEL = "meta-models/Muse-Glimmer-30B"
BASE_REVISION = "a4e59da52a7bc87ae7251dd5545c0dd437c44b68"
SAM_KINDS = {"present", "absent", "moved_to", "count"}


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _json(path: Path):
    return json.loads(path.read_bytes())


def _lines(path: Path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _decision(verdict: Verdict) -> tuple[str, bool]:
    return verdict.done, bool(verdict.contradiction)


def _evidence(run: Path, label: TrialLabel, events: list, graph: dict) -> list[str] | None:
    """Match the exact question, step and crop, never a nearby frame's answer."""
    step = next((s for s in graph["steps"] if s["id"] == label.step_id), None)
    if step is None:
        return None
    judges = [c for c in step["checks"] if c["kind"] == "judge" and c["question"] == label.question]
    requests = {e.payload["intent_id"]: e.payload for e in events if e.type == "model_request"}
    relevant = {iid for iid, req in requests.items() if req["role"] == "verdict"
                and req.get("frame_sha") == label.frame_sha
                and any(iid.startswith(f"{label.step_id}/{c['id']}/") for c in judges)}
    calls = []
    for event in events:
        p = event.payload
        if event.type != "model_result" or p["intent_id"] not in relevant or p["status"] != "ok":
            continue
        store = DirStore(run)
        meta = _json(store.call_path(p["call_dir"]) / "meta.json")
        raw, kind = store.call_response(p["call_dir"])
        reading = read_body(raw, kind, meta["endpoint"])
        try:
            verdict = Verdict.model_validate_json(reading.text)
        except ValueError:
            continue
        if reading.status != "ok":
            continue
        if _decision(verdict) != _decision(label.expected):
            return None
        calls.append(p["call_dir"])
    if not calls:
        return None
    # Required SAM checks must agree as an aggregate. Pending/error evidence cannot validate a positive label.
    sam = {c["id"] for c in step["checks"] if c["kind"] in SAM_KINDS and c.get("required", True)}
    states = {}
    for event in events:
        p = event.payload
        if (event.type == "check" and p["step"] == label.step_id and p["check"] in sam
                and f"frame:{label.frame_sha}" in p["evidence"]):
            states[p["check"]] = p["status"]
    if sam:
        observed = "no" if "fail" in states.values() else (
            "yes" if sam == states.keys() and all(s == "pass" for s in states.values()) else "unsure")
        if observed != label.expected.done:
            return None
    return sorted(set(calls))


def build_dataset(root: Path, output: Path, *, holdout_sessions: list[str], seed: int = 0,
                  mode: str = "full", licence: str, attribution: str,
                  allow_synthetic_for_tests: bool = False) -> dict:
    if mode not in {"full", "smoke"}:
        raise ValueError("mode must be full or smoke")
    if not licence.strip() or not attribution.strip():
        raise ValueError("explicit image/label licence and attribution are required")
    if not holdout_sessions or len(set(holdout_sessions)) != len(holdout_sessions):
        raise ValueError("supply distinct explicit holdout session ids")
    runs = find_runs(Path(root))
    if not runs:
        raise ValueError("no recorded runs; need reviewed real labels (500 train/60 holdout, or 16-row smoke)")
    rows, sources, dropped = [], {}, Counter()
    seen, frame_sessions, trial_ids = set(), {}, set()
    for run in runs:
        report = verify_run(run, write=False)
        if not report.ok:
            raise ValueError(f"run verification failed: {run.name}: {report.diffs}")
        manifest = _json(run / "manifest.json")
        synthetic = manifest["provenance"] == "synthetic"
        if not (allow_synthetic_for_tests and synthetic):
            problems = audit_run(run)
            if problems:
                raise ValueError(f"Meta provenance audit failed: {run.name}: {problems}")
        if manifest["run_id"] in sources:
            raise ValueError("duplicate source run id")
        sources[manifest["run_id"]] = {"manifest_sha256": digest((run / "manifest.json").read_bytes()),
                                      "provenance": manifest["provenance"], "files": manifest["files"]}
        labels_path = run / "labels.jsonl"
        if not labels_path.exists():
            continue
        if "labels.jsonl" not in manifest["files"]:
            raise ValueError("labels must be covered by the closed run manifest")
        events = read_events(run / "events.jsonl")
        graph = _json(run / "stepgraph.json")
        frames = {e.payload["frame_sha"] for e in events if e.type == "keyframe" and e.payload["verified"]}
        run_session = None
        for value in _lines(labels_path):
            label = TrialLabel.model_validate(value)
            if label.run_id != manifest["run_id"] or label.frame_sha not in frames:
                raise ValueError("label does not reference its run's verified crop")
            if run_session is not None and label.session_id != run_session:
                raise ValueError("duplicate physical session assignment within one run (session leakage)")
            run_session = label.session_id
            previous = frame_sessions.setdefault(label.frame_sha, label.session_id)
            if previous != label.session_id:
                raise ValueError("duplicate frame across physical sessions (session leakage)")
            identity = (label.session_id, label.trial_id)
            if identity in trial_ids:
                raise ValueError("duplicate trial id in a session")
            trial_ids.add(identity)
            key = (label.frame_sha, label.question)
            if key in seen:
                raise ValueError("duplicate crop/question training example")
            seen.add(key)
            if not label.reviewed:
                dropped["unreviewed"] += 1
                continue
            evidence = _evidence(run, label, events, graph)
            if evidence is None:
                dropped["disagreement_or_missing_evidence"] += 1
                continue
            jpeg = DirStore(run).frame_bytes(label.frame_sha)
            rows.append({"label": label.model_dump(mode="json"), "calls": evidence, "jpeg": jpeg})
    sessions = {r["label"]["session_id"] for r in rows}
    holdout = set(holdout_sessions)
    if not holdout <= sessions or not sessions - holdout:
        raise ValueError("holdout ids must have accepted rows and leave at least one train session")
    splits = {name: [r for r in rows if (r["label"]["session_id"] in holdout) == (name == "holdout")]
              for name in ("train", "holdout")}
    for values in splits.values():
        values.sort(key=lambda r: digest(f"{seed}:{canonical(r['label'])}".encode()))
    if mode == "smoke":
        if len(splits["train"]) < 16 or not splits["holdout"]:
            raise ValueError("smoke requires 16 reviewed train rows plus an independent holdout session")
        splits["train"] = splits["train"][:16]
    elif len(splits["train"]) < 500 or len(splits["holdout"]) < 60:
        raise ValueError(f"full dataset requires >=500 train and >=60 holdout; got {len(splits['train'])}/{len(splits['holdout'])}")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    (output / "images").mkdir()
    files, row_hashes = {}, {}
    def emit(name: str, data: bytes):
        (output / name).write_bytes(data)
        files[name] = digest(data)
    for split, values in splits.items():
        chats, provenance = [], []
        for row in values:
            label = row["label"]
            image = f"images/{label['frame_sha']}.jpg"
            if image not in files:
                emit(image, row["jpeg"])
            chat = {"messages": [{"role": "system", "content": INSTRUCTIONS},
                    {"role": "user", "content": [
                        {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + base64.b64encode(row["jpeg"]).decode()}},
                        {"type": "text", "text": "Question: " + label["question"]}]},
                    {"role": "assistant", "content": canonical(label["expected"])}]}
            line = canonical(chat)
            chats.append(line)
            provenance.append(canonical({"label": label, "image": image, "calls": row["calls"], "row_sha256": digest(line.encode())}))
        emit(f"{split}.jsonl", ("\n".join(chats) + "\n").encode())
        emit(f"{split}-provenance.jsonl", ("\n".join(provenance) + "\n").encode())
        row_hashes[split] = [digest(line.encode()) for line in chats]
    result = {"schema_version": 1, "mode": mode, "release_eligible": not allow_synthetic_for_tests,
              "provenance": "test-only" if allow_synthetic_for_tests else "meta",
              "split_seed": seed, "sessions": {s: sorted({r['label']['session_id'] for r in values}) for s, values in splits.items()},
              "counts": {s: len(values) for s, values in splits.items()}, "dropped": dict(sorted(dropped.items())),
              "source_runs": sources, "files": dict(sorted(files.items())), "row_sha256": row_hashes,
              "lineage": {"base_model": BASE_MODEL, "revision": BASE_REVISION, "prompt_version": VERSION,
                          "targets": "operator/trial-script reviewed short Verdict JSON; no reasoning traces"},
              "licence": licence, "attribution": attribution}
    (output / "manifest.json").write_text(canonical(result) + "\n")
    return result


def audit_dataset(root: Path, *, release: bool = True, source_root: Path | None = None) -> dict:
    """Offline integrity and release gate. Source authentication is established during build, not by filenames."""
    root = Path(root)
    m = _json(root / "manifest.json")
    if release and (not m["release_eligible"] or m["provenance"] != "meta"
                    or any(s["provenance"] != "meta" for s in m["source_runs"].values())):
        raise ValueError("test-only/synthetic dataset cannot pass release audit")
    actual = {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()} - {"manifest.json"}
    if actual != set(m["files"]):
        raise ValueError("dataset manifest file set mismatch")
    for name, sha in m["files"].items():
        path = root / name
        if not path.resolve().is_relative_to(root.resolve()) or path.is_symlink() or digest(path.read_bytes()) != sha:
            raise ValueError(f"dataset manifest hash mismatch: {name}")
    if set(m["sessions"]["train"]) & set(m["sessions"]["holdout"]):
        raise ValueError("session leakage")
    if release:
        if source_root is None:
            raise ValueError("release audit requires --source-runs to reauthenticate original runs")
        sources = {_json(p / "manifest.json")["run_id"]: p for p in find_runs(Path(source_root))}
        for run_id, source in m["source_runs"].items():
            run = sources.get(run_id)
            if run is None or digest((run / "manifest.json").read_bytes()) != source["manifest_sha256"]:
                raise ValueError("source run manifest missing or changed")
            if audit_run(run) or not verify_run(run, write=False).ok:
                raise ValueError("source run authentication/verification failed")
    authenticated_labels = {}
    if release:
        for run_id in m["source_runs"]:
            run = sources[run_id]
            events, graph = read_events(run / "events.jsonl"), _json(run / "stepgraph.json")
            accepted = {}
            if (run / "labels.jsonl").is_file():
                for value in _lines(run / "labels.jsonl"):
                    label = TrialLabel.model_validate(value)
                    if label.reviewed:
                        calls = _evidence(run, label, events, graph)
                        if calls:
                            accepted[canonical(label)] = calls
            authenticated_labels[run_id] = accepted
    seen, frame_split = set(), {}
    for split in ("train", "holdout"):
        rows = _lines(root / f"{split}.jsonl")
        provenance = _lines(root / f"{split}-provenance.jsonl")
        if len(rows) != len(provenance) or len(rows) != m["counts"][split]:
            raise ValueError("row count mismatch")
        if [digest(canonical(row).encode()) for row in rows] != m["row_sha256"][split]:
            raise ValueError("row hash mismatch")
        for row, origin, sha in zip(rows, provenance, m["row_sha256"][split], strict=True):
            label = TrialLabel.model_validate(origin["label"])
            if not label.reviewed or label.session_id not in m["sessions"][split] or origin["row_sha256"] != sha:
                raise ValueError("invalid reviewed row provenance")
            if label.run_id not in m["source_runs"] or not origin["calls"]:
                raise ValueError("missing source run/call provenance")
            if release and authenticated_labels[label.run_id].get(canonical(label)) != origin["calls"]:
                raise ValueError("dataset label/evidence differs from authenticated source")
            if origin["image"] != f"images/{label.frame_sha}.jpg":
                raise ValueError("invalid crop path")
            if digest((root / origin["image"]).read_bytes()) != label.frame_sha:
                raise ValueError("image hash mismatch")
            expected_image = "data:image/jpeg;base64," + base64.b64encode((root / origin["image"]).read_bytes()).decode()
            expected_messages = [
                {"role": "system", "content": INSTRUCTIONS},
                {"role": "user", "content": [
                    {"type": "image_url", "image_url": {"url": expected_image}},
                    {"type": "text", "text": "Question: " + label.question}]},
                {"role": "assistant", "content": canonical(label.expected)},
            ]
            if row != {"messages": expected_messages}:
                raise ValueError("training messages differ from reviewed crop/question/target")
            key = (label.frame_sha, label.question)
            if key in seen or frame_split.setdefault(label.frame_sha, split) != split:
                raise ValueError("duplicate example or frame leakage")
            seen.add(key)
    if m["mode"] == "full":
        if m["counts"]["train"] < 500 or m["counts"]["holdout"] < 60:
            raise ValueError("insufficient full dataset rows")
    elif m["mode"] != "smoke" or m["counts"]["train"] != 16 or m["counts"]["holdout"] < 1:
        raise ValueError("invalid smoke dataset counts")
    return m