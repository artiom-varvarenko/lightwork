from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import pytest

from lightwork.runs import export as exporter
from lightwork.runs.export import ExportError, build_run_data, export_site, safe_path, validate_site

pytestmark = pytest.mark.usefixtures("no_network")


@pytest.fixture
def run_copy(tmp_path, synthetic_kit_run):
    return Path(shutil.copytree(synthetic_kit_run, tmp_path / "source"))


def snapshot_files(root):
    return {p.relative_to(root).as_posix(): p.read_bytes() for p in root.rglob("*") if p.is_file()}


def write_manifest(run, manifest):
    (run / "manifest.json").write_text(json.dumps(manifest))


def test_export_recomputes_and_retains_original_files(tmp_path, run_copy):
    before = snapshot_files(run_copy)
    site = tmp_path / "site"
    bundle = export_site(run_copy, site, allow_synthetic=True)
    entry = bundle["runs"][0]
    dest = site / "runs" / entry["id"]
    replay = json.loads((dest / "replay.json").read_bytes())
    assert replay["verify"]["ok"] and replay["verify"]["diffs"] == []
    assert not bundle["publishable"]
    assert "SYNTHETIC" in replay["banner"]
    assert len(replay["timeline"]) == replay["verify"]["n_inputs"]
    assert any(row["scene"]["prims"] for row in replay["timeline"])
    assert len({json.dumps(row["scene"]) for row in replay["timeline"]}) > 1
    assert replay["timeline"][-1]["state"]["chips"]
    assert len(replay["trace"]) == replay["verify"]["n_calls"]
    assert len(replay["timeline"][-1]["state"]["trace"]) == len(replay["trace"])
    for row in replay["timeline"]:
        if row["image"]:
            assert (dest / row["image"]["path"]).is_file()
            assert len(row["image"]["crop"]) == 4
    assert (dest / "manifest.json").read_bytes() == before["manifest.json"]
    for rel, sha in replay["manifest"]["files"].items():
        assert hashlib.sha256((dest / rel).read_bytes()).hexdigest() == sha
    assert snapshot_files(run_copy) == before
    assert validate_site(site, require_publishable=False) == bundle
    with pytest.raises(ExportError, match="not publishable"):
        validate_site(site)
    # Re-export is deterministic and replaces an existing bundle safely.
    exported = snapshot_files(site)
    export_site(run_copy, site, allow_synthetic=True)
    assert snapshot_files(site) == exported


def test_default_rejects_synthetic_and_spoofed_meta(run_copy, tmp_path):
    with pytest.raises(ExportError, match="synthetic"):
        export_site(run_copy, tmp_path / "site")
    manifest = json.loads((run_copy / "manifest.json").read_bytes())
    manifest["provenance"] = "meta"
    write_manifest(run_copy, manifest)
    with pytest.raises(ExportError, match="audit failed"):
        build_run_data(run_copy, allow_synthetic=True)


def test_dev_standin_never_allowed(run_copy):
    manifest = json.loads((run_copy / "manifest.json").read_bytes())
    manifest["provenance"] = "dev-standin"
    write_manifest(run_copy, manifest)
    with pytest.raises(ExportError, match="dev-standin"):
        build_run_data(run_copy, allow_synthetic=True)


@pytest.mark.parametrize("target", ["frame", "events", "call"])
def test_tampered_recording_rejected(run_copy, target):
    path = {"frame": next((run_copy / "frames").glob("*.jpg")), "events": run_copy / "events.jsonl", "call": next((run_copy / "calls").glob("*/meta.json"))}[target]
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(ExportError, match="hash mismatch"):
        build_run_data(run_copy, allow_synthetic=True)


def test_rehashed_event_tamper_does_not_bypass_reducer(run_copy):
    path = run_copy / "events.jsonl"
    lines = path.read_text().splitlines()
    event = json.loads(lines[-1])
    event["t"] += 1
    lines[-1] = json.dumps(event, sort_keys=True, separators=(",", ":"))
    path.write_text("\n".join(lines) + "\n")
    manifest = json.loads((run_copy / "manifest.json").read_bytes())
    manifest["files"]["events.jsonl"] = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest["size_bytes"] = sum((run_copy / rel).stat().st_size for rel in manifest["files"])
    write_manifest(run_copy, manifest)
    with pytest.raises(ExportError, match="verification failed"):
        build_run_data(run_copy, allow_synthetic=True)


@pytest.mark.parametrize("relative", ["../secret", "/etc/passwd", "frames/../../secret", "frames\\secret", "./events.jsonl", "frames//secret"])
def test_traversal_rejected(tmp_path, relative):
    with pytest.raises(ExportError):
        safe_path(tmp_path, relative)


def test_manifest_private_path_rejected(run_copy, tmp_path):
    secret = tmp_path / "secret"
    secret.write_text("private")
    manifest = json.loads((run_copy / "manifest.json").read_bytes())
    manifest["files"]["../secret"] = hashlib.sha256(secret.read_bytes()).hexdigest()
    write_manifest(run_copy, manifest)
    with pytest.raises(ExportError, match="public run artifact"):
        build_run_data(run_copy, allow_synthetic=True)


@pytest.mark.parametrize("kind", ["file", "directory", "root"])
def test_symlink_rejected(run_copy, tmp_path, kind):
    if kind == "root":
        link = tmp_path / "linked"
        link.symlink_to(run_copy, target_is_directory=True)
        run_copy = link
    elif kind == "directory":
        target = tmp_path / "outside"
        shutil.move(run_copy / "frames", target)
        (run_copy / "frames").symlink_to(target, target_is_directory=True)
    else:
        path = next((run_copy / "frames").glob("*.jpg"))
        target = tmp_path / "outside.jpg"
        shutil.move(path, target)
        path.symlink_to(target)
    with pytest.raises(ExportError, match="symlink"):
        build_run_data(run_copy, allow_synthetic=True)


def test_size_failure_preserves_output(run_copy, tmp_path, monkeypatch):
    site = tmp_path / "site"
    export_site(run_copy, site, allow_synthetic=True)
    before = snapshot_files(site)
    monkeypatch.setattr(exporter, "SITE_SIZE_LIMIT", 1)
    with pytest.raises(ExportError, match="100 MiB"):
        export_site(run_copy, site, allow_synthetic=True)
    assert snapshot_files(site) == before


def test_bundle_tamper_and_extra_files(run_copy, tmp_path):
    site = tmp_path / "site"
    export_site(run_copy, site, allow_synthetic=True)
    path = site / "index.html"
    original = path.read_bytes()
    path.write_bytes(original + b"tamper")
    with pytest.raises(ExportError, match="bundle hash mismatch"):
        validate_site(site, require_publishable=False)
    path.write_bytes(original)
    (site / "private.txt").write_text("unlisted")
    with pytest.raises(ExportError, match="unlisted"):
        validate_site(site, require_publishable=False)
    # Rehashing a private payload into the bundle inventory must not make it public.
    bundle = json.loads((site / "bundle.json").read_bytes())
    bundle["files"]["private.txt"] = hashlib.sha256((site / "private.txt").read_bytes()).hexdigest()
    (site / "bundle.json").write_text(json.dumps(bundle))
    with pytest.raises(ExportError, match="allow-list"):
        validate_site(site, require_publishable=False)


def test_no_runs_and_unsafe_output(run_copy, tmp_path):
    with pytest.raises(ExportError, match="no runs"):
        export_site(tmp_path / "absent", tmp_path / "site")
    with pytest.raises(ExportError, match="contain one another"):
        export_site(run_copy, run_copy / "site", allow_synthetic=True)
    output = tmp_path / "user-files"
    output.mkdir()
    (output / "keep").write_text("untouched")
    with pytest.raises(ExportError, match="non-bundle"):
        export_site(run_copy, output, allow_synthetic=True)
    assert (output / "keep").read_text() == "untouched"


def test_directory_selection(run_copy, tmp_path):
    bundle = export_site(run_copy.parent, tmp_path.parent / (tmp_path.name + "-site"), allow_synthetic=True)
    assert len(bundle["runs"]) == 1
