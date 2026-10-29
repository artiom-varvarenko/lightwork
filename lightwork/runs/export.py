"""Offline static replay bundles: verified reducer snapshots, never logged scene hashes."""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import tempfile
from pathlib import Path, PurePosixPath
from typing import Any

from ..contracts.canonical import canonical
from ..contracts.run import Manifest
from ..server.screen_state import ScreenState
from .audit import audit_run
from .replay import replay_banner
from .store import DirStore, find_runs
from .verify import verify_run

SITE_SIZE_LIMIT = 100 * 1024 * 1024
WEB = Path(__file__).resolve().parents[1] / "web"
ASSETS = {"index.html": "viewer.html", "static/js/viewer.js": "js/viewer.js", "static/js/scene.js": "js/scene.js"}
RUN_FILE = re.compile(r"(?:events\.jsonl|stepgraph\.json|layout\.json|calib\.json|frames/[0-9a-f]{64}\.jpg|preview/[0-9]+\.jpg|calls/[A-Za-z0-9._-]+/(?:request\.json|meta\.json|response\.(?:json|sse)|attempts/[A-Za-z0-9._-]+\.(?:json|sse)))\Z")


class ExportError(ValueError):
    """A release, integrity, or path-safety gate failed."""


def safe_path(root: Path, relative: str) -> Path:
    parts = PurePosixPath(relative)
    if not relative or parts.is_absolute() or any(p in ("", ".", "..") for p in relative.split("/")) or "\\" in relative:
        raise ExportError(f"unsafe relative path: {relative!r}")
    path = root.joinpath(*parts.parts)
    if not path.resolve().is_relative_to(root.resolve()):
        raise ExportError(f"path escapes root: {relative!r}")
    _no_links(path)
    return path


def _no_links(path: Path) -> None:
    for part in (path, *path.parents):
        if part.is_symlink():
            raise ExportError(f"symlink is not allowed: {part}")


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _manifest(run: Path) -> Manifest:
    _no_links(run)
    for root, dirs, files in os.walk(run, followlinks=False):
        for name in dirs + files:
            _no_links(Path(root) / name)
    manifest = Manifest.model_validate_json(safe_path(run, "manifest.json").read_bytes())
    for rel, sha in manifest.files.items():
        if not RUN_FILE.fullmatch(rel):
            raise ExportError(f"not a public run artifact: {rel!r}")
        path = safe_path(run, rel)
        if not path.is_file() or _digest(path) != sha:
            raise ExportError(f"manifest hash mismatch: {rel}")
    return manifest


class _Snapshots:
    def __init__(self, run: Path):
        self.run = run
        self.timeline: list[dict[str, Any]] = []
        self.trace: list[dict[str, Any]] = []
        self.image = None

    def start(self, verifier):
        if verifier.inputs is None or verifier.manifest is None:
            raise ExportError("run inputs or manifest missing")
        graph, layout, _ = verifier.inputs
        self.screen = ScreenState(graph)
        self.crop = list(layout.work_area_xywh)
        self.previews = DirStore(self.run).previews()
        self.banner = replay_banner(verifier.manifest.created_utc, verifier.manifest.provenance)

    def before_input(self, event):
        pass  # Export is not paced by wall-clock time.

    def after_input(self, event, derived, reducer, last_seq):
        for kind, payload in [(event.type, event.payload), *derived]:
            self.screen.feed(kind, payload)
            if kind == "model_result":
                call = safe_path(self.run, payload["call_dir"] + "/meta.json")
                meta = json.loads(call.read_bytes())
                self.screen.set_call_cost(payload["intent_id"], meta["cost_usd"])
                self.trace.append({"t": event.t, "intent_id": payload["intent_id"], "call_dir": payload["call_dir"], "meta": meta})
        if event.type == "keyframe":
            self.image = {"path": f"frames/{event.payload['frame_sha']}.jpg", "crop": event.payload["crop_xywh"]}
        due = [p for p in self.previews if p[0] <= last_seq]
        if due:
            self.previews = self.previews[len(due):]
            self.image = {"path": due[-1][1].relative_to(self.run).as_posix(), "crop": self.crop}
        self.timeline.append({"t": event.t, "seq": last_seq, "scene": reducer.scene.model_dump(mode="json"), "state": self.screen.snapshot(), "image": self.image})


def build_run_data(run_dir: Path, *, allow_synthetic: bool = False) -> dict[str, Any]:
    """Recompute a single run without modifying it; also used by the C8 HTTP viewer."""
    run = Path(run_dir)
    manifest = _manifest(run)
    if manifest.provenance == "synthetic" and allow_synthetic:
        pass
    elif manifest.provenance != "meta":
        raise ExportError(f"provenance {manifest.provenance}: only audited Meta runs may be exported (local fixtures need --allow-synthetic)")
    else:
        problems = audit_run(run)
        if problems:
            raise ExportError("audit failed: " + "; ".join(problems))
    observer = _Snapshots(run)
    report = verify_run(run, write=False, observer=observer)
    if not report.ok:
        raise ExportError(f"verification failed: {len(report.diffs)} diffs")
    if _manifest(run) != manifest:
        raise ExportError("manifest changed while exporting")
    return {"schema_version": 1, "manifest": manifest.model_dump(mode="json"), "verify": report.model_dump(mode="json"), "banner": observer.banner, "crop": observer.crop, "timeline": observer.timeline, "trace": observer.trace}


def _json(path: Path, data: Any) -> None:
    path.write_text(canonical(data) + "\n", encoding="utf-8")


def export_site(source: Path, output: Path, *, allow_synthetic: bool = False) -> dict[str, Any]:
    """Build in a sibling temporary directory and replace output only after all gates pass."""
    source, output = Path(source), Path(output)
    _no_links(source)
    _no_links(output)
    if output.exists() and (not output.is_dir() or (any(output.iterdir()) and not (output / "bundle.json").is_file())):
        raise ExportError("refusing to replace a non-bundle output directory")
    runs = find_runs(source)
    if not runs:
        raise ExportError(f"no runs found at {source}")
    if source.resolve().is_relative_to(output.resolve()) or output.resolve().is_relative_to(source.resolve()):
        raise ExportError("output and source must not contain one another")
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=".lightwork-export-", dir=output.parent))
    try:
        entries = []
        for run in runs:
            data = build_run_data(run, allow_synthetic=allow_synthetic)
            manifest = data["manifest"]
            run_id = manifest["run_id"]
            if not re.fullmatch(r"[A-Za-z0-9_-]+", run_id) or any(e["id"] == run_id for e in entries):
                raise ExportError("unsafe or duplicate run id")
            dest = stage / "runs" / run_id
            dest.mkdir(parents=True)
            for rel, sha in manifest["files"].items():
                src = safe_path(run, rel)
                target = safe_path(dest, rel)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(src, target)
                if _digest(target) != sha:
                    raise ExportError(f"file changed while copying: {rel}")
            shutil.copyfile(safe_path(run, "manifest.json"), dest / "manifest.json")
            if Manifest.model_validate_json((dest / "manifest.json").read_bytes()).model_dump(mode="json") != manifest:
                raise ExportError("manifest changed while copying")
            _json(dest / "verify.json", data["verify"])
            _json(dest / "replay.json", data)
            entries.append({"id": run_id, "provenance": manifest["provenance"], "created_utc": manifest["created_utc"], "path": f"runs/{run_id}/replay.json"})
        for target, asset in ASSETS.items():
            path = stage / target
            path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(WEB / asset, path)
        (stage / ".nojekyll").touch()
        files = {p.relative_to(stage).as_posix(): _digest(p) for p in sorted(stage.rglob("*")) if p.is_file()}
        bundle = {"schema_version": 1, "publishable": all(e["provenance"] == "meta" for e in entries), "runs": entries, "files": files}
        _json(stage / "bundle.json", bundle)
        validate_site(stage, require_publishable=False)
        backup = None
        if output.exists():
            backup = Path(tempfile.mkdtemp(prefix=".lightwork-previous-", dir=output.parent))
            backup.rmdir()
            output.rename(backup)
        try:
            stage.rename(output)
        except OSError:
            if backup:
                backup.rename(output)
            raise
        if backup:
            shutil.rmtree(backup)
        return bundle
    finally:
        if stage.exists():
            shutil.rmtree(stage)


def validate_site(site: Path, *, require_publishable: bool = True) -> dict[str, Any]:
    """Recheck every bundle and original run hash before deploying an immutable bundle."""
    site = Path(site)
    _no_links(site)
    actual = set()
    size = 0
    for root, dirs, files in os.walk(site, followlinks=False):
        for name in dirs + files:
            path = Path(root) / name
            _no_links(path)
            if path.is_file():
                actual.add(path.relative_to(site).as_posix())
                size += path.stat().st_size
    if size > SITE_SIZE_LIMIT:
        raise ExportError(f"site exceeds 100 MiB cap: {size} bytes")
    bundle = json.loads(safe_path(site, "bundle.json").read_bytes())
    if not bundle.get("runs") or bundle.get("schema_version") != 1:
        raise ExportError("empty or unsupported bundle")
    if require_publishable and not bundle.get("publishable"):
        raise ExportError("synthetic/dev bundle is not publishable")
    if actual != set(bundle["files"]) | {"bundle.json"}:
        raise ExportError("bundle contains missing or unlisted files")
    permitted = set(ASSETS) | {".nojekyll", "bundle.json"}
    for entry in bundle["runs"]:
        run = safe_path(site, "runs/" + entry["id"])
        manifest = _manifest(run)
        permitted.update(f"runs/{entry['id']}/{rel}" for rel in [*manifest.files, "manifest.json", "verify.json", "replay.json"])
    if actual != permitted:
        raise ExportError("bundle contains artifacts outside the public viewer/run allow-list")
    for rel, sha in bundle["files"].items():
        if _digest(safe_path(site, rel)) != sha:
            raise ExportError(f"bundle hash mismatch: {rel}")
    for entry in bundle["runs"]:
        run = safe_path(site, "runs/" + entry["id"])
        manifest = _manifest(run)
        if require_publishable:
            if manifest.provenance != "meta" or entry["provenance"] != "meta":
                raise ExportError("non-Meta run is not publishable")
            # Export metadata is intentionally outside C6's original hash set.
            with tempfile.TemporaryDirectory(prefix="lightwork-publish-audit-") as tmp:
                original = Path(tmp)
                for rel in ["manifest.json", *manifest.files]:
                    target = safe_path(original, rel)
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(safe_path(run, rel), target)
                recomputed = build_run_data(original)
                if canonical(recomputed) != canonical(json.loads((run / "replay.json").read_bytes())):
                    raise ExportError("published replay snapshots differ from fresh reduction")
        report = json.loads((run / "verify.json").read_bytes())
        if not report.get("ok") or report.get("diffs"):
            raise ExportError("bundle verify report does not have 0 diffs")
    return bundle
