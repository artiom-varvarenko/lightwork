"""Provenance audit (D15, D21): a run may be published under ``runs/`` only if it is complete, intact and Meta-only."""

from __future__ import annotations

from pathlib import Path

from ..contracts.calls import CallMeta
from ..contracts.run import RUN_SIZE_LIMIT
from ..models.provenance import classify
from ..models.wire import read_body
from .verify import Verifier, format_diff


def audit_run(run_dir: Path) -> list[str]:
    """Release gate: complete, intact, authentic Meta-only records; synthetic fixtures are never evidence.

    The platform and response-id checks also reject legacy fixtures that predate ``provenance="synthetic"``.
    """
    run_dir = Path(run_dir)
    v = Verifier(run_dir)
    v.check_log()
    v.check_files()
    v.load_inputs()
    v.check_remaining_calls()
    v.check_unreferenced_calls()
    problems = [format_diff(d) for d in v.diffs]
    m = v.manifest
    if m is not None:
        if m.provenance != "meta":
            if m.provenance == "synthetic":
                problems.append("provenance synthetic: generated test data must never be published as model evidence")
            else:
                problems.append(f"provenance {m.provenance}: dev stand-in runs stay in runs-dev/ (D15)")
        if m.size_bytes > RUN_SIZE_LIMIT:
            problems.append(f"{m.size_bytes / 2**20:.1f} MB is over the 25 MB hard limit")
    if (m is not None and m.platform == "synthetic") or (v.started and v.started["platform"] == "synthetic"):
        problems.append("platform synthetic: generated test data is not authentic Meta evidence")
    for meta_path in sorted(run_dir.glob("calls/*/meta.json")):
        rel = meta_path.relative_to(run_dir).as_posix()
        try:
            meta = CallMeta.model_validate_json(meta_path.read_bytes())
        except (OSError, ValueError) as e:
            problems.append(f"{rel}: not a C5 call meta ({type(e).__name__})")
            continue
        if meta.provenance == "meta" and classify(meta.backend, meta.base_host, meta.model, meta.server_props) != "meta":
            problems.append(f"{rel}: model/server provenance is not Meta despite the meta label (D15)")
        ids = [meta.response_id]
        try:
            raw, kind = v.store.call_response(meta_path.parent.relative_to(run_dir).as_posix())
            ids.append(read_body(raw, kind, meta.endpoint).response_id)
            for attempt in sorted((meta_path.parent / "attempts").glob("*")):
                if attempt.is_file() and attempt.suffix in (".sse", ".json"):
                    ids.append(read_body(attempt.read_bytes(), attempt.suffix[1:], meta.endpoint).response_id)
        except (OSError, ValueError) as e:
            problems.append(f"{rel}: cannot inspect response identity ({type(e).__name__})")
        if any(value and value.startswith(("resp_synthetic", "chatcmpl-synthetic")) for value in ids):
            problems.append(f"{rel}: synthetic response id; generated test data is not authentic Meta evidence")
        if not meta.attempts:
            problems.append(f"{rel}: served from the dev cache, not a recorded call (D21)")
    return problems
