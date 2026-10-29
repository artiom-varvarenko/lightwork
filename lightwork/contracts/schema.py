"""JSON Schemas for every contract, generated from the pydantic models.

Regenerate after changing a contract: ``uv run python -m lightwork.contracts.schema``.
``tests/test_contracts.py`` fails when the files on disk are stale.
"""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, TypeAdapter

from . import backend, calls, evaluation, events, layout, render, run, stepgraph

SCHEMA_DIR = Path(__file__).parent / "schemas"


def _event_schema() -> dict:
    schema = events.Event.model_json_schema()
    schema["$defs"] = schema.get("$defs", {})
    schema["x-payloads"] = {t: m.model_json_schema() for t, m in sorted(events.PAYLOADS.items())}
    return schema


SCHEMAS = {
    "C1-event": _event_schema,
    "C2-page-to-server": lambda: TypeAdapter(render.PageToServer).json_schema(),
    "C2-server-to-page": lambda: TypeAdapter(render.ServerToPage).json_schema(),
    "C3-stepgraph": stepgraph.StepGraph.model_json_schema,
    "C4-layout": layout.Layout.model_json_schema,
    "C5-call-meta": calls.CallMeta.model_json_schema,
    "C6-manifest": run.Manifest.model_json_schema,
    "C6-verify": run.VerifyReport.model_json_schema,
    "C7-calibration": layout.Calibration.model_json_schema,
    "C9-verdict": backend.Verdict.model_json_schema,
    "C9-mask": backend.Mask.model_json_schema,
    "trial-card": evaluation.TrialCard.model_json_schema,
    "trial-label": evaluation.TrialLabel.model_json_schema,
}

MODELS: dict[str, type[BaseModel]] = {
    "C3-stepgraph": stepgraph.StepGraph,
    "C4-layout": layout.Layout,
    "C5-call-meta": calls.CallMeta,
    "C6-manifest": run.Manifest,
    "C6-verify": run.VerifyReport,
    "C7-calibration": layout.Calibration,
    "C9-verdict": backend.Verdict,
    "C9-mask": backend.Mask,
    "trial-card": evaluation.TrialCard,
    "trial-label": evaluation.TrialLabel,
}


def render_all() -> dict[str, str]:
    return {f"{name}.schema.json": json.dumps(fn(), indent=2, sort_keys=True) + "\n" for name, fn in SCHEMAS.items()}


def write_all(directory: Path = SCHEMA_DIR) -> list[Path]:
    directory.mkdir(parents=True, exist_ok=True)
    out = []
    for fname, text in render_all().items():
        p = directory / fname
        p.write_text(text, encoding="utf-8")
        out.append(p)
    return out


if __name__ == "__main__":
    for p in write_all():
        print(p)
