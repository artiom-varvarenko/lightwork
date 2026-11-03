from __future__ import annotations

import json
import math

import jsonschema
import pytest
from pydantic import TypeAdapter, ValidationError

from lightwork.contracts import canonical, file_sha
from lightwork.contracts.calls import request_fingerprint
from lightwork.contracts.events import ChainError, Event, check_chain, read_events, seal
from lightwork.contracts.layout import Calibration
from lightwork.contracts.render import PageToServer, ServerToPage
from lightwork.contracts.run import UNHASHED_FILES, Manifest
from lightwork.contracts.schema import MODELS, SCHEMA_DIR, render_all
from lightwork.contracts.stepgraph import StepGraph, validate_semantics

ADAPTERS = {
    **{name: TypeAdapter(model) for name, model in MODELS.items()},
    "C2-page-to-server": TypeAdapter(PageToServer),
    "C2-server-to-page": TypeAdapter(ServerToPage),
}


def test_schema_files_are_current():
    for fname, text in render_all().items():
        assert (SCHEMA_DIR / fname).read_text(encoding="utf-8") == text, f"{fname} stale: run python -m lightwork.contracts.schema"


@pytest.mark.parametrize("name", sorted(ADAPTERS))
def test_example_round_trip_and_schema(fixtures_dir, name):
    raw = json.loads((fixtures_dir / "contracts" / f"{name}.json").read_text())
    adapter = ADAPTERS[name]
    obj = adapter.validate_python(raw)
    again = adapter.validate_json(adapter.dump_json(obj))
    assert again == obj
    assert canonical(adapter.dump_python(again, mode="json")) == canonical(adapter.dump_python(obj, mode="json"))
    schema = json.loads((SCHEMA_DIR / f"{name}.schema.json").read_text())
    jsonschema.validate(raw, schema)


def test_calibration_is_projector_or_screen_only(fixtures_dir):
    """A half-projector calibration would project without a mapping, or with one on a rig that has no projector."""
    projector = json.loads((fixtures_dir / "contracts" / "C7-calibration.json").read_text())
    screen = {**projector, "H_cam_to_page": None, "page": None, "residuals": None, "flat_field_crop_xywh": None,
              "H_plate_cam_to_page": None, "measurement_mode": "normal"}
    assert Calibration.model_validate(screen).screen_only
    for key in ("H_cam_to_page", "page", "residuals"):
        with pytest.raises(ValidationError, match="needs H_cam_to_page"):
            Calibration.model_validate({**projector, key: None})
        with pytest.raises(ValidationError, match="screen-only"):
            Calibration.model_validate({**screen, key: projector[key]})


def test_render_effects_are_optional_and_left_out_at_default(fixtures_dir):
    """Scenes without effects must dump, and so hash (C1 ``scene``), exactly as before effects existed."""
    adapter = ADAPTERS["C2-server-to-page"]
    schema = json.loads((SCHEMA_DIR / "C2-server-to-page.schema.json").read_text())
    raw = json.loads((fixtures_dir / "contracts" / "C2-server-to-page.json").read_text())
    assert adapter.dump_python(adapter.validate_python(raw), mode="json") == raw
    raw["prims"][0].update(glow=True, draw_ms=600)
    raw["prims"][3]["type_ms"] = 400
    ring = {"kind": "ring", "id": "r", "c": [1.0, 2.0], "r": 3.0, "role": "done", "pulse": False, "glow": True}
    raw["prims"].append(ring)
    jsonschema.validate(raw, schema)
    assert adapter.dump_python(adapter.validate_python(raw), mode="json") == raw
    sfx = {"type": "sfx", "name": "found", "gain": 0.5}
    jsonschema.validate(sfx, schema)
    assert adapter.dump_python(adapter.validate_python(sfx), mode="json") == sfx
    assert adapter.validate_python({"type": "sfx", "name": "scan"}).gain == 1.0
    for bad in ({"type": "sfx", "name": "boom"}, {"type": "sfx", "name": "scan", "gain": 1.5}):
        with pytest.raises(ValidationError):
            adapter.validate_python(bad)
    with pytest.raises(ValidationError):
        adapter.validate_python({**raw, "prims": [{**raw["prims"][0], "draw_ms": 5001}]})


def test_canonical_rules():
    assert canonical({"b": 1, "a": [1.5, -0.0, "é"]}) == '{"a":[1.5,0.0,"é"],"b":1}'
    with pytest.raises(ValueError):
        canonical({"x": math.nan})
    with pytest.raises(ValueError):
        canonical({"x": math.inf})


def test_fixture_chain_and_payloads(synthetic_kit_run):
    events = read_events(synthetic_kit_run / "events.jsonl")
    run_id = json.loads((synthetic_kit_run / "manifest.json").read_text())["run_id"]
    check_chain(run_id, events)
    for ev in events:
        ev.typed_payload()
    # every line is already canonical
    lines = (synthetic_kit_run / "events.jsonl").read_text().splitlines()
    assert lines == [ev.line() for ev in events]


def test_chain_detects_edit_and_gap(synthetic_kit_run):
    events = read_events(synthetic_kit_run / "events.jsonl")
    run_id = json.loads((synthetic_kit_run / "manifest.json").read_text())["run_id"]
    edited = list(events)
    p = dict(edited[6].payload)
    p["text"] = "Put the cable in."
    edited[6] = Event.model_validate({**edited[6].model_dump(), "payload": p})
    with pytest.raises(ChainError, match="seq 7: hash mismatch"):
        check_chain(run_id, edited)
    with pytest.raises(ChainError, match="seq gap"):
        check_chain(run_id, events[:3] + events[4:])
    with pytest.raises(ChainError):
        check_chain("other-run", events)


def test_seal_rejects_wrong_payload():
    with pytest.raises(ValidationError):
        seal("0" * 64, 1, 0.0, "say", "reducer", {"text": "hi"})  # missing priority
    with pytest.raises(ValidationError):
        seal("0" * 64, 1, 0.0, "say", "reducer", {"text": "hi", "priority": "normal", "extra": 1})


def test_manifest_hashes(synthetic_kit_run):
    m = Manifest.model_validate_json((synthetic_kit_run / "manifest.json").read_text())
    on_disk = {
        p.relative_to(synthetic_kit_run).as_posix()
        for p in synthetic_kit_run.rglob("*")
        if p.is_file() and p.name not in UNHASHED_FILES
    }
    assert set(m.files) == on_disk
    for rel, sha in m.files.items():
        assert file_sha(synthetic_kit_run / rel) == sha, rel


def test_call_fingerprint_matches_intent(synthetic_kit_run):
    events = read_events(synthetic_kit_run / "events.jsonl")
    intents = {e.payload["intent_id"]: e.payload for e in events if e.type == "model_request"}
    for e in events:
        if e.type != "model_result":
            continue
        call = synthetic_kit_run / e.payload["call_dir"]
        req = json.loads((call / "request.json").read_text())
        meta = json.loads((call / "meta.json").read_text())
        fp = request_fingerprint(req)
        assert fp == meta["fingerprint"] == intents[e.payload["intent_id"]]["fingerprint"]
        # stream is transport-only: toggling it must not change the fingerprint
        assert request_fingerprint({**req, "stream": not req.get("stream")}) == fp


def _graph(**over):
    base = json.loads(json.dumps({
        "id": "g", "title": "g",
        "source": {"kind": "text", "ref": "x", "licence": "CC BY 4.0", "attribution": "a"},
        "objects": [
            {"id": "plate", "noun": "well plate", "kind": "grid", "grid": {"rows": 8, "cols": 12, "naming": "A1"}},
            {"id": "tube", "noun": "tube", "kind": "item"},
        ],
        "steps": [{
            "id": "s1", "say": "Move A5 to A6.", "targets": ["plate:A6"], "arrow": {"from": "plate:A5", "to": "plate:A6"},
            "checks": [
                {"id": "c1", "kind": "colour_change", "object": "plate", "cells": ["A6"], "direction": "darker"},
                {"id": "j", "kind": "judge", "question": "Did liquid go into A6?"},
            ],
            "recovery": [
                {"on": "wrong_cell:A7", "action": "exclude_cells", "cells": ["plate:A6"], "say": "liquid went into A7, expected A6"},
                {"on": "wrong_cell:A7", "action": "skip_steps", "steps": ["s1"]},
            ],
        }],
    }))
    base.update(over)
    return StepGraph.model_validate(base)


def test_stepgraph_semantics_ok_and_errors():
    assert validate_semantics(_graph()) == []
    g = _graph()
    bad = g.model_dump(by_alias=True)
    bad["steps"][0]["targets"] = ["plate:Z99"]
    bad["steps"][0]["checks"] = [{"id": "c1", "kind": "colour_change", "object": "plate", "cells": ["A6"], "direction": "darker"}]
    errs = validate_semantics(StepGraph.model_validate(bad))
    assert any("Z99" in e for e in errs)
    assert any("no Meta-reading" in e for e in errs)
    confirm_only = g.model_dump(by_alias=True)
    confirm_only["steps"][0]["checks"] = [{"id": "ok", "kind": "confirm"}]
    confirm_only["steps"][0]["recovery"] = []
    assert validate_semantics(StepGraph.model_validate(confirm_only)) == []
