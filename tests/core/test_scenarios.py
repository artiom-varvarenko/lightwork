"""H acceptance: generated camera JPEGs + recorded SSE replies, deterministic and offline."""

from collections import defaultdict

import pytest

from lightwork.contracts.canonical import canonical, canonical_sha
from lightwork.contracts.events import check_chain
from lightwork.contracts.stepgraph import META_CHECKS
from lightwork.core.scenarios import SCENARIOS, build, run_memory


def states(events):
    return {e.payload["step"]: e.payload["state"] for e in events if e.type == "step"}


def assert_invariants(scenario, events):
    current = {}
    checks = defaultdict(dict)
    kinds = {s.id: {c.id: c.kind for c in s.checks} for s in scenario.stepgraph.steps}
    intent_ids = set()
    for ev in events:
        p = ev.payload
        if ev.type == "model_request":
            assert p["intent_id"] not in intent_ids
            intent_ids.add(p["intent_id"])
        if ev.type == "check":
            checks[p["step"]][p["check"]] = p
        if ev.type == "step":
            current[p["step"]] = p["state"]
            assert sum(s == "provisional" for s in current.values()) <= 1
            if p["state"] == "active":
                checks[p["step"]] = {}
            if p["state"] == "passed":
                meta = [o for cid, o in checks[p["step"]].items()
                        if kinds[p["step"]][cid] in META_CHECKS and o["status"] == "pass"]
                assert meta, p
                assert any(ref.startswith("intent:") and ref[7:] in intent_ids for o in meta for ref in o["evidence"])
                step = next(s for s in scenario.stepgraph.steps if s.id == p["step"])
                assert all(checks[p["step"]].get(c.id, {}).get("status") == "pass"
                           for c in step.checks if c.required), p


@pytest.mark.parametrize("name", list(SCENARIOS))
def test_scenario_is_byte_identical_and_meets_acceptance(name, no_network):
    scenario = build(name)
    scenes = []
    events, calls = run_memory(scenario, on_scene=scenes.append)
    again, again_calls = run_memory(build(name))
    assert [e.line() for e in events] == [e.line() for e in again]
    assert [(c.intent_id, c.raw) for c in calls] == [(c.intent_id, c.raw) for c in again_calls]
    check_chain(scenario.run_started["run_id"], events)
    assert_invariants(scenario, events)
    assert scenario.run_started["platform"] == "synthetic"
    assert all(c.raw is None or b"resp_synthetic" in c.raw for c in calls)
    assert [e.payload["hash"] for e in events if e.type == "scene"] == [
        canonical_sha(s.model_dump(mode="json", exclude={"seq"})) for s in scenes]
    assert all(e.t == round(e.t, 3) for e in events)
    final = states(events)
    failed = [e for e in events if e.type == "step" and e.payload["state"] == "failed"]
    if name.startswith("kit") or name == "contradiction-next-active":
        assert set(final.values()) == {"passed"}, final
        if name == "kit-clean":
            assert failed == []
        elif name == "kit-remote-missing":
            assert any(e.payload["step"] == "remote-in" for e in failed)
            assert any("remote control" in e.payload["reason"] for e in failed)
            assert any(p.kind == "ring" and p.role == "error" for s in scenes for p in s.prims)
            assert any(e.type == "step" and e.payload["step"] == "remote-in" and e.payload["state"] == "recovery"
                       for e in events)
        else:
            f = next(e for e in failed if e.payload["step"] == "projector-in")
            assert f.payload["reason"] == "the projector is upside down"
            assert any(e.type == "step" and e.payload["step"] == "cable-in" and e.payload["state"] == "active"
                       and e.seq < f.seq for e in events)
            assert any(e.type == "step" and e.payload["step"] == "cable-in" and e.payload["state"] == "waiting"
                       and "suspended" in e.payload["reason"] for e in events)
    elif name.startswith("dilution"):
        assert len(scenario.stepgraph.steps) == 15  # setup, water, dye, 10 transfers, waste, read
        assert final["read"] == "passed", final
        series = [e for e in events if e.type == "check" and e.payload["check"] == "series"
                  and e.payload["status"] == "pass"]
        assert series
        assert any(e.type == "measurement" for e in events)
        assert final["waste"] == "unverified"  # colour below the camera floor is not a green pass
        assert all(final[f"transfer-{k}"] == "unverified" for k in range(7, 11))
        if name == "dilution-clean":
            assert failed == []
            assert "series over A1-A11" in series[-1].payload["reason"]
            assert final["transfer-5"] == "passed"
        else:
            assert final["transfer-5"] == "failed"
            assert final["transfer-6"] == "skipped"
            assert any(e.payload["reason"] == "liquid went into A7, expected A6" for e in failed)
            assert "series over A1-A5, A7-A11 (A6 excluded)" in series[-1].payload["reason"]
            from lightwork.geometry import Regions
            regions = Regions(scenario.layout, scenario.stepgraph)
            a6, a7 = list(regions.centre("plate:A6")), list(regions.centre("plate:A7"))
            assert any(any(p.kind == "ring" and p.role == "error" and p.c == a7 for p in s.prims)
                       and any(p.kind == "ring" and p.role == "warn" and p.c == a6 for p in s.prims) for s in scenes)
    else:
        assert final == {"place": "unverified"}
        assert not failed
        assert any(e.type == "step" and "needs confirm" in e.payload["reason"] for e in events)
        assert any(c.status == "timeout" for c in calls)


def test_scene_and_measurement_numbers_are_quantized(no_network):
    scenes = []
    events, _ = run_memory(build("dilution-clean"), on_scene=scenes.append)
    for ev in events:
        if ev.type == "measurement":
            for cell in ev.payload["cells"].values():
                assert cell["od"] == round(cell["od"], 4)
                assert cell["sd"] == round(cell["sd"], 4)
                assert all(v == round(v, 2) for v in cell["lab"])
        canonical(ev.payload)  # rejects NaN/Infinity
    for s in scenes:
        for p in s.prims:
            d = p.model_dump()
            for key in ("c", "at", "a", "b"):
                if key in d:
                    assert all(x == round(x, 1) for x in d[key])
            if "pts" in d:
                assert all(x == round(x, 1) for pt in d["pts"] for x in pt)
