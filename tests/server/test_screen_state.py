from __future__ import annotations

import json

from lightwork.contracts.stepgraph import StepGraph
from lightwork.server.screen_state import ScreenState


def test_events_fold_into_screen_state(synthetic_kit_run):
    sg = StepGraph.model_validate(json.loads((synthetic_kit_run / "stepgraph.json").read_text()))
    ss = ScreenState(sg)
    snap = ss.snapshot()
    assert [s["state"] for s in snap["steps"]] == ["waiting"] * len(sg.steps)
    first = sg.steps[0].id
    assert ss.feed("step", {"step": first, "state": "active", "reason": ""}) == {
        "steps": [{"id": s.id, "say": s.say, "state": "active" if s.id == first else "waiting", "reason": ""}
                  for s in sg.steps],
        "active": first,
    }
    assert ss.feed("check", {"step": first, "check": "c1", "status": "pending", "reason": "", "evidence": []}) == {
        "chips": [{"step": first, "check": "c1", "status": "pending", "reason": ""}]
    }
    patch = ss.feed("check", {"step": first, "check": "c1", "status": "pass", "reason": "ok", "evidence": ["x"]})
    assert patch["chips"] == [{"step": first, "check": "c1", "status": "pass", "reason": "ok"}]
    ss.feed("model_request", {"intent_id": "s1/v/1", "role": "verdict", "model": "m", "fingerprint": "f",
                              "frame_sha": None, "prompt_version": "1"})
    patch = ss.feed("model_result", {"intent_id": "s1/v/1", "call_dir": "calls/0001", "status": "ok", "latency_ms": 812})
    assert patch["trace"] == [{"intent_id": "s1/v/1", "role": "verdict", "model": "m", "status": "ok",
                               "latency_ms": 812, "cost_usd": None}]
    assert ss.set_call_cost("s1/v/1", 0.0012)["trace"][0]["cost_usd"] == 0.0012
    assert ss.set_call_cost("unknown", 1.0) == {}
    assert ss.feed("model_result", {"intent_id": "zz", "call_dir": "c", "status": "ok", "latency_ms": 1}) == {}
    assert ss.feed("cost", {"usd_total": 0.0042}) == {"cost_usd": 0.0042}
    assert ss.feed("say", {"text": "Put the remote in its slot", "priority": "normal"}) == {
        "last_say": "Put the remote in its slot"
    }
    assert ss.feed("step", {"step": "inserted-1", "state": "active", "reason": ""})["steps"][-1]["id"] == "inserted-1"
    assert ss.feed("tick", {}) == {}
