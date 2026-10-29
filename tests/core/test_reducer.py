import dataclasses

import pytest

from lightwork.contracts.canonical import canonical_sha
from lightwork.contracts.events import Event, genesis_hash, seal
from lightwork.contracts.stepgraph import Step
from lightwork.core import MemoryStore, Reducer, ReducerError
from lightwork.core.params import Params
from lightwork.core.scenarios import MODELS, SPARK, build
from lightwork.models.pricing import cost_usd
from lightwork.models.synthetic import responses_sse


def judged(sid="place", *, recovery=None, confirm=False):
    return Step.model_validate({"id": sid, "say": "Place the item.", "detail": "It belongs in its marked slot.",
        "targets": ["slot"], "checks": [
            {"id": "activity", "kind": "activity_at", "region": "slot"},
            {"id": "judge", "kind": "judge", "question": "Is the item in place?"},
            *([{"id": "confirm", "kind": "confirm"}] if confirm else []),
        ], "recovery": recovery or []})


def confirmed(sid):
    return Step.model_validate({"id": sid, "say": "Say done.", "checks": [{"id": "done", "kind": "confirm"}]})


class Harness:
    def __init__(self, steps=None, *, params=None):
        self.scenario = s = build("verdict-timeout")
        self.graph = s.stepgraph.model_copy(update={"steps": steps}) if steps is not None else s.stepgraph
        self.store = MemoryStore(s.frames)
        self.reducer = Reducer(self.graph, s.layout, s.calib, self.store, backend="meta", models=MODELS, params=params)
        self.events = []
        self.prev = genesis_hash(s.run_started["run_id"])
        self.sha = next(iter(s.frames))
        self.input(0., "run_started", **s.run_started)
        self.input(0., "procedure_loaded", stepgraph_sha=canonical_sha(self.graph))
        self.input(0., "calibration_loaded", calib_sha=canonical_sha(s.calib))
        self.input(0., "layout_loaded", layout_sha=canonical_sha(s.layout))
        self.frame(.1)

    def input(self, t, type_, **payload):
        ev = seal(self.prev, len(self.events)+1, t, type_, "test", payload)
        self.events.append(ev)
        self.prev = ev.h
        out = self.reducer.feed(ev)
        for et, p in out:
            derived = seal(self.prev, len(self.events)+1, t, et, "reducer", p)
            self.events.append(derived)
            self.prev = derived.h
        return out

    def frame(self, t, *, verified=True):
        return self.input(t, "keyframe", frame_sha=self.sha, crop_xywh=self.scenario.layout.work_area_xywh,
                          mode="grey", verified=verified, t_capture=t)

    def action(self, t, *, hands=True):
        self.input(t, "motion", state="moving", active_regions=["slot"])
        if hands:
            self.input(t+.1, "hands", over_regions=["slot"], tip_xy=[800., 250.])
            self.input(t+.4, "hands", over_regions=[], tip_xy=None)
        self.input(t+1., "motion", state="settled", active_regions=["slot"])
        return self.frame(t+1.2)

    def answer(self, t, iid, *, value=None, status="ok", raw=None, usage=None, terminal="completed", http_status=None):
        from lightwork.contracts.canonical import canonical
        if value is None:
            value = {"done": "yes", "contradiction": None, "evidence": "Synthetic evidence."}
        if raw is None:
            raw = responses_sse(value if isinstance(value, str) else canonical(value), model=SPARK, terminal=terminal)
        path = "calls/" + iid.replace("/", "-")
        self.store.put_call(path, raw, "sse")
        return self.input(t, "model_result", intent_id=iid, call_dir=path, status=status, latency_ms=200,
                          usage=usage, http_status=http_status)

    def steps(self, sid):
        return [e.payload["state"] for e in self.events if e.type == "step" and e.payload["step"] == sid]

    def requests(self):
        return [e.payload for e in self.events if e.type == "model_request"]


def test_only_one_provisional_and_freeze_ready_keyframe(no_network):
    h = Harness([judged("one"), judged("two"), confirmed("three")])
    h.action(2.)
    h.action(4.)
    assert [p["intent_id"] for p in h.requests()] == ["one/judge/1"]
    assert h.steps("two") == ["active"]
    h.input(5.5, "motion", state="moving", active_regions=["slot"])
    h.answer(6., "one/judge/1")
    assert h.steps("one")[-1] == "passed"
    assert h.steps("two")[-1] == "provisional"
    assert h.steps("three") == ["active"]
    assert [p["intent_id"] for p in h.requests()] == ["one/judge/1", "two/judge/1"]
    h.answer(6.5, "two/judge/1")
    assert h.steps("two")[-1] == "passed"


@pytest.mark.parametrize("terminal,value", [("completed", "not JSON"), ("incomplete", "")])
def test_one_derived_repair_or_length_retry(terminal, value, no_network):
    h = Harness()
    h.action(2.)
    h.answer(4., "place/judge/1", value=value, terminal=terminal)
    assert [p["intent_id"] for p in h.requests()] == ["place/judge/1", "place/judge/1/retry"]
    _, before = h.reducer.request_for("place/judge/1")
    _, after = h.reducer.request_for("place/judge/1/retry")
    if terminal == "incomplete":
        assert after["max_output_tokens"] == before["max_output_tokens"] * 2
    else:
        assert len(after["input"]) > len(before["input"])
    h.answer(4.5, "place/judge/1/retry")
    assert h.steps("place")[-1] == "passed"


def test_second_invalid_reply_is_inconclusive_not_infinite_retry(no_network):
    h = Harness()
    h.action(2.)
    h.answer(4., "place/judge/1", value="not JSON")
    h.answer(4.5, "place/judge/1/retry", value="still not JSON")
    assert len(h.requests()) == 2
    h.input(23.2, "tick")
    h.input(24., "key", key="space")
    assert h.steps("place")[-1] == "unverified"
    assert "passed" not in h.steps("place")


def test_tick_deadline_and_late_response_never_upgrades_confirmed_step(no_network):
    h = Harness()
    h.action(2.)
    h.input(23.199, "tick")
    assert not any(e.type == "step" and "needs confirm" in e.payload["reason"] for e in h.events)
    h.input(23.2, "tick")
    assert any(e.type == "step" and "needs confirm" in e.payload["reason"] for e in h.events)
    h.input(23.3, "key", key="space")
    h.answer(24., "place/judge/1")
    assert h.steps("place")[-1] == "unverified"
    assert "passed" not in h.steps("place")
    assert any(p.kind == "label" and "UNVERIFIED" in p.text for p in h.reducer.scene.prims)


def test_timeout_recovery_abort(no_network):
    h = Harness([judged(recovery=[{"on": "timeout", "action": "abort", "say": "Stop here."}]), confirmed("next")])
    h.action(2.)
    h.answer(4., "place/judge/1", status="timeout")
    assert h.steps("place")[-1] == "failed"
    assert h.steps("next")[-1] == "skipped"
    h.input(5., "key", key="space")
    assert "unverified" not in h.steps("next")


def test_unsure_plus_confirm_is_never_passed(no_network):
    h = Harness()
    h.action(2.)
    h.answer(4., "place/judge/1", value={"done": "unsure", "contradiction": None, "evidence": "Occluded."})
    assert h.steps("place")[-1] == "provisional"
    h.input(23.2, "tick")
    h.input(24., "ui", action="done", args={})
    assert h.steps("place")[-1] == "unverified"


def test_confirm_only_next_back_repeat_why_and_rejected_speech(no_network):
    h = Harness([confirmed("one"), confirmed("two"), judged("three")])
    h.input(1., "key", key="space")
    assert h.steps("one")[-1] == "unverified"
    h.input(2., "voice", command="next", transcript="next", session_id="s", turn_id="1")
    assert h.steps("two")[-1] == "skipped"
    h.input(3., "key", key="b")
    assert h.steps("two")[-1] == "active"
    assert h.steps("three")[-1] == "waiting"
    out = h.input(4., "key", key="r")
    assert any(et == "say" and p["text"] == "Say done." for et, p in out)
    out = h.input(5., "ui", action="why", args={})
    assert any(et == "say" and p["text"] == "Say done." for et, p in out)
    assert h.input(6., "voice_rejected", transcript="not done") == []
    assert h.requests() == []


def test_confirm_alone_without_activity_is_unverified(no_network):
    h = Harness()
    h.input(1., "key", key="space")
    assert h.steps("place")[-1] == "unverified"
    assert h.requests() == []


def test_required_confirm_only_passes_with_camera_and_meta_evidence(no_network):
    h = Harness([judged(confirm=True)])
    h.action(2.)
    h.answer(4., "place/judge/1")
    assert h.steps("place")[-1] == "provisional"
    h.input(5., "key", key="space")
    assert h.steps("place")[-1] == "passed"


def test_next_cannot_skip_checked_step_and_back_cannot_undo_pass(no_network):
    h = Harness([judged("one"), confirmed("two")])
    h.input(1., "ui", action="next", args={})
    assert h.steps("one")[-1] == "active"
    h.action(2.)
    h.answer(4., "one/judge/1")
    h.input(5., "key", key="b")
    assert h.steps("one")[-1] == "passed"
    assert h.steps("two")[-1] == "active"


def test_insert_then_redo_and_stale_results_do_not_affect_new_attempt(no_network):
    recovery = [{"on": "judge", "action": "insert_step", "step": confirmed("inspect").model_dump(mode="json")},
                {"on": "judge", "action": "redo_step"}]
    h = Harness([judged("one", recovery=recovery), confirmed("two")])
    h.action(2.)
    h.answer(4., "one/judge/1", value={"done": "no", "contradiction": "wrong item", "evidence": "Synthetic."})
    assert h.steps("inspect") == ["waiting", "active"]
    assert h.steps("one")[-1] == "recovery"
    assert h.steps("two")[-1] == "waiting"
    h.input(5., "key", key="space")
    assert h.steps("one")[-1] == "active"
    h.action(6.)
    assert h.requests()[-1]["intent_id"] == "one/judge/2"
    h.answer(8., "one/judge/2")
    assert h.steps("one")[-1] == "passed"


def test_stale_answer_after_back_is_trace_only(no_network):
    h = Harness([judged("one"), confirmed("two")])
    h.action(2.)
    h.input(3.5, "key", key="b")
    h.answer(4., "one/judge/1")
    assert h.steps("one")[-1] == "active"
    assert "passed" not in h.steps("one")
    assert any(e.type == "parsed" for e in h.events)


def test_moving_or_unverified_keyframe_cannot_gate(no_network):
    h = Harness()
    h.input(1., "motion", state="moving", active_regions=["slot"])
    h.frame(1.1)
    h.input(2., "motion", state="settled", active_regions=["slot"])
    h.frame(2.1, verified=False)
    assert h.requests() == []
    h.frame(2.2)
    assert len(h.requests()) == 1
    assert any(e.type == "check" and "motion-only" in e.payload["reason"] for e in h.events)


def test_hand_over_without_fingertip_does_not_count(no_network):
    h = Harness()
    h.input(1., "motion", state="moving", active_regions=["slot"])
    h.input(1.1, "hands", over_regions=["slot"], tip_xy=None)
    h.input(2., "motion", state="settled", active_regions=["slot"])
    h.frame(2.2)
    assert h.requests() == []


def test_aggregate_billing_includes_error_and_does_not_double_charge_raw_usage(no_network):
    h = Harness()
    h.action(2.)
    usage = {"input_tokens": 10_000, "output_tokens": 100}
    h.answer(4., "place/judge/1", usage=usage, status="error")
    costs = [e.payload["usd_total"] for e in h.events if e.type == "cost"]
    assert costs == [cost_usd(SPARK, usage)]
    assert not any(e.type == "parsed" for e in h.events)  # guard rejection: never trust the body
    h2 = Harness()
    h2.action(2.)
    h2.answer(4., "place/judge/1", usage=usage)
    assert [e.payload["usd_total"] for e in h2.events if e.type == "cost"] == costs


def test_duplicate_unknown_and_derived_inputs_are_errors(no_network):
    h = Harness()
    with pytest.raises(ReducerError, match="unknown intent"):
        h.reducer.request_for("not-derived")
    with pytest.raises(ReducerError, match="not an input"):
        h.reducer.feed(Event(seq=999, t=1., type="scene", src="test", derived=True, payload={"hash": "x"}))
    with pytest.raises(ReducerError, match="does not match"):
        h.input(1., "layout_loaded", layout_sha="0"*64)
    with pytest.raises(ReducerError, match="unknown intent"):
        h.answer(1.5, "unknown")
    h.action(2.)
    h.answer(4., "place/judge/1")
    with pytest.raises(ReducerError, match="second model_result"):
        h.answer(5., "place/judge/1")


def test_threshold_is_runtime_default_for_verify_tamper(monkeypatch, no_network):
    from lightwork.core import params
    monkeypatch.setattr(params, "DEFAULT", dataclasses.replace(Params(), verify_timeout_meta_s=.5))
    h = Harness()
    h.action(2.)
    h.input(3.7, "tick")
    assert any(e.type == "step" and "needs confirm" in e.payload["reason"] for e in h.events)


def test_even_optional_verdict_contradiction_blocks_pass(no_network):
    step = judged()
    step.checks[1] = step.checks[1].model_copy(update={"required": False})
    h = Harness([step])
    h.action(2.)
    h.answer(4., "place/judge/1", value={"done": "yes", "contradiction": "wrong object", "evidence": "Synthetic."})
    assert "failed" in h.steps("place")
    assert "passed" not in h.steps("place")


def test_judge_yes_cannot_override_later_sam_failure(no_network):
    from lightwork.core.scenarios import run_memory
    s = build("kit-remote-missing")
    respond = s.respond

    def delayed(iid, role, model, endpoint, request):
        reply = respond(iid, role, model, endpoint, request)
        return dataclasses.replace(reply, latency_ms=1500) if role == "segment" else reply

    s.respond = delayed
    events, _ = run_memory(s)
    failed = next(e for e in events if e.type == "step" and e.payload["step"] == "remote-in"
                  and e.payload["state"] == "failed")
    assert any(e.type == "check" and e.payload["step"] == "remote-in" and e.payload["check"] == "judge"
               and e.payload["status"] == "pass" and e.seq < failed.seq for e in events)
    assert not any(e.type == "step" and e.payload["step"] == "remote-in" and e.payload["state"] == "passed"
                   and e.seq < failed.seq for e in events)


@pytest.mark.parametrize("http_status,fallback", [(429, True), (400, False), (401, False), (503, False)])
def test_contributor_fallback_is_separate_derived_request(http_status, fallback, no_network):
    h = Harness()
    h.action(2.)
    h.answer(4., "place/judge/1", status="error", http_status=http_status)
    assert len(h.requests()) == (2 if fallback else 1)
    if fallback:
        intent = h.requests()[1]
        assert intent["intent_id"] == "place/judge/1/fallback"
        assert intent["model"] == "muse-spark-1.3"
        assert intent["fingerprint"] != h.requests()[0]["fingerprint"]
        endpoint, request = h.reducer.request_for(intent["intent_id"])
        assert request["model"] == "muse-spark-1.3"
        h.answer(4.5, intent["intent_id"], status="error", http_status=429)
        assert len(h.requests()) == 2  # never recurse or switch back


def test_late_reply_and_late_rate_limit_do_not_bypass_deadline(no_network):
    for status, http in (("ok", 200), ("error", 429)):
        h = Harness()
        h.action(2.)
        h.answer(24., "place/judge/1", status=status, http_status=http)
        assert "passed" not in h.steps("place")
        assert len(h.requests()) == 1
        h.input(24.1, "key", key="space")
        assert h.steps("place")[-1] == "unverified"
