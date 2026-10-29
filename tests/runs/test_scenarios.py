"""All six H scenarios traverse the real Recorder, C5 writer, Reducer, parser, verifier and replay engine offline."""

from pathlib import Path

import pytest

from lightwork.contracts.calls import CallMeta, request_fingerprint
from lightwork.contracts.events import read_events
from lightwork.core import Reducer
from lightwork.core.scenarios import SCENARIOS, build, drive, run_memory
from lightwork.models.pricing import cost_usd
from lightwork.models.record import write_call
from lightwork.models.wire import read_body
from lightwork.runs import Recorder, ReplayEngine, audit_run, verify_run

pytestmark = pytest.mark.usefixtures("no_network")


def record_scenario(root: Path, scenario) -> Path:
    now = [0.0]
    recorder = Recorder(root, scenario.pack, scenario.run_started, provenance="synthetic", clock=lambda: now[0])
    for name, obj in (("stepgraph.json", scenario.stepgraph), ("layout.json", scenario.layout), ("calib.json", scenario.calib)):
        recorder.put_run_file(name, obj)
    for sha, jpeg in scenario.frames.items():
        assert recorder.put_frame(jpeg) == sha
    reducer = Reducer(scenario.stepgraph, scenario.layout, scenario.calib, recorder.store,
                      backend=scenario.run_started["backend"], models=scenario.run_started["models"])
    recorder.attach(reducer)
    allocated = {}
    stopped = []

    def ingest(type_, src, payload, t):
        now[0] = t
        if type_ == "run_stopped":
            stopped.append(payload["reason"])
            return []  # Recorder.close seals this input and any final derived events exactly once
        out = recorder.ingest(type_, src, payload, t)
        for event in out:
            if event.type == "model_request":
                p = event.payload
                allocated[p["intent_id"]] = recorder.alloc_call(p["role"], p["model"])
        return out

    def on_call(call):
        index, rel = allocated[call.intent_id]
        raw = call.raw if call.raw is not None else b""  # a timeout received no response bytes
        kind = call.kind if call.raw is not None else "json"
        response_id = read_body(raw, kind, call.endpoint).response_id
        meta = CallMeta(
            backend=scenario.run_started["backend"], base_host="api.meta.ai", endpoint=call.endpoint,
            model=call.model, status=call.status, http_status=200 if call.status == "ok" else None,
            attempts=[{"n": 1, "status": 200 if call.status == "ok" else None,
                       "error": None if call.status == "ok" else call.status, "wait_s": 0.0, "ms": call.latency_ms}],
            headers={}, response_id=response_id, ttfb_ms=0 if raw else None, total_ms=call.latency_ms,
            usage=call.usage, cost_usd=cost_usd(call.model, call.usage), fingerprint=request_fingerprint(call.request),
            provenance="synthetic",
        )
        assert write_call(recorder.run_dir, index=index, role=call.role, request=call.request,
                          raw=raw, kind=kind, meta=meta) == rel
        return rel

    drive(scenario, ingest, reducer.request_for, on_call)
    assert len(stopped) == 1
    return recorder.close(stopped[0])


@pytest.mark.parametrize("name", sorted(SCENARIOS))
def test_real_scenario_record_verify_and_replay(tmp_path, name):
    scenario = build(name)
    run_dir = record_scenario(tmp_path, scenario)
    report = verify_run(run_dir)
    assert report.ok, report.diffs
    assert report.n_calls > 0
    expected, _calls = run_memory(build(name))
    assert [event.line() for event in read_events(run_dir / "events.jsonl")] == [event.line() for event in expected]
    replay = ReplayEngine(run_dir, speed=1_000_000.0, echo=lambda _: None).run()
    assert replay == report
    assert any("provenance synthetic" in reason for reason in audit_run(run_dir))
