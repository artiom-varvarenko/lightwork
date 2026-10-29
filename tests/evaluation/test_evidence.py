"""Integrity boundaries and offline determinism over clearly synthetic test fixtures."""
import json
import shutil
from pathlib import Path

import pytest

from lightwork.contracts.evaluation import TrialLabel
from lightwork.eval.evidence import EvidenceError, RunEvidence, load_runs, observations
from lightwork.eval.reporting import markdown_report

FIXTURE = Path(__file__).parents[1] / 'fixtures/runs/synthetic-kit'
pytestmark = pytest.mark.usefixtures('no_network')


def label(**changes):
    return TrialLabel.model_validate(dict(session_id='physical-one', run_id='run', frame_sha='a'*64,
        trial_id='trial', step_id='pack', question='Is it packed?',
        expected={'done':'yes','contradiction':None,'evidence':'synthetic script'}, source='trial-script', reviewed=True) | changes)


def event(type_, **payload):
    return {'type':type_, 'payload':payload}


def test_no_evidence_report_is_explicit_and_identical(tmp_path):
    first = markdown_report(tmp_path / 'absent')
    assert first == markdown_report(tmp_path / 'absent')
    assert 'No evidence available' in first and 'unavailable' in first


def test_synthetic_needs_explicit_optin_and_report_is_deterministic():
    with pytest.raises(EvidenceError, match='synthetic'):
        load_runs(FIXTURE)
    first = markdown_report(FIXTURE, include_synthetic=True)
    assert first == markdown_report(FIXTURE, include_synthetic=True)
    assert 'SYNTHETIC TEST EVIDENCE' in first


def test_tampered_manifest_rejected_before_metrics(tmp_path):
    path = tmp_path / 'fixture'
    shutil.copytree(FIXTURE, path)
    meta = next(path.glob('calls/*/meta.json'))
    data = json.loads(meta.read_text())
    data['total_ms'] = 1
    meta.write_text(json.dumps(data))
    with pytest.raises(EvidenceError, match='verification failed'):
        markdown_report(path, include_synthetic=True)


def test_observations_require_exact_step_frame_and_attempt():
    events = [event('step',step='pack',state='active'),
              event('model_request',intent_id='other/judge/1',frame_sha='a'*64,model='muse-spark-1.3'),
              event('step',step='pack',state='passed'),
              event('model_request',intent_id='pack/judge/1',frame_sha='b'*64,model='muse-spark-1.3'),
              event('step',step='pack',state='passed')]
    run = RunEvidence(Path('.'), {}, events, [])
    assert observations(run, label()) == {'gate':None,'spark':None}
    events += [event('model_request',intent_id='pack/judge/1',frame_sha='a'*64,model='muse-spark-1.3'),
               event('parsed',intent_id='pack/judge/1',kind='verdict',result=label().expected.model_dump()),
               event('step',step='pack',state='failed'),
               event('step',step='pack',state='active'), event('step',step='pack',state='passed')]
    assert observations(run, label()) == {'gate':'no','spark':label().expected.model_dump()}


@pytest.mark.parametrize('change, message', [
    ({'run_id':'different'}, 'does not match'),
    ({'frame_sha':'0'*64}, 'does not match'),
    ({'step_id':'absent'}, 'does not match'),
    ({'session_id':'second-session'}, 'mismatched physical sessions'),
])
def test_labels_must_match_session_run_frame_and_step(tmp_path, change, message):
    from lightwork.contracts.canonical import canonical
    from lightwork.contracts.events import read_events
    from lightwork.runs.manifest import build_manifest
    path = tmp_path / 'synthetic-label-fixture'
    shutil.copytree(FIXTURE, path)
    events = read_events(path / 'events.jsonl')
    frame = next(e.payload['frame_sha'] for e in events if e.type == 'keyframe' and e.payload['verified'])
    graph = json.loads((path / 'stepgraph.json').read_text())
    first = label(run_id=events[0].payload['run_id'], frame_sha=frame, step_id=graph['steps'][0]['id'])
    second = first.model_copy(update={'trial_id':'second', **change})
    # This is construction of an explicitly synthetic test fixture, not editing a real run.
    (path / 'labels.jsonl').write_text(canonical(first)+'\n'+canonical(second)+'\n')
    (path / 'manifest.json').write_text(canonical(build_manifest(path, events, 'synthetic'))+'\n')
    with pytest.raises(EvidenceError, match=message):
        load_runs(path, include_synthetic=True)


def test_recorded_voice_durations_and_missing_stats(tmp_path):
    from lightwork.eval.reporting import latency_report
    run = RunEvidence(tmp_path, {'files': {}, 'created_utc': '2026-10-29T12:00:00Z'}, [], [])
    assert latency_report([run])['Voice']['n'] == 0
    (tmp_path / 'voice-stats.json').write_text(json.dumps({'latencies_ms': [100, 200, 300]}))
    run.manifest['files']['voice-stats.json'] = 'test-only-already-verified'
    assert latency_report([run])['Voice'] == {
        'n': 3, 'p50_ms': 200, 'p95_ms': 290, 'dates': ['2026-10-29']}
    (tmp_path / 'voice-stats.json').write_text(json.dumps({'latencies_ms': [-1]}))
    with pytest.raises(EvidenceError, match='invalid measured latencies'):
        latency_report([run])
