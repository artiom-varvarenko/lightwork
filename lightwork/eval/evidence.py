"""Offline, integrity-checked evidence. Missing observations never become negative labels."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from lightwork.contracts.evaluation import TrialLabel
from lightwork.runs.verify import verify_run


class EvidenceError(ValueError):
    """An artifact failed integrity or label-to-observation checks."""


@dataclass
class RunEvidence:
    path: Path
    manifest: dict
    events: list[dict]
    labels: list[TrialLabel]


def load_runs(root: Path, *, include_synthetic: bool = False) -> list[RunEvidence]:
    root = Path(root)
    if not root.exists():
        return []
    paths = [root] if (root / 'manifest.json').exists() else sorted(p.parent for p in root.glob('*/manifest.json'))
    runs = []
    seen = set()
    for path in paths:
        report = verify_run(path, write=False)
        if not report.ok:
            raise EvidenceError(f'{path}: verification failed: ' + '; '.join(d.detail for d in report.diffs[:3]))
        manifest = json.loads((path / 'manifest.json').read_text())
        provenance = manifest['provenance']
        if provenance != 'meta' and not (include_synthetic and provenance == 'synthetic'):
            raise EvidenceError(f'{path}: {provenance} evidence is not eligible (synthetic needs --include-synthetic)')
        if manifest['run_id'] in seen:
            raise EvidenceError('duplicate run id')
        seen.add(manifest['run_id'])
        events = [json.loads(line) for line in (path / 'events.jsonl').read_text().splitlines()]
        labels_path = path / 'labels.jsonl'
        labels = [TrialLabel.model_validate_json(line) for line in labels_path.read_text().splitlines()] if labels_path.exists() else []
        frames = {e['payload']['frame_sha'] for e in events if e['type'] == 'keyframe' and e['payload']['verified']}
        steps = {s['id'] for s in json.loads((path / 'stepgraph.json').read_text())['steps']}
        sessions = {label.session_id for label in labels}
        if len(sessions) > 1:
            raise EvidenceError(f'{path}: mismatched physical sessions within one run')
        keys = set()
        for label in labels:
            key = (label.trial_id, label.frame_sha, label.step_id)
            if key in keys:
                raise EvidenceError(f'{path}: duplicate trial label')
            keys.add(key)
            if label.run_id != manifest['run_id'] or label.frame_sha not in frames or label.step_id not in steps:
                raise EvidenceError(f'{path}: label does not match run, verified frame and step')
        runs.append(RunEvidence(path, manifest, events, labels))
    return runs


def observations(run: RunEvidence, label: TrialLabel) -> dict:
    """Match exact frame and step; reset associations on a new step attempt.

    A transition is counted only when that attempt has a check citing this frame or a
    model request for this frame. A later attempt cannot overwrite that decision.
    """
    requests = {}
    matched = False
    gate = None
    spark = None
    for event in run.events:
        p = event['payload']
        if event['type'] == 'model_request':
            requests[p['intent_id']] = p
            if p.get('frame_sha') == label.frame_sha and p['intent_id'].split('/')[0] == label.step_id:
                matched = True
        elif event['type'] == 'parsed' and p.get('kind') == 'verdict':
            req = requests.get(p['intent_id'], {})
            if req.get('frame_sha') == label.frame_sha and p['intent_id'].split('/')[0] == label.step_id and 'spark' in req.get('model', '').lower():
                if p['result'].get('done') in ('yes', 'no', 'unsure'):
                    spark = p['result']
        elif event['type'] == 'check' and p.get('step') == label.step_id:
            if f'frame:{label.frame_sha}' in p.get('evidence', []):
                matched = True
        elif event['type'] == 'step' and p['step'] == label.step_id:
            state = p['state']
            if state == 'active':
                matched = False
            elif matched and state in ('passed', 'failed', 'unverified') and gate is None:
                gate = {'passed': 'yes', 'failed': 'no', 'unverified': 'unsure'}[state]
    return {'gate': gate, 'spark': spark}


def labelled_rows(runs: list[RunEvidence]):
    for run in runs:
        for label in run.labels:
            if label.reviewed:
                yield run, label, observations(run, label)
