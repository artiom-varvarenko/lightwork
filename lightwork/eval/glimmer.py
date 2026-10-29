"""Rejudge reviewed crops without changing source runs; preserve raw C5 call evidence."""
from __future__ import annotations

import json
from pathlib import Path

from lightwork.contracts.canonical import canonical, sha256_hex
from lightwork.contracts.calls import request_fingerprint
from lightwork.models.parse import parse_response
from .evidence import EvidenceError, labelled_rows, load_runs
from .metrics import prediction_summary


def _files(root):
    files = {}
    for path in sorted(Path(root).rglob('*')):
        if path.is_symlink():
            raise EvidenceError('symlinks are not evaluation artifacts')
        if path.is_file() and path != Path(root) / 'manifest.json':
            files[path.relative_to(root).as_posix()] = sha256_hex(path.read_bytes())
    return files


def parse_call(root, call_dir):
    directory = Path(root) / call_dir
    meta = json.loads((directory / 'meta.json').read_text())
    path = directory / 'response.sse'
    kind = 'sse'
    if not path.exists():
        path, kind = directory / 'response.json', 'json'
    parsed = parse_response('verdict', path.read_bytes(), kind, meta['endpoint'])
    return parsed, meta


def load_evaluation(root):
    root = Path(root)
    manifest = json.loads((root / 'manifest.json').read_text())
    if manifest.get('schema_version') != 1 or manifest.get('tag') not in ('base', 'tuned'):
        raise EvidenceError('unsupported evaluation manifest')
    if manifest['files'] != _files(root):
        raise EvidenceError('evaluation artifact manifest mismatch')
    rows = [json.loads(line) for line in (root / 'predictions.jsonl').read_text().splitlines()]
    identities = set()
    for row in rows:
        identity = tuple(row[key] for key in ('session_id', 'run_id', 'frame_sha', 'step_id', 'trial_id'))
        if identity in identities:
            raise EvidenceError('duplicate evaluation observation')
        identities.add(identity)
        rel = Path(row['call_dir'])
        if rel.is_absolute() or '..' in rel.parts:
            raise EvidenceError('unsafe call path')
        parsed, meta = parse_call(root, row['call_dir'])
        request = json.loads((root / rel / 'request.json').read_text())
        if request_fingerprint(request) != meta['fingerprint']:
            raise EvidenceError('evaluation request fingerprint mismatch')
        usage = meta.get('usage') or {}
        if row.get('decode_tokens') != usage.get('completion_tokens', usage.get('output_tokens')):
            raise EvidenceError('decode token count differs from raw call metadata')
        prediction = parsed['result'] if parsed['status'] == 'ok' else None
        if prediction != row['prediction'] or parsed['status'] != row['status'] or meta['provenance'] != 'meta':
            raise EvidenceError('prediction does not match raw Meta call')
    return manifest, rows


def rejudge(root, output, *, settings, tag, include_synthetic=False, client=None, session_ids=None):
    """Create a new immutable eval root. No retries of malformed outputs: report first-call quality."""
    from lightwork.models.budget import Budget
    from lightwork.models.client import ModelClient, run_provenance
    from lightwork.models.requests import build_request, is_glimmer

    if tag not in ('base', 'tuned'):
        raise ValueError('tag must be base or tuned')
    runs = load_runs(root, include_synthetic=include_synthetic)
    selected = set(session_ids or ())
    samples = [(run, label, observation) for run, label, observation in labelled_rows(runs)
               if not selected or label.session_id in selected]
    if selected - {label.session_id for _, label, _ in samples}:
        raise ValueError('selected session has no reviewed evidence')
    if not samples:
        raise ValueError('no reviewed keyframes to rejudge')
    model = settings.chat_model
    if not model or not is_glimmer(model):
        raise ValueError('eval glimmer requires LW_CHAT_MODEL identifying Muse Glimmer')
    if client is None:
        provenance, _ = run_provenance(settings, 'chat', {'verdict': model})
        if provenance != 'meta':
            raise EvidenceError('Glimmer eval requires authenticated Meta open-weight provenance')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    owned = client is None
    if owned:
        client = ModelClient(settings, run_dir=output, provenance='meta', budget=Budget.from_settings(settings), cache=None)
    rows = []
    try:
        for index, (run, label, observation) in enumerate(samples, 1):
            graph = json.loads((run.path / 'stepgraph.json').read_text())
            step = next(step for step in graph['steps'] if step['id'] == label.step_id)
            # Never disclose the script/operator expected label to the model.
            endpoint, request = build_request('verdict', model=model, backend='chat', frame_sha=label.frame_sha,
                                             ctx={'step_id': label.step_id, 'instruction': step['say'], 'question': label.question})
            result = client.execute(intent_id=f'eval/{index}', role='verdict', endpoint=endpoint, request=request,
                                    frames={label.frame_sha: (run.path / 'frames' / f'{label.frame_sha}.jpg').read_bytes()},
                                    index=index, timeout_s=180)
            parsed, meta = parse_call(output, result['call_dir'])
            usage = meta.get('usage') or {}
            row = label.model_dump(mode='json') | observation | {
                'prediction': parsed['result'] if parsed['status'] == 'ok' else None,
                'status': parsed['status'], 'call_dir': result['call_dir'],
                'decode_tokens': usage.get('completion_tokens', usage.get('output_tokens')),
                'date': run.manifest['created_utc'], 'provenance': run.manifest['provenance']}
            rows.append(row)
        (output / 'predictions.jsonl').write_text(''.join(canonical(row) + '\n' for row in rows))
        summary = prediction_summary(rows)
        (output / 'summary.json').write_text(canonical(summary) + '\n')
        manifest = {'schema_version': 1, 'tag': tag, 'model': model, 'files': _files(output),
                    'source_runs': {r.manifest['run_id']: sha256_hex((r.path / 'manifest.json').read_bytes()) for r in runs},
                    'synthetic': any(r.manifest['provenance'] == 'synthetic' for r in runs)}
        (output / 'manifest.json').write_text(canonical(manifest) + '\n')
        return summary
    finally:
        if owned:
            client.close()
