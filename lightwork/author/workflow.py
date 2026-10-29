"""Recorded authoring and auditable human edits; model calls use the shared transport and budget."""
from __future__ import annotations

import difflib
import json
import math
import re
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path

from lightwork.contracts.canonical import canonical_bytes
from lightwork.contracts.stepgraph import StepGraph, validate_semantics
from lightwork.models.budget import Budget
from lightwork.models.client import ModelClient, run_provenance
from lightwork.models.parse import parse_response
from lightwork.models.wire import read_body
from lightwork.models.requests import backend_models, build_request
from .sources import Source


class AuthoringFailed(ValueError):
    pass


def validated_graph(value: dict) -> StepGraph:
    graph = StepGraph.model_validate(value)
    errors = validate_semantics(graph)
    for step in graph.steps:
        for check in step.checks:
            if check.kind == 'series' and check.ratio is not None and not 0 < check.ratio < 1:
                errors.append(f'{step.id}/{check.id}: series ratio must be between zero and one (0.5 for 1:2)')
    if errors:
        raise ValueError('; '.join(errors))
    return graph


def read_call(root: Path, result: dict, role: str, endpoint: str) -> dict:
    folder = root / result['call_dir']
    sse = folder / 'response.sse'
    raw = sse if sse.exists() else folder / 'response.json'
    return parse_response(role, raw.read_bytes(), 'sse' if sse.exists() else 'json', endpoint)


def author(source: Source, *, slug: str, output: Path, settings, backend: str,
           timeout_s: float = 180, transport=None, notes: str = '') -> StepGraph:
    if not re.fullmatch(r'[a-z0-9]+(?:-[a-z0-9]+)*', slug):
        raise ValueError('slug must contain lowercase words separated by hyphens')
    if not math.isfinite(timeout_s) or timeout_s <= 0:
        raise ValueError('timeout must be positive and finite')
    target = output / 'stepgraph.json'
    if target.exists():
        raise ValueError('stepgraph.json exists; use --human-edit to preserve its model-output diff')
    wire_backend, models = backend_models(backend, settings.chat_model)
    provenance, _ = run_provenance(settings, wire_backend, {'author': models['author']}, transport=transport)
    if provenance != 'meta':
        raise ValueError('Procedure authoring packs require actual Meta models, not dev stand-ins')
    session = output / 'authoring' / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    session.mkdir(parents=True, exist_ok=False)
    ctx = {'text': source.text, 'notes': 'Preserve all explicit source step IDs and actions. Use series ratio=0.5 for 1:2. ' + notes}
    frames, frame_sha = {}, None
    if source.pdf is not None and backend == 'meta':
        frame_sha = sha256(source.pdf).hexdigest()
        frames[frame_sha] = source.pdf
        ctx.pop('text')
        ctx['filename'] = 'source.pdf'
        (session / 'source.pdf').write_bytes(source.pdf)
    (session / 'source.json').write_bytes(canonical_bytes(source.metadata))
    (session / 'source.txt').write_text(source.text, encoding='utf-8')
    summary = {'provenance': provenance, 'backend': wire_backend, 'model': models['author'],
               'timeout_s': timeout_s, 'attempts': [], 'status': 'error'}
    try:
        with ModelClient(settings, run_dir=session, provenance=provenance,
                         budget=Budget.from_settings(settings), cache=None, transport=transport) as client:
            for index in (1, 2):
                endpoint, request = build_request('author', model=models['author'], backend=wire_backend,
                                                  frame_sha=frame_sha, ctx=ctx)
                result = client.execute(intent_id=f'author-{index}', role='author', endpoint=endpoint,
                                        request=request, frames=frames, index=index, timeout_s=timeout_s)
                parsed = read_call(session, result, 'author', endpoint)
                summary['attempts'].append({'call': result, 'parsed': parsed})
                if result['status'] != 'ok':
                    raise AuthoringFailed(f"author call {result['status']}: {parsed.get('error')}; retained at {session}")
                if parsed['status'] == 'ok':
                    try:
                        graph = validated_graph({**parsed['result']['stepgraph'], 'id': slug, 'source': source.metadata})
                    except ValueError as exc:
                        folder = session / result['call_dir']
                        raw_path = folder / 'response.sse'
                        kind = 'sse'
                        if not raw_path.exists():
                            raw_path, kind = folder / 'response.json', 'json'
                        text = read_body(raw_path.read_bytes(), kind, endpoint).text
                        parsed.update(status='invalid', error=str(exc), result={'text': text})
                    else:
                        data = canonical_bytes(graph.model_dump(mode='json', by_alias=True))
                        (session / 'model-stepgraph.json').write_bytes(data)
                        target.write_bytes(data)
                        summary['status'] = 'ok'
                        return graph
                if index == 1 and parsed['status'] in ('invalid', 'incomplete'):
                    ctx['retry'] = ({'reason': 'length'} if parsed['status'] == 'incomplete' else
                                    {'reason': 'invalid', 'previous': parsed['result'].get('text', ''), 'error': parsed['error']})
                    continue
                raise AuthoringFailed(f"invalid author output after {index} call(s): {parsed['error']}; retained at {session}")
    except Exception as exc:
        summary['error'] = str(exc)
        raise
    finally:
        (session / 'result.json').write_bytes(canonical_bytes(summary))


def human_edit(output: Path, edited: Path) -> StepGraph:
    originals = sorted((output / 'authoring').glob('*/model-stepgraph.json'))
    if not originals:
        raise ValueError('No saved model output to compare; author this pack first')
    graph = validated_graph(json.loads(edited.read_text()))
    original = json.loads(originals[-1].read_text())
    if graph.id != original['id'] or graph.source.model_dump() != original['source']:
        raise ValueError('Human edits must preserve graph id and source attribution')
    before = json.dumps(original, indent=2, sort_keys=True).splitlines(keepends=True)
    after = json.dumps(graph.model_dump(mode='json', by_alias=True), indent=2, sort_keys=True).splitlines(keepends=True)
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    (output / 'authoring' / f'human-{stamp}.diff').write_text(''.join(difflib.unified_diff(before, after, fromfile=str(originals[-1]), tofile='stepgraph.json')))
    (output / 'stepgraph.json').write_bytes(canonical_bytes(graph.model_dump(mode='json', by_alias=True)))
    return graph
