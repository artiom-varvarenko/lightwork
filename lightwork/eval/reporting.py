"""Offline reports: observed dates, explicit empty evidence and reproducible commands."""
from __future__ import annotations

import json
import math
import shlex

from .evidence import labelled_rows, load_runs
from .metrics import gate_metrics, latency_summary


def gate_report(runs):
    rows = list(labelled_rows(runs))
    return gate_metrics((label.expected.model_dump(), observation['gate']) for _, label, observation in rows)


def latency_report(runs):
    values = {name: [] for name in ('SAM', 'Spark', 'Glimmer', 'Voice')}
    dates = {name: set() for name in values}
    for run in runs:
        for rel in sorted(run.manifest['files']):
            if not rel.startswith('calls/') or not rel.endswith('/meta.json'):
                continue
            meta = json.loads((run.path / rel).read_text())
            model = meta['model'].lower()
            family = next((name for name, marker in (('SAM', 'sam-'), ('Spark', 'spark'), ('Glimmer', 'glimmer'), ('Voice', 'voice')) if marker in model), None)
            if family is not None and not (family == 'Voice' and 'voice-stats.json' in run.manifest['files']):
                values[family].append(meta['total_ms'])
                dates[family].add(run.manifest['created_utc'][:10])
        if 'voice-stats.json' in run.manifest['files']:
            from .evidence import EvidenceError
            stats = json.loads((run.path / 'voice-stats.json').read_text())
            samples = stats.get('latencies_ms', [])
            if not isinstance(samples, list) or any(
                isinstance(value, bool) or not isinstance(value, (int, float))
                or not math.isfinite(value) or value < 0 for value in samples
            ):
                raise EvidenceError('voice-stats.json has invalid measured latencies')
            values['Voice'].extend(samples)
            if samples:
                dates['Voice'].add(run.manifest['created_utc'][:10])
    return {name: latency_summary(samples) | {'dates': sorted(dates[name])} for name, samples in values.items()}


def markdown_report(root, *, include_synthetic=False, evaluations=()):
    runs = load_runs(root, include_synthetic=include_synthetic)
    flag = ' --include-synthetic' if include_synthetic else ''
    arg = shlex.quote(str(root))
    dates = ', '.join(sorted({run.manifest['created_utc'][:10] for run in runs})) or 'unavailable'
    gate = gate_report(runs)
    lines = ['# Lightwork evidence report', '',
             'SYNTHETIC TEST EVIDENCE — not submission metrics.' if any(r.manifest['provenance'] == 'synthetic' for r in runs) else 'Source: verified Meta-only recorded runs.', '',
             'No evidence available: collect and review real trials before claiming performance.' if not runs else f'Verified runs: {len(runs)}.', '',
             '## Gate against reviewed script/operator ground truth', '',
             '| Metric | Count | N | Source dates | Regeneration command |', '|---|---:|---:|---|---|']
    for name in ('caught', 'missed', 'false_alarm', 'correct_pass', 'refusal', 'expected_refusal', 'correct_refusal', 'missing'):
        value = gate[name] if gate['n'] else 'unavailable'
        lines.append(f'| {name} | {value} | {gate["n"]} | {dates} | `uv run lightwork eval gate {arg}{flag}` |')
    lines += ['', 'N is reviewed labels, not attempted predictions. Missing matches are separate; refusals are not counted as caught or missed.', '',
              '## Recorded call latency (all outcomes; milliseconds)', '',
              '| Model | N | p50 | p95 | Source dates | Regeneration command |', '|---|---:|---:|---:|---|---|']
    for name, item in latency_report(runs).items():
        p50 = item['p50_ms'] if item['n'] else 'unavailable'
        p95 = item['p95_ms'] if item['n'] else 'unavailable'
        lines.append(f'| {name} | {item["n"]} | {p50} | {p95} | {", ".join(item["dates"]) or "unavailable"} | `uv run lightwork eval latency {arg}{flag}` |')
    lines += ['', 'Percentiles use linear interpolation. Voice uses manifest-covered voice-stats.json latencies (local VAD end to transcription completion); absent measured durations provide no samples.', '']
    from .evidence import EvidenceError
    from .glimmer import load_evaluation
    from .metrics import prediction_summary
    lines += ['## Saved Glimmer evaluations', '',
              '| Tag / metric | Value | N | Source dates | Regeneration command |',
              '|---|---:|---:|---|---|']
    if not evaluations:
        lines.append('| unavailable | unavailable | 0 | unavailable | `uv run lightwork report --evaluation EVAL_ROOT` |')
    for path in sorted(evaluations, key=str):
        manifest, rows = load_evaluation(path)
        if manifest.get('synthetic') and not include_synthetic:
            raise EvidenceError('synthetic evaluation needs --include-synthetic')
        stats = prediction_summary(rows)
        dates = ', '.join(sorted({row['date'][:10] for row in rows})) or 'unavailable'
        tag = manifest['tag'] + (' (SYNTHETIC)' if manifest.get('synthetic') else '')
        command = f'uv run lightwork report {arg}{flag} --evaluation {shlex.quote(str(path))}'
        for metric in ('json_valid_rate', 'truncation_rate', 'median_decode_tokens'):
            n = stats['decode_n'] if metric == 'median_decode_tokens' else stats['n']
            value = stats[metric] if n else 'unavailable'
            lines.append(f'| {tag} / {metric} | {value} | {n} | {dates} | `{command}` |')
        for reference in ('expected', 'spark', 'gate'):
            pair = stats[reference]
            value = f'{pair["agree"]}/{pair["n"]}' if pair['n'] else 'unavailable'
            lines.append(f'| {tag} / agreement with {reference} | {value} | {pair["n"]} | {dates} | `{command}` |')
    lines.append('')
    return '\n'.join(lines)
