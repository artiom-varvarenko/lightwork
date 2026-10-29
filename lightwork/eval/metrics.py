"""Small deterministic metrics with explicit missing/refusal denominators."""
from __future__ import annotations

import math
from collections import Counter
from statistics import median


def decision(verdict):
    if verdict is None:
        return None
    if isinstance(verdict, str):
        return verdict
    return 'no' if verdict.get('contradiction') else verdict['done']


def gate_metrics(rows):
    counts = Counter({key: 0 for key in ('n', 'matched', 'missing', 'caught', 'missed', 'false_alarm', 'correct_pass', 'refusal', 'expected_refusal', 'correct_refusal')})
    for expected, actual in rows:
        truth, prediction = decision(expected), decision(actual)
        counts['n'] += 1
        if prediction is None:
            counts['missing'] += 1
            continue
        counts['matched'] += 1
        if prediction == 'unsure':
            counts['refusal'] += 1
        if truth == 'unsure':
            counts['expected_refusal'] += 1
            counts['correct_refusal'] += prediction == 'unsure'
        elif truth == 'no':
            counts['caught'] += prediction == 'no'
            counts['missed'] += prediction == 'yes'
        else:
            counts['false_alarm'] += prediction == 'no'
            counts['correct_pass'] += prediction == 'yes'
    return dict(counts)


def percentile(values, q):
    values = sorted(values)
    if not values:
        return None
    index = (len(values) - 1) * q
    low = math.floor(index)
    high = math.ceil(index)
    return round(values[low] + (values[high] - values[low]) * (index - low), 3)


def latency_summary(values):
    return {'n': len(values), 'p50_ms': percentile(values, .5), 'p95_ms': percentile(values, .95)}


def mcnemar(left_correct, right_correct):
    if len(left_correct) != len(right_correct):
        raise ValueError('paired observations must have equal lengths')
    b = sum(a and not b for a, b in zip(left_correct, right_correct))
    c = sum(b and not a for a, b in zip(left_correct, right_correct))
    n = b + c
    # Exact two-sided binomial test; integer arithmetic avoids overflow for large N.
    p = min(1.0, 2 * sum(math.comb(n, k) for k in range(min(b, c) + 1)) / (1 << n)) if n else 1.0
    return {'n': len(left_correct), 'left_only_correct': b, 'right_only_correct': c, 'p_exact': p}


def prediction_summary(rows):
    rows = list(rows)
    result = {'n': len(rows), 'json_valid': sum(r['status'] == 'ok' for r in rows),
              'truncated': sum(r['status'] == 'incomplete' for r in rows)}
    result['json_valid_rate'] = result['json_valid'] / len(rows) if rows else None
    result['truncation_rate'] = result['truncated'] / len(rows) if rows else None
    tokens = [r['decode_tokens'] for r in rows if r.get('decode_tokens') is not None]
    result.update(decode_n=len(tokens), median_decode_tokens=median(tokens) if tokens else None)
    for reference in ('expected', 'spark', 'gate'):
        paired = [r for r in rows if decision(r.get(reference)) is not None]
        result[reference] = {'n': len(paired), 'agree': sum(decision(r['prediction']) == decision(r[reference]) for r in paired)}
    paired = [r for r in rows if r.get('spark') is not None]
    result['vs_spark_mcnemar'] = mcnemar(
        [decision(r['spark']) == decision(r['expected']) for r in paired],
        [decision(r['prediction']) == decision(r['expected']) for r in paired])
    return result


def row_key(row):
    return tuple(row[key] for key in ('session_id', 'run_id', 'frame_sha', 'step_id', 'trial_id'))


def compare_rows(base, tuned):
    left, right = {row_key(r): r for r in base}, {row_key(r): r for r in tuned}
    if len(left) != len(base) or len(right) != len(tuned):
        raise ValueError('duplicate evaluation observation')
    keys = sorted(left.keys() & right.keys())
    for key in keys:
        if left[key]['expected'] != right[key]['expected']:
            raise ValueError('paired ground truth differs')
    pairs = [(left[k], right[k]) for k in keys]
    result = {'base': prediction_summary(base), 'tuned': prediction_summary(tuned),
              'unpaired_base': len(left.keys() - right.keys()), 'unpaired_tuned': len(right.keys() - left.keys())}
    result['mcnemar'] = mcnemar(
        [decision(a.get('prediction')) == decision(a['expected']) for a, b in pairs],
        [decision(b.get('prediction')) == decision(b['expected']) for a, b in pairs])
    return result


def compare_saved(base_root, tuned_root, *, include_synthetic=False):
    """O training acceptance entry point; verifies both immutable artifact manifests first."""
    from .glimmer import load_evaluation
    base, base_rows = load_evaluation(base_root)
    tuned, tuned_rows = load_evaluation(tuned_root)
    from .evidence import EvidenceError
    if not include_synthetic and (base.get('synthetic') or tuned.get('synthetic')):
        raise EvidenceError('synthetic evaluation needs --include-synthetic')
    if base['tag'] != 'base' or tuned['tag'] != 'tuned':
        raise ValueError('expected base and tuned artifact tags')
    return compare_rows(base_rows, tuned_rows) | {'synthetic': bool(base.get('synthetic') or tuned.get('synthetic'))}
