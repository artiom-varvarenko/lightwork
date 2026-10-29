"""Consumer-visible metric boundaries; all labels here are synthetic test inputs."""
from lightwork.eval.metrics import compare_rows, decision, gate_metrics, latency_summary, mcnemar, prediction_summary
import pytest


def verdict(done, contradiction=None):
    return {'done': done, 'contradiction': contradiction, 'evidence': 'synthetic test'}


def row(session='one', **changes):
    return dict(session_id=session, run_id='run', frame_sha='a'*64, step_id='step', trial_id='trial',
                expected=verdict('yes'), prediction=verdict('yes'), spark=verdict('yes'), gate='yes',
                status='ok', decode_tokens=10) | changes


def test_missing_refusals_and_contradictions_are_separate():
    result = gate_metrics([(verdict('no'), None), (verdict('no'), 'unsure'), (verdict('no'), 'no'),
                           (verdict('no'), 'yes'), (verdict('yes'), verdict('yes', 'wrong object')),
                           (verdict('unsure'), 'unsure'), (verdict('yes'), 'yes')])
    assert result == dict(n=7, matched=6, missing=1, caught=1, missed=1, false_alarm=1,
                          correct_pass=1, refusal=2, expected_refusal=1, correct_refusal=1)
    assert decision(verdict('unsure', 'wrong object')) == 'no'


def test_percentiles_empty_singleton_and_linear_interpolation():
    assert latency_summary([]) == {'n': 0, 'p50_ms': None, 'p95_ms': None}
    assert latency_summary([10]) == {'n': 1, 'p50_ms': 10, 'p95_ms': 10}
    assert latency_summary([0, 100]) == {'n': 2, 'p50_ms': 50, 'p95_ms': 95}


def test_mcnemar_exact_and_pair_lengths():
    assert mcnemar([], [])['p_exact'] == 1
    assert mcnemar([True]*6, [False]*6) == {'n': 6, 'left_only_correct': 6, 'right_only_correct': 0, 'p_exact': .03125}
    assert mcnemar([True, False], [False, True])['p_exact'] == 1
    with pytest.raises(ValueError):
        mcnemar([True], [])


def test_invalid_predictions_stay_in_accuracy_denominator():
    result = prediction_summary([row(), row(status='incomplete', prediction=None, decode_tokens=None)])
    assert result['n'] == 2 and result['json_valid'] == 1 and result['truncated'] == 1
    assert result['expected'] == {'n': 2, 'agree': 1}
    assert result['vs_spark_mcnemar']['n'] == 2
    assert result['decode_n'] == 1


def test_pairing_never_crosses_sessions_or_changed_labels():
    result = compare_rows([row('one')], [row('two')])
    assert result['mcnemar']['n'] == 0
    assert result['unpaired_base'] == result['unpaired_tuned'] == 1
    with pytest.raises(ValueError, match='ground truth'):
        compare_rows([row()], [row(expected=verdict('no'))])
    with pytest.raises(ValueError, match='duplicate'):
        compare_rows([row(), row()], [row()])
