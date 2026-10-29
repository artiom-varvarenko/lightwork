"""Offline artifact checks using a recorded real Glimmer call and explicit test-only labels."""
import json
import shutil
from pathlib import Path

import pytest

from lightwork.contracts.canonical import canonical
from lightwork.eval.evidence import EvidenceError
from lightwork.eval.glimmer import _files, load_evaluation, parse_call
from lightwork.eval.metrics import compare_saved

CALL = Path(__file__).parents[1] / 'fixtures/calls/chat/calls/0002-verdict-muse-glimmer'
pytestmark = pytest.mark.usefixtures('no_network')


def artifact(path, tag):
    call_dir = 'calls/0001-verdict-muse-glimmer'
    shutil.copytree(CALL, path / call_dir)
    parsed, meta = parse_call(path, call_dir)
    row = {'session_id':'synthetic-session','run_id':'synthetic-fixture','trial_id':'synthetic-trial',
           'step_id':'synthetic-step','frame_sha':'a'*64,
           'expected':{'done':'yes','contradiction':None,'evidence':'TEST ONLY'},
           'prediction': parsed['result'] if parsed['status']=='ok' else None,
           'spark':None,'gate':None,'status':parsed['status'],'call_dir':call_dir,
           'decode_tokens':(meta.get('usage') or {}).get('completion_tokens'),
           'date':'2026-10-29','provenance':'synthetic'}
    (path/'predictions.jsonl').write_text(canonical(row)+'\n')
    (path/'manifest.json').write_text(canonical({'schema_version':1,'tag':tag,'files':_files(path),'synthetic':True}))
    return path


def test_saved_comparison_and_raw_integrity(tmp_path):
    base = artifact(tmp_path/'base','base')
    tuned = artifact(tmp_path/'tuned','tuned')
    result = compare_saved(base,tuned,include_synthetic=True)
    assert result['mcnemar']['n'] == 1
    assert result['mcnemar']['p_exact'] == 1
    raw = next(tuned.glob('calls/*/response.sse'))
    raw.write_bytes(raw.read_bytes()+b'changed')
    with pytest.raises(EvidenceError,match='manifest mismatch'):
        compare_saved(base,tuned,include_synthetic=True)


def test_prediction_cannot_disagree_with_hashed_raw_call(tmp_path):
    root = artifact(tmp_path/'base','base')
    rows = root/'predictions.jsonl'
    data = json.loads(rows.read_text())
    data['prediction'] = {'done':'no','contradiction':'fake edit','evidence':'TEST ONLY'}
    rows.write_text(canonical(data)+'\n')
    manifest = json.loads((root/'manifest.json').read_text())
    manifest['files'] = _files(root)
    (root/'manifest.json').write_text(canonical(manifest))
    with pytest.raises(EvidenceError,match='raw Meta call'):
        load_evaluation(root)
