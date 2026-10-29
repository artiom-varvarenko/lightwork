#!/usr/bin/env bash
# Official Managed SFT starter; never provisions deployments or opts into paid training.
set -euo pipefail
export LW_REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
exec python3 - "$@" <<'PY'
import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

p = argparse.ArgumentParser(description='Free Glimmer Managed LoRA SFT: audited Lightwork data only; no paid opt-in.')
p.add_argument('action', choices=['plan', 'upload', 'train', 'status', 'watch', 'download'])
p.add_argument('--dataset', type=Path, required=True)
p.add_argument('--source-runs', type=Path, required=True)
p.add_argument('--artifacts', type=Path, required=True)
p.add_argument('--run-id', required=True, help='fresh provider resource suffix; dataset id is lightwork-<run-id>')
p.add_argument('--cookbook', type=Path, default=os.environ.get('FIREWORKS_COOKBOOK'),
               help='checkout of fw-ai/cookbook; install training/ in its own venv')
p.add_argument('--python', default=os.environ.get('FIREWORKS_PYTHON', 'python3'), help='cookbook virtualenv interpreter')
a = p.parse_args()
if not a.run_id or len(a.run_id) > 40 or any(c not in 'abcdefghijklmnopqrstuvwxyz0123456789-' for c in a.run_id):
    p.error('run-id must be <=40 lowercase letters/digits/hyphens')
repo = Path(os.environ['LW_REPO_ROOT'])
if not os.environ.get('FIREWORKS_API_KEY') or not os.environ.get('FIREWORKS_ACCOUNT_ID'):
    p.error('export FIREWORKS_API_KEY and FIREWORKS_ACCOUNT_ID; values are never printed')
a.dataset = a.dataset.resolve()
a.artifacts = a.artifacts.resolve()
a.source_runs = a.source_runs.resolve()
subprocess.run(['uv', 'run', 'lightwork', 'dataset', 'audit', str(a.dataset), '--source-runs', str(a.source_runs)], cwd=repo, check=True)
manifest_bytes = (a.dataset / 'manifest.json').read_bytes()
manifest = json.loads(manifest_bytes)
if a.action in {'upload', 'train'} and dt.datetime.now(dt.timezone.utc) >= dt.datetime(2026, 12, 1, 8, tzinfo=dt.timezone.utc):
    sys.exit('Free promotion expired; paid training is deliberately unsupported by this script.')
if not a.cookbook:
    p.error('--cookbook or FIREWORKS_COOKBOOK is required')
training = a.cookbook.resolve() / 'training'
starter = training / 'examples/sft/muse_glimmer_starter.py'
if not starter.is_file():
    p.error('official cookbook starter missing; see docs/training.md installation steps')
sha = lambda b: hashlib.sha256(b).hexdigest()
config = {'base_model': 'accounts/fireworks/models/muse-glimmer-30b', 'surface': 'managed',
          'lora_rank': 8, 'epochs': 1, 'learning_rate': 0.0001, 'max_context_length': 8192,
          'eval_auto_carveout': True, 'batch_size': 'provider-managed (not asserted as 16)',
          'dataset_mode': manifest['mode'], 'dataset_manifest_sha256': sha(manifest_bytes),
          'train_sha256': manifest['files']['train.jsonl'], 'holdout_sha256': manifest['files']['holdout.jsonl'],
          'starter_sha256': sha(starter.read_bytes()), 'run_id': a.run_id,
          'promotion_deadline_utc': '2026-12-01T08:00:00Z',
          'source': 'https://docs.fireworks.ai/fine-tuning/training-api/cookbook/muse-glimmer'}
a.artifacts.mkdir(parents=True, exist_ok=True)
config_path = a.artifacts / 'training-config.json'
encoded = json.dumps(config, sort_keys=True, indent=2) + '\n'
if config_path.exists() and config_path.read_text() != encoded:
    sys.exit('Artifact directory belongs to a different dataset/config/starter; use a fresh directory.')
if not config_path.exists():
    config_path.write_text(encoded)
dataset_id = 'lightwork-' + a.run_id
cmd = [a.python, '-m', 'examples.sft.muse_glimmer_starter']
shared = ['--surface', 'managed', '--run-id', a.run_id, '--dataset', dataset_id,
          '--lora-rank', '8', '--epochs', '1', '--learning-rate', '0.0001', '--max-context-length', '8192']
if a.action == 'download':
    destination = a.artifacts / 'adapter'
    if destination.exists():
        sys.exit('Adapter directory already exists; refusing to overwrite training artifacts.')
    cmd = ['firectl', '--account-id', os.environ['FIREWORKS_ACCOUNT_ID'],
           'model', 'download', 'muse-starter-' + a.run_id, str(destination)]
elif a.action == 'upload':
    cmd += ['upload', *shared, '--file', str(a.dataset / 'train.jsonl')]
elif a.action == 'train':
    cmd += ['launch', *shared]
else:
    cmd += [('watch' if a.action in {'status', 'watch'} else 'plan'), *shared]
    if a.action == 'status':
        cmd += ['--once']
# Holdout is NEVER uploaded. Managed carveout is additional internal loss validation, not our test split.
if a.action == 'train':
    plan_cmd = [a.python, '-m', 'examples.sft.muse_glimmer_starter', 'plan', *shared]
    planned = subprocess.run(plan_cmd, cwd=training, capture_output=True, text=True)
    if planned.returncode:
        sys.exit('Provider plan failed; run the plan action for the recorded diagnostic. No job launched.')
    plan = json.loads(planned.stdout)
    if (not plan['promotion']['active'] or plan['base_model'] != config['base_model']
            or plan['account'] != os.environ['FIREWORKS_ACCOUNT_ID']
            or plan.get('dataset', {}).get('rows') != manifest['counts']['train']):
        sys.exit('Provider promotion/account/base/dataset rows do not match the audited plan. No job launched.')
    (a.artifacts / 'launch-plan.json').write_text(json.dumps(plan, sort_keys=True, indent=2) + '\n')
result = subprocess.run(cmd, cwd=training, capture_output=True, text=True)
key = os.environ.get('FIREWORKS_API_KEY', '')
def redact(text):
    return text.replace(key, '[REDACTED]') if key else text
stamp = dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')
record = {'action': a.action, 'command': cmd, 'returncode': result.returncode,
          'stdout': redact(result.stdout), 'stderr': redact(result.stderr), 'utc': stamp}
(a.artifacts / f'{stamp}-{a.action}.json').write_text(json.dumps(record, sort_keys=True, indent=2) + '\n')
files = {}
for f in sorted(a.artifacts.rglob('*')):
    if f.is_file() and f.name != 'artifacts.json':
        with f.open('rb') as stream:
            files[str(f.relative_to(a.artifacts))] = hashlib.file_digest(stream, 'sha256').hexdigest()
(a.artifacts / 'artifacts.json').write_text(json.dumps({'schema_version': 1, 'files': files}, sort_keys=True, indent=2) + '\n')
print(redact(result.stdout), end='')
print(redact(result.stderr), file=sys.stderr, end='')
sys.exit(result.returncode)
PY
