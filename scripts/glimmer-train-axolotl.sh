#!/usr/bin/env bash
# Local preparation is free. Training requires an already-provisioned GPU and explicit paid-execution approval.
set -euo pipefail
export LW_REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
exec python3 - "$@" <<'PY'
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

p = argparse.ArgumentParser(description='Prepare/run an audited vision QLoRA recipe; NEVER rents or provisions GPUs.')
p.add_argument('action', choices=['prepare', 'train'])
p.add_argument('--dataset', type=Path, required=True)
p.add_argument('--source-runs', type=Path, required=True)
p.add_argument('--output', type=Path, required=True)
p.add_argument('--approve-paid-execution', action='store_true', help='only after the operator approved this GPU spend')
p.add_argument('--axolotl', default='axolotl', help='Axolotl 0.19.0 executable from an isolated CUDA environment')
p.add_argument('--python', default='python3', help='same isolated environment, for image/token preflight')
a = p.parse_args()
a.dataset, a.output = a.dataset.resolve(), a.output.resolve()
subprocess.run(['uv', 'run', 'lightwork', 'dataset', 'audit', str(a.dataset), '--source-runs', str(a.source_runs.resolve())],
               cwd=os.environ['LW_REPO_ROOT'], check=True)
if a.action == 'train' and not a.approve_paid_execution:
    sys.exit('Training blocked: obtain operator spend approval, then pass --approve-paid-execution. No GPU was provisioned.')
m = json.loads((a.dataset / 'manifest.json').read_text())
if a.action == 'prepare':
    a.output.mkdir(parents=True, exist_ok=False)
    for split in ('train', 'holdout'):
        rows = [json.loads(s) for s in (a.dataset / f'{split}.jsonl').read_text().splitlines()]
        origins = [json.loads(s) for s in (a.dataset / f'{split}-provenance.jsonl').read_text().splitlines()]
        for row, origin in zip(rows, origins, strict=True):
            row['messages'][1]['content'][0] = {'type': 'image', 'path': str(a.dataset / origin['image'])}
        (a.output / f'{split}.jsonl').write_text(''.join(json.dumps(r, sort_keys=True) + '\n' for r in rows))
    # Adapted from the official 0.19.0 vision recipe; only language projections train, preserving the vision tower.
    config = {
        'base_model': 'meta-models/Muse-Glimmer-30B',
        'revision': 'a4e59da52a7bc87ae7251dd5545c0dd437c44b68', 'processor_type': 'AutoProcessor',
        'plugins': ['axolotl.integrations.cut_cross_entropy.CutCrossEntropyPlugin'],
        'load_in_4bit': True, 'skip_prepare_dataset': True, 'remove_unused_columns': False,
        'sample_packing': False, 'eval_sample_packing': False,
        'datasets': [{'path': str(a.output / 'train.jsonl'), 'type': 'chat_template', 'field_messages': 'messages'}],
        'val_set_size': 0.0, 'output_dir': str(a.output / 'adapter'),
        'dataset_prepared_path': str(a.output / 'prepared'), 'adapter': 'qlora',
        'sequence_len': 8192, 'pad_to_sequence_len': False, 'train_on_inputs': False,
        'lora_r': 16, 'lora_alpha': 32, 'lora_dropout': 0.05,
        'lora_target_modules': r'model\.language_model\.layers\.\d+\.(self_attn\.(q|k|v|o)_proj|mlp\.(gate|up|down)_proj)',
        'gradient_accumulation_steps': 16, 'micro_batch_size': 1,
        'num_epochs': 1 if m['mode'] == 'smoke' else 2,
        'optimizer': 'adamw_bnb_8bit', 'lr_scheduler': 'cosine', 'learning_rate': 0.0002,
        'bf16': True, 'tf32': True, 'gradient_checkpointing': True,
        'logging_steps': 1, 'attn_implementation': 'flash_attention_2',
        'warmup_ratio': 0.1, 'evals_per_epoch': 0, 'saves_per_epoch': 1, 'weight_decay': 0.0,
    }
    (a.output / 'axolotl.json').write_text(json.dumps(config, sort_keys=True, indent=2) + '\n')
    provenance = {'dataset_manifest_sha256': hashlib.sha256((a.dataset / 'manifest.json').read_bytes()).hexdigest(),
                  'mode': m['mode'], 'base_model': config['base_model'], 'revision': config['revision'],
                  'source': 'https://github.com/axolotl-ai-cloud/axolotl/blob/v0.19.0/examples/muse-glimmer/qlora-vision.yaml'}
    (a.output / 'lineage.json').write_text(json.dumps(provenance, sort_keys=True, indent=2) + '\n')
else:
    lineage = json.loads((a.output / 'lineage.json').read_text())
    if lineage['dataset_manifest_sha256'] != hashlib.sha256((a.dataset / 'manifest.json').read_bytes()).hexdigest():
        sys.exit('Dataset differs from prepared recipe')
    old = json.loads((a.output / 'artifacts.json').read_text())
    for name, sha in old['files'].items():
        if hashlib.sha256((a.output / name).read_bytes()).hexdigest() != sha:
            sys.exit('Prepared recipe changed: ' + name)
    # preprocess alone skips the multimodal path; actually decode every image and render every conversation.
    preflight = '''import json, sys
from transformers import AutoProcessor
processor = AutoProcessor.from_pretrained("meta-models/Muse-Glimmer-30B", revision="a4e59da52a7bc87ae7251dd5545c0dd437c44b68")
for line in open(sys.argv[1]):
    row = json.loads(line)
    encoded = processor.apply_chat_template(row["messages"], tokenize=True, return_dict=True, return_tensors="pt")
    n = encoded["input_ids"].shape[-1]
    if n > 8192: raise ValueError(f"Training row has {n} tokens; refusing truncation")
print("Image/token preflight passed")
'''
    subprocess.run([a.python, '-c', preflight, str(a.output / 'train.jsonl')], check=True)
    cmd = [a.axolotl, 'train', str(a.output / 'axolotl.json')]
    result = subprocess.run(cmd, capture_output=True, text=True, env={**os.environ, 'CUDA_VISIBLE_DEVICES': '0', 'WANDB_MODE': 'disabled'})
    (a.output / 'train.log').write_text(result.stdout + result.stderr)
    (a.output / 'execution.json').write_text(json.dumps({'command': cmd, 'returncode': result.returncode}) + '\n')
    print(result.stdout, end='')
    print(result.stderr, file=sys.stderr, end='')
files = {}
for f in sorted(a.output.rglob('*')):
    if f.is_file() and f.name != 'artifacts.json':
        with f.open('rb') as stream:
            files[str(f.relative_to(a.output))] = hashlib.file_digest(stream, 'sha256').hexdigest()
(a.output / 'artifacts.json').write_text(json.dumps({'schema_version': 1, 'files': files}, sort_keys=True, indent=2) + '\n')
sys.exit(result.returncode if a.action == 'train' else 0)
PY
