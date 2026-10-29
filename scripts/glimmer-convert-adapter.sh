#!/usr/bin/env bash
# Conversion only: does not start a server or provision inference.
set -euo pipefail
exec python3 - "$@" <<'PY'
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

p = argparse.ArgumentParser(description='Convert a downloaded PEFT Glimmer LoRA with llama.cpp; record hashes and failures.')
p.add_argument('--adapter', type=Path, required=True)
p.add_argument('--llama-cpp', type=Path, required=True)
p.add_argument('--base-config', type=Path, required=True, help='pinned HF base snapshot config/tokenizer directory (not GGUF)')
p.add_argument('--output', type=Path, required=True, help='new artifact directory')
p.add_argument('--python', default='python3', help='llama.cpp conversion environment interpreter')
a = p.parse_args()
a.adapter, a.llama_cpp, a.base_config = (v.resolve() for v in (a.adapter, a.llama_cpp, a.base_config))
converter = a.llama_cpp / 'convert_lora_to_gguf.py'
for f in (converter, a.adapter / 'adapter_config.json', a.base_config / 'config.json'):
    if not f.is_file():
        p.error(f'missing prerequisite: {f}')
if not (a.adapter / 'adapter_model.safetensors').is_file():
    p.error('adapter_model.safetensors required; unsafe pickle checkpoints are not accepted')
config = json.loads((a.adapter / 'adapter_config.json').read_text())
if config.get('base_model_name_or_path') not in {'meta-models/Muse-Glimmer-30B', 'accounts/fireworks/models/muse-glimmer-30b'}:
    p.error('adapter does not attest the Glimmer base; inspect provider metadata, do not relabel another base')
a.output = a.output.resolve()
a.output.mkdir(parents=True, exist_ok=False)
cmd = [a.python, str(converter), '--base', str(a.base_config), '--outfile', str(a.output / 'lightwork-lora.gguf'),
       '--outtype', 'f16', str(a.adapter)]
result = subprocess.run(cmd, capture_output=True, text=True)
(a.output / 'conversion.log').write_text(result.stdout + result.stderr)
def hashes(root):
    files = {}
    for f in sorted(root.rglob('*')):
        if f.is_file():
            with f.open('rb') as stream:
                files[str(f.relative_to(root))] = hashlib.file_digest(stream, 'sha256').hexdigest()
    return files
manifest = {'schema_version': 1, 'command': cmd, 'returncode': result.returncode,
            'base_model': 'meta-models/Muse-Glimmer-30B', 'base_config_files': hashes(a.base_config),
            'adapter_files': hashes(a.adapter), 'converter_sha256': hashlib.sha256(converter.read_bytes()).hexdigest(),
            'artifacts': hashes(a.output), 'serving_tested': False}
(a.output / 'manifest.json').write_text(json.dumps(manifest, sort_keys=True, indent=2) + '\n')
print(result.stdout, end='')
print(result.stderr, file=sys.stderr, end='')
if result.returncode == 0 and not (a.output / 'lightwork-lora.gguf').is_file():
    sys.exit('Converter exited zero without output; conversion not accepted.')
sys.exit(result.returncode)
PY
