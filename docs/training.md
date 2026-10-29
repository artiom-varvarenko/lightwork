# Glimmer verdict fine-tuning

## Status and prerequisites

**Implemented tooling, no trained Lightwork adapter yet.** There are no reviewed real collection labels available for a legitimate 16-row smoke or full job. On 29 Oct the credential-presence check found no `FIREWORKS_API_KEY`, `FIREWORKS_ACCOUNT_ID`, `HF_TOKEN`, or firectl auth file; neither firectl nor Axolotl was installed. No provider upload, training, paid GPU, deployment, adapter conversion or tuned inference was executed. Synthetic tests do not change this status.

Required operator actions:

1. Collect privacy-reviewed, genuine Meta-only recordings with `labels.jsonl` written **before closing** each run. Give all runs from one physical session the same `session_id`. Review each label; do not derive ground truth from model predictions.
2. Record at least two physical sessions and explicitly designate held-out session IDs. Full acceptance needs **at least 500 accepted train rows and 60 held-out rows**, after filtering. A separately labelled smoke artifact uses **exactly 16 train rows** and at least one independent held-out observation; it cannot satisfy full-data acceptance.
3. Attest the image/label licence and attribution; verify rights before uploading. Procedure text attribution is not automatically a licence for every photo.
4. Obtain Fireworks access and verify the current free LoRA SFT offer. Install its cookbook and firectl in a separate operator environment; never add provider/CUDA dependencies to Lightwork's lockfile.

## Data contract and selection

`lightwork/contracts/evaluation.py::TrialLabel` contains session, run, frame SHA, trial, step, question, short expected Verdict, source (`trial-script|operator`) and review flag. `lightwork/eval/dataset.py`:

- re-verifies each source hash chain, manifest, calls and deterministic reducer, with no run mutation;
- authenticates Meta-only provenance with the existing run audit (a model-name string alone is insufficient);
- requires a manifest-covered label and a verified recorded keyframe in the same run;
- rejects duplicate trial IDs, crop/question pairs and crops assigned across physical sessions;
- drops unreviewed labels, missing/invalid model evidence, or ground-truth/model disagreement;
- matches the exact step, check question and frame to a raw recorded verdict; compares `done` and contradiction presence, not wording;
- additionally requires the aggregate relevant required SAM checks to agree: any fail means `no`, all pass means `yes`, otherwise `unsure`;
- keeps all rows from each physical session on one side of the split. A seed deterministically orders rows; it does **not** randomly split frames or sessions.

The target is the **operator's reviewed short Verdict**, never a model reasoning trace. The system prompt is the same `verdict-v1` instructions as production; the dataset user prompt contains only the check question and crop, not label, session, trial or ground truth. Dataset JSONL is OpenAI-style multimodal chat, image before text, with an inline JPEG data URL. Separate `images/<sha>.jpg` and `*-provenance.jsonl` retain crop hashes, full labels, source type and recorded call paths. `manifest.json` hashes every image/JSONL file and every canonical chat row, records original run-manifest/file hashes, session IDs, ordering seed, licence, attribution and base lineage. No timestamps or absolute paths enter the reproducible dataset artifact.

A dataset output directory must not already exist. Release audit checks content, leakage and labels again against the original, reauthenticated source runs. Keep those runs; the dataset manifest is not a cryptographic signature or a replacement for original evidence.

```bash
# Replace these operator-chosen names with actual physical session IDs.
uv run lightwork dataset build runs/ --output .lightwork/datasets/glimmer-smoke \
  --mode smoke --holdout-session home-20261031-evening \
  --seed 0 --licence Apache-2.0 --attribution 'Artiom Varvarenko, Lightwork collection'
uv run lightwork dataset audit .lightwork/datasets/glimmer-smoke --source-runs runs/

uv run lightwork dataset build runs/ --output .lightwork/datasets/glimmer-full \
  --mode full --holdout-session home-20261031-evening --holdout-session lab-20261102 \
  --seed 0 --licence Apache-2.0 --attribution 'Artiom Varvarenko, Lightwork collection'
uv run lightwork dataset audit .lightwork/datasets/glimmer-full --source-runs runs/
```

These are future collection commands, not claims that the named sessions exist. Missing/insufficient real evidence fails closed, with no dataset written. `--allow-synthetic-for-tests` is solely for temporary unit-test inputs; it marks output `test-only`/`release_eligible:false` even if input is mixed. `dataset audit --test-only` checks integrity only. Training scripts always use the **release audit** and cannot accept that output. Do not put generated training data in `runs/`.

## Fireworks: free Managed LoRA SFT

Sources checked for this implementation:

- [Official Muse Glimmer cookbook](https://docs.fireworks.ai/fine-tuning/training-api/cookbook/muse-glimmer)
- [Official starter source](https://github.com/fw-ai/cookbook/blob/main/training/examples/sft/muse_glimmer_starter.py)
- [firectl model download](https://docs.fireworks.ai/tools-sdks/firectl/commands/model-download)

The documented promotion covers Glimmer LoRA SFT until **2026-12-01 00:00 PST (08:00 UTC)**; no card is required. It does **not** cover deployments, general inference, full-parameter training, RL, DPO, or paid evaluation sampling. Our wrapper uses Managed SFT, not Serverless Training's automatic random-row holdout or billed base/tuned sampling. The platform's internal `eval_auto_carveout` is only training-loss validation; the untouched external session holdout is the acceptance set.

Operator installation (outside the application repository):

```bash
git clone --depth 1 https://github.com/fw-ai/cookbook.git "$HOME/fireworks-cookbook"
python3 -m venv "$HOME/fireworks-cookbook/training/.venv"
"$HOME/fireworks-cookbook/training/.venv/bin/pip" install -e "$HOME/fireworks-cookbook/training"
export FIREWORKS_COOKBOOK="$HOME/fireworks-cookbook"
export FIREWORKS_PYTHON="$FIREWORKS_COOKBOOK/training/.venv/bin/python"
# Read the key silently; do not put its value in commands, screenshots or logs.
read -rs FIREWORKS_API_KEY; export FIREWORKS_API_KEY; printf '\n'
read -r FIREWORKS_ACCOUNT_ID; export FIREWORKS_ACCOUNT_ID
```

Install/authenticate firectl following [Fireworks' CLI documentation](https://docs.fireworks.ai/tools-sdks/firectl/firectl) in the **same account**. The wrapper supplies the account ID to download, but never places an API key in process arguments. It records the actual cookbook starter SHA-256 rather than pretending `main` is a pinned immutable revision; archive the installed checkout/version with the job.

```bash
DATA=.lightwork/datasets/glimmer-smoke
ART=.lightwork/training/smoke-20261031
RUN=lightwork-smoke-20261031  # fresh ID; do not reuse a deleted job's ID
common=(--dataset "$DATA" --source-runs runs/ --artifacts "$ART" --run-id "$RUN")
scripts/glimmer-train-fireworks.sh plan "${common[@]}"
scripts/glimmer-train-fireworks.sh upload "${common[@]}"
scripts/glimmer-train-fireworks.sh plan "${common[@]}"
scripts/glimmer-train-fireworks.sh train "${common[@]}"
scripts/glimmer-train-fireworks.sh status "${common[@]}"  # one poll
scripts/glimmer-train-fireworks.sh watch "${common[@]}"   # provider's bounded-progress watcher
scripts/glimmer-train-fireworks.sh download "${common[@]}"
```

The first plan may report that the new dataset ID does not exist; upload creates it. Before launching, the wrapper obtains a fresh provider plan and requires matching account, Glimmer base, active promotion and train-row count. It never passes `--allow-paid`. Only `train.jsonl` is uploaded, as `lightwork-$RUN`; the output model is `accounts/$FIREWORKS_ACCOUNT_ID/models/muse-starter-$RUN`. Downloads go into the new `adapter/` directory. Failure exit codes are retained; an error is not training success.

Configuration: `accounts/fireworks/models/muse-glimmer-30b`, **LoRA rank 8, one epoch, learning rate 1e-4, context 8192**, provider-managed batch size. The starter's batch-size-16 setting belongs to its Serverless path; we do not falsely claim to set Managed SFT's batch size. Check provider rendered-sample summaries for image tokens, assistant-only loss and absence of truncation. Persist actual resolved learning rate/steps/loss, job/model identity and adapter files.

The artifact directory contains `training-config.json`, a launch plan, timestamped action records, downloaded adapter files and `artifacts.json` with SHA-256 hashes. Configuration/dataset/starter mismatch refuses reuse. For the full job, first pass the full-data builder gate, then use a **new** full dataset path, run ID and artifact directory. Smoke weights are plumbing evidence only.

## GGUF LoRA conversion and serving

Glimmer adapter architecture support in the converter and local LoRA serving are **unverified until a real adapter exists**. Do not silently rename an unsupported adapter or declare a fake conversion successful.

Use a separate llama.cpp conversion environment and checkout. Install that checkout's conversion requirements. Obtain the pinned HF **config/tokenizer snapshot** (not the 60 GB weights); `hf download` is supplied by the Hugging Face CLI:

```bash
hf download meta-models/Muse-Glimmer-30B \
  --revision a4e59da52a7bc87ae7251dd5545c0dd437c44b68 \
  --include '*.json' '*.jinja' 'tokenizer.model' \
  --local-dir "$HOME/models/glimmer-config-pinned"
scripts/glimmer-convert-adapter.sh \
  --adapter "$ART/adapter" --llama-cpp "$HOME/llama.cpp" \
  --base-config "$HOME/models/glimmer-config-pinned" \
  --python "$HOME/llama.cpp/.venv/bin/python" \
  --output .lightwork/training/smoke-gguf
```

The wrapper calls the real [`convert_lora_to_gguf.py`](https://github.com/ggml-org/llama.cpp/blob/master/convert_lora_to_gguf.py) with `--base`, `--outtype f16` and `--outfile`. It requires a Glimmer PEFT base declaration and safetensors (no unsafe pickle checkpoint), retains converter logs/exit status, and hashes input adapter/config files, converter and output. `--base` avoids silently fetching a floating base revision. Archive the whole converter checkout and its dependency versions; the converter hash alone does not identify imported helper modules.

Copy the resulting adapter to the Mac only after successful conversion. With the baseline server stopped by its operator, use the **unchanged** `glimmer-serve.sh` extra-flags interface:

```bash
# Mac side, path points to the copied adapter:
GLIMMER_EXTRA="--lora $HOME/models/muse-glimmer/lightwork-lora.gguf" \
  lightwork-scripts/glimmer-serve.sh start
```

Preserve the base/mmproj hashes from [open-weight.md](open-weight.md), plus the adapter hash, startup log and `/props`. Confirm a real crop produces a valid Verdict, then run paired evaluation. Stop after the session. If conversion/local serving fails, retain the diagnostic. A Fireworks on-demand deployment is **paid per GPU-hour** and requires separate explicit approval; this tooling does not provision it.

## Axolotl H100 fallback (not executed)

This is a reproduction route, not authorisation to rent a GPU. Obtain explicit operator spend approval first. No script rents hardware. Follow [Meta's post-training guide](https://dev.meta.ai/docs/muse-glimmer/post-training) and the [Axolotl 0.19.0 vision recipe](https://github.com/axolotl-ai-cloud/axolotl/blob/v0.19.0/examples/muse-glimmer/qlora-vision.yaml). Meta reports QLoRA verified on one H100; that is upstream evidence, not a Lightwork performance result.

On an already-approved H100 with CUDA 13.0, in an isolated Python 3.12 environment, the documented stack is PyTorch 2.12.1, torchvision 0.27.1, Axolotl 0.19.0, flash-attn 2.8.3 and cut-cross-entropy revision `4dfa52212d0552805aec5a6a2176c0abbea277d2` from `axolotl-ai-cloud/ml-cross-entropy`. Build compiled extensions with `CUDA_HOME=/usr/local/cuda-13.0`, `CC=/usr/bin/gcc`, `CXX=/usr/bin/g++`, `TORCH_CUDA_ARCH_LIST=9.0`. Do not install the deepspeed extra. Use the upstream installation instructions for the CUDA wheel index and compiled extensions; never install these into Lightwork.

```bash
scripts/glimmer-train-axolotl.sh prepare \
  --dataset .lightwork/datasets/glimmer-full --source-runs runs/ \
  --output .lightwork/training/axolotl-full
# Only after spend approval, on the already-provisioned GPU host with original source runs available:
scripts/glimmer-train-axolotl.sh train \
  --dataset .lightwork/datasets/glimmer-full --source-runs runs/ \
  --output .lightwork/training/axolotl-full \
  --python /path/to/axolotl-venv/bin/python \
  --axolotl /path/to/axolotl-venv/bin/axolotl --approve-paid-execution
```

Preparation writes actual JSON-as-YAML configuration, provenance and hashes, and changes the image representation to inline `{"type":"image","path":...}` inside each message (not a top-level images column). The base revision is pinned. Settings: frozen four-bit base, bf16 compute, language-only LoRA r=16/alpha=32/dropout=.05, lr=2e-4, cosine, microbatch 1, accumulation 16, two epochs (one for smoke), context 8192, `sample_packing:false`, `eval_sample_packing:false`. Vision tower/projector stay frozen. The external session holdout is not used for fitting or model selection.

Training re-audits sources and prepared hashes, runs the real processor on every train row and refuses sequences over 8192 instead of truncating. `axolotl preprocess` alone would not exercise the vision path because `skip_prepare_dataset:true`. W&B is disabled, GPU visibility is one device. The script records actual stdout/stderr, return code and final adapter artifact hashes. CUDA availability, processor support and training remain untested on this machine. Convert the resulting PEFT adapter with the same converter wrapper; there is no automatic paid fallback.

## Base versus tuned acceptance

Configure the shared chat backend to the loopback Glimmer server, with non-Meta development opt-in **off**. Base and tuned use identical dataset, prompts, sampling, crop preprocessing and output budgets. Start from the same cold/warm-cache policy; do not run competing jobs on the Mac. Switch only the adapter, and retain its hash with the tuned artifact.

```bash
# --session-id is repeatable and must match the dataset's holdout sessions exactly.
uv run lightwork eval glimmer runs/ --output .lightwork/eval/base --tag base \
  --session-id home-20261031-evening --session-id lab-20261102
# Restart the same base server with the recorded LoRA, then:
uv run lightwork eval glimmer runs/ --output .lightwork/eval/tuned --tag tuned \
  --session-id home-20261031-evening --session-id lab-20261102
uv run lightwork eval compare .lightwork/eval/base .lightwork/eval/tuned
```

These commands re-judge reviewed observations in the held-out sessions, including any additional reviewed hard cases not selected for training. Freeze the label/run set before the first call; never add/remove cases between base and tuned. N's immutable evaluation manifest hashes predictions and raw calls; `lightwork.eval.metrics.compare_saved(base,tuned)` provides the programmatic interface. Reject unpaired keys or differing expected labels; report absent predictions rather than excluding them from a headline denominator.

Full acceptance requires all of the following, measured on the same held-out observations:

| Criterion | Required | Current Lightwork result |
|---|---|---|
| Gate agreement | Tuned ≥ base; report exact paired McNemar (p<0.1 for a claimed significant improvement) | Not measured |
| Strict Verdict JSON | ≥98% of all attempted observations | Not measured |
| Median decode tokens | Tuned ≤70% of base; compare recorded completion counts including reasoning, disclose missing counts | Not measured |
| General sanity | No visible regression on the ten fixed prompts below, manually reviewed blind | Not measured |

Also report ground-truth and Spark agreement, truncation/refusal rates and latency p50/p95 with N/date. The synthetic SP2 baseline is not this held-out baseline. Shorter answers only matter if accuracy and JSON validity do not regress. Do not claim token reduction from character counts or omit invalid attempts.

### Fixed general-sanity prompts

Run each once against base and tuned using the same chat template, `reasoning_strength:low`, temperature 1.0/top_p .95/top_k 64/min_p 0, `max_tokens:2048`. Save the exact prompt, raw response including reasoning/usage/finish reason, served adapter identity, and hashes outside immutable run directories. A second operator (or an answer-order-blinded review) records accept/reject for correctness, instruction following and inappropriate refusal. None of these prompts belongs in training.

1. What is 17 multiplied by 23? (391.)
2. Explain why a shadow changes length when a light source moves. (Geometry of the light/object/surface.)
3. Return only valid JSON with the keys `colour` and `count`, values `blue` and `3`. (Exact keys/types.)
4. Sort these numbers ascending: 9, -2, 0, 4. (-2, 0, 4, 9.)
5. In one sentence, explain the difference between correlation and causation. (Association is not proof of cause.)
6. Translate “The red box is beside the blue cup” into French. (Preserve colours and spatial relationship.)
7. Summarise in five words: “The camera stopped because its USB cable was unplugged.” (Cause preserved; five words.)
8. A rectangle is 3 cm wide and 4 cm long. What is its area? (12 cm².)
9. Someone asks you to pretend a missing measurement was observed. What should you do? (Refuse fabrication; state missing evidence.)
10. Give two ways to make a photograph of a small object less blurry. (Two valid techniques, such as steadier camera/faster shutter/focus.)

Local script syntax/help and offline dataset tests can be exercised without credentials:

```bash
uv run pytest tests/dataset -q
bash -n scripts/glimmer-train-fireworks.sh scripts/glimmer-train-axolotl.sh scripts/glimmer-convert-adapter.sh
scripts/glimmer-train-fireworks.sh --help
scripts/glimmer-train-axolotl.sh --help
scripts/glimmer-convert-adapter.sh --help
```

A passing software suite is not a successful provider job, a working GGUF LoRA, or a passing tuned-model evaluation.
