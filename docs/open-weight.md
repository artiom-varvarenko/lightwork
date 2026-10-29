# Muse Glimmer: lineage, runtime and evidence

Lightwork uses Meta's open-weight Muse Glimmer for the same short step-verdict task as Spark, for authoring, and for offline re-judging. It is an explicitly slow backend, not the live hero gate. **No Lightwork fine-tuned weights or tuned results exist yet.** The implemented training path is in [training.md](training.md); real reviewed labels and Fireworks account access are prerequisites, not completed experiments.

## Model lineage and artifacts

[Meta's model card](https://huggingface.co/meta-models/Muse-Glimmer-30B) describes Glimmer as a dense approximately 30B-parameter model distilled from Muse Spark outputs, released under Apache-2.0. Meta's upstream training is not Lightwork training. The model/config revision used by the documented fine-tuning recipe is `a4e59da52a7bc87ae7251dd5545c0dd437c44b68`. The GGUF downloads came from [Meta's GGUF repository](https://huggingface.co/meta-models/Muse-Glimmer-30B-GGUF); their repository revision was not recorded in the spike, so the following byte hashes—not an invented revision—identify the actual served artifacts.

| Artifact | Bytes | SHA-256 |
|---|---:|---|
| `Muse-Glimmer-30B-KQuant-17GB-Q4_K_M.gguf` | 16,756,683,904 | `4cc57c0f51040a226e5a72cc47b7613f7772950e460a665f7083de89f183f60e` |
| `mmproj-Muse-Glimmer-30B-Q4_K_M.gguf` | 1,400,328,928 | `f48b452316f9b213758e8659444029b961a24a07f99a1abb2a9f88b06f7c00c6` |
| `dflash-Muse-Glimmer-30B-Q4_K_M.gguf` (not enabled) | 1,631,208,128 | `b2e808bf656086fe86bd0d0bd990f01d33e377537a07c02d45371517c8b264ef` |

The Q4_K_M release uses mixed precision (Q4_K body, higher precision for selected tensors); Lightwork did not quantize or alter these base weights. The text GGUF reports 27,854,794,240 parameters; this is not the multimodal model's total parameter count.

## Measured local runtime

The 29 Oct 2026 SP2 engineering session used an M1 Max Mac Studio, **24 GPU cores and 32 GiB RAM**, with other services left running. Runtime: `/opt/homebrew/bin/llama-server`, version 0.5.0, build **b11146**, commit **`7fe450e19`**, AppleClang 21.0.0.21000334, Darwin arm64. No wired-memory limit was changed. The following baseline command was exercised:

```bash
llama-server \
  -m "$HOME/models/muse-glimmer/Muse-Glimmer-30B-KQuant-17GB-Q4_K_M.gguf" \
  --mmproj "$HOME/models/muse-glimmer/mmproj-Muse-Glimmer-30B-Q4_K_M.gguf" \
  -a muse-glimmer -ngl 99 -c 16384 -np 1 \
  --host 127.0.0.1 --port 8080 \
  --chat-template-kwargs '{"reasoning_strength":"low"}' \
  --temp 1.0 --top-p 0.95 --top-k 64 --min-p 0 \
  --reasoning-budget -1 --metrics --no-webui --sleep-idle-seconds 1800 \
  --log-file "$HOME/models/muse-glimmer/server.log"
```

Use the existing launch/tunnel scripts rather than leaving a second process running:

```bash
ssh mac-studio 'cd ~/models/muse-glimmer && shasum -a 256 -c SHA256SUMS'
ssh mac-studio 'lightwork-scripts/glimmer-serve.sh start'
scripts/glimmer-tunnel.sh up
curl --noproxy '*' http://127.0.0.1:8080/health
# Stop after the session; do not interrupt another operator's authoring/evaluation job.
scripts/glimmer-tunnel.sh down
ssh mac-studio 'lightwork-scripts/glimmer-serve.sh stop'
```

The optional server key lives in `~/.config/lightwork/glimmer.key`; no key file existed during SP2. Keep it secret, bind loopback, and reach the server through SSH. The startup log confirmed the multimodal projector was loaded and `n_ctx_slot = 16384`; no context-capping line appeared. `/props` reported `build_info: b11146-7fe450e19`, the served model path, and `modalities: {vision:true,video:true,audio:false}`. For each new deployment preserve those fields, `/v1/models`, base/adapter hashes, the exact command and startup log. Call metadata and raw responses retain usage and timing; `timings.prompt_n` is **uncached processed tokens**, not total prompt/image tokens.

## Measured limitations, not production accuracy

The source is the private SP2 engineering record (29 Oct); private research and Discord exports are not release assets. All images in these measurements were generated engineering fixtures, not photographs or real training examples.

| Test | Observed result | Product consequence |
|---|---|---|
| G1/G2 text and streamed image | Correct arithmetic and colour/location answers; separate reasoning and final content | Basic local inference works |
| G3 structured schema | Nested `response_format`: **0/10 schema-valid**, 8/10 raw JSON | No `response_format`; prompt schema + strict validation + one repair |
| G4 reasoning cap | Budgets 256 and 0 still produced reasoning; `max_tokens:150` produced empty content with `finish_reason:length` | Do not claim a hard reasoning cap; truncation is not a negative verdict |
| G5 precise grounding | **11/30** centres within 30/1000 (required 27/30), median box IoU 0.511 | Spark/SAM or manually accepted clicks, not trusted Glimmer coordinates |
| G6 baseline latency, N=20 distinct synthetic crops | p50 **51.80 s**, p90 **66.65 s**, median decode 381 tokens; strict verdict JSON 16/20 | Offline evaluation/authoring/slow mode only |
| G6 DFlash, same N=20 | p50 **54.27 s**, p90 **66.35 s** | Disabled: median 4.78% slower, not the required 20% faster |

Baseline median decode throughput was 10.49 tokens/s. Repeating one exact image had much lower latency; cached repeats must not replace fresh-crop measurements. The G6 question also had imperfect dilution wording. These counts must **not** be advertised as real-bench accuracy or fine-tune quality. Tuned accuracy, JSON validity, token reduction and general-sanity results remain **unmeasured**.

## Product task and adaptation

Versioned instructions and output schema are in `lightwork/models/prompts/verdict.py` (`verdict-v1`). Input is a verified work-area crop plus the step/check question. Output is only:

```json
{"done":"yes","contradiction":null,"evidence":"The requested object is in its marked slot."}
```

This is an output-format example, not an inference record. `done` is `yes|no|unsure`; contradiction is null or at most 120 characters; evidence is at most 160 characters. No private chain-of-thought or long reasoning targets are used. Scene text is untrusted. Sampling is temperature 1.0, top_p 0.95, top_k 64, min_p 0; `chat_template_kwargs.reasoning_strength` is `low`. The model's own template/stop tokens remain intact. Model calls use the shared client, budget counter, provenance classification, and raw recording; failed parsing remains inconclusive, not an invented yes/no.

Only work-area crops may be stored or sent. Operators must review privacy before collection/upload; hash verification cannot detect a face or private text. Real runs are immutable after close (except `verify.json`). Ground truth comes from trial scripts or an operator, with an explicit review, not copied model answers. Dataset selection requires real Meta-only run audit and deterministic verification, then agreement with the exact recorded verdict and relevant required SAM checks. Test-only generated datasets cannot pass release audit or training upload.

## Evaluation and the open-weight ruling

Use the held-out physical sessions and the exact comparison commands in [training.md](training.md). Report agreement against reviewed ground truth, recorded Spark and the gate separately; raw/strict JSON validity, truncation, median decode tokens, latency and exact paired McNemar are not interchangeable. Do not compare different observations or silently remove model failures.

The 29 Oct organiser response described the mixed hosted/open-weight path as “a little borderline”, but said providing the “full training methodology” and incorporating Muse Voice should be fine (recorded in the internal Glimmer brief and project ruling notes). This is not a new ruling or a guarantee of bonus eligibility. Muse Voice Transcribe remains the voice-input model. Lightwork's decision is to fine-tune Glimmer regardless; no staff follow-up answer or completed fine-tune is claimed here.
