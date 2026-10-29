# Lightwork

**Written workbench procedures become guidance on the real objects, with camera evidence checked before a step is committed.** A laptop does the work; a webcam watches the bench; a projector phone or screen overlay shows the next action.

## Try it without an API key

Requires [uv](https://docs.astral.sh/uv/getting-started/installation/) and Python **3.12**. From this checkout:

```sh
uv sync --frozen
uv run lightwork verify tests/fixtures/runs/synthetic-kit
uv run lightwork replay tests/fixtures/runs/synthetic-kit
```

Open **http://127.0.0.1:8000/screen**; playback begins when the page connects. Ctrl-C stops the server. Installation may download packages; replay and verification need no runtime API key, model server, webcam or outbound network connection.

**This bundled example is SYNTHETIC TEST DATA, not a recording of Meta inference or evidence of real-world accuracy.** Its visible replay banner identifies it. A successful `verify` establishes reproducibility of that fixture, not that a model or physical task succeeded. `audit` deliberately rejects synthetic data as release evidence.

**Demo/evidence status:** the bench demo has not been filmed. Authentic Glimmer call fixtures exist, but there are no accepted real hardware runs, reviewed trial dataset, measured tuned-model results or published viewer URL. No video/GIF or hosted-demo claim is made here. This is an in-progress Indie entry for the Meta Global AI Developer Hackathon, “Push the Frontier”.

For an authentic recording, use the same commands with a privacy-reviewed, Meta-only run directory:

```sh
# RUN must name an actual recorded run, not the synthetic fixture.
uv run lightwork verify "$RUN"
uv run lightwork audit "$RUN"
uv run lightwork replay "$RUN"
```

`RUN` is an operator-selected path. A release must include authentic runs under `runs/`; the absence of these runs is a release blocker, not a reason to relabel fixtures. [Usage](docs/usage.md) covers local static export and [release gates](docs/release.md) cover publication. No public hosting is required for local replay.

## Live setup

Live operation needs accepted procedure/layout/calibration files and either Meta API access or a configured Glimmer server. Judges do **not** need a key. The projector is optional for screen-overlay operation; camera position and the work-area crop are not optional.

```sh
uv run lightwork doctor
# Optional local hand-tracking package; not needed for replay:
uv sync --frozen --extra live
```

Put credentials only in the environment or an uncommitted `.env`; [.env.example](.env.example) lists configuration names. See [the operator workflow](docs/usage.md) for exact `author`, `layout`, `calibrate`, `live`, `collect`, evaluation and dataset commands. Linux camera controls are supported; other camera platforms can report uncontrolled settings. Hardware acceptance and a fresh macOS replay installation remain unmeasured here; this is not a cross-platform certification.

The intended benches are kit return-checking, small LEGO and breadboard assemblies, and water/food-dye serial dilution. A pack must fit the supported check vocabulary; Lightwork does not understand arbitrary physical procedures or guarantee hidden object properties.

## What replay recomputes

| Part | Replay behavior |
|---|---|
| Encoded keyframes, calibration, layout and procedure | Loaded from the recording and integrity-checked |
| Grid geometry, optical measurements, evidence checks, step transitions, recovery and scene hashes | Recomputed by the same deterministic reducer |
| Model requests | Re-derived and fingerprint-matched against recorded calls |
| Model responses | Recorded raw JSON/SSE is parsed again; no fresh inference or fabricated replacement |
| Motion/settle, hand events, user commands and voice transcripts | Recorded inputs, not fresh perception or transcription |

`verify` checks the event hash chain, manifest hashes and derived stream. Exit status is `0` for a match, `1` for a diff/check failure, and `2` for a usage/environment error. It detects changes; it does not independently authenticate the physical scene or the service that produced a response. `audit` adds provenance checks.

### Architecture and invariants

```text
camera -> crop + change/settle gate -> measurement keyframe
                    |                        |
              local hand helper        recorded model calls
                    |                        |
                    +------ event log -------+
                                |
                  deterministic evidence reducer
                                |
                  projector / screen / replay
```

Coordinates are locked-camera pixels. Projector calibration maps them to page pixels. Measurement capture uses an acknowledged grey/dark page and fresh frames, rather than treating projected highlights as object colour. Screen-only capture does not invent a projector ACK.

- At most **one** step is provisional. Guidance can move ahead while its reading is pending, but cannot accumulate unchecked steps.
- A `passed` step needs all required checks plus a passing Meta-model reading. A verdict does not override failed local evidence; human confirmation alone cannot produce `passed`.
- Low optical density, unreadable cells, missing evidence, timeouts and `unsure` are not fake successes. An explicit user override is logged as **`unverified`**, distinct from `passed`; confirm-only steps are also unverified.
- A contradiction can suspend the active step and apply the pack's recovery. Recovery, exclusions and skips stay in the event history.
- The local hand helper is advisory. It never supplies the sole proof of completion. Explicit motion-only operation is labelled as such.

Checks are `present`, `absent`, `moved_to`, `count`, `colour_change`, `series`, `activity_at`, `judge` and the explicit human `confirm` action. Transparent grid cells come from accepted geometry, not a claim that SAM can segment every well.

## Provenance and immutable recordings

Each run contains `events.jsonl`, `manifest.json`, the procedure, layout and calibration, cropped `frames/`, optional cropped `preview/`, raw `calls/` and a verification report. Calls preserve the redacted canonical request, response, model identity, request fingerprint and available timing/usage metadata. The manifest records code identity and file hashes.

Treat closed recordings as immutable. Do not edit a result to make verification green; fix code and re-record, or keep the incompatible run out of the release. A zero-diff replay of the wrong code/data pair is not a benchmark. A future demo must identify its matching run and label replay or sped-up footage; there is no uncut take to link yet.

Development stand-ins are **development-only**, opt-in through `LW_DEV_ALLOW_NON_META=1`, and isolated in `runs-dev/` as `dev-standin`. They are not shipped model evidence, never replace SAM or Voice, and are rejected by release audit. Cached development responses are excluded from live recording and real evidence. Submitted runtime model work uses Meta models; this does not mean every local component is a Meta model.

## Reproducible numbers

The tables below are unedited table sections from `uv run lightwork report`, run against `runs/`. **No real run evidence is available.** Counts of zero mean no eligible samples, not perfect performance. Unit-test totals and synthetic scenarios are not accuracy measurements.

### Gate against reviewed script/operator ground truth

| Metric | Count | N | Source dates | Regeneration command |
|---|---:|---:|---|---|
| caught | unavailable | 0 | unavailable | `uv run lightwork eval gate runs` |
| missed | unavailable | 0 | unavailable | `uv run lightwork eval gate runs` |
| false_alarm | unavailable | 0 | unavailable | `uv run lightwork eval gate runs` |
| correct_pass | unavailable | 0 | unavailable | `uv run lightwork eval gate runs` |
| refusal | unavailable | 0 | unavailable | `uv run lightwork eval gate runs` |
| expected_refusal | unavailable | 0 | unavailable | `uv run lightwork eval gate runs` |
| correct_refusal | unavailable | 0 | unavailable | `uv run lightwork eval gate runs` |
| missing | unavailable | 0 | unavailable | `uv run lightwork eval gate runs` |

N is reviewed labels, not attempted predictions. Missing matches are separate; refusals are not counted as caught or missed.

### Recorded call latency (all outcomes; milliseconds)

| Model | N | p50 | p95 | Source dates | Regeneration command |
|---|---:|---:|---:|---|---|
| SAM | 0 | unavailable | unavailable | unavailable | `uv run lightwork eval latency runs` |
| Spark | 0 | unavailable | unavailable | unavailable | `uv run lightwork eval latency runs` |
| Glimmer | 0 | unavailable | unavailable | unavailable | `uv run lightwork eval latency runs` |
| Voice | 0 | unavailable | unavailable | unavailable | `uv run lightwork eval latency runs` |

Percentiles use linear interpolation. Realtime voice commands without recorded durations provide no latency samples. Standalone smoke-call timings are not substitutes for recorded-run latency or gate performance. No measured calibration accuracy, task speedup, user-test benefit, cost-per-run or tuned-model improvement is claimed.

## Meta models and their jobs

This describes implemented model roles, not a claim that every role has passed a real bench evaluation.

| Model | Job |
|---|---|
| Muse Wonder (`muse-wonder-1.0`) | Author source text, URL or PDF into a validated procedure graph |
| Muse Spark (`muse-spark-1.3-contributor`, or configured full Spark) | Step verdicts, object grounding/count cross-checks, model explanation capability |
| SAM (`sam-3.1`) | Masks supporting presence, movement and count checks; layout suggestions |
| Muse Voice Transcribe (`muse-voice-transcribe-1.0`) | Spoken command input; browser speech is output, not a transcription replacement |
| Muse Glimmer | Open-weight authoring/verdict/explanation backend and offline re-judging; deliberately not used for precise layout grounding |

Glimmer is not a local SAM replacement. Current Glimmer live mode requires a graph without SAM checks and does not silently rewrite a pack. Hosted SAM/Voice still need Meta access wherever used. Long model latency and inconclusive readings remain visible; this is not “real-time AI verification”.

## Non-Meta components

| Component | Role | Licence |
|---|---|---|
| MediaPipe HandLandmarker (optional) | Local fingertips and hand presence; advisory only | Apache-2.0 code and upstream hand model card; model downloaded separately |
| OpenCV (`opencv-contrib-python`) | Capture, motion, ArUco/homography and optical readings; classic CV | Apache-2.0, with wheel third-party notices |
| Browser `speechSynthesis` | Spoken output | Browser/OS voice-provider terms; not a bundled model |
| llama.cpp | Serve Glimmer weights | MIT; weights licensed separately |
| NumPy | Arrays and numerics | BSD-3-Clause plus bundled component licences |
| FastAPI, Pydantic | HTTP application and validated schemas | MIT |
| Uvicorn, HTTPX | ASGI serving and model HTTP transport | BSD-3-Clause |
| `meta-sam-parser` | Decode Meta SAM responses (a library, not a second inference model) | SAM License, not Apache-2.0 |

[THIRD_PARTY.md](THIRD_PARTY.md) records licence sources, transitive runtime notices, procedure attribution and the distinction between code and model licences.

## Licences and open weights

Lightwork code is [Apache-2.0](LICENSE). Procedure texts carry their own licences: the original packs are CC BY 4.0; the dilution text adapts Baldwin and colleagues' protocol with attribution and listed changes. Model weights, hosted services and browser voices are not relicensed by this repository. No iFixit instructions or copied LEGO manual are included.

[Open-weight reproduction](docs/open-weight.md) documents Glimmer lineage and operation. No trained adapter or measured tuned-model gain is claimed. Dataset eligibility and training evidence must be established before describing a model as fine-tuned.

## Limitations and privacy

- The physical rig, calibration, lighting sensitivity, sustained voice operation and real staged-trial accuracy still require operator measurement. A successful synthetic replay does not close these gates.
- Reflections, occlusion, dim/uneven illumination, camera auto-controls and changing vessel geometry can invalidate colour readings. Top-down OD tracks dye amount under the flat-bottom/equal-area assumptions, not concentration or delivered volume by itself. Below the readable floor, a long dilution cannot be presented as quantitatively verified.
- Grid corner selection and camera registration matter. A bumped camera or projector requires re-checking calibration; invisible contents, polarity and volume are not proven by a plausible image.
- Fine-tuning, lab runs, filmed demos and public release are not complete. Replay reproduces recorded inputs, not a new live interaction.
- Crop before recording, previewing or sending images. Keep faces, room details, personal identifiers and confidential material outside the accepted work area. Crop-only is a boundary, not automatic anonymisation; text, reflections, audio and model responses also need review.
- Do not assume hosted inference has zero retention. Contributor-model content may be used for training; send only content you are authorised to share. Never expose a live endpoint backed by someone else's API key.

The model client's persisted billing guard reserves estimated spend before hosted model calls and preserves unresolved usage rather than treating it as free. `LW_BUDGET_USD` configures the local guard; it is not an account-wide billing cap for other processes, services or tools. See [usage and privacy](docs/usage.md) and [the release runbook](docs/release.md).
