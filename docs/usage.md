# Operator workflow

[Back to README](../README.md). These are operating instructions, not evidence that a physical session has already passed. Shell variables such as `CAMERA`, `LAYOUT`, `CALIB`, `RUN`, `SESSION_ID` and `HELDOUT_SESSION` mean operator-selected paths/identifiers that actually exist. Do not copy test calibration or labels into a real run.

## Install and offline replay

Install [uv](https://docs.astral.sh/uv/getting-started/installation/), then from the repository root:

```sh
uv sync --frozen
uv run lightwork --help
uv run lightwork verify tests/fixtures/runs/synthetic-kit
uv run lightwork replay tests/fixtures/runs/synthetic-kit --no-server
```

Python must be 3.12. `uv.lock` pins the application environment. The optional `live` extra is not required for replay, verify, audit, export or evaluation of existing recordings. Installing dependencies can use the network; subsequent keyless replay does not contact model APIs.

For the browser, omit `--no-server` and open `http://127.0.0.1:8000/screen`. `--speed 2` changes playback speed, not decisions. The server waits for the screen, plays once and retains the last frame until Ctrl-C. Stop it before another command needs the same port.

The fixture is explicitly synthetic. It demonstrates deterministic replay, not genuine model output, physical sensing or benchmark accuracy. To inspect authentic evidence, select a real run and use:

```sh
uv run lightwork verify "$RUN"
uv run lightwork audit "$RUN"
uv run lightwork replay "$RUN"
```

An empty/missing real `runs/` tree is not release-ready. Do not replace `audit` with verification alone.

For a local static viewer, export the fixture with explicit test permission:

```sh
uv run lightwork export tests/fixtures/runs/synthetic-kit \
  --output /tmp/lightwork-viewer --allow-synthetic
python3 -m http.server 8080 --bind 127.0.0.1 --directory /tmp/lightwork-viewer
```

Use a fresh output directory and open `http://127.0.0.1:8080/`. The viewer reuses the scene renderer and plays snapshots recomputed during export; it does not independently run Python verification in the browser. The synthetic banner must remain visible. For real reviewed data use `uv run lightwork export runs/ --output site` without test permission. [Public-only publishing](release.md#4-static-viewer-then-public-only-pages) remains operator-controlled; no hosted URL is claimed.

## Credentials, billing and local state

[.env.example](../.env.example) describes the supported settings. Never paste secrets into source, terminal captures or browser URLs. Environment variables override an uncommitted `.env`.

- `MODEL_API_KEY`: hosted Meta authoring, Spark, SAM and Voice access.
- `META_API_BASE`: hosted endpoint override; provenance rules still apply.
- `LW_BACKEND=meta|glimmer`: default chat/verdict backend; commands can override it.
- `LW_CHAT_BASE_URL`, `LW_CHAT_MODEL`, `LW_CHAT_API_KEY`: Glimmer's OpenAI-compatible endpoint, model identity and optional server key. Follow [the Glimmer runbook](open-weight.md), including server provenance and SSH tunnelling. Glimmer is not synonymous with a fully offline workflow: SAM and Muse Voice remain separate hosted capabilities.
- `LW_BUDGET_USD`: local persisted model-client spend guard. It reserves estimated hosted-call costs before a request; missing billed usage stays unresolved, not zero. Do not delete the counter to evade the limit. This is not an account-wide cap on other clients or provider training/deployment charges.
- `LW_DEV_ALLOW_NON_META=1`: development stand-ins only. Their recordings belong in `runs-dev/`, are marked `dev-standin`, and cannot be published as Meta evidence. Never use a stand-in for SAM or Voice. Do not enable it for submission collection.

`.lightwork/` holds local layouts/calibration, downloaded hand assets, spend state and development cache. Cached fixture calls cannot be used for live recording. Keep state and secrets out of the release.

## Rig and calibration

Fix the camera above the accepted work area. Use matte backing, stable lighting and locked exposure, white balance and focus. Linux by-id devices are preferable to unstable camera indices. `doctor` reports devices and environment; it is not a hardware acceptance test.

```sh
uv run lightwork doctor
uv sync --frozen --extra live
uv run lightwork calibrate --camera "$CAMERA" --output "$CALIB"
```

The calibration command starts its own local server and waits for `/projector`. For the Tank, enable USB debugging and projector mode, focus, disable keystone correction, then in another terminal:

```sh
uv run lightwork tank up
```

Open/tap Start on the projector page if a programmatic tap does not grant fullscreen. `tank up` maintains the USB reverse connection; `tank status` inspects it and `tank down` removes it. The page's size/orientation must stay fixed after calibration. An HDMI projector can display `http://127.0.0.1:8000/projector` instead.

`calibrate check --camera "$CAMERA" --output "$CALIB"` checks a saved calibration; it needs the documented physical reference/card and operator alignment inspection. Nominal scale is not measured millimetre accuracy. Recalibrate after movement or a page-size change. Do not run calibration alongside live/replay on the same port.

Screen overlay does not need a connected projector during live operation, but the current `live` CLI still requires a valid calibration file matching its actual camera device/mode/control readback. There is no advertised screen-only calibration bootstrap command; do not fabricate a calibrated rig by copying the fixture's file.

## Author a procedure

The repository includes original, attributed [kit instructions](../procedures/kit-return-check/source.md). Authoring is a real model call and records its response; it is not offline:

```sh
uv run lightwork author procedures/kit-return-check/source.md \
  --slug kit-return-check --backend meta --licence 'CC BY 4.0' \
  --attribution 'Original procedure text and trial design by the Lightwork contributors, 2026' \
  --privacy-confirmed
```

On success this produces a validated `procedures/kit-return-check/stepgraph.json` with immutable authoring evidence. Do not overwrite an existing pack's evidence. `--output NEW_PACK_DIRECTORY` selects a new destination. `--backend glimmer` uses the configured Glimmer endpoint instead of Wonder; `--timeout SECONDS` explicitly changes its per-call deadline. A longer deadline is not a guarantee of valid authoring.

Current authoring evidence is limited: a tiny Glimmer authoring smoke succeeded, but full-source Glimmer attempts for the supplied packs timed out at the default deadline. Those attempts do not establish runnable authored packs. Review the preserved attempt evidence and explicitly choose a backend/deadline before retrying; local Glimmer service availability is an operator prerequisite, not an always-running service.

The source argument accepts text/Markdown/PDF files or readable HTTP(S) URLs; PDF extraction needs the external `pdftotext` tool. Confirm privacy and redistribution rights before either fetching or uploading a source. CC BY sources need attribution, source URL and a description of changes. Review the resulting checks, references, thresholds and recovery; a valid schema is not a physically safe procedure.

To preserve a reviewed human edit separately from the model result:

```sh
uv run lightwork author --slug kit-return-check --human-edit "$EDITED_GRAPH"
```

This validates the replacement and saves the diff. It does not pretend the model authored the edited content.

## Accept a cropped layout

```sh
uv run lightwork layout procedures/kit-return-check/stepgraph.json \
  --camera "$CAMERA" --manual --mode normal --output "$LAYOUT"
```

Open `/screen`. Before any preview, select/confirm a work-area crop on the coordinate canvas; alternatively pass `--work-area X Y W H` with actual camera-pixel values. Capture the work area, pick each object, draw the accepted polygons/rectangles, then Accept. For a grid, click the four **extreme cell centres** in TL, TR, BR, BL order, not the vessel's outer corners.

`--mode normal` is only for screen-only operation with no projected overlay. With a projector, use `--mode dark` (the default) or `--mode grey` and wait for its measurement ACK. Omit `--manual` to make explicit Spark grounding/SAM mask suggestions available; these require Meta access. Glimmer is not used for precise layout grounding. The operator must accept geometry either way. Nothing claims an unmeasured setup duration.

## Live operation

After authoring, layout acceptance and matching calibration:

```sh
uv run lightwork live --pack procedures/kit-return-check \
  --layout "$LAYOUT" --calib "$CALIB" --source "$CAMERA" --backend meta
```

Open `/screen` and press Start to enable speech/microphone permissions. `/projector` is optional. With a connected projector, capture uses the calibrated grey/dark ACK sequence; losing that page does not silently downgrade to an illuminated normal frame. Without a projector, screen-only capture uses normal mode.

Terminal controls: Space = confirm, `r` = repeat, `b` = back, `q` = stop. Stop/SIGINT/SIGTERM drains outstanding calls before closing the run. Confirmation is not fabricated model evidence: unresolved steps remain explicitly `unverified`.

Options:

- `--no-hands`: explicit motion-only advisory mode, not hand evidence.
- `--no-voice`: disable Muse Voice input.
- `--quiet`: suppress speech and voice commands.
- `--voice-dry-run`: exercise the voice HUD/pipeline without letting recognized commands reach the reducer; not a real voice-command acceptance result.
- `--source`: camera by-id path/index, video or image directory. **A file source still calls the real selected backend.** It is not an offline-test switch.
- `--code-commit`: record the actual source revision. The current CLI marks the working tree dirty; a supplied revision string does not prove a clean release.

`--backend glimmer` requires a deliberately authored graph without SAM checks. A SAM-based kit graph is not automatically converted to weaker checks. Follow the open-weight docs for supported tasks and limits.

## Collect ground truth, evaluate and build a dataset

Use real physical session IDs; all runs from the same setup/session must share the ID. With an accepted pack and rig:

```sh
uv run lightwork collect --pack procedures/kit-return-check \
  --layout "$LAYOUT" --calib "$CALIB" --source "$CAMERA" --backend meta \
  --session-id "$SESSION_ID" --seed 42
```

The default `trials.json` sits beside the graph; `--trials PATH` chooses another validated card array. In `/screen`, stage the randomized card, Capture, inspect the crop and **script ground truth**, then Accept or Reject and Next. Labels are not reviewed until accepted; do not label from the model's prediction. `collection.json` preserves cards/seed and `labels.jsonl` preserves reviewed labels before the run is sealed.

A card's step must actually be evaluated by the live reducer on that captured frame to supply gate evidence. Collection never resets or fabricates reducer state. An out-of-active-step card can provide a labelled crop for re-judging, but produces missing gate evidence, not a guessed match.

Offline real-run analysis:

```sh
uv run lightwork eval gate runs/
uv run lightwork eval latency runs/
uv run lightwork report runs/
```

Unreviewed labels, absent durations and missing evidence are reported separately. In particular, realtime voice commands without recorded durations do not create latency samples. `--include-synthetic` is an explicitly labelled engineering-test mode, never a way to fill the real evidence table.

Glimmer re-judging makes real calls and writes a **new** immutable evaluation directory:

```sh
uv run lightwork eval glimmer runs/ --output "$BASE_EVALUATION" \
  --tag base --session-id "$HELDOUT_SESSION"
uv run lightwork report runs/ --evaluation "$BASE_EVALUATION"
```

Only after a genuine adapter exists, run the same held-out sessions with that adapter loaded, use `--tag tuned` and a different new output, and compare with `uv run lightwork eval compare "$BASE_EVALUATION" "$TUNED_EVALUATION"`. A tag is not proof of training or improvement.

Dataset construction requires real reviewed eligible evidence, independently held-out physical sessions, and rights to the photos/labels:

```sh
uv run lightwork dataset build runs/ --output "$NEW_DATASET" --mode smoke \
  --holdout-session "$HELDOUT_SESSION" --seed 0 \
  --licence "$IMAGE_LABEL_LICENCE" --attribution "$IMAGE_LABEL_ATTRIBUTION"
uv run lightwork dataset audit "$NEW_DATASET" --source-runs runs/
```

Use `--mode full` only with sufficient eligible real data. Procedure licensing does not grant image rights. See [training methodology and prerequisites](training.md); synthetic-test options cannot pass a release audit and are not training evidence.

## Privacy before and after capture

Only the accepted work-area crop belongs in previews, stored images and model requests. The camera may still see more locally: check the actual cropped image, not just the intended rectangle. Inspect reflective surfaces, readable labels, backgrounds at crop edges and any personal/confidential material. Voice sends microphone audio to Muse Voice; an image crop does not protect nearby conversations. Authoring sources and model outputs also require privacy review.

The local server binds loopback. Do not expose it or a model endpoint publicly with an operator key. Keep raw video separate from committed run bundles; publish a reviewed take only with permission. Closed recordings are immutable: reject/re-record an unsafe run rather than quietly editing its media and retaining old hashes. [Release review](release.md) covers the clean public copy and operator approvals.
