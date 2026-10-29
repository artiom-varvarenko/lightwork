# Release runbook

[Back to README](../README.md). This is an operator-controlled release procedure, **not a publication record**. Authentic hardware recordings, filmed evidence, a clean public repository, hosted Pages acceptance and a fresh public-clone test are still external prerequisites. Synthetic fixtures cannot satisfy them.

## 1. Select authentic evidence

Use completed, immutable Meta-only runs recorded with the intended code, prompts, graph, layout and calibration. Include clean behavior, a caught mistake with recovery and an honest refusal/hard case. Keep disagreements and explicit `unverified` outcomes visible rather than selecting only plausible frames. Record the actual revision; a manually supplied commit identifier alone does not prove a clean tree.

Before any publication, the operator must review:

- Every keyframe and preview, including edge pixels, reflections and readable text; only the consented work area is acceptable.
- Raw requests/responses, authoring sources, transcripts, labels, calibration/device metadata and traces for secrets, personal information or confidential material.
- Every file included in a static export. An export includes raw evidence, not just the visible canvas.
- Rights to procedure text, photos/labels, any recorded people/voices, video and music; keep the attribution and any required consent.
- That the matching video/run identifiers are recorded and replay, synthetic data and speed-ups are labelled honestly.

Do not alter sealed recording media to pass privacy or verification checks. Exclude an unsafe run and re-record. Hashes establish integrity, not consent or authenticity of the scene.

From the selected release checkout:

```sh
uv sync --frozen
uv run lightwork verify runs/
uv run lightwork audit runs/
uv run lightwork report runs/
```

Require verification with no diffs and successful audit for every published run. Refresh README quantitative tables only from the generated report, retaining sample counts, dates and regeneration commands. Missing measurements must stay unavailable. Unit tests and synthetic counts are not real-model benchmark evidence. No adapter or tuned-model claim is publishable without real training artifacts and held-out evaluation.

## 2. Build a separate clean public copy

**Never flip the development repository's visibility to public.** Deleting private files in a normal commit does not remove their history. The release owner uses the maintenance checkout's `scripts/prepare-public.sh NEW_DIRECTORY`; it does not push, rename repositories or publish anything. The output directory must not exist.

The maintenance script:

1. Checks all committed history for ineligible run manifests and forbidden development-run, synthetic-run, local-state and secret-environment paths.
2. Requires authentic committed run evidence, not just the synthetic test fixture.
3. Clones a separate copy, removes the development remote and filters private planning/research material and the maintenance-only preparation script out of history.
4. Checks the filtered history for private material and possible credentials; its pattern scan is not a substitute for manual review.
5. Installs the frozen environment and runs keyless verification and provenance audit on the clean committed run tree.

It is expected to refuse the current no-real-runs state. Do not weaken it or relabel test data to make it print `ok`. No claim is made here that the preparation gate has passed.

Only after reviewing the clean copy may the operator approve creation/push of a **new public repository**. Public naming, remotes, repository settings and visibility are operator decisions. The existing private repository remains private. This runbook deliberately supplies no push/rename command that could expose its history.

## 3. Fresh-clone test on the Mac

This acceptance step needs an actual approved public URL and authentic runs; it has not been executed for a release. Use a clean Mac Studio checkout/user environment, Python 3.12 and uv, with no `.env`, API keys or pre-existing project state. Clone the **public copy**, not the development repository. In that fresh directory:

```sh
uv sync --frozen
uv run lightwork verify runs/
uv run lightwork audit runs/
uv run lightwork replay "$RUN"
```

Dependency installation may require a network. Then disable outbound networking and confirm that verification and replay still work; browser loopback must remain available. Open `http://127.0.0.1:8000/screen`, inspect the recording banner, overlays, evidence chips and trace, and confirm the selected genuine run reproduces without diffs. Record the exact machine, revision, commands and observed results; do not infer macOS acceptance from Linux tests. No optional hand-tracking dependency is needed for replay.

## 4. Static viewer, then public-only Pages

Export is offline and recomputes snapshots with the Python reducer before packaging. The browser **plays those export-time snapshots**; it is not independently rerunning Python verification. The displayed verify status belongs to the exported verification report.

For authentic reviewed runs:

```sh
uv run lightwork export runs/ --output site
python3 -m http.server 8080 --bind 127.0.0.1 --directory site
```

Open `http://127.0.0.1:8080/`. The bundle includes an index, shared scene renderer, bundle metadata and each run's original evidence plus fresh `replay.json` and `verify.json`. Review its raw media/calls as well as every run's visible overlay, chips, trace and verification status. Stop the static server when done.

For **local engineering inspection only**:

```sh
uv run lightwork export tests/fixtures/runs/synthetic-kit \
  --output /tmp/lightwork-viewer --allow-synthetic
python3 -m http.server 8080 --bind 127.0.0.1 --directory /tmp/lightwork-viewer
```

Use a fresh output directory. This is synthetic test data, must retain its banner and is not eligible for public submission hosting.

The publish script requires an explicit public repository:

```sh
scripts/publish-viewer.sh --repo "$PUBLIC_REPO" --site site --dry-run
```

`PUBLIC_REPO` is the approved `OWNER/REPOSITORY`, not the private development origin. The script checks public visibility through GitHub, bundle hashes, run verification/provenance, and the site size limit. It refuses synthetic/dev data and bundles exceeding its size cap. Only with explicit operator approval, replacing `--dry-run` by `--publish` publishes the clean bundle to `gh-pages`; that operation replaces that branch's viewer content. Do not run publication merely to test documentation.

After Pages is configured, an operator must open the real URL logged out/in a private browser window and replay every published run. Confirm overlays, chips, trace and verify status, with no credentials, private paths or live API endpoint. Only then add that observed URL to the README. Local browser success is not hosted acceptance, and no hosted URL exists to advertise now.

## 5. Video, post and submission approval

The operator must approve the final video, public repository, viewer and submission text. Show real behavior and disclose limitations; do not present a screen replay as live sensing. Check video duration, captions, any speed-up labels and all public links logged out. Tie the uncut take to its run directory.

Do not post to X, upload/publish a video or submit/edit the form without operator approval. Include the actual model roles and honest open-weight/training status, not planned results. Freeze the public artifact at submission and retain the exact submitted revision and URLs. Outstanding real-evidence gates cannot be closed by writing this runbook.
