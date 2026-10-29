# Original brick build: operator notes

Original Lightwork instructions, notes and trial design, 2026. CC BY 4.0, https://creativecommons.org/licenses/by/4.0/. No commercial LEGO instructions, set design or images were copied. LEGO is a third-party trademark, not an endorsement.

## Materials and placement

Bring an 8×8-or-larger baseplate, red/blue/yellow/green 2×2 bricks, white/black 2×4 bricks and an orange 2×6 brick; matte backing, removable tape and two opaque cards. All placed bricks have footprints at least 2×2. Seven bricks plus the baseplate make eight pieces. Check actual parts fit; colour substitutions require a reviewed source/graph/trial change before collecting, not a hidden exception. Keep small parts away from young children.

Lock the baseplate and camera. Label camera-left-to-right x=1…8 and camera-back-to-front y=1…8 beside the baseplate without covering studs. Use the source's explicit inclusive coordinates; do not guess positions from colour alone. Manually accept the target regions. A projected mark at baseplate height may drift at pillar height, so verify alignment and use the screen overlay if needed. Keep loose bricks separated on the bench and all evidence within the privacy crop.

## Exact collection sequence

1. Obtain the real authored `stepgraph.json`, inspect the nine source IDs and colour/coordinate details, and run its semantic validator. Confirm lower-brick visibility is not overclaimed after the bridge covers it.
2. Start layout with `uv run lightwork layout procedures/lego-build/stepgraph.json --manual --work-area X Y W H --mode normal` for screen-only capture; use measured crop coordinates, not sample numbers. Accept the actual zones and save layout/calibration. Automatic Spark/SAM assistance, if used, must be reviewed; real Glimmer alone is not SAM.
3. Assign a physical session ID, e.g. `home-bricks-20261031-am`, shared across every run from that session. Run `uv run lightwork collect --pack procedures/lego-build --layout .lightwork/layouts/lego-build.json --calib .lightwork/calib/bricks.json --session-id home-bricks-20261031-am --seed 42 --backend meta`. Substitute files you actually created. A Glimmer option requires a supported explicitly reviewed graph, not an implicit replacement for SAM.
4. Open `http://localhost:8000/screen`, press Start and read each randomized card. Rebuild its prerequisite state from scratch; do not retain a previous trial's misplaced brick. For moved-to evidence, expose the loose brick before placement. Keep the card and expected answer outside the crop.
5. Remove hands and request Capture (`trial_capture`). Inspect the stored verified crop, actual stud coordinates, card/question and expected script label. Reject (`trial_reject`) any wrong setup, concealed marking or ambiguous colour that does not match the card. Restage rather than accepting what the model predicted. For an unsure card, ensure the target is truly hidden as scripted. Accept (`trial_accept`) only after operator review, then Next (`trial_next`).
6. Stop with q/Ctrl-C, keep the printed run path, and run `uv run lightwork verify runs/<actual-run-id>` and `uv run lightwork audit runs/<actual-run-id>`. Do not modify a closed run. Summarize actual labels/outcomes using `uv run lightwork eval gate runs` and `uv run lightwork report runs`.

## Demonstrations and review

The twelve cards include five clean, two missing-part, three wrong-colour/orientation/position, and two refusal cases. Run all twelve with fresh preconditions. They are not a claim of twelve correct predictions. A no label means a visible scripted contradiction; an unsure label means the requested evidence is hidden, even if the operator knows the underlying build is correct.

For a continuous clean run assemble all nine steps in order. For a separate caught-mistake run, use green at the yellow rear-marker step, let the system identify the mismatch, replace it with yellow, then continue. Alternatively use blue one row forward and correct it before adding the bridge. Record the actual failure, recovery and final state, and require zero verify diffs for each real run. Do not claim an image of the completed model proves hidden earlier placements or physical attachment strength.

Pending: actual brick availability, accepted layout and height-alignment check, operator trials, clean/caught-mistake Meta recordings and measured performance. Source text and script labels are preparation, not model or hardware evidence. No runs are fabricated for this pack.
