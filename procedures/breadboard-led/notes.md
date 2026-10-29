# Breadboard snippet: operator and safety notes

Original Lightwork notes and trial cards, 2026. CC BY 4.0, https://creativecommons.org/licenses/by/4.0/. No external circuit manual is reproduced.

## Bring and inspect

Bring a known standard solderless breadboard with labelled rows/columns, a 5 mm red LED with known polarity marking, verified 1 kΩ resistor rated at least 0.25 W, insulated jumper, switched two-AA holder with insulated leads and two matched intact AA cells. Use a matte backing, removable tape and opaque nonconductive card. A meter may verify continuity/resistance only while the circuit is disconnected; do not connect any external power supply. No mains, adapters, lithium cells or soldering are involved.

Operator verifies the breadboard's actual connected strips and the resistor's value; the distant camera cannot certify either. Check LED markings directly; reused/clipped lead length is unreliable. Keep the supply OFF, remove batteries and unplug supply leads before every change. Never energise the deliberately wrong/missing-part trials. Stop and disconnect for warmth, smell, leakage or unexpected behaviour. Do not bypass the resistor or increase voltage to improve visibility.

Fix the board at camera-top row 1, a–e left. Put readable row labels outside the wiring area and manually register the source's zones. The intended series path is +3 V at c10 → 1 kΩ resistor a10–a15 → LED anode e15/cathode e20 → jumper a20–a25 → negative at c25. The rails are unused. Use close visual inspection because individual holes may be too small for projector resolution. Keep hands, faces, unrelated wiring and room details out of the saved crop.

## Exact collection sequence

1. Obtain and inspect the real generated graph and its semantic validation. Preserve seven source IDs. `polarity-review` and `power-off` must be operator confirmations logged unverified, never visually certified safety passes. A powered trial is prohibited until manual review of the complete circuit.
2. Use `uv run lightwork layout procedures/breadboard-led/stepgraph.json --manual --work-area X Y W H --mode normal` with the measured crop for screen-only setup; inspect/accept row zones. Save the real layout and calibration. Automatic assistance, if used, does not replace pin-level operator inspection.
3. Choose one physical session ID, for example `home-led-20261101-am`, reused for all runs in that session. Start `uv run lightwork collect --pack procedures/breadboard-led --layout .lightwork/layouts/breadboard-led.json --calib .lightwork/calib/led.json --session-id home-led-20261101-am --seed 42 --backend meta`, replacing paths with actual accepted files. A supported Glimmer graph must be explicitly reviewed if used; Glimmer does not provide SAM.
4. On `/screen`, press Start and read the randomized card. Before every card, remove batteries and disconnect supply leads, then stage its exact precondition. `led-clean-light` is the ONLY trial that deliberately powers the inspected correct circuit. For it, first perform all manual checks, connect only the specified battery source, briefly illuminate, then switch off and remove batteries after capture. Every missing/wrong/hard case stays unpowered.
5. Clear hands, Capture (`trial_capture`) and inspect the stored verified crop with row labels and required markings. Reject (`trial_reject`) if the script was not performed, polarity cannot be seen on a yes/no card, or the exact endpoint is ambiguous. Do not alter wiring while powered to improve the view. Restage unpowered and recapture.
6. Accept (`trial_accept`) only after the actual crop and performed script support the expected answer, then Next (`trial_next`). A card's expected yes does not establish a real electrical test succeeded. Model output is never ground truth. Hidden pins/LED use unsure, and operator safety confirmations remain separate from image labels.
7. Stop with q/Ctrl-C and leave the hardware disconnected. Run `uv run lightwork verify runs/<actual-run-id>` and `uv run lightwork audit runs/<actual-run-id>` on the printed closed-run path. Do not modify closed labels/media. Use `uv run lightwork eval gate runs` and `uv run lightwork report runs` for actual outcomes.

## Real runs and pending gates

The twelve cards cover four clean states, three missing components, three wrong placements/polarity and two hidden-evidence refusals. Keep exact trial counts and false alarms; do not count a safety stop as a visual pass. If LED polarity is too small to read reliably, report that limit and do not manufacture a confident label from invisible markings.

For the clean continuous demonstration, complete all seven source steps with explicit manual polarity review and switch-off. For a caught-mistake demonstration, stage the missing resistor or wrong row while unpowered, show the actual detection/recovery, then restore and manually inspect the full circuit before proceeding. Never power a deliberately unsafe mistake for the video. Both actual runs must independently verify with zero diffs.

Pending operator prerequisites: actual part identification, continuity/resistance/polarity inspection, readable close-up framing, accepted layout/calibration, power-safe physical staging and review, genuine Meta access or a supported explicitly reviewed Glimmer graph, and clean/caught-mistake recordings. No hardware safety certification, model accuracy or real recording is claimed here.
