# Tea optical engineering notes — not filmed

Original Lightwork notes and trial scripts, 2026. CC BY 4.0, https://creativecommons.org/licenses/by/4.0/. This pack is a home engineering check only. It is not the submission's lab procedure, not a filmed use-case snippet, and not proof of calibrated laboratory concentration measurement.

## Materials, layout and preparation

Bring eight identical clear straight-sided flat-bottom glasses/cups of at least 25 mL capacity, fully cooled filtered black tea or coffee, water, needle-free 10 mL dosing syringe, waste cup, tissues, white matte backing and opaque cards. Do not use hot liquids, chipped glass or unequal vessel shapes. Keep liquids away from electrical equipment; do not drink the used liquids.

Register a 2×4 `glasses` grid with A1–A4 at the camera-back and B1–B4 at the front. Corner centres are A1,A4,B4,B1. Logical sequence positions 1–7 map to A1,A2,A3,A4,B1,B2,B3; B4 is the blank. This nondegenerate arrangement permits a real four-corner homography. Preserve transfer step IDs from the source but check actual arrow/cell references, especially `transfer-a4-a5` meaning A4→B1. Channel is blue (`b`), not the red used for blue dye. Mark vessel labels beside, not over, the liquid ROIs.

Lock exposure, white balance, focus and geometry; reduce specular reflection and record actual blank references/flat field when available. Capture with verified measurement choreography, not with the guidance overlay colouring the glass. Choose stock strength from real preliminary readings; no numerical separability is assumed from the recipe.

## Exact collection sequence

1. Obtain a real generated graph, semantic validation and human review of all eleven source steps, two-row mappings, channel=b, equal-OD transfer ratio and final-only series. Do not use a one-row graph or uncalibrated synthetic layout.
2. Run `uv run lightwork layout procedures/tea-dilution/stepgraph.json --manual --work-area X Y W H --mode normal` with actual crop coordinates for screen-only operation. Manually accept glass centres and material zones, save real layout/calibration, and check the cell order visually.
3. Choose a physical session ID such as `home-tea-20261030-pm` and retain it across all runs from that same session. Run `uv run lightwork collect --pack procedures/tea-dilution --layout .lightwork/layouts/tea-dilution.json --calib .lightwork/calib/tea.json --session-id home-tea-20261030-pm --seed 42 --backend meta`, substituting actual files. A real Glimmer alternative needs an explicitly reviewed supported graph; it must not silently replace SAM.
4. Open `/screen`, press Start and read the randomized card. Empty/rinse and reset glasses before each independent trial; prepare exactly the required prior transfers. Do not reuse a contaminated blank or accumulate wrong transfers across cards. Record pre-transfer references and activity when the question concerns a change. Keep expected answers outside the crop.
5. Execute the script, clear hands and request Capture (`trial_capture`). Review the actual stored measurement crop, card/question and script ground truth. Reject (`trial_reject`) any wrong cell, uncertain volume, unresolved setup mismatch or different visibility condition. Restage; never accept a label because a model agrees with it. If the required contrast cannot be established for a yes/no card, do not accept it as confident data.
6. Accept (`trial_accept`) only after operator review, then Next (`trial_next`). Covered-destination or covered-blank cards require unsure even when the operator knows what is underneath. Keep numerical OD measurements separate from visual verdict labels.
7. Stop with q/Ctrl-C; use the printed closed path in `uv run lightwork verify runs/<actual-run-id>` and `uv run lightwork audit runs/<actual-run-id>`. Do not edit closed runs. Use `uv run lightwork eval gate runs` and `uv run lightwork report runs` to report actual collected evidence.

## Quantitative engineering checks

For equal-OD testing, perform five independent fresh transfers with before/after frames. Record source-before, source-after, target-after and blank-corrected OD. Report the actual number with OD(target)/OD(source-after) in 0.75–1.25 and source roughly halved. A halved OD is not a constant 0.3 OD decrease. Exclude specular/saturated samples transparently; do not silently substitute a successful frame.

After a complete fresh series and final B3 discard, obtain ten hands-out measurement keyframes over two minutes. For each glass report mean and SD of blue-channel OD. Count adjacent pairs separated by more than three SD, state which SD convention is used, and test ratios 0.35–0.65 only above OD 0.03. Below that floor report monotonic-with-noise behaviour rather than precise ratios. Seven tea vessels give six neighbouring pairs: six separable pairs is a target, not a measured claim. If fewer resolve, publish the measured limit; change stock/vessel/lighting in a new documented session rather than relabelling the old data.

Perform at least three independent planted-skip trials on fresh preparations and record caught/missed outcomes. The twelve TrialCards also cover clean, omitted, wrong-cell and hidden-evidence cases; they are staged scripts, not completed measurements. Default wrong-target handling is stop/reset. An exclude-and-continue recovery variant requires a separately reviewed graph and actual recording.

Pending: vessel availability, optical validation/noise, real layout/calibration, reviewed collection and measured transfer/series results. Do not film this pack as a submission use case or claim success from authored source text. No fabricated lab or model evidence accompanies these notes.
