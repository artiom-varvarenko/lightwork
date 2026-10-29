# Serial dilution across row A

Licence: CC BY 4.0, https://creativecommons.org/licenses/by/4.0/.

Adapted from: Baldwin G, Haddock-Angelli T, Beal J, Dwijayanti A, Storch M, Farny N, Telmer C, Vignoni A, Tennant R, Rutten P (2019). *Calibration Protocol – Plate Reader Fluorescence Calibration* V.3. protocols.io. https://dx.doi.org/10.17504/protocols.io.6zrhf56. Licensed CC BY 4.0 (https://creativecommons.org/licenses/by/4.0/). Changes: fluorescein → blue food dye, PBS → water, one row (A) only, camera reading instead of a plate reader, fresh tip per transfer added. These practical instructions are newly worded; no source figures are included.

## Scope and equipment assumptions

A nonclinical demonstration with water and food dye only: no biological samples, patients or diagnostic decisions. The planned equipment is a clear flat-bottom 96-well plate, P200 or P20–200 pipette, compatible fresh tips, water reservoir, diluted blue food dye and a waste cup. Equipment availability and lab access have not been surveyed or confirmed. Follow local bench and waste rules. Do not eat or drink demonstration liquids.

Keep the locked camera above a matte white backing, with the plate slightly off-nadir to reduce reflections. Plate object `plate` is an 8 by 12 grid named A1 through H12, red OD channel. Four layout corners are the centres of A1, A12, H12 and H1, not the plate's outer corners. Define `work-area`, `tips`, `water`, `dye`, and `waste` objects/zones. A1 must be at the top left as seen by the camera; inspect the embossed row/column markings rather than guessing from the symmetric outline.

## Steps (preserve these step IDs)

1. `setup`: Put the empty plate, tips, water, dye and waste in their accepted zones. Check their presence and ask the Meta reader whether A1 is top-left. Operator confirms orientation. Capture a blank reference before adding dye.
2. `add-water`: Put 100 µL water in each well A2 through A12, exactly once; leave A1 empty. One clean tip may dispense this water. Record activity at all eleven wells and ask for a no-contradiction Meta reading. Clear water is not reliably visible: activity is not proof of delivered volume, so operator checks the pipette and tally. If a dispense is uncertain, reset the row rather than adding an unknown second dose.
3. `add-stock`: Fit a fresh tip and place 200 µL blue stock in A1. No other well receives stock. Require A1 darker by at least 0.5 OD plus a Meta reading. Pre-test stock to obtain initial red-channel OD 0.8–1.2; reset if saturated or too faint.
4. `transfer-a1-a2`: Mix A1 by pipetting gently three times, then transfer 100 µL A1 to A2. Avoid bubbles. Eject the tip.
5. `transfer-a2-a3`: Fresh tip; mix A2 three times; transfer 100 µL A2 to A3; eject tip.
6. `transfer-a3-a4`: Fresh tip; mix A3 three times; transfer 100 µL A3 to A4; eject tip.
7. `transfer-a4-a5`: Fresh tip; mix A4 three times; transfer 100 µL A4 to A5; eject tip.
8. `transfer-a5-a6`: Fresh tip; mix A5 three times; transfer 100 µL A5 to A6; eject tip.
9. `transfer-a6-a7`: Fresh tip; mix A6 three times; transfer 100 µL A6 to A7; eject tip.
10. `transfer-a7-a8`: Fresh tip; mix A7 three times; transfer 100 µL A7 to A8; eject tip.
11. `transfer-a8-a9`: Fresh tip; mix A8 three times; transfer 100 µL A8 to A9; eject tip.
12. `transfer-a9-a10`: Fresh tip; mix A9 three times; transfer 100 µL A9 to A10; eject tip.
13. `transfer-a10-a11`: Fresh tip; mix A10 three times; transfer 100 µL A10 to A11; eject tip.
14. `discard-a11`: Fresh tip; mix A11 three times; remove 100 µL to waste, never A12. Require A11 lighter, activity at A11 and waste, A12 unchanged, and a Meta reading. All used wells now contain 100 µL.
15. `final-read`: Remove hands and obtain the final verified measurement frame. Check the series on A1–A11 and that A12 remains the blank. Ask the Meta reader about visible contradictions, bubbles or contamination; operator confirms the final read. Do not claim a quantitative concentration measurement below the optical detection floor.

## Binding measurement and recovery rules

Top-down OD for equal-area flat-bottom wells is proportional to dye concentration times liquid depth, hence to dye amount per well, not concentration alone. This is the optical model to test, not an already measured hardware result. After each correctly mixed transfer, source and destination contain equal dye amounts and therefore approximately **equal OD**. The source OD halves relative to its pre-transfer value; the target darkens from blank. This is a halving of OD, not a fixed 0.3 OD subtraction and not half the camera intensity.

Every transfer uses `activity_at` for source and target, `colour_change(source, lighter)`, `colour_change(target, darker)` and a Meta `judge` for contradiction/uniformity. The arrow must join the two `plate:<cell>` references. Put `ratio: 1.0, tol: 0.25` on the target's darker check: OD(target)/OD(source after) must be 0.75–1.25. Set positive change thresholds from blank noise on the actual rig; do not invent measured sensitivity. Tip replacement is operator practice/advisory evidence, not a visually guaranteed pass.

Only `final-read` uses `series`: ratio 0.5, tolerance 0.15, ordered cells A1–A11. Use ratios above OD 0.03 and monotonic non-increase within noise below it. Linearise sRGB (including its low-value linear branch), use the red channel for blue dye, reject specular pixels at or above 250 and sample the central half-radius disc. Reference each cell before dye arrived or a valid blank, with flat-field correction where available. Neither a human confirmation nor a model verdict by itself proves a passed liquid step.

Planted mistake at `transfer-a5-a6`: deliberately put the 100 µL from A5 into A7 instead. A5 gets lighter, A6 stays unchanged, A7 darkens and activity hits A7. Report exactly: "liquid went into A7, expected A6". Highlight A7 red and A6 yellow. On `wrong_cell:A7`, apply `exclude_cells` to `plate:A6` and `skip_steps` to `transfer-a6-a7`; do not redo the emptied A5 transfer. Keep A6 as an extra blank. Continue with `transfer-a7-a8` using the 200 µL now in A7. Never discard A7 and never try to transfer dye from blank A6. Still perform `discard-a11`. The final series is [A1,A2,A3,A4,A5,A7,A8,A9,A10,A11]. Other wrong-well or uncertain-volume mistakes abort this demonstration row and restart on a newly registered row; do not improvise recovery. If A12 is contaminated, stop and obtain a separately recorded valid blank before any quantitative interpretation.

## If the lab has no 96-well plate

These are proposed adaptations, not surveyed equipment. Prefer a 12/24-well flat-bottom plate with safe volumes selected within vessel and pipette capacity; record a new grid and re-author the step count before collection. Alternatively register a row of identical clear cuvettes as a 1×N grid: 500 µL diluent, 1 mL initial stock, 500 µL transfers and final discard, provided the vessels allow mixing. A nominal 10 mm cuvette dimension does not establish the vertical optical path: validate the top-down response and equal-OD model. Open 1.5 mL tubes in a rack are a last resort; folded-back lids and conical bottoms weaken/alter the optical model. For tubes, do not reuse plate OD thresholds or claim the same quantitative series without calibration. Use activity plus Meta reading and explicitly unverified operator confirmation where optical evidence is insufficient. Never silently run the 96-well graph on another geometry.
