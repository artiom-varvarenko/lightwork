# Tea dilution: engineering test only

Original procedure text and trial design by the Lightwork contributors, 2026. Licence: CC BY 4.0, https://creativecommons.org/licenses/by/4.0/. This is a home optical engineering test, not a filmed use-case demonstration and not evidence of laboratory accuracy.

## Materials and setup

Use eight identical clear, straight-sided, flat-bottom vessels holding at least 25 mL, matte white backing, fully cooled filtered black tea or coffee, room-temperature water, a 10 mL dosing syringe without a needle, mixing spoon/syringe and a waste cup. Do not use hot liquid, chipped glasses or variable-shaped vessels. Do not drink liquids after the demonstration. Keep liquid away from cables and the laptop. If only six vessels are available, re-author a shorter graph rather than executing the eight-cell graph unchanged.

Arrange vessels in two rows of four: A1–A4 at the back, B1–B4 at the front. Physical sequence is A1,A2,A3,A4,B1,B2,B3, with B4 the blank. A two-row arrangement avoids degenerate four-corner registration of a single row. In the procedure's step IDs, logical A1…A7 mean those seven sequence positions; the graph MUST map them to the actual cells just listed. Use grid object `glasses`, rows=2, cols=4, naming=A1, channel=b. Register centres A1,A4,B4,B1. Keep vessel spacing uniform and never move them after registration. Blue-channel OD is appropriate to tea/coffee; validate this on the actual brew.

## Eleven steps (preserve these IDs)

1. `setup`: Place eight empty clean glasses, water, cooled stock, syringe and waste in their zones. Check presence and a Meta no-contradiction reading. Capture an empty/blank reference, then pre-test stock concentration so the signal is neither near-black nor lost in noise.
2. `add-water`: Put 10 mL water into each of the seven vessels except the first (actual cells A2,A3,A4,B1,B2,B3,B4). Leave A1 empty. Tally the dispenses and record activity; clear water cannot be certified by its colour. Ask for a Meta contradiction check.
3. `add-stock`: Put 20 mL cooled tea/coffee into A1. Require darker OD there plus a Meta reading. Do not pour into the blank.
4. `transfer-a1-a2`: Mix A1 gently three times and move 10 mL to A2; keep bubbles out.
5. `transfer-a2-a3`: Mix A2 three times; move 10 mL to A3.
6. `transfer-a3-a4`: Mix A3 three times; move 10 mL to A4.
7. `transfer-a4-a5`: Mix A4 three times; move 10 mL to B1 (logical fifth vessel).
8. `transfer-a5-a6`: Mix B1 three times; move 10 mL to B2.
9. `transfer-a6-a7`: Mix B2 three times; move 10 mL to B3.
10. `discard-a7`: Mix B3 three times; remove 10 mL to waste, never B4. All seven tea vessels now contain 10 mL, and B4 contains only 10 mL water.
11. `final-read`: With hands out, record ten verified measurement keyframes across two minutes. Check the ordered final series A1,A2,A3,A4,B1,B2,B3, with B4 unchanged as blank; ask a Meta reader for visible contradictions. Record actual mean and SD and how many adjacent pairs can be distinguished. No fixed success count is presumed.

## Measurement and planted mistakes

For every transfer use source lighter and target darker colour checks, source/target activity, and a Meta judge. Arrows connect the actual `glasses:<cell>` references. Set target darker `ratio=1.0, tol=0.25`: target/source-after OD should lie within 0.75–1.25. In equal-area vessels viewed top-down, OD tracks dye amount: after transfer source and target should have equal OD, while source OD halves. Do not expect a final geometric gradient until after the last discard. This optical model must be tested for the chosen vessels.

Use only a final `series` check, ratio 0.5, tolerance 0.15. Apply ratios where OD exceeds 0.03 and monotonic-with-noise checks below it. Linearise sRGB, use blue-channel OD, central half-radius samples, specular rejection and cell-specific blank references. Report adjacent separation relative to three SD from the repeated frames; report fewer separable steps honestly. Seven tea vessels supply six adjacent pairs; the engineering target is six separable pairs, not a claimed result.

For five fresh-row transfer tests, measure the equal-OD ratio and report the actual count within 0.75–1.25. For a planted skip, at `transfer-a3-a4` send the 10 mL from A3 to B1 instead of A4. Expected evidence: A3 lighter, A4 unchanged, B1 darker. Do not continue an undocumented mixture: stop, empty/rinse and restage before the next trial. For a recovery rehearsal, a separately reviewed graph may exclude A4 and skip `transfer-a4-a5`, then continue B1→B2; record this variant explicitly rather than pretending the default graph recovered.
