# Projector rental kit: pack and return-check

Original procedure text and trial design by the Lightwork contributors, 2026. Licence: CC BY 4.0, https://creativecommons.org/licenses/by/4.0/. No rental-company manual or iFixit material is reproduced. Brand names describe possible owned equipment, not affiliation.

## Materials and placement

Use an unplugged, cool projector; its matching remote; a coiled power cable; a coiled HDMI cable; two loose matching batteries; and an open rental case. Check that the projector fits without force. Keep battery terminals away from loose metal; never use leaking or damaged cells. Do not energise the projector during this procedure.

Fix the case in `case-zone`, with a clear `bench` staging zone beside it. Put light matte card inserts in dark foam so cables and projected outlines can be seen. Define separate `slot-p`, `slot-c`, `slot-h` and `slot-r` polygons on a correctly packed reference frame, then accept them manually. Slot R holds the remote and a small nonconductive battery compartment. Items start outside all slots. Keep connectors visible to distinguish HDMI from power; do not rely on cable colour. Object nouns: projector, remote, power cable, HDMI cable, battery, case. Put labels beside the case rather than over evidence.

## Steps (preserve these IDs)

1. `setup`: Open the case and remove all loose items to the bench. Check case present in its zone and the named projector, remote, cables and batteries absent from their slots. Operator checks the case is dry and undamaged. If an unexpected object remains, remove it before restarting the check.
2. `projector`: Move the cool projector from the bench into slot P, seated flat in its intended orientation. Use `moved_to(projector, slot-p)` and a Meta `judge` for flat seating. Do not close the lid to force a poor fit. On failure lift, turn and reseat the projector.
3. `power-cable`: Coil the disconnected power cable loosely and place it fully in slot C with its connector visible. Use `moved_to(power-cable, slot-c)` and a Meta check that it is not in the HDMI slot. On failure move it to slot C.
4. `hdmi-cable`: Coil the HDMI cable and put it in slot H, showing a connector. Use `moved_to(hdmi-cable, slot-h)` and a Meta identity/location check. On failure correct the slot; do not accept a power cable as a substitute.
5. `remote-batteries`: Put the matching remote and exactly two loose batteries into slot R's separate recesses. Use `moved_to(remote, slot-r)`, `count(battery, slot-r, 2)`, `absent(remote, bench)` and a Meta contradiction check. Planted mistake: leave the remote visible on the bench while placing both batteries correctly. Ring the bench remote in red, identify it as missing from slot R, and redo this step by moving it into the case; do not advance on a spoken "done" alone.
6. `final-check`: Keep the lid OPEN. Individually check projector in P, power cable in C, HDMI cable in H, remote in R and exactly two batteries in R. Ask the Meta reader whether the visible set matches these requirements. There are six physical pieces, not five; avoid an ambiguous `count(item, 5)`. The lid must not hide the final evidence. If something is wrong, reopen its packing step and repeat the open-case final check.
7. `close-lid`: Only after the recorded open-case check, operator closes and latches the lid without force. This is a separate confirm-only step, recorded unverified, not proof that invisible contents are present.

For an arriving return case, inspect the OPEN contents against `final-check`; use present/count checks, not a fictitious moved-to history. Record missing parts and re-pack after inspection. An actual return-mode graph needs explicit present checks; do not silently reinterpret movement checks. Never claim the software inspected condition, battery charge, electrical safety or projector function from this view.
