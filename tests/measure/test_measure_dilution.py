"""1:2 serial dilution physics on synthetic plates (briefs/procedure.md §1, PLAN.md §6.1-6.2)."""

from __future__ import annotations

import pytest

from lightwork.contracts.stepgraph import Grid
from lightwork.geometry import GridGeom
from lightwork.measure.colour import cell_stats, od, series_ok, transfer_ok, wrong_cell
from lightwork.measure.grid import cell_rois, neighbours
from lightwork.testing.synth import well_plate

ROWS, COLS = 8, 12
ROW_A = [f"A{i}" for i in range(1, 13)]
A1_START = 1.2  # A1's red-channel OD right after the dye goes in (dye tuned to 0.8-1.2)


def _cells():
    _, g = well_plate({})
    return cell_rois(GridGeom(g.corners, Grid(rows=ROWS, cols=COLS, naming="A1")), ROW_A + ["B5", "B6", "B7"])


CENTRES, RADIUS = _cells()


def _measure(ods: dict[str, float], seed: int, noise: float = 2.0) -> dict[str, float]:
    """Render the plate, then read every cell's OD against a pre-dye reference keyframe."""
    img, _ = well_plate(ods, noise_sd=noise, seed=seed)
    ref, _ = well_plate({}, noise_sd=noise, seed=seed + 1000)
    s, s0 = cell_stats(img, CENTRES, RADIUS, "r"), cell_stats(ref, CENTRES, RADIUS, "r")
    return {c: od(s[c].i_lin, s0[c].i_lin) for c in CENTRES}


def _transfer(amounts: dict[str, float], src: str, tgt: str) -> None:
    """Equal-volume transfer: half the source's dye goes to the target (OD ~ amount of dye in the well)."""
    half = amounts[src] / 2
    amounts[src] = half
    if tgt != "waste":
        amounts[tgt] = amounts.get(tgt, 0.0) + half


@pytest.mark.parametrize("noise,seed", [(0.0, 0), (2.0, 1), (2.0, 2)])
def test_final_series_of_11_passes(noise, seed):
    ods = {f"A{k}": A1_START / 2**k for k in range(1, 12)}
    m = _measure(ods, seed, noise)
    ok, reason = series_ok([m[c] for c in ROW_A[:11]], 0.5, 0.15, names=ROW_A[:11])
    assert ok, reason


def test_series_with_skipped_halving_fails():
    ods = {f"A{k}": A1_START / 2 ** (k if k < 4 else k + 1) for k in range(1, 12)}  # A3 -> A4 drops two halvings
    m = _measure(ods, 3)
    ok, reason = series_ok([m[c] for c in ROW_A[:11]], 0.5, 0.15, names=ROW_A[:11])
    assert not ok and reason.startswith("A3->A4: ") and "dilution steps" in reason, reason
    assert 1.8 < float(reason.split()[1]) < 2.2


@pytest.mark.parametrize("seed", [10, 20, 30])
def test_clean_five_transfer_row(seed):
    amounts = {"A1": A1_START}
    before = _measure(dict(amounts), seed)
    for k in range(1, 6):
        src, tgt = f"A{k}", f"A{k + 1}"
        _transfer(amounts, src, tgt)
        after = _measure(dict(amounts), seed + k)
        status, reason = transfer_ok(after[src], after[tgt], before[src], before[tgt], names=(src, tgt))
        assert status == "pass", (k, reason)
        deltas = {c: after[c] - before[c] for c in neighbours(tgt, ROWS, COLS, "A1") if c in after}
        assert wrong_cell(tgt, deltas, exclude=[src]) is None
        before = after


@pytest.mark.parametrize("seed", [100, 200, 300])
def test_planted_a5_to_a7_detected_and_recovered(seed):
    amounts = {"A1": A1_START}
    for k in range(1, 5):
        _transfer(amounts, f"A{k}", f"A{k + 1}")
    before = _measure(dict(amounts), seed)
    _transfer(amounts, "A5", "A7")  # planted mistake: expected A6
    after = _measure(dict(amounts), seed + 1)

    status, reason = transfer_ok(after["A5"], after["A6"], before["A5"], before["A6"], names=("A5", "A6"))
    assert status == "fail" and reason.startswith("A6 did not darken")
    deltas = {c: after[c] - before[c] for c in neighbours("A6", ROWS, COLS, "A1") if c in after}
    assert wrong_cell("A6", deltas, exclude=["A5"]) == "A7"

    # recovery: A6 becomes an extra blank, continue from A7 (A7 -> A8 ... A11 -> waste)
    for k in range(7, 12):
        _transfer(amounts, f"A{k}", f"A{k + 1}" if k < 11 else "waste")
    final = _measure(dict(amounts), seed + 2)
    kept = ROW_A[:5] + ROW_A[6:11]
    ok, why = series_ok([final[c] for c in kept], 0.5, 0.15, names=kept)
    assert ok, why
    ok, why = series_ok([final[c] for c in ROW_A[:11]], 0.5, 0.15, names=ROW_A[:11])
    assert not ok and why.startswith("A6->A7")


def test_transfer_failure_modes():
    # pipette set to 50 uL instead of 100: source keeps 2/3, target gets 1/3
    assert transfer_ok(0.4, 0.2, 0.6)[0] == "fail"
    assert "ratio 0.50" in transfer_ok(0.4, 0.2, 0.6)[1]
    # double dispense into the target
    assert transfer_ok(0.3, 0.6, 0.6)[0] == "fail"
    # nothing moved
    assert transfer_ok(0.6, 0.0, 0.6) == ("fail", "target did not darken (OD change +0.0000, need +0.1500)")
    # target darkened but source unchanged (dye from elsewhere)
    assert transfer_ok(0.6, 0.3, 0.6)[1].startswith("source did not get lighter")
    # too dilute to read
    assert transfer_ok(0.01, 0.01, 0.02)[0] == "inconclusive"
    # readable before, below floor after: direction is enough
    assert transfer_ok(0.025, 0.025, 0.05)[0] == "pass"


def test_wrong_cell_rules():
    assert wrong_cell("A6", {"A6": 0.1, "A7": 0.2}) is None
    assert wrong_cell("A6", {"A6": 0.0, "A5": -0.1, "A7": 0.005}) is None
    assert wrong_cell("A6", {"A6": 0.0, "A5": 0.3, "A7": 0.1}, exclude=["A5"]) == "A7"
    assert wrong_cell("A6", {"A6": 0.0, "B6": 0.1, "A7": 0.1}) == "A7"  # tie: first in name order


def test_series_rules():
    assert series_ok([0.8, 0.4, 0.2, 0.1, 0.05, 0.025, 0.012, 0.0], 0.5, 0.15)[0]
    assert series_ok([0.01, 0.0], 0.5, 0.15) == (False, "no cell above OD 0.03")
    assert series_ok([0.8], 0.5, 0.15) == (False, "series needs at least 2 cells")
    assert series_ok([0.8, 0.4, 0.02, 0.035], 0.5, 0.15)[1] == "#3->#4: OD rises from 0.0200 to 0.0350"
    assert series_ok([0.8, 0.4, 0.2, 0.012, 0.025], 0.5, 0.15, names=list("abcde"))[1].startswith("d->e: OD rises")
    with pytest.raises(ValueError):
        series_ok([0.8, 0.4], 2.0, 0.15)
    with pytest.raises(ValueError):
        series_ok([0.8, 0.4], 0.5, 0.15, names=["a"])
