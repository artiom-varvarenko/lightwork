"""Reducer thresholds (workstream H).

The reducer reads :data:`DEFAULT` when it is constructed. ``verify`` re-derives recorded runs with the code's values,
so changing a value here changes the derived output of recorded runs (PLAN.md §4.2); tests flip a value with
``dataclasses.replace`` to prove that.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Params:
    # colour_change (OD on the grid's channel, or CIEDE2000 for "hue"; PLAN.md §6.1)
    od_min_delta: float = 0.02  # default OD change for darker / lighter / none
    hue_min_delta: float = 4.0  # default delta E for hue
    # transfer steps (the step's arrow joins two cells of one grid; briefs/procedure.md §1 physics):
    # need = max(min_delta, frac x OD of the source before); unreadable below the source floor.
    transfer_drop_frac: float = 0.25
    transfer_min_source_od: float = 0.06  # = 3 x od_min_delta: the expected change (half the source) must clear it
    ratio_tol: float = 0.25  # default tol of a colour_change ratio (equal-OD band of a transfer)
    # series
    series_tol: float = 0.15
    series_floor: float = 0.03
    # wrong-cell search radius (Chebyshev cells around the expected target)
    wrong_cell_reach: int = 2
    # SAM checks: share of a mask's area that must lie inside the region
    present_min_overlap: float = 0.5
    # verification deadline after a step turns provisional (D3: Spark verdict 20 s, Glimmer 180 s)
    verify_timeout_meta_s: float = 20.0
    verify_timeout_chat_s: float = 180.0
    # scene
    cell_ring_frac: float = 0.45  # ring radius / cell pitch: neighbouring rings never touch
    mark_pad_px: float = 8.0  # padding of a ring drawn around a SAM mask


DEFAULT = Params()
