"""Per-cell regions of interest on top of :class:`lightwork.geometry.GridGeom` (D8).

Cell centres and local pitch come from the grid homography in ``geometry.py``; this module only turns them into
measurement discs. All coordinates are camera px unless a function says otherwise.

ROI rule (``briefs/rig.md`` §6): the central disc of radius ``0.5 x well radius`` avoids the meniscus and the
refracting well wall. The well radius is estimated from the cell pitch with :data:`WELL_RADIUS_PER_PITCH`.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np

from ..geometry import GridGeom, cell_index, cell_name

# 96-well SBS plate: well diameter ~6.86 mm at 9 mm pitch -> radius/pitch = 0.381. Ice-cube trays and 12/24-well
# plates sit at 0.38-0.42, so 0.38 keeps the ROI inside the well for every grid object we plan to use.
WELL_RADIUS_PER_PITCH = 0.38
# Fraction of the well radius that is measured.
ROI_FRACTION = 0.5


def roi_radius(pitch_px: float) -> float:
    """ROI disc radius in px for a cell pitch: ``ROI_FRACTION * WELL_RADIUS_PER_PITCH * pitch`` (0.19 x pitch)."""
    if pitch_px <= 0:
        raise ValueError(f"pitch must be positive, got {pitch_px!r}")
    return ROI_FRACTION * WELL_RADIUS_PER_PITCH * float(pitch_px)


def cell_rois(geom: GridGeom, names: Iterable[str] | None = None) -> tuple[dict[str, tuple[float, float]], float]:
    """Centres (camera px) of ``names`` (default: every cell) and one shared ROI radius.

    The radius uses the smallest local pitch among the cells, so under perspective no disc spills over its well.
    """
    names = list(geom.names() if names is None else names)
    if not names:
        raise ValueError("no cells")
    centres = {n: geom.centre(n) for n in names}
    return centres, roi_radius(min(geom.pitch(n) for n in names))


def disc_indices(shape: tuple[int, ...], centre: tuple[float, float], radius: float) -> tuple[np.ndarray, np.ndarray]:
    """(ys, xs) of the pixels whose centres lie within ``radius`` of ``centre`` (image px, pixel centres at integers).

    Pixels outside the image are dropped, so a disc near the border yields fewer (possibly zero) pixels.
    """
    h, w = shape[:2]
    cx, cy = float(centre[0]), float(centre[1])
    x0, x1 = max(int(np.floor(cx - radius)), 0), min(int(np.ceil(cx + radius)), w - 1)
    y0, y1 = max(int(np.floor(cy - radius)), 0), min(int(np.ceil(cy + radius)), h - 1)
    if x1 < x0 or y1 < y0:
        return np.empty(0, np.intp), np.empty(0, np.intp)
    ys, xs = np.mgrid[y0 : y1 + 1, x0 : x1 + 1]
    inside = (xs - cx) ** 2 + (ys - cy) ** 2 <= radius * radius
    return ys[inside], xs[inside]


def neighbours(name: str, rows: int, cols: int, naming: str, reach: int = 2) -> list[str]:
    """Cells within ``reach`` steps (Chebyshev distance) of ``name``, excluding it, in row-major order.

    The candidate set for :func:`lightwork.measure.colour.wrong_cell`: a misplaced dispense lands near the target.
    """
    r0, c0 = cell_index(name, naming)
    return [
        cell_name(r, c, naming)
        for r in range(max(r0 - reach, 0), min(r0 + reach, rows - 1) + 1)
        for c in range(max(c0 - reach, 0), min(c0 + reach, cols - 1) + 1)
        if (r, c) != (r0, c0)
    ]
