"""Projected calibration pattern (D5, rig brief §4.1): ArUco ``DICT_4X4_50`` ids 0-14 on a 3x5 grid.

The server lays the pattern out in page CSS px from the page's ``hello``. The page (``web/js/aruco.js``) snaps each
marker to whole device px, draws it with a 1-module white quiet zone on black and echoes the rects as drawn in its
``ack``. A rect is the marker's outer black border: ``(x, y)`` its top-left corner, ``size`` its side, which spans
6 modules (1-module border around the 4x4 bits).
"""

from __future__ import annotations

import itertools
import math
from functools import cache

import cv2
import numpy as np

from ...contracts.render import MarkerRect, Pattern

ARUCO_DICT = cv2.aruco.DICT_4X4_50
PATTERN_IDS = tuple(range(15))
MODULES = 6  # 4x4 bits plus a 1-module border on each side
SIDE_FRAC = 0.15  # target marker side / page height (capped to fit narrow pages)
MARGIN_FRAC = 0.06  # page edge to the nearest border / short page side
COLS, ROWS = 5, 3


@cache
def dictionary() -> cv2.aruco.Dictionary:
    return cv2.aruco.getPredefinedDictionary(ARUCO_DICT)


def pattern_for_page(page_w: float, page_h: float) -> Pattern:
    """15 markers spread over the whole page (the solve is best conditioned when the points span it).

    Sides are a multiple of 6 CSS px so that at ``dpr`` 1 every module is a whole pixel; neighbouring borders stay
    3 modules apart (two quiet zones plus one module), which caps the side on squarish pages.
    """
    margin = math.ceil(MARGIN_FRAC * min(page_w, page_h))
    fit = min(SIDE_FRAC * page_h, (page_w - 2 * margin) / (COLS + (COLS - 1) / 2),
              (page_h - 2 * margin) / (ROWS + (ROWS - 1) / 2))
    side = MODULES * math.floor(fit / MODULES)
    if side < 2 * MODULES:
        raise ValueError(f"page {page_w}x{page_h} CSS px is too small for the calibration pattern")
    xs = _spread(margin, page_w - margin - side, COLS)
    ys = _spread(margin, page_h - margin - side, ROWS)
    rects = [MarkerRect(id=i, x=x, y=y, size=float(side)) for i, (y, x) in zip(PATTERN_IDS, itertools.product(ys, xs))]
    return Pattern(markers=rects)


def _spread(lo: float, hi: float, n: int) -> list[float]:
    return [float(round(lo + k * (hi - lo) / (n - 1))) for k in range(n)]


def marker_corners(rect: MarkerRect) -> np.ndarray:
    """Outer border corners TL, TR, BR, BL (the ArUco corner order), page CSS px."""
    x, y, s = rect.x, rect.y, rect.size
    return np.array([[x, y], [x + s, y], [x + s, y + s], [x, y + s]], dtype=np.float64)


def marker_modules(marker_id: int) -> np.ndarray:
    """6x6 uint8 module grid of a ``DICT_4X4_50`` marker, 1 = white, border included."""
    return (cv2.aruco.generateImageMarker(dictionary(), marker_id, MODULES) > 127).astype(np.uint8)
