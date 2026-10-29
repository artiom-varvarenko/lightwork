"""Marker detection and the camera px -> page CSS px homography (rig brief §4.1-§4.3).

Corners are sub-pixel refined by ArUco's ``CORNER_REFINE_SUBPIX`` with a window of half a marker module (at most
12 px): wide enough to average a blurred edge, too small to reach the next edge one module away. On synthetic
projections that halves the error of the default window (0.06 -> 0.035 page px mean, blur sigma 1.5 px).
``cv2.findHomography`` with RANSAC (threshold in destination units: 2 page px, about 1 mm at 0.5 m) refines the
model on the inliers; residuals are reported over those inliers.
"""

from __future__ import annotations

from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass

import cv2
import numpy as np

from ...contracts.layout import Residuals
from ...contracts.render import MarkerRect
from ...geometry import apply_h
from .pattern import dictionary, marker_corners

MIN_MARKERS = 6
RANSAC_PX = 2.0
CLIP_DN = 250


class CalibrationError(RuntimeError):
    pass


def gray(image: np.ndarray) -> np.ndarray:
    return image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def _detector() -> cv2.aruco.ArucoDetector:
    p = cv2.aruco.DetectorParameters()
    p.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    p.cornerRefinementWinSize = 12
    p.relativeCornerRefinmentWinSize = 0.5
    p.cornerRefinementMaxIterations = 50
    p.cornerRefinementMinAccuracy = 0.01
    return cv2.aruco.ArucoDetector(dictionary(), p)


def detect_markers(image: np.ndarray, ids: Collection[int]) -> dict[int, np.ndarray]:
    """Refined corners (4x2 TL, TR, BR, BL, camera px) of every marker in ``ids`` that is seen exactly once."""
    corners, found, _ = _detector().detectMarkers(gray(image))
    if found is None:
        return {}
    seen: dict[int, np.ndarray] = {}
    twice: set[int] = set()
    for c, i in zip(corners, found.ravel().tolist()):
        if i in ids:
            if i in seen:
                twice.add(i)
            seen[i] = c.reshape(4, 2).astype(np.float64)
    return {i: c for i, c in seen.items() if i not in twice}


def merge(detections: Sequence[Mapping[int, np.ndarray]]) -> dict[int, np.ndarray]:
    """Per marker, the mean corners over the frames that saw it (averages sensor noise across frames)."""
    acc: dict[int, list[np.ndarray]] = {}
    for det in detections:
        for i, c in det.items():
            acc.setdefault(i, []).append(c)
    return {i: np.mean(cs, axis=0) for i, cs in sorted(acc.items())}


@dataclass(frozen=True)
class Solution:
    H: np.ndarray  # 3x3, camera px -> page CSS px, H[2, 2] == 1
    residuals: Residuals
    marker_px: dict[int, float]  # per marker id: largest corner residual, page px (outliers included)
    corners: dict[int, np.ndarray]  # camera px corners the solve used
    clipped: float | None = None  # fraction of saturated pixels on the markers and quiet zones


def _r(v: float) -> float:
    return round(float(v), 3)


def fit(corners: Mapping[int, np.ndarray], drawn: Sequence[MarkerRect]) -> Solution:
    rects = {m.id: m for m in drawn}
    ids = sorted(i for i in corners if i in rects)
    if len(ids) < MIN_MARKERS:
        raise CalibrationError(f"found {len(ids)} of {len(rects)} pattern markers, need >= {MIN_MARKERS}")
    cam = np.concatenate([corners[i] for i in ids])
    page = np.concatenate([marker_corners(rects[i]) for i in ids])
    H, mask = cv2.findHomography(cam, page, cv2.RANSAC, RANSAC_PX)
    if H is None or not np.all(np.isfinite(H)) or abs(H[2, 2]) < 1e-12:
        raise CalibrationError("homography solve failed")
    H = H / H[2, 2]
    inliers = mask.ravel().astype(bool)
    if inliers.sum() < 4 * MIN_MARKERS:
        raise CalibrationError(f"only {int(inliers.sum())} of {len(cam)} corners fit within {RANSAC_PX} page px")
    err = np.linalg.norm(apply_h(H, cam) - page, axis=1)
    e = err[inliers]
    residuals = Residuals(
        rms_px=_r(np.sqrt(np.mean(e**2))),
        p95_px=_r(np.percentile(e, 95)),
        max_px=_r(e.max()),
        n_markers=len(ids),
        n_inliers=int(inliers.sum()),
    )
    marker_px = {i: _r(err[4 * k : 4 * k + 4].max()) for k, i in enumerate(ids)}
    return Solution(H=H, residuals=residuals, marker_px=marker_px, corners={i: corners[i] for i in ids})


def clipped_fraction(image: np.ndarray, corners: Mapping[int, np.ndarray]) -> float:
    """Share of pixels >= ``CLIP_DN`` over the markers and their quiet zones. Clipped whites bias the corners
    towards the black side; the fix is a lower page white or a shorter exposure, not more refinement."""
    g = gray(image)
    mask = np.zeros(g.shape, np.uint8)
    for c in corners.values():
        centre = c.mean(axis=0)
        quad = centre + (c - centre) * (8 / 6)  # border + quiet zone: 8 modules across instead of 6
        cv2.fillConvexPoly(mask, np.round(quad).astype(np.int32), 255)
    inside = g[mask > 0]
    return round(float(np.mean(inside >= CLIP_DN)), 4) if inside.size else 0.0


def solve(frame_bgr: np.ndarray, drawn: Sequence[MarkerRect]) -> tuple[np.ndarray, Residuals]:
    """``H_cam_to_page`` and its residuals from one camera frame of the projected pattern."""
    sol = fit(detect_markers(frame_bgr, {m.id for m in drawn}), drawn)
    return sol.H, sol.residuals
