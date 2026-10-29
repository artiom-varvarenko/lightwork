"""Bump detection (rig brief §4.5) between calibrations.

- **Camera moved:** ``cv2.phaseCorrelate`` on 4 corner patches of the camera frame that lie outside the projected
  page and the work area (so the overlay never changes them). Only patches with texture correlate (response
  >= ``MIN_RESPONSE``); a similarity transform fitted to the responding patches is the camera motion.
- **Projector moved (or keystone fired):** 4 small projected ticks near the page corners (:func:`tick_prims`, drawn
  as ``dot`` prims) are re-found around their reference centroids; drift beyond what the camera motion explains
  means the projector moved. Ticks no longer found near their reference also count as moved.
- One check is a sample; ``moved`` needs ``CONFIRM`` positive samples in a row (a hand passing is not a bump).
  Run :meth:`BumpMonitor.check` on settled, hand-free frames (the live loop: every ~1 s), then re-solve with
  ``flow.recalibrate`` and start a new monitor from a fresh reference frame.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from ...contracts import q_xy
from ...contracts.layout import Calibration
from ...contracts.render import Dot
from ...geometry import apply_h
from .solve import gray

PATCH_PX = 128
EDGE_PX = 8  # patch inset from the frame edge
KEEP_OUT_PX = 16  # clearance around the projected page
MIN_RESPONSE = 0.2  # textured, unchanged content correlates at 0.3-1.0; noise or new content stays below 0.1
MIN_TEXTURE_DN = 1.5  # std after 1 px denoising; a flat patch cannot reveal camera motion
AGREE_PX = 1.0
CAMERA_SHIFT_PX = 2.0
TICK_DRIFT_PX = 3.0
TICK_INSET_FRAC = 0.05  # tick centres inset from the page corners, fraction of the short page side
TICK_SEARCH_PX = 40
MIN_TICK_CONTRAST = 25.0
CONFIRM = 2


def tick_points_page(page_w: float, page_h: float) -> np.ndarray:
    m = TICK_INSET_FRAC * min(page_w, page_h)
    return np.array([[m, m], [page_w - m, m], [page_w - m, page_h - m], [m, page_h - m]], dtype=np.float64)


def tick_prims(calib: Calibration) -> list[Dot]:
    """The 4 corner ticks as scene prims (camera px), to be drawn in every normal scene while monitoring."""
    cam = apply_h(np.linalg.inv(np.asarray(calib.H_cam_to_page)), tick_points_page(calib.page.w, calib.page.h))
    return [Dot(id=f"calib-tick-{i}", c=q_xy(p), role="info") for i, p in enumerate(cam)]


def find_blob(g: np.ndarray, xy: np.ndarray, reach: float) -> np.ndarray | None:
    """Intensity-weighted centroid of the brightest blob within ``reach`` px of ``xy``, or None."""
    h, w = g.shape
    x0, y0 = max(0, int(xy[0] - reach)), max(0, int(xy[1] - reach))
    x1, y1 = min(w, int(xy[0] + reach) + 1), min(h, int(xy[1] + reach) + 1)
    if x1 - x0 < 5 or y1 - y0 < 5:
        return None
    win = g[y0:y1, x0:x1].astype(np.float32)
    bg, peak = float(np.median(win)), float(win.max())
    if peak - bg < MIN_TICK_CONTRAST:
        return None
    _, labels = cv2.connectedComponents((win > bg + 0.5 * (peak - bg)).astype(np.uint8))
    iy, ix = np.unravel_index(int(np.argmax(win)), win.shape)
    wts = np.where(labels == labels[iy, ix], win - bg, 0.0)
    ys, xs = np.mgrid[y0:y1, x0:x1]
    s = float(wts.sum())
    return np.array([float((wts * xs).sum()) / s, float((wts * ys).sum()) / s])


@dataclass(frozen=True)
class PatchShift:
    centre: tuple[float, float]  # camera px
    shift: tuple[float, float]  # content motion, camera px
    response: float


@dataclass(frozen=True)
class BumpCheck:
    moved: bool  # confirmed over CONFIRM samples: re-solve now
    camera_moved: bool  # this sample: the patches moved more than CAMERA_SHIFT_PX
    projector_moved: bool  # this sample: ticks drifted beyond the camera motion, or were lost
    camera_observable: bool  # >= 2 patches correlated; otherwise a camera move looks like a projector move
    camera_motion: np.ndarray | None  # 3x3 similarity, old camera px -> new camera px (patch fit)
    tick_drift_px: float | None  # median tick drift, camera px
    patches: tuple[PatchShift, ...]
    ticks_found: int
    ticks_ref: int


class BumpMonitor:
    def __init__(self, ref_image: np.ndarray, calib: Calibration, *, work_area_xywh=None, ticks: bool = True, confirm: int = CONFIRM):
        g = gray(ref_image).astype(np.float32)
        h, w = g.shape
        keep_out = np.zeros((h, w), np.uint8)
        H_p2c = np.linalg.inv(np.asarray(calib.H_cam_to_page))
        page = apply_h(H_p2c, [[0, 0], [calib.page.w, 0], [calib.page.w, calib.page.h], [0, calib.page.h]])
        cv2.fillConvexPoly(keep_out, np.round(page).astype(np.int32), 255)
        if work_area_xywh is not None:
            x, y, ww, hh = (int(v) for v in work_area_xywh)
            keep_out[max(0, y) : y + hh, max(0, x) : x + ww] = 255
        keep_out = cv2.dilate(keep_out, np.ones((2 * KEEP_OUT_PX + 1, 2 * KEEP_OUT_PX + 1), np.uint8))
        s = PATCH_PX
        self._window = cv2.createHanningWindow((s, s), cv2.CV_32F)
        self._patches: list[tuple[int, int, np.ndarray]] = []
        for x, y in ((EDGE_PX, EDGE_PX), (w - EDGE_PX - s, EDGE_PX), (w - EDGE_PX - s, h - EDGE_PX - s), (EDGE_PX, h - EDGE_PX - s)):
            if x >= 0 and y >= 0 and not keep_out[y : y + s, x : x + s].any():
                patch = cv2.GaussianBlur(g[y : y + s, x : x + s], (5, 5), 1.0)
                if float(patch.std()) >= MIN_TEXTURE_DN:
                    self._patches.append((x, y, cv2.multiply(patch - patch.mean(), self._window)))
        self._ticks: list[np.ndarray] = []
        if ticks:
            for p in apply_h(H_p2c, tick_points_page(calib.page.w, calib.page.h)):
                found = find_blob(g, p, TICK_SEARCH_PX)
                if found is not None:
                    self._ticks.append(found)
        self._confirm = confirm
        self._streak = 0

    @property
    def n_patches(self) -> int:
        return len(self._patches)

    @property
    def n_ticks(self) -> int:
        return len(self._ticks)

    def check(self, image: np.ndarray) -> BumpCheck:
        g = gray(image).astype(np.float32)
        s = PATCH_PX
        shifts = []
        for x, y, ref in self._patches:
            # phaseCorrelate(window=...) mutates same-size input buffers; keep the reference immutable.
            patch = cv2.GaussianBlur(g[y : y + s, x : x + s], (5, 5), 1.0)
            current = cv2.multiply(patch - patch.mean(), self._window)
            (dx, dy), resp = cv2.phaseCorrelate(ref, current)
            if resp >= MIN_RESPONSE:
                shifts.append(PatchShift((x + s / 2, y + s / 2), (float(dx), float(dy)), float(resp)))
        motion = _similarity(shifts)
        observable = motion is not None
        camera_moved = False
        if motion is not None:
            c = np.array([p.centre for p in shifts])
            disp = np.linalg.norm(apply_h(motion, c) - c, axis=1)
            camera_moved = bool(disp.max() > CAMERA_SHIFT_PX)
        found = [(ref, find_blob(g, ref, TICK_SEARCH_PX)) for ref in self._ticks]
        found = [(ref, cur) for ref, cur in found if cur is not None]
        drift = None
        projector_moved = len(self._ticks) - len(found) >= 2
        if found:
            ref = np.array([r for r, _ in found])
            cur = np.array([c for _, c in found])
            drift = float(np.median(np.linalg.norm(cur - ref, axis=1)))
            expected = apply_h(motion, ref) if motion is not None else ref
            projector_moved |= bool(np.median(np.linalg.norm(cur - expected, axis=1)) > TICK_DRIFT_PX)
        self._streak = self._streak + 1 if (camera_moved or projector_moved) else 0
        return BumpCheck(
            moved=self._streak >= self._confirm,
            camera_moved=camera_moved,
            projector_moved=projector_moved,
            camera_observable=observable,
            camera_motion=motion,
            tick_drift_px=None if drift is None else round(drift, 2),
            patches=tuple(shifts),
            ticks_found=len(found),
            ticks_ref=len(self._ticks),
        )


def _similarity(shifts: list[PatchShift]) -> np.ndarray | None:
    """Rotation + scale + translation from >= 2 agreeing patches (RANSAC when there are more), else None."""
    if len(shifts) < 2:
        return None
    src = np.array([p.centre for p in shifts], dtype=np.float64)
    dst = src + np.array([p.shift for p in shifts], dtype=np.float64)
    M, inl = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC, ransacReprojThreshold=AGREE_PX)
    if M is None or inl is None or int(inl.sum()) < 2:
        return None
    return np.vstack([M, [0.0, 0.0, 1.0]])
