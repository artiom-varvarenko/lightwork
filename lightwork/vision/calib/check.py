"""rig:C1 accuracy check (rig brief §8): project a ring on each printed cross of the card, find each ring's centre in
the camera by on/off difference, map it to mm through the card and compare with the cross.

This is the README accuracy figure: in-sample residuals are optimistic (rig brief §4.3). It measures the whole
projector-camera chain on the bench plane; the card's own fit is cross-checked with a ruler by hand.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from ...contracts import q_px, q_xy
from ...contracts.layout import Calibration
from ...contracts.render import Calib, Ring, SceneMsg
from ...geometry import apply_h
from .flow import ACK_TIMEOUT_S, SETTLE_S, Camera, PageLink, lit_frame
from .mm import CARD, CardSpec, fit_card
from .solve import CalibrationError, gray

RING_MM = 10.0
STROKE_MM = 3.5  # 5 projector px at up to 0.65 m, rounded up
MEAN_MM, MAX_MM = 2.0, 3.0  # rig:C1 pass criteria


class SceneLink(PageLink, Protocol):
    def publish_scene(self, scene: SceneMsg) -> None: ...

    def set_calib(self, calib: Calib) -> None: ...


@dataclass(frozen=True)
class AccuracyReport:
    errors_mm: tuple[float | None, ...]  # per cross in CardSpec.crosses order; None = ring not found
    mean_mm: float | None
    max_mm: float | None
    card_rms_mm: float
    passed: bool


def calib_message(calib: Calibration) -> Calib:
    return Calib(H_cam_to_page=calib.H_cam_to_page, page_size=[calib.page.w, calib.page.h])


def _frame_after(cam: Camera, link: SceneLink, scene: SceneMsg) -> np.ndarray:
    link.publish_scene(scene)
    t = link.show_mode("normal", timeout=ACK_TIMEOUT_S)
    if t is None:
        raise CalibrationError("the projector page did not ACK the scene")
    frame = cam.wait_frame(t + SETTLE_S, timeout=ACK_TIMEOUT_S)
    if frame is None:
        raise CalibrationError("no camera frame")
    return gray(frame.image).astype(np.float32)


def ring_centre(diff: np.ndarray, xy: np.ndarray, reach: float) -> np.ndarray | None:
    """Centroid of the projected light (on - off) within ``reach`` px of ``xy``; a ring's centroid is its centre."""
    h, w = diff.shape
    x0, y0 = max(0, int(xy[0] - reach)), max(0, int(xy[1] - reach))
    x1, y1 = min(w, int(xy[0] + reach) + 1), min(h, int(xy[1] + reach) + 1)
    win = diff[y0:y1, x0:x1]
    if win.size == 0 or float(win.max()) < 25.0:
        return None
    ys, xs = np.mgrid[y0:y1, x0:x1]
    wts = np.where((win > 0.3 * float(win.max())) & ((xs - xy[0]) ** 2 + (ys - xy[1]) ** 2 <= reach**2), win, 0.0)
    s = float(wts.sum())
    return np.array([float((wts * xs).sum()) / s, float((wts * ys).sum()) / s]) if s > 0 else None


def check_accuracy(cam: Camera, link: SceneLink, calib: Calibration, *, spec: CardSpec = CARD) -> AccuracyReport:
    link.set_calib(calib_message(calib))
    try:
        card = fit_card(lit_frame(cam, link), spec)
        if card is None:
            raise CalibrationError("calibration card (ArUco 40-43) not found: >= 3 of its markers must be visible")
        to_mm = np.asarray(card.scale.H_cam_to_mm)
        to_cam = np.linalg.inv(to_mm)
        crosses = np.asarray(spec.crosses, dtype=np.float64)
        targets = apply_h(to_cam, crosses)
        radii = np.linalg.norm(apply_h(to_cam, crosses + [RING_MM, 0.0]) - targets, axis=1)
        off = _frame_after(cam, link, SceneMsg(seq=0, mode="normal", prims=[]))
        rings = [Ring(id=f"c1-{i}", c=q_xy(t), r=q_px(r), role="info") for i, (t, r) in enumerate(zip(targets, radii))]
        on = _frame_after(cam, link, SceneMsg(seq=0, mode="normal", prims=rings))
    finally:
        link.publish_scene(SceneMsg(seq=0, mode="normal", prims=[]))
        link.show_mode("normal", timeout=ACK_TIMEOUT_S)
    diff = on - off
    # A ring drawn up to 2 x MAX_MM off target still falls inside its search window.
    cam_per_mm = float(np.median(radii)) / RING_MM
    errors: list[float | None] = []
    for ring, x_mm in zip(rings, crosses):
        reach = ring.r + cam_per_mm * (2 * MAX_MM + STROKE_MM)
        c = ring_centre(diff, np.asarray(ring.c), reach)
        errors.append(None if c is None else round(float(np.linalg.norm(apply_h(to_mm, [c])[0] - x_mm)), 3))
    got = [e for e in errors if e is not None]
    mean = round(float(np.mean(got)), 3) if got else None
    worst = round(max(got), 3) if got else None
    passed = bool(got) and len(got) == len(errors) and mean <= MEAN_MM and worst <= MAX_MM
    return AccuracyReport(tuple(errors), mean, worst, card.rms_mm, passed)
