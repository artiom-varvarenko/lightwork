"""Calibration on the rig (PLAN §4.4, D5, D7; C7): flash the pattern, solve, add the mm scale and write
``.lightwork/calib/calib.json``; optional consented-crop flat field, bump re-solve and plate-height homography.
Screen-only guidance (no projector) records just the locked camera in ``.lightwork/calib/screen.json``.

Choreography: ``show_mode("pattern")`` -> ACK -> frames older than ACK + 120 ms are skipped (they may still show the
previous page) -> the next frames are detected until >= ``N_FRAMES`` frames together hold >= 6 markers (a hand may
cover some) or the 2 s budget (rig:C2) runs out -> per-marker corners averaged over frames -> one RANSAC solve.
``calib.json`` is canonical JSON, so its file sha equals ``canonical_sha`` of the model.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol

import numpy as np

from ...config import load_settings
from ...contracts import canonical_bytes, sha256_hex
from ...contracts.layout import Calibration, CameraInfo, MmScale, PageInfo
from ...contracts.render import Ack, Hello, Pattern
from ...geometry import apply_h
from ..measure_frames import FlatField, capture_flat_field
from ..source import Frame
from .bump import AGREE_PX, BumpCheck
from .mm import CardFit, fit_card, find_id1_card, nominal_scale, quad_scale
from .pattern import PATTERN_IDS, pattern_for_page
from .solve import MIN_MARKERS, CalibrationError, Solution, clipped_fraction, detect_markers, fit, merge

SETTLE_S = 0.12  # D7
N_FRAMES = 3
BUDGET_S = 2.0  # rig:C2
ACK_TIMEOUT_S = 1.0
MAX_RMS_PX = 1.5  # rig:C2
DEFAULT_MOUNT_MM = 500.0
LIT_MODE = "grey"  # frame for finding printed references: lights the bench without clipping it
MM_SOURCES = ("auto", "card", "idcard", "nominal")
CALIB_DIR, CALIB_FILE, SCREEN_CALIB_FILE = "calib", "calib.json", "screen.json"
CLIP_NOTE = 0.01  # share of clipped marker pixels worth a warning


class PageLink(Protocol):  # J's RenderHub
    def show_mode(self, mode: str, pattern: Pattern | None = None, timeout: float = 1.0) -> float | None: ...

    def last_ack(self) -> Ack | None: ...


class Camera(Protocol):  # A's LiveCamera
    def latest(self) -> Frame | None: ...

    def wait_frame(self, after_t: float, timeout: float) -> Frame | None: ...

    def describe(self) -> dict: ...


# ---------------------------------------------------------------------------------------------------- storage


def calib_dir(state_dir: Path | None = None) -> Path:
    return (state_dir if state_dir is not None else load_settings().state_dir) / CALIB_DIR


def calib_path(state_dir: Path | None = None) -> Path:
    return calib_dir(state_dir) / CALIB_FILE


def screen_calib_path(state_dir: Path | None = None) -> Path:
    return calib_dir(state_dir) / SCREEN_CALIB_FILE


def flat_field_path(calib_file: Path, sha: str) -> Path:
    """The flat-field JPEG named by ``flat_field_sha`` sits next to ``calib.json``; runs store it as
    ``frames/<sha>.jpg``."""
    return calib_file.parent / f"flat-{sha}.jpg"


def save_calibration(calib: Calibration, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_bytes(canonical_bytes(calib))
    tmp.replace(path)
    return path


def load_calibration(path: Path | None = None) -> Calibration:
    path = path or calib_path()
    if not path.is_file():
        raise FileNotFoundError(f"no calibration at {path}: run `lightwork calibrate`")
    return Calibration.model_validate_json(path.read_bytes())


def _now_utc() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def screen_calibration(cam: Camera) -> Calibration:
    """C7 for guidance shown only on the screen: the camera identity and control readback that live checks before
    every capture. No projector is involved, so there is no mapping, residual, flat field or mm scale to record, and
    live never projects or switches measurement modes with it. An incomplete lock is refused: live would reject it."""
    report = getattr(cam, "report", None)
    if report is not None and not report.ok:
        raise CalibrationError(f"{getattr(cam, 'status', 'camera lock incomplete')}; fix the camera lock first")
    if cam.latest() is None:
        raise CalibrationError("camera gave no frame")
    info = cam.describe()
    return Calibration(
        H_cam_to_page=None, page=None, residuals=None,
        camera=CameraInfo(device_id=info["device_id"], mode=info["mode"], controls=info["controls"]),
        mm=MmScale(source="nominal"), measurement_mode="normal", created=_now_utc(),
    )


# ---------------------------------------------------------------------------------------------------- pattern flash


@dataclass(frozen=True)
class Flash:
    solution: Solution
    n_frames: int
    t_s: float  # pattern requested -> solved


def flash_pattern(cam: Camera, link: PageLink, page_w: float, page_h: float, *, settle_s: float = SETTLE_S, n_frames: int = N_FRAMES, budget_s: float = BUDGET_S) -> Flash:
    """Show the pattern and solve. Leaves the pattern up; the caller switches the page to its next mode."""
    t0 = time.monotonic()
    pattern = pattern_for_page(page_w, page_h)
    t_ack = link.show_mode("pattern", pattern, timeout=ACK_TIMEOUT_S)
    if t_ack is None:
        raise CalibrationError("the projector page did not ACK the pattern (is /projector open? `lightwork tank up`)")
    ack = link.last_ack()
    if ack is None or not ack.drawn:
        raise CalibrationError("the projector ACK carries no drawn marker rects")
    drawn = list(ack.drawn)
    ids = set(PATTERN_IDS)
    detections: list[dict[int, np.ndarray]] = []
    last: np.ndarray | None = None
    after = t_ack + settle_s
    deadline = t0 + budget_s
    while (left := deadline - time.monotonic()) > 0:
        frame = cam.wait_frame(after, timeout=left)
        if frame is None:
            break
        after = frame.t_mono
        detections.append(detect_markers(frame.image, ids))
        last = frame.image
        if len(detections) >= n_frames and len(merge(detections)) >= MIN_MARKERS:
            break
    if last is None:
        raise CalibrationError(f"no camera frame within {budget_s:g} s of the pattern")
    sol = fit(merge(detections), drawn)
    sol = replace(sol, clipped=clipped_fraction(last, sol.corners))
    return Flash(sol, len(detections), round(time.monotonic() - t0, 3))


def lit_frame(cam: Camera, link: PageLink, *, settle_s: float = SETTLE_S) -> np.ndarray:
    t = link.show_mode(LIT_MODE, timeout=ACK_TIMEOUT_S)
    if t is None:
        raise CalibrationError(f"the projector page did not ACK mode {LIT_MODE!r}")
    frame = cam.wait_frame(t + settle_s, timeout=ACK_TIMEOUT_S)
    if frame is None:
        raise CalibrationError("no camera frame")
    return frame.image


def _check_rms(sol: Solution, max_rms_px: float) -> None:
    if sol.residuals.rms_px > max_rms_px:
        raise CalibrationError(f"residual rms {sol.residuals.rms_px} page px > {max_rms_px} (blur, a moving hand or a bent board?)")


# ---------------------------------------------------------------------------------------------------- calibrate


@dataclass(frozen=True)
class CalibrationRun:
    calib: Calibration
    path: Path
    flash: Flash
    card: CardFit | None
    flat_field: FlatField | None
    t_s: float
    notes: tuple[str, ...]


def run_calibration(
    cam: Camera,
    link: PageLink,
    page: Hello,
    *,
    measurement_mode: str = "grey",
    mm: str = "auto",
    mount_mm: float = DEFAULT_MOUNT_MM,
    flat_field: bool = False,
    work_area_xywh: tuple[int, int, int, int] | None = None,
    max_rms_px: float = MAX_RMS_PX,
    path: Path | None = None,
) -> CalibrationRun:
    """Write C7. Flat-field capture is opt-in and needs an explicit privacy crop, on an empty board."""
    if mm not in MM_SOURCES:
        raise ValueError(f"mm must be one of {MM_SOURCES}")
    if measurement_mode not in ("grey", "dark"):
        raise ValueError("measurement_mode must be grey or dark (D7)")
    if flat_field and work_area_xywh is None:
        raise ValueError("flat-field storage needs work_area_xywh (D10 privacy)")
    path = path or calib_path()
    t0 = time.monotonic()
    notes: list[str] = []
    flat: FlatField | None = None
    try:
        flash = flash_pattern(cam, link, page.innerWidth, page.innerHeight)
        _check_rms(flash.solution, max_rms_px)
        card = None
        scale = nominal_scale(flash.solution.H, page.innerWidth, mount_mm)
        if mm != "nominal":
            lit = lit_frame(cam, link)
            if mm in ("auto", "card"):
                card = fit_card(lit)
                if card is not None:
                    scale = card.scale
                elif mm == "card":
                    raise CalibrationError("calibration card (ArUco 40-43) not found: >= 3 of its markers must be visible")
            else:
                found = find_id1_card(lit)
                if found is None:
                    raise CalibrationError("no ID-1 card found: put a card that contrasts with the board in view")
                scale = quad_scale(found[0], found[1], "idcard")
    finally:
        link.show_mode("normal", timeout=ACK_TIMEOUT_S)
    if flash.solution.clipped is not None and flash.solution.clipped > CLIP_NOTE:
        notes.append(f"{flash.solution.clipped:.1%} of the marker pixels are clipped: lower the exposure or the page white")
    cam_info = cam.describe()
    calib = Calibration(
        H_cam_to_page=flash.solution.H.tolist(),
        page=PageInfo(w=page.innerWidth, h=page.innerHeight, dpr=page.dpr),
        camera=CameraInfo(device_id=cam_info["device_id"], mode=cam_info["mode"], controls=cam_info["controls"]),
        residuals=flash.solution.residuals,
        mm=scale,
        flat_field_sha=None,
        measurement_mode=measurement_mode,
        created=_now_utc(),
    )
    if flat_field:
        calib, flat = add_flat_field(cam, link, calib, path=path, work_area_xywh=work_area_xywh)
        if flat is None:
            notes.append("flat field: the page did not ACK or the camera gave no frames")
        elif not flat.verified:
            notes.append("flat field not verified: " + "; ".join(flat.reasons))
    save_calibration(calib, path)
    return CalibrationRun(calib, path, flash, card, flat, round(time.monotonic() - t0, 3), tuple(notes))


def calibrate(cam: Camera, link: PageLink, page: Hello, **kw) -> Calibration:
    """Binding interface D: calibrate and write ``.lightwork/calib/calib.json``. Raises :class:`CalibrationError`."""
    return run_calibration(cam, link, page, **kw).calib


def add_flat_field(
    cam: Camera, link: PageLink, calib: Calibration, *, path: Path, work_area_xywh: tuple[int, int, int, int]
) -> tuple[Calibration, FlatField | None]:
    """Store only the consenting user's empty work-area crop, with its camera origin in C7.

    A rejected capture leaves the existing calibration unchanged. Remove printed references before calling.
    """
    first = cam.latest()
    if first is None:
        return calib, None
    h, w = first.image.shape[:2]
    x, y, cw, ch = work_area_xywh
    if x < 0 or y < 0 or cw <= 0 or ch <= 0 or x + cw > w or y + ch > h:
        raise ValueError("work area must fit inside the camera frame")
    flat = capture_flat_field(cam, link, calib.measurement_mode, crop_xywh=work_area_xywh)
    if flat is None or not flat.verified:
        return calib, flat
    sha = sha256_hex(flat.jpeg)
    out = flat_field_path(path, sha)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(flat.jpeg)
    new = calib.model_copy(update={"flat_field_sha": sha, "flat_field_crop_xywh": list(flat.crop_xywh)})
    save_calibration(new, path)
    return new, flat


def calibrate_plate_height(cam: Camera, link: PageLink, page: Hello, calib: Calibration, *, path: Path, max_rms_px: float = MAX_RMS_PX) -> tuple[Calibration, Flash]:
    """Second homography with the board raised to plate height (rig:C4, parallax > 3 mm); stored in
    ``H_plate_cam_to_page`` next to the table solve."""
    try:
        flash = flash_pattern(cam, link, page.innerWidth, page.innerHeight)
    finally:
        link.show_mode("normal", timeout=ACK_TIMEOUT_S)
    _check_rms(flash.solution, max_rms_px)
    new = calib.model_copy(update={"H_plate_cam_to_page": flash.solution.H.tolist()})
    save_calibration(new, path)
    return new, flash


def with_mm(calib: Calibration, scale: MmScale, path: Path) -> Calibration:
    """Replace the mm scale (e.g. from a plate found during layout) and rewrite ``calib.json``."""
    new = calib.model_copy(update={"mm": scale})
    save_calibration(new, path)
    return new


# ---------------------------------------------------------------------------------------------------- re-solve


@dataclass(frozen=True)
class Recalibration:
    calib: Calibration
    cause: str  # camera | projector | both | unobserved (layout must be checked again)
    camera_motion: np.ndarray | None  # old -> new camera px on the bench; None when unchanged or unobservable
    flash: Flash
    t_s: float


def recalibrate(cam: Camera, link: PageLink, old: Calibration, page: Hello, bump: BumpCheck, *, path: Path, max_rms_px: float = MAX_RMS_PX) -> Recalibration:
    """Re-solve after a confirmed bump and carry the mm scale across it; invalidate flat/raised-plane captures.

    If the camera moved and the projector stayed, ``inv(H_new) @ H_old`` is the exact camera motion on the bench;
    if the patches disagree with it, both moved and the patch fit is used. Camera-px state elsewhere (layout,
    work area) must be mapped through ``camera_motion`` by its owner.
    """
    t0 = time.monotonic()
    lit = None
    try:
        flash = flash_pattern(cam, link, page.innerWidth, page.innerHeight)
        if old.mm.source == "card":
            lit = lit_frame(cam, link)
    finally:
        link.show_mode("normal", timeout=ACK_TIMEOUT_S)
    _check_rms(flash.solution, max_rms_px)
    H_old, H_new = np.asarray(old.H_cam_to_page), flash.solution.H
    via_page = np.linalg.inv(H_new) @ H_old
    motion: np.ndarray | None = None
    if not bump.camera_observable:
        cause = "unobserved"
    elif not bump.camera_moved:
        cause = "projector"
    else:
        centres = np.array([p.centre for p in bump.patches])
        gap = np.linalg.norm(apply_h(via_page, centres) - apply_h(bump.camera_motion, centres), axis=1).max()
        cause, motion = ("camera", via_page) if gap <= 1.5 * AGREE_PX else ("both", bump.camera_motion)
    card = fit_card(lit) if lit is not None else None
    old_mm = None if old.mm.H_cam_to_mm is None else np.asarray(old.mm.H_cam_to_mm)
    if card is not None:
        scale = card.scale
        if motion is not None and old_mm is not None:
            motion = np.linalg.inv(np.asarray(card.scale.H_cam_to_mm)) @ old_mm  # exact on the card plane
    elif old_mm is None:
        scale = old.mm
    elif old.mm.source == "nominal":  # page-anchored: keep page -> mm
        scale = MmScale(source="nominal", H_cam_to_mm=(old_mm @ np.linalg.inv(H_old) @ H_new).tolist())
    elif motion is not None:  # bench-anchored references stay put while the camera moves
        scale = MmScale(source=old.mm.source, H_cam_to_mm=(old_mm @ np.linalg.inv(motion)).tolist())
    elif cause == "unobserved":
        scale = MmScale(source="nominal", H_cam_to_mm=None)
    else:
        scale = old.mm
    plate = None
    # A table-plane correction cannot recover a different plane's mapping after a physical bump.
    # Re-measure the raised plane instead of silently carrying an approximate H across it.
    new = old.model_copy(update={"H_cam_to_page": H_new.tolist(), "residuals": flash.solution.residuals, "mm": scale,
                                 "H_plate_cam_to_page": plate, "flat_field_sha": None,
                                 "flat_field_crop_xywh": None, "created": _now_utc()})
    save_calibration(new, path)
    return Recalibration(new, cause, motion, flash, round(time.monotonic() - t0, 3))
