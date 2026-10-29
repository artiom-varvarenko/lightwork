"""Measurement frames: keyframes without our overlay in them (PLAN.md §4.4, D7; ``briefs/rig.md`` §5.1-§5.2).

:func:`capture_keyframe` choreography:

1. Keep the newest normal-mode camera frame as the reference and tell the gate's :class:`OverlayMask` that the bench
   is about to be lit (the gate ignores those frames).
2. ``link.show_mode('grey'|'dark')``: the page draws the mode and ACKs after ``requestAnimationFrame``.
3. Discard every camera frame dequeued before ACK + ``settle_ms`` (120 ms: compositor, DLP, exposure and rolling
   readout), then take frames until two consecutive ones pass content verification (at most ``MAX_FRAMES``).
4. Restore the overlay (``show_mode('normal')``) before any encoding, so the bench is lit as briefly as possible.
5. Encode the work-area crop of the pair's mean as JPEG q95 with 4:4:4 chroma (half the OD error of 4:2:0 on
   20-30 px wells, for +35 % bytes).

Content verification never trusts timing alone. A pair passes when, inside the crop:

- exposure is sane: mean grey in [``EXPOSURE_MIN``, ``EXPOSURE_MAX``], at most ``CLIP_MAX`` of pixels clipped;
- the frames agree: per-band gains within ``AGREE_GAIN`` of 1 and at most ``AGREE_FRAC`` of pixels off by more than
  ``AGREE_DN`` after them (a rolling-shutter half frame or a moving hand fails);
- ``grey``: every row band is ``GREY_RISE`` brighter in linear light than the normal-mode reference (the page lit it);
- ``grey``/``dark``: where the previous overlay's line contrast was >= ``OVERLAY_SIGNAL_DN`` against its surroundings,
  that contrast fell to ``OVERLAY_FRAC`` of it or below ``OVERLAY_TOL_DN``; calibration marker IDs 0-14 are absent;
- ``normal`` (setups without a projector): no overlay is drawn inside the crop.

:func:`capture_flat_field` grabs the empty board in the measurement mode for E's flat-field correction (C7
``flat_field_sha``). Rig checks rig:M1/M2 run from this module: ``python -m lightwork.vision.measure_frames --help``.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Protocol

import cv2
import numpy as np

from ..contracts import render
from ..contracts.canonical import q_s
from .gate import band_gain
from .overlay_mask import OverlayMask
from .source import Frame

KEYFRAME_MODES = ("grey", "dark", "normal")  # C1 keyframe.mode
FLAT_MODES = ("grey", "dark")
SETTLE_MS = 120
MAX_FRAMES = 4  # frames looked at after the settle for one verified pair
ACK_TIMEOUT_S = 1.0
FRAME_TIMEOUT_S = 0.25  # camera silent this long: give up
JPEG_PARAMS = [cv2.IMWRITE_JPEG_QUALITY, 95, cv2.IMWRITE_JPEG_SAMPLING_FACTOR, cv2.IMWRITE_JPEG_SAMPLING_FACTOR_444]

CHECK_WIDTH = 480  # agreement and brightness checks run on the crop downscaled to about this width
CHECK_BANDS = 6  # row bands, so a rolling-shutter transition anywhere in the frame is caught
EXPOSURE_MIN, EXPOSURE_MAX = 10.0, 240.0
CLIP_DN, CLIP_MAX = 250, 0.05
AGREE_GAIN, AGREE_DN, AGREE_FRAC = 0.10, 10.0, 0.005
GREY_RISE = 1.15
OVERLAY_SIGNAL_DN, OVERLAY_FRAC, OVERLAY_TOL_DN, OVERLAY_MIN_PX = 10.0, 0.3, 4.0, 30
FLAT_LIT_MIN = 0.15  # grey flat field: at least this much of the crop must be lit by the page
FLAT_SIGMA = 0.03  # flatness reference blur, fraction of the check width
FLAT_TOL, FLAT_FRAC = 0.08, 0.01  # at most FLAT_FRAC of the lit board may deviate more than FLAT_TOL from smooth


class PageLink(Protocol):
    """Implemented by ``lightwork.server.hub.RenderHub``."""

    def show_mode(self, mode: str, pattern: render.Pattern | None = None, timeout: float = 1.0) -> float | None:
        """Switch the page and return the laptop ``time.monotonic()`` of its ACK (after rAF), None on timeout."""
        ...


class FrameWaiter(Protocol):
    """The part of ``lightwork.vision.capture.LiveCamera`` used here."""

    def latest(self) -> Frame | None: ...

    def wait_frame(self, after_t: float, timeout: float) -> Frame | None: ...


@dataclass(frozen=True)
class Capture:
    image: np.ndarray  # BGR uint8 work-area crop: mean of the verified pair (the last frame seen when unverified)
    crop_xywh: tuple[int, int, int, int]  # camera px, clamped to the frame
    mode: str
    verified: bool
    reasons: tuple[str, ...]  # failed checks of the last pair tried; empty when verified
    t_capture: float  # t_mono of the newest frame used
    t_send: float  # clock() just before the mode switch
    t_ack: float | None  # ACK of the measurement mode (None for mode "normal")
    t_restore: float | None  # ACK of show_mode("normal"); None for mode "normal" or on timeout
    frames_seen: int  # frames looked at after the settle


@dataclass(frozen=True)
class FlatField:
    jpeg: bytes  # JPEG q95 of the mean of n frames (crop of the camera frame; crop None = the full frame)
    crop_xywh: tuple[int, int, int, int]
    mode: str
    verified: bool
    reasons: tuple[str, ...]


def encode_jpeg(img: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(".jpg", img, JPEG_PARAMS)
    if not ok:
        raise RuntimeError("JPEG encode failed")
    return buf.tobytes()


def _clamp_crop(crop_xywh: Sequence[int] | None, shape: tuple[int, ...]) -> tuple[int, int, int, int]:
    H, W = shape[:2]
    x, y, w, h = (0, 0, W, H) if crop_xywh is None else (int(v) for v in crop_xywh)
    x0, y0, x1, y1 = max(0, x), max(0, y), min(W, x + w), min(H, y + h)
    if x1 <= x0 or y1 <= y0:
        raise ValueError(f"crop {crop_xywh} lies outside the {W}x{H} camera frame")
    return x0, y0, x1 - x0, y1 - y0


def _lin(dn: np.ndarray | float) -> np.ndarray:
    v = np.asarray(dn, dtype=np.float64) / 255.0
    return np.where(v <= 0.04045, v / 12.92, ((v + 0.055) / 1.055) ** 2.4)


def _crop(img: np.ndarray, crop: tuple[int, int, int, int]) -> np.ndarray:
    x, y, w, h = crop
    return img[y : y + h, x : x + w]


def _bands(n_rows: int) -> list[tuple[int, int]]:
    edges = np.linspace(0, n_rows, CHECK_BANDS + 1).round().astype(int)
    return [(int(a), int(b)) for a, b in zip(edges[:-1], edges[1:]) if b > a]


class _Checks:
    """Content checks for one capture; per-frame derived images are cached (each frame is in two pairs)."""

    def __init__(self, mode: str, crop: tuple[int, int, int, int], before: Frame | None, drawn: np.ndarray | None, dilate_px: int):
        self.mode = mode
        self.crop = crop
        w, h = crop[2], crop[3]
        self._f = max(1, w // CHECK_WIDTH)
        self._small_size = (max(1, w // self._f), max(1, h // self._f))
        self._small_bands = _bands(self._small_size[1])
        self._full_bands = _bands(h)
        self._cache: dict[tuple[int, float, str], np.ndarray] = {}
        self._patterns: dict[tuple[int, float], str | None] = {}
        self._detector = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50))
        self._inside = self._ring = None
        self.overlay_in_crop = False
        if drawn is not None:
            d = _crop(drawn, crop) > 0
            self.overlay_in_crop = bool(d.any())
            if self.overlay_in_crop and mode in FLAT_MODES:
                u8 = d.view(np.uint8)
                near = cv2.dilate(u8, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * dilate_px + 1,) * 2))
                far = cv2.dilate(u8, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (4 * dilate_px + 1,) * 2))
                self._inside, self._ring = d, (far > 0) & (near == 0)
        self._before = before

    def small(self, f: Frame) -> np.ndarray:
        key = (f.idx, f.t_mono, "small")
        if key not in self._cache:
            c = _crop(f.image, self.crop)
            if self._f > 1:
                c = cv2.resize(c, self._small_size, interpolation=cv2.INTER_AREA)
            self._cache[key] = cv2.GaussianBlur(cv2.cvtColor(c, cv2.COLOR_BGR2GRAY).astype(np.float32), (5, 5), 1.0)
        return self._cache[key]

    def full(self, f: Frame) -> np.ndarray:
        key = (f.idx, f.t_mono, "full")
        if key not in self._cache:
            self._cache[key] = cv2.cvtColor(_crop(f.image, self.crop), cv2.COLOR_BGR2GRAY).astype(np.float32)
        return self._cache[key]

    def exposure(self, f: Frame) -> str | None:
        c = _crop(f.image, self.crop)
        mean = float(self.small(f).mean())
        if not EXPOSURE_MIN <= mean <= EXPOSURE_MAX:
            return f"exposure: mean grey {mean:.0f} outside [{EXPOSURE_MIN:.0f}, {EXPOSURE_MAX:.0f}]"
        clipped = float(np.count_nonzero(c.max(axis=2) >= CLIP_DN)) / (c.shape[0] * c.shape[1])
        if clipped > CLIP_MAX:
            return f"exposure: {100 * clipped:.1f}% of pixels clipped"
        return None

    def agree(self, f1: Frame, f2: Frame) -> str | None:
        a, b = self.small(f1), self.small(f2)
        gain = band_gain(b, a)
        off = np.abs(gain - 1.0) > AGREE_GAIN
        if off.any():
            rows = np.nonzero(off)[0] * self._f
            return f"frames disagree: brightness {100 * (float(gain[off].mean()) - 1):+.0f}% in rows {rows[0]}-{rows[-1]}"
        frac = float(np.count_nonzero(np.abs(b - gain[:, None] * a) > AGREE_DN)) / a.size
        if frac > AGREE_FRAC:
            return f"frames disagree: {100 * frac:.1f}% of pixels changed"
        return None

    def lit(self, f: Frame) -> str | None:
        """grey: every band brighter than the normal-mode reference by GREY_RISE (linear light)."""
        if self._before is None:
            return "no normal-mode reference frame"
        m, b = self.small(f), self.small(self._before)
        for r0, r1 in self._small_bands:
            rise = float(_lin(np.median(m[r0:r1])) / max(float(_lin(np.median(b[r0:r1]))), 1e-6))
            if rise < GREY_RISE:
                return f"page did not light rows {r0 * self._f}-{r1 * self._f}: x{rise:.2f} vs normal mode"
        return None

    def overlay_gone(self, f: Frame) -> str | None:
        """The previous overlay's line contrast is gone in every band where it showed."""
        if self._inside is None or self._before is None:
            return None
        m, b = self.full(f), self.full(self._before)
        for r0, r1 in self._full_bands:
            ins, ring = self._inside[r0:r1], self._ring[r0:r1]
            if np.count_nonzero(ins) < OVERLAY_MIN_PX or np.count_nonzero(ring) < OVERLAY_MIN_PX:
                continue
            c0 = float(b[r0:r1][ins].mean() - b[r0:r1][ring].mean())
            if c0 < OVERLAY_SIGNAL_DN:
                continue
            c1 = float(m[r0:r1][ins].mean() - m[r0:r1][ring].mean())
            if c1 > max(OVERLAY_TOL_DN, OVERLAY_FRAC * c0):
                return f"overlay still visible in rows {r0}-{r1}: contrast {c1:.0f} DN ({c0:.0f} in normal mode)"
        return None

    def pattern(self, f: Frame) -> str | None:
        key = (f.idx, f.t_mono)
        if key not in self._patterns:
            # IDs 40-43 on the optional printed scale card are legitimate; projected calibration uses IDs 0-14.
            g = cv2.cvtColor(_crop(f.image, self.crop), cv2.COLOR_BGR2GRAY)
            if g.shape[1] > 960:
                g = cv2.resize(g, (960, max(1, round(g.shape[0] * 960 / g.shape[1]))), interpolation=cv2.INTER_AREA)
            _, ids, _ = self._detector.detectMarkers(g)
            self._patterns[key] = "calibration pattern still visible" if ids is not None and (ids < 15).any() else None
        return self._patterns[key]

    def pair(self, f1: Frame, f2: Frame) -> tuple[str, ...]:
        found: list[str] = []
        for r in (self.exposure(f1), self.exposure(f2), self.agree(f1, f2), self.pattern(f1), self.pattern(f2)):
            if r:
                found.append(r)
        if self.mode == "grey":
            r = self.lit(f1) or self.lit(f2)
            if r:
                found.append(r)
        if self.mode in FLAT_MODES:
            r = self.overlay_gone(f1) or self.overlay_gone(f2)
            if r:
                found.append(r)
        elif self.overlay_in_crop:
            found.append("overlay drawn inside the crop (mode normal)")
        return tuple(found)


def _take(cam: FrameWaiter, after_t: float, n: int, max_frames: int, pair_ok: Callable[[Frame, Frame], tuple[str, ...]]) -> tuple[list[Frame], int, tuple[str, ...]]:
    """Consecutive frames dequeued after ``after_t`` until ``n`` in a row pass ``pair_ok`` pairwise.
    Returns (the best run, frames seen, reasons the last failing pair gave)."""
    run: list[Frame] = []
    best: list[Frame] = []
    seen = 0
    reasons: tuple[str, ...] = ("camera gave no frame after the page ACK",)
    while seen < max_frames:
        f = cam.wait_frame(after_t, FRAME_TIMEOUT_S)
        if f is None:
            break
        if f.t_mono <= after_t:
            raise ValueError("camera.wait_frame returned a stale frame")
        seen += 1
        after_t = f.t_mono
        if run:
            bad = pair_ok(run[-1], f)
            if bad:
                reasons = bad
                run = []
        run.append(f)
        if len(run) >= len(best):
            best = list(run)
        if len(run) >= n:
            return run, seen, ()
    if seen == 1:
        reasons = ("only one frame after the page ACK",)
    return best, seen, reasons


def _mean(frames: list[Frame], crop: tuple[int, int, int, int]) -> np.ndarray:
    if len(frames) == 1:
        return _crop(frames[0].image, crop).copy()
    acc = np.zeros((crop[3], crop[2], 3), np.float32)
    for f in frames:
        acc += _crop(f.image, crop)
    return np.clip(np.round(acc / len(frames)), 0, 255).astype(np.uint8)


def _restore(link: PageLink, overlay: OverlayMask | None) -> float | None:
    """Restore normal mode. Keep the gate held if the page does not ACK: it may still light the whole board."""
    t = link.show_mode("normal", timeout=ACK_TIMEOUT_S)
    if overlay is not None and t is not None:
        overlay.release(t)
    return t


def _normal_reference(cam: FrameWaiter, overlay: OverlayMask | None) -> Frame | None:
    """Newest frame that no longer shows an earlier measurement mode."""
    f = cam.latest()
    clear = overlay.clear_after() if overlay is not None else None
    if clear is None or f is None or f.t_mono > clear:
        return f
    if clear == float("inf"):
        raise RuntimeError("a measurement-mode capture is already in progress")
    return cam.wait_frame(clear, clear - f.t_mono + FRAME_TIMEOUT_S)


def measure_frame(
    cam: FrameWaiter,
    link: PageLink,
    mode: str,
    crop_xywh: Sequence[int],
    *,
    settle_ms: float = SETTLE_MS,
    overlay: OverlayMask | None = None,
    max_frames: int = MAX_FRAMES,
    clock: Callable[[], float] = time.monotonic,
) -> Capture | None:
    """Run the keyframe choreography and return the crop with its verification; None when the page did not ACK the
    mode or the camera gave no frame. Captures must not overlap (one capture thread)."""
    if mode not in KEYFRAME_MODES:
        raise ValueError(f"mode must be one of {KEYFRAME_MODES}, got {mode!r}")
    if settle_ms < 0 or max_frames < 2:
        raise ValueError("settle_ms must be nonnegative and max_frames at least 2")
    before = _normal_reference(cam, overlay)
    if before is None:
        return None
    crop = _clamp_crop(crop_xywh, before.image.shape)
    t_send = clock()
    drawn = overlay.drawn_at(t_send) if overlay is not None else None
    checks = _Checks(mode, crop, before, drawn, overlay.dilate_px if overlay is not None else 0)
    t_ack = t_restore = None
    if mode == "normal":
        frames, seen, reasons = _take(cam, before.t_mono, 2, max_frames, checks.pair)
    else:
        frames, seen, reasons = [], 0, ()
        if overlay is not None:
            overlay.hold(t_send)
        try:
            t_ack = link.show_mode(mode, timeout=ACK_TIMEOUT_S)
            if t_ack is not None:
                frames, seen, reasons = _take(cam, t_ack + settle_ms / 1000.0, 2, max_frames, checks.pair)
        finally:
            t_restore = _restore(link, overlay)
        if t_ack is None:
            return None
    if not frames:
        return None
    verified = len(frames) >= 2 and not reasons
    used = frames if verified else frames[-1:]
    return Capture(
        image=_mean(used, crop),
        crop_xywh=crop,
        mode=mode,
        verified=verified,
        reasons=reasons,
        t_capture=used[-1].t_mono,
        t_send=t_send,
        t_ack=t_ack,
        t_restore=t_restore,
        frames_seen=seen,
    )


def capture_keyframe(
    cam: FrameWaiter,
    link: PageLink,
    mode: str,
    crop_xywh: Sequence[int],
    *,
    settle_ms: float = SETTLE_MS,
    overlay: OverlayMask | None = None,
    t0: float = 0.0,
    clock: Callable[[], float] = time.monotonic,
) -> tuple[bytes, dict] | None:
    """JPEG q95 of the crop in measurement ``mode`` plus the C1 ``keyframe`` payload minus ``frame_sha``:
    ``{crop_xywh, mode, verified, t_capture}``, with ``t_capture = q_s(frame t_mono - t0)`` (pass the run's start
    ``t0`` to get run time). Pass the gate's ``overlay`` so the gate ignores the lit frames and the dark check knows
    where the overlay was. None when the page did not ACK or the camera gave no frame."""
    cap = measure_frame(cam, link, mode, crop_xywh, settle_ms=settle_ms, overlay=overlay, clock=clock)
    if cap is None:
        return None
    payload = {"crop_xywh": list(cap.crop_xywh), "mode": cap.mode, "verified": cap.verified, "t_capture": q_s(cap.t_capture - t0)}
    return encode_jpeg(cap.image), payload


def _flat_reasons(img: np.ndarray, dark: np.ndarray | None, mode: str) -> list[str]:
    f = max(1, img.shape[1] // CHECK_WIDTH)
    size = (max(1, img.shape[1] // f), max(1, img.shape[0] // f))

    def small(a: np.ndarray) -> np.ndarray:
        s = cv2.resize(a, size, interpolation=cv2.INTER_AREA) if f > 1 else a
        return cv2.cvtColor(s, cv2.COLOR_BGR2GRAY).astype(np.float32)

    out: list[str] = []
    g = small(img)
    clipped = float(np.count_nonzero(img.max(axis=2) >= CLIP_DN)) / (img.shape[0] * img.shape[1])
    if clipped > CLIP_MAX:
        out.append(f"exposure: {100 * clipped:.1f}% of pixels clipped")
    sigma = max(1.0, FLAT_SIGMA * size[0])
    region = np.ones(g.shape, np.float32)
    if mode == "grey" and dark is not None:
        lit = (_lin(g) / np.maximum(_lin(small(dark)), 1e-6)) >= GREY_RISE
        if np.count_nonzero(lit) < FLAT_LIT_MIN * lit.size:
            out.append(f"page lit only {100 * np.count_nonzero(lit) / lit.size:.0f}% of the crop (projector off or not grey?)")
            return out
        contours, _ = cv2.findContours(lit.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        footprint = np.zeros(lit.shape, np.uint8)
        cv2.fillConvexPoly(footprint, cv2.convexHull(np.concatenate(contours)), 1)
        holes = np.count_nonzero((footprint > 0) & ~lit) / max(1, np.count_nonzero(footprint))
        if holes > FLAT_FRAC:
            out.append(f"board not flat: {100 * holes:.1f}% dark holes inside the projected field")
        # the footprint edge is blurred over a few px; judge flatness strictly inside it
        region = cv2.erode(lit.astype(np.uint8), np.ones((7, 7), np.uint8)).astype(np.float32)
    inside = region > 0
    if not inside.any():
        return out + ["no contiguous lit board area"]
    # smooth reference from the region only (normalized blur), so the lit/unlit edge does not leak into it
    smooth = cv2.GaussianBlur(g * region, (0, 0), sigma) / np.maximum(cv2.GaussianBlur(region, (0, 0), sigma), 1e-3)
    dev = np.abs(g - smooth) / np.maximum(smooth, 1.0)
    frac = float(np.count_nonzero((dev > FLAT_TOL) & inside)) / np.count_nonzero(inside)
    if frac > FLAT_FRAC:
        out.append(f"board not flat: {100 * frac:.1f}% of it deviates >{100 * FLAT_TOL:.0f}% (objects on the board?)")
    mean = float(g[inside].mean())
    if not EXPOSURE_MIN <= mean <= EXPOSURE_MAX:
        out.append(f"exposure: mean grey {mean:.0f} outside [{EXPOSURE_MIN:.0f}, {EXPOSURE_MAX:.0f}]")
    return out


def capture_flat_field(
    cam: FrameWaiter,
    link: PageLink,
    mode: str = "grey",
    *,
    crop_xywh: Sequence[int] | None = None,
    n: int = 5,
    settle_ms: float = SETTLE_MS,
    overlay: OverlayMask | None = None,
    clock: Callable[[], float] = time.monotonic,
) -> FlatField | None:
    """The empty board in measurement ``mode``: mean of ``n`` agreeing frames taken after ACK + ``settle_ms``.
    For ``grey`` the page first goes dark, which gives the reference for "the page lit the board". Verification:
    frames agree, exposure sane, lit area (grey) and the board flat (no objects). Restores ``normal``. None when
    the page did not ACK or the camera gave no frame. Call it on the empty board right after calibration."""
    if mode not in FLAT_MODES:
        raise ValueError(f"mode must be one of {FLAT_MODES}, got {mode!r}")
    if n < 2 or settle_ms < 0:
        raise ValueError("n must be at least 2 and settle_ms nonnegative")
    first = cam.latest()
    if first is None:
        return None
    crop = _clamp_crop(crop_xywh, first.image.shape)
    checks = _Checks(mode, crop, None, None, 0)

    def agree(f1: Frame, f2: Frame) -> tuple[str, ...]:
        return tuple(r for r in (checks.agree(f1, f2), checks.pattern(f1), checks.pattern(f2)) if r)

    settle = settle_ms / 1000.0
    dark: list[Frame] = []
    frames: list[Frame] = []
    reasons: tuple[str, ...] = ()
    if overlay is not None:
        overlay.hold(clock())
    try:
        if mode == "grey":
            t_ack = link.show_mode("dark", timeout=ACK_TIMEOUT_S)
            if t_ack is None:
                return None
            dark, _, dark_reasons = _take(cam, t_ack + settle, 2, 2 + MAX_FRAMES, agree)
            if not dark:
                return None
            reasons += tuple(f"dark reference: {r}" for r in dark_reasons)
        t_ack = link.show_mode(mode, timeout=ACK_TIMEOUT_S)
        if t_ack is None:
            return None
        frames, _, run_reasons = _take(cam, t_ack + settle, n, n + MAX_FRAMES, agree)
    finally:
        _restore(link, overlay)
    if not frames:
        return None
    if len(frames) < n:
        reasons += (f"only {len(frames)} of {n} frames agreed",) + run_reasons
    img = _mean(frames, crop)
    reasons += tuple(_flat_reasons(img, _mean(dark, crop) if dark else None, mode))
    return FlatField(jpeg=encode_jpeg(img), crop_xywh=crop, mode=mode, verified=not reasons, reasons=reasons)


def _rig_scene(crop: tuple[int, int, int, int], seq: int) -> render.SceneMsg:
    x, y, w, h = crop
    source = [x + 0.2 * w, y + 0.25 * h]
    target = [x + (0.35 + 0.1 * (seq % 5)) * w, y + 0.75 * h]
    radius = 0.075 * min(w, h)
    return render.SceneMsg(seq=seq, mode="normal", prims=[
        render.Ring(id="source", c=source, r=radius, role="source"),
        render.Ring(id="target", c=target, r=radius, role="target", pulse=True),
        render.Arrow(id="arrow", a=source, b=target, role="info"),
    ])


def _rig_m1(cam, hub, overlay, crop, seconds: float) -> dict:
    from .gate import Gate

    gate = Gate(None, crop)
    started = time.monotonic()
    next_scene, seq = started, 0
    events: list[dict] = []
    frames = triggers = moving = 0
    had_motion = False
    while time.monotonic() - started < seconds:
        now = time.monotonic()
        if now >= next_scene:
            seq += 1
            scene = _rig_scene(crop, seq)
            overlay.set_scene(scene, now)
            hub.publish_scene(scene)
            next_scene = now + 0.1
        frame = cam.read()
        if frame is None:
            raise RuntimeError("camera stopped during rig:M1")
        frames += 1
        for _, payload in gate.update(frame, overlay.mask_at(frame.t_mono)):
            events.append({"t": q_s(frame.t_mono - started), **payload})
            if payload["state"] == "moving":
                moving += 1
                had_motion = True
            elif had_motion:
                triggers += 1
                had_motion = False
    return {"test": "rig:M1", "seconds": q_s(time.monotonic() - started), "frames": frames,
            "moving_events": moving, "keyframe_triggers": triggers, "events": events,
            "passed": moving <= 1 and triggers == 0}


def _rig_m2(cam, hub, overlay, crop, modes, count: int, output) -> dict:
    results = []
    for mode in modes:
        samples = []
        for i in range(count):
            before = _normal_reference(cam, overlay)
            drawn = overlay.drawn_at(time.monotonic())
            cap = measure_frame(cam, hub, mode, crop, overlay=overlay)
            if cap is None:
                samples.append({"verified": False, "reasons": ["no ACK or camera frame"], "file": None})
                continue
            jpeg = encode_jpeg(cap.image)
            total_ms = round(1000 * (time.monotonic() - cap.t_send), 3)
            filename = f"{mode}-{i + 1:03d}.jpg"
            (output / filename).write_bytes(jpeg)
            decoded = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
            h, w = decoded.shape[:2]
            ref = None if before is None else Frame(_crop(before.image, crop), before.t_mono, before.idx)
            check = _Checks(mode, (0, 0, w, h), ref, None if drawn is None else _crop(drawn, crop), overlay.dilate_px)
            saved = Frame(decoded, cap.t_capture, -1)
            reasons = cap.reasons + check.pair(saved, saved)
            blank_ms = None if cap.t_restore is None else round(1000 * (cap.t_restore - cap.t_send), 3)
            samples.append({"verified": cap.verified and not reasons, "reasons": list(dict.fromkeys(reasons)),
                            "file": filename, "blank_ms": blank_ms, "total_ms": total_ms,
                            "frames_seen": cap.frames_seen})
            if cap.t_restore is None:
                raise RuntimeError("normal-mode restore was not acknowledged; gate remains masked")
        durations = [s["blank_ms"] for s in samples if s.get("blank_ms") is not None]
        p95 = round(float(np.percentile(durations, 95)), 3) if durations else None
        total_p95 = round(float(np.percentile([s["total_ms"] for s in samples if "total_ms" in s], 95)), 3) if durations else None
        verified = sum(s["verified"] for s in samples)
        results.append({"mode": mode, "n": count, "verified": verified, "blank_p95_ms": p95,
                        "total_p95_ms": total_p95, "samples": samples,
                        "passed": verified == count and p95 is not None and p95 <= 300 and total_p95 <= 300})
    return {"test": "rig:M2", "modes": results, "passed": all(r["passed"] for r in results)}


def main(argv: list[str] | None = None) -> int:
    """Hardware checks, no model calls. Only the selected work-area crop is saved.

    Start this module, then ``lightwork tank up`` and tap Start on the projector page. Stop any other Lightwork
    server first. Keep the board empty and hands out. ``m1`` animates for 120 s; ``m2`` saves 50 cropped JPEGs per
    mode and reports content checks plus blank/total p95. Review those crops for residual projection. Run dark
    mode only with >=150 lux ambient. The crop comes from C7's flat-field crop or an explicit --crop override;
    never from the full camera frame by default. Output stays private under .lightwork/checks/measurement.
    """
    import argparse
    import json
    import sys
    from pathlib import Path

    from ..contracts.layout import Calibration
    from ..server.app import create_app
    from ..server.hub import RenderHub
    from ..server.runner import start_server
    from .capture import open_camera

    parser = argparse.ArgumentParser(prog="python -m lightwork.vision.measure_frames", description=main.__doc__)
    parser.add_argument("test", choices=["m1", "m2"])
    parser.add_argument("--calib", type=Path, default=Path(".lightwork/calib/calib.json"))
    parser.add_argument("--crop", type=int, nargs=4, metavar=("X", "Y", "W", "H"), help="work-area crop in camera px (default: calibration's flat-field crop)")
    parser.add_argument("--camera", help="camera by-id path or video source (default: calibration's camera)")
    parser.add_argument("--seconds", type=float, default=120.0)
    parser.add_argument("--count", type=int, default=50)
    parser.add_argument("--mode", choices=["grey", "dark", "both"], default="both")
    parser.add_argument("--output", type=Path, default=Path(".lightwork/checks/measurement"))
    args = parser.parse_args(argv)
    if args.seconds <= 0 or args.count <= 0:
        parser.error("--seconds and --count must be positive")
    cam = server = None
    try:
        calib = Calibration.model_validate_json(args.calib.read_text())
        selected_crop = args.crop or calib.flat_field_crop_xywh
        if selected_crop is None:
            raise ValueError("calibration has no work-area crop; supply --crop X Y W H")
        cam = open_camera(args.camera or calib.camera.device_id)
        first = cam.read()
        if first is None:
            raise RuntimeError("camera gave no frame")
        crop = _clamp_crop(selected_crop, first.image.shape)
        if crop != tuple(selected_crop):
            raise ValueError("work-area crop must lie entirely within the camera frame")
        hub = RenderHub()
        hub.set_calib(render.Calib(H_cam_to_page=calib.H_cam_to_page, page_size=[calib.page.w, calib.page.h]))
        overlay = OverlayMask((first.image.shape[1], first.image.shape[0]), calib.H_cam_to_page, (calib.page.w, calib.page.h))
        scene = _rig_scene(crop, 0)
        overlay.set_scene(scene, time.monotonic())
        hub.publish_scene(scene)
        server = start_server(create_app(render_hub=hub))
        print("Open http://localhost:8000/projector on the Tank (lightwork tank up); tap Start. Waiting 60 s.", flush=True)
        ack = hub.show_mode("normal", timeout=60.0)
        if ack is None or hub.hello("projector") is None:
            raise RuntimeError("no projector ACK (a screen-only connection cannot test projection)")
        hello = hub.hello("projector")
        if abs(hello.innerWidth - calib.page.w) > 1 or abs(hello.innerHeight - calib.page.h) > 1:
            raise RuntimeError("projector CSS dimensions differ from calibration; recalibrate first")
        if cam.wait_frame(ack + 0.3, 2.0) is None:
            raise RuntimeError("no normal-mode reference frame")
        args.output.mkdir(parents=True, exist_ok=True)
        if args.test == "m1":
            report = _rig_m1(cam, hub, overlay, crop, args.seconds)
        else:
            modes = FLAT_MODES if args.mode == "both" else (args.mode,)
            report = _rig_m2(cam, hub, overlay, crop, modes, args.count, args.output)
        report["crop_xywh"] = list(crop)
        report["camera_controlled"] = cam.controlled
        report["page_stats"] = hub.stats()
        text = json.dumps(report, sort_keys=True, indent=2)
        (args.output / f"{args.test}.json").write_text(text + "\n")
        print(text)
        return 0 if report["passed"] else 1
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"measurement check: {exc}", file=sys.stderr)
        return 2
    finally:
        if server is not None:
            server.stop()
        if cam is not None:
            cam.close()


if __name__ == "__main__":
    raise SystemExit(main())
