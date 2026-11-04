"""Session calibration for ``lightwork play``: camera gain, projector-camera mapping, hand check, drift checks.

Session start, in this order:

1. Camera gain: with the page dark, step the C920 gain (set while streaming; it shows ~4 frames later) until the
   surface sits mid-grey (``LEVEL_BAND``). One camera then serves a bright wall and a dark ceiling.
2. Geometry: flash the ArUco pattern and solve ``H_cam_to_page``. Bright projected markers can clip at a high gain, so
   a failed flash retries at lower gains; the ambient gain is restored afterwards (the mapping does not depend on it).
3. Hand check: targets are projected one at a time and the user touches them (table, wall) or points at them
   (ceiling). Where the camera sees the fingertip, against where the target was drawn, tells whether the hand works on
   the calibrated surface. A consistent offset is the fingertip landmark sitting past the contact point and becomes a
   cursor correction; scattered errors mean another plane or a stale mapping.

During the session the gain is nudged when the surface level leaves the band, and the pattern is flashed again at an
interval when no hand is around; a mapping that drifted is adopted (a large jump, a bumped rig, repeats the hand check).
"""

from __future__ import annotations

import math
import time
from collections import deque
from dataclasses import dataclass

import cv2
import numpy as np

from .contracts import render
from .geometry import apply_h
from .vision.calib.flow import MAX_RMS_PX, flash_pattern
from .vision.calib.solve import CalibrationError
from .vision.camera import V4l2Ctl, have_v4l2_ctl

GAIN_STEPS = (0, 16, 32, 48, 64, 96, 128, 160, 200, 255)
LEVEL_BAND = (60.0, 150.0)  # median grey of the surface under a black page: hands visible, nothing clipped
LEVEL_TARGET = 110.0  # session start aims here: hand tracking weakens towards the dark end of the band
CLIP_LEVEL = 245.0
SETTLE_FRAMES = 6  # a gain change shows up 3-4 frames later

TARGETS = ((120.0, 110.0), (630.0, 110.0), (375.0, 320.0))  # design px, spread over the page
ACQUIRE_R = {"touch": 110.0, "point": 160.0}  # a still fingertip this close to the target counts as an answer
STABLE_S = 0.7  # a touch is held; a pause while the hand approaches is shorter
STABLE_PX = 12.0  # 80 % of the window's fingertips within this of their median (MediaPipe jitters a few px)
MIN_SAMPLES = 6  # tracking drops frames; a still window needs this many fingertips, not every frame
TARGET_TIMEOUT_S = 12.0
FIRST_HAND_TIMEOUT_S = 30.0  # time to walk up to the surface before the check is skipped
PASS_RESIDUAL = 15.0  # design px rms around the mean offset: the hand is on the calibrated plane
PASS_OFFSET = 45.0  # design px: larger than any fingertip-landmark offset, so the mapping itself is off
TARGET_R = 22.0


# ---------------------------------------------------------------------------------------------------- camera gain


class CameraGain:
    """C920 gain written while streaming. ``available`` is False for uncontrolled cameras (gain stays as opened)."""

    def __init__(self, cam, gain: int):
        dev = getattr(cam, "device", None)
        # not `cam.controlled`: the C920 quantizes some gains (64 reads back 63), which marks the lock incomplete
        usable = dev is not None and getattr(dev, "is_c920", False) and have_v4l2_ctl()
        self.ctl = V4l2Ctl(dev.node) if usable else None
        self.cam = cam
        self.gain = gain

    @property
    def available(self) -> bool:
        return self.ctl is not None

    def set(self, gain: int) -> None:
        if self.ctl is None or gain == self.gain:
            return
        err = self.ctl.set("gain", gain)
        for _ in range(3):  # the C920 sometimes refuses a write just after the stream starts (4 Nov: "out of range" on 48)
            if not err:
                break
            time.sleep(0.2)
            err = self.ctl.set("gain", gain)
        if err:
            print(f"camera gain stays {self.gain}: setting {gain}: {err}")
            return
        self.gain = gain
        t = time.monotonic()
        for _ in range(SETTLE_FRAMES):  # let the change reach the frames
            frame = self.cam.wait_frame(t, 1.0)
            if frame is None:
                break
            t = frame.t_mono


def surface_level(cam, mask: np.ndarray | None = None, frames: int = 3) -> tuple[float, float]:
    """(median, 99th percentile) grey level of fresh frames inside ``mask`` (default: the central 60 %)."""
    t = time.monotonic()
    meds, highs = [], []
    for _ in range(frames):
        frame = cam.wait_frame(t, 1.0)
        if frame is None:
            break
        t = frame.t_mono
        g = cv2.cvtColor(frame.image[::4, ::4], cv2.COLOR_BGR2GRAY)
        if mask is not None:
            v = g[mask[::4, ::4] > 0]
        else:
            h, w = g.shape
            v = g[h // 5:h - h // 5, w // 5:w - w // 5].ravel()
        if v.size:
            meds.append(float(np.median(v)))
            highs.append(float(np.percentile(v, 99)))
    if not meds:
        raise CalibrationError("camera gave no frame")
    return float(np.median(meds)), float(np.median(highs))


def _step_index(gain: int) -> int:
    return min(range(len(GAIN_STEPS)), key=lambda k: abs(GAIN_STEPS[k] - gain))


def choose_gain(gain: CameraGain, cam, mask: np.ndarray | None = None) -> float:
    """Step the gain towards ``LEVEL_TARGET`` and keep the unclipped step closest to it; returns its level."""
    i = _step_index(gain.gain)
    gain.set(GAIN_STEPS[i])
    levels: dict[int, tuple[float, float]] = {}
    while True:
        level, high = surface_level(cam, mask)
        levels[i] = (level, high)
        if not gain.available:
            return level
        if level < LEVEL_TARGET and high <= CLIP_LEVEL and i + 1 < len(GAIN_STEPS) and i + 1 not in levels:
            i += 1
        elif (level > LEVEL_BAND[1] or high > CLIP_LEVEL) and i > 0 and i - 1 not in levels:
            i -= 1
        else:
            break
        gain.set(GAIN_STEPS[i])
    ok = {k: v for k, v in levels.items() if v[1] <= CLIP_LEVEL and v[0] <= LEVEL_BAND[1]} or levels
    best = min(ok, key=lambda k: abs(ok[k][0] - LEVEL_TARGET))
    gain.set(GAIN_STEPS[best])
    return levels[best][0]


def nudge_gain(gain: CameraGain, cam, mask: np.ndarray | None) -> tuple[bool, float]:
    """One gain step when the surface level has left the band (with 10 % hysteresis). Returns (changed, level)."""
    level, _ = surface_level(cam, mask, frames=2)
    if not gain.available:
        return False, level
    i = _step_index(gain.gain)
    lo, hi = LEVEL_BAND
    if level < 0.9 * lo and i + 1 < len(GAIN_STEPS):
        gain.set(GAIN_STEPS[i + 1])
    elif level > 1.1 * hi and i > 0:
        gain.set(GAIN_STEPS[i - 1])
    else:
        return False, level
    return True, level


# ---------------------------------------------------------------------------------------------------- geometry


def flash_geometry(cam, hub, page, gain: CameraGain, log=print):
    """Flash the pattern and solve, retrying at lower gains; the ambient gain is restored. Returns the ``Flash``."""
    ambient = gain.gain
    tries = [ambient] + ([g for g in (ambient // 2, 0) if g < ambient] if gain.available else [])
    last: CalibrationError | None = None
    try:
        for g in dict.fromkeys(tries):
            gain.set(g)
            try:
                flash = flash_pattern(cam, hub, page.innerWidth, page.innerHeight)
            except CalibrationError as exc:
                log(f"  pattern at gain {g}: {exc}")
                last = exc
                continue
            r = flash.solution.residuals
            if r.rms_px > MAX_RMS_PX:
                last = CalibrationError(f"residual rms {r.rms_px:.2f} > {MAX_RMS_PX} page px")
                log(f"  pattern at gain {g}: {last}")
                continue
            return flash
        raise last or CalibrationError("no pattern solve")
    finally:
        hub.show_mode("normal")
        gain.set(ambient)


def drift_px(H_old, H_new, page_w: float, page_h: float) -> float:
    """Largest page-px move of the page corners and centre between two mappings (seen from the same camera px)."""
    pts = np.array([(0, 0), (page_w, 0), (page_w, page_h), (0, page_h), (page_w / 2, page_h / 2)], dtype=np.float64)
    cam = apply_h(np.linalg.inv(np.asarray(H_old, dtype=np.float64)), pts)
    return float(np.max(np.linalg.norm(apply_h(np.asarray(H_new, dtype=np.float64), cam) - pts, axis=1)))


def footprint_mask(geo, frame_size: tuple[int, int]) -> np.ndarray:
    """Camera px covered by the projected page."""
    w, h = frame_size
    poly = apply_h(geo.Hi, [(0, 0), (geo.page_w, 0), (geo.page_w, geo.page_h), (0, geo.page_h)])
    mask = np.zeros((h, w), np.uint8)
    cv2.fillPoly(mask, [np.round(poly).astype(np.int32)], 255)
    return mask


# ---------------------------------------------------------------------------------------------------- hand check


@dataclass(frozen=True)
class HandResult:
    mode: str  # touch | point
    status: str  # pass | fail | aligned | skipped
    errors: tuple[tuple[float, float], ...]  # fingertip - target per answered target, design px
    offset: tuple[float, float]  # mean error: the cursor correction
    residual: float  # rms of errors around the offset
    worst: float
    message: str


def evaluate(errors, mode: str) -> HandResult:
    e = np.asarray(errors, dtype=np.float64).reshape(-1, 2)
    off = e.mean(axis=0)
    residual = float(np.sqrt(np.mean(np.sum((e - off) ** 2, axis=1))))
    worst = float(np.max(np.linalg.norm(e, axis=1)))
    offset = (round(float(off[0]), 1), round(float(off[1]), 1))
    errs = tuple((round(float(x), 1), round(float(y), 1)) for x, y in e)
    if mode == "point":
        return HandResult(mode, "aligned", errs, offset, residual, worst,
                          f"Pointer aligned: offset {offset[0]:+.0f}, {offset[1]:+.0f} px, scatter {residual:.0f} px.")
    if residual <= PASS_RESIDUAL and math.hypot(*off) <= PASS_OFFSET:
        return HandResult(mode, "pass", errs, offset, residual, worst,
                          f"Surface check passed: scatter {residual:.0f} px, fingertip offset {math.hypot(*off):.0f} px corrected.")
    return HandResult(mode, "fail", errs, offset, residual, worst,
                      f"Surface check failed: scatter {residual:.0f} px, offset {math.hypot(*off):.0f} px.")


class HandCheck:
    """Projected targets answered by a still fingertip. Feed :meth:`step` every frame; draw with :meth:`draw`."""

    def __init__(self, geo, mode: str = "touch", attempt: int = 1):
        self.geo, self.mode, self.attempt = geo, mode, attempt
        self.i = 0
        self.errors: list[tuple[float, float]] = []
        self.result: HandResult | None = None
        self.tip: tuple[float, float] | None = None
        self._hist: deque[tuple[float, tuple[float, float]]] = deque()
        self._shown: float | None = None
        self._hand_seen = False
        self._pause_until = 0.0
        self._created: float | None = None
        self.closest: float = math.inf  # nearest fingertip to the current target, design px
        self.frames = 0  # frames while a hand was expected (after the first one appeared)
        self.tip_frames = 0

    @property
    def tracked(self) -> float:
        return self.tip_frames / self.frames if self.frames else 0.0

    @property
    def verb(self) -> str:
        return "TOUCH" if self.mode == "touch" else "POINT AT"

    def _still(self, now: float) -> tuple[float, float] | None:
        if len(self._hist) < MIN_SAMPLES or now - self._hist[0][0] < 0.8 * STABLE_S:
            return None
        pts = np.array([p for _, p in self._hist])
        med = np.median(pts, axis=0)
        if np.mean(np.linalg.norm(pts - med, axis=1) <= STABLE_PX) < 0.8:
            return None
        return float(med[0]), float(med[1])

    def step(self, tip: tuple[float, float] | None, now: float) -> HandResult | None:
        """``tip``: fingertip in design px without cursor correction, or None."""
        if self.result is not None:
            return self.result
        if self._created is None:
            self._created = now
        self.tip = tip
        if self._hand_seen or tip is not None:
            self.frames += 1
            self.tip_frames += tip is not None
        if tip is not None:
            if not self._hand_seen:
                self._hand_seen = True
                self._shown = now  # the first target's clock starts when a hand arrives
            self._hist.append((now, tip))
        while self._hist and now - self._hist[0][0] > STABLE_S:  # dropped frames do not reset the window
            self._hist.popleft()
        if now < self._pause_until:
            return None
        target = TARGETS[self.i]
        if tip is not None:
            self.closest = min(self.closest, math.dist(tip, target))
        still = self._still(now)
        if still is not None and math.dist(still, target) <= ACQUIRE_R[self.mode]:
            self.errors.append((still[0] - target[0], still[1] - target[1]))
            self.i += 1
            self._hist.clear()
            self.closest = math.inf
            self._pause_until = now + 0.6
            self._shown = now + 0.6
            if self.i == len(TARGETS):
                self.result = evaluate(self.errors, self.mode)
        elif not self._hand_seen and now - self._created > FIRST_HAND_TIMEOUT_S:
            self.result = HandResult(self.mode, "skipped", (), (0.0, 0.0), 0.0, 0.0, "No hand seen: surface not verified.")
        elif self._hand_seen and now - self._shown > TARGET_TIMEOUT_S:
            base = evaluate(self.errors, self.mode) if self.errors else None
            self.result = HandResult(self.mode, "fail", base.errors if base else (), base.offset if base else (0.0, 0.0),
                                     base.residual if base else 0.0, base.worst if base else 0.0,
                                     f"Target {self.i + 1} was not reached: hand tracked in {self.tracked:.0%} of "
                                     f"frames, closest fingertip {self.closest:.0f} px away ({ACQUIRE_R[self.mode]:.0f} "
                                     "needed).")
        return self.result

    def draw(self, c) -> None:
        """Targets on a playground ``Canvas`` (design space)."""
        c.label("check-title", (18, 24), "SURFACE CHECK" if self.mode == "touch" else "POINTER CHECK", "info")
        if self.attempt > 1:
            c.label("check-retry", (560, 24), f"TRY {self.attempt}", "warn")
        for k, t in enumerate(TARGETS):
            if k < self.i:
                c.circle(f"check-done-{k}", t, 10, "target", n=20)
            elif k == self.i:
                cam = self.geo.d2c([t, (t[0] + TARGET_R, t[1])])
                r_cam = float(np.linalg.norm(cam[1] - cam[0]))
                c.prims.append(render.Ring(id=f"check-t-{k}", c=c._xy([cam[0]])[0], r=round(r_cam, 1), role="warn", pulse=True))
                c.poly(f"check-h-{k}", [(t[0] - 34, t[1]), (t[0] + 34, t[1])], "warn")
                c.poly(f"check-v-{k}", [(t[0], t[1] - 34), (t[0], t[1] + 34)], "warn")
                lx = t[0] + 44 if t[0] < 500 else t[0] - 210
                c.label(f"check-l-{k}", (lx, t[1] + 44), f"{self.verb} {k + 1}/{len(TARGETS)}", "info")
        if self.tip is not None:
            c.circle("check-tip", self.tip, 12, "source", n=20)


# ---------------------------------------------------------------------------------------------------- border frame
#
# A white line around the whole page shows where the projection ends, to the camera and to people. The camera looks for
# it across short profiles at the expected position (from the mapping): a border that shifted against its baseline means
# the camera or the projector moved; a border that cannot be found means it moved a lot or something covers it. Either
# triggers a fresh pattern flash without waiting for the interval.
#
# Other projected lines can fall inside a profile too: the drop zone's right edge is 13.8 page px inside the border on a
# 749 px page, and label tops sit about 12 px below the top line. They are as bright as the border, so each profile takes
# the peak nearest the expected position, never simply the brightest one (measured 1 Nov, wall: brightest-peak readings
# jumped between 0, 6.4 and 12.7 px with nothing moving, and flashed the pattern every 10-20 s).

FRAME_SAMPLES = 14  # profiles per side
FRAME_REACH = 14.0  # page px searched on each side of the expected line
FRAME_STEP = 0.5  # page px between profile samples
FRAME_MIN_CONTRAST = 15.0  # grey levels the line must stand above its surroundings
FRAME_SIDE_FOUND = 0.4  # share of a side's profiles that must find the line
FRAME_MOVE_PX = 2.5  # page px of shift (vs baseline) that counts as moved
FRAME_CONFIRM = 2  # readings in a row before acting (a passing hand is not a bump)
SIDES = ("top", "right", "bottom", "left")


def frame_inset(page_w: float, page_h: float) -> float:
    """Page px from the page edge to the frame line's centre, so the whole stroke is on the page."""
    from .server import projector_style as style

    return style.LINE_PROJ_PX * style.unit_css_px(page_w, page_h) / 2 + 2.0


def frame_corners_page(page_w: float, page_h: float) -> np.ndarray:
    m = frame_inset(page_w, page_h)
    return np.array([(m, m), (page_w - m, m), (page_w - m, page_h - m), (m, page_h - m)], dtype=np.float64)


def frame_prims(geo) -> list:
    cam = apply_h(geo.Hi, frame_corners_page(geo.page_w, geo.page_h))
    return [render.Poly(id="frame", pts=[[round(float(x), 1), round(float(y), 1)] for x, y in cam], closed=True, role="info")]


@dataclass(frozen=True)
class BorderReading:
    sides: tuple[float | None, ...]  # per side: median offset of the line from where it was drawn, page px (+ = inward)
    found: tuple[float, ...]  # per side: share of profiles that found the line
    contrast: float  # median line contrast, grey levels

    @property
    def n_seen(self) -> int:
        return sum(o is not None for o in self.sides)


class BorderWatch:
    """Finds the projected frame in camera images and reports when it moved against its baseline."""

    def __init__(self, geo):
        corners = frame_corners_page(geo.page_w, geo.page_h)
        d = np.arange(-FRAME_REACH, FRAME_REACH + 1e-9, FRAME_STEP)
        self.d = d
        maps = []
        for s in range(4):
            a, b = corners[s], corners[(s + 1) % 4]
            along = (b - a) / np.linalg.norm(b - a)
            inward = np.array([-along[1], along[0]])  # corners run clockwise in page px (y down): left normal is inward
            for k in range(1, FRAME_SAMPLES + 1):
                p = a + (b - a) * k / (FRAME_SAMPLES + 1)
                maps.append(apply_h(geo.Hi, p + d[:, None] * inward))
        pts = np.array(maps, dtype=np.float32)  # (4 * samples, len(d), 2) camera px
        self.map_x, self.map_y = np.ascontiguousarray(pts[..., 0]), np.ascontiguousarray(pts[..., 1])
        self.baseline: BorderReading | None = None
        self.last: BorderReading | None = None
        self._moved = 0
        self._lost = 0

    def read(self, image: np.ndarray) -> BorderReading:
        g = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        prof = cv2.remap(g, self.map_x, self.map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        prof = prof.astype(np.float32)
        above = prof - np.median(prof, axis=1, keepdims=True)
        # candidate lines: local maxima above the contrast floor; a line cut off by the window end is not located
        mid = above[:, 1:-1]
        is_peak = (mid >= FRAME_MIN_CONTRAST) & (mid >= above[:, :-2]) & (mid >= above[:, 2:])
        offsets = np.full(len(prof), np.nan)
        heights = []
        for i in np.nonzero(is_peak.any(axis=1))[0]:
            cand = np.nonzero(is_peak[i])[0] + 1
            j = int(cand[np.argmin(np.abs(self.d[cand]))])  # the line nearest where the border was drawn
            w = np.clip(above[i] - above[i, j] / 2, 0, None)  # upper half of that line's profile
            lo, hi = j, j
            while lo > 0 and w[lo - 1] > 0:
                lo -= 1
            while hi < len(w) - 1 and w[hi + 1] > 0:
                hi += 1
            offsets[i] = float(np.sum(w[lo:hi + 1] * self.d[lo:hi + 1]) / max(np.sum(w[lo:hi + 1]), 1e-6))
            heights.append(float(above[i, j]))
        sides, found = [], []
        for s in range(4):
            o = offsets[s * FRAME_SAMPLES:(s + 1) * FRAME_SAMPLES]
            ok = ~np.isnan(o)
            found.append(round(float(ok.mean()), 2))
            sides.append(round(float(np.median(o[ok])), 2) if ok.mean() >= FRAME_SIDE_FOUND else None)
        return BorderReading(tuple(sides), tuple(found), round(float(np.median(heights)), 1) if heights else 0.0)

    def shift(self, r: BorderReading) -> float | None:
        """Largest per-side change against the baseline, page px (None before a baseline or with no common side)."""
        if self.baseline is None:
            return None
        diffs = [abs(a - b) for a, b in zip(r.sides, self.baseline.sides, strict=False) if a is not None and b is not None]
        return max(diffs) if diffs else None

    def update(self, image: np.ndarray) -> str | None:
        """Feed a hand-free frame. Returns "moved" or "lost" once confirmed, else None; the first good reading
        becomes the baseline."""
        r = self.read(image)
        self.last = r
        if self.baseline is None:
            if r.n_seen >= 3:
                self.baseline = r
            return None
        shift = self.shift(r)
        self._moved = self._moved + 1 if shift is not None and shift > FRAME_MOVE_PX else 0
        self._lost = self._lost + 1 if r.n_seen < 2 else 0
        if self._moved >= FRAME_CONFIRM:
            self._moved = 0
            return "moved"
        if self._lost >= FRAME_CONFIRM + 1:
            self._lost = 0
            return "lost"
        return None
