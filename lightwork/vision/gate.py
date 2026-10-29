"""Change/settle gate with activity attribution (PLAN.md §4.1, §7 B; ``briefs/rig.md`` §5.3).

Every camera frame is judged inside the work area only, at 640 px across the full frame (640x360 for 16:9):

- grey, blurred, compared with the frame ``lag_s`` earlier (slow hands still move a few px in 0.1 s);
- :func:`band_gain` cancels multiplicative light changes per 8-row band (DLP subframes and 100 Hz mains roll through
  a rolling shutter as horizontal bands; exposure drift is global) using robust medians, so hands do not bias it;
- our own projection (``overlay_mask`` from :mod:`lightwork.vision.overlay_mask`) is ignored in both frames, and a
  frame that is mostly masked (the page is in a measurement mode) is skipped and never becomes a reference;
- a pixel changed when the corrected readings differ by more than ``diff_dn``. A clipped reading is only a lower
  bound, not an exact value; two clipped readings carry no evidence. A frame moves at ``min_area_px`` changed pixels.

The scene is ``settled`` once comparisons have shown no change across ``still_s``: evidence of stillness starts at
the reference frame of the first still comparison, so the settle comes ``still_s`` after the last change, plus at
most one frame.

Events (C1 ``motion`` input payloads, only on change). An episode runs from the first moving frame to the settle;
``active_regions`` is the sorted set of region refs (layout object ids and ``<grid>:<cell>``) that moved at any time
during the episode. ``moving`` is emitted when motion starts and again, at most every ``EMIT_MIN_S``, when that set
grows; ``settled`` carries the episode's final set. The first ``settled`` (empty) comes ``still_s`` after start-up.
A region moved in a frame when ``REGION_FRAC`` of its area changed, capped at ``REGION_MAX_PX`` proc px, so a
fingertip is enough for a large container while a 96-well cell needs a fifth of itself.
"""

from __future__ import annotations

import math
from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass

import cv2
import numpy as np

from ..geometry import Regions
from .source import Frame

PROC_WIDTH = 640  # judge at 640 px across the full camera frame
BLUR_SIGMA = 1.0
BAND_ROWS = 8  # gain bands, proc rows
GAIN_OUTLIER = 0.25  # pixels whose ratio is further than this from the frame median do not vote (hands, moved objects)
BAND_MIN_VOTES = 0.1  # fraction of a band's samples that must vote for its own gain; others are interpolated
SATURATED = 245.0  # blurred grey at or above this is clipped
MIN_VALID = 0.25  # unmasked fraction of the work area needed to judge a frame
EMIT_MIN_S = 0.25
REGION_FRAC = 0.2
REGION_MAX_PX = 12
HISTORY = 8  # reference candidates kept (0.1 s lag at up to 60 fps, plus margin)
_EPS = 1e-6


def band_gain(cur: np.ndarray, ref: np.ndarray, valid: np.ndarray | None = None, band_rows: int = BAND_ROWS) -> np.ndarray:
    """Per-row gain ``g`` (float32, one per row) such that ``cur ~ g * ref`` on the static background.

    Ratios are sampled on every second row and column. A pixel clipped in one frame votes with its clipped reading:
    that ratio is only a bound, but it is the gain that predicts the clipped reading, so a stretch of board that
    clips in one frame stays consistent. Pixels clipped in both frames, masked pixels and pixels far from the frame's
    median ratio (hands, objects that moved) do not vote. Band medians are interpolated between band centres; a band
    with too few votes takes its value from its neighbours.
    """
    cs, rs = cur[::2, ::2], ref[::2, ::2]
    ratio = (cs + 4.0) / (rs + 4.0)
    usable = (cs < SATURATED) | (rs < SATURATED)
    if valid is not None:
        usable &= valid[::2, ::2]
    sample = ratio[usable]
    if sample.size == 0:
        return np.ones(cur.shape[0], np.float32)
    g0 = float(np.median(sample))
    keep = usable & (np.abs(ratio - g0) <= GAIN_OUTLIER * g0)
    b = max(1, band_rows // 2)
    nb = -(-ratio.shape[0] // b)
    votes = np.full((nb * b, ratio.shape[1]), np.inf, np.float32)
    votes[: ratio.shape[0]][keep] = ratio[keep]
    votes = np.sort(votes.reshape(nb, -1), axis=1)  # votes first, padding (inf) last
    n = np.count_nonzero(np.isfinite(votes), axis=1)
    ok = n >= max(1.0, BAND_MIN_VOTES * b * ratio.shape[1])
    if not ok.any():
        return np.full(cur.shape[0], g0, np.float32)
    i = np.nonzero(ok)[0]
    med = (votes[i, (n[i] - 1) // 2] + votes[i, n[i] // 2]) / 2
    centres = 2.0 * (i * b + (b - 1) / 2.0)
    return np.interp(np.arange(cur.shape[0]), centres, med).astype(np.float32)


def region_threshold(area_px: np.ndarray | int) -> np.ndarray | int:
    """Changed proc px needed for a region of ``area_px`` proc px to count as moved."""
    return np.maximum(1, np.minimum(REGION_MAX_PX, np.ceil(REGION_FRAC * np.asarray(area_px)))).astype(np.int64)


@dataclass(frozen=True)
class _Snap:
    t: float
    grey: np.ndarray  # float32 (ph, pw)
    masked: np.ndarray | None  # bool (ph, pw); None = nothing masked


@dataclass(frozen=True)
class _Geometry:
    shape: tuple[int, int]  # camera frame (h, w)
    x: int
    y: int
    w: int
    h: int  # work area, camera px, clamped to the frame
    pw: int
    ph: int  # proc size of the work area
    grids: list[tuple[str, np.ndarray, list[str], np.ndarray]]  # (grid id, int32 labels, cell names, threshold per label)
    objects: list[tuple[str, slice, slice, np.ndarray, int]]  # (object id, rows, cols, bool mask crop, threshold)


class Gate:
    """Feed every camera frame to :meth:`update`; log the returned events with the frame's time."""

    def __init__(
        self,
        regions: Regions | None,
        work_area_xywh: Sequence[int] | None,
        *,
        still_s: float = 0.6,
        lag_s: float = 0.1,
        diff_dn: float = 12.0,
        min_area_px: int = 16,
    ):
        self.regions = regions
        self.work_area = None if work_area_xywh is None else tuple(int(v) for v in work_area_xywh)
        self.still_s = still_s
        self.lag_s = lag_s
        self.diff_dn = diff_dn
        self.min_area_px = min_area_px
        self.state: str | None = None  # "moving" | "settled"; None until the first decision
        self.last_changed_px = 0  # changed proc px in the last judged frame (diagnostics, rig:M1)
        self._geo: _Geometry | None = None
        self._hist: deque[_Snap] = deque(maxlen=HISTORY)
        self._still_since: float | None = None
        self._episode: set[str] = set()
        self._emitted: frozenset[str] = frozenset()
        self._last_emit_t = -math.inf
        self._mask_src: np.ndarray | None = None
        self._mask_proc: np.ndarray | None = None

    def update(self, frame: Frame, overlay_mask: np.ndarray | None = None) -> list[tuple[str, dict]]:
        """Judge one frame. ``overlay_mask``: uint8/bool (frame h, w), nonzero = our projection (ignored); treated
        as immutable, so passing the same array object again skips its downscale."""
        geo = self._geometry(frame.image.shape)
        masked = self._masked(overlay_mask, geo)
        if masked is not None and np.count_nonzero(masked) > (1 - MIN_VALID) * masked.size:
            return []
        grey = self._grey(frame.image, geo)
        t = frame.t_mono
        ref = self._reference(t)
        self._hist.append(_Snap(t, grey, masked))
        if ref is None:
            self._still_since = t
            return []
        changed = self._changed(grey, masked, ref)
        self.last_changed_px = int(np.count_nonzero(changed))
        if self.last_changed_px >= self.min_area_px:
            return self._on_motion(t, self._active(changed, geo))
        return self._on_still(t, ref.t)

    # ---- decisions

    def _on_motion(self, t: float, regions: set[str]) -> list[tuple[str, dict]]:
        self._still_since = None
        if self.state != "moving":
            self.state = "moving"
            self._episode = set(regions)
            return [self._emit(t, "moving")]
        self._episode |= regions
        if self._episode != self._emitted and t - self._last_emit_t >= EMIT_MIN_S - _EPS:
            return [self._emit(t, "moving")]
        return []

    def _on_still(self, t: float, t_ref: float) -> list[tuple[str, dict]]:
        if self._still_since is None:
            self._still_since = t_ref
        if self.state == "settled" or t - self._still_since < self.still_s - _EPS:
            return []
        event = self._emit(t, "settled")
        self.state = "settled"
        self._episode = set()
        self._emitted = frozenset()
        return [event]

    def _emit(self, t: float, state: str) -> tuple[str, dict]:
        self._emitted = frozenset(self._episode)
        self._last_emit_t = t
        return "motion", {"state": state, "active_regions": sorted(self._episode)}

    # ---- image work

    def _reference(self, t: float) -> _Snap | None:
        for snap in reversed(self._hist):
            if snap.t <= t - self.lag_s + _EPS:
                return snap
        return self._hist[0] if self._hist else None

    def _grey(self, img: np.ndarray, geo: _Geometry) -> np.ndarray:
        crop = img[geo.y : geo.y + geo.h, geo.x : geo.x + geo.w]
        small = cv2.resize(crop, (geo.pw, geo.ph), interpolation=cv2.INTER_AREA)
        grey = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).astype(np.float32)
        return cv2.GaussianBlur(grey, (5, 5), BLUR_SIGMA)

    def _masked(self, overlay_mask: np.ndarray | None, geo: _Geometry) -> np.ndarray | None:
        if overlay_mask is None:
            return None
        if overlay_mask is self._mask_src:
            return self._mask_proc
        if overlay_mask.shape[:2] != geo.shape:
            raise ValueError(f"overlay_mask is {overlay_mask.shape[:2]}, camera frames are {geo.shape}")
        crop = overlay_mask[geo.y : geo.y + geo.h, geo.x : geo.x + geo.w]
        if crop.dtype == np.bool_:
            crop = crop.view(np.uint8)
        small = cv2.resize(crop, (geo.pw, geo.ph), interpolation=cv2.INTER_AREA) > 0
        self._mask_src = overlay_mask
        self._mask_proc = small if small.any() else None
        return self._mask_proc

    def _changed(self, grey: np.ndarray, masked: np.ndarray | None, ref: _Snap) -> np.ndarray:
        if masked is None:
            ignore = ref.masked
        else:
            ignore = masked if ref.masked is None else masked | ref.masked
        valid = None if ignore is None else ~ignore
        gain = band_gain(grey, ref.grey, valid)[:, None]
        cur_exact, ref_exact = grey < SATURATED, ref.grey < SATURATED
        # A clipped reading denotes [SATURATED, infinity): change requires disjoint possible-value intervals.
        brighter = (np.minimum(grey, SATURATED) - gain * ref.grey > self.diff_dn) & ref_exact
        darker = (gain * np.minimum(ref.grey, SATURATED) - grey > self.diff_dn) & cur_exact
        changed = brighter | darker
        if valid is not None:
            changed &= valid
        return changed

    def _active(self, changed: np.ndarray, geo: _Geometry) -> set[str]:
        out: set[str] = set()
        for gid, labels, names, thr in geo.grids:
            counts = np.bincount(labels[changed], minlength=len(names) + 1)
            out.update(f"{gid}:{names[i - 1]}" for i in np.nonzero(counts[1:] >= thr[1:])[0] + 1)
        for oid, rows, cols, m, thr in geo.objects:
            if np.count_nonzero(changed[rows, cols] & m) >= thr:
                out.add(oid)
        return out

    # ---- geometry (per camera frame size)

    def _geometry(self, shape: tuple[int, ...]) -> _Geometry:
        hw = (int(shape[0]), int(shape[1]))
        if self._geo is not None and self._geo.shape == hw:
            return self._geo
        H, W = hw
        x, y, w, h = self.work_area if self.work_area is not None else (0, 0, W, H)
        x0, y0, x1, y1 = max(0, x), max(0, y), min(W, x + w), min(H, y + h)
        if x1 <= x0 or y1 <= y0:
            raise ValueError(f"work area {self.work_area} lies outside the {W}x{H} camera frame")
        x, y, w, h = x0, y0, x1 - x0, y1 - y0
        s = PROC_WIDTH / W
        pw, ph = max(1, round(w * s)), max(1, round(h * s))
        scale = np.array([pw / w, ph / h])
        origin = np.array([x, y], dtype=np.float64)

        def proc(poly: np.ndarray) -> np.ndarray:
            return np.round((np.asarray(poly, dtype=np.float64) - origin) * scale * 8).astype(np.int32)

        grids: list[tuple[str, np.ndarray, list[str], np.ndarray]] = []
        objects: list[tuple[str, slice, slice, np.ndarray, int]] = []
        if self.regions is not None:
            for gid, gg in self.regions.grids.items():
                names = gg.names()
                labels = np.zeros((ph, pw), np.int32)
                for i, name in enumerate(names, 1):
                    cv2.fillPoly(labels, [proc(gg.quad(name))], i, cv2.LINE_8, 3)
                area = np.bincount(labels.ravel(), minlength=len(names) + 1)
                grids.append((gid, labels, names, region_threshold(area)))
            for oid in self.regions.layout.objects:
                m = np.zeros((ph, pw), np.uint8)
                cv2.fillPoly(m, [proc(self.regions.polygon(oid))], 1, cv2.LINE_8, 3)
                ys, xs = np.nonzero(m)
                if ys.size == 0:
                    continue  # outside the work area: never active
                rows, cols = slice(int(ys.min()), int(ys.max()) + 1), slice(int(xs.min()), int(xs.max()) + 1)
                objects.append((oid, rows, cols, m[rows, cols].astype(bool), int(region_threshold(ys.size))))
        self._geo = _Geometry(hw, x, y, w, h, pw, ph, grids, objects)
        self._hist.clear()
        self._mask_src = self._mask_proc = None
        return self._geo
