"""Mask of our own projected overlay in camera px (PLAN.md D7; ``briefs/rig.md`` §5.3).

The projector page maps every C2 primitive point by point through ``H_cam_to_page`` and strokes it with the sizes in
:mod:`lightwork.server.projector_style`. :func:`rasterize` draws the same geometry on a page-space canvas and warps it
back into the camera, so the mask covers what the camera can see of our overlay.

:class:`OverlayMask` keeps a timeline of what the page shows. Its masks are dilated (blur plus calibration error:
8 px at 1080p) and, after any change, cover the union of the old and new overlay for ``hold_s`` (page draw, DLP and
camera lag, ``briefs/rig.md`` §5.2). While a measurement or pattern mode lights the whole bench, and for ``hold_s``
after the page is back to normal, the mask is the full frame. The gate ignores masked pixels;
:mod:`lightwork.vision.measure_frames` uses :meth:`OverlayMask.drawn_at` to check that a keyframe no longer shows it.

Times are laptop ``time.monotonic()`` seconds, the clock of ``Frame.t_mono`` and of page ACKs. Returned masks are
read-only uint8 arrays of the camera frame size, 255 = overlay; the same object is returned while nothing changes.
"""

from __future__ import annotations

import bisect
import itertools
import math
import threading
from collections.abc import Sequence
from dataclasses import dataclass, replace

import cv2
import numpy as np

from ..contracts import render
from ..geometry import apply_h
from ..server import projector_style as style

HOLD_S = 0.2  # union of old and new overlay after a change (camera sees the page this late, at worst)
DILATE_PX_1080 = 8  # rig brief §5.3: 7-9 px at 1080p; scaled with the frame height
SUPERSAMPLE = 2  # page canvas px per page CSS px
SHIFT = 3  # fixed-point bits for sub-pixel drawing
LABEL_ADVANCE_EM = 1.1  # conservative bold sans/CJK width, including wide glyphs; the page measures text, we cannot
LABEL_HALF_HEIGHT_EM = 0.65  # label box half height in font px (`at` is the vertical middle)
KEEP_S = 1.0  # entries that stopped being visible this long before the newest change are dropped (frames lag less)


def _fixed(pts: np.ndarray) -> np.ndarray:
    return np.round(np.asarray(pts, dtype=np.float64) * SUPERSAMPLE * (1 << SHIFT)).astype(np.int32)


def _draw(canvas: np.ndarray, prim, H: np.ndarray, unit: float) -> None:
    """Draw one primitive on a page-space canvas (``SUPERSAMPLE`` px per CSS px), as projector.js strokes it. A glowing
    ring or poly covers its halo (faint, but the camera sees it); ``draw_ms`` outlines and ``type_ms`` labels are drawn
    whole, as they are once revealed."""
    stroke = style.LINE_PROJ_PX * unit
    if prim.kind == "ring" and prim.pulse:
        stroke *= 1 + style.PULSE_GAIN
    if getattr(prim, "glow", False):  # ring and poly only
        stroke *= style.GLOW_WIDTH_LINES
    thick = max(1, round(stroke * SUPERSAMPLE))
    if prim.kind == "ring":
        a = np.linspace(0.0, 2 * math.pi, style.RING_SEGMENTS, endpoint=False)
        cam = np.column_stack([prim.c[0] + prim.r * np.cos(a), prim.c[1] + prim.r * np.sin(a)])
        cv2.polylines(canvas, [_fixed(apply_h(H, cam))], True, 255, thick, cv2.LINE_8, SHIFT)
    elif prim.kind == "arrow":
        a, b = apply_h(H, [prim.a, prim.b])
        back = a - b
        n = float(np.hypot(*back))
        head = style.ARROW_HEAD_LINES * stroke
        lines = [np.array([a, b])]
        if n > 0:
            u = back / n
            for s in (-style.ARROW_HEAD_RAD, style.ARROW_HEAD_RAD):
                c, si = math.cos(s), math.sin(s)
                lines.append(np.array([b, b + head * np.array([c * u[0] - si * u[1], si * u[0] + c * u[1]])]))
        else:
            cv2.circle(canvas, tuple(int(v) for v in _fixed(b)), round(head * SUPERSAMPLE * (1 << SHIFT)), 255, -1, cv2.LINE_8, SHIFT)
        cv2.polylines(canvas, [_fixed(p) for p in lines], False, 255, thick, cv2.LINE_8, SHIFT)
    elif prim.kind == "poly":
        pts = _fixed(apply_h(H, prim.pts))
        if len(pts) == 1:
            cv2.circle(canvas, tuple(int(v) for v in pts[0]), round(stroke / 2 * SUPERSAMPLE * (1 << SHIFT)), 255, -1, cv2.LINE_8, SHIFT)
        elif len(pts) > 1:
            cv2.polylines(canvas, [pts], prim.closed, 255, thick, cv2.LINE_8, SHIFT)
    elif prim.kind == "dot":
        c = _fixed(apply_h(H, [prim.c])[0])
        r = style.DOT_RADIUS_LINES * stroke
        cv2.circle(canvas, (int(c[0]), int(c[1])), round(r * SUPERSAMPLE * (1 << SHIFT)), 255, -1, cv2.LINE_8, SHIFT)
    elif prim.kind == "label":
        x, y = apply_h(H, [prim.at])[0]
        font = style.TEXT_PROJ_PX * unit
        x0, x1 = x - stroke, x + LABEL_ADVANCE_EM * font * len(prim.text) + stroke
        y0, y1 = y - LABEL_HALF_HEIGHT_EM * font, y + LABEL_HALF_HEIGHT_EM * font
        cv2.fillPoly(canvas, [_fixed([[x0, y0], [x1, y0], [x1, y1], [x0, y1]])], 255, cv2.LINE_8, SHIFT)
    elif prim.kind in ("picture", "map", "solid"):  # all of it: what is black in it we cannot tell here, and a mask may cover too much
        cv2.fillPoly(canvas, [_fixed(apply_h(H, prim.corners))], 255, cv2.LINE_8, SHIFT)
    else:
        raise ValueError(f"unknown primitive kind {prim.kind!r}")


def rasterize(prims: Sequence[render.Prim], H_cam_to_page, page_size: Sequence[float], frame_size: Sequence[int]) -> np.ndarray:
    """Undilated uint8 mask (frame h x w, 255 = lit) of the camera px the page lights when it draws ``prims``."""
    H = np.asarray(H_cam_to_page, dtype=np.float64)
    page_w, page_h = float(page_size[0]), float(page_size[1])
    w, h = int(frame_size[0]), int(frame_size[1])
    canvas = np.zeros((math.ceil(page_h * SUPERSAMPLE), math.ceil(page_w * SUPERSAMPLE)), np.uint8)
    unit = style.unit_css_px(page_w, page_h)
    for p in prims:
        _draw(canvas, p, H, unit)
    if not canvas.any():
        return np.zeros((h, w), np.uint8)
    canvas_to_cam = np.linalg.inv(H) @ np.diag([1 / SUPERSAMPLE, 1 / SUPERSAMPLE, 1.0])
    warped = cv2.warpPerspective(canvas, canvas_to_cam, (w, h), flags=cv2.INTER_LINEAR, borderValue=0)
    return cv2.threshold(warped, 0, 255, cv2.THRESH_BINARY)[1]


@dataclass(frozen=True)
class _Shown:
    """Something the page was told to show from ``t`` on."""

    n: int  # unique, for cache keys
    t: float
    full: bool  # a measurement/pattern mode: the whole bench is lit
    drawn: np.ndarray | None  # undilated overlay; None = nothing drawn
    mask: np.ndarray | None  # dilated overlay


def _readonly(a: np.ndarray) -> np.ndarray:
    a.flags.writeable = False
    return a


class OverlayMask:
    """Timeline of what the projector page shows, queried by camera-frame time. Thread-safe.

    ``frame_size`` = camera (w, h); ``H_cam_to_page`` and ``page_size`` (CSS px) come from the calibration (C7
    ``H_cam_to_page``, ``page.w``/``page.h``) that the page itself uses.
    """

    def __init__(self, frame_size: Sequence[int], H_cam_to_page, page_size: Sequence[float], *, dilate_px: int | None = None, hold_s: float = HOLD_S):
        self.frame_size = (int(frame_size[0]), int(frame_size[1]))
        H = np.asarray(H_cam_to_page, dtype=np.float64)
        if H.shape != (3, 3) or abs(np.linalg.det(H)) < 1e-12:
            raise ValueError("H_cam_to_page must be an invertible 3x3 matrix")
        self._H = H
        self._page_size = (float(page_size[0]), float(page_size[1]))
        w, h = self.frame_size
        self.dilate_px = dilate_px if dilate_px is not None else max(1, round(DILATE_PX_1080 * h / 1080))
        self._kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * self.dilate_px + 1, 2 * self.dilate_px + 1))
        self.hold_s = hold_s
        self._full = _readonly(np.full((h, w), 255, np.uint8))
        self._lock = threading.Lock()
        self._ids = itertools.count()
        self._timeline: list[_Shown] = []
        self._latest: _Shown | None = None  # latest normal scene; the page shows it again after a measurement mode
        self._held = False
        self._released_t: float | None = None
        self._cache_key: tuple[int, ...] | None = None
        self._cache: np.ndarray | None = None

    # ---- writers (any thread)

    def set_scene(self, scene: render.SceneMsg, t: float) -> None:
        """The page was sent ``scene`` at ``t``. Call it wherever the scene is published to the render hub."""
        if scene.mode != "normal":
            entry = _Shown(next(self._ids), t, True, None, None)
        else:
            drawn = rasterize(scene.prims, self._H, self._page_size, self.frame_size) if scene.prims else None
            if drawn is not None and not drawn.any():
                drawn = None
            mask = None if drawn is None else _readonly(cv2.dilate(drawn, self._kernel))
            entry = _Shown(next(self._ids), t, False, None if drawn is None else _readonly(drawn), mask)
        with self._lock:
            if not entry.full:
                self._latest = entry
            if not self._held:
                self._push(entry)

    def hold(self, t: float) -> None:
        """A measurement mode lights the whole bench from ``t`` (call before asking the page to switch)."""
        with self._lock:
            if not self._held:
                self._held = True
                self._push(_Shown(next(self._ids), t, True, None, None))

    def release(self, t: float) -> None:
        """The page shows the latest scene again from ``t`` (its ACK of ``show_mode('normal')``)."""
        with self._lock:
            if not self._held:
                return
            self._held = False
            self._released_t = t
            if self._latest is not None:
                self._push(replace(self._latest, n=next(self._ids), t=t))
            else:
                self._push(_Shown(next(self._ids), t, False, None, None))

    # ---- readers (any thread)

    def mask_at(self, t: float) -> np.ndarray | None:
        """Pixels to ignore in a camera frame dequeued at ``t``: full frame during a measurement mode, the dilated
        union of every overlay possibly visible, or None when nothing is projected."""
        with self._lock:
            visible = self._visible(t)
            if any(e.full for e in visible):
                return self._full
            shown = [e for e in visible if e.mask is not None]
            if not shown:
                return None
            if len(shown) == 1:
                return shown[0].mask
            key = tuple(e.n for e in shown)
            if key != self._cache_key:
                union = shown[0].mask.copy()
                for e in shown[1:]:
                    cv2.bitwise_or(union, e.mask, dst=union)
                self._cache_key, self._cache = key, _readonly(union)
            return self._cache

    def drawn_at(self, t: float) -> np.ndarray | None:
        """Undilated union of the overlays possibly visible at ``t`` (None when nothing is drawn)."""
        with self._lock:
            drawn = [e.drawn for e in self._visible(t) if e.drawn is not None]
        if not drawn:
            return None
        out = drawn[0].copy()
        for d in drawn[1:]:
            cv2.bitwise_or(out, d, dst=out)
        return out

    def clear_after(self) -> float | None:
        """Frames dequeued after this time no longer show the last measurement mode (None: never held)."""
        with self._lock:
            if self._held:
                return math.inf
            return None if self._released_t is None else self._released_t + self.hold_s

    # ---- internals (lock held)

    def _push(self, entry: _Shown) -> None:
        tl = self._timeline
        tl.insert(bisect.bisect_right(tl, entry.t, key=lambda e: e.t), entry)
        newest = tl[-1].t
        drop = 0
        while drop < len(tl) - 1 and tl[drop + 1].t + self.hold_s < newest - KEEP_S:
            drop += 1
        del tl[:drop]

    def _visible(self, t: float) -> list[_Shown]:
        tl = self._timeline
        return [e for i, e in enumerate(tl) if e.t <= t and (i + 1 == len(tl) or t < tl[i + 1].t + self.hold_s)]
