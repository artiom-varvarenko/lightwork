"""Projected playground (``lightwork play``): an interactive dashboard drawn on the bench.

Not a procedure run and not evidence: no model calls, nothing recorded or saved. It exercises the rig end to end:
calibration flash -> C2 scenes on the projector (and the camera view on /screen) -> local MediaPipe fingertip
tracking -> OpenCV change detection in a projected drop zone.

Widgets are laid out in projector page CSS px and mapped into camera px (the C2 coordinate system) through the
inverse of ``H_cam_to_page``; the page maps them back, so they land where they were designed. Circles are sent as
polygons for the same reason: a camera-px ring would become an ellipse on a keystoned page.

Interaction: hold a fingertip on a button for ``DWELL_S`` to press it; the slider follows the fingertip directly.
The drop zone compares each frame with an empty-zone reference (masking our own projected outlines) and commits a
new object list only after it has been still for ``SETTLE_S`` with no hand in or near the zone.
"""

from __future__ import annotations

import math
import time
from collections import deque
from dataclasses import dataclass, field

import cv2
import numpy as np

from .contracts import render
from .geometry import apply_h
from .vision.hands import HandPose, HandTracker
from .vision.overlay_mask import rasterize

DESIGN_W, DESIGN_H = 750.0, 420.0  # layout space; scaled uniformly onto the real page
DWELL_S = 0.8
TIP_HOLD_S = 0.25  # keep the last fingertip through short tracking dropouts
SMOOTH = 0.5  # fingertip EMA weight of the newest sample
SETTLE_S = 0.6  # drop zone: candidate must hold this long before it is committed
HISTORY_DT = 0.25
HISTORY_LEN = 60
OVERLAY_HOLD_S = 0.5  # camera sees a scene change this late, at worst
EDGE_PX = 9  # band inside the zone edge; a blob reaching it is an arm or an object across the edge
INSET_PX = 14  # zone interior starts this far inside the projected outline (camera px)
MATCH_PX = 14  # same object if its centroid moved less than this
LIGHT_FG = 0.25  # foreground share of the zone that means "the light changed", not "an object arrived"
LIGHT_GAIN = 0.12  # per-channel gain change that triggers a relearn even when compensation copes
RELEARN_AFTER_S = 2.0  # a light change must hold this long before the reference is relearned
STUCK_EDGE_S = 5.0  # something still across the zone edge this long is absorbed into the reference
STILL_S = 1.0  # ... with the view still and no hand seen for this long
STILL_DIFF = 3.0  # mean grey-level change between updates below which the view counts as still
ADAPT_RATE = 0.03  # per update, quiet background pixels move this far toward the current frame
THEMES = (("target", "source"), ("warn", "error"), ("source", "warn"))

# Seven-segment digits in a 1 x 2 box.
SEGMENTS = {
    "a": ((0, 0), (1, 0)), "b": ((1, 0), (1, 1)), "c": ((1, 1), (1, 2)), "d": ((0, 2), (1, 2)),
    "e": ((0, 1), (0, 2)), "f": ((0, 0), (0, 1)), "g": ((0, 1), (1, 1)),
}
DIGITS = {
    "0": "abcdef", "1": "bc", "2": "abged", "3": "abgcd", "4": "fgbc",
    "5": "afgcd", "6": "afgedc", "7": "abc", "8": "abcdefg", "9": "abcdfg",
}
COLOUR_ROLE = {"red": "error", "orange": "warn", "yellow": "warn", "green": "target", "cyan": "source",
               "blue": "source", "purple": "source"}


# ---------------------------------------------------------------------------------------------------- geometry


class Geo:
    """Page CSS px <-> camera px, plus the design-space -> page transform."""

    def __init__(self, H_cam_to_page, page_w: float, page_h: float):
        self.H = np.asarray(H_cam_to_page, dtype=np.float64)
        self.Hi = np.linalg.inv(self.H)
        self.page_w, self.page_h = float(page_w), float(page_h)
        self.s = min(self.page_w / DESIGN_W, self.page_h / DESIGN_H)
        self.ox = (self.page_w - DESIGN_W * self.s) / 2
        self.oy = (self.page_h - DESIGN_H * self.s) / 2

    def d2p(self, pts) -> np.ndarray:
        p = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
        return np.column_stack([self.ox + p[:, 0] * self.s, self.oy + p[:, 1] * self.s])

    def p2d(self, xy) -> tuple[float, float]:
        return ((xy[0] - self.ox) / self.s, (xy[1] - self.oy) / self.s)

    def d2c(self, pts) -> np.ndarray:  # design -> camera px
        return apply_h(self.Hi, self.d2p(pts))

    def c2d(self, xy) -> tuple[float, float]:  # camera px -> design
        return self.p2d(apply_h(self.H, [xy])[0])


class Canvas:
    """Collects C2 primitives from design-space geometry (or camera px for detected objects)."""

    def __init__(self, geo: Geo):
        self.geo = geo
        self.prims: list = []

    @staticmethod
    def _xy(cam) -> list[list[float]]:
        return [[round(float(x), 1), round(float(y), 1)] for x, y in cam]

    def poly(self, id: str, pts, role: str, closed: bool = False, **fx) -> None:
        """``fx``: optional C2 effects (``glow``, ``draw_ms``)."""
        self.prims.append(render.Poly(id=id, pts=self._xy(self.geo.d2c(pts)), closed=closed, role=role, **fx))

    def cam_poly(self, id: str, pts_cam, role: str, closed: bool = True, **fx) -> None:
        self.prims.append(render.Poly(id=id, pts=self._xy(pts_cam), closed=closed, role=role, **fx))

    def cam_dot(self, id: str, xy_cam, role: str) -> None:
        self.prims.append(render.Dot(id=id, c=self._xy([xy_cam])[0], role=role))

    def rect(self, id: str, x0, y0, x1, y1, role: str) -> None:
        self.poly(id, [(x0, y0), (x1, y0), (x1, y1), (x0, y1)], role, closed=True)

    def circle(self, id: str, c, r: float, role: str, n: int = 40) -> None:
        a = np.linspace(0, 2 * math.pi, n, endpoint=False)
        self.poly(id, np.column_stack([c[0] + r * np.cos(a), c[1] + r * np.sin(a)]), role, closed=True)

    def arc(self, id: str, c, r: float, frac: float, role: str) -> None:
        frac = min(max(frac, 0.0), 1.0)
        if frac <= 0.02:
            return
        a = -math.pi / 2 + np.linspace(0, 2 * math.pi * frac, max(3, int(48 * frac) + 1))
        self.poly(id, np.column_stack([c[0] + r * np.cos(a), c[1] + r * np.sin(a)]), role)

    def label(self, id: str, at, text: str, role: str, **fx) -> None:
        """``fx``: optional C2 effects (``type_ms``)."""
        self.prims.append(render.Label(id=id, at=self._xy(self.geo.d2c([at]))[0], text=text, role=role, **fx))

    def dot(self, id: str, c, role: str) -> None:
        self.prims.append(render.Dot(id=id, c=self._xy(self.geo.d2c([c]))[0], role=role))

    def picture(self, id: str, src: str, x0: float, y0: float, x1: float, y1: float, **fx) -> None:
        """A served picture over a design-px rect, its top along the rect's top: upright for the person at the table."""
        corners = self.geo.d2c([(x0, y0), (x1, y0), (x1, y1), (x0, y1)])
        self.prims.append(render.Picture(id=id, src=src, corners=self._xy(corners), **fx))

    def wire(self, id: str, src: str, at, unit: float, yaw: float, tilt: float, **fx) -> None:
        """A served wireframe: its middle at ``at``, one model unit ``unit`` design px, seen turned and from above."""
        o, x, y = self._xy(self.geo.d2c([at, (at[0] + unit, at[1]), (at[0], at[1] - unit)]))
        self.prims.append(render.Wire3D(id=id, src=src, origin=o, ex=x, ey=y, yaw=round(float(yaw), 4),
                                        tilt=round(float(tilt), 4), **fx))

    def map_view(self, id: str, rect, center, zoom: float, bearing: float, pitch: float, flight: int = 0,
                 fly_ms: int = 0) -> None:
        """A live map over a design-px rect. ``center`` is (lon, lat)."""
        x0, y0, x1, y1 = rect
        corners = self.geo.d2c([(x0, y0), (x1, y0), (x1, y1), (x0, y1)])
        self.prims.append(render.MapView(id=id, corners=self._xy(corners), center=[round(float(center[0]), 6), round(float(center[1]), 6)],
                                         zoom=round(float(zoom), 3), bearing=round(float(bearing), 2),
                                         pitch=round(float(pitch), 2), flight=flight, fly_ms=fly_ms))

    def number(self, id: str, value: str, x: float, y: float, w: float, role: str) -> None:
        """Seven-segment digits, each w wide and 2w tall, top-left at (x, y)."""
        inset = 0.14
        for i, ch in enumerate(value):
            ox = x + i * w * 1.45
            for seg in DIGITS.get(ch, ""):
                (ax, ay), (bx, by) = SEGMENTS[seg]
                # shorten each segment a little so the digit reads as segments, not a wireframe
                dx, dy = (bx - ax) * inset, (by - ay) * inset
                pts = [(ox + (ax + dx) * w, y + (ay + dy) * w), (ox + (bx - dx) * w, y + (by - dy) * w)]
                self.poly(f"{id}-{i}{seg}", pts, role)


# ---------------------------------------------------------------------------------------------------- drop zone


@dataclass(frozen=True)
class Obj:
    centre: tuple[float, float]  # camera px
    box: tuple[tuple[float, float], ...]  # rotated bounding box, camera px
    area: int
    colour: str


def colour_name(bgr: np.ndarray, board_v: float) -> str:
    """Rough colour word for the median of ``bgr`` (N x 3), relative to the empty board's brightness."""
    if len(bgr) == 0:
        return "?"
    med = np.median(bgr, axis=0).astype(np.uint8).reshape(1, 1, 3)
    h, s, v = (int(x) for x in cv2.cvtColor(med, cv2.COLOR_BGR2HSV)[0, 0])
    if v < 0.3 * board_v:
        return "black"
    if s < 55:
        return "white" if v > 0.8 * board_v else "grey"
    for limit, name in ((8, "red"), (22, "orange"), (35, "yellow"), (85, "green"), (100, "cyan"), (130, "blue"), (165, "purple")):
        if h < limit:
            return name
    return "red"


class DropZone:
    """Change detection inside a projected rectangle against an empty-zone reference.

    Light changes, in three layers:

    1. Every update divides out a global per-channel gain, the median ratio of current to reference brightness over
       background pixels (zone interior minus our overlay and the committed objects). A lamp dimming evenly or a
       colour-temperature shift then does not light up the whole zone.
    2. Background pixels (not foreground, not an object, not overlay) blend slowly into the reference, which absorbs
       gradual uneven drift such as daylight moving across the zone. The reference keeps its overall level while it
       blends, so an even change shows in the gain alone (and the gain cannot creep on a static scene).
    3. A large change (foreground over ``LIGHT_FG`` of the zone, or a gain beyond ``LIGHT_GAIN``) that holds while the
       view is still and no hand is seen for ``RELEARN_AFTER_S`` relearns the reference from the current frame,
       keeping the old background, rescaled, under committed objects so that they stay detected. Something still
       across the zone edge (a local light change, a shadow, an object) is absorbed the same way after
       ``STUCK_EDGE_S``.
    """

    def __init__(self, geo: Geo, rect_design, frame_size: tuple[int, int], *, threshold: int = 28, min_area: int = 500):
        x0, y0, x1, y1 = rect_design
        poly = geo.d2c([(x0, y0), (x1, y0), (x1, y1), (x0, y1)])
        fw, fh = frame_size
        bx, by, bw, bh = cv2.boundingRect(np.round(poly).astype(np.int32))
        bx, by = max(0, bx), max(0, by)
        bw, bh = min(fw, bx + bw) - bx, min(fh, by + bh) - by
        if bw <= 2 * INSET_PX or bh <= 2 * INSET_PX:
            raise ValueError("drop zone is outside the camera view")
        self.roi = (bx, by, bw, bh)
        full = np.zeros((bh, bw), np.uint8)
        cv2.fillPoly(full, [np.round(poly - (bx, by)).astype(np.int32)], 255)
        k = lambda r: cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
        self.inner = cv2.erode(full, k(INSET_PX))
        self.edge = cv2.subtract(self.inner, cv2.erode(self.inner, k(EDGE_PX)))
        self.inner_px = int(np.count_nonzero(self.inner))
        self.threshold = threshold
        self.min_area = min_area
        self._open, self._close, self._grow = k(2), k(5), k(12)
        self.ref: np.ndarray | None = None  # float32 blurred crop, in the light of the last (re)learn
        self.board_v = 128.0
        self.objects: tuple[Obj, ...] = ()
        self.status = "learning"  # learning | ready | busy | edge | light
        self.gain = np.ones(3, np.float32)  # current light / reference light, per channel
        self.fg_fraction = 0.0
        self.relearns = 0  # automatic relearns since start
        self._objmask = np.zeros((bh, bw), np.uint8)
        self._cand: tuple[Obj, ...] | None = None
        self._cand_since = 0.0
        self._prev: np.ndarray | None = None
        self._still_since = 0.0
        self._light_since: float | None = None
        self._edge_since: float | None = None
        self._hand_seen = -1e9

    def _crop(self, img: np.ndarray) -> np.ndarray:
        x, y, w, h = self.roi
        return img[y:y + h, x:x + w]

    def _prep(self, img: np.ndarray) -> np.ndarray:
        return cv2.GaussianBlur(self._crop(img), (7, 7), 0).astype(np.float32)

    def _set_ref(self, ref: np.ndarray, img: np.ndarray) -> None:
        self.ref = ref
        hsv = cv2.cvtColor(self._crop(img), cv2.COLOR_BGR2HSV)
        self.board_v = float(np.median(hsv[..., 2][self.inner > 0])) or 128.0
        self.gain = np.ones(3, np.float32)
        self._light_since = self._edge_since = None

    def learn(self, img: np.ndarray) -> None:
        """Manual (re)learn: the zone must be empty."""
        self._set_ref(self._prep(img), img)
        self.objects = ()
        self._objmask[:] = 0
        self._cand = None
        self.status = "ready"

    def _relearn(self, cur: np.ndarray, img: np.ndarray) -> None:
        """Automatic relearn after a light change; committed objects keep the old background under them."""
        ref = cur.copy()
        keep = self._objmask > 0
        ref[keep] = self.ref[keep] * self.gain
        self._set_ref(ref, img)
        self._cand = None
        self.relearns += 1

    def _background(self, overlay: np.ndarray | None) -> np.ndarray:
        bg = self.inner.copy()
        bg[self._objmask > 0] = 0
        if overlay is not None:
            bg[overlay > 0] = 0
        return bg

    def _estimate_gain(self, cur: np.ndarray, bg: np.ndarray) -> np.ndarray:
        sel = bg[::4, ::4] > 0
        if np.count_nonzero(sel) < 200:
            return self.gain
        c, r = cur[::4, ::4][sel], self.ref[::4, ::4][sel]
        return (np.median(c, axis=0) / np.maximum(np.median(r, axis=0), 1.0)).astype(np.float32)

    def _adapt(self, cur: np.ndarray, bg: np.ndarray, fg: np.ndarray) -> None:
        """Blend quiet background into the reference (uneven drift), keeping the reference's overall level: an even
        change belongs to ``gain`` alone. Unanchored, the two fed each other and the gain crept on a static wall
        (1 Nov: 1.00 -> 0.88 within a minute, a relearn every 1-2 min with nothing changing)."""
        sel = bg[::4, ::4] > 0
        if np.count_nonzero(sel) < 200:
            return
        before = np.median(self.ref[::4, ::4][sel], axis=0)
        adapt = (bg > 0) & (fg == 0)
        self.ref[adapt] += ADAPT_RATE * (cur[adapt] / self.gain - self.ref[adapt])
        after = np.median(self.ref[::4, ::4][sel], axis=0)
        self.ref *= (before / np.maximum(after, 1.0)).astype(np.float32)

    def _detect(self, cur: np.ndarray, img: np.ndarray, overlay: np.ndarray | None) -> tuple[tuple[Obj, ...], bool, np.ndarray]:
        d = np.abs(cur - self.ref * self.gain).max(axis=2)
        fg = (d > self.threshold).astype(np.uint8) * 255
        fg &= self.inner
        if overlay is not None:
            fg &= cv2.bitwise_not(overlay)
        fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, self._open)
        fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, self._close)
        self.fg_fraction = float(np.count_nonzero(fg)) / max(1, self.inner_px)
        n, labels, stats, cents = cv2.connectedComponentsWithStats(fg, connectivity=8)
        crop = np.clip(self._crop(img) / np.maximum(self.gain, 0.05), 0, 255).astype(np.uint8)  # colours in learned light
        x0, y0 = self.roi[:2]
        objs, edge = [], False
        for i in range(1, n):
            area = int(stats[i, cv2.CC_STAT_AREA])
            if area < self.min_area:
                continue
            comp = labels == i
            if np.count_nonzero(comp & (self.edge > 0)) > 20:
                edge = True
                continue
            pts = np.column_stack(np.nonzero(comp)[::-1]).astype(np.float32)
            box = cv2.boxPoints(cv2.minAreaRect(pts)) + (x0, y0)
            objs.append(Obj(centre=(float(cents[i][0] + x0), float(cents[i][1] + y0)),
                            box=tuple((float(px), float(py)) for px, py in box), area=area,
                            colour=colour_name(crop[comp], self.board_v)))
        objs.sort(key=lambda o: (o.centre[0], o.centre[1]))
        return tuple(objs), edge, fg

    @staticmethod
    def _same(a: tuple[Obj, ...], b: tuple[Obj, ...] | None) -> bool:
        return b is not None and len(a) == len(b) and all(math.dist(p.centre, q.centre) < MATCH_PX for p, q in zip(a, b, strict=False))

    def _commit(self, objs: tuple[Obj, ...]) -> None:
        self.objects = objs
        self._objmask[:] = 0
        x0, y0 = self.roi[:2]
        for o in objs:
            cv2.fillPoly(self._objmask, [np.round(np.array(o.box) - (x0, y0)).astype(np.int32)], 255)
        self._objmask = cv2.dilate(self._objmask, self._grow)

    def update(self, img: np.ndarray, t: float, *, overlay: np.ndarray | None, hand_near: bool,
               hand_visible: bool = False) -> bool:
        """Returns True when the committed object list changed. ``overlay`` is a full-frame camera-px mask."""
        if self.ref is None:
            return False
        if hand_visible or hand_near:
            self._hand_seen = t
        cur = self._prep(img)
        ov = None if overlay is None else self._crop(overlay)
        if self._prev is None or float(np.mean(np.abs(cur[::4, ::4] - self._prev[::4, ::4]))) > STILL_DIFF:
            self._still_since = t
        self._prev = cur
        bg = self._background(ov)
        self.gain = self._estimate_gain(cur, bg)
        objs, edge, fg = self._detect(cur, img, ov)

        quiet = t - self._still_since >= STILL_S and t - self._hand_seen >= STILL_S
        if hand_near:
            self.status, self._cand = "busy", None
            self._light_since = self._edge_since = None
            return False
        light = self.fg_fraction > LIGHT_FG or float(np.max(np.abs(self.gain - 1.0))) > LIGHT_GAIN
        if light:
            self._light_since = t if self._light_since is None else self._light_since
            self.status, self._cand = "light", None
            if t - self._light_since >= RELEARN_AFTER_S and quiet:
                self._relearn(cur, img)
            return False
        self._light_since = None
        if edge:
            self._edge_since = t if self._edge_since is None else self._edge_since
            self.status, self._cand = "edge", None
            if t - self._edge_since >= STUCK_EDGE_S and quiet:
                self._relearn(cur, img)
            return False
        self._edge_since = None
        self.status = "ready"
        self._adapt(cur, bg, fg)
        if not self._same(objs, self._cand):
            self._cand, self._cand_since = objs, t
            return False
        if t - self._cand_since < SETTLE_S or self._same(objs, self.objects):
            return False
        self._commit(objs)
        return True


# ---------------------------------------------------------------------------------------------------- dashboard


@dataclass
class Button:
    key: str
    text: str
    c: tuple[float, float]
    r: float = 26.0
    dwell: float = 0.0
    armed: bool = True
    flash_until: float = 0.0


@dataclass
class Dashboard:
    """Widget state and the scene builder. All geometry in design space (``DESIGN_W`` x ``DESIGN_H``)."""

    geo: Geo
    zone_rect: tuple[float, float, float, float] = (440.0, 52.0, 732.0, 330.0)
    slider: tuple[float, float, float] = (190.0, 405.0, 350.0)  # x0, x1, y
    buttons: list[Button] = field(default_factory=lambda: [
        Button("plus", "+1", (52.0, 105.0)), Button("reset", "RESET", (52.0, 180.0)),
        Button("theme", "THEME", (52.0, 255.0)), Button("learn", "LEARN", (52.0, 330.0)),
    ])
    count: int = 0
    level: float = 0.5
    theme: int = 0
    history: deque = field(default_factory=lambda: deque(maxlen=HISTORY_LEN))
    tip: tuple[float, float] | None = None  # design space, smoothed
    tip_seen: float = -1.0
    sliding: bool = False
    offset: tuple[float, float] = (0.0, 0.0)  # fingertip landmark - contact point, from the hand check (design px)
    _hist_t: float = 0.0

    @property
    def accent(self) -> str:
        return THEMES[self.theme][0]

    @property
    def second(self) -> str:
        return THEMES[self.theme][1]

    def track(self, tip_cam: tuple[float, float] | None, now: float) -> None:
        if tip_cam is not None:
            raw = self.geo.c2d(tip_cam)
            d = (raw[0] - self.offset[0], raw[1] - self.offset[1])
            if self.tip is None or now - self.tip_seen > TIP_HOLD_S:
                self.tip = d
            else:
                self.tip = (self.tip[0] + SMOOTH * (d[0] - self.tip[0]), self.tip[1] + SMOOTH * (d[1] - self.tip[1]))
            self.tip_seen = now
        elif self.tip is not None and now - self.tip_seen > TIP_HOLD_S:
            self.tip = None

    def near_zone(self, margin: float = 60.0) -> bool:
        if self.tip is None:
            return False
        x0, y0, x1, y1 = self.zone_rect
        return x0 - margin <= self.tip[0] <= x1 + margin and y0 - margin <= self.tip[1] <= y1 + margin

    def step(self, now: float, dt: float) -> list[str]:
        """Advance buttons and slider; returns the keys of buttons pressed this frame."""
        dt = min(max(dt, 0.0), 0.1)
        fired = []
        for b in self.buttons:
            inside = self.tip is not None and math.dist(self.tip, b.c) <= b.r + 10
            if not inside:
                b.dwell, b.armed = 0.0, True
            elif b.armed:
                b.dwell += dt
                if b.dwell >= DWELL_S:
                    b.dwell, b.armed, b.flash_until = 0.0, False, now + 0.5
                    fired.append(b.key)
        self.sliding = self.on_slider()
        if self.sliding:
            self.set_level(self.tip[0])
        self._sample(now)
        for key in fired:
            self.apply(key)
        return fired

    def on_slider(self) -> bool:
        x0, x1, y = self.slider
        return self.tip is not None and abs(self.tip[1] - y) < 26 and x0 - 12 <= self.tip[0] <= x1 + 12

    def set_level(self, x: float) -> None:
        x0, x1, _ = self.slider
        self.level = min(max((x - x0) / (x1 - x0), 0.0), 1.0)

    def button_at(self) -> Button | None:
        if self.tip is None:
            return None
        return next((b for b in self.buttons if math.dist(self.tip, b.c) <= b.r + 10), None)

    def _sample(self, now: float) -> None:
        if now - self._hist_t >= HISTORY_DT:
            self._hist_t = now
            self.history.append(self.level)

    def apply(self, key: str) -> None:
        if key == "plus":
            self.count = min(999, self.count + 1)
        elif key == "reset":
            self.count = 0
            self.history.clear()
        elif key == "theme":
            self.theme = (self.theme + 1) % len(THEMES)

    def scene(self, zone: DropZone | None, clock: str) -> tuple[list, list]:
        """(all prims, the zone's object prims). Object prims are listed separately for the overlay mask."""
        c = Canvas(self.geo)
        self._widgets(c, clock)
        zone_prims, objs = self._zone(zone)
        c.prims += zone_prims
        self._cursor(c)
        return c.prims, objs

    def _widgets(self, c: Canvas, clock: str) -> None:
        now = time.monotonic()
        c.label("title", (18, 24), "LIGHTWORK", "info")
        c.label("clock", (628, 24), clock, "info")
        for b in self.buttons:
            hot = self.tip is not None and math.dist(self.tip, b.c) <= b.r + 10
            role = self.accent if now < b.flash_until else ("warn" if hot and b.armed else "info")
            c.circle(f"btn-{b.key}", b.c, b.r, role)
            c.arc(f"btn-{b.key}-arc", b.c, b.r + 9, b.dwell / DWELL_S, self.accent)
            c.label(f"btn-{b.key}-txt", (b.c[0] + b.r + 12, b.c[1]), b.text, "info")
        # counter
        c.label("count-k", (190, 78), "COUNT", "info")
        c.number("count", str(self.count), 190, 100, 30, self.accent)
        # sparkline of the level
        sx0, sy0, sx1, sy1 = 190, 195, 405, 268
        c.poly("spark-axis", [(sx0, sy0), (sx0, sy1), (sx1, sy1)], "info")
        if len(self.history) >= 2:
            n = HISTORY_LEN - 1
            pts = [(sx1 - (len(self.history) - 1 - i) * (sx1 - sx0) / n, sy1 - v * (sy1 - sy0)) for i, v in enumerate(self.history)]
            c.poly("spark", pts, self.second)
        # slider
        x0, x1, y = self.slider
        c.label("level-k", (190, 305), f"LEVEL {round(self.level * 100):d}%", "info")
        c.poly("slider", [(x0, y), (x1, y)], "info")
        kx = x0 + self.level * (x1 - x0)
        c.circle("slider-knob", (kx, y), 13, self.second if self.sliding else self.accent, n=24)

    def _zone(self, zone: DropZone | None) -> tuple[list, list]:
        """(zone outline, status and object prims; the object prims alone, for the overlay mask)."""
        c = Canvas(self.geo)
        zx0, zy0, zx1, zy1 = self.zone_rect
        if zone is None:
            c.rect("zone", zx0, zy0, zx1, zy1, "info")
            c.label("zone-k", (zx0, 362), "NO ZONE", "info")
            return c.prims, []
        role = {"busy": "warn", "edge": "warn", "light": "warn", "learning": "info"}.get(zone.status)
        role = role or (self.accent if zone.objects else "info")
        c.rect("zone", zx0, zy0, zx1, zy1, role)
        c.label("zone-k", (zx0, 362), zone_text(zone), role)
        for i, o in enumerate(zone.objects, 1):
            centre = np.array(o.centre)
            box = centre + (np.array(o.box) - centre) * 1.15  # a little outside the object
            c.cam_poly(f"obj-{i}", box, COLOUR_ROLE.get(o.colour, self.accent))
            top = box[np.argmin(box[:, 1])]
            d = self.geo.c2d(top)
            c.label(f"obj-{i}-txt", (max(zx0 + 4, d[0] - 20), max(zy0 + 14, d[1] - 18)), f"{i} {o.colour.upper()}", "info")
        return c.prims, c.prims[2:]  # the outline and status sit outside the detection interior

    def _cursor(self, c: Canvas) -> None:
        """Fingertip cursor; drawn last so it is on top."""
        if self.tip is not None:
            hot = self.button_at() is not None or self.sliding
            c.circle("cursor", self.tip, 15, self.accent if hot else "info", n=24)


def zone_text(zone: DropZone) -> str:
    if zone.status == "learning":
        return "LEARNING ZONE..."
    if zone.status == "busy":
        return "HAND IN ZONE"
    if zone.status == "edge":
        return "KEEP INSIDE THE ZONE"
    if zone.status == "light":
        return "LIGHT CHANGED..."
    n = len(zone.objects)
    return "ZONE EMPTY" if n == 0 else f"{n} OBJECT{'S' if n != 1 else ''}"


def spoken(zone: DropZone) -> str:
    n = len(zone.objects)
    if n == 0:
        return "Zone empty."
    words = ["One", "Two", "Three", "Four", "Five", "Six", "Seven", "Eight", "Nine"]
    head = words[n - 1] if n <= len(words) else str(n)
    colours = [o.colour for o in zone.objects if o.colour != "?"]
    tail = ": " + ", ".join(colours) if colours else ""
    return f"{head} object{'s' if n != 1 else ''}{tail}."


class OverlayTimeline:
    """Camera-px mask of our projected object outlines, held as the union of old and new after each change."""

    def __init__(self, geo: Geo, frame_size: tuple[int, int]):
        self.geo, self.frame_size = geo, frame_size
        self._key: object = None
        self._mask: np.ndarray | None = None
        self._prev: np.ndarray | None = None
        self._changed = -1e9
        self._kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))

    def set(self, prims: list, t: float) -> None:
        key = tuple(p.model_dump_json() for p in prims)
        if key == self._key:
            return
        self._key = key
        mask = None
        if prims:
            mask = cv2.dilate(rasterize(prims, self.geo.H, (self.geo.page_w, self.geo.page_h), self.frame_size), self._kernel)
        self._prev, self._mask, self._changed = self._mask, mask, t

    def at(self, t: float) -> np.ndarray | None:
        if t - self._changed < OVERLAY_HOLD_S and self._prev is not None:
            return self._prev if self._mask is None else cv2.bitwise_or(self._prev, self._mask)
        return self._mask


# ---------------------------------------------------------------------------------------------------- ceiling mode
#
# Projection on a ceiling (or a wall): nothing can be touched or put down, so the hand works in mid-air between the
# camera and the surface. The cursor lands where the camera sees the fingertip against the surface. Pinching thumb
# and index finger clicks a button, drags the slider, or paints a light trail; an open hand held still clears the
# paint. The drop zone becomes a gesture panel, and the hand skeleton is projected where the camera sees the hand.

HAND_CHAINS = ((0, 1, 2, 3, 4), (0, 5, 6, 7, 8), (9, 10, 11, 12), (13, 14, 15, 16), (0, 17, 18, 19, 20), (5, 9, 13, 17))
FINGERS = ((1, 2, 3, 4), (5, 6, 7, 8), (9, 10, 11, 12), (13, 14, 15, 16), (17, 18, 19, 20))  # thumb..pinky joints
TIPS = (4, 8, 12, 16, 20)
PINCH_ON, PINCH_OFF = 0.30, 0.42  # thumb-index tip distance / palm length (wrist to middle knuckle), 3-D
CEILING_DWELL_S = 1.2  # dwell still presses buttons, more slowly than a pinch
CLEAR_HOLD_S = 1.2
GESTURE_FRAMES = 4  # a gesture must be seen this many frames in a row before it is shown
MAX_PAINT_PTS = 900
PAINT_STEP = 5.0  # design px between stored stroke points


class PoseTracker(HandTracker):
    """HandTracker that returns all 21 landmarks of the most confident hand (image and world coordinates)."""

    def pose(self, frame) -> HandPose | None:
        poses = self.poses(frame)
        return poses[0] if poses else None


def _cos(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    return float(a @ b) / (na * nb) if na > 1e-9 and nb > 1e-9 else 0.0


def palm_length(world: np.ndarray) -> float:
    return float(np.linalg.norm(world[9] - world[0])) or 1e-6


def fingers_extended(world: np.ndarray) -> tuple[bool, ...]:
    """Thumb..pinky straight and pointing away from the wrist, from 3-D landmarks (works for a tilted hand)."""
    palm = palm_length(world)
    thumb = _cos(world[3] - world[2], world[4] - world[3]) > 0.6 and np.linalg.norm(world[4] - world[9]) > 0.7 * palm
    out = [bool(thumb)]
    for mcp, pip, _dip, tip in FINGERS[1:]:
        straight = _cos(world[pip] - world[mcp], world[tip] - world[pip]) > 0.5
        away = np.linalg.norm(world[tip] - world[0]) > 1.1 * np.linalg.norm(world[pip] - world[0])
        out.append(bool(straight and away))
    return tuple(out)


def pinch_ratio(world: np.ndarray) -> float:
    """Thumb-index tip distance over palm length; infinite when the index finger is curled into the palm, so a
    fist (thumb resting on the curled index finger) is not a pinch."""
    palm = palm_length(world)
    if np.linalg.norm(world[8] - world[5]) < 0.5 * palm:
        return math.inf
    return float(np.linalg.norm(world[4] - world[8])) / palm


def gesture_name(ext: tuple[bool, ...], pinching: bool) -> str:
    n = sum(ext)
    if pinching:
        return "OK" if ext[2] and ext[3] and ext[4] else "PINCH"
    named = {
        (0, 0, 0, 0, 0): "FIST", (1, 1, 1, 1, 1): "OPEN HAND", (0, 1, 0, 0, 0): "POINT", (0, 1, 1, 0, 0): "PEACE",
        (1, 0, 0, 0, 0): "THUMB", (0, 1, 0, 0, 1): "ROCK", (1, 1, 0, 0, 1): "ROCK", (1, 0, 0, 0, 1): "SHAKA",
        (1, 1, 0, 0, 0): "L", (0, 1, 1, 1, 0): "THREE", (0, 1, 1, 1, 1): "FOUR",
    }
    return named.get(tuple(int(e) for e in ext), f"{n} FINGERS")


@dataclass
class CeilingDashboard(Dashboard):
    """Mid-air variant of :class:`Dashboard`; ``zone_rect`` holds the gesture panel."""

    buttons: list[Button] = field(default_factory=lambda: [
        Button("plus", "+1", (52.0, 105.0)), Button("reset", "RESET", (52.0, 180.0)),
        Button("theme", "THEME", (52.0, 255.0)), Button("clear", "CLEAR", (52.0, 330.0)),
    ])
    pose: HandPose | None = None
    ext: tuple[bool, ...] = ()
    pinching: bool = False
    gesture: str = ""
    drag: str | None = None  # "slider" | "paint" | None, set by a pinch onset
    strokes: list[list[tuple[float, float]]] = field(default_factory=list)
    clear_hold: float = 0.0
    clicks: int = 0
    _raw: str = ""
    _raw_n: int = 0

    @property
    def fingers(self) -> int:
        return sum(self.ext)

    def apply(self, key: str) -> None:
        if key == "clear":
            self.strokes.clear()
        else:
            super().apply(key)

    def on_panel(self) -> bool:
        x0, y0, x1, y1 = self.zone_rect
        return self.tip is not None and x0 - 10 <= self.tip[0] <= x1 + 10 and y0 - 10 <= self.tip[1] <= y1 + 10

    def update(self, pose: HandPose | None, now: float, dt: float) -> list[str]:
        """Advance with this frame's hand pose; returns the keys of buttons pressed."""
        dt = min(max(dt, 0.0), 0.1)
        was = self.pinching
        self.pose = pose
        if pose is None:
            self.track(None, now)
            self.pinching, self.ext, raw = False, (), ""
        else:
            ratio = pinch_ratio(pose.world)
            self.pinching = ratio < (PINCH_OFF if was else PINCH_ON)
            point = (pose.img[4] + pose.img[8]) / 2 if self.pinching else pose.img[8]
            self.track((float(point[0]), float(point[1])), now)
            self.ext = fingers_extended(pose.world)
            raw = gesture_name(self.ext, self.pinching)
        self._raw_n = self._raw_n + 1 if raw == self._raw else 1
        self._raw = raw
        if self._raw_n >= GESTURE_FRAMES or raw == "":
            self.gesture = raw

        fired: list[str] = []
        if self.pinching and not was and self.tip is not None:
            self.clicks += 1
            b = self.button_at()
            if b is not None:
                b.flash_until = now + 0.5
                fired.append(b.key)
            elif self.on_slider():
                self.drag = "slider"
            elif not self.on_panel():
                self.drag = "paint"
                self.strokes.append([self.tip])
        if not self.pinching:
            self.drag = None
        if self.drag == "slider" and self.tip is not None:
            self.set_level(self.tip[0])
        elif self.drag == "paint" and self.tip is not None and self.strokes:
            stroke = self.strokes[-1]
            if math.dist(stroke[-1], self.tip) >= PAINT_STEP:
                stroke.append(self.tip)
            while sum(len(s) for s in self.strokes) > MAX_PAINT_PTS and len(self.strokes) > 1:
                self.strokes.pop(0)
        self.sliding = self.drag == "slider"

        for b in self.buttons:  # dwell fallback for hands that do not pinch cleanly
            inside = not self.pinching and self.tip is not None and math.dist(self.tip, b.c) <= b.r + 10
            if not inside:
                b.dwell, b.armed = 0.0, True
            elif b.armed:
                b.dwell += dt
                if b.dwell >= CEILING_DWELL_S:
                    b.dwell, b.armed, b.flash_until = 0.0, False, now + 0.5
                    fired.append(b.key)

        if self.gesture == "OPEN HAND" and self.strokes:
            self.clear_hold += dt
            if self.clear_hold >= CLEAR_HOLD_S:
                self.strokes.clear()
                self.clear_hold = 0.0
        else:
            self.clear_hold = 0.0
        self._sample(now)
        for key in fired:
            self.apply(key)
        return fired

    def scene(self, zone: DropZone | None, clock: str) -> tuple[list, list]:
        c = Canvas(self.geo)
        self._widgets(c, clock)
        for b in self.buttons:  # the base draws dwell progress against DWELL_S; redraw it against CEILING_DWELL_S
            c.prims = [p for p in c.prims if p.id != f"btn-{b.key}-arc"]
            c.arc(f"btn-{b.key}-arc", b.c, b.r + 9, b.dwell / CEILING_DWELL_S, self.accent)
        self._panel(c)
        for i, stroke in enumerate(self.strokes):
            if len(stroke) >= 2:
                c.poly(f"paint-{i}", stroke, self.second)
            else:
                c.dot(f"paint-{i}", stroke[0], self.second)
        self._skeleton(c)
        self._cursor(c)
        return c.prims, []

    def _panel(self, c: Canvas) -> None:
        x0, y0, x1, y1 = self.zone_rect
        c.rect("panel", x0, y0, x1, y1, self.accent if self.pose is not None else "info")
        if self.pose is None:
            c.label("panel-k", (x0 + 14, (y0 + y1) / 2), "RAISE A HAND", "info")
        else:
            c.label("panel-k", (x0 + 14, y0 + 24), "FINGERS", "info")
            c.number("fingers", str(self.fingers), (x0 + x1) / 2 - 30, y0 + 50, 60, self.accent)
            if self.gesture:
                c.label("gesture", (x0 + 14, y1 - 28), self.gesture, self.second)
        c.label("hint-1", (x0, 362), "PINCH: CLICK, DRAW", "info")
        c.label("hint-2", (x0, 396), "OPEN HAND: CLEAR", "info")

    def _skeleton(self, c: Canvas) -> None:
        if self.pose is None:
            return
        for i, chain in enumerate(HAND_CHAINS):
            c.cam_poly(f"bone-{i}", self.pose.img[list(chain)], self.second, closed=False)
        for f, (tip, up) in enumerate(zip(TIPS, self.ext, strict=False)):
            if up:
                c.cam_dot(f"tip-{f}", self.pose.img[tip], self.accent)

    def _cursor(self, c: Canvas) -> None:
        if self.tip is None:
            return
        x = min(max(self.tip[0], 8.0), DESIGN_W - 8)
        y = min(max(self.tip[1], 8.0), DESIGN_H - 8)
        off = (x, y) != self.tip
        hot = self.button_at() is not None or self.sliding
        role = "warn" if off else (self.accent if hot or self.pinching else "info")
        c.circle("cursor", (x, y), 9 if self.pinching else 15, role, n=24)
        if self.clear_hold > 0:
            c.arc("clear-arc", (x, y), 26, self.clear_hold / CLEAR_HOLD_S, "error")
