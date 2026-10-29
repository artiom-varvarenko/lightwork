"""Simulated projector-camera rig: calibration tests and dry runs without the Tank or the C920.

World model: a bench plane in mm. The projector maps page CSS px -> bench mm (``page_to_bench``), the camera maps
bench mm -> camera px (``bench_to_cam``); bumping either changes one of them. :class:`SimPage` draws what
``web/js/projector.js`` draws (black page; the ArUco pattern of ``aruco.js``; measurement fills; ring and dot
prims through ``H_cam_to_page``). :class:`SimCamera` renders ``albedo x (ambient + gain x projector light)``,
blurred, with sensor noise, on demand on a fixed frame clock, showing what the page displayed ``photon_s`` earlier.

Pixel conventions: camera px are OpenCV pixel-centre coordinates; page CSS px follow the canvas (device pixel
``i`` covers ``[i, i + 1) / dpr``), so a page image pixel centre sits at ``(i + 0.5) / dpr`` CSS px.
"""

from __future__ import annotations

import math
import threading
import time
from collections.abc import Sequence

import cv2
import numpy as np

from ...contracts.render import Ack, Calib, Dot, Hello, MarkerRect, Pattern, Ring, SceneMsg
from ...geometry import apply_h
from ...server import projector_style as style
from ..source import Frame
from .mm import CARD, ID1_MM, CardSpec
from .pattern import MODULES, marker_modules

PATTERN_WHITE = 166 / 255  # aruco.js draws quiet zones and white bits with MEASURE_FILL.grey (#a6a6a6)
MEASURE_FILL = {"dark": 0.0, "grey": 166 / 255, "white": 1.0}
PALETTE_BGR = {  # scene.js PALETTES.projector
    "target": (0, 255, 0),
    "source": (255, 255, 0),
    "error": (0, 0, 255),
    "warn": (0, 255, 255),
    "done": (255, 0, 0),
    "info": (255, 255, 255),
}
BOARD = (0.84, 0.85, 0.85)  # matte white board albedo, BGR


def _translate(dx: float, dy: float) -> np.ndarray:
    return np.array([[1.0, 0.0, dx], [0.0, 1.0, dy], [0.0, 0.0, 1.0]])


def _quad_h(src: Sequence[Sequence[float]], dst: Sequence[Sequence[float]]) -> np.ndarray:
    return cv2.getPerspectiveTransform(np.float32(src), np.float32(dst)).astype(np.float64)


def _js_round(v: float) -> int:
    return math.floor(v + 0.5)


class SimRig:
    """The physical scene. Defaults: Tank at 0.5 m (0.52 mm per page px), C920 seeing ~740 mm across."""

    def __init__(
        self,
        *,
        cam_size: tuple[int, int] = (1920, 1080),
        bench_mm: tuple[float, float] = (760.0, 520.0),
        px_per_mm: float = 2.0,
        page_to_bench: np.ndarray | None = None,
        bench_to_cam: np.ndarray | None = None,
        ambient: float = 40.0,
        gain: float = 200.0,
        blur: float = 1.2,
        noise: float = 2.5,
        photon_s: float = 0.04,
        fps: float = 30.0,
        seed: int = 0,
        texture: bool = True,
    ):
        self.cam_size = cam_size
        self.bench_mm = bench_mm
        self.k = px_per_mm
        bw, bh = bench_mm
        if page_to_bench is None:  # 854x480 page -> 444x250 mm, slight keystone, centred
            cx, cy = bw / 2, bh / 2
            page_to_bench = _quad_h([[0, 0], [854, 0], [854, 480], [0, 480]], [[cx - 224, cy - 123], [cx + 220, cy - 126], [cx + 223, cy + 126], [cx - 221, cy + 124]])
        if bench_to_cam is None:  # ~2.6 camera px per mm at 1920x1080, mild perspective
            sx, sy = cam_size[0] / 1920, cam_size[1] / 1080
            view = [[-24, -138], [1958, -128], [1938, 1218], [-10, 1206]]
            bench_to_cam = _quad_h([[0, 0], [bw, 0], [bw, bh], [0, bh]], [[x * sx, y * sy] for x, y in view])
        self.page_to_bench = np.asarray(page_to_bench, dtype=np.float64)
        self.bench_to_cam = np.asarray(bench_to_cam, dtype=np.float64)
        self.ambient, self.gain, self.blur, self.noise = ambient, gain, blur, noise
        self.photon_s, self.fps = photon_s, fps
        self.rng = np.random.default_rng(seed)
        self.albedo = np.empty((round(bh * px_per_mm), round(bw * px_per_mm), 3), np.float32)
        self.albedo[:] = BOARD
        if texture:
            self._texture_margins(np.random.default_rng(seed + 1))
        self._blank = self.albedo.copy()
        self.occluders: list[np.ndarray] = []
        self._lock = threading.Lock()
        self._shown: list[tuple[float, int, np.ndarray, float]] = []  # (t_visible, show counter, page BGR 0..1, dpr)
        self._n_shown = 0
        self._cache: tuple[tuple[int, int], np.ndarray] | None = None
        self._albedo_cam: tuple[int, np.ndarray] | None = None
        self._version = 0

    # -- truth and bumps

    @property
    def H_page_to_cam(self) -> np.ndarray:
        h = self.bench_to_cam @ self.page_to_bench
        return h / h[2, 2]

    @property
    def H_cam_to_page(self) -> np.ndarray:
        h = np.linalg.inv(self.H_page_to_cam)
        return h / h[2, 2]

    def move_camera(self, dx_px: float, dy_px: float) -> None:
        with self._lock:
            self.bench_to_cam = _translate(dx_px, dy_px) @ self.bench_to_cam
            self._version += 1

    def move_projector(self, dx_mm: float, dy_mm: float) -> None:
        with self._lock:
            self.page_to_bench = _translate(dx_mm, dy_mm) @ self.page_to_bench
            self._version += 1

    def occlude(self, polygon_cam_px: Sequence[Sequence[float]]) -> None:
        """An opaque hand-coloured blob over the camera view (it blocks markers like a hand would)."""
        with self._lock:
            self.occluders.append(np.asarray(polygon_cam_px, dtype=np.float64))
            self._version += 1

    def clear_prints(self) -> None:
        """Take every card off the bench."""
        with self._lock:
            self.albedo[:] = self._blank
            self._version += 1

    # -- bench prints

    def _texture_margins(self, rng: np.random.Generator) -> None:
        """Random clutter outside the projection (tools, tape, wood grain) for the camera-bump patches."""
        h, w = self.albedo.shape[:2]
        for _ in range(900):
            x, y = int(rng.integers(0, w)), int(rng.integers(0, h))
            if 0.18 * w < x < 0.82 * w and 0.18 * h < y < 0.82 * h:
                continue
            colour = tuple(float(v) for v in rng.uniform(0.1, 0.8, 3))
            if rng.random() < 0.5:
                cv2.circle(self.albedo, (x, y), int(rng.integers(3, 14)), colour, -1, cv2.LINE_AA)
            else:
                cv2.rectangle(self.albedo, (x, y), (x + int(rng.integers(4, 30)), y + int(rng.integers(4, 30))), colour, -1)

    def _paint_mm(self, poly_mm: np.ndarray, albedo: Sequence[float]) -> None:
        pts = np.round((np.asarray(poly_mm) * self.k - 0.5) * 16).astype(np.int32)
        cv2.fillPoly(self.albedo, [pts], tuple(float(a) for a in albedo), cv2.LINE_AA, shift=4)

    def place_card(self, origin_mm: tuple[float, float], angle_deg: float = 0.0, spec: CardSpec = CARD) -> np.ndarray:
        """Print the calibration card on the bench; returns its sheet mm -> bench mm transform."""
        a = math.radians(angle_deg)
        T = np.array([[math.cos(a), -math.sin(a), origin_mm[0]], [math.sin(a), math.cos(a), origin_mm[1]], [0, 0, 1]])
        w, h = spec.sheet_mm
        with self._lock:
            self._paint_mm(apply_h(T, [[0, 0], [w, 0], [w, h], [0, h]]), (0.9, 0.9, 0.9))
            mod = spec.marker_mm / MODULES
            for mid, (x, y) in spec.markers.items():
                cells = marker_modules(mid)
                for r in range(MODULES):
                    for c in range(MODULES):
                        sq = [[x + c * mod, y + r * mod], [x + (c + 1) * mod, y + r * mod], [x + (c + 1) * mod, y + (r + 1) * mod], [x + c * mod, y + (r + 1) * mod]]
                        self._paint_mm(apply_h(T, sq), (0.9, 0.9, 0.9) if cells[r, c] else (0.05, 0.05, 0.05))
            arm, half = spec.cross_arm_mm, 0.3
            for cx, cy in spec.crosses:
                for bar in ([[cx - arm, cy - half], [cx + arm, cy - half], [cx + arm, cy + half], [cx - arm, cy + half]],
                            [[cx - half, cy - arm], [cx + half, cy - arm], [cx + half, cy + arm], [cx - half, cy + arm]]):
                    self._paint_mm(apply_h(T, bar), (0.05, 0.05, 0.05))
            self._version += 1
        return T

    def place_id1_card(self, centre_mm: tuple[float, float], angle_deg: float = 0.0, albedo=(0.45, 0.2, 0.1)) -> np.ndarray:
        """An ID-1 card (rounded corners, r = 3.18 mm); returns its bench-mm outline corners TL, TR, BR, BL."""
        w, h = ID1_MM
        r = 3.18
        ang = np.linspace(0, np.pi / 2, 8)
        outline = []
        for (cx, cy), start in (((w - r, r), -np.pi / 2), ((w - r, h - r), 0.0), ((r, h - r), np.pi / 2), ((r, r), np.pi)):
            outline += [[cx + r * math.cos(start + t), cy + r * math.sin(start + t)] for t in ang]
        rot = np.array([[math.cos(math.radians(angle_deg)), -math.sin(math.radians(angle_deg)), 0], [math.sin(math.radians(angle_deg)), math.cos(math.radians(angle_deg)), 0], [0, 0, 1]])
        T = _translate(centre_mm[0], centre_mm[1]) @ rot @ _translate(-w / 2, -h / 2)
        with self._lock:
            self._paint_mm(apply_h(T, outline), albedo)
            self._version += 1
        return apply_h(T, [[0, 0], [w, 0], [w, h], [0, h]])

    # -- display (page -> photons)

    def show(self, page_img: np.ndarray, dpr: float, t_drawn: float) -> None:
        with self._lock:
            self._n_shown += 1
            self._shown.append((t_drawn + self.photon_s, self._n_shown, page_img, dpr))
            self._shown = self._shown[-8:]

    def _visible(self, t: float) -> tuple[int, np.ndarray | None, float]:
        """(show counter, page image, dpr) of the newest page state whose photons reached the bench by ``t``."""
        out: tuple[int, np.ndarray | None, float] = (0, None, 1.0)
        for tv, n, im, d in self._shown:
            if tv <= t:
                out = (n, im, d)
        return out

    # -- camera

    def render(self, t: float) -> np.ndarray:
        """Camera image (BGR uint8) at laptop time ``t``."""
        with self._lock:
            n, page_img, dpr = self._visible(t)
            key = (self._version, n)
            if self._cache is None or self._cache[0] != key:
                self._cache = (key, self._base(page_img, dpr))
            base = self._cache[1]
            noise = self.rng.standard_normal(base.shape[:2], dtype=np.float32) * self.noise
        return np.clip(base + noise[..., None] + 0.5, 0, 255).astype(np.uint8)

    def _base(self, page_img: np.ndarray | None, dpr: float) -> np.ndarray:
        w, h = self.cam_size
        if self._albedo_cam is None or self._albedo_cam[0] != self._version:
            to_bench_px = np.array([[1 / self.k, 0, 0.5 / self.k], [0, 1 / self.k, 0.5 / self.k], [0, 0, 1]])
            warped = cv2.warpPerspective(self.albedo, self.bench_to_cam @ to_bench_px, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
            self._albedo_cam = (self._version, warped)
        light = np.zeros((h, w, 3), np.float32)
        if page_img is not None:
            to_css = np.array([[1 / dpr, 0, 0.5 / dpr], [0, 1 / dpr, 0.5 / dpr], [0, 0, 1]])
            light = cv2.warpPerspective(page_img, self.H_page_to_cam @ to_css, (w, h), flags=cv2.INTER_LINEAR, borderValue=(0, 0, 0))
        img = self._albedo_cam[1] * (self.ambient + self.gain * light)
        for poly in self.occluders:
            cv2.fillPoly(img, [np.round(poly * 16).astype(np.int32)], (90.0, 120.0, 165.0), cv2.LINE_AA, shift=4)
        return cv2.GaussianBlur(img, (0, 0), self.blur) if self.blur > 0 else img


class SimCamera:
    """``LiveCamera``-compatible camera over a :class:`SimRig`: frames on a fixed clock, rendered on demand."""

    controlled = True

    def __init__(self, rig: SimRig):
        self.rig = rig
        self._t0 = time.monotonic()
        self.closed = False

    def _frame_at(self, k: int) -> Frame:
        t = self._t0 + k / self.rig.fps
        return Frame(image=self.rig.render(t), t_mono=t, idx=k)

    def wait_frame(self, after_t: float, timeout: float) -> Frame | None:
        k = max(0, math.floor((after_t - self._t0) * self.rig.fps) + 1)
        t = self._t0 + k / self.rig.fps
        # Floating-point subtraction can round a frame boundary down to the previous index.
        while t <= after_t:
            k += 1
            t = self._t0 + k / self.rig.fps
        now = time.monotonic()
        if t > now + timeout:
            time.sleep(max(0.0, timeout))
            return None
        if t > now:
            time.sleep(t - now)
        return self._frame_at(k)

    def latest(self) -> Frame:
        return self._frame_at(math.floor((time.monotonic() - self._t0) * self.rig.fps))

    def describe(self) -> dict:
        w, h = self.rig.cam_size
        return {"device_id": "sim", "mode": f"sim {w}x{h}@{self.rig.fps:g}", "controls": {}}

    def close(self) -> None:
        self.closed = True


class SimPage:
    """The projector page: draws a C2 scene into a device-px image and shows it on the rig."""

    def __init__(self, rig: SimRig, w: int = 854, h: int = 480, dpr: float = 1.0):
        self.rig, self.w, self.h, self.dpr = rig, w, h, dpr
        self.H_cam_to_page: np.ndarray | None = None

    def hello(self) -> Hello:
        return Hello(role="projector", innerWidth=self.w, innerHeight=self.h, dpr=self.dpr, screen={"width": self.w, "height": self.h},
                     orientation="landscape-primary", fullscreen=True, secure=True, wakelock=True, ua="lightwork-sim")

    def set_calib(self, H_cam_to_page: Sequence[Sequence[float]]) -> None:
        self.H_cam_to_page = np.asarray(H_cam_to_page, dtype=np.float64)

    def draw(self, scene: SceneMsg) -> list[MarkerRect] | None:
        """Render like ``projector.js`` ``draw()``; returns the drawn marker rects in pattern mode."""
        W, H = round(self.w * self.dpr), round(self.h * self.dpr)
        img = np.zeros((H, W, 3), np.float32)
        drawn = None
        if scene.mode == "pattern":
            drawn = self._draw_markers(img, scene.pattern)
        elif scene.mode != "normal":
            img[:] = MEASURE_FILL[scene.mode]
        elif self.H_cam_to_page is not None:
            self._draw_prims(img, scene)
        self.rig.show(img, self.dpr, time.monotonic())
        return drawn

    def _draw_markers(self, img: np.ndarray, pattern: Pattern) -> list[MarkerRect]:
        out = []
        for m in pattern.markers:
            mod = max(1, _js_round(m.size * self.dpr / MODULES))
            x, y = _js_round(m.x * self.dpr), _js_round(m.y * self.dpr)
            img[max(0, y - mod) : y + 7 * mod, max(0, x - mod) : x + 7 * mod] = PATTERN_WHITE
            cells = np.kron(marker_modules(m.id), np.ones((mod, mod), np.uint8)).astype(np.float32) * PATTERN_WHITE
            img[y : y + 6 * mod, x : x + 6 * mod] = cells[..., None]
            out.append(MarkerRect(id=m.id, x=x / self.dpr, y=y / self.dpr, size=6 * mod / self.dpr))
        return out

    def _draw_prims(self, img: np.ndarray, scene: SceneMsg) -> None:
        unit = style.unit_css_px(self.w, self.h)
        line = style.LINE_PROJ_PX * unit * self.dpr  # device px

        def dev(pts) -> np.ndarray:  # camera px -> cv2 fixed-point device coords (pixel centres at i + 0.5)
            p = apply_h(self.H_cam_to_page, pts) * self.dpr - 0.5
            return np.round(p * 16).astype(np.int32)

        for p in scene.prims:
            colour = tuple(c / 255 for c in PALETTE_BGR[p.role])
            if isinstance(p, Ring):
                a = np.arange(style.RING_SEGMENTS) * 2 * np.pi / style.RING_SEGMENTS
                circle = np.stack([p.c[0] + p.r * np.cos(a), p.c[1] + p.r * np.sin(a)], axis=1)
                cv2.polylines(img, [dev(circle)], True, colour, max(1, round(line)), cv2.LINE_AA, shift=4)
            elif isinstance(p, Dot):
                c = dev([p.c])[0]
                cv2.circle(img, (int(c[0]), int(c[1])), round(style.DOT_RADIUS_LINES * line * 16), colour, -1, cv2.LINE_AA, shift=4)
            else:
                raise ValueError(f"the simulated page draws ring and dot prims only, not {p.kind}")


class SimLink:
    """In-process stand-in for ``RenderHub`` driving one :class:`SimPage` (PageLink plus the hub calls the
    calibration flows use): ``show_mode`` draws, waits ``ack_s`` and returns the ACK time."""

    def __init__(self, page: SimPage, ack_s: float = 0.03):
        self.page, self.ack_s = page, ack_s
        self._scene = SceneMsg(seq=0, mode="normal", prims=[])
        self._mode: SceneMsg | None = None
        self._ack: Ack | None = None
        self._seq = 0
        self.modes: list[str] = []  # every show_mode call, in order

    def hello(self, role: str) -> Hello | None:
        return self.page.hello() if role == "projector" else None

    def set_calib(self, calib: Calib) -> None:
        self.page.set_calib(calib.H_cam_to_page)

    def publish_scene(self, scene: SceneMsg) -> None:
        self._scene = scene
        if self._mode is None:
            self.page.draw(scene)

    def show_mode(self, mode: str, pattern: Pattern | None = None, timeout: float = 1.0) -> float | None:
        self.modes.append(mode)
        self._mode = None if mode == "normal" else SceneMsg(seq=0, mode=mode, prims=[], pattern=pattern)
        drawn = self.page.draw(self._mode or self._scene)
        time.sleep(self.ack_s)
        self._seq += 1
        self._ack = Ack(seq=self._seq, drawn=drawn)
        return time.monotonic()

    def last_ack(self) -> Ack | None:
        return self._ack
