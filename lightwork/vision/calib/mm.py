"""mm scale of the bench (C7 ``mm``; rig brief §4.3-§4.4). Camera px stay canonical (D5); mm only scales reports
(residuals, rig:C1 accuracy) and tolerances. Sources, best first:

- ``card``: the printed sheet (:data:`CARD`, ArUco ids 40-43 at known positions), found in a lit frame;
- ``plate``: a well plate's corner-cell centres and pitch (SBS 96-well: 9 mm), e.g. from the layout session;
- ``idcard``: an ISO/IEC 7810 ID-1 card (85.60 x 53.98 mm) found in a lit frame;
- ``nominal``: the Tank's throw (44.2 cm wide at 0.5 m) at an assumed mount height, through ``H_cam_to_page``.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field

import cv2
import numpy as np

from ...contracts.layout import Calibration, MmScale
from ...geometry import apply_h
from .pattern import MODULES, marker_modules
from .solve import CalibrationError, detect_markers, gray

TANK_WIDTH_MM_PER_MM_THROW = 442.0 / 500.0  # projected image width / mount height (rig brief §1)
ID1_MM = (85.60, 53.98)
MIN_CARD_MARKERS = 3


@dataclass(frozen=True)
class CardSpec:
    """A4 landscape calibration sheet: mm scale, camera-bump reference and the rig:C1 accuracy targets."""

    sheet_mm: tuple[float, float] = (297.0, 210.0)
    marker_mm: float = 40.0
    markers: dict[int, tuple[float, float]] = field(
        default_factory=lambda: {40: (12.0, 12.0), 41: (245.0, 12.0), 42: (245.0, 158.0), 43: (12.0, 158.0)}
    )  # id -> top-left of the outer border, mm from the sheet's top-left
    crosses: tuple[tuple[float, float], ...] = tuple((x, y) for y in (25.0, 105.0, 185.0) for x in (68.5, 148.5, 228.5))
    cross_arm_mm: float = 6.0
    ruler: tuple[float, float, float] = (98.5, 198.5, 201.0)  # x0, x1, y of the 100 mm print-scale bar

    def marker_corners_mm(self, marker_id: int) -> np.ndarray:
        x, y = self.markers[marker_id]
        s = self.marker_mm
        return np.array([[x, y], [x + s, y], [x + s, y + s], [x, y + s]], dtype=np.float64)


CARD = CardSpec()


def card_svg(spec: CardSpec = CARD) -> str:
    """Printable SVG of the card in true mm. Print at 100 % (no fit-to-page); the bar must measure 100 mm."""
    w, h = spec.sheet_mm
    mod = spec.marker_mm / MODULES
    out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}mm" height="{h}mm" viewBox="0 0 {w} {h}">',
        f'<rect width="{w}" height="{h}" fill="#fff"/>',
    ]
    for mid, (x, y) in sorted(spec.markers.items()):
        out.append(f'<rect x="{x}" y="{y}" width="{spec.marker_mm}" height="{spec.marker_mm}" fill="#000"/>')
        cells = marker_modules(mid)
        for r, c in zip(*np.nonzero(cells)):
            out.append(f'<rect x="{x + c * mod:.4f}" y="{y + r * mod:.4f}" width="{mod:.4f}" height="{mod:.4f}" fill="#fff"/>')
    a = spec.cross_arm_mm
    for x, y in spec.crosses:
        out.append(f'<path d="M{x - a} {y}H{x + a}M{x} {y - a}V{y + a}" stroke="#000" stroke-width="0.6" fill="none"/>')
    x0, x1, y = spec.ruler
    ticks = "".join(f"M{x0 + k * 10} {y - (2.5 if k % 5 == 0 else 1.5)}V{y}" for k in range(11))
    out.append(f'<path d="M{x0} {y}H{x1}{ticks}" stroke="#000" stroke-width="0.4" fill="none"/>')
    out.append(
        f'<text x="{x0}" y="{y + 5}" font-family="sans-serif" font-size="3.2">Lightwork calibration card (DICT_4X4_50 ids '
        f"40-43). Print at 100 %: this bar is 100 mm.</text>"
    )
    out.append("</svg>")
    return "\n".join(out) + "\n"


# ---------------------------------------------------------------------------------------------------- sources


@dataclass(frozen=True)
class CardFit:
    scale: MmScale
    rms_mm: float
    n_markers: int


def fit_card(image: np.ndarray, spec: CardSpec = CARD) -> CardFit | None:
    """``H_cam_to_mm`` from the card's markers in a lit frame; None when fewer than ``MIN_CARD_MARKERS`` are seen."""
    found = detect_markers(image, set(spec.markers))
    if len(found) < MIN_CARD_MARKERS:
        return None
    ids = sorted(found)
    cam = np.concatenate([found[i] for i in ids])
    mm = np.concatenate([spec.marker_corners_mm(i) for i in ids])
    H, _ = cv2.findHomography(cam, mm, 0)
    if H is None:
        raise CalibrationError("card homography solve failed")
    H = H / H[2, 2]
    rms = float(np.sqrt(np.mean(np.sum((apply_h(H, cam) - mm) ** 2, axis=1))))
    return CardFit(MmScale(source="card", H_cam_to_mm=H.tolist()), round(rms, 3), len(ids))


def quad_scale(corners_cam_px: Sequence[Sequence[float]], size_mm: tuple[float, float], source: str) -> MmScale:
    """A rectangle of known size seen at TL, TR, BR, BL camera px; ``size_mm`` = (TL->TR, TR->BR) lengths."""
    w, h = size_mm
    src = np.asarray(corners_cam_px, dtype=np.float32).reshape(4, 2)
    dst = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)
    return MmScale(source=source, H_cam_to_mm=cv2.getPerspectiveTransform(src, dst).astype(np.float64).tolist())


def grid_scale(corners_cam_px: Sequence[Sequence[float]], rows: int, cols: int, pitch_mm: float) -> MmScale:
    """A plate from its corner-cell centres (C4 grid convention: TL, TR, BR, BL) and its well pitch."""
    if rows < 2 or cols < 2:
        raise ValueError("a plate scale needs at least 2 rows and 2 columns")
    return quad_scale(corners_cam_px, ((cols - 1) * pitch_mm, (rows - 1) * pitch_mm), "plate")


def nominal_scale(H_cam_to_page: np.ndarray, page_w: float, mount_mm: float) -> MmScale:
    """mm from the throw: the page width spans ``0.884 x mount height``. Labelled nominal, never measured."""
    s = TANK_WIDTH_MM_PER_MM_THROW * mount_mm / page_w
    return MmScale(source="nominal", H_cam_to_mm=(np.diag([s, s, 1.0]) @ np.asarray(H_cam_to_page)).tolist())


def mm_per_page_px(calib: Calibration) -> float | None:
    """Local mm per page CSS px at the page centre (scales page-px residuals to mm)."""
    if calib.mm.H_cam_to_mm is None:
        return None
    to_mm = np.asarray(calib.mm.H_cam_to_mm) @ np.linalg.inv(np.asarray(calib.H_cam_to_page))
    c = np.array([calib.page.w / 2, calib.page.h / 2])
    p = apply_h(to_mm, [c, c + (1, 0), c + (0, 1)])
    u, v = p[1] - p[0], p[2] - p[0]
    return float(math.sqrt(abs(u[0] * v[1] - u[1] * v[0])))


# ---------------------------------------------------------------------------------------------------- ID-1 card

MIN_CARD_AREA_FRAC = 0.002
ASPECT_TOL = 0.08


def find_id1_card(image: np.ndarray) -> tuple[np.ndarray, tuple[float, float]] | None:
    """The largest card-shaped quad (aspect 1.586 +-8 %) in a lit frame: corners TL, TR, BR, BL (camera px, edge
    lines fitted at sub-pixel precision, so rounded card corners do not matter) and its (TL->TR, TR->BR) size in mm.
    The card must contrast with the board."""
    g = cv2.GaussianBlur(gray(image), (0, 0), 1.2)
    edges = cv2.morphologyEx(cv2.Canny(g, 30, 90), cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    ratio = ID1_MM[0] / ID1_MM[1]
    best, best_area = None, MIN_CARD_AREA_FRAC * g.size
    for c in contours:
        q = cv2.approxPolyDP(c, 0.02 * cv2.arcLength(c, True), True)
        if len(q) != 4 or not cv2.isContourConvex(q):
            continue
        area = cv2.contourArea(q)
        quad = _clockwise(q.reshape(4, 2).astype(np.float64))
        sides = np.linalg.norm(quad - np.roll(quad, -1, axis=0), axis=1)
        a, b = (sides[0] + sides[2]) / 2, (sides[1] + sides[3]) / 2
        if area > best_area and abs(max(a, b) / min(a, b) / ratio - 1) <= ASPECT_TOL:
            best, best_area = quad, area
    if best is None:
        return None
    corners = _refine_quad(g, best)
    top, right = np.linalg.norm(corners[1] - corners[0]), np.linalg.norm(corners[2] - corners[1])
    return corners, (ID1_MM if top >= right else ID1_MM[::-1])


def _clockwise(quad: np.ndarray) -> np.ndarray:
    """TL, TR, BR, BL: clockwise on screen (y down), starting at the corner nearest the image origin."""
    c = quad.mean(axis=0)
    q = quad[np.argsort(np.arctan2(quad[:, 1] - c[1], quad[:, 0] - c[0]))]
    return np.roll(q, -int(np.argmin(q.sum(axis=1))), axis=0)


def _refine_quad(g: np.ndarray, quad: np.ndarray, reach: float = 6.0, n: int = 24) -> np.ndarray:
    """Fit each side to the strongest-gradient points along its normals (middle 70 % of the side), intersect."""
    lines = []
    offs = np.arange(-reach, reach + 0.25, 0.25)
    for k in range(4):
        a, b = quad[k], quad[(k + 1) % 4]
        d = (b - a) / np.linalg.norm(b - a)
        nrm = np.array([-d[1], d[0]])
        base = a + np.linspace(0.15, 0.85, n)[:, None] * (b - a)
        pts = base[:, None, :] + offs[None, :, None] * nrm
        prof = cv2.remap(g.astype(np.float32), pts[..., 0].astype(np.float32), pts[..., 1].astype(np.float32), cv2.INTER_LINEAR)
        grad = np.abs(np.diff(prof, axis=1))
        i = np.clip(np.argmax(grad, axis=1), 1, grad.shape[1] - 2)
        rows = np.arange(n)
        y0, y1, y2 = grad[rows, i - 1], grad[rows, i], grad[rows, i + 1]
        den = y0 - 2 * y1 + y2
        sub = np.where(np.abs(den) > 1e-9, 0.5 * (y0 - y2) / np.where(den == 0, 1, den), 0.0)
        s = offs[0] + (i + 0.5 + sub) * (offs[1] - offs[0])  # diff sample k sits between offsets k and k+1
        edge = base + s[:, None] * nrm
        vx, vy, x0, y0_ = cv2.fitLine(edge.astype(np.float32), cv2.DIST_HUBER, 0, 0.01, 0.01).ravel()
        lines.append((np.array([x0, y0_], dtype=np.float64), np.array([vx, vy], dtype=np.float64)))
    out = []
    for k in range(4):
        (p1, v1), (p2, v2) = lines[(k - 1) % 4], lines[k]
        t = np.linalg.solve(np.column_stack([v1, -v2]), p2 - p1)
        out.append(p1 + t[0] * v1)
    return np.array(out)
