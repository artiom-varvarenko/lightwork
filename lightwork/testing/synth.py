"""Deterministic synthetic bench images for tests and fake-camera runs.

All functions return BGR uint8 images (OpenCV order) and are pure functions of their arguments.

- :func:`well_plate`: a white board with a plate whose wells carry a blue dye of a given optical density
  (top-down OD on the red channel, linearised sRGB, as measured by ``lightwork.measure``).
- :func:`kit_scene`: coloured shapes standing in for kit items, optionally inside a case outline.
- :func:`add_hand`: a skin-toned blob with a finger, for motion/hand-region tests.
- :func:`warp_to_camera`: place a page-space image into a camera view through a homography.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

WHITE = (235, 235, 235)
SKIN = (120, 160, 210)  # BGR


def srgb_to_linear(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64) / 255.0
    return np.where(v <= 0.04045, v / 12.92, ((v + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(v: np.ndarray) -> np.ndarray:
    v = np.clip(np.asarray(v, dtype=np.float64), 0, 1)
    s = np.where(v <= 0.0031308, v * 12.92, 1.055 * np.power(v, 1 / 2.4) - 0.055)
    return np.clip(np.round(s * 255.0), 0, 255).astype(np.uint8)


# Relative absorbance of a blue dye per channel (B, G, R): mostly absorbs red, some green.
BLUE_DYE = (0.08, 0.45, 1.0)
# Tea/coffee: mostly absorbs blue.
BROWN_DYE = (1.0, 0.55, 0.25)


def dyed_colour(od: float, base_bgr=WHITE, dye=BLUE_DYE) -> tuple[int, int, int]:
    """Colour of a well with ``od`` on the dye's main channel (Beer-Lambert in linear light)."""
    lin = srgb_to_linear(np.array(base_bgr, dtype=np.float64))
    k = np.array(dye, dtype=np.float64)
    out = lin * np.power(10.0, -od * k)
    return tuple(int(x) for x in linear_to_srgb(out))


@dataclass(frozen=True)
class PlateGeometry:
    corners: np.ndarray  # 4x2 float, TL TR BR BL of the well-centre grid (A1, A<cols>, <last row><cols>, <last>1)
    radius: float


def well_plate(
    ods: dict[str, float],
    rows: int = 8,
    cols: int = 12,
    size: tuple[int, int] = (1280, 720),
    pitch: float = 60.0,
    origin: tuple[float, float] = (230.0, 140.0),
    dye=BLUE_DYE,
    noise_sd: float = 0.0,
    seed: int = 0,
) -> tuple[np.ndarray, PlateGeometry]:
    """Plate on a white board; ``ods`` maps "A1"-style names to OD (missing cells are clear)."""
    w, h = size
    img = np.full((h, w, 3), WHITE, np.uint8)
    ox, oy = origin
    r = pitch * 0.38
    x0, y0 = ox - pitch * 0.7, oy - pitch * 0.7
    x1, y1 = ox + pitch * (cols - 1) + pitch * 0.7, oy + pitch * (rows - 1) + pitch * 0.7
    cv2.rectangle(img, (int(x0), int(y0)), (int(x1), int(y1)), (225, 228, 228), -1, cv2.LINE_AA)
    cv2.rectangle(img, (int(x0), int(y0)), (int(x1), int(y1)), (150, 150, 150), 2, cv2.LINE_AA)
    clear = (238, 240, 240)
    for ri in range(rows):
        for ci in range(cols):
            name = f"{chr(65 + ri)}{ci + 1}"
            c = (ox + ci * pitch, oy + ri * pitch)
            colour = dyed_colour(ods[name], clear, dye) if name in ods else clear
            cv2.circle(img, (round(c[0]), round(c[1])), round(r), colour, -1, cv2.LINE_AA)
            cv2.circle(img, (round(c[0]), round(c[1])), round(r), (170, 170, 170), 1, cv2.LINE_AA)
    if noise_sd > 0:
        rng = np.random.default_rng(seed)
        img = np.clip(img.astype(np.float64) + rng.normal(0, noise_sd, img.shape), 0, 255).astype(np.uint8)
    corners = np.array(
        [[ox, oy], [ox + (cols - 1) * pitch, oy], [ox + (cols - 1) * pitch, oy + (rows - 1) * pitch], [ox, oy + (rows - 1) * pitch]],
        dtype=np.float64,
    )
    return img, PlateGeometry(corners=corners, radius=r)


@dataclass(frozen=True)
class Item:
    name: str
    shape: str  # rect | circle | bar
    colour: tuple[int, int, int]  # BGR
    xy: tuple[int, int]  # centre
    size: tuple[int, int]  # w, h (circle uses w as diameter)


def kit_scene(
    items: list[Item],
    size: tuple[int, int] = (1280, 720),
    case_rect: tuple[int, int, int, int] | None = (700, 120, 520, 480),
    slots: dict[str, tuple[int, int, int, int]] | None = None,
) -> np.ndarray:
    """White board, optional open case (light insert) with slot outlines, coloured items."""
    w, h = size
    img = np.full((h, w, 3), WHITE, np.uint8)
    if case_rect:
        x, y, cw, ch = case_rect
        cv2.rectangle(img, (x, y), (x + cw, y + ch), (60, 60, 60), 6)
        cv2.rectangle(img, (x + 6, y + 6), (x + cw - 6, y + ch - 6), (205, 210, 210), -1)
    for sx, sy, sw, sh in (slots or {}).values():
        cv2.rectangle(img, (sx, sy), (sx + sw, sy + sh), (160, 160, 160), 2)
    for it in items:
        cx, cy = it.xy
        iw, ih = it.size
        if it.shape == "circle":
            cv2.circle(img, (cx, cy), iw // 2, it.colour, -1, cv2.LINE_AA)
        else:
            cv2.rectangle(img, (cx - iw // 2, cy - ih // 2), (cx + iw // 2, cy + ih // 2), it.colour, -1)
    return img


def add_hand(img: np.ndarray, tip: tuple[int, int], angle_deg: float = 90.0, scale: float = 1.0) -> np.ndarray:
    """Skin blob (palm) with one extended finger whose tip is at ``tip``. Returns a new image."""
    out = img.copy()
    a = np.deg2rad(angle_deg)
    d = np.array([np.cos(a), np.sin(a)])
    finger_len, palm_r = 90 * scale, 55 * scale
    tip_v = np.array(tip, dtype=np.float64)
    base = tip_v + d * finger_len
    palm = base + d * palm_r
    cv2.line(out, tuple(np.round(tip_v).astype(int)), tuple(np.round(base).astype(int)), SKIN, int(22 * scale), cv2.LINE_AA)
    cv2.circle(out, tuple(np.round(tip_v).astype(int)), int(11 * scale), SKIN, -1, cv2.LINE_AA)
    cv2.ellipse(out, tuple(np.round(palm).astype(int)), (int(palm_r), int(palm_r * 1.2)), angle_deg, 0, 360, SKIN, -1, cv2.LINE_AA)
    return out


def warp_to_camera(page_img: np.ndarray, H_page_to_cam: np.ndarray, cam_size=(1920, 1080), background=(20, 20, 20)) -> np.ndarray:
    w, h = cam_size
    return cv2.warpPerspective(page_img, H_page_to_cam, (w, h), flags=cv2.INTER_LINEAR, borderValue=background)


def encode_jpeg(img: np.ndarray, quality: int = 95) -> bytes:
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise RuntimeError("JPEG encode failed")
    return buf.tobytes()
