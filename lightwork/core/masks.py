"""SAM masks in camera px.

``models.parse`` returns C9 masks in the px of the image SAM received: the keyframe crop, sent at crop resolution.
The reducer adds the crop origin. Each mask's RLE is COCO RLE (column-major) covering exactly its half-open box.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from ..geometry import Regions


@dataclass(frozen=True)
class CamMask:
    object_id: int
    box: tuple[int, int, int, int]  # half-open [x0, y0, x1, y1), camera px
    area: int  # mask pixels
    centroid: tuple[float, float]  # camera px
    bits: np.ndarray  # bool (y1 - y0, x1 - x0), box-local

    @property
    def radius(self) -> float:
        x0, y0, x1, y1 = self.box
        return max(x1 - x0, y1 - y0) / 2


def _counts_from_string(s: str) -> list[int]:
    """COCO compressed RLE counts (6-bit chunks offset by 48, deltas against the count two back)."""
    counts: list[int] = []
    p = 0
    while p < len(s):
        x, k, more = 0, 0, True
        while more:
            if p >= len(s):
                raise ValueError("truncated RLE counts")
            c = ord(s[p]) - 48
            x |= (c & 0x1F) << (5 * k)
            more = bool(c & 0x20)
            p += 1
            k += 1
            if not more and c & 0x10:
                x |= -1 << (5 * k)
        if len(counts) > 2:
            x += counts[-2]
        counts.append(x)
    return counts


def rle_decode(rle: Mapping[str, Any]) -> np.ndarray:
    """COCO RLE ``{"size": [h, w], "counts": str | list[int]}`` -> bool array (h, w)."""
    h, w = (int(v) for v in rle["size"])
    counts = rle["counts"]
    counts = _counts_from_string(counts) if isinstance(counts, str) else [int(c) for c in counts]
    if any(c < 0 for c in counts) or sum(counts) != h * w:
        raise ValueError(f"RLE counts do not cover a {h}x{w} mask")
    flat = np.zeros(h * w, dtype=bool)
    pos = 0
    for i, c in enumerate(counts):
        if i % 2:
            flat[pos : pos + c] = True
        pos += c
    return flat.reshape(w, h).T


def to_camera(mask: Mapping[str, Any], crop_xywh: Sequence[int]) -> CamMask:
    """C9 mask in sent-image px -> :class:`CamMask` in camera px."""
    ox, oy = int(crop_xywh[0]), int(crop_xywh[1])
    x0, y0, x1, y1 = (int(v) for v in mask["box"])
    bits = rle_decode(mask["rle"])
    if bits.shape != (y1 - y0, x1 - x0):
        raise ValueError(f"mask RLE {bits.shape} does not cover its box {[x0, y0, x1, y1]}")
    cx, cy = mask["centroid"]
    return CamMask(
        object_id=int(mask["object_id"]),
        box=(x0 + ox, y0 + oy, x1 + ox, y1 + oy),
        area=int(bits.sum()),
        centroid=(float(cx) + ox, float(cy) + oy),
        bits=bits,
    )


def overlap(mask: CamMask, regions: Regions, ref: str) -> float:
    """Share of the mask's pixels inside region ``ref``."""
    if mask.area == 0:
        return 0.0
    x0, y0, x1, y1 = mask.box
    inside = regions.mask(ref, (y1 - y0, x1 - x0), origin=(x0, y0)) > 0
    return int((mask.bits & inside).sum()) / mask.area
