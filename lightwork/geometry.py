"""Shared region geometry in camera px (owned by the integration owner; used by gate, hands, measure, reducer).

Region references (C3): an object id, or ``<grid_object>:<cell>`` (e.g. ``plate:A6``).
Grid convention (C4): a grid object's ``corners_cam_px`` are the *centres of its four corner cells*, in order
TL (A1), TR (A<cols>), BR (<last row><cols>), BL (<last row>1). Cells come from the homography that maps grid
coordinates ``(col, row)`` to camera px (D8). A cell's region is the quad ``(col +- 0.5, row +- 0.5)``.
"""

from __future__ import annotations

import re

import cv2
import numpy as np

from .contracts.layout import Layout, LayoutObject
from .contracts.stepgraph import Grid, StepGraph

_A1 = re.compile(r"^([A-Z])(\d+)$")
_RC = re.compile(r"^r(\d+)c(\d+)$")


def cell_index(name: str, naming: str) -> tuple[int, int]:
    """Cell name -> zero-based (row, col)."""
    m = (_A1 if naming == "A1" else _RC).match(name)
    if not m:
        raise ValueError(f"bad cell name {name!r} for naming {naming}")
    if naming == "A1":
        return ord(m.group(1)) - 65, int(m.group(2)) - 1
    return int(m.group(1)) - 1, int(m.group(2)) - 1


def cell_name(row: int, col: int, naming: str) -> str:
    return f"{chr(65 + row)}{col + 1}" if naming == "A1" else f"r{row + 1}c{col + 1}"


def grid_homography(corners: np.ndarray | list, rows: int, cols: int) -> np.ndarray:
    """3x3 homography from grid coords (col, row) to camera px. Degenerate 1-row/1-col grids use an affine line."""
    dst = np.asarray(corners, dtype=np.float64).reshape(4, 2)
    if rows == 1 or cols == 1:
        # Corner cells coincide pairwise; build the transform from the two ends plus a perpendicular unit.
        a, b = (dst[0], dst[1]) if rows == 1 else (dst[0], dst[3])
        n = max(cols, rows) - 1
        step = (b - a) / max(n, 1)
        perp = np.array([-step[1], step[0]])
        if rows == 1:
            M = np.array([[step[0], perp[0], a[0]], [step[1], perp[1], a[1]], [0, 0, 1]])
        else:
            M = np.array([[-perp[0], step[0], a[0]], [-perp[1], step[1], a[1]], [0, 0, 1]])
        return M
    src = np.array([[0, 0], [cols - 1, 0], [cols - 1, rows - 1], [0, rows - 1]], dtype=np.float64)
    return cv2.getPerspectiveTransform(src.astype(np.float32), dst.astype(np.float32)).astype(np.float64)


def apply_h(H: np.ndarray, pts: np.ndarray | list) -> np.ndarray:
    p = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
    ph = np.hstack([p, np.ones((len(p), 1))]) @ H.T
    return ph[:, :2] / ph[:, 2:3]


class GridGeom:
    def __init__(self, corners, grid: Grid):
        self.grid = grid
        self.H = grid_homography(corners, grid.rows, grid.cols)

    def centre(self, name: str) -> tuple[float, float]:
        r, c = cell_index(name, self.grid.naming)
        x, y = apply_h(self.H, [[c, r]])[0]
        return float(x), float(y)

    def quad(self, name: str) -> np.ndarray:
        r, c = cell_index(name, self.grid.naming)
        return apply_h(self.H, [[c - 0.5, r - 0.5], [c + 0.5, r - 0.5], [c + 0.5, r + 0.5], [c - 0.5, r + 0.5]])

    def pitch(self, name: str) -> float:
        """Local cell pitch in px (mean of the quad's side lengths)."""
        q = self.quad(name)
        return float(np.mean(np.linalg.norm(q - np.roll(q, -1, axis=0), axis=1)))

    def names(self) -> list[str]:
        return [cell_name(r, c, self.grid.naming) for r in range(self.grid.rows) for c in range(self.grid.cols)]


class Regions:
    """Resolves region refs against a layout + step graph."""

    def __init__(self, layout: Layout, stepgraph: StepGraph):
        self.layout = layout
        self.grids: dict[str, GridGeom] = {}
        grids = {o.id: o.grid for o in stepgraph.objects if o.grid is not None}
        for oid, obj in layout.objects.items():
            if oid in grids and obj.corners_cam_px is not None:
                self.grids[oid] = GridGeom(obj.corners_cam_px, grids[oid])

    def polygon(self, ref: str) -> np.ndarray:
        if ":" in ref:
            oid, cell = ref.split(":", 1)
            return self.grids[oid].quad(cell)
        obj: LayoutObject = self.layout.objects[ref]
        pts = obj.corners_cam_px if obj.corners_cam_px is not None else obj.polygon_cam_px
        if obj.kind == "grid" and ref in self.grids:
            g = self.grids[ref]
            n = g.grid
            last = [cell_name(0, 0, n.naming), cell_name(0, n.cols - 1, n.naming), cell_name(n.rows - 1, n.cols - 1, n.naming), cell_name(n.rows - 1, 0, n.naming)]
            return np.array([g.quad(last[0])[0], g.quad(last[1])[1], g.quad(last[2])[2], g.quad(last[3])[3]])
        return np.asarray(pts, dtype=np.float64)

    def centre(self, ref: str) -> tuple[float, float]:
        if ":" in ref:
            oid, cell = ref.split(":", 1)
            return self.grids[oid].centre(cell)
        p = self.polygon(ref)
        c = p.mean(axis=0)
        return float(c[0]), float(c[1])

    def all_refs(self) -> list[str]:
        refs = list(self.layout.objects)
        for oid, g in self.grids.items():
            refs.extend(f"{oid}:{n}" for n in g.names())
        return refs

    def contains(self, ref: str, xy) -> bool:
        poly = self.polygon(ref).astype(np.float32)
        return cv2.pointPolygonTest(poly, (float(xy[0]), float(xy[1])), False) >= 0

    def mask(self, ref: str, shape: tuple[int, int], origin: tuple[int, int] = (0, 0)) -> np.ndarray:
        """uint8 mask (h, w) of the region in an image whose top-left is ``origin`` in camera px."""
        m = np.zeros(shape[:2], np.uint8)
        poly = np.round(self.polygon(ref) - np.asarray(origin, dtype=np.float64)).astype(np.int32)
        cv2.fillPoly(m, [poly], 255)
        return m
