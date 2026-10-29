from __future__ import annotations

import math

import numpy as np
import pytest

from lightwork.contracts.stepgraph import Grid
from lightwork.geometry import GridGeom
from lightwork.measure.grid import cell_rois, disc_indices, neighbours, roi_radius
from lightwork.testing.synth import well_plate


def test_roi_radius_is_half_the_well_radius():
    # synth plate: well radius 0.38 x pitch
    _, g = well_plate({}, pitch=60.0)
    assert roi_radius(60.0) == pytest.approx(0.5 * g.radius)
    with pytest.raises(ValueError):
        roi_radius(0)


def test_cell_rois_uses_grid_centres_and_smallest_pitch():
    # perspective: the right side is compressed, so its pitch sets the radius
    corners = np.array([[100, 100], [600, 120], [600, 380], [100, 400]], dtype=np.float64)
    geom = GridGeom(corners, Grid(rows=4, cols=6, naming="A1"))
    centres, r = cell_rois(geom, ["A1", "D6"])
    assert centres == {"A1": pytest.approx((100, 100)), "D6": pytest.approx((600, 380))}
    assert r == pytest.approx(roi_radius(min(geom.pitch("A1"), geom.pitch("D6"))))
    all_centres, _ = cell_rois(geom)
    assert len(all_centres) == 24
    with pytest.raises(ValueError):
        cell_rois(geom, [])


def test_disc_indices_area_and_border():
    ys, xs = disc_indices((200, 200), (100.0, 100.0), 10.0)
    assert abs(len(ys) - math.pi * 100) < 15
    assert ((xs - 100) ** 2 + (ys - 100) ** 2 <= 100).all()
    ys2, _ = disc_indices((200, 200), (0.0, 100.0), 10.0)
    assert 0 < len(ys2) < len(ys)
    ys3, _ = disc_indices((200, 200), (-50.0, -50.0), 10.0)
    assert len(ys3) == 0


def test_neighbours():
    assert neighbours("A6", 8, 12, "A1", reach=1) == ["A5", "A7", "B5", "B6", "B7"]
    assert neighbours("A6", 1, 12, "A1") == ["A4", "A5", "A7", "A8"]
    assert neighbours("r1c1", 2, 2, "r1c1", reach=1) == ["r1c2", "r2c1", "r2c2"]
