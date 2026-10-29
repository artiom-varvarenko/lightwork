from __future__ import annotations

import pytest

from lightwork.contracts.layout import Layout
from lightwork.contracts.stepgraph import StepGraph
from lightwork.geometry import Regions, cell_index, cell_name
from lightwork.testing.synth import well_plate


def _setup(rows=8, cols=12):
    _, geom = well_plate({}, rows=rows, cols=cols)
    sg = StepGraph.model_validate({
        "id": "p", "title": "p", "source": {"kind": "text", "ref": "x", "licence": "x", "attribution": "x"},
        "objects": [{"id": "plate", "noun": "well plate", "kind": "grid", "grid": {"rows": rows, "cols": cols, "naming": "A1"}}],
        "steps": [{"id": "s", "say": "x", "checks": [{"id": "c", "kind": "confirm"}]}],
    })
    lay = Layout.model_validate({"pack": "p", "created": "x", "work_area_xywh": [0, 0, 1280, 720],
        "objects": {"plate": {"kind": "grid", "corners_cam_px": geom.corners.tolist(), "sam": {"mode": "noun", "noun": "well plate"}}}})
    return Regions(lay, sg)


def test_cell_names_round_trip():
    assert cell_index("H12", "A1") == (7, 11) and cell_name(7, 11, "A1") == "H12"
    assert cell_index("r2c3", "r1c1") == (1, 2) and cell_name(1, 2, "r1c1") == "r2c3"


@pytest.mark.parametrize("rows,cols", [(8, 12), (1, 12), (3, 1)])
def test_cell_centres_match_synthetic_plate(rows, cols):
    reg = _setup(rows, cols)
    g = reg.grids["plate"]
    # synth: origin (230,140), pitch 60
    for name in (cell_name(0, 0, "A1"), cell_name(rows - 1, cols - 1, "A1")):
        r, c = cell_index(name, "A1")
        assert g.centre(name) == pytest.approx((230 + 60 * c, 140 + 60 * r), abs=1e-6)
    assert g.pitch(cell_name(0, 0, "A1")) == pytest.approx(60, abs=1e-6)
    assert reg.contains(f"plate:{cell_name(0, 0, 'A1')}", (230 + 25, 140 - 25))
    assert not reg.contains(f"plate:{cell_name(0, 0, 'A1')}", (230 + 35, 140))
    assert reg.contains("plate", (230 - 25, 140 - 25))
