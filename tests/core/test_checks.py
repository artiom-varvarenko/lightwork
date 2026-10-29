from dataclasses import replace

import numpy as np
import pytest

from lightwork.contracts.layout import Layout
from lightwork.contracts.stepgraph import Check, Grid, Step, StepGraph
from lightwork.core import checks as ck
from lightwork.core.graph import validate
from lightwork.core.masks import rle_decode, to_camera
from lightwork.core.params import Params
from lightwork.core.readings import CellRead, Readings
from lightwork.core.scenarios import SOURCE, build
from lightwork.geometry import Regions
from lightwork.models.parse import coco_rle_counts
from lightwork.testing.synth import encode_jpeg, well_plate

P = Params()
G = Grid(rows=1, cols=12, naming="A1")


def reading(od, lab=(80., 0., 0.)):
    return CellRead(od, lab, 100, .001)


def mask(box, *, crop=(0, 0, 100, 100), bits=None):
    x0, y0, x1, y1 = box
    bits = np.ones((y1-y0, x1-x0), dtype=bool) if bits is None else bits
    ys, xs = np.nonzero(bits)
    return to_camera({"object_id": 1, "box": box, "centroid": [x0+xs.mean(), y0+ys.mean()],
                      "rle": {"size": list(bits.shape), "counts": coco_rle_counts(bits)}}, crop)


@pytest.fixture
def regions():
    graph = StepGraph.model_validate({"id": "checks", "title": "Checks", "source": SOURCE,
        "objects": [{"id": "slot", "kind": "zone", "noun": "slot"}], "steps": []})
    layout = Layout.model_validate({"pack": "checks", "created": "synthetic", "work_area_xywh": [0, 0, 100, 100],
        "objects": {"slot": {"kind": "zone", "corners_cam_px": [[0, 0], [49, 0], [49, 99], [0, 99]],
                              "sam": {"mode": "noun", "noun": "slot"}}}})
    return Regions(layout, graph)


@pytest.mark.parametrize("kind,want", [("present", "pass"), ("absent", "fail")])
def test_mask_overlap_uses_pixels_not_box_or_centroid(regions, kind, want):
    c = Check(id="c", kind=kind, object="item", region="slot")
    m = mask((0, 10, 100, 20))  # exactly half the filled mask lies within x<=49
    f = getattr(ck, kind)
    assert f(c, "item", [m], regions, P).status == want
    tighter = replace(P, present_min_overlap=.51)
    assert f(c, "item", [m], regions, tighter).status != want
    bits = np.zeros((10, 100), dtype=bool)
    bits[:, :10] = True
    m = mask((0, 10, 100, 20), bits=bits)
    assert ck.present(c, "item", [m], regions, tighter).status == "pass"


def test_absence_and_movement_require_actual_before_evidence(regions):
    inside, outside = mask((5, 5, 15, 15)), mask((65, 5, 75, 15))
    c = Check(id="m", kind="moved_to", object="item", region="slot")
    assert ck.moved_to(c, "item", [outside], [inside], regions, P).status == "pass"
    assert ck.moved_to(c, "item", [inside], [inside], regions, P).status == "fail"
    assert ck.moved_to(c, "item", [], [inside], regions, P).status == "inconclusive"
    assert ck.moved_to(c, "item", [outside], [outside], regions, P).status == "fail"
    assert ck.absent(c, "item", [outside], regions, P).status == "pass"
    assert ck.present(c, "item", [], regions, P).status == "fail"


def test_count_cross_checks_spark_points_in_region(regions):
    c = Check(id="n", kind="count", object="item", region="slot", n=2)
    ms = [mask((5, 5, 15, 15)), mask((25, 5, 35, 15)), mask((65, 5, 75, 15))]
    assert ck.count(c, "item", ms, [(10, 10), (30, 10), (70, 10)], regions, P).status == "pass"
    assert ck.count(c, "item", ms, [(10, 10)], regions, P).status == "inconclusive"
    assert ck.count(c, "item", ms[:1], [(10, 10)], regions, P).status == "fail"


@pytest.mark.parametrize("tracked,motion,tips,status", [
    (True, {"plate:A6"}, {"plate:A6"}, "pass"),
    (True, {"plate:A6"}, {"plate:A7", "plate"}, "pending"),
    (True, {"plate:A7"}, {"plate:A6"}, "pending"),
    (True, {"plate:A6"}, set(), "pending"),
    (False, {"plate:A6"}, set(), "pass"),
])
def test_activity_at_requires_same_region(tracked, motion, tips, status):
    c = Check(id="a", kind="activity_at", region="plate:A6")
    out = ck.activity_at(c, motion, tips, tracked)
    assert out.status == status
    if not tracked:
        assert "motion-only" in out.reason
    if "plate:A7" in motion:
        assert "suspect" in out.reason


def test_colour_directions_ratio_and_unreadable():
    def check(direction, **kw):
        return Check(id="colour", kind="colour_change", object="plate", cells=["A2"], direction=direction, **kw)
    step = Step.model_validate({"id": "transfer", "say": "Transfer.", "arrow": {"from": "plate:A1", "to": "plate:A2"},
                               "checks": []})
    before = {"plate:A1": reading(.8), "plate:A2": reading(0.)}
    after = {"plate:A1": reading(.4), "plate:A2": reading(.4, (50., 10., 0.))}
    assert ck.colour_change(check("darker", ratio=1.), step, after, before, P).status == "pass"
    wrong_ratio = {**after, "plate:A2": reading(.7)}
    assert ck.colour_change(check("darker", ratio=1.), step, wrong_ratio, before, P).status == "fail"
    assert ck.colour_change(check("none"), step, after, before, P).status == "fail"
    assert ck.colour_change(check("hue", min_delta=4.), step, after, before, P).status == "pass"
    assert ck.colour_change(check("hue", min_delta=4.), step, before, before, P).status == "pending"
    assert ck.colour_change(check("darker"), step, {}, before, P).status == "inconclusive"
    weak = {"plate:A1": reading(.03), "plate:A2": reading(0.)}
    assert ck.colour_change(check("darker"), step, after, weak, P).status == "inconclusive"
    lighter = Check(id="l", kind="colour_change", object="plate", cells=["A1"], direction="lighter")
    assert ck.colour_change(lighter, step, after, before, P).status == "pass"


def test_wrong_cell_and_excluded_series():
    c = Check(id="dark", kind="colour_change", object="plate", cells=["A6"], direction="darker")
    step = Step(id="s", say="Transfer.", checks=[c])
    before = {f"plate:A{k}": reading(0.) for k in range(4, 9)}
    after = {**before, "plate:A7": reading(.04)}
    wrong = ck.wrong_cell(c, step, G, after, before, .02, P)
    assert wrong.wrong == "A7"
    assert wrong.reason == "liquid went into A7, expected A6"
    assert [m.role for m in wrong.marks] == ["error", "warn"]
    assert ck.wrong_cell(c, step, G, {**after, "plate:A6": reading(.03)}, before, .02, P) is None
    series = Check(id="series", kind="series", object="plate", cells=["A1", "A2", "A3", "A4"], ratio=.5, tol=.15)
    now = {f"plate:A{k}": reading(v) for k, v in enumerate([.4, .2, 0., .1], 1)}
    assert ck.series(series, G, now, set(), P).status == "fail"
    assert ck.series(series, G, now, {"plate:A3"}, P).status == "pass"
    assert "A3 excluded" in ck.series(series, G, now, {"plate:A3"}, P).reason


@pytest.mark.parametrize("verdict,status", [
    ({"done": "yes", "contradiction": None}, "pass"),
    ({"done": "yes", "contradiction": "wrong target"}, "fail"),
    ({"done": "no", "contradiction": None}, "fail"),
    ({"done": "unsure", "contradiction": None}, "inconclusive"),
])
def test_judge(verdict, status):
    assert ck.judge(verdict).status == status


def test_rle_round_trip_and_crop_origin():
    bits = np.array([[1, 0, 1, 1], [1, 1, 0, 1], [0, 0, 0, 1]], dtype=bool)
    rle = {"size": list(bits.shape), "counts": coco_rle_counts(bits)}
    np.testing.assert_array_equal(rle_decode(rle), bits)
    m = mask((2, 3, 6, 6), crop=(100, 200, 400, 300), bits=bits)
    assert m.box == (102, 203, 106, 206)
    assert m.area == bits.sum()
    assert m.centroid[0] > 100 and m.centroid[1] > 200
    with pytest.raises(ValueError, match="do not cover"):
        rle_decode({"size": [4, 4], "counts": [0, 10]})


def test_readings_use_encoded_crop_and_first_readable_reference(no_network):
    s = build("dilution-clean")
    readings = Readings(s.stepgraph, Regions(s.layout, s.stepgraph), P)
    x, y, w, h = s.layout.work_area_xywh
    blank, _ = well_plate({})
    dye, _ = well_plate({"A1": .8})
    for i, img in enumerate((blank, dye)):
        jpeg = encode_jpeg(img[y:y+h, x:x+w])
        from lightwork.contracts.canonical import sha256_hex
        sha = sha256_hex(jpeg)
        out = readings.measure(sha, jpeg, [x, y, w, h])
        od = out[0]["cells"]["plate:A1"]["od"]
        assert abs(od - (0. if i == 0 else .8)) < .015
        assert readings.measure(sha, jpeg, [x, y, w, h]) == []


def test_graph_uses_contract_validator_and_reducer_geometry():
    s = build("kit-clean")
    assert validate(s.stepgraph, s.layout) == []
    missing = s.layout.model_copy(update={"objects": {k: v for k, v in s.layout.objects.items() if k != "slot-r"}})
    assert any("no geometry" in e for e in validate(s.stepgraph, missing))
    graph = s.stepgraph.model_copy(deep=True)
    graph.steps[0].checks[:] = [Check(id="bad", kind="activity_at", region="slot-p")]
    assert any("no Meta-reading" in e for e in validate(graph, s.layout))
