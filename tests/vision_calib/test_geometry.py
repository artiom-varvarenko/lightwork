from __future__ import annotations

import re
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pytest

from lightwork.contracts.render import Pattern, SceneMsg
from lightwork.geometry import apply_h
from lightwork.vision.calib import CalibrationError, pattern_for_page, solve
from lightwork.vision.calib.mm import CARD, ID1_MM, card_svg, find_id1_card, fit_card, grid_scale, mm_per_page_px, nominal_scale, quad_scale
from lightwork.vision.calib.pattern import marker_corners, marker_modules
from lightwork.vision.calib.solve import detect_markers, fit


@pytest.mark.parametrize("w,h", [(854, 480), (1920, 1080), (480, 854), (800, 800)])
def test_pattern_spans_page_with_quiet_zones(w, h):
    p = pattern_for_page(w, h)
    assert [m.id for m in p.markers] == list(range(15))
    assert len({m.x for m in p.markers}) == 5 and len({m.y for m in p.markers}) == 3
    for m in p.markers:
        assert m.size % 6 == 0
        assert 0.10 * min(w, h) <= m.size <= 0.15 * min(w, h)
        mod = m.size / 6
        assert m.x >= mod and m.y >= mod
        assert m.x + m.size + mod <= w and m.y + m.size + mod <= h
    for i, a in enumerate(p.markers):
        for b in p.markers[i + 1:]:
            assert abs(a.x - b.x) >= a.size * 1.5 or abs(a.y - b.y) >= a.size * 1.5


def test_tiny_page_fails():
    with pytest.raises(ValueError, match="too small"):
        pattern_for_page(60, 40)


def test_opencv_bits_match_browser():
    js = (Path(__file__).parents[2] / "lightwork/web/js/aruco.js").read_text()
    codes = [int(x, 16) for x in re.findall(r"0x[0-9a-fA-F]+", re.search(r"CODES\s*=\s*\[([^]]+)\]", js).group(1))]
    for i, code in enumerate(codes):
        actual = sum(int(bit) << (15 - k) for k, bit in enumerate(marker_modules(i)[1:5, 1:5].ravel()))
        assert actual == code


@pytest.mark.parametrize("dpr,ambient,gain,blur,noise", [
    (1.0, 40, 200, 1.2, 2.5), (2.625, 40, 200, 1.2, 2.5),
    (1.0, 15, 100, 2.5, 3.0), (2.625, 60, 130, 2.5, 4.0),
    (1.0, 40, 200, 1.2, 8.0),
])
def test_warped_pattern_accuracy(make_rig, truth_error, dpr, ambient, gain, blur, noise):
    rig, _, page, _ = make_rig(dpr=dpr, ambient=ambient, gain=gain, blur=blur, noise=noise, seed=12)
    drawn = page.draw(SceneMsg(seq=1, mode="pattern", prims=[], pattern=pattern_for_page(page.w, page.h)))
    image = rig.render(time.monotonic() + 1)
    start = time.perf_counter()
    H, residuals = solve(image, drawn)
    elapsed = time.perf_counter() - start
    e = truth_error(H, rig)
    print(f"solve dpr={dpr} light={ambient}/{gain} blur={blur} noise={noise}: "
          f"mean={e.mean():.4f}, max={e.max():.4f} page px; rms={residuals.rms_px}; {elapsed:.4f}s")
    assert residuals.n_markers == 15
    assert residuals.n_inliers >= 56
    assert residuals.rms_px < 1.5
    assert e.mean() < 0.25 and e.max() < 0.9
    assert elapsed < 2.0


def test_use_drawn_rectangles_not_requests(make_rig, truth_error):
    rig, _, page, _ = make_rig(dpr=2.625)
    p = pattern_for_page(854, 480)
    drawn = page.draw(SceneMsg(seq=1, mode="pattern", prims=[], pattern=p))
    image = rig.render(time.monotonic() + 1)
    actual, _ = solve(image, drawn)
    requested, _ = solve(image, p.markers)
    assert truth_error(actual, rig).mean() < 0.15
    assert truth_error(requested, rig).mean() > truth_error(actual, rig).mean() * 3


def test_fewer_than_six_markers_fails(make_rig):
    rig, _, page, _ = make_rig()
    p = pattern_for_page(854, 480)
    page.draw(SceneMsg(seq=1, mode="pattern", prims=[], pattern=Pattern(markers=p.markers[:5])))
    image = rig.render(time.monotonic() + 1)
    with pytest.raises(CalibrationError, match=r"found 5 .*need >= 6"):
        solve(image, p.markers)
    with pytest.raises(CalibrationError, match="found 0"):
        solve(np.zeros_like(image), p.markers)


def test_hand_partial_occlusion_still_solves(make_rig, truth_error):
    rig, _, page, _ = make_rig()
    rig.occlude([[0, 0], [800, 0], [800, 1080], [0, 1080]])
    drawn = page.draw(SceneMsg(seq=1, mode="pattern", prims=[], pattern=pattern_for_page(854, 480)))
    H, r = solve(rig.render(time.monotonic() + 1), drawn)
    assert 6 <= r.n_markers < 15
    assert truth_error(H, rig).max() < 0.6


def test_ransac_rejects_one_wrong_marker(make_rig, truth_error):
    rig, _, _, _ = make_rig()
    rects = pattern_for_page(854, 480).markers
    corners = {m.id: apply_h(rig.H_page_to_cam, marker_corners(m)) for m in rects}
    corners[7] += [45, -30]
    sol = fit(corners, rects)
    assert sol.residuals.n_inliers == 56
    assert sol.marker_px[7] > 20
    assert truth_error(sol.H, rig).max() < 0.001


def test_duplicate_detected_marker_is_ignored(make_rig):
    rig, _, page, _ = make_rig()
    p = pattern_for_page(854, 480)
    markers = list(p.markers)
    markers[1] = markers[1].model_copy(update={"id": 0})
    page.draw(SceneMsg(seq=1, mode="pattern", prims=[], pattern=Pattern(markers=markers)))
    det = detect_markers(rig.render(time.monotonic() + 1), range(15))
    assert 0 not in det and 1 not in det and len(det) == 13


def test_printed_card_mm_scale(make_rig):
    rig, _, page, _ = make_rig()
    card_to_bench = rig.place_card((230, 155), angle_deg=1.0)
    page.draw(SceneMsg(seq=1, mode="grey", prims=[]))
    found = fit_card(rig.render(time.monotonic() + 1))
    assert found is not None and found.n_markers == 4
    camera = apply_h(rig.bench_to_cam @ card_to_bench, CARD.crosses)
    errors = np.linalg.norm(apply_h(np.asarray(found.scale.H_cam_to_mm), camera) - CARD.crosses, axis=1)
    print(f"card mm truth: mean={errors.mean():.4f}, max={errors.max():.4f}, rms={found.rms_mm}")
    assert errors.max() < 0.3


def test_printable_card_dimensions_and_ruler():
    root = ET.fromstring(card_svg())
    assert root.attrib["width"] == "297.0mm" and root.attrib["height"] == "210.0mm"
    assert "100 mm" in card_svg()
    assert len(root.findall("{http://www.w3.org/2000/svg}path")) == 10
    assert len(CARD.markers) == 4
    assert set(CARD.markers) == {40, 41, 42, 43}


def test_id1_card_refined_edges(make_rig):
    rig, _, page, _ = make_rig(texture=False)
    truth = rig.place_id1_card((380, 260), angle_deg=7)
    page.draw(SceneMsg(seq=1, mode="grey", prims=[]))
    found = find_id1_card(rig.render(time.monotonic() + 1))
    assert found is not None
    corners, size = found
    assert size == ID1_MM
    errors = np.linalg.norm(corners - apply_h(rig.bench_to_cam, truth), axis=1)
    print(f"ID-1 corners: max={errors.max():.4f} camera px")
    assert errors.max() < 2.0


def test_no_id1_false_detection_from_page(make_rig):
    rig, _, page, _ = make_rig(texture=False)
    page.draw(SceneMsg(seq=1, mode="grey", prims=[]))
    assert find_id1_card(rig.render(time.monotonic() + 1)) is None


def test_plate_and_nominal_mm(make_rig, truth_calib):
    rig, _, page, _ = make_rig()
    mm = np.array([[0, 0], [99, 0], [99, 63], [0, 63]])
    camera = apply_h(rig.bench_to_cam, mm + [280, 200])
    scale = grid_scale(camera, 8, 12, 9)
    np.testing.assert_allclose(apply_h(np.asarray(scale.H_cam_to_mm), camera), mm, atol=1e-4)
    with pytest.raises(ValueError, match="2 rows"):
        grid_scale(camera, 1, 12, 9)
    rectangular = quad_scale(camera, (99, 63), "plate")
    np.testing.assert_allclose(rectangular.H_cam_to_mm, scale.H_cam_to_mm)
    nominal = nominal_scale(rig.H_cam_to_page, 854, 500)
    calib = truth_calib(rig, page, mm=nominal)
    assert mm_per_page_px(calib) == pytest.approx(442 / 854, abs=1e-9)
