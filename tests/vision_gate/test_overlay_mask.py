from __future__ import annotations

import math

import numpy as np
import pytest

from lightwork.contracts import render
from lightwork.server import projector_style as style
from lightwork.vision.overlay_mask import OverlayMask, rasterize

# Page CSS px = 0.5 * camera px - (100, 50); page 854x480 CSS px, so 1 projector px = 1 CSS px = 2 camera px.
H = np.array([[0.5, 0.0, -100.0], [0.0, 0.5, -50.0], [0.0, 0.0, 1.0]])
PAGE = (854.0, 480.0)
FRAME = (1920, 1080)
STROKE_CAM = style.LINE_PROJ_PX * 2  # 10 camera px


def _scene(*prims, mode="normal") -> render.SceneMsg:
    return render.SceneMsg(seq=1, mode=mode, prims=list(prims))


def ring(c=(800.0, 400.0), r=60.0, pulse=False, id_="r"):
    return render.Ring(id=id_, c=list(c), r=r, role="target", pulse=pulse)


def lit(mask, x, y) -> bool:
    return bool(mask[int(round(y)), int(round(x))])


def test_ring_stroke_matches_projector_style():
    m = rasterize([ring()], H, PAGE, FRAME)
    assert m.shape == (1080, 1920) and m.dtype == np.uint8
    half = STROKE_CAM / 2
    for a in np.linspace(0, 2 * math.pi, 16, endpoint=False):
        for dr, expect in ((0, True), (half - 1.5, True), (-(half - 1.5), True), (half + 3, False), (-(half + 3), False)):
            x, y = 800 + (60 + dr) * math.cos(a), 400 + (60 + dr) * math.sin(a)
            assert lit(m, x, y) is expect, (a, dr)
    assert not lit(m, 800, 400)


def test_pulsing_ring_masks_its_widest_stroke():
    m = rasterize([ring(pulse=True)], H, PAGE, FRAME)
    half = STROKE_CAM * (1 + style.PULSE_GAIN) / 2  # 8 camera px
    assert lit(m, 800 + 60 + half - 1.5, 400) and lit(m, 800 + 60 - half + 1.5, 400)
    assert not lit(m, 800 + 60 + half + 3, 400)


def test_arrow_dot_poly_label_geometry():
    arrow = render.Arrow(id="a", a=[500.0, 500.0], b=[700.0, 500.0], role="source")
    dot = render.Dot(id="d", c=[300.0, 300.0], role="info")
    closed = render.Poly(id="p", pts=[[100.0, 700.0], [300.0, 700.0], [300.0, 900.0]], closed=True, role="warn")
    label = render.Label(id="l", at=[1000.0, 200.0], text="A6", role="info")
    m = rasterize([arrow, dot, closed, label], H, PAGE, FRAME)
    # shaft and both head strokes (4 widths long at +-0.5 rad, back from b), in page px then camera px
    assert lit(m, 600, 500) and lit(m, 700, 500)
    head = style.ARROW_HEAD_LINES * style.LINE_PROJ_PX
    for s in (-1, 1):
        px, py = 250 - head / 2 * math.cos(0.5), 200 + s * head / 2 * math.sin(0.5)
        assert lit(m, (px + 100) * 2, (py + 50) * 2)
    # dot: radius 1.5 widths = 7.5 CSS px = 15 camera px
    assert lit(m, 300 + 13, 300) and not lit(m, 300 + 18, 300)
    # closed poly draws its closing edge
    assert lit(m, 200, 800)
    # Conservative label box from `at` (left edge, vertical middle), including wide glyphs.
    font_cam = style.TEXT_PROJ_PX * 2
    assert lit(m, 1000 + 0.6 * font_cam * 2, 200) and lit(m, 1000, 200 + 0.6 * font_cam)
    assert not lit(m, 1000, 200 + 0.8 * font_cam)
    wide = render.Label(id="wide", at=[1000.0, 200.0], text="WWW", role="info")
    mw = rasterize([wide], H, PAGE, FRAME)
    assert lit(mw, 1000 + 0.95 * font_cam * 3, 200)
    open_ = rasterize([closed.model_copy(update={"closed": False})], H, PAGE, FRAME)
    assert not lit(open_, 200, 800)


def test_perspective_ring_follows_mapped_points():
    Hp = np.array([[0.48, 0.03, -90.0], [-0.02, 0.5, -40.0], [2e-5, 4e-5, 1.0]])
    m = rasterize([ring(c=(900.0, 500.0), r=80.0)], Hp, PAGE, FRAME)
    for a in np.linspace(0, 2 * math.pi, style.RING_SEGMENTS, endpoint=False):
        assert lit(m, 900 + 80 * math.cos(a), 500 + 80 * math.sin(a))
    assert not lit(m, 900, 500)


def test_prims_off_the_page_are_not_masked():
    assert not rasterize([ring(c=(50.0, 50.0), r=20.0)], H, PAGE, FRAME).any()  # page starts at camera (200, 100)


def test_mask_is_dilated_and_scales_with_frame_height():
    om = OverlayMask(FRAME, H, PAGE)
    assert om.dilate_px == 8
    om.set_scene(_scene(ring()), 1.0)
    m = om.mask_at(1.0)
    half = STROKE_CAM / 2
    assert lit(m, 800 + 60 + half + 6, 400) and not lit(m, 800 + 60 + half + 11, 400)
    assert not m.flags.writeable
    assert OverlayMask((1280, 720), H, PAGE).dilate_px == 5


def test_timeline_union_hold_and_release():
    om = OverlayMask(FRAME, H, PAGE)
    a, b = ring(c=(700.0, 400.0), id_="a"), ring(c=(1100.0, 400.0), id_="b")
    assert om.mask_at(5.0) is None
    om.set_scene(_scene(a), 1.0)
    om.set_scene(_scene(b), 2.0)
    in_a, in_b = (700 + 60, 400), (1100 + 60, 400)
    assert om.mask_at(0.9) is None  # before anything was shown
    m = om.mask_at(1.5)
    assert lit(m, *in_a) and not lit(m, *in_b)
    m = om.mask_at(2.1)  # within hold_s of the change: old and new
    assert lit(m, *in_a) and lit(m, *in_b)
    assert om.mask_at(2.1) is m  # cached union: same object
    m = om.mask_at(2.25)
    assert not lit(m, *in_a) and lit(m, *in_b)
    assert lit(om.drawn_at(2.1), *in_a) and not lit(om.drawn_at(2.25), *in_a)

    om.hold(3.0)
    assert om.mask_at(3.1).all()
    assert om.mask_at(2.95) is not None and not om.mask_at(2.95).all()  # frames dequeued before the switch
    assert om.clear_after() == math.inf
    om.set_scene(_scene(a), 3.1)  # published while the page shows the measurement mode
    assert om.mask_at(3.2).all()
    om.release(3.3)
    assert om.clear_after() == pytest.approx(3.5)
    assert om.mask_at(3.45).all()
    m = om.mask_at(3.55)
    assert lit(m, *in_a) and not lit(m, *in_b)  # the page shows the latest scene again


def test_non_normal_scene_masks_everything_and_empty_scene_masks_nothing():
    om = OverlayMask(FRAME, H, PAGE)
    om.set_scene(_scene(ring()), 1.0)
    om.set_scene(_scene(mode="pattern"), 2.0)
    assert om.mask_at(2.5).all()
    om.set_scene(_scene(), 3.0)
    assert om.mask_at(3.1).all()
    assert om.mask_at(3.3) is None


def test_rejects_singular_homography():
    with pytest.raises(ValueError):
        OverlayMask(FRAME, np.zeros((3, 3)), PAGE)
