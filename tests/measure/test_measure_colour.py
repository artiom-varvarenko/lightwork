from __future__ import annotations

import cv2
import numpy as np
import pytest

from lightwork.contracts.stepgraph import Grid
from lightwork.geometry import GridGeom
from lightwork.measure.colour import CellStats, cell_stats, decode, delta_e, od, od_sd
from lightwork.measure.grid import cell_rois
from lightwork.testing.synth import BROWN_DYE, encode_jpeg, linear_to_srgb, srgb_to_linear, well_plate

ROWS, COLS = 8, 12


def _rois(names, **plate_kw):
    _, g = well_plate({}, **plate_kw)
    return cell_rois(GridGeom(g.corners, Grid(rows=ROWS, cols=COLS, naming="A1")), names)


def _od(img, ref, cell="A1", channel="r", **kw):
    centres, r = _rois([cell])
    s, s0 = cell_stats(img, centres, r, channel, **kw)[cell], cell_stats(ref, centres, r, channel, **kw)[cell]
    return od(s.i_lin, s0.i_lin)


OD_LEVELS = [0.05, 0.1, 0.2, 0.4, 0.8, 1.2, 1.5]


@pytest.mark.parametrize("target", OD_LEVELS)
def test_known_od_noise_free(target):
    img, _ = well_plate({"A1": target})
    ref, _ = well_plate({})
    assert _od(img, ref) == pytest.approx(target, abs=0.02)


@pytest.mark.parametrize("seed", [1, 2, 3])
@pytest.mark.parametrize("target", OD_LEVELS)
def test_known_od_with_noise(target, seed):
    img, _ = well_plate({"A1": target}, noise_sd=2, seed=seed)
    ref, _ = well_plate({}, noise_sd=2, seed=seed + 100)
    assert _od(img, ref) == pytest.approx(target, abs=0.05)


@pytest.mark.parametrize("target", OD_LEVELS)
def test_known_od_through_keyframe_jpeg(target):
    img, _ = well_plate({"A1": target}, noise_sd=2, seed=7)
    ref, _ = well_plate({}, noise_sd=2, seed=8)
    assert _od(decode(encode_jpeg(img)), decode(encode_jpeg(ref))) == pytest.approx(target, abs=0.05)


@pytest.mark.parametrize("target", [0.1, 0.5, 1.0])
def test_tea_on_blue_channel(target):
    img, _ = well_plate({"A1": target}, dye=BROWN_DYE, noise_sd=2, seed=4)
    ref, _ = well_plate({}, dye=BROWN_DYE, noise_sd=2, seed=5)
    assert _od(img, ref, channel="b") == pytest.approx(target, abs=0.05)
    # the red channel barely sees tea
    assert _od(img, ref, channel="r") < 0.5 * target


def test_deterministic_and_origin_offset():
    img, _ = well_plate({"A1": 0.7, "B2": 0.3}, noise_sd=2, seed=9)
    centres, r = _rois(["A1", "B2"])
    a = cell_stats(img, centres, r, "r")
    assert a == cell_stats(img.copy(), centres, r, "r")
    crop = img[100:400, 150:600]
    assert cell_stats(crop, centres, r, "r", origin=(150, 100)) == a


def test_specular_pixels_rejected():
    img, _ = well_plate({"A1": 0.8}, noise_sd=2, seed=11)
    ref, _ = well_plate({}, noise_sd=2, seed=12)
    centres, r = _rois(["A1"])
    clean = cell_stats(img, centres, r, "r")["A1"]
    hot = img.copy()
    cx, cy = (round(v) for v in centres["A1"])
    cv2.circle(hot, (cx + 3, cy - 2), 4, (255, 255, 255), -1)  # specular hotspot inside the ROI
    hot[cy + 6, cx] = (40, 60, 252)  # one clipped channel is enough to reject a pixel
    n_hot = int(((hot != img).any(axis=2)).sum())
    got = cell_stats(hot, centres, r, "r")["A1"]
    assert got.n == clean.n - n_hot
    assert od(got.i_lin, cell_stats(ref, centres, r, "r")["A1"].i_lin) == pytest.approx(0.8, abs=0.02)
    # fully saturated cell -> unreadable, not a fake reading
    sat = img.copy()
    cv2.circle(sat, (cx, cy), round(r) + 2, (255, 255, 255), -1)
    assert cell_stats(sat, centres, r, "r")["A1"] == CellStats(0.0, (0.0, 0.0, 0.0), 0, 0.0)


def _shade(img: np.ndarray, gain: np.ndarray) -> np.ndarray:
    """Multiply linear light by a per-column gain (uneven projector illumination)."""
    return linear_to_srgb(srgb_to_linear(img) * gain[None, :, None])


def test_flat_field_cancels_uneven_illumination():
    target = 0.6
    img, _ = well_plate({"A1": target})
    gain = np.linspace(0.6, 1.0, img.shape[1])
    lit = _shade(img, gain)
    flat = _shade(np.full_like(img, 235), gain)
    centres, r = _rois(["A1", "A12"])
    # I_ref = blank cell A12 in the same frame, far to the right (brighter) of A1
    raw = cell_stats(lit, centres, r, "r")
    assert od(raw["A1"].i_lin, raw["A12"].i_lin) - target > 0.05
    ff = cell_stats(lit, centres, r, "r", flat_field=flat)
    assert od(ff["A1"].i_lin, ff["A12"].i_lin) == pytest.approx(target, abs=0.02)
    # a crop with its origin reads the full-frame flat field at the right place
    crop = lit[80:420, 150:1000]
    ffc = cell_stats(crop, centres, r, "r", origin=(150, 80), flat_field=flat)
    assert ffc == ff
    with pytest.raises(ValueError):
        cell_stats(lit, centres, r, "r", origin=(150, 80), flat_field=flat)


def _lab_d65(bgr_u8) -> tuple[float, float, float]:
    """Textbook sRGB -> XYZ (D65) -> CIE Lab, independent of OpenCV."""
    b, g, r = srgb_to_linear(np.asarray(bgr_u8, dtype=np.float64))
    x = 0.412453 * r + 0.357580 * g + 0.180423 * b
    y = 0.212671 * r + 0.715160 * g + 0.072169 * b
    z = 0.019334 * r + 0.119193 * g + 0.950227 * b

    def f(t):
        return np.cbrt(t) if t > (6 / 29) ** 3 else t / (3 * (6 / 29) ** 2) + 4 / 29

    fx, fy, fz = f(x / 0.950456), f(y), f(z / 1.088754)
    return 116 * fy - 16, 500 * (fx - fy), 200 * (fy - fz)


def test_lab_is_d65_of_the_mean_linear_colour():
    img, _ = well_plate({"A1": 0.8})
    centres, r = _rois(["A1", "A2"])
    s = cell_stats(img, centres, r, "r")
    for cell in ("A1", "A2"):
        x, y = (round(v) for v in centres[cell])
        assert s[cell].lab == pytest.approx(_lab_d65(img[y, x]), abs=0.02)
    assert s["A1"].lab[2] < -10  # blue dye: negative b*
    assert abs(s["A2"].lab[1]) < 2 and abs(s["A2"].lab[2]) < 2  # clear well: near neutral


@pytest.mark.parametrize(
    "lab1,lab2,want",
    [  # Sharma, Wu, Dalal 2005 test data
        ((50.0, 2.6772, -79.7751), (50.0, 0.0, -82.7485), 2.0425),
        ((50.0, 0.0, 0.0), (50.0, -1.0, 2.0), 2.3669),
        ((50.0, 2.49, -0.001), (50.0, -2.49, 0.0009), 7.1792),
        ((50.0, 2.5, 0.0), (73.0, 25.0, -18.0), 27.1492),
    ],
)
def test_delta_e_ciede2000(lab1, lab2, want):
    assert delta_e(lab1, lab2) == pytest.approx(want, abs=0.01)
    assert delta_e(lab2, lab1) == delta_e(lab1, lab2)
    assert delta_e(lab1, lab1) == 0.0


def test_od_semantics():
    assert od(0.5, 1.0) == pytest.approx(0.301, abs=1e-4)
    assert od(1.1, 1.0) < 0  # brighter than the reference stays visible
    assert od(0.0, 1.0) > 4  # clamped, finite
    assert od(0.123456789, 0.7) == round(od(0.123456789, 0.7), 4)
    for bad in [(0.5, 0.0), (-0.1, 1.0), (float("nan"), 1.0)]:
        with pytest.raises(ValueError):
            od(*bad)


def test_od_sd():
    assert od_sd(CellStats(0.5, (0.0, 0.0, 0.0), 100, 0.01)) == pytest.approx(0.01 / (0.5 * np.log(10)), abs=1e-4)
    assert od_sd(CellStats(0.0, (0.0, 0.0, 0.0), 0, 0.0)) == 0.0


def test_bad_inputs():
    img, _ = well_plate({})
    centres, r = _rois(["A1"])
    with pytest.raises(ValueError):
        cell_stats(img, centres, r, "lab")
    with pytest.raises(ValueError):
        cell_stats(img[:, :, 0], centres, r, "r")
    with pytest.raises(ValueError):
        decode(b"not a jpeg")
