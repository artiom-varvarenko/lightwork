"""``python -m lightwork.measure.colour`` (rig:D1 separable-steps tool) on synthetic tea photos."""

from __future__ import annotations

import json

import cv2

from lightwork.measure.colour import main, separable_steps
from lightwork.testing.synth import BROWN_DYE, encode_jpeg, well_plate

COLS = 9  # 8 dilutions + blank A9
CORNERS = ["230,140", f"{230 + 60 * (COLS - 1)},140", f"{230 + 60 * (COLS - 1)},140", "230,140"]


def _photos(tmp_path, n: int, ods: dict[str, float]):
    paths = []
    for i in range(n):
        img, _ = well_plate(ods, rows=1, cols=COLS, dye=BROWN_DYE, noise_sd=2, seed=50 + i)
        p = tmp_path / f"tea{i}.jpg"
        p.write_bytes(encode_jpeg(img))
        paths.append(str(p))
    return paths


def _ods(start=1.0):
    return {f"A{k}": start / 2 ** (k - 1) for k in range(1, COLS)}


def test_series_report_with_blank_cell(tmp_path, capsys):
    photos = _photos(tmp_path, 5, _ods())
    code = main([*photos, "--corners", *CORNERS, "--cols", str(COLS), "--blank", "A9", "--json"])
    rep = json.loads(capsys.readouterr().out)
    assert code == 0
    assert rep["frames"] == 5 and rep["sd_kind"] == "frame-to-frame SD" and rep["channel"] == "b"
    assert list(rep["cells"]) == [f"A{k}" for k in range(1, COLS)]
    assert abs(rep["cells"]["A1"]["od"] - 1.0) < 0.05
    assert rep["separable_steps"] == 7 and rep["series_ok"], rep


def test_series_report_with_reference_photo_and_annotation(tmp_path, capsys):
    photos = _photos(tmp_path, 1, _ods())
    ref = tmp_path / "ref.png"
    cv2.imwrite(str(ref), well_plate({}, rows=1, cols=COLS, dye=BROWN_DYE, noise_sd=2, seed=9)[0])
    ann = tmp_path / "rois.png"
    code = main([*photos, "--corners", *CORNERS, "--cols", str(COLS), "--cells", "A1", "A2", "A3", "A4", "A5",
                 "A6", "A7", "--ref", str(ref), "--annotate", str(ann), "--min-steps", "7"])
    out = capsys.readouterr().out
    assert code == 1  # 6 steps among 7 cells < 7
    assert "separable steps: 6 (target >= 7)" in out and "single frame" in out
    assert cv2.imread(str(ann)) is not None


def test_merged_cells_stop_the_count():
    # A3 and A4 indistinguishable: count stops at the first non-separable step
    assert separable_steps([1.0, 0.5, 0.25, 0.25, 0.12], [0.01] * 5) == 2
    # separable but wrong ratio (two halvings) also stops it
    assert separable_steps([1.0, 0.5, 0.125], [0.01] * 3) == 1
    # below the ratio floor only the 3-SD separation counts
    assert separable_steps([0.05, 0.026, 0.01], [0.001] * 3) == 2


def test_usage_errors(tmp_path, capsys):
    bad = tmp_path / "x.jpg"
    bad.write_bytes(b"nope")
    assert main([str(bad), "--corners", *CORNERS, "--cols", str(COLS), "--blank", "A9"]) == 2
    assert "error:" in capsys.readouterr().err
