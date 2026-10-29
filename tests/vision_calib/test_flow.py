from __future__ import annotations

import time

import cv2
import numpy as np
import pytest

from lightwork.contracts import canonical_bytes, file_sha
from lightwork.contracts.layout import Calibration
from lightwork.contracts.render import SceneMsg
from lightwork.geometry import apply_h
from lightwork.vision.calib import CalibrationError, calibrate
from lightwork.vision.calib.bump import BumpMonitor, tick_prims
from lightwork.vision.calib.check import calib_message, check_accuracy, ring_centre
from lightwork.vision.calib.flow import add_flat_field, calibrate_plate_height, flash_pattern, flat_field_path, load_calibration, recalibrate, run_calibration, with_mm
from lightwork.vision.calib.mm import nominal_scale


def test_binding_calibrate_writes_canonical_c7(make_rig, truth_error, tmp_path):
    rig, cam, page, link = make_rig(dpr=2.625)
    path = tmp_path / "calib/calib.json"
    result = run_calibration(cam, link, page.hello(), path=path)
    assert result.calib.mm.source == "nominal"
    assert result.flash.n_frames >= 3
    assert result.flash.t_s < 2
    assert truth_error(result.calib.H_cam_to_page, rig).max() < 0.4
    assert path.read_bytes() == canonical_bytes(result.calib)
    assert load_calibration(path) == result.calib
    assert Calibration.model_validate_json(path.read_bytes()) == result.calib
    assert result.calib.flat_field_sha is None
    assert list(path.parent.iterdir()) == [path]
    assert link.modes[-1] == "normal"
    print(f"full flash: {result.flash.t_s:.3f}s, RMS={result.calib.residuals.rms_px}; "
          f"truth max={truth_error(result.calib.H_cam_to_page, rig).max():.4f} page px")
    # Binding entry point returns C7, not the diagnostic wrapper.
    assert isinstance(calibrate(cam, link, page.hello(), path=path, mm="nominal"), Calibration)


def test_card_scale_in_full_flow(make_rig, tmp_path):
    rig, cam, page, link = make_rig()
    rig.place_card((230, 155))
    result = run_calibration(cam, link, page.hello(), mm="card", path=tmp_path / "calib.json")
    assert result.card.n_markers == 4
    assert result.calib.mm.source == "card"
    assert result.calib.flat_field_sha is None


def test_flat_field_crop_only_storage(make_rig, truth_calib, tmp_path):
    rig, cam, page, link = make_rig()
    calib = truth_calib(rig, page)
    path = tmp_path / "calib.json"
    crop = (640, 360, 640, 320)
    updated, flat = add_flat_field(cam, link, calib, path=path, work_area_xywh=crop)
    assert flat is not None and flat.verified, None if flat is None else flat.reasons
    assert updated.flat_field_crop_xywh == list(crop)
    frame_path = flat_field_path(path, updated.flat_field_sha)
    assert frame_path.read_bytes() == flat.jpeg
    assert file_sha(frame_path) == updated.flat_field_sha
    image = cv2.imread(str(frame_path))
    assert image.shape == (320, 640, 3)
    assert load_calibration(path) == updated
    assert link.modes[-1] == "normal"


def test_flat_field_requires_selected_crop(make_rig, truth_calib, tmp_path):
    rig, cam, page, link = make_rig()
    with pytest.raises(ValueError, match="privacy"):
        run_calibration(cam, link, page.hello(), path=tmp_path / "calib.json", flat_field=True)
    with pytest.raises(ValueError, match="inside"):
        add_flat_field(cam, link, truth_calib(rig, page), path=tmp_path / "calib.json", work_area_xywh=(-1, 0, 100, 100))
    assert not list(tmp_path.iterdir())


def test_objects_reject_flat_without_overwriting(make_rig, truth_calib, tmp_path):
    rig, cam, page, link = make_rig()
    rig.place_card((230, 155))
    calib = truth_calib(rig, page)
    updated, flat = add_flat_field(cam, link, calib, path=tmp_path / "calib.json", work_area_xywh=(600, 320, 750, 620))
    assert flat is not None and not flat.verified
    assert updated == calib
    assert not list(tmp_path.iterdir())
    assert link.modes[-1] == "normal"


def test_failure_restores_normal_and_writes_nothing(make_rig, tmp_path):
    _, cam, page, _ = make_rig()

    class NoAck:
        def __init__(self):
            self.modes = []

        def show_mode(self, mode, *args, **kw):
            self.modes.append(mode)
            return None

    link = NoAck()
    with pytest.raises(CalibrationError, match="ACK"):
        calibrate(cam, link, page.hello(), path=tmp_path / "calib.json")
    assert link.modes == ["pattern", "normal"]
    assert not list(tmp_path.iterdir())


def test_flash_rejects_persistent_heavy_occlusion(make_rig):
    rig, cam, page, link = make_rig()
    rig.occlude([[0, 0], [1500, 0], [1500, 1080], [0, 1080]])
    with pytest.raises(CalibrationError, match="need >= 6"):
        flash_pattern(cam, link, page.w, page.h, budget_s=0.45)


def test_plate_plane_stored_separately(make_rig, truth_calib, truth_error, tmp_path):
    rig, cam, page, link = make_rig()
    old = truth_calib(rig, page)
    # A different projected-plane mapping is represented by the simulator's independent projector transform.
    rig.move_projector(4, -2)
    new, flash = calibrate_plate_height(cam, link, page.hello(), old, path=tmp_path / "calib.json")
    assert new.H_cam_to_page == old.H_cam_to_page
    assert truth_error(new.H_plate_cam_to_page, rig).max() < 0.4
    assert flash.t_s < 2
    assert link.modes[-1] == "normal"


def test_replacing_mm_scale(make_rig, truth_calib, tmp_path):
    rig, _, page, _ = make_rig()
    old = truth_calib(rig, page)
    mm = nominal_scale(rig.H_cam_to_page, page.w, 650)
    out = with_mm(old, mm, tmp_path / "calib.json")
    assert out.mm == mm and out.H_cam_to_page == old.H_cam_to_page
    assert load_calibration(tmp_path / "calib.json") == out


def _tick_reference(rig, page, link, calib):
    link.set_calib(calib_message(calib))
    link.publish_scene(SceneMsg(seq=0, mode="normal", prims=tick_prims(calib)))
    return rig.render(time.monotonic() + 0.2)


@pytest.mark.parametrize("cause", ["camera", "projector", "both"])
def test_bump_detected_and_resolved(make_rig, truth_calib, truth_error, tmp_path, cause):
    rig, cam, page, link = make_rig(seed=28)
    old = truth_calib(rig, page).model_copy(update={
        "flat_field_sha": "a" * 64, "flat_field_crop_xywh": [640, 360, 640, 320],
        "H_plate_cam_to_page": rig.H_cam_to_page.tolist(),
    })
    ref = _tick_reference(rig, page, link, old)
    monitor = BumpMonitor(ref, old)
    assert monitor.n_patches == 3 and monitor.n_ticks == 4  # one flat/noise-only corner is correctly ignored
    for _ in range(8):
        assert not monitor.check(rig.render(time.monotonic() + 0.2)).moved
    before = time.perf_counter()
    if cause in ("camera", "both"):
        rig.move_camera(5, -4)
    if cause in ("projector", "both"):
        rig.move_projector(3, 2)
    first = monitor.check(rig.render(time.monotonic() + 0.2))
    confirmed = monitor.check(rig.render(time.monotonic() + 0.2))
    assert not first.moved and confirmed.moved
    detection_s = time.perf_counter() - before
    assert detection_s < 2
    result = recalibrate(cam, link, old, page.hello(), confirmed, path=tmp_path / "calib.json")
    assert result.cause == cause
    assert result.t_s < 5
    error = truth_error(result.calib.H_cam_to_page, rig)
    assert error.max() < 0.5
    assert result.calib.flat_field_sha is None and result.calib.flat_field_crop_xywh is None
    assert result.calib.H_plate_cam_to_page is None
    if cause == "camera":
        np.testing.assert_allclose(apply_h(result.camera_motion, [[500, 500], [1000, 800]]), [[505, 496], [1005, 796]], atol=0.4)
    print(f"bump {cause}: detect={detection_s:.3f}s, re-solve={result.t_s:.3f}s, truth max={error.max():.4f} page px")


def test_transient_bump_is_not_confirmed(make_rig, truth_calib):
    rig, _, page, link = make_rig(seed=2)
    calib = truth_calib(rig, page)
    ref = _tick_reference(rig, page, link, calib)
    monitor = BumpMonitor(ref, calib)
    shifted = cv2.warpAffine(ref, np.float32([[1, 0, 5], [0, 1, 4]]), (ref.shape[1], ref.shape[0]))
    assert not monitor.check(shifted).moved
    assert not monitor.check(ref).moved
    assert not monitor.check(shifted).moved
    for _ in range(20):
        assert not monitor.check(ref).moved


def test_unobservable_motion_invalidates_measured_scale(make_rig, truth_calib, tmp_path):
    rig, cam, page, link = make_rig(texture=False, noise=0)
    calib = truth_calib(rig, page)
    ref = _tick_reference(rig, page, link, calib)
    monitor = BumpMonitor(ref, calib)
    rig.move_camera(5, 4)
    monitor.check(rig.render(time.monotonic() + 0.2))
    bump = monitor.check(rig.render(time.monotonic() + 0.2))
    assert bump.moved and not bump.camera_observable
    result = recalibrate(cam, link, calib, page.hello(), bump, path=tmp_path / "calib.json")
    assert result.cause == "unobserved" and result.calib.mm.H_cam_to_mm is None


def test_card_ring_accuracy_and_deliberate_misalignment(make_rig, truth_calib):
    rig, cam, page, link = make_rig()
    rig.place_card((230, 155))
    calib = truth_calib(rig, page)
    good = check_accuracy(cam, link, calib)
    print(f"C1 good: mean={good.mean_mm}, max={good.max_mm} mm")
    assert good.passed and good.max_mm < 0.5
    translation = np.array([[1, 0, 8], [0, 1, 0], [0, 0, 1]])
    bad_calib = calib.model_copy(update={"H_cam_to_page": (translation @ np.asarray(calib.H_cam_to_page)).tolist()})
    bad = check_accuracy(cam, link, bad_calib)
    print(f"C1 deliberate 8-page-px offset: mean={bad.mean_mm}, max={bad.max_mm} mm")
    assert not bad.passed and bad.mean_mm > 3
    assert link.modes[-1] == "normal"


def test_ring_absence_not_sensor_noise():
    image = np.random.default_rng(1).normal(0, 3.5, (150, 150)).astype(np.float32)
    assert ring_centre(image, np.array([75, 75]), 40) is None
