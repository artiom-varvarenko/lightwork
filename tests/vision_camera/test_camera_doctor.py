"""``lightwork doctor`` camera section: no camera, a non-C920 only, and a C920 compared read-only against the lock."""

from __future__ import annotations

import sys

import pytest

from lightwork.commands import doctor
from lightwork.vision import camera

pytestmark = pytest.mark.skipif(sys.platform != "linux", reason="by-id resolution is Linux-only")


def test_no_camera_reports_none_and_doctor_exits_0(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(camera, "BY_ID_DIR", tmp_path / "missing")
    rows = doctor.section_camera()
    assert rows == [("camera", None, f"none found under {tmp_path / 'missing'} (live needs one; replay/verify don't)")]
    assert doctor.run(None) == 0
    assert "none found under" in capsys.readouterr().out


def test_non_c920_only(tmp_path):
    (tmp_path / "video0").touch()
    (tmp_path / "usb-Generic_Laptop_Camera-video-index0").symlink_to(tmp_path / "video0")
    rows = doctor.section_camera(by_id_dir=tmp_path)
    assert rows[0] == ("camera", None, f"usb-Generic_Laptop_Camera-video-index0 -> {tmp_path / 'video0'}")
    assert rows[1][0] == "camera chosen" and rows[1][1] is None and "no C920" in rows[1][2]


def test_c920_controls_vs_lock(tmp_path, monkeypatch, fake_v4l2):
    by_id = tmp_path / "by-id"
    by_id.mkdir()
    (tmp_path / "video2").touch()
    (by_id / "usb-046d_HD_Pro_Webcam_C920_ABCDEF-video-index0").symlink_to(tmp_path / "video2")
    state = tmp_path / "state"
    camera.save_lock({"exposure_time_absolute": 333}, camera.lock_path(state))
    fake = fake_v4l2(names="old")
    real_ctl = camera.V4l2Ctl
    monkeypatch.setattr(camera, "V4l2Ctl", lambda node: real_ctl(node, fake.run))
    monkeypatch.setattr(camera, "have_v4l2_ctl", lambda: True)
    rows = doctor.section_camera(by_id_dir=by_id, state_dir=state)
    by_name = {name.strip(): (ok, detail) for name, ok, detail in rows}
    assert by_name["camera chosen"] == (True, "usb-046d_HD_Pro_Webcam_C920_ABCDEF-video-index0")
    assert by_name["camera lock"] == (True, str(state / "camera.json"))
    assert by_name["camera control names"] == (True, "old")
    assert by_name["exposure_absolute"] == (None, "now 250, lock 333 (applied when the camera opens)")
    assert by_name["brightness"] == (True, "now 128, lock 128")
    assert not fake.writes  # doctor never changes the camera
    assert not any(ok is False for ok, _ in by_name.values())


def test_bad_lock_file_fails(tmp_path):
    by_id = tmp_path / "by-id"
    by_id.mkdir()
    (tmp_path / "video2").touch()
    (by_id / "usb-046d_HD_Pro_Webcam_C920_ABCDEF-video-index0").symlink_to(tmp_path / "video2")
    (tmp_path / "camera.json").write_text('{"controls": {"exposure": 3}}')
    rows = doctor.section_camera(by_id_dir=by_id, state_dir=tmp_path)
    assert rows[-1][0] == "camera lock" and rows[-1][1] is False and "unknown control" in rows[-1][2]
