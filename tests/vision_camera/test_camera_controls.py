"""v4l2 control locking against a fake C920 (``briefs/rig.md`` §3.1)."""

from __future__ import annotations

import json

import pytest

from lightwork.vision import camera
from lightwork.vision.camera import (
    LOCK_ORDER,
    V4l2Ctl,
    apply_lock,
    compare_lock,
    load_lock,
    parse_get_ctrls,
    parse_list_ctrls,
    save_lock,
)

LAPTOP_LIST_CTRLS = """
User Controls

                     brightness 0x00980900 (int)    : min=-64 max=64 step=1 default=0 value=0 flags=has-min-max
        white_balance_automatic 0x0098090c (bool)   : default=1 value=1
           power_line_frequency 0x00980918 (menu)   : min=0 max=2 default=2 value=1 (50 Hz)
      white_balance_temperature 0x0098091a (int)    : min=2800 max=6500 step=10 default=4600 value=4600 flags=inactive, has-min-max

Camera Controls

                  auto_exposure 0x009a0901 (menu)   : min=0 max=3 default=3 value=3 (Aperture Priority Mode)
         exposure_time_absolute 0x009a0902 (int)    : min=10 max=625 step=1 default=156 value=156 flags=inactive, has-min-max
"""


def defaults() -> dict[str, int | None]:
    return {s.name: s.default for s in LOCK_ORDER}


def written_names(fake) -> list[str]:
    return [name for name, _ in fake.writes]


def test_parse_list_ctrls_real_output():
    c = parse_list_ctrls(LAPTOP_LIST_CTRLS)
    assert set(c) == {"brightness", "white_balance_automatic", "power_line_frequency", "white_balance_temperature",
                      "auto_exposure", "exposure_time_absolute"}
    assert c["brightness"].min == -64 and c["brightness"].value == 0
    assert c["power_line_frequency"].value == 1 and c["power_line_frequency"].type == "menu"
    assert c["auto_exposure"].value == 3
    assert c["white_balance_temperature"].inactive and not c["brightness"].inactive
    assert c["white_balance_automatic"].min == 0 and c["white_balance_automatic"].max == 1
    assert c["white_balance_temperature"].clamp(5003) == 5000  # snapped to step 10
    assert parse_get_ctrls("auto_exposure: 1 (Manual Mode)\ngain: 0\n") == {"auto_exposure": 1, "gain": 0}


@pytest.mark.parametrize("names", ["new", "old"])
def test_apply_lock_both_name_sets_in_order(fake_v4l2, names):
    fake = fake_v4l2(names=names)
    report = apply_lock(V4l2Ctl("/dev/video9", fake.run), defaults())
    assert report.ok, report.lines()
    assert report.name_set == names
    assert not report.absent
    order = [spec.name for spec in LOCK_ORDER]
    assert [r.name for r in report.rows] == order
    # every write went out under the device's own name, in LOCK_ORDER, one control per call
    assert len(fake.writes) == len(order)
    assert written_names(fake) == [r.device_name for r in report.rows]
    if names == "old":
        assert "exposure_absolute" in written_names(fake) and "exposure_time_absolute" not in written_names(fake)
    # auto switches before their manual values (the fake raises an I/O error otherwise)
    w = written_names(fake)
    exp_auto, exp_abs = ("auto_exposure", "exposure_time_absolute") if names == "new" else ("exposure_auto", "exposure_absolute")
    assert w.index(exp_auto) < w.index(exp_abs)
    focus_auto = "focus_automatic_continuous" if names == "new" else "focus_auto"
    assert w.index(focus_auto) < w.index("focus_absolute")
    # readback = defaults from the brief, None-default controls = the device default written explicitly
    controls = report.controls()
    assert controls[exp_auto] == 1 and controls[exp_abs] == defaults()["exposure_time_absolute"]
    assert controls["white_balance_temperature"] == 5000 and controls["power_line_frequency"] == 1
    assert controls["focus_absolute"] == 20 and controls["brightness"] == 128 and controls["gain"] == 0
    # the readback was one -C call after all writes
    assert fake.calls[-1][3] == "-C"


def test_manual_before_auto_off_is_rejected_by_fake(fake_v4l2):
    """Sanity check of the fake: the ordering requirement the lock is built around."""
    fake = fake_v4l2()
    ctl = V4l2Ctl("/dev/video9", fake.run)
    assert "Input/output error" in ctl.set("focus_absolute", 20)
    assert ctl.set("focus_automatic_continuous", 0) is None
    assert ctl.set("focus_absolute", 20) is None


def test_readback_mismatch_is_reported(fake_v4l2):
    fake = fake_v4l2(ignore=("power_line_frequency", "exposure_time_absolute"))
    report = apply_lock(V4l2Ctl("/dev/video9", fake.run), defaults())
    assert not report.ok
    assert [r.name for r in report.mismatches] == ["exposure_time_absolute", "power_line_frequency"]
    bad = report.mismatches[0]
    want = defaults()["exposure_time_absolute"]
    assert bad.value == want and bad.readback == 250 and bad.error is None
    text = "\n".join(report.lines())
    assert f"exposure_time_absolute: want {want}, MISMATCH read 250" in text
    assert "power_line_frequency: want 1, MISMATCH read 2" in text


def test_write_error_is_reported(fake_v4l2):
    fake = fake_v4l2(ignore=("focus_automatic_continuous",))  # AF stays on -> focus write fails with I/O error
    report = apply_lock(V4l2Ctl("/dev/video9", fake.run), defaults())
    rows = {r.name: r for r in report.mismatches}
    assert set(rows) == {"focus_automatic_continuous", "focus_absolute"}
    assert "Input/output error" in rows["focus_absolute"].error


def test_clamping_and_absent_controls(fake_v4l2, tmp_path):
    fake = fake_v4l2(drop=("zoom_absolute",))
    lock = load_lock(camera.lock_path(tmp_path))
    lock.update(white_balance_temperature=9000, focus_absolute=22, exposure_time_absolute=1)
    report = apply_lock(V4l2Ctl("/dev/video9", fake.run), lock)
    assert report.ok, report.lines()
    assert report.absent == ["zoom_absolute"]
    rows = {r.name: r for r in report.rows}
    assert (rows["white_balance_temperature"].value, rows["white_balance_temperature"].requested) == (6500, 9000)
    assert rows["focus_absolute"].value == 20  # step 5
    assert rows["exposure_time_absolute"].value == 3
    assert "want 6500 (clamped from 9000), ok" in "\n".join(report.lines())
    assert "zoom_absolute: not on this device" in report.lines()


def test_compare_lock_never_writes(fake_v4l2):
    fake = fake_v4l2()
    report = compare_lock(V4l2Ctl("/dev/video9", fake.run), defaults())
    assert not fake.writes and all(c[3] == "--list-ctrls" for c in fake.calls)
    assert not report.ok  # factory state is auto everything
    assert {r.name for r in report.mismatches} >= {"auto_exposure", "focus_automatic_continuous"}
    apply_lock(V4l2Ctl("/dev/video9", fake.run), defaults())
    assert compare_lock(V4l2Ctl("/dev/video9", fake.run), defaults()).ok


def test_lock_file(tmp_path):
    path = tmp_path / "camera.json"
    defaults = load_lock(path)
    assert defaults == {s.name: s.default for s in LOCK_ORDER}
    path.write_text(json.dumps({"controls": {"exposure_absolute": 333, "focus_auto": 0, "gain": None}}))
    lock = load_lock(path)
    assert lock["exposure_time_absolute"] == 333 and lock["focus_automatic_continuous"] == 0 and lock["gain"] is None
    lock["focus_absolute"] = 35
    save_lock(lock, path)
    assert load_lock(path) == lock
    path.write_text(json.dumps({"controls": {"exposure": 1}}))
    with pytest.raises(ValueError, match="unknown control"):
        load_lock(path)
    path.write_text(json.dumps({"controls": {"gain": 1.5}}))
    with pytest.raises(ValueError, match="integer"):
        load_lock(path)
