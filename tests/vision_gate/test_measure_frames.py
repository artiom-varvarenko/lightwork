from __future__ import annotations

from collections import deque

import cv2
import numpy as np
import pytest

from lightwork.contracts import render
from lightwork.contracts.canonical import sha256_hex
from lightwork.contracts.events import make_payload
from lightwork.vision import measure_frames as mf
from lightwork.vision.overlay_mask import OverlayMask
from lightwork.vision.source import Frame

SIZE = (320, 240)
CROP = (30, 20, 240, 180)


def image(value):
    return np.full((SIZE[1], SIZE[0], 3), value, np.uint8)


class FakeRig:
    """Camera dequeue and page ACK timestamps share a deterministic laptop clock."""

    def __init__(self, before, images=(), times=None, acks=(10.04, 10.23)):
        self.now = 10.0
        self.current = None if before is None else Frame(before, 9.99, 0)
        if times is None:
            times = [10.166667 + k / 30 for k in range(len(images))]
        self.frames = deque(Frame(img, t, k + 1) for k, (img, t) in enumerate(zip(images, times)))
        self.acks = deque(acks)
        self.modes = []
        self.after = []
        self.discarded = []
        self.returned = []

    def clock(self):
        return self.now

    def latest(self):
        return self.current

    def wait_frame(self, after_t, timeout):
        self.after.append(after_t)
        while self.frames:
            f = self.frames.popleft()
            if f.t_mono <= after_t:
                self.discarded.append(f)
                continue
            self.now = max(self.now, f.t_mono)
            self.current = f
            self.returned.append(f)
            return f
        self.now += timeout
        return None

    def show_mode(self, mode, pattern=None, timeout=1.0):
        self.modes.append(mode)
        ack = self.acks.popleft()
        if isinstance(ack, Exception):
            raise ack
        if ack is not None:
            self.now = max(self.now, ack)
            return self.now
        return None


def overlay():
    om = OverlayMask(SIZE, np.eye(3), SIZE)
    prims = [render.Ring(id=str(y), c=[160.0, float(y)], r=30.0, role="target") for y in (60, 180)]
    om.set_scene(render.SceneMsg(seq=1, mode="normal", prims=prims), 9.0)
    return om


def projected(base, om, *, bottom_only=False):
    out = base.copy()
    mask = om.drawn_at(9.99) > 0
    if bottom_only:
        mask[:120] = False
    out[mask] = np.minimum(out[mask].astype(np.int16) + 75, 245).astype(np.uint8)
    return out


def capture(rig, mode="grey", om=None, crop=CROP, **kw):
    return mf.measure_frame(rig, rig, mode, crop, overlay=om, clock=rig.clock, **kw)


def test_discard_every_frame_before_ack_plus_120ms():
    rig = FakeRig(image(100), [image(60), image(70), image(80), image(160), image(162)],
                  [10.02, 10.10, 10.159999, 10.166667, 10.20])
    cap = capture(rig)
    assert cap.verified and cap.reasons == ()
    assert rig.modes == ["grey", "normal"]
    assert rig.after[0] == pytest.approx(10.16)
    assert len(rig.discarded) == 3
    assert all(f.t_mono > cap.t_ack + 0.12 for f in rig.returned)
    assert cap.t_capture == 10.20
    assert cap.image.shape == (180, 240, 3)
    assert np.all(cap.image == 161)
    assert cap.frames_seen == 2
    assert cap.t_restore - cap.t_send < 0.3  # a deterministic 40 ms page ACK, 30 fps camera


def test_jpeg_is_cropped_and_encoded_after_restoration(monkeypatch):
    img = image(5)
    img[20:200, 30:270] = 160
    rig = FakeRig(image(100), [img, img])
    real_encode = mf.encode_jpeg

    def encode(a):
        assert rig.modes[-1] == "normal"
        return real_encode(a)

    monkeypatch.setattr(mf, "encode_jpeg", encode)
    jpeg, payload = mf.capture_keyframe(rig, rig, "grey", CROP, t0=10.0, clock=rig.clock)
    decoded = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
    assert decoded.shape == (180, 240, 3)
    assert np.max(np.abs(decoded.astype(np.int16) - 160)) <= 1
    assert payload == {"crop_xywh": list(CROP), "mode": "grey", "verified": True, "t_capture": 0.2}
    full = {"frame_sha": sha256_hex(jpeg), **payload}
    assert make_payload("keyframe", full) == full


@pytest.mark.parametrize("mode", ["grey", "dark"])
@pytest.mark.parametrize("bottom_only", [False, True])
def test_stale_overlay_is_rejected_even_with_good_ack(mode, bottom_only):
    om = overlay()
    before = projected(image(100), om)
    stale = projected(image(160 if mode == "grey" else 100), om, bottom_only=bottom_only)
    rig = FakeRig(before, [stale] * 4)
    cap = capture(rig, mode, om)
    assert not cap.verified
    assert any("overlay still visible" in r for r in cap.reasons)
    assert rig.modes == [mode, "normal"]
    assert om.clear_after() == pytest.approx(cap.t_restore + 0.2)


@pytest.mark.parametrize("mode", ["grey", "dark"])
def test_removed_overlay_is_verified(mode):
    om = overlay()
    rig = FakeRig(projected(image(100), om), [image(160 if mode == "grey" else 100)] * 2)
    cap = capture(rig, mode, om)
    assert cap.verified, cap.reasons
    assert om.mask_at(10.1).all()
    assert om.mask_at(10.5) is not None and not om.mask_at(10.5).all()


def test_grey_ack_without_actual_grey_light_is_rejected():
    rig = FakeRig(image(100), [image(100)] * 4)
    cap = capture(rig)
    assert not cap.verified
    assert any("page did not light" in r for r in cap.reasons)


def test_rolling_shutter_partial_grey_frame_is_rejected():
    half = image(160)
    half[120:] = 100
    rig = FakeRig(image(100), [half] * 4)
    cap = capture(rig)
    assert not cap.verified
    assert any("page did not light" in r for r in cap.reasons)


@pytest.mark.parametrize("mode", ["grey", "dark", "normal"])
def test_calibration_pattern_is_never_a_keyframe(mode):
    pattern = image(200)
    marker = cv2.aruco.generateImageMarker(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50), 7, 90)
    pattern[60:150, 90:180] = marker[..., None]
    rig = FakeRig(image(80), [pattern] * 4)
    cap = capture(rig, mode)
    assert not cap.verified
    assert "calibration pattern still visible" in cap.reasons


def test_printed_scale_card_is_not_a_projected_pattern():
    card = image(180)
    marker = cv2.aruco.generateImageMarker(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50), 40, 90)
    card[60:150, 90:180] = marker[..., None]
    rig = FakeRig(card, [card] * 2)
    cap = capture(rig, "normal")
    assert cap.verified, cap.reasons


def test_moving_hand_rejects_pair_then_uses_next_agreeing_pair():
    moved = image(160)
    cv2.circle(moved, (160, 100), 30, (20, 20, 20), -1)
    rig = FakeRig(image(100), [image(160), moved, image(162), image(162)])
    cap = capture(rig)
    assert cap.verified and cap.frames_seen == 4
    assert cap.t_capture == pytest.approx(10.266667)
    assert np.all(cap.image == 162)


def test_persistent_motion_is_unverified():
    a, b = image(160), image(160)
    cv2.circle(a, (100, 100), 25, (20, 20, 20), -1)
    cv2.circle(b, (220, 100), 25, (20, 20, 20), -1)
    cap = capture(FakeRig(image(100), [a, b, a, b]))
    assert not cap.verified
    assert any("frames disagree" in r for r in cap.reasons)


@pytest.mark.parametrize("value", [0, 255])
def test_bad_exposure_is_rejected(value):
    cap = capture(FakeRig(image(100), [image(value)] * 4), "dark")
    assert not cap.verified
    assert any("exposure:" in r for r in cap.reasons)


def test_both_members_of_pair_need_sane_exposure():
    cap = capture(FakeRig(image(100), [image(255), image(235)]), "dark", max_frames=2)
    assert not cap.verified
    assert any("exposure:" in r for r in cap.reasons)


def test_normal_mode_needs_no_page_or_overlay_in_crop():
    rig = FakeRig(image(100), [image(100)] * 2, acks=())
    cap = capture(rig, "normal")
    assert cap.verified and rig.modes == [] and cap.t_ack is None and cap.t_restore is None
    om = overlay()
    rig = FakeRig(image(100), [image(100)] * 4, acks=())
    cap = capture(rig, "normal", om)
    assert not cap.verified
    assert "overlay drawn inside the crop (mode normal)" in cap.reasons


def test_ack_timeout_still_restores_the_scene():
    rig = FakeRig(image(100), acks=(None, 10.3))
    om = overlay()
    assert capture(rig, om=om) is None
    assert rig.modes == ["grey", "normal"]
    assert om.clear_after() == 10.5


def test_camera_timeout_still_restores_and_returns_none():
    rig = FakeRig(image(100))
    assert capture(rig) is None
    assert rig.modes == ["grey", "normal"]


def test_one_frame_cannot_be_verified():
    cap = capture(FakeRig(image(100), [image(160)]))
    assert not cap.verified
    assert cap.reasons == ("only one frame after the page ACK",)


def test_restore_timeout_keeps_gate_masked_until_confirmed():
    rig = FakeRig(image(100), [image(160)] * 2, acks=(10.04, None))
    om = overlay()
    cap = capture(rig, om=om)
    assert cap.t_restore is None
    assert om.mask_at(100.0).all() and om.clear_after() == float("inf")
    with pytest.raises(RuntimeError, match="already in progress"):
        capture(rig, om=om)
    om.release(10.4)  # the caller has obtained a new normal-mode ACK
    assert om.clear_after() == pytest.approx(10.6)


def test_page_error_still_attempts_restore():
    rig = FakeRig(image(100), acks=(OSError("link failed"), 10.3))
    om = overlay()
    with pytest.raises(OSError, match="link failed"):
        capture(rig, om=om)
    assert rig.modes == ["grey", "normal"]
    assert om.clear_after() == 10.5


def test_stale_camera_contract_violation_never_uses_the_frame():
    rig = FakeRig(image(100))
    rig.wait_frame = lambda after_t, timeout: Frame(image(160), after_t - 0.01, 1)
    with pytest.raises(ValueError, match="stale frame"):
        capture(rig)
    assert rig.modes == ["grey", "normal"]


def test_previous_measurement_cannot_be_normal_reference():
    om = overlay()
    om.hold(9.5)
    om.release(9.8)  # normal frames after 10.0 only
    rig = FakeRig(image(160), [image(100), image(160), image(160)], [10.01, 10.17, 10.2])
    cap = capture(rig, om=om)
    assert cap.verified, cap.reasons
    assert rig.after[0] == pytest.approx(10.0)


def test_no_initial_frame_makes_no_page_changes():
    rig = FakeRig(None)
    assert capture(rig) is None
    assert rig.modes == []


def test_invalid_capture_parameters():
    rig = FakeRig(image(100))
    for args in ({"mode": "pattern"}, {"settle_ms": -1}, {"max_frames": 1}, {"crop": (400, 0, 20, 20)}):
        with pytest.raises(ValueError):
            capture(rig, **args)
    assert rig.modes == []


def flat_rig(lit):
    return FakeRig(image(100), [image(100)] * 2 + [lit] * 5,
                   [10.17, 10.20] + [10.36 + i / 30 for i in range(5)], acks=(10.04, 10.22, 10.52))


def test_flat_field_is_a_cropped_jpeg_of_five_agreeing_frames():
    lit = image(180)
    lit[:, :30] = 255  # room outside the selected work area is neither stored nor judged
    rig = flat_rig(lit)
    result = mf.capture_flat_field(rig, rig, crop_xywh=CROP, clock=rig.clock)
    assert result.verified, result.reasons
    assert result.crop_xywh == CROP
    assert rig.modes == ["dark", "grey", "normal"]
    assert len(rig.returned) == 7
    decoded = cv2.imdecode(np.frombuffer(result.jpeg, np.uint8), cv2.IMREAD_COLOR)
    assert decoded.shape == (180, 240, 3) and np.max(np.abs(decoded.astype(np.int16) - 180)) <= 1


def test_flat_field_accepts_smooth_illumination_gradient():
    lit = np.broadcast_to(np.linspace(140, 210, 320).astype(np.uint8)[None, :, None], (240, 320, 3)).copy()
    rig = flat_rig(lit)
    result = mf.capture_flat_field(rig, rig, crop_xywh=CROP, clock=rig.clock)
    assert result.verified, result.reasons


def test_flat_field_rejects_objects_and_projector_off():
    dirty = image(180)
    cv2.rectangle(dirty, (90, 80), (220, 140), (30, 30, 30), -1)
    for lit, reason in ((dirty, "board not flat"), (image(100), "page lit only")):
        rig = flat_rig(lit)
        result = mf.capture_flat_field(rig, rig, crop_xywh=CROP, clock=rig.clock)
        assert not result.verified
        assert any(reason in r for r in result.reasons)


def test_dark_flat_field_needs_no_second_mode():
    rig = FakeRig(image(100), [image(120)] * 5, acks=(10.04, 10.4))
    result = mf.capture_flat_field(rig, rig, "dark", crop_xywh=CROP, clock=rig.clock)
    assert result.verified, result.reasons
    assert rig.modes == ["dark", "normal"]


def test_flat_field_no_ack_and_too_few_frames():
    rig = FakeRig(image(100), acks=(None, 10.2))
    assert mf.capture_flat_field(rig, rig, crop_xywh=CROP, clock=rig.clock) is None
    assert rig.modes == ["dark", "normal"]
    rig = FakeRig(image(100), [image(120)], acks=(10.04, 10.4))
    result = mf.capture_flat_field(rig, rig, "dark", crop_xywh=CROP, clock=rig.clock)
    assert not result.verified and any("only 1 of 5" in r for r in result.reasons)


def test_flat_field_invalid_parameters():
    rig = FakeRig(image(100))
    for args in ({"mode": "normal"}, {"n": 1}, {"settle_ms": -1}):
        with pytest.raises(ValueError):
            mf.capture_flat_field(rig, rig, crop_xywh=CROP, **args)


def test_rig_m2_reports_saved_crop_content_and_timings(tmp_path, monkeypatch, no_network):
    rig = FakeRig(image(100), [image(160)] * 2)
    om = OverlayMask(SIZE, np.eye(3), SIZE)
    monkeypatch.setattr(mf.time, "monotonic", rig.clock)
    # measure_frame's default clock is bound at import, so make the call use the fake clock explicitly.
    real_measure = mf.measure_frame
    monkeypatch.setattr(mf, "measure_frame", lambda *a, **kw: real_measure(*a, clock=rig.clock, **kw))
    report = mf._rig_m2(rig, rig, om, CROP, ["grey"], 1, tmp_path)
    assert report["passed"]
    mode = report["modes"][0]
    assert mode["verified"] == 1 and mode["blank_p95_ms"] == pytest.approx(230)
    saved = cv2.imread(str(tmp_path / mode["samples"][0]["file"]))
    assert saved.shape == (180, 240, 3)


def test_hardware_harness_help_is_offline(capsys, no_network):
    with pytest.raises(SystemExit) as exc:
        mf.main(["--help"])
    assert exc.value.code == 0
    text = capsys.readouterr().out
    assert "--crop" in text and "--count" in text and "m1" in text and "m2" in text


def test_hardware_harness_never_defaults_to_full_camera(tmp_path, fixtures_dir, monkeypatch, capsys, no_network):
    import json

    from lightwork.vision import capture as camera

    calib = json.loads((fixtures_dir / "contracts" / "C7-calibration.json").read_text())
    calib["flat_field_crop_xywh"] = None
    path = tmp_path / "calib.json"
    path.write_text(json.dumps(calib))

    def forbidden(*args, **kwargs):
        raise AssertionError("camera must not open before a work-area crop is selected")

    monkeypatch.setattr(camera, "open_camera", forbidden)
    assert mf.main(["m1", "--calib", str(path)]) == 2
    assert "no work-area crop" in capsys.readouterr().err
