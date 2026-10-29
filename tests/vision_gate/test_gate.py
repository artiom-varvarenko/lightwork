from __future__ import annotations

import cv2
import numpy as np
import pytest

from lightwork.contracts import render
from lightwork.contracts.events import make_payload
from lightwork.contracts.layout import Layout
from lightwork.contracts.stepgraph import StepGraph
from lightwork.geometry import Regions
from lightwork.testing.synth import add_hand, well_plate
from lightwork.vision.gate import Gate
from lightwork.vision.overlay_mask import OverlayMask
from lightwork.vision.source import Frame

FPS = 30.0
WORK = (100, 60, 1080, 620)  # x 100..1180, y 60..680 of a 1280x720 camera
# page CSS px = (camera px - work origin) * 854/1080: the page covers the work area; 1 projector px = 1.26 camera px
SCALE = 854 / 1080
H_PAGE = np.array([[SCALE, 0.0, -100 * SCALE], [0.0, SCALE, -60 * SCALE], [0.0, 0.0, 1.0]])
PAGE = (854.0, 490.0)


def bench():
    img, geom = well_plate({"A1": 1.0, "A2": 0.5, "B2": 0.25}, size=(1280, 720))
    sg = StepGraph.model_validate({
        "id": "p", "title": "p", "source": {"kind": "text", "ref": "x", "licence": "x", "attribution": "x"},
        "objects": [
            {"id": "plate", "noun": "well plate", "kind": "grid", "grid": {"rows": 8, "cols": 12, "naming": "A1"}},
            {"id": "tray", "noun": "tray", "kind": "zone"},
            {"id": "rack", "noun": "tube rack", "kind": "item"},
            {"id": "shelf", "noun": "shelf", "kind": "zone"},
        ],
        "steps": [{"id": "s", "say": "x", "checks": [{"id": "c", "kind": "confirm"}]}],
    })
    lay = Layout.model_validate({
        "pack": "p", "created": "x", "work_area_xywh": list(WORK),
        "objects": {
            "plate": {"kind": "grid", "corners_cam_px": geom.corners.tolist(), "sam": {"mode": "noun", "noun": "well plate"}},
            "tray": {"kind": "zone", "polygon_cam_px": [[470, 600], [600, 600], [600, 675], [470, 675]], "sam": {"mode": "noun", "noun": "tray"}},
            "rack": {"kind": "item", "corners_cam_px": [[960, 100], [1100, 100], [1100, 200], [960, 200]], "sam": {"mode": "noun", "noun": "tube rack"}},
            "shelf": {"kind": "zone", "polygon_cam_px": [[1200, 300], [1270, 300], [1270, 400], [1200, 400]], "sam": {"mode": "noun", "noun": "shelf"}},
        },
    })
    return img, Regions(lay, sg)


def noisy(img, rng, sd=3.0, jpeg_q=80):
    out = np.clip(img.astype(np.float32) + rng.normal(0, sd, img.shape), 0, 255).astype(np.uint8)
    ok, buf = cv2.imencode(".jpg", out, [cv2.IMWRITE_JPEG_QUALITY, jpeg_q])  # the C920 streams MJPEG
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)


def flicker(img, k, amp=0.06, glob=0.04, period_rows=240.0, roll=0.37):
    """Rolling-shutter banding (mains or DLP light vs the camera's frame rate: the bands roll by ``roll`` of a period
    per frame, not commensurate with the gate's 3-frame lag) plus a global gain wobble."""
    y = np.arange(img.shape[0], dtype=np.float32)[:, None, None]
    gain = (1 + amp * np.sin(2 * np.pi * (y / period_rows + roll * k))) * (1 + glob * np.sin(1.7 * k))
    return np.clip(img.astype(np.float32) * gain, 0, 255).astype(np.uint8)


def run(gate, images, t0=0.0, masks=None):
    out = []
    for i, im in enumerate(images):
        t = t0 + i / FPS
        m = masks(t) if masks is not None else None
        out += [(t, ev) for ev in gate.update(Frame(im, t, i), m)]
    return out


def hand_path(start, end, n):
    return [(round(start[0] + (end[0] - start[0]) * i / (n - 1)), round(start[1] + (end[1] - start[1]) * i / (n - 1))) for i in range(n)]


def hand_mask(tip):
    return (add_hand(np.zeros((720, 1280, 3), np.uint8), tip) > 0).any(axis=2)


def test_events_validate_as_c1_motion_payloads():
    img, reg = bench()
    rng = np.random.default_rng(0)
    path = hand_path((530, 700), (530, 320), 30)
    frames = [noisy(img, rng) for _ in range(20)] + [noisy(add_hand(img, p), rng) for p in path] + [noisy(add_hand(img, path[-1]), rng) for _ in range(30)]
    events = run(Gate(reg, WORK), frames)
    assert events
    for _, (type_, payload) in events:
        assert type_ == "motion"
        assert make_payload("motion", payload) == payload
        assert payload["active_regions"] == sorted(payload["active_regions"])


@pytest.mark.parametrize("sd", [2.0, 5.0])
def test_sensor_noise_and_mjpeg_only_settle_once(sd):
    img, reg = bench()
    rng = np.random.default_rng(1)
    events = run(Gate(reg, WORK), [noisy(img, rng, sd) for _ in range(150)])
    assert [e for _, e in events] == [("motion", {"state": "settled", "active_regions": []})]
    assert events[0][0] == pytest.approx(0.6, abs=1 / FPS)


def test_rolling_flicker_is_not_motion():
    img, reg = bench()
    rng = np.random.default_rng(2)
    frames = [noisy(flicker(img, k), rng) for k in range(150)]
    raw = np.abs(frames[10].astype(np.int16) - frames[13].astype(np.int16)).max(axis=2) > 12
    assert raw.mean() > 0.2  # without gain correction most of the bench would count as changed
    events = run(Gate(reg, WORK), frames)
    assert [e for _, e in events] == [("motion", {"state": "settled", "active_regions": []})]


@pytest.mark.parametrize("with_flicker", [False, True])
def test_hand_over_cells_gives_the_cells_it_moved_over(with_flicker):
    img, reg = bench()
    rng = np.random.default_rng(3)
    path = hand_path((530, 700), (530, 320), 30)  # fingertip ends on D6's centre
    scene = [img] * 20 + [add_hand(img, p) for p in path] + [add_hand(img, path[-1])] * 30
    frames = [noisy(flicker(s, k) if with_flicker else s, rng) for k, s in enumerate(scene)]
    events = run(Gate(reg, WORK), frames)
    states = [e[1]["state"] for _, e in events]
    assert states[0] == "settled" and states[1] == "moving" and states[-1] == "settled"
    final = set(events[-1][1][1]["active_regions"])
    assert {"plate", "plate:D6", "plate:E6", "plate:H6", "tray"} <= final
    swept = np.zeros((720, 1280), np.uint8)
    for p in path:
        swept |= hand_mask(p)
    swept = cv2.dilate(swept, np.ones((17, 17), np.uint8)) > 0
    for ref in reg.all_refs():
        if not swept[reg.mask(ref, (720, 1280)) > 0].any():
            assert ref not in final, ref  # the hand never came near it
    assert "rack" not in final and "shelf" not in final
    # moving events only grow the set, and the settled event carries the union
    sets = [set(e[1]["active_regions"]) for _, e in events if e[1]["state"] == "moving"]
    assert all(a < b for a, b in zip(sets, sets[1:]))
    assert sets[-1] <= final


@pytest.mark.parametrize("leave", [False, True])
def test_settle_latency_at_30fps(leave):
    img, reg = bench()
    rng = np.random.default_rng(4)
    path = hand_path((900, 700), (700, 400), 25)
    scene = [img] * 20 + [add_hand(img, p) for p in path]
    scene += [img if leave else add_hand(img, path[-1])] * 40
    last_change = (20 + len(path) - (0 if leave else 1)) / FPS  # first frame of the final, still scene
    events = run(Gate(reg, WORK), [noisy(s, rng) for s in scene])
    settled = [t for t, e in events if e[1]["state"] == "settled" and t > 1.0]
    assert len(settled) == 1
    latency = settled[0] - last_change
    assert 0.6 - 1e-6 <= latency <= 0.8


def test_motion_outside_the_work_area_is_ignored():
    img, reg = bench()
    rng = np.random.default_rng(5)
    path = hand_path((1245, 700), (1245, 250), 30)  # palm stays right of x = 1185; the work area ends at 1180
    frames = [noisy(img, rng) for _ in range(20)] + [noisy(add_hand(img, p), rng) for p in path] + [noisy(img, rng)] * 5
    events = run(Gate(reg, WORK), frames)
    assert [e[1]["state"] for _, e in events] == ["settled"]


def test_moving_events_are_rate_limited_and_still_frames_silent():
    img, reg = bench()
    rng = np.random.default_rng(6)
    path = hand_path((250, 700), (870, 330), 60)  # 2 s diagonal across many cells
    scene = [img] * 150 + [add_hand(img, p) for p in path] + [add_hand(img, path[-1])] * 30
    events = run(Gate(reg, WORK), [noisy(s, rng) for s in scene])
    moving = [t for t, e in events if e[1]["state"] == "moving"]
    assert [e[1]["state"] for _, e in events].count("settled") == 2
    assert 2 <= len(moving) <= 1 + 2.0 * 4 + 1
    assert all(b - a >= 0.25 - 1e-6 for a, b in zip(moving, moving[1:]))


def overlay_frame(img, prims, pulse_phase, offset=(2, -1)):
    """What the camera sees of the projected overlay, drawn independently of OverlayMask: saturated light on the
    bench, offset by a calibration error, blurred edges, pulsing rings at 1..1.6x their stroke."""
    layer = np.zeros(img.shape[:2], np.uint8)
    stroke = 5 / SCALE  # 5 projector px = 5 CSS px here, in camera px
    for p in prims:
        if p.kind == "ring":
            w = stroke * (1 + 0.6 * pulse_phase if p.pulse else 1)
            cv2.circle(layer, (round(p.c[0]) + offset[0], round(p.c[1]) + offset[1]), round(p.r), 255, max(1, round(w)), cv2.LINE_AA)
        elif p.kind == "arrow":
            a = (round(p.a[0]) + offset[0], round(p.a[1]) + offset[1])
            b = (round(p.b[0]) + offset[0], round(p.b[1]) + offset[1])
            cv2.line(layer, a, b, 255, round(stroke), cv2.LINE_AA)
            angle = np.arctan2(a[1] - b[1], a[0] - b[0])
            for side in (-0.5, 0.5):
                tip = (round(b[0] + 4 * stroke * np.cos(angle + side)), round(b[1] + 4 * stroke * np.sin(angle + side)))
                cv2.line(layer, b, tip, 255, round(stroke), cv2.LINE_AA)
        elif p.kind == "dot":
            cv2.circle(layer, (round(p.c[0]) + offset[0], round(p.c[1]) + offset[1]), round(1.5 * stroke), 255, -1, cv2.LINE_AA)
    a = cv2.GaussianBlur(layer, (5, 5), 1.2).astype(np.float32)[..., None] / 255.0
    light = np.array([40, 60, 255], np.float32)
    return np.clip(img * (1 - a) + light * a, 0, 255).astype(np.uint8)


def animated_scenes(n=20, dt=0.2):
    out = []
    for i in range(n):
        col = i % 10
        prims = [
            render.Ring(id="t", c=[290.0 + 60 * col, 320.0], r=36.0, role="target", pulse=True),
            render.Ring(id="s", c=[230.0, 140.0 + 60 * (i % 8)], r=36.0, role="source"),
            render.Dot(id="d", c=[300.0 + 40 * i, 620.0], role="info"),
        ]
        if i % 2:
            prims.append(render.Arrow(id="a", a=[230.0, 140.0 + 60 * (i % 8)], b=[290.0 + 60 * col, 320.0], role="info"))
        out.append((1.0 + i * dt, render.SceneMsg(seq=i + 1, mode="normal", prims=prims)))
    return out


@pytest.mark.parametrize("lag_s", [0.07, 0.15])
def test_projected_overlay_changes_are_not_motion(lag_s):
    img, reg = bench()
    rng = np.random.default_rng(7)
    scenes = animated_scenes()
    om = OverlayMask((1280, 720), H_PAGE, PAGE)
    frames = []
    for k in range(int(6.0 * FPS)):
        t = k / FPS
        shown = [s for ts, s in scenes if ts + lag_s <= t]  # the camera sees a scene lag_s after it was sent
        im = overlay_frame(img, shown[-1].prims, 0.5 + 0.5 * np.sin(2 * np.pi * 2 * t)) if shown else img
        frames.append(noisy(im, rng))
    next_scene = 0

    def mask_at(t):
        nonlocal next_scene
        while next_scene < len(scenes) and scenes[next_scene][0] <= t:
            sent, scene = scenes[next_scene]
            om.set_scene(scene, sent)
            next_scene += 1
        return om.mask_at(t)

    events = run(Gate(reg, WORK), frames, masks=mask_at)
    assert [e for _, e in events] == [("motion", {"state": "settled", "active_regions": []})]
    control = run(Gate(reg, WORK), frames)  # same frames without the mask: the overlay would look like motion
    assert any(e[1]["state"] == "moving" for _, e in control)


def test_measurement_mode_frames_are_skipped():
    img, reg = bench()
    rng = np.random.default_rng(8)
    om = OverlayMask((1280, 720), H_PAGE, PAGE)
    om.set_scene(render.SceneMsg(seq=1, mode="normal", prims=[]), 0.0)
    om.hold(2.0)
    om.release(2.3)
    grey = np.clip(img.astype(np.float32) * 1.6 + 30, 0, 255).astype(np.uint8)  # the page lights the bench
    frames = [noisy(grey if 2.05 <= k / FPS < 2.35 else img, rng) for k in range(int(4 * FPS))]
    gate = Gate(reg, WORK)
    events = run(gate, frames, masks=om.mask_at)
    assert [e for _, e in events] == [("motion", {"state": "settled", "active_regions": []})]


def test_change_during_a_measurement_mode_is_seen_afterwards():
    img, reg = bench()
    rng = np.random.default_rng(9)
    om = OverlayMask((1280, 720), H_PAGE, PAGE)
    om.hold(2.0)
    om.release(2.3)
    moved = img.copy()
    cv2.rectangle(moved, (980, 120), (1080, 180), (40, 120, 200), -1)  # something put on the rack meanwhile
    frames = [noisy(img if k / FPS < 2.0 else moved, rng) for k in range(int(4 * FPS))]
    events = run(Gate(reg, WORK), frames, masks=om.mask_at)
    moving = [(t, e) for t, e in events if e[1]["state"] == "moving"]
    assert moving and moving[0][0] >= 2.5 - 1e-9  # first unmasked frame after release + hold_s
    assert "rack" in moving[0][1][1]["active_regions"]


def test_overlay_mask_must_match_the_frame_size():
    img, reg = bench()
    with pytest.raises(ValueError):
        Gate(reg, WORK).update(Frame(img, 0.0, 0), np.zeros((10, 10), np.uint8))


def test_without_regions_or_work_area():
    img, _ = bench()
    rng = np.random.default_rng(10)
    path = hand_path((530, 700), (530, 320), 30)
    frames = [noisy(img, rng) for _ in range(20)] + [noisy(add_hand(img, p), rng) for p in path] + [noisy(add_hand(img, path[-1]), rng)] * 25
    events = run(Gate(None, None), frames)
    assert [e[1]["state"] for _, e in events] == ["settled", "moving", "settled"]
    assert all(e[1]["active_regions"] == [] for _, e in events)
