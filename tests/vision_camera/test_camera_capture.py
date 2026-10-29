"""LiveCamera semantics (latest / read / wait_frame / close), file sources, reconnect + lock re-apply, resolution."""

from __future__ import annotations

import itertools
import queue
import threading
import time

import cv2
import numpy as np
import pytest

from lightwork.vision import capture
from lightwork.vision.camera import (
    LOCK_ORDER,
    VideoDevice,
    device_for_node,
    find_c920,
    list_devices,
)
from lightwork.vision.capture import (
    CameraNotFound,
    DeviceBackend,
    LiveCamera,
    open_camera,
)
from lightwork.vision.source import FrameSource


def image_dir(tmp_path, n=5, name="frames"):
    d = tmp_path / name
    d.mkdir()
    for i in range(n):
        cv2.imwrite(str(d / f"{i:03d}.png"), np.full((24, 32, 3), i * 10, np.uint8))
    (d / "notes.txt").write_text("ignored")
    return d


class ScriptedBackend:
    """Frames pushed by the test: ``push(value, t)``; ``end()`` finishes the source."""

    reconnect = False
    controlled = True
    status = "scripted"

    def __init__(self):
        self.q: queue.Queue = queue.Queue()
        self.released = False

    def open(self):
        return np.zeros((2, 2, 3), np.uint8)

    def push(self, value: int, t: float):
        self.q.put((np.full((2, 2, 3), value, np.uint8), t, None))

    def end(self):
        self.q.put(None)

    def grab(self):
        return self.q.get()

    def interrupt(self):
        self.q.put(None)

    def release(self):
        self.released = True

    def describe(self):
        return {"device_id": "scripted", "mode": "test", "controls": {}}


def wait_until(cond, timeout=2.0):
    end = time.monotonic() + timeout
    while not cond():
        assert time.monotonic() < end, "condition not reached"
        time.sleep(0.002)


def test_image_dir_source_paced_and_ends(tmp_path):
    cam = open_camera(str(image_dir(tmp_path)), fps=50)
    try:
        assert isinstance(cam, FrameSource) and cam.controlled
        assert cam.latest() is not None  # first frame is there as soon as open returns
        frames = []
        while (f := cam.read()) is not None:
            frames.append(f)
        assert cam.read() is None
        idx = [f.idx for f in frames]
        assert idx == sorted(set(idx)) and idx[-1] == 4
        ts = [f.t_mono for f in frames]
        assert all(b > a for a, b in itertools.pairwise(ts))
        if len(frames) == 5:  # paced at 50 fps (0.02 s); lower bound only, CI may be slow
            assert ts[-1] - ts[0] >= 4 * 0.02 * 0.8
        assert [int(f.image[0, 0, 0]) for f in frames] == [i * 10 for i in idx]
        d = cam.describe()
        assert d == {"device_id": "file:frames", "mode": "file 32x24", "controls": {}}
        assert cam.wait_frame(time.monotonic() + 10, 5.0) is None  # ended: returns at once, no waiting
    finally:
        cam.close()


def test_loop_keeps_going_and_t_mono_is_laptop_clock(tmp_path):
    t0 = time.monotonic()
    with open_camera(str(image_dir(tmp_path, n=3)), fps=200, loop=True) as cam:
        frames = [cam.read() for _ in range(10)]
        assert all(f is not None for f in frames)
        assert frames[-1].idx >= 9
        assert all(t0 <= f.t_mono <= time.monotonic() for f in frames)
    assert cam.read() is None


def test_video_file_source(tmp_path):
    path = tmp_path / "clip.avi"
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 100.0, (32, 24))
    assert w.isOpened()
    for i in range(6):
        w.write(np.full((24, 32, 3), i * 40, np.uint8))
    w.release()
    with open_camera(str(path)) as cam:
        n = 0
        while cam.read() is not None:
            n += 1
        assert n >= 1 and cam.latest().idx == 5
        assert cam.describe()["device_id"] == "file:clip.avi"


def test_read_returns_newest_and_wait_frame_returns_first_after():
    b = ScriptedBackend()
    cam = LiveCamera(b)
    try:
        base = time.monotonic()
        for i, dt in enumerate((0.1, 0.2, 0.3), start=1):
            b.push(i, base + dt)
        wait_until(lambda: cam.latest().idx == 3)
        f = cam.read()
        assert f.idx == 3 and f.image[0, 0, 0] == 3  # newest; frames 1-2 skipped
        assert cam.wait_frame(base + 0.15, 0.0).idx == 2  # first frame after the given time, not the newest
        assert cam.wait_frame(base + 0.05, 0.0).idx == 1
        # blocking: nothing newer yet -> waits for the next push
        got = {}
        th = threading.Thread(target=lambda: got.update(f=cam.wait_frame(base + 0.3, 2.0)))
        th.start()
        time.sleep(0.05)
        assert th.is_alive()
        b.push(4, base + 0.4)
        th.join(2)
        assert got["f"].idx == 4
        # timeout
        t = time.monotonic()
        assert cam.wait_frame(base + 99, 0.05) is None
        assert 0.04 <= time.monotonic() - t < 1.0
        # read() after a read blocks until a newer frame
        assert cam.read().idx == 4
        th = threading.Thread(target=lambda: got.update(r=cam.read()))
        th.start()
        time.sleep(0.05)
        assert th.is_alive()
        b.push(5, base + 0.5)
        th.join(2)
        assert got["r"].idx == 5
    finally:
        cam.close()


def test_ring_is_bounded():
    b = ScriptedBackend()
    with LiveCamera(b) as cam:
        base = time.monotonic()
        for i in range(1, 20):
            b.push(i, base + i)
        wait_until(lambda: cam.latest().idx == 19)
        oldest = cam.wait_frame(-1.0, 0.0)
        assert oldest.idx == 19 - capture.RING_LEN + 1


def test_close_wakes_blocked_reader_and_is_idempotent():
    b = ScriptedBackend()
    cam = LiveCamera(b)
    assert cam.read().idx == 0
    got = {}
    th = threading.Thread(target=lambda: got.update(r=cam.read(), w=cam.wait_frame(time.monotonic() + 9, 30)))
    th.start()
    time.sleep(0.05)
    t = time.monotonic()
    cam.close()
    th.join(2)
    assert not th.is_alive() and time.monotonic() - t < 1.0
    assert got == {"r": None, "w": None}
    assert b.released and not cam._thread.is_alive()
    cam.close()
    assert cam.latest().idx == 0  # the last frame stays readable


def test_end_of_source_wakes_readers():
    b = ScriptedBackend()
    with LiveCamera(b) as cam:
        cam.read()
        got = {}
        th = threading.Thread(target=lambda: got.update(r=cam.read()))
        th.start()
        b.end()
        th.join(2)
        assert got == {"r": None}


def c920(tmp_path) -> VideoDevice:
    return VideoDevice(path=str(tmp_path / "fake-node"), node=str(tmp_path / "fake-node"),
                       by_id="usb-046d_HD_Pro_Webcam_C920_ABCDEF-video-index0")


def test_device_lock_applied_and_reapplied_after_reopen(tmp_path, monkeypatch, fake_v4l2, fake_capture):
    monkeypatch.setattr(capture, "REOPEN_INTERVAL_S", 0.01)
    fake = fake_v4l2()
    caps = fake_capture([5, None, None, 10**9], on_open=lambda n: fake.reset() if n == 2 else None)
    # 5 frames, unplug (controls drop to factory defaults, as after a power cycle), two failed reopens, back
    lock = {s.name: s.default for s in LOCK_ORDER}
    backend = DeviceBackend(c920(tmp_path), width=1920, height=1080, fps=30, lock=lock, run=fake.run, capture_factory=caps)
    with LiveCamera(backend) as cam:
        assert cam.controlled and cam.status.startswith("controlled camera")
        assert caps.props[cv2.CAP_PROP_BUFFERSIZE] == 1 and caps.props[cv2.CAP_PROP_FPS] == 30
        assert caps.props[cv2.CAP_PROP_FRAME_WIDTH] == 1920
        assert caps.props[cv2.CAP_PROP_FOURCC] == cv2.VideoWriter_fourcc(*"MJPG")
        assert caps.opened[0] == (str(tmp_path / "fake-node"), cv2.CAP_V4L2)
        wait_until(lambda: cam.stats()["reopens"] == 1)
        wait_until(lambda: cam.read() is not None and cam.stats()["frames"] > 8)
        s = cam.stats()
        assert s["reopens"] == 1 and s["age_ms_p50"] >= 5
        d = cam.describe()
        assert d["device_id"] == "usb-046d_HD_Pro_Webcam_C920_ABCDEF-video-index0"
        assert d["mode"] == "MJPG 64x48@30"
        assert d["controls"]["auto_exposure"] == 1 and d["controls"]["exposure_time_absolute"] == 167
    assert written_count(fake, "auto_exposure") == 2  # first open + the reopen
    assert len(caps.opened) == 4


def written_count(fake, name: str) -> int:
    return sum(1 for n, _ in fake.writes if n == name)


def test_device_lock_mismatch_makes_camera_uncontrolled(tmp_path, fake_v4l2, fake_capture):
    fake = fake_v4l2(names="old", ignore=("exposure_absolute",))
    lock = {s.name: s.default for s in LOCK_ORDER}
    backend = DeviceBackend(c920(tmp_path), width=1920, height=1080, fps=30, lock=lock, run=fake.run,
                            capture_factory=fake_capture([10**9]))
    with LiveCamera(backend) as cam:
        assert not cam.controlled
        assert cam.status == "camera lock incomplete: exposure_time_absolute=250 (want 167)"
        assert cam.report.name_set == "old"


def test_uncontrolled_device_writes_nothing(tmp_path, fake_v4l2, fake_capture):
    fake = fake_v4l2()
    dev = VideoDevice(path=str(tmp_path / "n"), node=str(tmp_path / "n"), by_id="usb-Generic_Laptop_Camera-video-index0")
    backend = DeviceBackend(dev, width=1920, height=1080, fps=30, lock=None, uncontrolled_reason="not a C920",
                            run=fake.run, capture_factory=fake_capture([10**9]))
    with LiveCamera(backend) as cam:
        assert not cam.controlled and cam.status == "uncontrolled camera: not a C920"
        assert not fake.writes


def test_device_open_failure_raises(tmp_path, fake_capture):
    backend = DeviceBackend(c920(tmp_path), width=1920, height=1080, fps=30, lock=None, capture_factory=fake_capture([None]))
    with pytest.raises(OSError, match="cannot open camera"):
        LiveCamera(backend)


def by_id_tree(tmp_path):
    dev = tmp_path / "dev"
    by_id = dev / "v4l" / "by-id"
    by_id.mkdir(parents=True)
    for n in (0, 1, 2, 3):
        (dev / f"video{n}").touch()
    (by_id / "usb-046d_HD_Pro_Webcam_C920_ABCDEF-video-index0").symlink_to(dev / "video2")
    (by_id / "usb-046d_HD_Pro_Webcam_C920_ABCDEF-video-index1").symlink_to(dev / "video3")
    (by_id / "usb-Generic_Laptop_Camera_2009-video-index0").symlink_to(dev / "video0")
    return dev, by_id


def test_device_resolution(tmp_path):
    dev, by_id = by_id_tree(tmp_path)
    assert [d.by_id for d in list_devices(by_id)] == [
        "usb-046d_HD_Pro_Webcam_C920_ABCDEF-video-index0",
        "usb-Generic_Laptop_Camera_2009-video-index0",
    ]
    c = find_c920(by_id)
    assert c.is_c920 and c.node == str(dev / "video2") and c.path.startswith(str(by_id))
    assert device_for_node(str(dev / "video2"), by_id) == c  # a /dev/videoN path maps back to its by-id name
    other = device_for_node(str(dev / "video1"), by_id)
    assert other.by_id is None and not other.is_c920 and other.device_id == str(dev / "video1")
    assert find_c920(tmp_path / "missing") is None and list_devices(tmp_path / "missing") == []


def test_open_camera_errors(tmp_path):
    by_id_tree(tmp_path)
    if capture.sys.platform == "linux":
        with pytest.raises(CameraNotFound, match="no C920 under .*usb-Generic_Laptop_Camera_2009-video-index0"):
            _no_c920(tmp_path)
        with pytest.raises(CameraNotFound, match="does not exist"):
            open_camera("/dev/lightwork-no-such-video")
    with pytest.raises(CameraNotFound, match="no camera, video file or image directory"):
        open_camera(str(tmp_path / "nope.mp4"))
    (tmp_path / "empty").mkdir()
    with pytest.raises(CameraNotFound, match="no .* images"):
        open_camera(str(tmp_path / "empty"))


def _no_c920(tmp_path):
    by_id = tmp_path / "only-laptop"
    by_id.mkdir()
    (by_id / "usb-Generic_Laptop_Camera_2009-video-index0").symlink_to(tmp_path / "dev" / "video0")
    open_camera(None, by_id_dir=by_id)
