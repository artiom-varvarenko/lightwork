"""Live capture: one thread per camera feeding a latest-frame slot; :func:`open_camera` picks the source.

``Frame.t_mono`` is ``time.monotonic()`` when the frame was dequeued (``read()`` returned), for live devices and for
file sources alike, so frame times compare directly with page ACK times. A short ring of recent frames lets
:meth:`LiveCamera.wait_frame` return the *first* frame after a given time rather than only the newest one.

Bench and rig checks (``briefs/rig.md`` C0, PLAN §7 A)::

    uv run python -m lightwork.vision.capture                 # open, lock, print controls read back vs requested
    uv run python -m lightwork.vision.capture --bench 300     # 30.0+-1 fps over 5 min; capture-to-available <= 1 frame
    uv run python -m lightwork.vision.capture --c0            # rig:C0: lock, read back, replug, read back
    uv run python -m lightwork.vision.capture --focus-sweep --save-lock   # pick focus_absolute at rig height
"""

from __future__ import annotations

import argparse
import logging
import sys
import threading
import time
from collections import deque
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Protocol, Self

import cv2
import numpy as np

from .camera import (
    BY_ID_DIR,
    ControlReport,
    Runner,
    V4l2Ctl,
    VideoDevice,
    apply_lock,
    compare_lock,
    device_for_node,
    find_c920,
    have_v4l2_ctl,
    list_devices,
    load_lock,
    lock_path,
    save_lock,
    subprocess_runner,
)
from .source import Frame, FrameSource, ImageSequenceSource, VideoFileSource

log = logging.getLogger(__name__)

RING_LEN = 8  # ~0.27 s at 30 fps: enough for the keyframe choreography (ACK + 120 ms, then 2 frames)
REOPEN_INTERVAL_S = 0.5
READ_TIMEOUT_MS = 2000
IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".bmp")


class CameraNotFound(RuntimeError):
    pass


class Backend(Protocol):
    reconnect: bool  # True: a failed grab means "lost, reopen"; False: it means end of source
    controlled: bool
    status: str

    def open(self) -> np.ndarray: ...  # (re)open and return the first image; OSError on failure

    def grab(self) -> tuple[np.ndarray, float, float | None] | None: ...  # (image, t_mono, driver age s) or None

    def interrupt(self) -> None: ...  # make a blocked grab() return soon (called from another thread)

    def release(self) -> None: ...

    def describe(self) -> dict: ...


class DeviceBackend:
    """An OpenCV capture device. With a ``lock`` (Linux C920), v4l2 controls are applied after every (re)open."""

    reconnect = True

    def __init__(
        self,
        device: VideoDevice | int,
        *,
        width: int,
        height: int,
        fps: int,
        lock: Mapping[str, int | None] | None,
        uncontrolled_reason: str = "",
        api: int = cv2.CAP_V4L2,
        run: Runner = subprocess_runner,
        capture_factory: Callable[..., cv2.VideoCapture] = cv2.VideoCapture,
    ):
        self.device = device if isinstance(device, VideoDevice) else None
        self._index = device if isinstance(device, int) else None
        self._size = (width, height)
        self._fps = fps
        self._lock = dict(lock) if lock is not None else None
        self._api = api
        self._run = run
        self._factory = capture_factory
        self._cap: cv2.VideoCapture | None = None
        self.mode = ""
        self.report: ControlReport | None = None
        self._uncontrolled_reason = uncontrolled_reason
        self._readback: dict[str, int] = {}

    @property
    def controlled(self) -> bool:
        return self._lock is not None and self.report is not None and self.report.ok

    @property
    def status(self) -> str:
        if self._lock is None:
            return f"uncontrolled camera: {self._uncontrolled_reason}"
        if self.report is None:
            return "camera lock not applied yet"
        if not self.report.ok:
            bad = ", ".join(f"{r.name}={r.readback} (want {r.value})" for r in self.report.mismatches)
            return f"camera lock incomplete: {bad}"
        return f"controlled camera ({self.report.name_set} control names)"

    def _target(self) -> str | int:
        if self.device is None:
            return self._index if self._index is not None else 0
        return self.device.node_now()

    def open(self) -> np.ndarray:
        self.release()
        target = self._target()
        cap = self._factory(target, self._api)
        if not cap.isOpened():
            cap.release()
            raise OSError(f"cannot open camera {target}")
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))  # before the size
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._size[0])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._size[1])
        cap.set(cv2.CAP_PROP_FPS, self._fps)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if self._api == cv2.CAP_V4L2:
            cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, READ_TIMEOUT_MS)
        ok, img = cap.read()  # starts the stream; controls are applied after this (briefs/rig.md §3.1)
        if not ok or img is None:
            cap.release()
            raise OSError(f"camera {target} opened but returned no frame")
        fourcc = int(cap.get(cv2.CAP_PROP_FOURCC)).to_bytes(4, "little").decode("ascii", "replace").strip("\x00")
        w, h, f = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), cap.get(cv2.CAP_PROP_FPS)
        self.mode = f"{fourcc or '?'} {w}x{h}@{f:g}"
        self._cap = cap
        if isinstance(target, str) and self._lock is not None:
            self.report = apply_lock(V4l2Ctl(target, self._run), self._lock)
            if not self.report.ok:
                log.warning("%s", self.status)
        elif isinstance(target, str) and have_v4l2_ctl():
            try:
                self._readback = {k: c.value for k, c in V4l2Ctl(target, self._run).list_ctrls().items()}
            except OSError as e:
                log.warning("control readback failed: %s", e)
        return img

    def grab(self) -> tuple[np.ndarray, float, float | None] | None:
        cap = self._cap
        if cap is None:
            return None
        ok, img = cap.read()
        t = time.monotonic()
        if not ok or img is None:
            return None
        age = None
        if self._api == cv2.CAP_V4L2:
            ts_ms = cap.get(cv2.CAP_PROP_POS_MSEC)  # V4L2 buffer timestamp, CLOCK_MONOTONIC like time.monotonic()
            if ts_ms > 0:
                age = t - ts_ms / 1000.0
        return img, t, age

    def interrupt(self) -> None:
        pass  # a V4L2 read returns within one frame period or READ_TIMEOUT_MS

    def release(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def describe(self) -> dict:
        device_id = self.device.device_id if self.device else f"index:{self._index or 0}"
        controls = self.report.controls() if self.report is not None else dict(self._readback)
        return {"device_id": device_id, "mode": self.mode, "controls": controls}


class FileBackend:
    """A video file or image sequence played at its own frame times (real-time pacing), optionally looped."""

    reconnect = False
    controlled = True

    def __init__(self, make_source: Callable[[], FrameSource], *, name: str, loop: bool = False, period_s: float = 1 / 30):
        self._make = make_source
        self._name = name
        self._loop = loop
        self._period = period_s
        self._src: FrameSource | None = None
        self._stop = threading.Event()
        self._base_due = 0.0  # monotonic time at which the current pass's first frame was due
        self._t0_src = 0.0
        self._last_due = 0.0
        self._shape: tuple[int, ...] = ()
        self.status = f"file source {name}"

    def _start_pass(self, due: float) -> Frame:
        if self._src is not None:
            self._src.close()
        self._src = self._make()
        f = self._src.read()
        if f is None:
            raise OSError(f"{self._name}: no frames")
        self._base_due, self._t0_src, self._last_due = due, f.t_mono, due
        return f

    def open(self) -> np.ndarray:
        f = self._start_pass(time.monotonic())
        self._shape = f.image.shape
        return f.image

    def grab(self) -> tuple[np.ndarray, float, float | None] | None:
        if self._src is None:
            return None
        f = self._src.read()
        if f is None:
            if not self._loop:
                return None
            f = self._start_pass(self._last_due + self._period)
        due = self._base_due + (f.t_mono - self._t0_src)
        delay = due - time.monotonic()
        if delay > 0 and self._stop.wait(delay):
            return None
        self._last_due = due
        return f.image, time.monotonic(), None

    def interrupt(self) -> None:
        self._stop.set()

    def release(self) -> None:
        if self._src is not None:
            self._src.close()
            self._src = None

    def describe(self) -> dict:
        h, w = self._shape[:2] if self._shape else (0, 0)
        return {"device_id": f"file:{self._name}", "mode": f"file {w}x{h}", "controls": {}}


class LiveCamera:
    """Capture thread + latest-frame slot over a :class:`Backend`; satisfies ``vision.source.FrameSource``."""

    def __init__(self, backend: Backend):
        self._backend = backend
        self._cond = threading.Condition()
        self._ring: deque[Frame] = deque(maxlen=RING_LEN)
        self._next_idx = 0
        self._last_read_idx = -1
        self._ended = False
        self._stop = threading.Event()
        self._reopens = 0
        self._ages: deque[float] = deque(maxlen=10_000)
        self._gap_max = 0.0
        img = backend.open()
        self._publish(img, time.monotonic(), None)
        self._thread = threading.Thread(target=self._loop, name="lightwork-capture", daemon=True)
        self._thread.start()

    # -- FrameSource / interface A

    @property
    def controlled(self) -> bool:
        return self._backend.controlled

    @property
    def status(self) -> str:
        return self._backend.status

    @property
    def report(self) -> ControlReport | None:
        return getattr(self._backend, "report", None)

    @property
    def device(self) -> VideoDevice | None:
        return getattr(self._backend, "device", None)

    def read(self) -> Frame | None:
        """Block for the newest frame newer than the last ``read()``; None once the source ended or was closed."""
        with self._cond:
            while True:
                if self._ring and self._ring[-1].idx > self._last_read_idx:
                    f = self._ring[-1]
                    self._last_read_idx = f.idx
                    return f
                if self._ended:
                    return None
                self._cond.wait()

    def latest(self) -> Frame | None:
        with self._cond:
            return self._ring[-1] if self._ring else None

    def wait_frame(self, after_t: float, timeout: float) -> Frame | None:
        """First frame with ``t_mono > after_t``, waiting up to ``timeout`` s; None on timeout or end.

        If ``after_t`` is older than the ring, the oldest frame still held is the earliest one available."""
        deadline = time.monotonic() + timeout
        with self._cond:
            while True:
                for f in self._ring:
                    if f.t_mono > after_t:
                        return f
                remaining = deadline - time.monotonic()
                if self._ended or remaining <= 0:
                    return None
                self._cond.wait(remaining)

    def describe(self) -> dict:
        return self._backend.describe()

    def stats(self) -> dict:
        """Frames produced, reopens, max inter-frame gap and driver-to-app age percentiles (live V4L2 only)."""
        with self._cond:
            ages = np.array(self._ages, dtype=np.float64)
            out = {"frames": self._next_idx, "reopens": self._reopens, "gap_ms_max": round(self._gap_max * 1000, 1)}
        if ages.size:
            out["age_ms_p50"] = round(float(np.percentile(ages, 50)) * 1000, 1)
            out["age_ms_p95"] = round(float(np.percentile(ages, 95)) * 1000, 1)
        return out

    def close(self) -> None:
        self._stop.set()
        self._backend.interrupt()
        if self._thread.is_alive() and self._thread is not threading.current_thread():
            self._thread.join(timeout=READ_TIMEOUT_MS / 1000 + 1)
        self._backend.release()
        with self._cond:
            self._ended = True
            self._cond.notify_all()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # -- capture thread

    def _publish(self, img: np.ndarray, t: float, age: float | None) -> None:
        with self._cond:
            if self._ring:
                self._gap_max = max(self._gap_max, t - self._ring[-1].t_mono)
            self._ring.append(Frame(image=img, t_mono=t, idx=self._next_idx))
            self._next_idx += 1
            if age is not None:
                self._ages.append(age)
            self._cond.notify_all()

    def _reopen(self) -> bool:
        self._backend.release()
        while not self._stop.wait(REOPEN_INTERVAL_S):
            try:
                img = self._backend.open()
            except OSError as e:
                log.debug("reopen failed: %s", e)
                continue
            with self._cond:
                self._reopens += 1
                self._gap_max = 0.0  # the outage itself is reported by "reopens"
                self._ring.clear()
            self._publish(img, time.monotonic(), None)
            log.info("camera reopened (%s)", self._backend.status)
            return True
        return False

    def _loop(self) -> None:
        try:
            while not self._stop.is_set():
                try:
                    got = self._backend.grab()
                except Exception:  # a driver error must not leave readers blocked forever
                    log.exception("camera grab failed")
                    got = None
                if self._stop.is_set():
                    break
                if got is not None:
                    self._publish(*got)
                    continue
                if not self._backend.reconnect:
                    break
                log.warning("camera lost; reopening")
                if not self._reopen():
                    break
        finally:
            with self._cond:
                self._ended = True
                self._cond.notify_all()


# ---------------------------------------------------------------------------------------------------- open


def _image_files(d: Path) -> list[Path]:
    return sorted(p for p in d.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)


def open_camera(
    spec: str | None = None,
    *,
    width: int = 1920,
    height: int = 1080,
    fps: int = 30,
    loop: bool = False,
    lock: Mapping[str, int | None] | None = None,
    state_dir: Path | None = None,
    by_id_dir: Path = BY_ID_DIR,
) -> LiveCamera:
    """Open a camera or file source.

    ``spec``: None = the C920 found under ``/dev/v4l/by-id`` (index 0 on macOS/Windows); a by-id or ``/dev/videoN``
    path; a device index; a video file; or a directory of images (played at ``fps``). Only a C920 on Linux with
    ``v4l2-ctl`` gets its controls locked (``lock``, default :func:`camera.load_lock` from
    ``<state_dir>/camera.json``); every other camera runs on auto settings with ``controlled=False``.
    ``loop`` replays file sources forever instead of ending.
    """
    if spec is not None and not spec.startswith("/dev/") and not spec.isdigit():
        p = Path(spec)
        if p.is_dir():
            files = _image_files(p)
            if not files:
                raise CameraNotFound(f"{spec}: no {'/'.join(IMAGE_SUFFIXES)} images")
            return LiveCamera(
                FileBackend(lambda: ImageSequenceSource(files, fps=fps), name=p.name, loop=loop, period_s=1 / fps)
            )
        if p.is_file():
            return LiveCamera(FileBackend(lambda: VideoFileSource(p), name=p.name, loop=loop, period_s=1 / fps))
        raise CameraNotFound(f"no camera, video file or image directory at {spec!r}")

    if sys.platform != "linux":
        index = int(spec) if spec is not None and spec.isdigit() else 0
        if spec is not None and not spec.isdigit():
            raise CameraNotFound(f"{spec!r}: device paths are Linux-only; pass a camera index")
        reason = f"{sys.platform} has no v4l2 control path; auto exposure, white balance and focus"
        return LiveCamera(DeviceBackend(index, width=width, height=height, fps=fps, lock=None, uncontrolled_reason=reason, api=cv2.CAP_ANY))

    if spec is None:
        dev = find_c920(by_id_dir)
        if dev is None:
            found = ", ".join(d.device_id for d in list_devices(by_id_dir)) or "none"
            raise CameraNotFound(f"no C920 under {by_id_dir} (cameras found: {found}); pass a camera path or index explicitly")
    else:
        node = f"/dev/video{spec}" if spec.isdigit() else spec
        if not Path(node).exists():
            raise CameraNotFound(f"{node} does not exist")
        dev = device_for_node(node, by_id_dir)

    def backend(lock_: Mapping[str, int | None] | None, reason: str = "") -> DeviceBackend:
        return DeviceBackend(dev, width=width, height=height, fps=fps, lock=lock_, uncontrolled_reason=reason)

    if not dev.is_c920:
        return LiveCamera(backend(None, f"{dev.device_id} is not a C920; auto settings"))
    if not have_v4l2_ctl():
        return LiveCamera(backend(None, "v4l2-ctl not found (install v4l-utils); auto settings"))
    return LiveCamera(backend(dict(lock) if lock is not None else load_lock(lock_path(state_dir))))


# ---------------------------------------------------------------------------------------------------- rig tools


def _print_camera(cam: LiveCamera) -> None:
    d = cam.describe()
    print(f"device: {d['device_id']}  mode: {d['mode']}")
    print(f"status: {cam.status}")
    if cam.report is not None:
        for line in cam.report.lines():
            print(f"  {line}")


def bench(cam: LiveCamera, seconds: float, fps: float) -> int:
    """Average capture fps over ``seconds`` (pass: within +-1 of ``fps``) and driver-to-app age (pass: p95 below two
    frame periods, i.e. no frame ever waits in the driver queue behind a newer one)."""
    first = last = None
    skipped = 0
    t_end = time.monotonic() + seconds
    next_report = time.monotonic() + 10
    while time.monotonic() < t_end:
        f = cam.read()
        if f is None:
            print("source ended")
            break
        if last is not None:
            skipped += f.idx - last.idx - 1
        first = first or f
        last = f
        if f.t_mono >= next_report:
            next_report += 10
            print(f"  {f.t_mono - first.t_mono:6.0f} s  frames {f.idx - first.idx + 1}")
    if first is None or last is None or last.idx == first.idx:
        print("FAIL: no frames")
        return 1
    measured = (last.idx - first.idx) / (last.t_mono - first.t_mono)
    s = cam.stats()
    period_ms = 1000 / fps
    fps_ok = abs(measured - fps) <= 1.0 and s["reopens"] == 0
    age_p95 = s.get("age_ms_p95")
    age_ok = age_p95 is not None and age_p95 < 2 * period_ms
    print(f"fps {measured:.2f} (target {fps}+-1): {'ok' if fps_ok else 'FAIL'}; reopens {s['reopens']}, max gap {s['gap_ms_max']} ms")
    if age_p95 is None:
        print("capture-to-available: n/a (no driver timestamps on this backend)")
    else:
        print(
            f"capture-to-available: age p50 {s['age_ms_p50']} ms, p95 {age_p95} ms "
            f"(< {2 * period_ms:.1f} ms = at most 1 frame behind): {'ok' if age_ok else 'FAIL'}"
        )
    print(f"consumer skipped {skipped} frames (latest-wins slot)")
    return 0 if fps_ok and (age_ok or age_p95 is None) else 1


def rig_c0(cam: LiveCamera, lock: Mapping[str, int | None], wait_s: float = 120) -> int:
    """rig:C0: controls equal the lock after start and again after a replug (re-applied on reopen)."""
    if cam.device is None or cam.report is None:
        print(f"C0 needs a controlled C920 ({cam.status})")
        return 2
    ok1 = compare_lock(V4l2Ctl(cam.device.node_now()), lock)
    print("after start:", "ok" if ok1.ok else "MISMATCH")
    for line in ok1.lines():
        print(f"  {line}")
    print(f"Unplug the camera, wait ~5 s, plug it back in (waiting up to {wait_s:.0f} s)...")
    t_end = time.monotonic() + wait_s
    while cam.stats()["reopens"] == 0 and time.monotonic() < t_end:
        time.sleep(0.2)
    if cam.stats()["reopens"] == 0:
        print("FAIL: no replug seen")
        return 1
    f = cam.wait_frame(time.monotonic() + 2.0, 5.0)  # let the restarted stream run for 2 s before reading back
    if f is None:
        print("FAIL: no frames after replug")
        return 1
    ok2 = compare_lock(V4l2Ctl(cam.device.node_now()), lock)
    print("after replug:", "ok" if ok2.ok else "MISMATCH")
    for line in ok2.lines():
        print(f"  {line}")
    passed = ok1.ok and ok2.ok
    print("rig:C0", "PASS" if passed else "FAIL")
    return 0 if passed else 1


def focus_sweep(cam: LiveCamera, step: int = 5) -> int | None:
    """Sharpness (variance of the Laplacian, central half of the frame) per ``focus_absolute``; returns the best."""
    if cam.device is None or cam.report is None:
        print(f"focus sweep needs a controlled C920 ({cam.status})")
        return None
    ctl = V4l2Ctl(cam.device.node_now())
    info = ctl.list_ctrls().get("focus_absolute")
    if info is None or info.min is None or info.max is None:
        print("device has no focus_absolute")
        return None
    scores: dict[int, float] = {}
    for v in range(info.min, info.max + 1, max(step, info.step or 1)):
        if err := ctl.set("focus_absolute", v):
            print(f"focus_absolute={v}: {err}")
            continue
        f = cam.wait_frame(time.monotonic() + 0.6, 3.0)  # focus motor settle
        if f is None:
            break
        h, w = f.image.shape[:2]
        centre = cv2.cvtColor(f.image[h // 4 : 3 * h // 4, w // 4 : 3 * w // 4], cv2.COLOR_BGR2GRAY)
        scores[v] = float(cv2.Laplacian(centre, cv2.CV_64F).var())
        print(f"  focus_absolute {v:4d}: sharpness {scores[v]:9.1f}")
    if not scores:
        return None
    best = max(scores, key=lambda k: scores[k])
    ctl.set("focus_absolute", best)
    print(f"best focus_absolute = {best}")
    return best


def _parse_override(s: str) -> tuple[str, int]:
    name, sep, value = s.partition("=")
    if not sep:
        raise argparse.ArgumentTypeError(f"expected NAME=VALUE, got {s!r}")
    return name.strip(), int(value)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m lightwork.vision.capture", description="Camera bench and rig checks.")
    ap.add_argument("--camera", default=None, help="by-id path, /dev/videoN, index, video file or image dir (default: C920)")
    ap.add_argument("--width", type=int, default=1920)
    ap.add_argument("--height", type=int, default=1080)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("-c", "--control", action="append", type=_parse_override, default=[], help="override a lock value, NAME=VALUE")
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--bench", type=float, metavar="SECONDS", help="measure fps and capture-to-available latency")
    mode.add_argument("--c0", action="store_true", help="rig:C0 lock / replug / read-back test")
    mode.add_argument("--focus-sweep", action="store_true", help="sweep focus_absolute and pick the sharpest")
    ap.add_argument("--save-lock", action="store_true", help=f"write the effective lock to .lightwork/{lock_path().name}")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    lock = load_lock()
    for name, value in args.control:
        if name not in lock:
            ap.error(f"unknown control {name!r}; known: {', '.join(lock)}")
        lock[name] = value
    try:
        cam = open_camera(args.camera, width=args.width, height=args.height, fps=args.fps, lock=lock)
    except (CameraNotFound, OSError) as e:
        print(f"camera: {e}", file=sys.stderr)
        return 2
    with cam:
        _print_camera(cam)
        rc = 0
        if args.bench:
            rc = bench(cam, args.bench, args.fps)
        elif args.c0:
            rc = rig_c0(cam, lock)
        elif args.focus_sweep:
            best = focus_sweep(cam)
            if best is None:
                rc = 1
            else:
                lock["focus_absolute"] = best
        elif cam.report is not None and not cam.report.ok:
            rc = 1
        if args.save_lock:
            print(f"lock saved to {save_lock(lock)}")
    return rc


if __name__ == "__main__":
    sys.exit(main())
