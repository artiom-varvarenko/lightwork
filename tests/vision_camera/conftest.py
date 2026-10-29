"""Fakes for workstream A: a v4l2-ctl device (both control name sets) and an OpenCV capture device."""

from __future__ import annotations

import time

import cv2
import numpy as np
import pytest

# name, old name, id, type, min, max, step, default (C920 ranges)
_C920 = [
    ("brightness", None, 0x00980900, "int", 0, 255, 1, 128),
    ("contrast", None, 0x00980901, "int", 0, 255, 1, 128),
    ("saturation", None, 0x00980902, "int", 0, 255, 1, 128),
    ("white_balance_automatic", "white_balance_temperature_auto", 0x0098090C, "bool", 0, 1, 1, 1),
    ("gain", None, 0x00980913, "int", 0, 255, 1, 0),
    ("power_line_frequency", None, 0x00980918, "menu", 0, 2, 1, 2),
    ("white_balance_temperature", None, 0x0098091A, "int", 2000, 6500, 1, 4000),
    ("sharpness", None, 0x0098091B, "int", 0, 255, 1, 128),
    ("backlight_compensation", None, 0x0098091C, "int", 0, 1, 1, 0),
    ("auto_exposure", "exposure_auto", 0x009A0901, "menu", 0, 3, 1, 3),
    ("exposure_time_absolute", "exposure_absolute", 0x009A0902, "int", 3, 2047, 1, 250),
    ("exposure_dynamic_framerate", "exposure_auto_priority", 0x009A0903, "bool", 0, 1, 1, 0),
    ("focus_absolute", None, 0x009A090A, "int", 0, 250, 5, 0),
    ("focus_automatic_continuous", "focus_auto", 0x009A090C, "bool", 0, 1, 1, 1),
    ("zoom_absolute", None, 0x009A090D, "int", 100, 500, 1, 100),
]
# manual control -> (its auto switch, auto switch value that permits a manual write)
_GUARDS = {
    "exposure_time_absolute": ("auto_exposure", 1),
    "white_balance_temperature": ("white_balance_automatic", 0),
    "focus_absolute": ("focus_automatic_continuous", 0),
}


class FakeV4l2:
    """Emulates ``v4l2-ctl`` on a C920: rejects a manual value while its auto mode is on (I/O error, as on the real
    camera), validates ranges, and can silently ignore writes to chosen controls (readback mismatch)."""

    def __init__(self, names: str = "new", ignore: tuple[str, ...] = (), drop: tuple[str, ...] = ()):
        self.names = names
        self.ignore = set(ignore)
        self.specs = {}
        for new, old, cid, typ, lo, hi, step, default in _C920:
            if new in drop:
                continue
            name = old if names == "old" and old else new
            self.specs[name] = (new, cid, typ, lo, hi, step, default)
        self.values = {n: s[6] for n, s in self.specs.items()}
        self.calls: list[list[str]] = []
        self.writes: list[tuple[str, int]] = []

    def reset(self) -> None:
        """Power cycle: every control back to its default."""
        self.values = {n: s[6] for n, s in self.specs.items()}

    def _dev(self, canonical: str) -> str:
        return next(n for n, s in self.specs.items() if s[0] == canonical)

    def _inactive(self, name: str) -> bool:
        guard = _GUARDS.get(self.specs[name][0])
        return guard is not None and self.values[self._dev(guard[0])] != guard[1]

    def run(self, argv):
        argv = list(argv)
        self.calls.append(argv)
        assert argv[0] == "v4l2-ctl" and argv[1] == "-d"
        op = argv[3]
        if op == "--list-ctrls":
            lines = []
            for name, (_, cid, typ, lo, hi, step, default) in self.specs.items():
                v = self.values[name]
                rng = f"default={default} value={v}" if typ == "bool" else f"min={lo} max={hi}" + (
                    "" if typ == "menu" else f" step={step}") + f" default={default} value={v}"
                flags = " flags=inactive" if self._inactive(name) else ""
                lines.append(f"{name:>31} 0x{cid:08x} ({typ})    : {rng}{' (Manual Mode)' if typ == 'menu' else ''}{flags}")
            return 0, "\n".join(lines) + "\n", ""
        if op == "-c":
            name, _, value = argv[4].partition("=")
            if name not in self.specs:
                return 1, "", f"unknown control '{name}'\n"
            v = int(value)
            _, _, typ, lo, hi, _, _ = self.specs[name]
            if not lo <= v <= hi:
                return 255, "", "VIDIOC_S_EXT_CTRLS: failed: Numerical result out of range\n"
            if self._inactive(name):
                return 255, "", "VIDIOC_S_EXT_CTRLS: failed: Input/output error\n"
            self.writes.append((name, v))
            if name not in self.ignore:
                self.values[name] = v
            return 0, "", ""
        if op == "-C":
            out = []
            for name in argv[4].split(","):
                if name not in self.specs:
                    return 255, "".join(out), f"unknown control '{name}'\n"
                out.append(f"{name}: {self.values[name]}\n")
            return 0, "".join(out), ""
        raise AssertionError(f"unexpected v4l2-ctl call {argv}")


class FakeCapture:
    """Stand-in for cv2.VideoCapture on a device. ``plan`` is shared across instances: each entry is how many frames
    that opening serves before read() fails (unplug); ``None`` = the open itself fails."""

    def __init__(self, plan: list[int | None], size=(64, 48), on_open=None):
        self.on_open = on_open  # called with the 1-based open count before each open
        self.plan = plan
        self.size = size
        self.opened: list[tuple[object, int]] = []
        self.props: dict[int, float] = {}

    def __call__(self, target, api):
        self.opened.append((target, api))
        if self.on_open is not None:
            self.on_open(len(self.opened))
        budget = self.plan.pop(0) if self.plan else 10**9
        return _Cap(self, budget)


class _Cap:
    def __init__(self, parent: FakeCapture, budget: int | None):
        self.parent = parent
        self.budget = budget
        self.n = 0
        self.ts = 0.0

    def isOpened(self):
        return self.budget is not None

    def set(self, prop, value):
        self.parent.props[prop] = value
        return True

    def get(self, prop):
        w, h = self.parent.size
        if prop == cv2.CAP_PROP_FOURCC:
            return float(int.from_bytes(b"MJPG", "little"))
        if prop == cv2.CAP_PROP_FRAME_WIDTH:
            return float(w)
        if prop == cv2.CAP_PROP_FRAME_HEIGHT:
            return float(h)
        if prop == cv2.CAP_PROP_FPS:
            return 30.0
        if prop == cv2.CAP_PROP_POS_MSEC:
            return self.ts * 1000
        return 0.0

    def read(self):
        if self.budget is None or self.n >= self.budget:
            time.sleep(0.005)
            return False, None
        self.n += 1
        time.sleep(0.005)
        self.ts = time.monotonic() - 0.010  # driver timestamp 10 ms before dequeue
        w, h = self.parent.size
        return True, np.full((h, w, 3), self.n % 256, np.uint8)

    def release(self):
        self.budget = None


@pytest.fixture
def fake_v4l2():
    return FakeV4l2


@pytest.fixture
def fake_capture():
    return FakeCapture
