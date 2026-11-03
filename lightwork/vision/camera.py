"""C920 device resolution and v4l2 control locking (Linux only; ``briefs/rig.md`` §3.1).

Controls are written in :data:`LOCK_ORDER` (every auto switch before its manual value, because the driver rejects a
manual value while its auto mode is on), one ``v4l2-ctl -c`` call each, then read back with ``v4l2-ctl -C``. Kernels
expose one of two name sets (``auto_exposure`` vs ``exposure_auto`` ...); a lock is keyed by the new names and each
control is written under whichever name the device lists. Lock values: ``.lightwork/camera.json``
(``{"controls": {"exposure_time_absolute": 312, ...}}``, old names accepted, ``null`` = device default) merged over
:data:`LOCK_ORDER` defaults. Requested values are clamped to the device's min/max/step before writing; the readback
must equal the clamped value.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from ..config import STATE_DIR_NAME

BY_ID_DIR = Path("/dev/v4l/by-id")
LOCK_FILE = "camera.json"
V4L2_CTL = "v4l2-ctl"


@dataclass(frozen=True)
class CtrlSpec:
    name: str  # new kernel name (canonical)
    old: str | None  # older kernels / most web guides
    default: int | None  # None = the device's own default, still written explicitly


LOCK_ORDER: tuple[CtrlSpec, ...] = (
    CtrlSpec("auto_exposure", "exposure_auto", 1),  # 1 = manual on V4L2
    CtrlSpec("exposure_dynamic_framerate", "exposure_auto_priority", 0),
    # 100 us units. A streaming C920 PRO (046d:08e5) applies only a few steps (156 and 312 read back exactly) and
    # silently keeps its old value for others such as 167, so the default is its own default step.
    CtrlSpec("exposure_time_absolute", "exposure_absolute", 156),
    CtrlSpec("gain", None, None),
    CtrlSpec("white_balance_automatic", "white_balance_temperature_auto", 0),
    CtrlSpec("white_balance_temperature", None, 5000),
    CtrlSpec("focus_automatic_continuous", "focus_auto", 0),
    CtrlSpec("focus_absolute", None, 20),
    CtrlSpec("power_line_frequency", None, 1),  # 50 Hz
    CtrlSpec("backlight_compensation", None, 0),
    CtrlSpec("brightness", None, None),
    CtrlSpec("contrast", None, None),
    CtrlSpec("saturation", None, None),
    CtrlSpec("sharpness", None, None),
    CtrlSpec("zoom_absolute", None, None),
)
_CANONICAL = {s.name: s.name for s in LOCK_ORDER} | {s.old: s.name for s in LOCK_ORDER if s.old}


# ---------------------------------------------------------------------------------------------------- devices


@dataclass(frozen=True)
class VideoDevice:
    path: str  # what to open: the by-id symlink when there is one, else /dev/videoN
    node: str  # resolved /dev/videoN
    by_id: str | None  # by-id link name, e.g. usb-046d_HD_Pro_Webcam_C920_XXXX-video-index0

    @property
    def is_c920(self) -> bool:
        return self.by_id is not None and "c920" in self.by_id.lower()

    @property
    def device_id(self) -> str:
        return self.by_id or self.node

    def node_now(self) -> str:
        """The node the path points at right now (a by-id link moves to a new /dev/videoN after a replug)."""
        return os.path.realpath(self.path)


def list_devices(by_id_dir: Path = BY_ID_DIR) -> list[VideoDevice]:
    """Capture nodes (``*-video-index0``) under ``/dev/v4l/by-id``, sorted by name; [] when the dir is absent."""
    if not by_id_dir.is_dir():
        return []
    out = []
    for link in sorted(by_id_dir.iterdir()):
        if link.name.endswith("-video-index0"):
            out.append(VideoDevice(path=str(link), node=str(link.resolve()), by_id=link.name))
    return out


def find_c920(by_id_dir: Path = BY_ID_DIR) -> VideoDevice | None:
    return next((d for d in list_devices(by_id_dir) if d.is_c920), None)


def device_for_node(node: str, by_id_dir: Path = BY_ID_DIR) -> VideoDevice:
    """A /dev/videoN or by-id path -> VideoDevice, naming it by its by-id link when one points at the same node."""
    p = Path(node)
    if p.parent == by_id_dir:
        return VideoDevice(path=str(p), node=str(p.resolve()), by_id=p.name)
    real = str(p.resolve())
    for d in list_devices(by_id_dir):
        if d.node == real:
            return d
    return VideoDevice(path=real, node=real, by_id=None)


# ---------------------------------------------------------------------------------------------------- v4l2-ctl

Runner = Callable[[Sequence[str]], tuple[int, str, str]]  # argv -> (returncode, stdout, stderr)


def subprocess_runner(argv: Sequence[str]) -> tuple[int, str, str]:
    try:
        p = subprocess.run(list(argv), capture_output=True, text=True, timeout=5, check=False)
    except subprocess.TimeoutExpired:
        return 124, "", f"{argv[0]}: timed out"
    return p.returncode, p.stdout, p.stderr


def have_v4l2_ctl() -> bool:
    return shutil.which(V4L2_CTL) is not None


@dataclass(frozen=True)
class CtrlInfo:
    name: str
    type: str  # int | bool | menu | ...
    value: int
    default: int | None
    min: int | None
    max: int | None
    step: int | None
    inactive: bool

    def clamp(self, v: int) -> int:
        if self.type == "bool":
            return 1 if v else 0
        if self.min is not None and v < self.min:
            v = self.min
        if self.max is not None and v > self.max:
            v = self.max
        if self.step and self.step > 1 and self.min is not None:
            v = self.min + round((v - self.min) / self.step) * self.step
            if self.max is not None and v > self.max:
                v -= self.step
        return v


_CTRL_LINE = re.compile(r"^\s*(\w+)\s+0x[0-9a-fA-F]+\s+\((\w+)\)\s*:\s*(.*)$")
_KV = re.compile(r"(\w+)=(-?\d+)")
_GET_LINE = re.compile(r"^\s*(\w+):\s*(-?\d+)")


def parse_list_ctrls(text: str) -> dict[str, CtrlInfo]:
    out: dict[str, CtrlInfo] = {}
    for line in text.splitlines():
        m = _CTRL_LINE.match(line)
        if not m:
            continue
        name, typ, rest = m.groups()
        kv = {k: int(v) for k, v in _KV.findall(rest)}
        if "value" not in kv:  # buttons, strings and compound controls carry no int value
            continue
        flags = rest.split("flags=", 1)[1] if "flags=" in rest else ""
        out[name] = CtrlInfo(
            name=name,
            type=typ,
            value=kv["value"],
            default=kv.get("default"),
            min=kv.get("min", 0 if typ == "bool" else None),
            max=kv.get("max", 1 if typ == "bool" else None),
            step=kv.get("step"),
            inactive="inactive" in flags,
        )
    return out


def parse_get_ctrls(text: str) -> dict[str, int]:
    return {m.group(1): int(m.group(2)) for line in text.splitlines() if (m := _GET_LINE.match(line))}


class V4l2Ctl:
    """``v4l2-ctl`` on one device node. ``run`` is injectable so tests drive a fake device."""

    def __init__(self, device: str, run: Runner = subprocess_runner):
        self.device = device
        self._run = run

    def list_ctrls(self) -> dict[str, CtrlInfo]:
        rc, out, err = self._run([V4L2_CTL, "-d", self.device, "--list-ctrls"])
        if rc != 0:
            raise OSError(f"{V4L2_CTL} --list-ctrls on {self.device} failed: {err.strip() or rc}")
        return parse_list_ctrls(out)

    def set(self, name: str, value: int) -> str | None:
        """Write one control; returns the error text, or None on success."""
        rc, _, err = self._run([V4L2_CTL, "-d", self.device, "-c", f"{name}={value}"])
        return None if rc == 0 else (err.strip() or f"exit {rc}")

    def get(self, names: Iterable[str]) -> dict[str, int]:
        names = list(names)
        if not names:
            return {}
        rc, out, err = self._run([V4L2_CTL, "-d", self.device, "-C", ",".join(names)])
        if rc != 0 and not out:
            raise OSError(f"{V4L2_CTL} -C on {self.device} failed: {err.strip() or rc}")
        return parse_get_ctrls(out)


# ---------------------------------------------------------------------------------------------------- lock


def lock_path(state_dir: Path | None = None) -> Path:
    return (state_dir or Path.cwd() / STATE_DIR_NAME) / LOCK_FILE


def load_lock(path: Path | None = None) -> dict[str, int | None]:
    """Requested values keyed by canonical name: LOCK_ORDER defaults, overridden by ``camera.json`` if present."""
    lock: dict[str, int | None] = {s.name: s.default for s in LOCK_ORDER}
    path = path or lock_path()
    if path.is_file():
        data = json.loads(path.read_text(encoding="utf-8"))
        controls = data.get("controls", {}) if isinstance(data, dict) else None
        if not isinstance(controls, dict):
            raise ValueError(f"{path}: expected {{\"controls\": {{name: int|null}}}}")
        for k, v in controls.items():
            name = _CANONICAL.get(k)
            if name is None:
                raise ValueError(f"{path}: unknown control {k!r}; known: {', '.join(s.name for s in LOCK_ORDER)}")
            if v is not None and (isinstance(v, bool) or not isinstance(v, int)):
                raise ValueError(f"{path}: {k} must be an integer or null, got {v!r}")
            lock[name] = v
    return lock


def save_lock(lock: Mapping[str, int | None], path: Path | None = None) -> Path:
    path = path or lock_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    ordered = {s.name: lock.get(s.name, s.default) for s in LOCK_ORDER}
    path.write_text(json.dumps({"controls": ordered}, indent=2) + "\n", encoding="utf-8")
    return path


@dataclass(frozen=True)
class PlannedCtrl:
    name: str  # canonical
    device_name: str  # the name this device lists
    value: int  # clamped value to write
    requested: int | None  # lock value before clamping (None = device default)


def plan_lock(available: Mapping[str, CtrlInfo], lock: Mapping[str, int | None]) -> tuple[list[PlannedCtrl], list[str]]:
    """Ordered writes for this device, plus the canonical names it lacks under either name."""
    planned: list[PlannedCtrl] = []
    absent: list[str] = []
    for spec in LOCK_ORDER:
        dev_name = spec.name if spec.name in available else spec.old if spec.old in available else None
        if dev_name is None:
            absent.append(spec.name)
            continue
        info = available[dev_name]
        req = lock.get(spec.name, spec.default)
        base = req if req is not None else info.default if info.default is not None else info.value
        planned.append(PlannedCtrl(spec.name, dev_name, info.clamp(base), req))
    return planned, absent


@dataclass(frozen=True)
class CtrlRow:
    name: str
    device_name: str
    requested: int | None
    value: int  # written (clamped)
    readback: int | None
    error: str | None

    @property
    def ok(self) -> bool:
        return self.error is None and self.readback == self.value


@dataclass(frozen=True)
class ControlReport:
    rows: list[CtrlRow]
    absent: list[str] = field(default_factory=list)  # lock controls the device lists under neither name

    @property
    def ok(self) -> bool:
        return bool(self.rows) and all(r.ok for r in self.rows)

    @property
    def mismatches(self) -> list[CtrlRow]:
        return [r for r in self.rows if not r.ok]

    @property
    def name_set(self) -> str:
        """``new``/``old``/``mixed`` over the controls that have two names."""
        two = {s.name for s in LOCK_ORDER if s.old}
        kinds = {"new" if r.device_name == r.name else "old" for r in self.rows if r.name in two}
        return kinds.pop() if len(kinds) == 1 else "mixed" if kinds else "n/a"

    def controls(self) -> dict[str, int]:
        """Read-back state keyed by the device's own names (C7 ``camera.controls``)."""
        return {r.device_name: r.readback for r in self.rows if r.readback is not None}

    def lines(self) -> list[str]:
        out = []
        for r in self.rows:
            alias = "" if r.device_name == r.name else f" (as {r.device_name})"
            want = f"{r.value}" + ("" if r.requested in (None, r.value) else f" (clamped from {r.requested})")
            state = "ok" if r.ok else f"MISMATCH read {r.readback}" + (f": {r.error}" if r.error else "")
            out.append(f"{r.name}{alias}: want {want}, {state}")
        out += [f"{n}: not on this device" for n in self.absent]
        return out


def apply_lock(ctl: V4l2Ctl, lock: Mapping[str, int | None]) -> ControlReport:
    """Write the lock in LOCK_ORDER, then read every written control back."""
    planned, absent = plan_lock(ctl.list_ctrls(), lock)
    errors = {p.device_name: ctl.set(p.device_name, p.value) for p in planned}
    back = ctl.get(p.device_name for p in planned)
    rows = [
        CtrlRow(p.name, p.device_name, p.requested, p.value, back.get(p.device_name), errors[p.device_name])
        for p in planned
    ]
    return ControlReport(rows=rows, absent=absent)


def compare_lock(ctl: V4l2Ctl, lock: Mapping[str, int | None]) -> ControlReport:
    """Current device state vs the lock, without writing anything (doctor)."""
    available = ctl.list_ctrls()
    planned, absent = plan_lock(available, lock)
    rows = [CtrlRow(p.name, p.device_name, p.requested, p.value, available[p.device_name].value, None) for p in planned]
    return ControlReport(rows=rows, absent=absent)
