"""adb plumbing for the Tank projector phone (rig brief §2, webvoice brief §4).

The phone reaches the laptop through ``adb reverse tcp:8000 tcp:8000`` so its page runs on ``localhost`` (a secure
context). The mapping is lost on USB replug or adb server restart; ``ReverseWatchdog`` re-creates it every 2 s.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import threading
from collections.abc import Callable

from ..contracts.server import PORT

CHROME_ACTIVITY = "com.android.chrome/com.google.android.apps.chrome.Main"
WATCHDOG_INTERVAL_S = 2.0


class AdbError(RuntimeError):
    pass


class Adb:
    def __init__(self, serial: str | None = None, timeout: float = 10.0):
        exe = shutil.which("adb")
        if exe is None:
            raise AdbError("adb not found on PATH (Arch: pacman -S android-tools)")
        self._base = [exe] + (["-s", serial] if serial else [])
        self._timeout = timeout

    def run(self, *args: str, check: bool = True) -> str:
        try:
            p = subprocess.run([*self._base, *args], capture_output=True, text=True, timeout=self._timeout)
        except subprocess.TimeoutExpired as e:
            raise AdbError(f"adb {' '.join(args)}: timed out") from e
        if check and p.returncode != 0:
            raise AdbError(f"adb {' '.join(args)}: {(p.stderr or p.stdout).strip()}")
        return p.stdout

    def state(self) -> str | None:
        """``device`` when one usable phone is attached; ``unauthorized``/``offline``; None when absent."""
        try:
            return self.run("get-state").strip() or None
        except AdbError as e:
            m = re.search(r"device (unauthorized|offline)", str(e))
            return m.group(1) if m else None

    def reverse_present(self, port: int = PORT) -> bool:
        spec = f"tcp:{port} tcp:{port}"
        return any(line.split(" ", 1)[-1].strip() == spec for line in self.run("reverse", "--list").splitlines())

    def reverse(self, port: int = PORT) -> None:
        """Re-create the mapping from scratch (a stale half-dead mapping survives a plain re-add)."""
        self.run("reverse", "--remove", f"tcp:{port}", check=False)
        self.run("reverse", f"tcp:{port}", f"tcp:{port}")

    def remove_reverse(self, port: int = PORT) -> None:
        self.run("reverse", "--remove", f"tcp:{port}", check=False)

    def shell(self, *args: str, check: bool = True) -> str:
        return self.run("shell", *args, check=check)

    def screen_size(self) -> tuple[int, int]:
        """Current display size in px (override size wins over physical)."""
        out = self.shell("wm", "size")
        sizes = dict(re.findall(r"(Physical|Override) size: (\d+x\d+)", out))
        w, h = (sizes.get("Override") or sizes["Physical"]).split("x")
        return int(w), int(h)

    def open_url(self, url: str) -> None:
        self.shell("am", "start", "-a", "android.intent.action.VIEW", "-n", CHROME_ACTIVITY, "-d", url)

    def tap(self, x: int, y: int) -> None:
        self.shell("input", "tap", str(x), str(y))


class ReverseWatchdog:
    """Every ``interval`` s: if the phone is attached and ``tcp:port`` is not reversed, restore it.
    ``on_change(state, detail)`` is called on transitions: ``restored``, ``no-device``, ``error``."""

    def __init__(self, adb: Adb, port: int = PORT, *, interval: float = WATCHDOG_INTERVAL_S,
                 on_change: Callable[[str, str], None] | None = None):
        self._adb = adb
        self._port = port
        self._interval = interval
        self._on_change = on_change or (lambda state, detail: None)
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, name="adb-reverse-watchdog", daemon=True)
        self._last: str | None = None

    def start(self) -> ReverseWatchdog:
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(self._interval + self._adb._timeout)

    def check_once(self) -> str:
        """One watchdog pass; returns ``ok``, ``restored``, ``no-device`` or ``error``."""
        try:
            if self._adb.state() != "device":
                result, detail = "no-device", "no authorised phone attached"
            elif self._adb.reverse_present(self._port):
                result, detail = "ok", ""
            else:
                self._adb.reverse(self._port)
                result, detail = "restored", f"adb reverse tcp:{self._port} re-created"
        except AdbError as e:
            result, detail = "error", str(e)
        if result != "ok" and (result == "restored" or result != self._last):
            self._on_change(result, detail)
        self._last = result
        return result

    def _loop(self) -> None:
        while not self._stop.is_set():
            self.check_once()
            self._stop.wait(self._interval)
