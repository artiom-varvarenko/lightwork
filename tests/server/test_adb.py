from __future__ import annotations

from lightwork.server.adb import Adb, AdbError, ReverseWatchdog


class FakeAdb(Adb):
    """Phone simulator: `attached` toggles a replug; the reverse list empties on detach like the real adb."""

    def __init__(self):
        self._timeout = 1.0
        self.attached = True
        self.reversed: set[str] = set()
        self.calls: list[tuple[str, ...]] = []

    def run(self, *args: str, check: bool = True) -> str:
        self.calls.append(args)
        if not self.attached:
            if check:
                raise AdbError("adb: no devices/emulators found")
            return ""
        if args == ("get-state",):
            return "device\n"
        if args[:2] == ("reverse", "--list"):
            return "".join(f"UsbFfs {r} {r}\n" for r in sorted(self.reversed))
        if args[:2] == ("reverse", "--remove"):
            self.reversed.discard(args[2])
            return ""
        if args[0] == "reverse":
            self.reversed.add(args[1])
            return ""
        if args == ("shell", "wm", "size"):
            return "Physical size: 1080x2400\nOverride size: 720x1600\n"
        return ""


def test_watchdog_restores_after_replug():
    adb = FakeAdb()
    changes: list[str] = []
    wd = ReverseWatchdog(adb, 8000, on_change=lambda s, d: changes.append(s))
    assert wd.check_once() == "restored" and adb.reverse_present(8000)
    assert wd.check_once() == "ok"
    adb.attached = False
    adb.reversed.clear()
    assert wd.check_once() == "no-device"
    assert wd.check_once() == "no-device"  # reported once per transition
    adb.attached = True
    assert wd.check_once() == "restored"
    assert changes == ["restored", "no-device", "restored"]
    assert ("reverse", "--remove", "tcp:8000") in adb.calls  # re-created from scratch


def test_screen_size_prefers_override():
    assert FakeAdb().screen_size() == (720, 1600)


def test_reverse_present_matches_exact_port():
    adb = FakeAdb()
    adb.reversed.add("tcp:80001")
    assert not adb.reverse_present(8000)
