"""``lightwork tank up|down|status``: set up the Tank projector phone over adb (webvoice brief §4, rig brief §2).

``up``: ``adb reverse`` (re-created from scratch), ``svc power stayon usb``, open Chrome on ``/projector``, tap the
page's "tap to start" layer (fullscreen + landscape + speech), report the page's ``hello`` if the server runs, then
keep the 2 s reverse watchdog in the foreground until Ctrl-C (``--no-watch`` to exit after setup).
"""

from __future__ import annotations

import time

COMMAND = "tank"
HELP = "Tank projector phone: adb reverse + watchdog, keep awake, open /projector fullscreen (up|down|status)."

PAGE_WAIT_S = 6.0


def configure(p) -> None:
    from ..contracts.server import PORT

    p.add_argument("action", choices=("up", "down", "status"))
    p.add_argument("--serial", help="adb serial when several devices are attached")
    p.add_argument("--port", type=int, default=PORT)
    p.add_argument("--no-watch", action="store_true", help="up: exit after setup instead of running the watchdog")
    p.add_argument("--no-open", action="store_true", help="up: only reverse + stay awake; do not open Chrome")
    p.add_argument("--load-wait", type=float, default=3.0, help="up: seconds between opening the page and the tap")


def _projector_pages(port: int) -> list[dict] | None:
    """Projector pages connected to a running local server, or None when no server answers."""
    import httpx

    try:
        r = httpx.get(f"http://127.0.0.1:{port}/api/pages", timeout=1.0)
        r.raise_for_status()
    except httpx.HTTPError:
        return None
    return [p for p in r.json()["render"] if p["role"] == "projector"]


def _describe(page: dict) -> str:
    h = page["hello"]
    ack = f", ACK p50 {page['ack_p50_ms']:.0f} ms" if page["ack_p50_ms"] is not None else ""
    return (
        f"{h['innerWidth']}x{h['innerHeight']} @{h['dpr']} {h['orientation']}, fullscreen={h['fullscreen']}, "
        f"secure={h['secure']}, wakelock={h['wakelock']}{ack}"
    )


def _up(adb, args) -> int:
    from ..server.adb import AdbError, ReverseWatchdog

    try:
        adb.reverse(args.port)
        print(f"ok   adb reverse tcp:{args.port} tcp:{args.port}")
        adb.shell("svc", "power", "stayon", "usb")
        print("ok   svc power stayon usb")
        if not args.no_open:
            adb.open_url(f"http://localhost:{args.port}/projector")
            print("ok   Chrome -> /projector")
            time.sleep(args.load_wait)
            w, h = adb.screen_size()
            adb.tap(w // 2, h // 2)
            print(f"ok   input tap {w // 2} {h // 2} (tap-to-start layer)")
    except AdbError as e:
        print(f"FAIL {e}")
        return 2
    if not args.no_open:
        deadline = time.monotonic() + PAGE_WAIT_S
        pages = _projector_pages(args.port)
        while pages is not None and not any(p["hello"]["fullscreen"] for p in pages) and time.monotonic() < deadline:
            time.sleep(0.5)
            pages = _projector_pages(args.port)
        if pages is None:
            print(f"--   no server on 127.0.0.1:{args.port}; the page connects once `lightwork live`/`calibrate` runs")
        elif not pages:
            print("FAIL no projector page connected (Chrome first-run screens? tap the phone once by hand)")
        else:
            print(f"page {_describe(pages[-1])}")
            if not pages[-1]["hello"]["fullscreen"]:
                print("--   not fullscreen: the injected tap did not count; tap the phone once (or click in scrcpy)")
    if args.no_watch:
        return 0
    print("watchdog: checking adb reverse every 2 s (Ctrl-C to stop)")
    watchdog = ReverseWatchdog(adb, args.port, on_change=lambda s, d: print(f"{time.strftime('%H:%M:%S')} {s}: {d}"))
    watchdog.start()
    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        watchdog.stop()
    return 0


def _down(adb, args) -> int:
    from ..server.adb import AdbError

    try:
        adb.remove_reverse(args.port)
        adb.shell("svc", "power", "stayon", "false")
    except AdbError as e:
        print(f"FAIL {e}")
        return 2
    print(f"ok   removed adb reverse tcp:{args.port}; stay-awake off")
    return 0


def _status(adb, args) -> int:
    from ..server.adb import AdbError

    state = adb.state()
    print(f"device   {state or 'none'}")
    if state != "device":
        return 2
    try:
        print(f"reverse  {'tcp:%d ok' % args.port if adb.reverse_present(args.port) else 'MISSING'}")
        w, h = adb.screen_size()
        print(f"display  {w}x{h}")
        print(f"stayon   {adb.shell('settings', 'get', 'global', 'stay_on_while_plugged_in').strip()} (2/3/7 = on USB)")
    except AdbError as e:
        print(f"FAIL {e}")
        return 2
    pages = _projector_pages(args.port)
    if pages is None:
        print(f"server   not running on 127.0.0.1:{args.port}")
    elif not pages:
        print("page     no projector page connected")
    for p in pages or []:
        print(f"page     {_describe(p)}")
    return 0


def run(args) -> int:
    from ..server.adb import Adb, AdbError

    try:
        adb = Adb(args.serial)
    except AdbError as e:
        print(f"FAIL {e}")
        return 2
    if args.action != "status" and adb.state() != "device":
        print(f"FAIL no authorised phone attached (adb state: {adb.state() or 'none'}); enable USB debugging")
        return 2
    return {"up": _up, "down": _down, "status": _status}[args.action](adb, args)
