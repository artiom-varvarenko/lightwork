"""Calibration and rig:C1-C4 checks. Starts a local server and waits for /projector (run `lightwork tank up`
in another terminal). No camera images are saved except an explicitly selected, verified flat-field crop.
`calibrate screen` needs no projector: it records the locked camera for screen-only guidance.
See `lightwork.vision.calib` for the hardware-check recipe; all library flows also accept synthetic camera/page peers.
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from pathlib import Path

COMMAND = "calibrate"
HELP = "Calibrate projector guidance (run, card, check, watch, plate, flat; rig:C1-C4) or a screen-only camera (screen)."


def configure(p) -> None:
    p.add_argument("action", nargs="?", choices=("run", "card", "check", "watch", "plate", "flat", "screen"), default="run")
    p.add_argument("--camera", help="by-id path, camera index, video, or image directory (default: C920)")
    p.add_argument("--output", type=Path, help="calib.json path (screen: .lightwork/calib/screen.json); for card, printable SVG path")
    p.add_argument("--port", type=int, default=8000, help="local server port (default 8000)")
    p.add_argument("--page-wait", type=float, default=90.0, help="seconds to wait for a stable projector hello")
    p.add_argument("--mm", choices=("auto", "card", "idcard", "nominal"), default="auto")
    p.add_argument("--mount-mm", type=float, default=500.0, help="mount height for labelled nominal mm scale")
    p.add_argument("--mode", choices=("grey", "dark"), default="grey", help="measurement mode saved in C7")
    p.add_argument("--repeat", type=int, default=1, help="run: repeated flashes for rig:C2, each must take <=2 s")
    p.add_argument("--plane", choices=("table", "plate"), default="table", help="check: which saved homography")
    p.add_argument("--work-area", type=int, nargs=4, metavar=("X", "Y", "W", "H"), help="flat: explicitly consented crop, camera pixels")
    p.add_argument("--interval", type=float, default=0.5, help="watch: seconds between hand-free samples")
    p.add_argument("--duration", type=float, default=0.0, help="watch: seconds to run (0 = until Ctrl-C)")
    p.add_argument("--check-after-bump", action="store_true", help="watch: run C1 after each re-solve; keep the card on the bench")


def _open_camera(spec):
    from ..vision.capture import open_camera

    return open_camera(spec)


def _wait_projector(hub, timeout):
    deadline = time.monotonic() + timeout
    last, since = None, time.monotonic()
    while time.monotonic() < deadline:
        page = hub.hello("projector")
        now = time.monotonic()
        if page != last:
            last, since = page, now
        if page is not None and now - since >= 0.4:
            return page
        time.sleep(0.05)
    raise RuntimeError("no stable projector page: open /projector, tap Start; run `lightwork tank up` in another terminal")


@contextmanager
def _rig(args):
    from ..server.app import create_app
    from ..server.hub import make_hubs
    from ..server.runner import start_server

    cam, server = _open_camera(args.camera), None
    try:
        hubs = make_hubs()
        try:
            server = start_server(create_app(mode="calibrate", **hubs), port=args.port)
        except RuntimeError as exc:
            raise RuntimeError(f"{exc}; stop live/replay before calibrating") from exc
        print(f"Waiting for http://localhost:{args.port}/projector (tap Start).", flush=True)
        page = _wait_projector(hubs["render_hub"], args.page_wait)
        print(f"page: {page.innerWidth}x{page.innerHeight} CSS px, dpr={page.dpr:g}; camera: {cam.describe()}")
        if not page.fullscreen:
            print("WARNING: page is not fullscreen; resizing it later invalidates this calibration")
        if not cam.controlled:
            print("WARNING: uncontrolled camera; lock exposure, focus and white balance for repeatable accuracy")
        yield cam, hubs["render_hub"], page
    finally:
        if server is not None:
            server.stop()
        cam.close()


def _existing(path, page):
    from ..vision.calib.flow import load_calibration

    calib = load_calibration(path)
    if (calib.page.w, calib.page.h, calib.page.dpr) != (page.innerWidth, page.innerHeight, page.dpr):
        raise ValueError("page size/dpr changed since calibration; run `lightwork calibrate` again")
    return calib


def _flash_report(flash):
    r = flash.solution.residuals
    print(f"rig:C2: {flash.t_s:.3f} s; markers={r.n_markers}, inlier corners={r.n_inliers}; "
          f"rms={r.rms_px:.3f}, p95={r.p95_px:.3f}, max={r.max_px:.3f} page px; frames={flash.n_frames}")
    return flash.t_s <= 2.0 and r.n_markers >= 6 and r.rms_px <= 1.5


def _accuracy(cam, link, calib):
    from ..vision.calib.check import check_accuracy

    report = check_accuracy(cam, link, calib)
    print(f"rig:C1: {'PASS' if report.passed else 'FAIL'}; errors mm={report.errors_mm}; "
          f"mean={report.mean_mm}, max={report.max_mm}, card fit rms={report.card_rms_mm}")
    print("Spot-check ring-to-cross alignment with a ruler; print-scale error is not measured by the camera.")
    return report.passed


def _reference(cam, link, calib):
    from ..contracts.render import SceneMsg
    from ..vision.calib.bump import BumpMonitor, tick_prims
    from ..vision.calib.check import calib_message
    from ..vision.calib.flow import SETTLE_S
    from ..vision.calib.solve import CalibrationError

    link.set_calib(calib_message(calib))
    link.publish_scene(SceneMsg(seq=0, mode="normal", prims=tick_prims(calib)))
    ack = link.show_mode("normal")
    frame = cam.wait_frame(ack + SETTLE_S, 1.0) if ack is not None else None
    if frame is None:
        raise CalibrationError("no acknowledged corner-tick reference frame")
    monitor = BumpMonitor(frame.image, calib)
    if monitor.n_ticks < 2:
        raise CalibrationError("fewer than two projected corner ticks visible; uncover the page corners")
    print(f"watch reference: {monitor.n_patches} unlit patches, {monitor.n_ticks} ticks; keep hands out")
    return monitor


def _screen(args) -> int:
    from ..vision.calib.flow import save_calibration, screen_calib_path, screen_calibration

    path = args.output or screen_calib_path()
    cam = _open_camera(args.camera)
    try:
        calib = screen_calibration(cam)
        controlled = cam.controlled
    finally:
        cam.close()
    save_calibration(calib, path)
    print(f"{path}: screen-only calibration for {calib.camera.device_id}, {calib.camera.mode}; nothing is projected")
    print(f"controls: {calib.camera.controls or 'none (auto settings)'}")
    if not controlled:
        print("WARNING: uncontrolled camera; lock exposure, focus and white balance for repeatable evidence")
    print("Recalibrate after changing the camera, its mode or its lock (.lightwork/camera.json).")
    return 0


def _watch(args, cam, link, page, path):
    from ..vision.calib.flow import recalibrate
    from ..vision.calib.solve import CalibrationError

    calib = _existing(path, page)
    monitor = _reference(cam, link, calib)
    deadline = time.monotonic() + args.duration if args.duration else float("inf")
    first_positive = None
    while time.monotonic() < deadline:
        time.sleep(args.interval)
        current = link.hello("projector")
        if current is None or (current.innerWidth, current.innerHeight, current.dpr) != (page.innerWidth, page.innerHeight, page.dpr):
            raise CalibrationError("projector disconnected or resized; restart calibration")
        frame = cam.wait_frame(time.monotonic(), 1.0)
        if frame is None:
            raise CalibrationError("camera stopped while monitoring")
        now = time.monotonic()
        check = monitor.check(frame.image)
        if check.camera_moved or check.projector_moved:
            first_positive = now if first_positive is None else first_positive
        else:
            first_positive = None
        if not check.moved:
            continue
        result = recalibrate(cam, link, calib, page, check, path=path)
        print(f"rig:C3: cause={result.cause}; confirmation after first positive sample "
              f"{now - first_positive:.3f} s; re-solve {result.t_s:.3f} s (bump onset must be timed externally)")
        _flash_report(result.flash)
        calib = result.calib
        print("Flat field and raised-plane calibration invalidated; re-capture them before live use.")
        if result.cause in ("camera", "both", "unobserved"):
            print("Camera-coordinate layout must be updated/rechecked before resuming a live run.")
        if args.check_after_bump and not _accuracy(cam, link, calib):
            return 1
        monitor = _reference(cam, link, calib)
        first_positive = None
    return 0


def run(args) -> int:
    from ..vision.calib.flow import add_flat_field, calib_path, calibrate_plate_height, load_calibration, run_calibration
    from ..vision.calib.mm import card_svg, mm_per_page_px
    from ..vision.calib.solve import CalibrationError

    try:
        if args.action == "card":
            path = args.output or calib_path().with_name("card.svg")
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(card_svg(), encoding="utf-8")
            print(f"{path}: print A4 landscape at 100 %, no fit-to-page; verify the 100 mm bar with a ruler")
            return 0
        if args.action == "screen":
            return _screen(args)
        if args.repeat < 1 or args.interval <= 0 or args.duration < 0 or args.page_wait <= 0 or args.mount_mm <= 0:
            raise ValueError("repeat >=1, interval/page-wait/mount-mm >0 and duration >=0 are required")
        if args.action == "flat" and args.work_area is None:
            raise ValueError("flat requires --work-area X Y W H; only explicitly selected pixels may be stored")
        path = args.output or calib_path()
        if args.action != "run" and load_calibration(path).screen_only:
            raise ValueError(f"{path} is a screen-only calibration; projector checks need `lightwork calibrate run`")
        with _rig(args) as (cam, link, page):
            if args.action == "run":
                ok = True
                for i in range(args.repeat):
                    result = run_calibration(cam, link, page, measurement_mode=args.mode, mm=args.mm,
                                             mount_mm=args.mount_mm, path=path)
                    print(f"run {i + 1}/{args.repeat}: {path}")
                    ok = _flash_report(result.flash) and ok
                    print(f"mm source={result.calib.mm.source}; nominal/measured scale={mm_per_page_px(result.calib):.4f} mm/page px")
                    for note in result.notes:
                        print(note)
                return 0 if ok else 1
            if args.action == "watch":
                return _watch(args, cam, link, page, path)
            calib = _existing(path, page)
            if args.action == "check":
                if args.plane == "plate":
                    if calib.H_plate_cam_to_page is None:
                        raise ValueError("no plate-plane homography: run `lightwork calibrate plate` with a raised board")
                    calib = calib.model_copy(update={"H_cam_to_page": calib.H_plate_cam_to_page})
                return 0 if _accuracy(cam, link, calib) else 1
            if args.action == "plate":
                _, flash = calibrate_plate_height(cam, link, page, calib, path=path)
                print(f"Plate-height homography stored in {path}; now run `calibrate check --plane plate` on that plane")
                return 0 if _flash_report(flash) else 1
            calib, flat = add_flat_field(cam, link, calib, path=path, work_area_xywh=tuple(args.work_area))
            if flat is None or not flat.verified:
                print("flat field failed: " + ("no ACK/frames" if flat is None else "; ".join(flat.reasons)))
                return 1
            print(f"flat field {calib.flat_field_sha}; crop={calib.flat_field_crop_xywh}; {path}")
            return 0
    except CalibrationError as exc:
        print(f"calibration failed: {exc}")
        return 1
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"calibration environment: {exc}")
        return 2
