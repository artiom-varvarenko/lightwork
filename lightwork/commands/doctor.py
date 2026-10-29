"""``lightwork doctor``: environment checks. Each section is a function returning (name, ok, detail) rows.

Sections are owned by their workstreams (camera: A, Tank/adb: J, models: F). Missing optional pieces are reported,
not failed; only a broken core install exits 2.
"""

from __future__ import annotations

import importlib.metadata
import platform
import shutil
import sys
from pathlib import Path

COMMAND = "doctor"
HELP = "Check the local environment (python, packages, keys, adb, camera)."

Row = tuple[str, bool | None, str]  # ok: True pass, False fail, None info/optional missing


def section_core() -> list[Row]:
    rows: list[Row] = [("python", sys.version_info[:2] == (3, 12), platform.python_version())]
    for mod in ("cv2", "numpy", "fastapi", "uvicorn", "httpx", "pydantic", "meta_sam_parser"):
        try:
            m = __import__(mod)
            rows.append((mod, True, getattr(m, "__version__", "ok")))
        except Exception as e:  # noqa: BLE001
            rows.append((mod, False, f"import failed: {e}"))
    # Metadata only: importing mediapipe pulls in sounddevice/PortAudio, whose PipeWire-JACK client sets
    # RLIMIT_RTTIME=0 for the process, and a later HandLandmarker in it (or a child) gets SIGKILLed.
    try:
        rows.append(("mediapipe (live extra)", True, importlib.metadata.version("mediapipe")))
    except importlib.metadata.PackageNotFoundError:
        rows.append(("mediapipe (live extra)", None, "not installed: `uv sync --extra live` for hand tracking"))
    return rows


def section_settings() -> list[Row]:
    from ..config import load_settings

    s = load_settings()
    rows: list[Row] = [
        ("MODEL_API_KEY", None if not s.model_api_key else True, "set" if s.model_api_key else "not set (replay/verify need none)"),
        ("META_API_BASE", True, s.meta_api_base),
        ("LW_BACKEND", s.backend in ("meta", "glimmer"), s.backend),
        ("LW_CHAT_BASE_URL", None, s.chat_base_url or "not set"),
        ("LW_BUDGET_USD", True, f"{s.budget_usd:.2f}"),
    ]
    if s.dev_allow_non_meta:
        rows.append(("LW_DEV_ALLOW_NON_META", None, "ON: dev stand-ins allowed; runs go to runs-dev/ (D15)"))
    return rows


def section_tools() -> list[Row]:
    rows: list[Row] = []
    for tool in ("adb", "v4l2-ctl", "ffmpeg"):
        path = shutil.which(tool)
        rows.append((tool, None if not path else True, path or "not found (live rig only)"))
    return rows


def section_camera(by_id_dir: Path | None = None, state_dir: Path | None = None) -> list[Row]:
    """By-id devices, the device ``open_camera()`` would pick, and its controls now vs the lock (read-only)."""
    from ..vision import camera

    if sys.platform != "linux":
        return [("camera", None, f"{sys.platform}: opened by index with auto settings (uncontrolled camera)")]
    by_id_dir = by_id_dir or camera.BY_ID_DIR
    devices = camera.list_devices(by_id_dir)
    if not devices:
        return [("camera", None, f"none found under {by_id_dir} (live needs one; replay/verify don't)")]
    rows: list[Row] = [
        ("camera", None, f"{d.device_id} -> {d.node}" + (" [C920]" if d.is_c920 else "")) for d in devices
    ]
    c920 = next((d for d in devices if d.is_c920), None)
    if c920 is None:
        rows.append(("camera chosen", None, "no C920: pass a camera path or index explicitly (runs uncontrolled)"))
        return rows
    rows.append(("camera chosen", True, c920.device_id))
    path = camera.lock_path(state_dir)
    try:
        lock = camera.load_lock(path)
    except ValueError as e:
        rows.append(("camera lock", False, str(e)))
        return rows
    rows.append(("camera lock", True, str(path) if path.is_file() else f"built-in defaults (no {path})"))
    if not camera.have_v4l2_ctl():
        rows.append(("camera controls", None, "v4l2-ctl not found (install v4l-utils); camera would run uncontrolled"))
        return rows
    try:
        report = camera.compare_lock(camera.V4l2Ctl(c920.node), lock)
    except OSError as e:
        rows.append(("camera controls", False, str(e)))
        return rows
    rows.append(("camera control names", True, report.name_set))
    for r in report.rows:
        detail = f"now {r.readback}, lock {r.value}" + ("" if r.ok else " (applied when the camera opens)")
        rows.append((f"  {r.device_name}", True if r.ok else None, detail))
    rows += [(f"  {n}", None, "not on this device") for n in report.absent]
    rows.append(("camera check", None, "uv run python -m lightwork.vision.capture [--bench 300 | --c0]"))
    return rows


SECTIONS = [("core", section_core), ("settings", section_settings), ("tools", section_tools), ("camera", section_camera)]


def configure(p) -> None:
    pass


def run(args) -> int:
    failed = False
    for title, fn in SECTIONS:
        print(f"== {title}")
        for name, ok, detail in fn():
            mark = {True: "ok  ", False: "FAIL", None: "--  "}[ok]
            print(f"  {mark} {name}: {detail}")
            failed |= ok is False
    return 2 if failed else 0
