"""``lightwork doctor``: environment checks. Each section is a function returning (name, ok, detail) rows.

Sections are owned by their workstreams (camera: A, Tank/adb: J, models: F). Missing optional pieces are reported,
not failed; only a broken core install exits 2.
"""

from __future__ import annotations

import platform
import shutil
import sys

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
    try:
        import mediapipe

        rows.append(("mediapipe (live extra)", True, mediapipe.__version__))
    except Exception:  # noqa: BLE001
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


SECTIONS = [("core", section_core), ("settings", section_settings), ("tools", section_tools)]


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
