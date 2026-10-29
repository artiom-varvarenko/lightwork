"""Shared live/collect CLI lifecycle.

Example (calibration and layout must match the actual camera readback)::

    uv run lightwork live --pack procedures/kit-return-check --layout .lightwork/layouts/kit.json \
        --calib .lightwork/calib/bench.json --backend meta

``--source`` accepts A's by-id camera, video, or image-directory sources; a file
source still calls the selected real model backend. Only ``--replay-as-live`` is
synthetic/offline. ``--quiet`` suppresses every utterance and mutes voice input;
``--voice-dry-run`` keeps the voice HUD/pacer but never feeds commands to the core.
Space confirms (never fabricates a passed reading); r repeats, b goes back, q or
SIGINT/SIGTERM drains and closes. Server and hub workers stop after the run seals.
"""

from __future__ import annotations

import os
import select
import signal
import sys
import threading
from pathlib import Path

from ..config import load_settings
from ..contracts.layout import Calibration, Layout
from ..contracts.stepgraph import StepGraph
from ..server.app import create_app
from ..server.runner import start_server
from .session import LiveSession


def add_live_arguments(parser) -> None:
    parser.add_argument("--pack", type=Path, help="procedure directory or stepgraph.json")
    parser.add_argument("--layout", type=Path, help="accepted C4 layout JSON")
    parser.add_argument("--calib", type=Path, help="C7 calibration JSON for this camera")
    parser.add_argument("--source", help="camera by-id path/index, video, or image directory (default: C920)")
    parser.add_argument("--backend", choices=("meta", "glimmer"), help="model backend (default: LW_BACKEND)")
    parser.add_argument("--quiet", action="store_true", help="no spoken output or voice commands")
    parser.add_argument("--voice-dry-run", action="store_true", help="voice HUD only; no commands reach the reducer")
    parser.add_argument("--no-hands", action="store_true", help="explicit motion-only advisory mode")
    parser.add_argument("--no-voice", action="store_true", help="disable Muse Voice input")
    parser.add_argument("--root", type=Path, default=Path.cwd(), help="project root for settings, spend and run directories")
    parser.add_argument("--code-commit", default=os.environ.get("LW_CODE_COMMIT", "unknown"),
                        help="recorded code revision; default unknown (working tree marked dirty)")


def session_from_args(args) -> LiveSession:
    if not args.pack or not args.layout or not args.calib:
        raise ValueError("live/collect needs --pack, --layout and --calib")
    graph_path = args.pack / "stepgraph.json" if args.pack.is_dir() else args.pack
    graph = StepGraph.model_validate_json(graph_path.read_bytes())
    layout = Layout.model_validate_json(args.layout.read_bytes())
    calib = Calibration.model_validate_json(args.calib.read_bytes())
    flat = None
    if calib.flat_field_sha:
        # Calibration keeps its content-addressed crop alongside the C7 JSON.
        from ..vision.calib.flow import flat_field_path
        path = flat_field_path(args.calib, calib.flat_field_sha)
        if not path.is_file():
            raise ValueError(f"calibration flat field missing: {path}")
        flat = path.read_bytes()
    return LiveSession(load_settings(args.root), graph, layout, calib, source=args.source,
                       backend=args.backend, quiet=args.quiet, voice_dry_run=args.voice_dry_run,
                       hands=False if args.no_hands else None, voice=False if args.no_voice else None,
                       flat_field=flat, code_commit=args.code_commit, dirty=True)


class _Keyboard:
    def __init__(self, session: LiveSession):
        self.session = session
        self.stopping = threading.Event()
        self.thread = None
        self.saved = None

    def start(self) -> None:
        if not sys.stdin.isatty() or os.name != "posix":
            return
        import termios
        import tty
        self.fd = sys.stdin.fileno()
        self.saved = termios.tcgetattr(self.fd)
        tty.setcbreak(self.fd)
        self.thread = threading.Thread(target=self._read, name="lightwork-keyboard")
        self.thread.start()

    def _read(self) -> None:
        while not self.stopping.is_set():
            readable, _, _ = select.select([self.fd], [], [], .1)
            if readable:
                char = os.read(self.fd, 1).decode("utf-8", errors="ignore")
                if not char:
                    return
                if char in (" ", "r", "b", "q"):
                    self.session.on_key("space" if char == " " else char)

    def close(self) -> None:
        self.stopping.set()
        if self.thread is not None:
            self.thread.join()
        if self.saved is not None:
            import termios
            termios.tcsetattr(self.fd, termios.TCSADRAIN, self.saved)


def run_session(session: LiveSession, *, on_started=None) -> Path:
    """Run live/collect with ASGI, terminal keys and signals; return the sealed run path."""
    server = None
    keyboard = _Keyboard(session)
    previous = {}
    try:
        session.start()
        app = create_app(mode="live", runs_root=session.run_dir.parent,
                         frames_dirs=[session.run_dir / "frames"], voice=session.voice or None, **session.hubs)
        server = start_server(app)
        if threading.current_thread() is threading.main_thread():
            for sig in (signal.SIGINT, signal.SIGTERM):
                previous[sig] = signal.signal(sig, lambda n, _frame: session.stop(f"signal {n}"))
        keyboard.start()
        if on_started is not None:
            on_started()
        print("Open http://127.0.0.1:8000/screen; Start enables speech/mic. Space=confirm r=repeat b=back q=stop.")
        path = session.join()
        print(f"Run closed: {path}")
        return path
    finally:
        keyboard.close()
        session.stop("command exit")
        if session._thread is not None and not session.finished:
            try:
                session.join()
            except BaseException:
                pass  # preserve the command's original exception; session.error retains it
        if server is not None:
            server.stop()
        for hub in (session.render, session.control):
            hub.close()
        for sig, handler in previous.items():
            signal.signal(sig, handler)
