"""``lightwork replay <run>``: replay a recorded run on the screen page, re-deriving every decision offline (D16).

Starts the server on 127.0.0.1:8000 and plays the run once a screen page connects; ``--no-server`` replays in the
terminal only. Exit 0 when the replay matches the log, 1 on any diff, 2 on a usage or environment error.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

COMMAND = "replay"
HELP = "Replay a recorded run on the screen page (no key, no network), re-deriving every decision offline."


def configure(p) -> None:
    p.add_argument("run", type=Path, help="a run directory, e.g. runs/<id>")
    p.add_argument("--speed", type=float, default=1.0, help="playback speed factor (default 1.0)")
    p.add_argument("--no-server", action="store_true", help="headless: print the replay in the terminal only")


def run(args) -> int:
    from ..runs.replay import ReplayEngine
    from ..runs.store import is_run_dir
    from ..runs.verify import report_lines

    if not is_run_dir(args.run):
        print(f"{args.run}: not a run directory", file=sys.stderr)
        return 2
    if not args.speed > 0:
        print("--speed must be > 0", file=sys.stderr)
        return 2
    if args.no_server:
        report = ReplayEngine(args.run, speed=args.speed).run()
        print("\n".join(report_lines(report)))
        return 0 if report.ok else 1

    from ..contracts.server import HOST, PORT
    from ..server.app import create_app
    from ..server.hub import make_hubs
    from ..server.runner import start_server

    run_dir = args.run.resolve()
    hubs = make_hubs()
    app = create_app(mode="replay", runs_root=run_dir.parent, frames_dirs=[run_dir / "frames"], **hubs)
    try:
        server = start_server(app)
    except RuntimeError as e:
        print(e, file=sys.stderr)
        return 2
    print(f"Open http://{HOST}:{PORT}/screen to watch; the replay starts when the page connects.")
    try:
        report = ReplayEngine(run_dir, speed=args.speed, hubs=hubs, wait_for_screen=True).run()
        print("\n".join(report_lines(report)))
        print("Replay finished; the page keeps the final state. Ctrl-C to quit.")
        try:
            while True:
                time.sleep(3600)
        except KeyboardInterrupt:
            pass
    finally:
        server.stop()
    return 0 if report.ok else 1
