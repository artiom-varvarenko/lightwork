"""``lightwork live``: real bench loop, or explicit offline synthetic replay-as-live."""

from __future__ import annotations

import sys
from pathlib import Path

COMMAND = "live"
HELP = "Run a camera-guided procedure; --replay-as-live exercises the same runtime offline on synthetic fixtures."


def configure(parser) -> None:
    from ..live.cli import add_live_arguments
    add_live_arguments(parser)
    parser.add_argument("--replay-as-live", type=Path, metavar="SYNTHETIC_RUN",
                        help="offline fake-camera/recorded-response diagnostic; output is runs-synthetic only")


def run(args) -> int:
    from ..live.cli import run_session, session_from_args
    from ..models.errors import ModelError
    from ..vision.capture import CameraNotFound
    try:
        if args.replay_as_live:
            if args.pack or args.layout or args.calib or args.source or args.backend:
                raise ValueError("--replay-as-live uses fixture artifacts, not --pack/--layout/--calib/--source/--backend")
            from ..live.diagnostic import replay_as_live
            result = replay_as_live(args.replay_as_live, args.root)
            print(f"SYNTHETIC OFFLINE replay-as-live: {result.run_dir}")
            print(f"verify: {len(result.verify.diffs)} diffs")
            print("source derived stream (media/request hashes normalized): "
                  + ("identical" if result.source_derived_equal else "DIFF"))
            return 0 if result.verify.ok and result.source_derived_equal else 1
        run_session(session_from_args(args))
        return 0
    except (OSError, ValueError, RuntimeError, ModelError, CameraNotFound) as exc:
        print(f"live: {exc}", file=sys.stderr)
        return 2
