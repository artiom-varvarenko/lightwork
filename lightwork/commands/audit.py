"""``lightwork audit [runs/]``: every run must be complete, intact and Meta-only before it is published (D15, D21).

Exit 0 when every run passes, 1 when any run is rejected, 2 when no run is found.
"""

from __future__ import annotations

import sys
from pathlib import Path

COMMAND = "audit"
HELP = "Check that every recorded run is complete, intact and Meta-only (provenance meta); exit 1 otherwise."


def configure(p) -> None:
    p.add_argument("path", nargs="?", type=Path, default=Path("runs"), help="a run or a directory of runs (default runs/)")


def run(args) -> int:
    from ..runs.audit import audit_run
    from ..runs.store import find_runs

    runs = find_runs(args.path)
    if not runs:
        print(f"no runs found at {args.path}", file=sys.stderr)
        return 2
    rejected = 0
    for run_dir in runs:
        problems = audit_run(run_dir)
        print(f"{run_dir.name}: {'REJECTED' if problems else 'ok'}")
        for line in problems:
            print(f"  {line}")
        rejected += bool(problems)
    if len(runs) > 1:
        print(f"{len(runs)} runs, {rejected} rejected")
    return 1 if rejected else 0
