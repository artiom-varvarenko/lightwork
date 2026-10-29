"""``lightwork verify <run|runs/>``: re-derive recorded runs offline and diff them against their logs (D1, D16).

Writes ``verify.json`` into each run. Exit 0 when every run has 0 diffs, 1 on any diff, 2 when no run is found.
"""

from __future__ import annotations

import sys
from pathlib import Path

COMMAND = "verify"
HELP = "Re-derive recorded runs offline (no key, no network) and diff them against the log; exit 1 on any diff."


def configure(p) -> None:
    p.add_argument("path", type=Path, help="a run directory, or a directory of runs such as runs/")


def run(args) -> int:
    from ..runs.store import find_runs
    from ..runs.verify import report_lines, verify_run

    runs = find_runs(args.path)
    if not runs:
        print(f"no runs found at {args.path}", file=sys.stderr)
        return 2
    failed = 0
    for run_dir in runs:
        report = verify_run(run_dir)
        print("\n".join(report_lines(report)))
        failed += not report.ok
    if len(runs) > 1:
        print(f"{len(runs)} runs, {failed} with diffs")
    return 1 if failed else 0
