"""Export audited, re-derived runs to a keyless static replay site."""
from __future__ import annotations

import sys
from pathlib import Path

COMMAND = "export"
HELP = "Export verified runs to a static replay site (Meta only unless --allow-synthetic)."


def configure(p) -> None:
    p.add_argument("path", type=Path, help="a run directory or directory containing runs")
    p.add_argument("--output", type=Path, default=Path("site"), help="destination site directory (default: site)")
    p.add_argument("--allow-synthetic", action="store_true", help="allow labelled local test fixtures; output cannot be published")


def run(args) -> int:
    from ..runs.export import export_site

    try:
        bundle = export_site(args.path, args.output, allow_synthetic=args.allow_synthetic)
    except (OSError, ValueError) as exc:
        print(f"export: {exc}", file=sys.stderr)
        return 1
    label = "Meta-only" if bundle["publishable"] else "SYNTHETIC TEST — NOT PUBLISHABLE"
    print(f"Exported {len(bundle['runs'])} run(s) to {args.output}: 0 diffs; {label}")
    print(f"Serve locally: python -m http.server 8080 --directory {args.output}")
    return 0
