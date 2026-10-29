"""Build and audit session-isolated short-verdict training data."""
from pathlib import Path
import sys

COMMAND = "dataset"
HELP = "Build/audit reviewed Meta-only session-split verdict datasets."


def configure(p):
    sub = p.add_subparsers(dest="action", required=True)
    build = sub.add_parser("build")
    build.add_argument("path", type=Path)
    build.add_argument("--output", type=Path, required=True)
    build.add_argument("--holdout-session", action="append", required=True)
    build.add_argument("--seed", type=int, default=0)
    build.add_argument("--mode", choices=("full", "smoke"), default="full")
    build.add_argument("--licence", required=True, help="operator-attested licence for the images and labels")
    build.add_argument("--attribution", required=True)
    build.add_argument("--allow-synthetic-for-tests", action="store_true",
                       help="test-only output, permanently ineligible for release/training upload")
    audit = sub.add_parser("audit")
    audit.add_argument("path", type=Path)
    audit.add_argument("--test-only", action="store_true", help="integrity only; NOT a release audit")
    audit.add_argument("--source-runs", type=Path, help="original run tree, required for release authentication")


def run(args):
    from ..contracts.canonical import canonical
    from ..eval.dataset import audit_dataset, build_dataset
    try:
        if args.action == "build":
            result = build_dataset(args.path, args.output, holdout_sessions=args.holdout_session,
                                   seed=args.seed, mode=args.mode, licence=args.licence, attribution=args.attribution,
                                   allow_synthetic_for_tests=args.allow_synthetic_for_tests)
        else:
            result = audit_dataset(args.path, release=not args.test_only, source_root=args.source_runs)
        print(canonical({k: result[k] for k in ("mode", "counts", "provenance", "release_eligible")}))
        return 0
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(f"dataset: {exc}", file=sys.stderr)
        return 2 if args.action == "build" else 1
