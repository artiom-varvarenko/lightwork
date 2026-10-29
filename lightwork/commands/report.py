"""Deterministic offline Markdown evidence tables (no keys or network)."""
from pathlib import Path
import sys

COMMAND = 'report'
HELP = 'Generate deterministic offline Markdown evidence tables; no data means no claimed performance.'


def configure(parser):
    parser.add_argument('root', type=Path, nargs='?', default=Path('runs'))
    parser.add_argument('--include-synthetic', action='store_true')
    parser.add_argument('--evaluation', type=Path, action='append', default=[], help='saved immutable Glimmer evaluation')


def run(args):
    from ..eval.evidence import EvidenceError
    from ..eval.reporting import markdown_report
    try:
        print(markdown_report(args.root, include_synthetic=args.include_synthetic, evaluations=args.evaluation), end='')
        return 0
    except EvidenceError as error:
        print(str(error), file=sys.stderr)
        return 1
    except (ValueError, OSError) as error:
        print(str(error), file=sys.stderr)
        return 2
