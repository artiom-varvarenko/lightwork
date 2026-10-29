"""Evaluate verified evidence, or record separate Glimmer rejudgements."""
from pathlib import Path
import sys

COMMAND = 'eval'
HELP = 'Gate, latency and Glimmer evaluation with verified evidence and explicit sample counts.'


def configure(parser):
    sub = parser.add_subparsers(dest='evaluation', required=True)
    for name in ('gate', 'latency', 'glimmer'):
        p = sub.add_parser(name)
        p.add_argument('root', type=Path, nargs='?', default=Path('runs'))
        p.add_argument('--include-synthetic', action='store_true')
        if name == 'glimmer':
            p.add_argument('--output', type=Path, required=True, help='new immutable evaluation directory')
            p.add_argument('--tag', choices=('base', 'tuned'), required=True)
            p.add_argument('--session-id', action='append', help='held-out physical session; repeat to include several')
    p = sub.add_parser('compare')
    p.add_argument('base', type=Path)
    p.add_argument('--include-synthetic', action='store_true')
    p.add_argument('tuned', type=Path)


def run(args):
    from ..contracts.canonical import canonical
    from ..eval.evidence import EvidenceError, load_runs
    from ..eval.reporting import gate_report, latency_report
    try:
        if args.evaluation == 'compare':
            from ..eval.metrics import compare_saved
            result = compare_saved(args.base, args.tuned, include_synthetic=args.include_synthetic)
        elif args.evaluation == 'glimmer':
            from ..config import load_settings
            from ..eval.glimmer import rejudge
            result = rejudge(args.root, args.output, settings=load_settings(), tag=args.tag,
                             include_synthetic=args.include_synthetic, session_ids=args.session_id)
        else:
            runs = load_runs(args.root, include_synthetic=args.include_synthetic)
            result = (gate_report if args.evaluation == 'gate' else latency_report)(runs)
            result = {'synthetic': any(r.manifest['provenance'] == 'synthetic' for r in runs),
                      'dates': sorted({r.manifest['created_utc'][:10] for r in runs}), 'metrics': result}
        print(canonical(result))
        return 0
    except EvidenceError as error:
        print(str(error), file=sys.stderr)
        return 1
    except (ValueError, OSError, RuntimeError) as error:
        print(str(error), file=sys.stderr)
        return 2
