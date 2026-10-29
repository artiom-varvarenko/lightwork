"""Convert licensed source text/Markdown/URL/PDF using Wonder or a real Glimmer server."""
from pathlib import Path

COMMAND = 'author'
HELP = 'Author a licensed procedure; retain real calls, validate once and repair once.'


def configure(p):
    p.add_argument('source', nargs='?', help='text/Markdown/PDF file or readable HTTP(S) URL')
    p.add_argument('--slug', required=True)
    p.add_argument('--output', type=Path, help='pack directory (default procedures/<slug>)')
    p.add_argument('--backend', choices=('meta', 'glimmer'), default=None)
    p.add_argument('--licence', choices=('original', 'CC BY 3.0', 'CC BY 4.0'))
    p.add_argument('--attribution', help='author/source; CC BY also needs source URL and changes')
    p.add_argument('--privacy-confirmed', action='store_true', help='confirm no private/patient information in source')
    p.add_argument('--timeout', type=float, default=180, help='per-call deadline seconds; default 180, explicitly raise for long Glimmer authoring')
    p.add_argument('--notes', default='')
    p.add_argument('--human-edit', type=Path, help='validated replacement graph; save diff from model output without inference')


def run(args):
    import sys
    import httpx
    from lightwork.config import load_settings
    from lightwork.models.errors import ModelError
    from lightwork.author.sources import load_source
    from lightwork.author.workflow import author, human_edit

    output = args.output or Path('procedures') / args.slug
    try:
        if args.human_edit:
            graph = human_edit(output, args.human_edit)
        else:
            if not args.source or not args.licence or not args.attribution:
                raise ValueError('source, --licence and --attribution are required')
            source = load_source(args.source, licence=args.licence, attribution=args.attribution,
                                 privacy_confirmed=args.privacy_confirmed)
            settings = load_settings()
            graph = author(source, slug=args.slug, output=output, settings=settings,
                           backend=args.backend or settings.backend, timeout_s=args.timeout, notes=args.notes)
        print(f'{output / "stepgraph.json"}: valid C3, {len(graph.steps)} steps; actual records in {output / "authoring"}')
        return 0
    except (OSError, ValueError, ModelError, httpx.HTTPError) as exc:
        print(f'author: {exc}', file=sys.stderr)
        return 2
