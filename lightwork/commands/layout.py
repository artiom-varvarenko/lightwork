"""Interactive screen layout, private crop selection first; --manual is completely offline."""
from pathlib import Path

COMMAND = 'layout'
HELP = 'Map bench objects in /screen after explicit private crop selection; manual mode needs no key.'


def configure(p):
    p.add_argument('graph', type=Path, help='authored stepgraph.json')
    p.add_argument('--camera', help='camera device, video or image directory')
    p.add_argument('--manual', action='store_true', help='disable all model calls; draw geometry by hand')
    p.add_argument('--work-area', type=int, nargs=4, metavar=('X', 'Y', 'W', 'H'), help='explicit private crop in camera pixels; otherwise select on blank coordinate canvas')
    p.add_argument('--mode', choices=('normal', 'dark', 'grey'), default='dark', help='normal only for screen-only operation with no projector')
    p.add_argument('--port', type=int, default=8000)
    p.add_argument('--output', type=Path, help='new layout file; default .lightwork/layouts/<pack>-<timestamp>.json')


def run(args):
    import json
    import sys
    from datetime import datetime, timezone
    from lightwork.config import load_settings
    from lightwork.author.layout import LayoutSession
    from lightwork.author.workflow import validated_graph
    from lightwork.server.app import create_app
    from lightwork.server.hub import make_hubs
    from lightwork.server.runner import start_server
    from lightwork.vision.capture import open_camera

    camera = server = None
    try:
        settings = load_settings()
        graph = validated_graph(json.loads(args.graph.read_text()))
        camera = open_camera(args.camera, loop=True, state_dir=settings.state_dir)
        hubs = make_hubs()
        stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
        output = args.output or settings.state_dir / 'layouts' / f'{graph.id}-{stamp}.json'
        session = LayoutSession(graph, camera, hubs, output=output, settings=settings, manual=args.manual, mode=args.mode)
        if args.work_area:
            session.action('layout_work_area', {'xywh': args.work_area})
        server = start_server(create_app(mode='layout', **hubs), port=args.port)
        print(f'Layout: http://127.0.0.1:{args.port}/screen; only selected work-area crop is previewed/stored.', flush=True)
        session.run()
        print(f'Saved {output}')
        return 0
    except KeyboardInterrupt:
        return 2
    except (ValueError, RuntimeError, OSError) as exc:
        print(f'layout: {exc}', file=sys.stderr)
        return 2
    finally:
        if server:
            server.stop()
        if camera:
            camera.close()
