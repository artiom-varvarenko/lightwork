"""Staged collection: live capture plus explicit physical-session identity and operator review.

Usage: lightwork collect --pack procedures/kit-return-check --layout layout.json --calib calib.json
       --session-id home-20261031-am --seed 42 --backend glimmer

Use /screen: perform the randomized card, Capture, inspect the work-area crop and
script ground truth, Accept (or Reject and recapture), then Next. Stop via q or
Ctrl-C. labels.jsonl and the seed/card order are covered by the closed manifest.
Reuse the session id for every run from the same physical session. No label is a
model prediction; unreviewed captures are excluded from metrics and training.

For gate evaluation the procedure must actually evaluate the card's step on that
captured frame. Collection does not reset or fabricate reducer transitions:
unmatched cards remain useful labelled crops but count as missing gate evidence.
"""
from pathlib import Path
import sys

COMMAND = 'collect'
HELP = 'Collect randomized operator-reviewed trials using the verified live capture pipeline.'


def configure(parser):
    from ..live.cli import add_live_arguments
    add_live_arguments(parser)
    parser.add_argument('--session-id', required=True, help='physical session grouping shared by all runs that session')
    parser.add_argument('--seed', type=int, required=True, help='logged randomization seed')
    parser.add_argument('--trials', type=Path, help='TrialCard array; default trials.json beside the step graph')


def run(args):
    from ..eval.collection import Collection, load_cards
    from ..live.cli import session_from_args, run_session
    try:
        session = session_from_args(args)
        pack_dir = args.pack if args.pack.is_dir() else args.pack.parent
        cards_path = args.trials or pack_dir / 'trials.json'
        cards = load_cards(cards_path, session.graph)
        collection = Collection(session, cards, session_id=args.session_id, seed=args.seed)
        session.set_scene_transform(collection.scene)
        run_session(session, on_started=collection.start)
        return 0
    except (ValueError, OSError, RuntimeError) as error:
        print(str(error), file=sys.stderr)
        return 2
