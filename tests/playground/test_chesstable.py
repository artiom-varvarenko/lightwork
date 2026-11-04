"""Chess on the table (lightwork.chesstable), played through a synthetic camera with the rig's real mapping: pieces are
drawn into the frames, a hand is a big blob that moves in and out."""

from __future__ import annotations

from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from lightwork.chess import BLACK, WHITE, Board, parse_uci, square
from lightwork.chesstable import (
    ANIM_S,
    BOARD,
    SQ,
    ChessTable,
    Reading,
    board_picture,
    leans_over,
    read_expected,
    read_move,
    sq_box,
    sq_xy,
)
from lightwork.games import DWELL_S, GameHost
from lightwork.playground import Canvas, Geo

H = [[0.6653339648598842, -0.028289534729032187, -275.0564675100224],
     [0.0777983313652961, 0.6648012834346899, -238.27260326678964],
     [6.672005521996616e-05, 9.27250924355104e-05, 1.0]]  # the rig's mapping, 2 Nov
DT = 1 / 30


class FakeJob:
    def __init__(self, text):
        self.text = text

    def poll(self):
        return self.text

    def stop(self):
        pass


class Table:
    """The camera's view: a table, the board's lit squares and the pieces on it."""

    def __init__(self, geo, seed=1):
        self.geo = geo
        self.rng = np.random.default_rng(seed)
        self.base = np.full((1080, 1920, 3), (120, 130, 150), np.uint8)
        for q in square_list():
            if ((q >> 4) + (q & 7)) % 2:
                x0, y0, x1, y1 = sq_box(q)
                poly = geo.d2c([(x0, y0), (x1, y0), (x1, y1), (x0, y1)]).astype(np.int32)
                cv2.fillConvexPoly(self.base, poly, (215, 225, 235))
        self.pieces: dict[int, int] = {}  # square -> piece
        self.noise = [self.rng.integers(0, 7, self.base.shape, dtype=np.uint8) for _ in range(5)]
        self._key, self._clean, self._n = None, self.base, 0

    def setup(self):
        board = Board()
        self.pieces = {q: board.b[q] for q in square_list() if board.b[q]}

    def apply(self, board, text):
        """The move on the real board, as ``board`` says it goes; ``board`` is the position before it."""
        m = parse_uci(text)
        where = board.changes(m)
        piece = self.pieces.pop(where[0])
        if len(where) == 4:
            self.pieces[where[3]] = self.pieces.pop(where[2])
        elif len(where) == 3:
            self.pieces.pop(where[2])
        self.pieces[where[1]] = piece

    def shift(self, a, b):
        self.pieces[square(b)] = self.pieces.pop(square(a))

    def image(self, t, hand=False):
        key = tuple(sorted(self.pieces.items()))
        if key != self._key:  # drawn once per position: a game is thousands of frames
            self._key, self._clean = key, self.base.copy()
            for q, p in self.pieces.items():
                x, y = self.geo.d2c([sq_xy(q)])[0]
                colour = (225, 232, 240) if p > 0 else (35, 40, 45)  # white on the lit squares differs mostly by its rim
                cv2.circle(self._clean, (int(x), int(y)), 27, colour, -1, cv2.LINE_AA)
                cv2.circle(self._clean, (int(x), int(y)), 27, (90, 95, 100), 3, cv2.LINE_AA)
                cv2.circle(self._clean, (int(x), int(y)), 13, (150, 155, 160) if p > 0 else (70, 75, 80), 3, cv2.LINE_AA)
        img = self._clean
        if hand:
            img = img.copy()
            x, y = self.geo.d2c([(BOARD[0] + 170 + 60 * np.sin(t * 7), BOARD[1] + 200)])[0]
            cv2.ellipse(img, (int(x), int(y)), (120, 200), 0, 0, 360, (90, 120, 190), -1)
        self._n += 1
        return cv2.add(img, self.noise[self._n % len(self.noise)])


def square_list():
    return [r * 16 + f for r in range(8) for f in range(8)]


class Run:
    def __init__(self, tmp_path, replies=()):
        self.host = GameHost(state_dir=tmp_path, seed=3)
        self.geo = Geo(H, 749.0, 421.0)
        self.table = Table(self.geo)
        self.t = 0.0
        self.said: list[str] = []
        self.replies = list(replies)
        self.link = None
        self.start()

    def start(self):
        self.said += self.host.start("chess", self.t)
        self.g.think = lambda played: FakeJob(self.replies.pop(0))

    @property
    def g(self) -> ChessTable:
        return self.host.game

    def frames(self, seconds, hand=False, points=()):
        end = self.t + seconds
        while self.t < end:
            self.t += DT
            say, _ = self.host.update(self.t, list(points), [])
            self.said += say
            g = self.host.game
            if isinstance(g, ChessTable):
                g.place(self.geo)
                g.watch(SimpleNamespace(image=self.table.image(self.t, hand), t_mono=self.t), self.t, self.link)
                self.said += g.say
                g.say = []
            self.host.draw(Canvas(self.geo), self.t)

    def hand(self):
        self.frames(0.4, hand=True)

    def settle(self):
        self.frames(2.6)

    def to_play(self):
        self.frames(1.6)
        assert self.g.state == "setup"
        assert not self.g.placed
        self.hand()
        self.table.setup()
        self.hand()
        self.settle()
        assert self.g.state == "you", self.g.log
        self.g.log.clear()

    def you(self, text):
        self.hand()
        self.table.apply(self.g.board, text)
        self.hand()
        self.settle()

    def mine(self):
        """Makes the move the table showed."""
        assert self.g.state == "show"
        self.hand()
        from lightwork.chess import uci
        self.table.apply(self.g.board, uci(self.g.reply))
        self.hand()
        self.settle()


def prims(run, kind=None):
    c = Canvas(run.geo)
    run.g.draw(c, run.t)
    return [p for p in c.prims if kind is None or p.kind == kind]


def test_the_board_fits_between_the_panels_and_the_status_lines():
    from lightwork.tabletop import COL_X1, STATUS_Y1

    assert BOARD[0] - 16 > COL_X1 + 100  # the clocks at its left
    assert BOARD[1] > STATUS_Y1
    assert BOARD[3] + 16 <= 420.0
    assert BOARD[2] + 10 < 552.0 + 1
    assert BOARD[2] - BOARD[0] == BOARD[3] - BOARD[1] == 8 * SQ


def test_a1_is_dark_and_h1_lit_in_the_picture():
    img = board_picture()
    px, margin = 3, 16 * 3
    s = int(SQ * px)
    bottom = 2 * px + 8 * s
    assert img[bottom - s // 2, margin + s // 2].max() == 0  # a1
    assert img[bottom - s // 2, margin + 7 * s + s // 2].min() == 255  # h1


def test_setting_up_shows_every_piece_then_starts_by_itself(tmp_path):
    run = Run(tmp_path)
    run.frames(1.6)
    assert run.g.state == "setup"
    ghosts = [p for p in prims(run, "label") if p.id.startswith("chess-ghost-")]
    assert len(ghosts) == 32
    assert run.host.hud(run.t)[0].startswith("PUT THE PIECES ON THE BOARD: 0 OF 32")
    for q in square_list()[:8]:  # the near rank first: what is placed loses its mark
        run.table.pieces[q] = Board().b[q]
    run.hand()
    run.settle()
    assert run.g.state == "setup"
    assert len(run.g.placed) == 8, run.g.log
    assert len([p for p in prims(run, "label") if p.id.startswith("chess-ghost-")]) == 24
    run.hand()
    run.table.setup()
    run.hand()
    run.settle()
    assert run.g.state == "you"
    assert any("Your move" in s for s in run.said)


def test_pieces_already_standing_when_it_starts_count(tmp_path):
    run = Run(tmp_path)
    run.table.setup()
    run.frames(1.6)
    assert run.g.state == "you", run.g.log


def test_the_projectors_reflection_on_an_empty_square_is_not_a_piece(tmp_path):
    run = Run(tmp_path)
    x, y = run.geo.d2c([sq_xy(square("d7"))])[0]
    hot = run.table.base.astype(np.float32)
    for k in range(6):  # a bright reflection with a little structure of its own (rig, 4 Nov: d7 read 14, empty)
        cv2.circle(hot, (int(x) + 7 * k - 18, int(y) - 6 + 3 * k), 9, (255, 255, 255), -1)
    run.table.base = np.clip(cv2.GaussianBlur(hot, (0, 0), 2.5), 0, 255).astype(np.uint8)
    run.table._key = None
    run.frames(1.6)
    assert square("d7") not in run.g.placed, run.g.log[-2:]
    run.hand()
    run.table.setup()
    run.hand()
    run.settle()
    assert run.g.state == "you", run.g.log[-2:]


def test_a_piece_taken_away_again_needs_its_place_shown_again(tmp_path):
    run = Run(tmp_path)
    for q in square_list()[:8]:
        run.table.pieces[q] = Board().b[q]
    run.frames(1.6)
    assert len(run.g.placed) == 8
    run.hand()
    del run.table.pieces[0]
    run.hand()
    run.settle()
    assert len(run.g.placed) == 7
    assert "chess-ghost-0" in {p.id for p in prims(run, "label")}


def test_the_start_bubble_starts_with_pieces_missing(tmp_path):
    run = Run(tmp_path)
    run.frames(1.6)
    run.frames(0.6 + DWELL_S + 0.2, points=[run.g.start_bubble.c])
    assert run.g.state in ("ready", "you")
    run.settle()
    assert run.g.state == "you"


def test_a_full_turn_read_animated_replied_and_checked(tmp_path):
    run = Run(tmp_path, ["e7e5"])
    run.to_play()
    left = run.g.left[WHITE]
    run.you("e2e4")
    assert run.g.moves == ["e4"], run.g.log
    assert run.g.left[WHITE] < left
    # the reply waits for the animation, then shows where it goes
    assert run.g.state == "show"
    assert any(s.startswith("Pawn to e5") for s in run.said)
    ids = {p.id for p in prims(run)}
    assert {"chess-from", "chess-to", "chess-arrow"} <= ids
    assert run.host.hud(run.t)[0] == "MY MOVE: PAWN E7 TO E5"
    black_left = run.g.left[BLACK]
    run.frames(0.5)
    assert run.g.left[BLACK] == black_left  # the table's clock does not run while the player makes its move
    run.mine()
    assert run.g.state == "you"
    assert run.g.moves == ["e4", "e5"]
    assert "chess-from" not in {p.id for p in prims(run)}


def test_the_move_is_animated_before_the_reply_shows(tmp_path):
    run = Run(tmp_path, ["e7e5"])
    run.to_play()
    run.hand()
    run.table.apply(run.g.board, "e2e4")
    run.hand()
    while run.g.state == "you":
        run.frames(DT)
    assert run.g.state == "think"
    ids = {p.id for p in prims(run)}
    assert "chess-trail" in ids and "chess-flying" in ids
    run.frames(ANIM_S - 0.1)
    assert run.g.state == "think"  # the engine answered at once, but the flight is not over
    run.frames(0.2)
    assert run.g.state == "show"


def test_marks_go_off_for_the_clean_picture_of_the_tables_move(tmp_path):
    run = Run(tmp_path, ["e7e5"])
    run.to_play()
    run.you("e2e4")
    run.hand()
    run.table.apply(run.g.board, "e7e5")
    run.hand()
    seen_blank = False
    for _ in range(90):
        run.frames(DT)
        if run.g.state != "show":
            break
        if run.g._scan is not None and run.g._scan.blank:
            seen_blank = True
            assert "chess-from" not in {p.id for p in prims(run)}
            assert "chess-board" in {p.id for p in prims(run)}
    assert seen_blank
    assert run.g.state == "you"


def test_another_piece_moved_for_the_table_is_refused(tmp_path):
    run = Run(tmp_path, ["e7e5"])
    run.to_play()
    run.you("e2e4")
    run.hand()
    run.table.shift("d7", "d5")
    run.hand()
    run.settle()
    assert run.g.state == "show"
    assert run.g.trouble and set(run.g.trouble) <= {square("d7"), square("d5")}
    assert any(s.startswith("Not that one") for s in run.said)
    run.hand()
    run.table.shift("d5", "d7")
    run.table.shift("e7", "e5")
    run.hand()
    run.settle()
    assert run.g.state == "you"
    assert not run.g.trouble


def test_an_illegal_move_is_refused_and_the_game_waits(tmp_path):
    run = Run(tmp_path)
    run.to_play()
    run.hand()
    run.table.shift("e2", "e5")  # three squares: no pawn does that
    run.hand()
    run.settle()
    assert run.g.state == "you"
    assert run.g.moves == []
    assert run.g.trouble
    assert any("legal" in s for s in run.said)
    assert run.host.hud(run.t)[2] == "error"
    run.hand()
    run.table.shift("e5", "e2")
    run.hand()
    run.settle()
    assert not run.g.trouble
    assert run.g.state == "you"


def test_the_tables_move_and_the_players_own_made_in_one_go_are_both_read(tmp_path):
    run = Run(tmp_path, ["e7e5", "g8f6"])
    run.to_play()
    run.you("e2e4")
    run.hand()
    run.table.apply(run.g.board, "e7e5")  # no hand away between the two
    run.table.shift("g1", "f3")
    run.hand()
    run.settle()
    assert run.g.moves[:3] == ["e4", "e5", "Nf3"], run.g.log[-4:]
    run.frames(0.8)
    assert run.g.state == "show"


def test_the_players_move_onto_the_square_the_tables_piece_just_left_in_one_go(tmp_path):
    # 1.e4 e5 2.f4 exf4 3.e5, the last two without a hand away (rig, 4 Nov): e5 changes either way, only e4 says anything
    run = Run(tmp_path, ["e7e5", "e5f4", "g8f6"])
    run.to_play()
    run.you("e2e4")
    run.mine()
    run.you("f2f4")
    assert run.g.state == "show"
    run.hand()
    run.table.apply(run.g.board, "e5f4")
    run.table.shift("e4", "e5")
    run.hand()
    run.settle()
    assert run.g.moves[-2:] == ["exf4", "e5"], run.g.log[-5:]


def test_the_player_takes_the_piece_the_table_just_moved_in_one_go(tmp_path):
    # 1.d3 e5 2.a3 e4 3.dxe4: the pawn takes the moment the table's pawn was put down
    run = Run(tmp_path, ["e7e5", "e5e4", "b8c6"])
    run.to_play()
    run.you("d2d3")
    run.mine()
    run.you("a2a3")
    assert run.g.state == "show"
    run.hand()
    run.table.apply(run.g.board, "e5e4")
    run.table.pieces[square("e4")] = run.table.pieces.pop(square("d3"))
    run.hand()
    run.settle()
    assert run.g.moves[-2:] == ["e4", "dxe4"], run.g.log[-5:]


def _flat(view, q):
    from lightwork.chesstable import EYE

    y, x = (7 - (q >> 4)) * EYE, (q & 7) * EYE
    patch = view[y:y + EYE, x:x + EYE]
    patch[:] = np.median(patch.reshape(-1, 3), axis=0)
    return patch


def _piece(view, q, colour=(35, 40, 45)):
    from lightwork.chesstable import EYE

    patch = _flat(view, q)
    cv2.circle(patch, (EYE // 2, EYE // 2), 18, colour, -1)
    cv2.circle(patch, (EYE // 2, EYE // 2), 8, (150, 155, 160), 2)


def test_a_piece_that_barely_shows_leaving_its_square_is_still_read_in_one_go(tmp_path):
    """A white knight leaving a washed-out lit square changes it by about 10 levels (rig, 4 Nov): under the level that says
    changed, but its square reads empty now, and it must stay in the reference for the next reading."""
    from lightwork.chesstable import EYE

    run = Run(tmp_path, ["e7e5"])
    run.to_play()
    run.you("e2e4")
    assert run.g.state == "show"
    g = run.g
    g1 = square("g1")
    ref = g.refs[-1].copy()
    patch = _flat(ref, g1)  # the knight as it looked: hardly different from the square
    cv2.circle(patch, (EYE // 2, EYE // 2), 16, (int(patch[0, 0, 0]) + 12, int(patch[0, 0, 1]) + 12, int(patch[0, 0, 2]) + 12), 2)
    g.refs[-1] = ref
    cur = ref.copy()
    _flat(cur, square("e7"))
    _piece(cur, square("e5"))
    _flat(cur, g1)
    _piece(cur, square("f3"), (200, 210, 220))
    g._read_mine(cur, run.t)  # first picture
    g._read_mine(cur, run.t)  # the second agrees
    assert g.moves[-1] == "e5" and g.state == "you", g.log[-3:]
    assert g.visit  # read at once: no hand has to pass again
    g.think = lambda played: FakeJob("b8c6")
    g._read_yours(cur, run.t)
    g._read_yours(cur, run.t)
    assert g.moves[-1] == "Nf3", g.log[-3:]


def test_a_piece_that_never_moved_is_not_read_as_the_one_that_did():
    """Something landed on f3 and the king on e2 reads about 10 levels of noise: he still stands there (rig, 4 Nov: Kf3)."""
    from lightwork.chesstable import square_contents  # noqa: F401

    board = Board("4k3/8/8/8/8/8/4K3/6N1 w - - 0 1")
    contents = dict.fromkeys(square_list(), 4.0)
    contents.update({square("e2"): 42.0, square("f3"): 35.0, square("e8"): 40.0})
    r = read_move(changes(f3=39.0, f2=20.0, e2=10.0, g1=10.0), board, contents, nadir=(3.0, 5.0))
    assert r.move == parse_uci("g1f3")


def test_one_square_changed_and_a_piece_standing_where_it_can_go_is_that_move():
    board = Board()
    contents = dict.fromkeys(square_list(), 3.0)
    contents.update({q: 40.0 for q in square_list() if board.b[q]})
    contents[square("e2")] = 3.0
    contents[square("e4")] = 40.0
    r = read_move(changes(e2=50.0), board, contents)
    assert r.kind == "move" and r.move == parse_uci("e2e4")
    contents[square("e4")] = 3.0  # lifted, and not put down yet
    assert read_move(changes(e2=50.0), board, contents).kind == "partial"


def test_the_tables_move_is_done_only_when_what_stands_there_says_so():
    where = (square("b8"), square("c6"))
    contents = dict.fromkeys(square_list(), 3.0)
    contents[square("c6")] = 60.0
    assert read_expected(changes(b8=30.0, c6=28.0), where, None, contents).kind == "move"
    contents[square("b8")] = 60.0  # a knight on each: one changed a little, the other is still there
    assert read_expected(changes(b8=12.0, c6=28.0), where, None, contents).kind == "partial"


def test_a_nudged_piece_is_not_a_wrong_move(tmp_path):
    run = Run(tmp_path, ["e7e5"])
    run.to_play()
    g = run.g
    cur = g.refs[-1].copy()
    from lightwork.chesstable import EYE

    for name in ("g1", "e2"):  # the pieces there turned a little: their squares change, both still hold a piece
        q = square(name)
        y, x = (7 - (q >> 4)) * EYE, (q & 7) * EYE
        cur[y:y + EYE, x:x + EYE] = np.roll(cur[y:y + EYE, x:x + EYE], 9, axis=1)
    g._read_yours(cur, run.t)
    g._read_yours(cur, run.t)
    assert not g.trouble and g.moves == [], g.log[-3:]
    assert any("nudged" in line for line in g.log)
    run.you("e2e4")  # and the next move reads against the board as it stands now
    assert g.moves == ["e4"], g.log[-3:]


def test_board_ok_makes_what_stands_there_the_position(tmp_path):
    run = Run(tmp_path)
    run.to_play()
    run.hand()
    run.table.shift("e2", "e5")
    run.hand()
    run.settle()
    assert run.g.trouble
    assert "chess-bub-ok" not in {p.id for p in prims(run)}
    assert "bub-ok" in {p.id for p in prims(run)}
    run.frames(0.6 + DWELL_S + 0.2, points=[run.g.ok_bubble.c])
    assert not run.g.trouble
    run.settle()
    assert not run.g.refresh_ref
    run.hand()
    run.settle()
    assert not run.g.trouble and run.g.moves == []  # nothing is out of place any more


def test_the_chess_tile_asks_for_a_longer_exposure_and_the_others_do_not(tmp_path):
    host = GameHost(state_dir=tmp_path)
    host.start("chess", 0.0)
    assert host.game.exposure == 78
    host.start("rps", 0.0)
    assert host.game.exposure is None


def test_a_move_counts_only_when_a_second_picture_agrees(tmp_path):
    run = Run(tmp_path, ["e7e5"])
    run.to_play()
    run.hand()
    run.table.apply(run.g.board, "e2e4")
    run.hand()
    while run.g.candidate is None:
        run.frames(DT)
    assert run.g.moves == []  # one picture said so
    run.frames(0.1)
    run.hand()  # ... the hand was still there: it moves again, and the picture is forgotten
    assert run.g.candidate is None and run.g.moves == []
    run.settle()
    assert run.g.moves == ["e4"]


def test_nothing_is_read_when_the_page_does_not_show_the_bare_board(tmp_path):
    run = Run(tmp_path, ["e7e5"])
    run.to_play()
    run.you("e2e4")
    assert run.g.state == "show"

    class DeadPage:
        def version(self):
            return 1

        def acked_at(self, version):
            return None

    run.link = DeadPage()
    run.hand()
    run.table.shift("d7", "d5")
    run.hand()
    run.settle()
    assert run.g.state == "show" and not run.g.trouble  # no reading
    assert any("does not answer" in line for line in run.g.log)
    run.link = None
    run.settle()
    assert run.g.trouble  # the page is back: now it looks, and d7-d5 was not asked for


def test_a_broken_camera_frame_is_not_a_hand():
    from lightwork.chesstable import BoardEye

    geo = Geo(H, 749.0, 421.0)
    eye = BoardEye(geo.d2c([(BOARD[0], BOARD[1]), (BOARD[2], BOARD[1]), (BOARD[2], BOARD[3]), (BOARD[0], BOARD[3])]))
    still = np.full((1080, 1920, 3), 120, np.uint8)
    broken = still.copy()
    broken[500:] = 128 - 60
    frames = [still] * 6 + [broken] + [still] * 6
    assert not any(eye.moving(f) for f in frames)
    moving = []
    for k in range(10):
        f = still.copy()
        cv2.circle(f, (1000 + 40 * k, 600), 80, (30, 60, 200), -1)
        moving.append(eye.moving(f))
    assert any(moving)


def test_a_hand_that_passes_over_changes_nothing(tmp_path):
    run = Run(tmp_path)
    run.to_play()
    scans = run.g.scans
    run.hand()
    run.settle()
    assert run.g.scans > scans
    assert run.g.state == "you" and run.g.moves == [] and not run.g.trouble


def test_nothing_is_read_while_a_hand_rests_on_the_board(tmp_path):
    run = Run(tmp_path)
    run.to_play()
    scans = run.g.scans
    run.frames(2.0, points=[sq_xy(square("e4"))])  # tracked, not moving
    assert run.g.scans == scans


def test_captures_castling_and_en_passant_are_read(tmp_path):
    # 1.e4 d5 2.exd5 Nf6 3.Nf3 Nxd5 4.Bc4 Nb6 5.O-O a6 6.d4 e6 7.d5 c5 8.dxc6
    run = Run(tmp_path, ["d7d5", "g8f6", "f6d5", "d5b6", "a7a6", "e7e6", "c7c5", "a6a5"])
    run.to_play()
    for text in ("e2e4", "e4d5", "g1f3", "f1c4", "e1g1", "d2d4", "d4d5", "d5c6"):
        run.you(text)
        assert run.g.state == "show", (text, run.g.moves, run.g.log[-3:])
        run.mine()
        assert run.g.state == "you", (text, run.g.log[-3:])
    assert run.g.moves[:3] == ["e4", "d5", "exd5"]
    assert run.g.moves[8] == "O-O"
    assert run.g.moves[14] == "dxc6"  # taken in passing
    assert square("c5") not in run.table.pieces


def test_the_table_castles_and_the_rook_is_shown_too(tmp_path):
    # 1.a3 Nf6 2.a4 g6 3.a5 Bg7 4.h3 O-O
    run = Run(tmp_path, ["g8f6", "g7g6", "f8g7", "e8g8"])
    run.to_play()
    for k, text in enumerate(("a2a3", "a3a4", "a4a5", "h2h3")):
        run.you(text)
        if k == 3:
            ids = {p.id for p in prims(run)}
            assert {"chess-rook-from", "chess-rook-to", "chess-rook-arrow"} <= ids
            assert any(s.startswith("Castle kingside") for s in run.said)
        run.mine()
    assert run.g.moves[-1] == "O-O"
    assert run.table.pieces[square("f8")] < 0 and run.table.pieces[square("g8")] < 0


def test_undo_takes_the_move_back_and_the_board_must_follow(tmp_path):
    run = Run(tmp_path, ["e7e5", "d7d5"])
    run.to_play()
    run.you("e2e4")
    run.mine()
    assert run.g.moves == ["e4", "e5"]
    run.frames(0.6 + DWELL_S + 0.2, points=[run.g.undo_bubble.c])
    assert run.g.moves == []
    assert run.g.state == "you"
    assert "PUT BACK" in run.host.hud(run.t)[1]
    run.hand()
    run.table.shift("e5", "e7")
    run.table.shift("e4", "e2")
    run.hand()
    run.settle()
    assert not run.g.trouble and run.g.moves == []
    run.you("e2e4")
    assert run.g.moves == ["e4"]


def test_the_clock_falls_and_ends_the_game(tmp_path):
    run = Run(tmp_path)
    run.to_play()
    run.g.left[WHITE] = 0.2
    run.frames(0.3)
    assert run.g.over
    assert run.g.result == ("time", BLACK)
    assert any("time is up" in s for s in run.said)


def test_checkmate_by_the_table_ends_after_the_player_made_the_move_and_play_again_keeps_the_empty_board(tmp_path):
    run = Run(tmp_path, ["e7e5", "d8h4"])
    run.to_play()
    run.you("f2f3")
    run.mine()
    run.you("g2g4")
    assert run.g.state == "show"
    assert any("checkmate" in s for s in run.said)
    run.mine()
    assert run.g.over and run.g.result == ("checkmate", BLACK)
    assert run.host.hud(run.t)[0] == "CHECKMATE: TABLE WINS"
    assert any("Checkmate. I win" in s for s in run.said)
    run.frames(1.4)  # the bubbles can be picked a moment later, and stand beside the board
    assert run.host.again is not None and run.host.again.c[0] < BOARD[0]
    run.frames(DWELL_S + 0.3, points=[run.host.again.c])
    run.g.think = lambda played: FakeJob("e7e5")
    assert run.g.state == "setup" and not run.g.over
    run.hand()
    run.table.setup()
    run.hand()
    run.settle()
    assert run.g.state == "you"


def test_recalibrating_retakes_the_reference(tmp_path):
    run = Run(tmp_path)
    run.to_play()
    run.geo = Geo(np.array(H) @ np.array([[1, 0, 3.0], [0, 1, 2.0], [0, 0, 1.0]]), 749.0, 421.0)
    run.frames(1.0)
    assert not run.g.refresh_ref
    assert run.g.state == "you" and not run.g.trouble


def test_it_says_so_when_the_camera_cannot_see_the_whole_board(tmp_path):
    run = Run(tmp_path)
    run.frames(0.1)
    assert not run.g.cut_off
    moved = Geo(np.array(H) @ np.array([[1, 0, 0.0], [0, 1, 700.0], [0, 0, 1.0]]), 749.0, 421.0)  # the camera turned away
    run.g.place(moved)
    assert run.g.cut_off
    assert run.host.hud(run.t)[0] == "THE CAMERA CANNOT SEE THE WHOLE BOARD"


def test_the_hud_never_overruns_a_status_line(tmp_path):
    run = Run(tmp_path, ["g8f6"])
    run.to_play()
    run.you("e2e4")
    for line in run.host.hud(run.t)[:2]:
        assert len(line) <= 44


# ---- reading a pattern of changed squares


def changes(**hot):
    out = dict.fromkeys(square_list(), 3.0)
    out.update({square(k): v for k, v in hot.items()})
    return out


def test_read_move_takes_the_legal_move_the_squares_say():
    r = read_move(changes(e2=40.0, e4=35.0), Board())
    assert r.kind == "move" and r.move == parse_uci("e2e4")


def test_read_move_picks_the_landing_square_over_a_tall_pieces_lean():
    board = Board("4k3/8/8/8/8/8/8/3QK3 w - - 0 1")
    r = read_move(changes(d1=40.0, d4=38.0, d5=16.0), board)  # the queen's head leans over d5
    assert r.move == parse_uci("d1d4")


def test_read_move_accepts_white_on_white_as_weak_as_it_comes():
    r = read_move(changes(e2=15.0, e4=9.0), Board())
    assert r.move == parse_uci("e2e4")


def test_read_move_one_square_is_not_a_move_yet():
    r = read_move(changes(e2=40.0), Board())
    assert r == Reading("partial", None, (square("e2"),))


def test_read_move_nothing_and_too_much():
    assert read_move(changes(), Board()).kind == "none"
    busy = {q: 3.0 for q in square_list()}
    for q in square_list()[8:22]:
        busy[q] = 50.0
    assert read_move(busy, Board()).kind == "busy"


def test_read_move_says_when_no_move_fits():
    assert read_move(changes(e2=40.0, e6=40.0), Board()).kind == "wrong"


def test_read_move_rejects_a_move_that_explains_less_than_what_else_changed():
    # a queen held still over d1 and d2, and 9 levels of nothing on f3 (rig, 4 Nov)
    board = Board("rnbqkbnr/pp1ppppp/2p5/8/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2")
    r = read_move(changes(d2=54.0, d1=28.0, c5=18.0, f3=9.0), board)
    assert r.kind == "wrong"


def test_a_hand_resting_over_several_squares_is_not_a_move():
    # rig, 4 Nov: f1 69, d1 58, e5 55, e2 48, e1 45, d2 30: read as Qe2
    board = Board("rnbqkb1r/pppp1ppp/5n2/4P3/5p2/2N5/PPPP2PP/R1BQKBNR b KQkq - 0 4")
    board.push(parse_uci("f6g8"))
    contents = dict.fromkeys(square_list(), 3.0)
    contents.update({q: 40.0 for q in square_list() if board.b[q]})
    contents[square("e2")] = contents[square("f1")] = 40.0
    contents[square("d1")] = 5.0
    r = read_move(changes(f1=69.0, d1=58.0, e5=55.0, e2=48.0, e1=45.0, d2=30.0), board, contents)
    assert r.kind == "wrong"


def test_a_tall_pieces_head_is_not_held_against_its_move():
    nadir = (3.0, 5.0)  # under the camera: about d6
    assert square("f2") in leans_over(square("f3"), nadir)
    assert square("g4") not in leans_over(square("f3"), nadir)
    assert leans_over(square("d6"), nadir) == ()
    board = Board("rnbqkbnr/pp1ppppp/2p5/8/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2")
    r = read_move(changes(f2=45.0, f3=36.0, d1=35.0), board, nadir=nadir)
    assert r.move == parse_uci("d1f3")  # not the pawn f2-f3, though f2 changed most


def test_a_head_over_the_next_square_is_not_where_the_piece_landed():
    nadir = (3.0, 5.0)
    board = Board("4k3/8/8/8/8/8/8/3QK3 w - - 0 1")
    # the queen went to g4: it stands on g4, its head reaches over h3
    r = read_move(changes(d1=40.0, g4=30.0, h3=31.0), board, nadir=nadir)
    assert r.move == parse_uci("d1g4")


def test_a_square_that_says_nothing_can_be_landed_on():
    board = Board("4k3/8/8/8/4P3/8/8/4K3 w - - 0 1")
    r = read_move(changes(e4=50.0), board, unknown={square("e5")})
    assert r.move == parse_uci("e4e5")
    assert read_move(changes(e4=50.0), board).kind == "partial"
    assert read_move(changes(), board, unknown={square("e5")}).kind == "none"


def test_read_move_takes_castling_over_the_rook_or_the_king_alone():
    board = Board("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1")
    r = read_move(changes(e1=30.0, f1=30.0, g1=30.0, h1=30.0), board)
    assert r.move == parse_uci("e1g1")


def test_a_noisy_pair_of_pictures_raises_the_level():
    noisy = {q: 12.0 for q in square_list()}
    assert read_move(noisy, Board()).kind == "none"  # 12 everywhere is noise (or a light change), not a move


@pytest.mark.parametrize("hot,kind", [
    ({"b8": 30.0, "c6": 28.0}, "move"),
    ({"b8": 30.0}, "partial"),
    ({"c6": 30.0}, "partial"),
    ({"b8": 30.0, "a6": 28.0}, "wrong"),
    ({}, "none"),
])
def test_read_expected(hot, kind):
    where = (square("b8"), square("c6"))
    assert read_expected(changes(**hot), where).kind == kind
