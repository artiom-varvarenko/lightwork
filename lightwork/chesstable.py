"""Chess against the table, with real pieces on a board made of light (``--game chess``, or the menu's CHESS tile).

The table draws the board and shows where each piece goes until the camera sees one standing there; the player sets the
pieces up and plays White from the near side. The player moves their own pieces: the camera reads the move, the table animates it and replies (:mod:`lightwork.chess`,
in its own process, no model). The table has no hands, so it shows its move, says it, and the player makes it; the camera
checks that it was made. The moves run down the right of the board, the two clocks stand on its left.

How a move is read: every comparison is of two clean pictures of the board, each taken with the view still and no hand
over it, with only the bare board projected (what we draw on top goes off for a moment first, and the page's ACK is waited
for, like the helper's look). Those pictures are rectified to the board and compared square by square against the one
taken after the last move; the legal moves say which pattern of changed squares means what. Nothing has to tell one
piece from another, so white on white only has to change a square a little.
"""

from __future__ import annotations

import hashlib
import math
import subprocess
import sys
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from . import chess
from .chess import BLACK, KNIGHT, PIECE_NAMES, SQUARES, WHITE, Board, move_from, move_to, parse_uci, square_name, uci
from .games import Bubble, Game, _label_c

SQ = 42.0  # design px per square: about 46 mm on the rig
BOARD = (206.0, 66.0, 206.0 + 8 * SQ, 66.0 + 8 * SQ)  # design px; rank 1 along the near edge, file a on the player's left
MARGIN = 16.0  # design px of the board's picture left of and below the squares: the coordinates
LEFT_CX = 130.0  # the left panel's middle
LIST_X = (552.0, 584.0, 660.0)  # the move list's columns: number, White, Black
LIST_Y0, LIST_STEP, LIST_ROWS = 80.0, 27.0, 13
CLOCK_S = 600.0  # 10 minutes each
THINK_S = 2.0
FONT = 24.6  # design px: a label's height (projector_style.TEXT_PROJ_PX on the rig's page)

ANIM_S = 1.0  # the player's move is animated this long
STILL_S = 0.5  # the view has not moved for this long ...
SCAN_FRAMES = 6  # ... then this many frames are averaged: 100 Hz room lights beat with 30 fps over 3 frames
AFTER_ACK_S = 0.3  # frames this long after the page drew the bare board (games fade things out over 0.18 s)
ACK_WAIT_S = 1.0
AGREE = 8.0  # mean levels between the first and last frame of a scan above which something moved
IGNORE_S = 0.45  # what we draw on the board just changed: the camera sees that for this long
CONFIRM_GAP = 0.25  # a move or a complaint counts when a second clean picture this long after the first says the same
EYE = 64  # px per square of the rectified view
THUMB = 160  # px across the motion view
MOTION_DIFF = 18  # levels of change over 3 frames that count as moving
MOTION_FRAC = 0.002  # share of the board that must change

HOT = 14.0  # levels (mean of the most changed part of a square's middle) that say a square changed ...
WEAK = 0.6  # ... with the others a move needs allowed this share of that: white on white changes little
FLOOR = 2.5  # ... and at least this many times what the quietest squares read
BUSY = 10  # more squares changed than this: something lies across the board, not a move
TOP_SHARE = 35  # per cent of a square's middle that is averaged: a piece's base covers about that
DISC = 0.38  # of a square's side: the radius of its middle; a tall neighbour's head leans in only from the edge
FILLED = 11.0  # levels of detail inside a square that say something stands on it (rig, 4 Nov: empty 1-9, pieces 15-74) ...
EMPTY_TYPICAL = 7.0  # ... once each square's own empty reading is taken off down to this: the projector's reflection on the
EMPTY_MAX = 16.0  # glossy table reads up to 14 on an empty square (4 Nov, d7); higher than this is never an empty square

GLYPHS = "♙♘♗♖♕♔"  # the outlined ones for both sides, told apart by colour: the solid pawn is a colour emoji on phones
ROLE_BGR = {"target": (0, 255, 0), "source": (255, 255, 0), "error": (0, 0, 255), "warn": (0, 255, 255)}


def sq_box(sq: int) -> tuple[float, float, float, float]:
    x0, y0 = BOARD[0] + (sq & 7) * SQ, BOARD[1] + (7 - (sq >> 4)) * SQ
    return x0, y0, x0 + SQ, y0 + SQ


def sq_xy(sq: int) -> tuple[float, float]:
    return BOARD[0] + ((sq & 7) + 0.5) * SQ, BOARD[1] + (7 - (sq >> 4) + 0.5) * SQ


def _ease(f: float) -> float:
    return 1 - (1 - f) ** 3


def _on_board(p, pad: float = 8.0) -> bool:
    return BOARD[0] - pad <= p[0] <= BOARD[2] + pad and BOARD[1] - pad <= p[1] <= BOARD[3] + pad


# ---------------------------------------------------------------------------------------------------- pictures


def _served(pictures: Path, img: np.ndarray) -> str | None:
    """Writes an image where the page loads pictures from and returns its URL."""
    ok, buf = cv2.imencode(".png", img)
    if not ok:
        return None
    name = hashlib.sha256(buf.tobytes()).hexdigest()[:32] + ".png"
    try:
        pictures.mkdir(parents=True, exist_ok=True)
        path = pictures / name
        if not path.exists():
            path.write_bytes(buf.tobytes())
    except OSError:
        return None
    return f"/pictures/{name}"


def board_picture(light: int = 255) -> np.ndarray:
    """The board as light: lit squares, unlit squares (black is no light on the table), a line around it, the coordinates
    in the margin at its left and bottom."""
    px = 3
    w = int((BOARD[2] - BOARD[0] + MARGIN + 2) * px)
    h = int((BOARD[3] - BOARD[1] + MARGIN + 2) * px)
    img = np.zeros((h, w, 3), np.uint8)
    x0, y0 = int(MARGIN * px), int(2 * px)
    s = int(SQ * px)
    for rank in range(8):
        for file in range(8):
            if (rank + file) % 2:  # a1 is dark
                x, y = x0 + file * s, y0 + (7 - rank) * s
                img[y:y + s, x:x + s] = light
    cv2.rectangle(img, (x0 - 3, y0 - 3), (x0 + 8 * s + 2, y0 + 8 * s + 2), (255, 255, 255), 3, cv2.LINE_AA)
    font = cv2.FONT_HERSHEY_SIMPLEX
    for k in range(8):
        cv2.putText(img, "abcdefgh"[k], (x0 + k * s + s // 2 - 11, y0 + 8 * s + 36), font, 1.15, (255, 255, 255), 3, cv2.LINE_AA)
        cv2.putText(img, str(k + 1), (6, y0 + (7 - k) * s + s // 2 + 13), font, 1.15, (255, 255, 255), 3, cv2.LINE_AA)
    return img


# ---------------------------------------------------------------------------------------------------- reading the board

_mask_y, _mask_x = np.ogrid[:EYE, :EYE]
_MIDDLE = (_mask_x - (EYE - 1) / 2) ** 2 + (_mask_y - (EYE - 1) / 2) ** 2 <= (DISC * EYE) ** 2
_WIDE = (_mask_x - (EYE - 1) / 2) ** 2 + (_mask_y - (EYE - 1) / 2) ** 2 <= (0.40 * EYE) ** 2


def square_changes(ref: np.ndarray, cur: np.ndarray) -> dict[int, float]:
    """How much each square changed between two rectified views, in colour levels. A square's own light is matched first
    (a person's soft shadow darkens a whole square, a piece only part of it)."""
    a = cv2.GaussianBlur(ref, (0, 0), 1.2).astype(np.float32)
    b = cv2.GaussianBlur(cur, (0, 0), 1.2).astype(np.float32)
    out = {}
    for sq in SQUARES:
        y, x = (7 - (sq >> 4)) * EYE, (sq & 7) * EYE
        pa, pb = a[y:y + EYE, x:x + EYE], b[y:y + EYE, x:x + EYE]
        gain = np.clip(np.median(pb, axis=(0, 1)) / np.maximum(np.median(pa, axis=(0, 1)), 1.0), 0.6, 1.6)
        d = np.abs(pb - pa * gain).max(axis=2)[_MIDDLE]
        k = max(1, d.size * TOP_SHARE // 100)
        out[sq] = float(np.partition(d, -k)[-k:].mean())
    return out


def square_contents(view: np.ndarray) -> dict[int, float]:
    """How much detail each square of one view has: an empty one is flat, whichever way the light falls on it. The most
    detailed fifth of its middle, against its own surroundings; a square is looked at alone, its neighbours' light stays out."""
    v = cv2.GaussianBlur(view, (0, 0), 1.2).astype(np.float32)
    out = {}
    for sq in SQUARES:
        y, x = (7 - (sq >> 4)) * EYE, (sq & 7) * EYE
        p = v[y:y + EYE, x:x + EYE]
        d = np.abs(p - cv2.blur(p, (21, 21), borderType=cv2.BORDER_REPLICATE)).max(axis=2)[_WIDE]
        k = max(1, d.size // 5)
        out[sq] = float(np.partition(d, -k)[-k:].mean())
    return out


def hot_level(change: dict[int, float]) -> float:
    """What counts as changed, here: what the quietest squares read says how noisy this pair of pictures is."""
    return max(HOT, FLOOR * float(np.percentile(list(change.values()), 25)))


@dataclass(frozen=True)
class Reading:
    kind: str  # none | move | partial | wrong | busy
    move: int | None = None
    hot: tuple[int, ...] = ()


def leans_over(q: int, nadir: tuple[float, float] | None) -> tuple[int, ...]:
    """The squares next to ``q`` a tall piece on it can stick out over: seen from a camera above the board, a piece's head
    lies further from the point under the camera (``nadir``: file, rank) than its base. Rig, 4 Nov: a queen on f3 changed f2
    (45) more than f3 itself (36)."""
    if nadir is None:
        return ()
    dx, dy = (q & 7) - nadir[0], (q >> 4) - nadir[1]
    d = math.hypot(dx, dy)
    if d < 0.75:
        return ()
    sx = (1 if dx > 0 else -1) if abs(dx) > 0.38 * d else 0
    sy = (1 if dy > 0 else -1) if abs(dy) > 0.38 * d else 0
    return tuple(q + b * 16 + a for a, b in {(sx, sy), (sx, 0), (0, sy)} - {(0, 0)} if not (q + b * 16 + a) & 0x88)


def _toward_camera(q: int, nadir: tuple[float, float] | None) -> list[int]:
    """The neighbours of ``q`` whose pieces can stick out over it."""
    near = (q + a + 16 * b for a in (-1, 0, 1) for b in (-1, 0, 1) if a or b)
    return [i for i in near if not i & 0x88 and q in leans_over(i, nadir)]


def _left_and_landed(where: tuple[int, ...]) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """The squares of a move (:meth:`lightwork.chess.Board.changes`) that end up empty, and the ones that end up occupied."""
    if len(where) == 4:  # castling: king, then rook
        return (where[0], where[2]), (where[1], where[3])
    if len(where) == 3:  # in passing: the pawn, where it lands, the pawn taken
        return (where[0], where[2]), (where[1],)
    return (where[0],), (where[1],)


def _stands_right(where: tuple[int, ...], change: dict[int, float], contents: dict[int, float], level: float, unknown=frozenset()) -> bool:
    """What stands on the board now fits this move: where it leaves is empty and where it lands is occupied, or the square
    changed so much that the picture must be right about it and the detail reading wrong. Rig, 4 Nov: empty squares read
    3-7, pieces 13-77, whatever the light; the change between two pictures can be as faint as 10 on a white piece leaving a
    washed-out lit square, no more than the noise on a king that never moved (read as Kf3)."""
    left, landed = _left_and_landed(where)
    return (all(contents[q] < FILLED or change[q] >= 2 * level for q in left if q not in unknown)
            and all(contents[q] >= FILLED or change[q] >= level for q in landed))


def read_move(change: dict[int, float], board: Board, contents: dict[int, float] | None = None,
              nadir: tuple[float, float] | None = None, unknown=frozenset()) -> Reading:
    """The legal move that explains what changed and fits what now stands there (``contents``: :func:`square_contents` of
    the new view). Squares that changed and no move explains (a piece nudged) count against a move; a move that explains far
    less than such a square is not what happened (rig, 4 Nov: a queen held still over d1 and d2, read as Qf3 from 9 levels on
    f3). ``unknown``: squares that say nothing either way (just taken into the reference with the table's move): a move may
    use them, and needs another square that changed."""
    level = hot_level(change)
    hot = tuple(q for q in SQUARES if change[q] >= level and q not in unknown)
    if not hot:
        return Reading("none")
    if len(hot) > BUSY:
        return Reading("busy", None, hot)
    best, best_score = None, -math.inf
    seen = set()
    cap = 3 * level
    for m in board.legal_moves():  # a promotion's queen comes first and stands for the others
        where = board.changes(m)
        key = frozenset(where)
        if key in seen:
            continue
        seen.add(key)
        known = [q for q in where if q not in unknown]
        if not known or max(change[q] for q in known) < level:
            continue
        taken = move_to(m)
        if contents is None:
            if min(change[q] for q in known) < WEAK * level:
                continue
            evidence = min(change[q] for q in known)
        else:
            if not _stands_right(where, change, contents, level, unknown):
                continue
            if board.b[taken] and taken not in unknown and change[taken] < WEAK * level:
                continue  # occupied before and after: only the change can say the piece there is another one
            evidence = max(change[q] for q in known)
        heads = {u for q in where for u in leans_over(q, nadir)}
        other = [q for q in hot if q not in where and q not in heads]
        if other and evidence < 0.5 * max(change[q] for q in other):
            continue
        if sum(change[q] >= 2 * level for q in other) >= 2:  # a move does not do that: a hand lies over the board
            continue
        score = sum(min(change[q], cap) for q in known) - 0.7 * sum(min(change[q], cap) for q in other)
        for q in known:  # this square may only have changed by a head from the camera's side, where something changed too
            if any(i not in where and change[i] >= level for i in _toward_camera(q, nadir)):
                score -= 2 * level
        if score > best_score:
            best, best_score = m, score
    if best is not None:
        return Reading("move", best, hot)
    return Reading("partial" if len(hot) == 1 else "wrong", None, hot)


def read_expected(change: dict[int, float], where: tuple[int, ...], nadir: tuple[float, float] | None = None,
                  contents: dict[int, float] | None = None, capture: bool = False) -> Reading:
    """Whether the table's move, which changes the squares ``where``, has been made on the board."""
    level = hot_level(change)
    hot = tuple(q for q in SQUARES if change[q] >= level)
    if not hot:
        return Reading("none")
    if contents is None:
        done = min(change[q] for q in where) >= WEAK * level
    else:
        done = _stands_right(where, change, contents, level) and (not capture or change[where[1]] >= WEAK * level)
    if done and max(change[q] for q in where) >= level:
        return Reading("move", None, hot)
    if len(hot) > BUSY:
        return Reading("busy", None, hot)
    heads = {u for q in where for u in leans_over(q, nadir)}
    others = tuple(q for q in hot if q not in where and q not in heads)
    return Reading("wrong", None, others) if others else Reading("partial", None, hot)


class BoardEye:
    """The board through the camera: ``corners`` are the board's corners in camera px, from the far left clockwise."""

    def __init__(self, corners):
        src = np.float32(corners)
        side = 8 * EYE
        self.view_m = cv2.getPerspectiveTransform(src, np.float32([(0, 0), (side, 0), (side, side), (0, side)]))
        t = float(THUMB)
        self.thumb_m = cv2.getPerspectiveTransform(src / 2, np.float32([(0, 0), (t, 0), (t, t), (0, t)]))
        self.past: deque[np.ndarray] = deque(maxlen=3)
        self.seen_moving: deque[bool] = deque(maxlen=3)
        self.corners = src

    def view(self, image: np.ndarray) -> np.ndarray:
        return cv2.warpPerspective(image, self.view_m, (8 * EYE, 8 * EYE), flags=cv2.INTER_LINEAR)

    def moving(self, image: np.ndarray, mask: np.ndarray | None = None) -> bool:
        """Something moves on the board. Against the frame 3 back: the room's 100 Hz light repeats after 3 frames at 30 fps,
        at a short exposure it flickers from frame to frame."""
        small = cv2.warpPerspective(image[::2, ::2], self.thumb_m, (THUMB, THUMB), flags=cv2.INTER_LINEAR).astype(np.int16)
        before = self.past[0] if len(self.past) == self.past.maxlen else None
        self.past.append(small)
        if before is None:
            return False
        change = np.abs(small - before).max(axis=2) > MOTION_DIFF
        if mask is not None:
            change &= mask == 0
        # in 2 of the last 3 frames: a truncated camera frame differs from 3 back, and 3 later, but not in between
        self.seen_moving.append(float(np.count_nonzero(change)) / (THUMB * THUMB) > MOTION_FRAC)
        return sum(self.seen_moving) >= 2


# ---------------------------------------------------------------------------------------------------- the engine's process


class EngineJob:
    """One search, in its own process: a search in this one would take the interpreter lock from the camera thread."""

    def __init__(self, played: list[str], think_s: float, seed: int):
        self.p = subprocess.Popen([sys.executable, "-m", "lightwork.chess", "--think", str(think_s), "--seed", str(seed), *played],
                                  stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
        self.result: str | None = None

    def poll(self) -> str | None:
        """The move as text ("" if the process failed), or None while it thinks."""
        if self.result is None and self.p.poll() is not None:
            self.result = (self.p.stdout.read() or "").strip()
            self.p.stdout.close()
        return self.result

    def stop(self) -> None:
        if self.p.poll() is None:
            self.p.kill()


@dataclass
class _Scan:
    t0: float
    blank: bool  # what we draw on the board goes off first
    purpose: str  # setup | ref0 | read | ref
    target: int | None = None
    clean_after: float | None = None
    total: np.ndarray | None = None
    first: np.ndarray | None = None
    n: int = 0


# ---------------------------------------------------------------------------------------------------- the game


class ChessTable(Game):
    key = "chess"
    draws_hands = True
    # Written to the C920, which takes 78 as 15.6 ms: one turn of the projector's colour wheel. At the other games' 7.8 ms the
    # lit squares show moving bands of colour (rig, 4 Nov: 15 levels of blue and red between two pictures, the whole board
    # read as changed). Pieces stand still; a hand only has to be found, not followed fast.
    exposure = 78
    END_BUBBLES = ((LEFT_CX, 130.0, 34.0), (LEFT_CX, 280.0, 34.0))  # PLAY AGAIN and MENU, beside the board: pieces stay on it

    def __init__(self, now, rng, *, state_dir: Path | None = None, think=None):
        super().__init__(now, rng)
        self.think = think or (lambda played: EngineJob(played, THINK_S, rng.randrange(1 << 30)))
        self.board = Board()
        self.you, self.me = WHITE, BLACK
        self.state = "setup"  # setup | ready | you | think | show | over
        self.eye: BoardEye | None = None
        self.geo = None
        self.refs: list[np.ndarray] = []  # the view after each move: refs[-1] is what the next move is read against
        self.refresh_ref = False  # the mapping changed: refs[-1] must be taken again
        self.placed: set[int] = set()
        self.home = [q for q in SQUARES if self.board.b[q]]
        self.played: list[str] = []
        self.moves: list[str] = []  # standard notation, for the list
        self.left = {WHITE: CLOCK_S, BLACK: CLOCK_S}
        self.clock_t = now
        self.job = None
        self.reply: int | None = None
        self.reply_t = now
        self.reply_said = ""
        self.anim: tuple[int, int, float, str, bool] | None = None  # from, to, start, glyph, a capture
        self.result: tuple[str, int] | None = None
        self.trouble: tuple[int, ...] = ()
        self.trouble_lines: tuple[str, str] = ("", "")
        self.partial = False
        self.back_hint = ""
        self.visit = True  # something moved on the board since the last clean picture (at the start: never looked)
        self.unknown: frozenset[int] = frozenset()  # squares the reference just took as they stood, with the table's move
        self.candidate = None  # what the last clean picture seemed to say, waiting for a second one to agree
        self.confirm_at: float | None = None
        self.nadir: tuple[float, float] | None = None  # the point under the camera: file, rank (may lie off the board)
        self.cut_off = False  # part of the board lies outside the camera's picture
        self.empty_base: dict[int, float] = {}  # the lowest detail each square read while it was empty: its own light
        self._mask_key = None
        self._mask: np.ndarray | None = None
        self._noack_t = -1e9
        self.move_t = now
        self.hand_t = -1e9
        self.ignore_until = now + IGNORE_S
        self._drew: tuple = ()
        self._scan: _Scan | None = None
        self.scans = 0
        self.pens: list = []
        self.last = now
        self.log: list[str] = []
        self.start_bubble = Bubble("start", "START", (LEFT_CX, 200.0), 24.0)
        self.undo_bubble = Bubble("undo", "UNDO", (LEFT_CX, 200.0), 24.0)
        self.ok_bubble = Bubble("ok", "BOARD OK", (LEFT_CX, 200.0), 24.0)
        self.undo_bubble.dwell = self.start_bubble.dwell = self.ok_bubble.dwell = -0.5
        self.src, self.fills = None, {}
        self.debug_dir: Path | None = None
        if state_dir is not None:
            pictures = Path(state_dir) / "pictures"
            self.src = _served(pictures, board_picture())
            for role, bgr in ROLE_BGR.items():
                self.fills[role] = _served(pictures, np.full((4, 4, 3), bgr, np.uint8))
            self.debug_dir = Path(state_dir) / "play-debug" / f"chess-{time.strftime('%Y%m%dT%H%M%S')}"

    # ---- where the camera sees the board

    def place(self, geo, frame_size: tuple[int, int] = (1920, 1080)) -> None:
        """The session's mapping, on start and after every recalibration: what was learned through the old one is void."""
        if geo is self.geo:
            return
        old, self.geo = self.eye, geo
        self.eye = BoardEye(geo.d2c([(BOARD[0], BOARD[1]), (BOARD[2], BOARD[1]), (BOARD[2], BOARD[3]), (BOARD[0], BOARD[3])]))
        c = self.eye.corners
        self.cut_off = bool(c[:, 0].min() < 0 or c[:, 1].min() < 0 or c[:, 0].max() >= frame_size[0] or c[:, 1].max() >= frame_size[1])
        if self.cut_off:
            self.log.append(f"chess: the camera does not see the whole board: corners {np.round(c).astype(int).tolist()}")
        x, y = geo.c2d((frame_size[0] / 2, frame_size[1] / 2))  # the camera looks about straight down
        self.nadir = ((x - BOARD[0]) / SQ - 0.5, 7 - ((y - BOARD[1]) / SQ - 0.5))
        if old is None or np.abs(old.corners - self.eye.corners).max() < 1.0:
            return
        self._scan = None
        if self.refs:
            self.refresh_ref = True
        else:
            self.visit = True

    # ---- the frame loop

    def update(self, now, points, tips, segments=(), pens=()):
        dt = min(max(now - self.last, 0.0), 0.1)
        self.last = now
        self.pens = [p for p in pens if not _on_board(p.at)]
        if any(_on_board(p) for p in points):
            self.hand_t = now
            self.visit = True
            self.candidate, self.confirm_at = None, None
        if self.over:
            return
        if self.state == "setup" and self.start_bubble.step(points, dt):
            self.sfx.append("press")
            self._ready(now)
        elif self.trouble or self.back_hint:  # the board is right as it stands, whatever the camera thinks
            if self.ok_bubble.step(points, dt):
                self.sfx.append("press")
                self.trouble, self.back_hint, self.partial = (), "", False
                self.refresh_ref = True
                self.log.append("chess: BOARD OK: the board as it stands is the position")
        elif self.moves and self.undo_bubble.step(points, dt):
            self.sfx.append("press")
            self._undo(now)
        self._tick(now)
        if self.state == "think":
            self._think_step(now)

    def _tick(self, now: float) -> None:
        d, self.clock_t = now - self.clock_t, now
        side = {"you": self.you, "think": self.me}.get(self.state)  # the table's clock stops once it has chosen
        if side is None:
            return
        self.left[side] -= d
        if self.left[side] <= 0:
            self.left[side] = 0.0
            self._finish(("time", -side), now)

    def watch(self, frame, now: float, link=None) -> None:
        """One camera frame: is the board still, and if it is, take the clean picture this state needs."""
        if self.eye is None or self.over:
            return
        if self._scan is not None:
            self._scan_step(frame, now, link)
            return
        moving = self.eye.moving(frame.image, self._busy_mask())
        if now < self.ignore_until:
            moving = False
        if moving:
            self.move_t = now
            self.visit = True
            self.candidate, self.confirm_at = None, None
            self.unknown = frozenset()  # whatever comes next is seen arriving
        if now - max(self.move_t, self.hand_t) < STILL_S:
            return
        purpose = self._wanted(now)
        if purpose is not None:
            self.visit = False
            self._scan = _Scan(now, self._marks_on(), purpose)

    def _wanted(self, now: float) -> str | None:
        if self.refresh_ref:
            return "ref"
        if self.state == "ready":
            return "ref0"
        if self.confirm_at is not None:
            return "read" if now >= self.confirm_at else None
        if not self.visit:
            return None
        return {"setup": "setup", "you": "read", "show": "read"}.get(self.state)

    def _marks_on(self) -> bool:
        """What we draw on the board is on, and has to go off before a clean picture."""
        if self.candidate is not None:  # still off from the first of the two pictures
            return False
        return self.state == "setup" or self.state == "show" or bool(self.trouble) or (self.state == "you" and self.board.in_check())

    def _agreed(self, what, now: float) -> bool:
        """A clean picture said ``what``; it counts when the next one, a moment later, says the same. A hand held still over
        the board for half a second looks like the end of a move in one picture, rarely in two."""
        if self.candidate == what:
            self.candidate, self.confirm_at = None, None
            return True
        self.candidate, self.confirm_at = what, now + CONFIRM_GAP
        return False

    def _busy_mask(self) -> np.ndarray | None:
        """Where our own light changes while nothing else moves: the animated light, and the filled squares, whose single
        colour flickers even at this exposure. The camera would take it for a hand; an arm reaches over more than these."""
        squares: set[int] = set(self.trouble)
        path = None
        if self.state == "setup":
            squares |= {q for q in self.home if q not in self.placed}
        elif self.state == "think" and self.anim is not None:
            path = self.anim[:2]
        elif self.state == "show" and self.reply is not None:
            path = (move_from(self.reply), move_to(self.reply))
            squares |= set(self.board.changes(self.reply))
        elif self.state == "you" and self.board.in_check():
            squares.add(self.board.kings[self.you])
        if not squares and path is None:
            return None
        key = (tuple(sorted(squares)), path)
        if key != self._mask_key:
            self._mask_key, self._mask = key, _light_mask(squares, path)
        return self._mask

    def _scan_step(self, frame, now: float, link) -> None:
        s = self._scan
        if s.clean_after is None:
            if not s.blank:
                s.clean_after = s.t0
            elif link is None:
                s.clean_after = s.t0 + AFTER_ACK_S
            else:
                if s.target is None:  # the bare board went out after the previous frame
                    s.target = link.version()
                acked = link.acked_at(s.target)
                if acked is not None:
                    s.clean_after = acked + AFTER_ACK_S
                elif now - s.t0 > ACK_WAIT_S:  # a picture of who knows what must not become the position
                    self._scan = None
                    self.visit, self.move_t = True, now
                    if now - self._noack_t > 30.0:
                        self._noack_t = now
                        self.log.append("chess: the page does not answer; not reading the board until it does")
                    return
                else:
                    return
        if frame.t_mono < s.clean_after:
            return
        view = self.eye.view(frame.image).astype(np.float32)
        if s.total is None:
            s.total, s.first = view.copy(), view
        else:
            s.total += view
        s.n += 1
        if s.n < SCAN_FRAMES:
            return
        self._scan = None
        drift = float(np.mean(np.abs(s.first - view)))
        if drift > AGREE:  # something moved, or a broken camera frame: look again when it is still
            self.log.append(f"chess: frames disagree ({drift:.1f}), looking again")
            self.visit, self.move_t = True, now
            self.candidate, self.confirm_at = None, None
            return
        self.scans += 1
        self._scanned(s.purpose, (s.total / s.n + 0.5).astype(np.uint8), now)

    def _save(self, view: np.ndarray, name: str) -> None:
        """Each clean picture, for setting the thresholds from what this camera really sees."""
        if self.debug_dir is None:
            return
        try:
            self.debug_dir.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(self.debug_dir / f"{self.scans:03d}-{name}.jpg"), view, [cv2.IMWRITE_JPEG_QUALITY, 92])
        except OSError:
            pass

    def _scanned(self, purpose: str, view: np.ndarray, now: float) -> None:
        self._save(view, purpose)
        if purpose == "setup":
            self._read_setup(view, now)
        elif purpose == "ref0":
            self.refs = [view]
            self.state = "you"
            self.clock_t = now
            self.sfx.append("answer")
            self.say.append("All set. You are white. Your move.")
            self.log.append("chess: game on")
        elif purpose == "ref":
            self.refs[-1] = view
            self.refresh_ref = False
        elif self.state == "you":
            self._read_yours(view, now)
        elif self.state == "show":
            self._read_mine(view, now)

    def _contents(self, view: np.ndarray) -> dict[int, float]:
        """:func:`square_contents`, less what each square reads when nothing stands on it: the projector's bright reflection
        on the table has detail of its own. A square counts as empty for that while the position says it is (during set-up,
        every square: nothing reads lower than an empty square, so the lowest reading is its own light)."""
        raw = square_contents(view)
        base = self.empty_base
        for q in SQUARES:
            if self.state == "setup" or not self.board.b[q]:
                base[q] = min(base.get(q, math.inf), raw[q])
        return {q: raw[q] - max(0.0, min(base.get(q, 0.0), EMPTY_MAX) - EMPTY_TYPICAL) for q in SQUARES}

    def _nudged(self, squares, contents: dict[int, float]) -> bool:
        """Every one of these squares still holds a piece where the position has one and none where it has none: they
        changed because a piece was nudged on its square (rig, 4 Nov: knight and pawns beside a castling king), not
        because anything moved. A move always empties the square it starts from."""
        return bool(squares) and all((contents[q] >= FILLED) == bool(self.board.b[q]) for q in squares)

    def _take_as_is(self, view: np.ndarray, squares) -> None:
        """These squares of the reference become what they look like now."""
        ref = self.refs[-1].copy()
        for q in squares:
            y, x = (7 - (q >> 4)) * EYE, (q & 7) * EYE
            ref[y:y + EYE, x:x + EYE] = view[y:y + EYE, x:x + EYE]
        self.refs[-1] = ref
        self.log.append("chess: nudged, not moved: " + " ".join(square_name(q) for q in squares))

    def _log_change(self, change: dict[int, float], text: str) -> None:
        top = sorted(change, key=change.get, reverse=True)[:6]
        self.log.append(f"chess: {text}; level {hot_level(change):.0f}, " +
                        " ".join(f"{square_name(q)} {change[q]:.0f}" for q in top) +
                        f", median {float(np.median(list(change.values()))):.1f}")

    # ---- setting up

    def _read_setup(self, view: np.ndarray, now: float) -> None:
        """What stands where, from this one picture: pieces that were there already count, ones taken away again do not."""
        contents = self._contents(view)
        placed = {q for q in self.home if contents[q] >= FILLED}
        missing = sorted((q for q in self.home if q not in placed), key=lambda q: -contents[q])[:6]
        self.log.append(f"chess: setup: {len(placed)} placed; missing " + " ".join(f"{square_name(q)} {contents[q]:.0f}" for q in missing))
        if placed - self.placed:
            self.sfx.append("found")
        self.placed = placed
        if len(placed) == len(self.home):
            self._ready(now)

    def _ready(self, now: float) -> None:
        self.state = "ready"
        self.placed = set(self.home)

    # ---- the player's move

    def _read_yours(self, view: np.ndarray, now: float) -> None:
        change = square_changes(self.refs[-1], view)
        contents = self._contents(view)
        r = read_move(change, self.board, contents, self.nadir, unknown=self.unknown)
        self._log_change(change, f"read {r.kind} {uci(r.move) if r.move else ''}")
        if r.kind in ("wrong", "partial") and not self.back_hint and self._nudged(r.hot, contents):
            self._take_as_is(view, r.hot)
            self.candidate, self.confirm_at, self.partial, self.trouble = None, None, False, ()
            return
        if self.back_hint:  # taken back: nothing counts until the board is back as it was
            self.partial = False
            if r.kind == "none":
                self.back_hint, self.trouble = "", ()
                self.sfx.append("found")
            else:
                self.trouble = r.hot[:8]
            return
        if r.kind == "none" or r.kind == "partial":
            self.candidate, self.confirm_at = None, None
            self.partial = r.kind == "partial"
            self.trouble = ()
            return
        if not self._agreed(r.move if r.kind == "move" else "wrong", now):
            return
        was_troubled = bool(self.trouble)
        self.partial, self.trouble = False, ()
        if r.kind == "move":
            self._played(r.move, view, now)
            return
        self.trouble = r.hot[:8]
        if r.kind == "busy":
            self.trouble_lines = ("SOMETHING IS ON THE BOARD", "TAKE IT OFF, THEN I LOOK AGAIN")
        else:
            names = " ".join(square_name(q).upper() for q in r.hot[:4])
            self.trouble_lines = ("THAT IS NOT A LEGAL MOVE: PUT IT BACK", f"I SEE A CHANGE AT {names}")
            if not was_troubled:
                self.say.append("That's not a legal move. Please put it back.")
        self.sfx.append("error")

    def _played(self, m: int, view: np.ndarray, now: float) -> None:
        board = self.board
        self.unknown = frozenset()
        kind = board.b[move_from(m)]
        self.anim = (move_from(m), move_to(m), now, GLYPHS[abs(kind) - 1], board.is_capture(m))
        self.moves.append(board.san(m))
        self.played.append(uci(m))
        board.push(m)
        self.refs.append(view)
        self.left[self.you] = min(self.left[self.you] + min(now - max(self.move_t, self.hand_t), 3.0), CLOCK_S)
        self.sfx.append("found")
        self.log.append(f"chess: you played {self.moves[-1]}")
        out = board.outcome()
        if out is not None:
            self._finish(out, now)
            return
        self.state = "think"
        self.clock_t = now
        self.job = self.think(list(self.played))

    # ---- the table's move

    def _think_step(self, now: float) -> None:
        text = self.job.poll()
        if text is None or (self.anim is not None and now - self.anim[2] < ANIM_S):
            return
        m = None
        try:
            m = parse_uci(text)
        except (ValueError, IndexError):
            pass
        if m is None or m not in self.board.legal_moves():
            self.log.append(f"chess: the engine said {text!r}; thinking here instead")
            m = chess.best_move(self.board, 0.3, self.rng)
        self.reply = m
        self.reply_t = now
        self.anim = None
        self.job = None
        self.state = "show"
        self.visit = False
        self.reply_said = self.board.spoken(m)
        self.sfx.append("answer")
        self.say.append(self.reply_said)
        self.log.append(f"chess: table plays {self.board.san(m)}")

    def _read_mine(self, view: np.ndarray, now: float) -> None:
        change = square_changes(self.refs[-1], view)
        where = self.board.changes(self.reply)
        contents = self._contents(view)
        r = read_expected(change, where, self.nadir, contents, self.board.is_capture(self.reply))
        self._log_change(change, f"check {uci(self.reply)}: {r.kind}")
        if r.kind == "wrong" and self._nudged(r.hot, contents):  # its move not made yet, and something only nudged
            self._take_as_is(view, r.hot)
            self.candidate, self.confirm_at, self.trouble = None, None, ()
            return
        if r.kind == "none" or r.kind == "partial":
            self.candidate, self.confirm_at = None, None
            self.trouble = ()
            return
        if not self._agreed(r.kind == "move", now):
            return
        was_troubled = bool(self.trouble)
        self.trouble = ()
        if r.kind != "move":
            self.trouble = r.hot[:8]
            self.trouble_lines = ("NOT THAT ONE: " + self._reply_text()[:31], "PUT THAT BACK AND MAKE MY MOVE")
            self.sfx.append("error")
            if not was_troubled:
                self.say.append(f"Not that one. {self.reply_said}")
            return
        # The reference takes the table's move and nothing else: the player may have moved their own piece in the same go,
        # or be in the middle of it, and that must still show as changed (rig, 4 Nov: a white knight leaving a washed-out lit
        # square changes it by only 10 levels). What stayed quiet is refreshed, for the light.
        # (A square under the head of the table's piece is taken too, unless one of the player's pieces stood there.)
        taken = set(where) | {u for q in where for u in leans_over(q, self.nadir) if self.board.b[u] * self.you <= 0}
        level = hot_level(change)
        before, ref = self.refs[-1], view.copy()
        more = False
        for q in SQUARES:
            if q not in taken and change[q] >= WEAK * level:
                y, x = (7 - (q >> 4)) * EYE, (q & 7) * EYE
                ref[y:y + EYE, x:x + EYE] = before[y:y + EYE, x:x + EYE]
                more = more or change[q] >= level
        self.moves.append(self.board.san(self.reply))
        self.played.append(uci(self.reply))
        self.board.push(self.reply)
        self.refs.append(ref)
        self.reply = None
        self.sfx.append("found")
        self.state = "you"
        self.clock_t = now
        out = self.board.outcome()
        if out is not None:
            self._finish(out, now)
            return
        self.unknown = frozenset(taken)
        if more:  # read it now: no hand has to pass again for that
            self.visit = True

    def _reply_text(self) -> str:
        m = self.reply
        kind = abs(self.board.b[move_from(m)])
        return f"{PIECE_NAMES[kind].upper()} {square_name(move_from(m)).upper()} TO {square_name(move_to(m)).upper()}"

    # ---- taking back and ending

    def _undo(self, now: float) -> None:
        """The last move of the player's comes off the list: the board must go back to how it was, which the next clean
        picture compares with the one taken then."""
        if self.job is not None:
            self.job.stop()
            self.job = None
        take = 2 if self.state == "you" and len(self.moves) >= 2 else 1
        back = []
        for _ in range(take):
            m = self.board.hist[-1][0]
            back.append(f"{square_name(move_to(m)).upper()} TO {square_name(move_from(m)).upper()}")
            self.board.pop()
            self.moves.pop()
            self.played.pop()
            self.refs.pop()
        self.reply, self.anim, self.trouble = None, None, ()
        self.candidate, self.confirm_at = None, None
        self.unknown = frozenset()
        self.state = "you"
        self.clock_t = now
        self.back_hint = "PUT BACK: " + ", ".join(back)
        self.visit = True
        self.say.append("Taken back. Please put the pieces back.")
        self.log.append(f"chess: took back {take}")

    def _finish(self, out: tuple[str, int], now: float) -> None:
        if self.job is not None:
            self.job.stop()
            self.job = None
        self.result = out
        self.over = True
        self.state = "over"
        self.sfx.append("answer")
        self.say.append(self._ending_said())
        self.log.append(f"chess: over: {out} after {' '.join(self.moves)}")

    def _ending_said(self) -> str:
        how, winner = self.result
        if how == "checkmate":
            return "Checkmate. You win. Well played!" if winner == self.you else "Checkmate. I win. Good game!"
        if how == "time":
            return "Your time is up. I win." if winner == self.me else "My time is up. You win."
        return f"It's a draw, by {how}."

    # ---- drawing

    def draw(self, c, now):
        self._draw_board(c)
        self._draw_list(c)
        hide = (self._scan is not None and self._scan.blank) or self.candidate is not None
        if not self.over:
            self._draw_clocks(c)
            if self.state == "setup":
                self.start_bubble.draw(c, "target")
            elif self.trouble or self.back_hint:
                self.ok_bubble.draw(c, "warn")
            elif self.moves:
                self.undo_bubble.draw(c, "info")
        marks = self.state in ("setup", "show") or bool(self.trouble) or (self.state == "you" and self.board.in_check())
        sig = (hide and marks, self.state, self.reply, self.trouble, self.visit and self.state == "show", len(self.placed), marks)
        if sig != self._drew:
            self._drew = sig
            self.ignore_until = max(self.ignore_until, now + IGNORE_S)
        if not hide:
            self._draw_marks(c, now)
        for k, p in enumerate(self.pens):
            c.circle(f"chess-pen{k}", p.at, 9, "info", n=16)

    def _fill(self, c, id: str, sq: int, role: str) -> None:
        x0, y0, x1, y1 = sq_box(sq)
        src = self.fills.get(role)
        if src is not None:
            c.picture(id, src, x0, y0, x1, y1)
        else:
            c.rect(id, x0 + 2, y0 + 2, x1 - 2, y1 - 2, role)

    def _draw_board(self, c) -> None:
        x0, y0, x1, y1 = BOARD
        if self.src is not None:
            c.picture("chess-board", self.src, x0 - MARGIN, y0 - 2, x1 + 2, y1 + MARGIN)
            return
        for k in range(9):
            c.poly(f"chess-v{k}", [(x0 + k * SQ, y0), (x0 + k * SQ, y1)], "info")
            c.poly(f"chess-h{k}", [(x0, y0 + k * SQ), (x1, y0 + k * SQ)], "info")

    def _draw_clocks(self, c) -> None:
        running = {"you": self.you, "think": self.me}.get(self.state)
        for name, side, y in (("TABLE", self.me, 96.0), ("YOU", self.you, 290.0)):
            left = self.left[side]
            role = "error" if left < 10 else "warn" if left < 60 else "target" if side == running else "info"
            _label_c(c, f"chess-clock-{name}", LEFT_CX, y, name, role)
            whole = int(math.ceil(left))
            w, x0, top = 15.0, 80.0, y + 16
            cell = w * 1.45
            c.number(f"chess-mm-{name}", f"{whole // 60:02d}", x0, top, w, role)
            c.number(f"chess-ss-{name}", f"{whole % 60:02d}", x0 + 2 * cell + 12, top, w, role)
            cx = x0 + 2 * cell + 2
            c.poly(f"chess-colon-{name}-a", [(cx, top + 0.55 * w), (cx + 0.1, top + 0.55 * w)], role)
            c.poly(f"chess-colon-{name}-b", [(cx, top + 1.45 * w), (cx + 0.1, top + 1.45 * w)], role)
            if side == running:
                c.rect(f"chess-clock-{name}-on", 72.0, top - 7, 188.0, top + 2 * w + 7, role)

    def _draw_list(self, c) -> None:
        rows = [(k // 2 + 1, self.moves[k], self.moves[k + 1] if k + 1 < len(self.moves) else "")
                for k in range(0, len(self.moves), 2)][-LIST_ROWS:]
        for i, (n, white, black) in enumerate(rows):
            y = LIST_Y0 + i * LIST_STEP
            last = i == len(rows) - 1
            c.label(f"chess-n{i}", (LIST_X[0], y), str(n), "info")
            c.label(f"chess-w{i}", (LIST_X[1], y), white, "target" if last and not black else "info")
            if black:
                c.label(f"chess-b{i}", (LIST_X[2], y), black, "target" if last else "info")

    def _draw_marks(self, c, now: float) -> None:
        if self.state == "setup":
            self._draw_ghosts(c)
            return
        if self.state == "think" and self.anim is not None:
            self._draw_flight(c, now)
        if self.state == "you" and self.board.in_check() and not self.trouble:
            self._fill(c, "chess-check", self.board.kings[self.you], "error")
        if self.state == "show":
            self._draw_reply(c, now)
        if self.state == "over" and self.result[0] == "checkmate":
            self._fill(c, "chess-mated", self.board.kings[-self.result[1]], "error")
        for k, q in enumerate(self.trouble):
            self._fill(c, f"chess-bad-{k}", q, "error")

    def _draw_ghosts(self, c) -> None:
        """Where each piece goes: green for the player's side, red for the table's."""
        for q in self.home:
            if q in self.placed:
                continue
            p = self.board.b[q]
            side = 1 if p > 0 else -1
            x, y = sq_xy(q)
            c.label(f"chess-ghost-{q}", (x - 0.5 * FONT, y), GLYPHS[abs(p) - 1], "target" if side == self.you else "error")

    def _draw_flight(self, c, now: float) -> None:
        """The player's move, as light: the piece's mark flies from where it stood to where it stands now, trailing a line,
        and rings open around it on arrival."""
        f, t, t0, glyph, cap = self.anim
        a, b = sq_xy(f), sq_xy(t)
        k = (now - t0) / ANIM_S
        flight = min(k / 0.65, 1.0)
        e = _ease(flight)
        x, y = a[0] + (b[0] - a[0]) * e, a[1] + (b[1] - a[1]) * e
        tail = _ease(max(flight - 0.35, 0.0) / 0.65)
        tx, ty = a[0] + (b[0] - a[0]) * tail, a[1] + (b[1] - a[1]) * tail
        role = "error" if cap else "source"
        if k < 1.0:
            c.poly("chess-trail", [(tx, ty), (x, y)], role, glow=True)
            if flight < 1.0:
                c.label("chess-flying", (x - 0.5 * FONT, y), glyph, role)
            else:
                g = (k - 0.65) / 0.35
                c.circle("chess-land", b, 0.45 * SQ * (1 + 0.9 * g), role, n=28)
        c.rect("chess-to", *sq_box(t), "target")

    def _draw_reply(self, c, now: float) -> None:
        """The table's move: the piece's square in blue-green, where it goes in green, what it takes in red, an arrow between
        them; a light runs along the arrow until a hand comes."""
        m = self.reply
        f, t = move_from(m), move_to(m)
        where = self.board.changes(m)
        capture = self.board.is_capture(m)
        self._fill(c, "chess-from", f, "source")
        self._fill(c, "chess-to", t, "error" if capture else "target")
        path = _path(self.board, m)
        c.poly("chess-arrow", path, "warn", glow=True)
        ax, ay = path[-2]
        bx, by = path[-1]
        angle = math.atan2(by - ay, bx - ax)
        for k, side in enumerate((-1, 1)):
            c.poly(f"chess-head-{k}", [(bx, by), (bx - 13 * math.cos(angle + side * 0.5), by - 13 * math.sin(angle + side * 0.5))], "warn")
        if len(where) == 4:  # castling: the rook's move too
            self._fill(c, "chess-rook-from", where[2], "source")
            self._fill(c, "chess-rook-to", where[3], "target")
            c.poly("chess-rook-arrow", [sq_xy(where[2]), sq_xy(where[3])], "warn", glow=True)
        elif len(where) == 3:  # the pawn taken in passing
            self._fill(c, "chess-passing", where[2], "error")
        if not self.visit:
            k = ((now - self.reply_t) % 1.3) / 1.1
            if k < 1.0:
                c.dot("chess-runner", _along(path, _ease(k)), "info")

    def hud(self, now, best):
        if self.cut_off:
            return "THE CAMERA CANNOT SEE THE WHOLE BOARD", "MOVE IT, THEN HOLD THE CROSS-HAIRS", "error"
        if (self._scan is not None or self.candidate is not None) and self.state == "you" and not self.trouble:
            return "YOUR MOVE", "LOOKING...", "target"
        if self.state == "setup":
            return f"PUT THE PIECES ON THE BOARD: {len(self.placed)} OF {len(self.home)}", \
                "GREEN IS YOUR SIDE. HANDS AWAY, I LOOK", "info"
        if self.state == "ready":
            return "ALL SET", "TAKE YOUR HANDS OFF THE BOARD", "target"
        if self.state == "you" and self.back_hint:
            return "UNDONE: PUT THE BOARD BACK", self.back_hint, "warn"
        if self.trouble:
            return (*self.trouble_lines, "error")
        if self.state == "you":
            if self.partial:
                return "YOUR MOVE", "I SEE ONE SQUARE CHANGED. FINISH YOUR MOVE", "warn"
            return "YOUR MOVE" + (": CHECK!" if self.board.in_check() else ""), "MOVE A PIECE, THEN TAKE YOUR HAND AWAY", "target"
        if self.state == "think":
            return f"YOU PLAYED {self.moves[-1]}", "THINKING...", "info"
        if self.state == "show":
            return f"MY MOVE: {self._reply_text()}", "PLEASE MOVE IT FOR ME, THEN HAND AWAY", "warn"
        return self._ending_shown(), "HOLD PLAY AGAIN, OR MENU", "target" if self.result[1] == self.you else "info"

    def _ending_shown(self) -> str:
        how, winner = self.result
        if not winner:
            return f"DRAW BY {how.upper()}"
        who = "YOU WIN" if winner == self.you else "TABLE WINS"
        return f"{'CHECKMATE' if how == 'checkmate' else 'TIME'}: {who}"


def _path(board: Board, m: int) -> list[tuple[float, float]]:
    """The arrow's way: a knight's bends, everything else goes straight."""
    f, t = move_from(m), move_to(m)
    a, b = sq_xy(f), sq_xy(t)
    if abs(board.b[f]) != KNIGHT:
        return [a, b]
    if abs((t & 7) - (f & 7)) == 2:  # along the rank first, the long way
        return [a, (b[0], a[1]), b]
    return [a, (a[0], b[1]), b]


def _along(path: list, f: float) -> tuple[float, float]:
    """The point a share ``f`` of the way along a path."""
    lengths = [math.dist(path[i], path[i + 1]) for i in range(len(path) - 1)]
    left = f * sum(lengths)
    for i, n in enumerate(lengths):
        if left <= n or i == len(lengths) - 1:
            k = 0.0 if n == 0 else min(left / n, 1.0)
            return (path[i][0] + (path[i + 1][0] - path[i][0]) * k, path[i][1] + (path[i + 1][1] - path[i][1]) * k)
        left -= n
    return path[-1]


def _light_mask(squares, path) -> np.ndarray:
    """The motion view's pixels around these squares and the way between the two squares of ``path``."""
    k = THUMB / (BOARD[2] - BOARD[0])
    mask = np.zeros((THUMB, THUMB), np.uint8)

    def px(p):
        return int((p[0] - BOARD[0]) * k), int((p[1] - BOARD[1]) * k)

    width = max(1, int(SQ * k))
    if path is not None:
        f, t = path
        a, b = sq_xy(f), sq_xy(t)
        if abs((t & 7) - (f & 7)) * abs((t >> 4) - (f >> 4)) == 2:  # a knight's way goes round one of these corners
            corners = [px(a), px((b[0], a[1])), px(b), px((a[0], b[1]))]
            for p, q in zip(corners, corners[1:] + corners[:1], strict=True):
                cv2.line(mask, p, q, 255, width)
        else:
            cv2.line(mask, px(a), px(b), 255, width)
        squares = {*squares, f, t}
    r = int(0.65 * SQ * k)
    for q in squares:
        x, y = px(sq_xy(q))
        cv2.rectangle(mask, (x - r, y - r), (x + r, y + r), 255, -1)
    return mask
