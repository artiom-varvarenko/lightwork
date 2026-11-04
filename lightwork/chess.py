"""Chess: the full rules and a small engine that plays them, with no libraries and no model call.

Squares are 0x88 numbers (``rank * 16 + file``, a1 = 0, h8 = 0x77); pieces are signed ints (white positive). A move is an
int, ``from | to << 7 | promotion << 14``: the search makes tens of thousands of them a second in plain Python.

``python -m lightwork.chess --think 2 e2e4 e7e5`` prints the reply to the moves so far: the table runs its engine like this,
in its own process, so a search never takes the interpreter lock from the camera loop.
"""

from __future__ import annotations

import argparse
import random
import time

START = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
WHITE, BLACK = 1, -1
PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING = 1, 2, 3, 4, 5, 6
LETTERS = " PNBRQK"
PIECE_NAMES = {PAWN: "pawn", KNIGHT: "knight", BISHOP: "bishop", ROOK: "rook", QUEEN: "queen", KING: "king"}
SQUARES = tuple(r * 16 + f for r in range(8) for f in range(8))

KNIGHT_STEPS = (33, 31, 18, 14, -14, -18, -31, -33)
KING_STEPS = (1, -1, 16, -16, 17, 15, -15, -17)
BISHOP_STEPS = (17, 15, -15, -17)
ROOK_STEPS = (1, -1, 16, -16)
SLIDES = {BISHOP: BISHOP_STEPS, ROOK: ROOK_STEPS, QUEEN: KING_STEPS}
PROMOTIONS = (QUEEN, KNIGHT, ROOK, BISHOP)

CASTLE_K, CASTLE_Q, CASTLE_k, CASTLE_q = 1, 2, 4, 8
_KEEP_CASTLE = [15] * 128  # castling rights that survive a move from or to this square
_KEEP_CASTLE[0x00] = 15 - CASTLE_Q
_KEEP_CASTLE[0x07] = 15 - CASTLE_K
_KEEP_CASTLE[0x04] = 15 - CASTLE_K - CASTLE_Q
_KEEP_CASTLE[0x70] = 15 - CASTLE_q
_KEEP_CASTLE[0x77] = 15 - CASTLE_k
_KEEP_CASTLE[0x74] = 15 - CASTLE_k - CASTLE_q


def square(name: str) -> int:
    """"e4" -> its number."""
    return (ord(name[1]) - 49) * 16 + ord(name[0]) - 97


def square_name(sq: int) -> str:
    return "abcdefgh"[sq & 7] + str((sq >> 4) + 1)


def make_move(frm: int, to: int, promo: int = 0) -> int:
    return frm | to << 7 | promo << 14


def move_from(m: int) -> int:
    return m & 127


def move_to(m: int) -> int:
    return (m >> 7) & 127


def move_promo(m: int) -> int:
    return m >> 14


def uci(m: int) -> str:
    p = m >> 14
    return square_name(m & 127) + square_name((m >> 7) & 127) + (LETTERS[p].lower() if p else "")


def parse_uci(text: str) -> int:
    promo = LETTERS.index(text[4].upper()) if len(text) > 4 else 0
    return make_move(square(text[:2]), square(text[2:4]), promo)


# ---------------------------------------------------------------------------------------------------- evaluation tables

VALUE = (0, 100, 320, 330, 500, 900, 0)
_MID = {  # from white's side, rank 8 first (Tomasz Michniewski's "simplified evaluation function")
    PAWN: ((0, 0, 0, 0, 0, 0, 0, 0), (50, 50, 50, 50, 50, 50, 50, 50), (10, 10, 20, 30, 30, 20, 10, 10),
           (5, 5, 10, 25, 25, 10, 5, 5), (0, 0, 0, 20, 20, 0, 0, 0), (5, -5, -10, 0, 0, -10, -5, 5),
           (5, 10, 10, -20, -20, 10, 10, 5), (0, 0, 0, 0, 0, 0, 0, 0)),
    KNIGHT: ((-50, -40, -30, -30, -30, -30, -40, -50), (-40, -20, 0, 0, 0, 0, -20, -40), (-30, 0, 10, 15, 15, 10, 0, -30),
             (-30, 5, 15, 20, 20, 15, 5, -30), (-30, 0, 15, 20, 20, 15, 0, -30), (-30, 5, 10, 15, 15, 10, 5, -30),
             (-40, -20, 0, 5, 5, 0, -20, -40), (-50, -40, -30, -30, -30, -30, -40, -50)),
    BISHOP: ((-20, -10, -10, -10, -10, -10, -10, -20), (-10, 0, 0, 0, 0, 0, 0, -10), (-10, 0, 5, 10, 10, 5, 0, -10),
             (-10, 5, 5, 10, 10, 5, 5, -10), (-10, 0, 10, 10, 10, 10, 0, -10), (-10, 10, 10, 10, 10, 10, 10, -10),
             (-10, 5, 0, 0, 0, 0, 5, -10), (-20, -10, -10, -10, -10, -10, -10, -20)),
    ROOK: ((0, 0, 0, 0, 0, 0, 0, 0), (5, 10, 10, 10, 10, 10, 10, 5), (-5, 0, 0, 0, 0, 0, 0, -5), (-5, 0, 0, 0, 0, 0, 0, -5),
           (-5, 0, 0, 0, 0, 0, 0, -5), (-5, 0, 0, 0, 0, 0, 0, -5), (-5, 0, 0, 0, 0, 0, 0, -5), (0, 0, 0, 5, 5, 0, 0, 0)),
    QUEEN: ((-20, -10, -10, -5, -5, -10, -10, -20), (-10, 0, 0, 0, 0, 0, 0, -10), (-10, 0, 5, 5, 5, 5, 0, -10),
            (-5, 0, 5, 5, 5, 5, 0, -5), (0, 0, 5, 5, 5, 5, 0, -5), (-10, 5, 5, 5, 5, 5, 0, -10),
            (-10, 0, 5, 0, 0, 0, 0, -10), (-20, -10, -10, -5, -5, -10, -10, -20)),
    KING: ((-30, -40, -40, -50, -50, -40, -40, -30), (-30, -40, -40, -50, -50, -40, -40, -30),
           (-30, -40, -40, -50, -50, -40, -40, -30), (-30, -40, -40, -50, -50, -40, -40, -30),
           (-20, -30, -30, -40, -40, -30, -30, -20), (-10, -20, -20, -20, -20, -20, -20, -10),
           (20, 20, 0, 0, 0, 0, 20, 20), (20, 30, 10, 0, 0, 10, 30, 20)),
}
_KING_END = ((-50, -40, -30, -20, -20, -30, -40, -50), (-30, -20, -10, 0, 0, -10, -20, -30),
             (-30, -10, 20, 30, 30, 20, -10, -30), (-30, -10, 30, 40, 40, 30, -10, -30),
             (-30, -10, 30, 40, 40, 30, -10, -30), (-30, -10, 20, 30, 30, 20, -10, -30),
             (-30, -30, 0, 0, 0, 0, -30, -30), (-50, -30, -30, -30, -30, -30, -30, -50))


def _tables() -> tuple[list[list[int]], list[int]]:
    """Per signed piece + 6, per square: what the piece is worth there from white's side. And, per square for white's king,
    how much better it stands in an endgame than the middlegame table says."""
    pst = [[0] * 128 for _ in range(13)]
    for kind, rows in _MID.items():
        for sq in SQUARES:
            r, f = sq >> 4, sq & 7
            pst[kind + 6][sq] = VALUE[kind] + rows[7 - r][f]
            pst[-kind + 6][sq] = -(VALUE[kind] + rows[r][f])
    delta = [0] * 128
    for sq in SQUARES:
        r, f = sq >> 4, sq & 7
        delta[sq] = _KING_END[7 - r][f] - _MID[KING][7 - r][f]
    return pst, delta


PST, _KING_DELTA = _tables()
_rand = random.Random(20261104)  # one set of position keys for every run
_Z = [[_rand.getrandbits(62) for _ in range(128)] for _ in range(13)]
_Z_STM = _rand.getrandbits(62)
_Z_CASTLE = [_rand.getrandbits(62) for _ in range(16)]
_Z_EP = [_rand.getrandbits(62) for _ in range(128)]


# ---------------------------------------------------------------------------------------------------- the rules


class Board:
    """A position and how it came about: moves made with :meth:`push` come off again with :meth:`pop`."""

    __slots__ = ("b", "stm", "castle", "ep", "half", "full", "kings", "key", "score", "hist", "keys")

    def __init__(self, fen: str = START):
        self.b = [0] * 128
        rows, stm, castle, ep, *rest = fen.split()
        self.kings = [0, 0, 0]  # [unused, white's, black's]: indexed with +1 and -1
        for k, row in enumerate(rows.split("/")):
            f = 0
            for ch in row:
                if ch.isdigit():
                    f += int(ch)
                    continue
                kind = LETTERS.index(ch.upper())
                piece = kind if ch.isupper() else -kind
                self.b[(7 - k) * 16 + f] = piece
                if kind == KING:
                    self.kings[1 if ch.isupper() else -1] = (7 - k) * 16 + f
                f += 1
        self.stm = WHITE if stm == "w" else BLACK
        self.castle = sum(bit for ch, bit in zip("KQkq", (1, 2, 4, 8), strict=True) if ch in castle)
        self.ep = -1 if ep == "-" else square(ep)
        self.half = int(rest[0]) if rest else 0
        self.full = int(rest[1]) if len(rest) > 1 else 1
        self.hist: list[tuple] = []
        self.key, self.score = 0, 0
        for sq in SQUARES:
            p = self.b[sq]
            if p:
                self.key ^= _Z[p + 6][sq]
                self.score += PST[p + 6][sq]
        self.key ^= _Z_CASTLE[self.castle] ^ (_Z_EP[self.ep] if self.ep >= 0 else 0) ^ (_Z_STM if self.stm < 0 else 0)
        self.keys = [self.key]

    def fen(self) -> str:
        rows = []
        for r in range(7, -1, -1):
            row, gap = "", 0
            for f in range(8):
                p = self.b[r * 16 + f]
                if p == 0:
                    gap += 1
                    continue
                row += (str(gap) if gap else "") + (LETTERS[p] if p > 0 else LETTERS[-p].lower())
                gap = 0
            rows.append(row + (str(gap) if gap else ""))
        castle = "".join(ch for ch, bit in zip("KQkq", (1, 2, 4, 8), strict=True) if self.castle & bit) or "-"
        ep = square_name(self.ep) if self.ep >= 0 else "-"
        return f"{'/'.join(rows)} {'w' if self.stm > 0 else 'b'} {castle} {ep} {self.half} {self.full}"

    def attacked(self, sq: int, by: int) -> bool:
        """A piece of side ``by`` attacks ``sq``."""
        b = self.b
        pawn = by
        s = sq - 15 * by
        if not s & 0x88 and b[s] == pawn:
            return True
        s = sq - 17 * by
        if not s & 0x88 and b[s] == pawn:
            return True
        knight = KNIGHT * by
        for d in KNIGHT_STEPS:
            s = sq + d
            if not s & 0x88 and b[s] == knight:
                return True
        king = KING * by
        for d in KING_STEPS:
            s = sq + d
            if not s & 0x88 and b[s] == king:
                return True
        rook, bishop, queen = ROOK * by, BISHOP * by, QUEEN * by
        for d in ROOK_STEPS:
            s = sq + d
            while not s & 0x88:
                p = b[s]
                if p:
                    if p == rook or p == queen:
                        return True
                    break
                s += d
        for d in BISHOP_STEPS:
            s = sq + d
            while not s & 0x88:
                p = b[s]
                if p:
                    if p == bishop or p == queen:
                        return True
                    break
                s += d
        return False

    def in_check(self, side: int | None = None) -> bool:
        side = self.stm if side is None else side
        return self.attacked(self.kings[side], -side)

    def pseudo(self, captures_only: bool = False) -> list[int]:
        """Moves that follow how the pieces move; their own king may be left in check. ``captures_only``: captures and
        promotions, for the search's quiet ending."""
        b, us, out = self.b, self.stm, []
        add = out.append
        for s in SQUARES:
            p = b[s]
            if p * us <= 0:
                continue
            kind = p * us
            if kind == PAWN:
                up = 16 * us
                t = s + up
                last = t >> 4 == (7 if us > 0 else 0)
                if not b[t]:
                    if last:
                        for pr in PROMOTIONS:
                            add(s | t << 7 | pr << 14)
                    elif not captures_only:
                        add(s | t << 7)
                        if s >> 4 == (1 if us > 0 else 6) and not b[t + up]:
                            add(s | (t + up) << 7)
                for t in (s + up - 1, s + up + 1):
                    if t & 0x88:
                        continue
                    if b[t] * us < 0:
                        if last:
                            for pr in PROMOTIONS:
                                add(s | t << 7 | pr << 14)
                        else:
                            add(s | t << 7)
                    elif t == self.ep:
                        add(s | t << 7)
            elif kind == KNIGHT or kind == KING:
                for d in KNIGHT_STEPS if kind == KNIGHT else KING_STEPS:
                    t = s + d
                    if t & 0x88:
                        continue
                    q = b[t] * us
                    if q < 0 or (q == 0 and not captures_only):
                        add(s | t << 7)
            else:
                for d in SLIDES[kind]:
                    t = s + d
                    while not t & 0x88:
                        q = b[t] * us
                        if q > 0:
                            break
                        if q < 0:
                            add(s | t << 7)
                            break
                        if not captures_only:
                            add(s | t << 7)
                        t += d
        if not captures_only and self.castle:
            self._castling(add)
        return out

    def _castling(self, add) -> None:
        us, b = self.stm, self.b
        home = 0x04 if us > 0 else 0x74
        short, long = (CASTLE_K, CASTLE_Q) if us > 0 else (CASTLE_k, CASTLE_q)
        if self.castle & short and not b[home + 1] and not b[home + 2] and b[home + 3] == ROOK * us:
            if not (self.attacked(home, -us) or self.attacked(home + 1, -us) or self.attacked(home + 2, -us)):
                add(home | (home + 2) << 7)
        if self.castle & long and not b[home - 1] and not b[home - 2] and not b[home - 3] and b[home - 4] == ROOK * us:
            if not (self.attacked(home, -us) or self.attacked(home - 1, -us) or self.attacked(home - 2, -us)):
                add(home | (home - 2) << 7)

    def push(self, m: int) -> None:
        b, us = self.b, self.stm
        f, t, pr = m & 127, (m >> 7) & 127, m >> 14
        p, cap = b[f], b[t]
        self.hist.append((m, cap, self.castle, self.ep, self.half, self.key, self.score))
        key = self.key ^ _Z_STM ^ _Z_CASTLE[self.castle]
        if self.ep >= 0:
            key ^= _Z_EP[self.ep]
        score = self.score
        half = self.half + 1
        ep = -1
        kind = p * us
        b[f] = 0
        key ^= _Z[p + 6][f]
        score -= PST[p + 6][f]
        if cap:
            half = 0
            key ^= _Z[cap + 6][t]
            score -= PST[cap + 6][t]
        placed = p
        if kind == PAWN:
            half = 0
            if t == self.ep:  # en passant: the pawn taken stands beside where this one came from
                c = t - 16 * us
                taken = b[c]
                b[c] = 0
                key ^= _Z[taken + 6][c]
                score -= PST[taken + 6][c]
            elif t - f == 32 * us:  # the square passed over counts only if a pawn could really take there
                a, z = t - 1, t + 1
                if (not a & 0x88 and b[a] == -p) or (not z & 0x88 and b[z] == -p):
                    ep = (f + t) // 2
            if pr:
                placed = pr * us
        elif kind == KING:
            self.kings[us] = t
            if t - f == 2 or f - t == 2:  # castling: the rook jumps over
                ra, rb = (t + 1, t - 1) if t > f else (t - 2, t + 1)
                rook = b[ra]
                b[ra], b[rb] = 0, rook
                key ^= _Z[rook + 6][ra] ^ _Z[rook + 6][rb]
                score += PST[rook + 6][rb] - PST[rook + 6][ra]
        b[t] = placed
        key ^= _Z[placed + 6][t]
        score += PST[placed + 6][t]
        self.castle &= _KEEP_CASTLE[f] & _KEEP_CASTLE[t]
        key ^= _Z_CASTLE[self.castle]
        if ep >= 0:
            key ^= _Z_EP[ep]
        self.ep, self.half, self.key, self.score = ep, half, key, score
        if us < 0:
            self.full += 1
        self.stm = -us
        self.keys.append(key)

    def pop(self) -> None:
        m, cap, self.castle, self.ep, self.half, self.key, self.score = self.hist.pop()
        self.keys.pop()
        b = self.b
        us = self.stm = -self.stm
        f, t, pr = m & 127, (m >> 7) & 127, m >> 14
        p = PAWN * us if pr else b[t]
        b[f], b[t] = p, cap
        kind = p * us
        if kind == PAWN:
            if t == self.ep:
                b[t - 16 * us] = -p
        elif kind == KING:
            self.kings[us] = f
            if t - f == 2 or f - t == 2:
                ra, rb = (t + 1, t - 1) if t > f else (t - 2, t + 1)
                b[ra], b[rb] = b[rb], 0
        if us < 0:
            self.full -= 1

    def legal_moves(self) -> list[int]:
        out = []
        us = self.stm
        for m in self.pseudo():
            self.push(m)
            if not self.attacked(self.kings[us], -us):
                out.append(m)
            self.pop()
        return out

    def changes(self, m: int) -> tuple[int, ...]:
        """The squares a move changes on a real board: where a piece leaves or lands, the rook's when castling, the pawn's
        taken in passing."""
        f, t = m & 127, (m >> 7) & 127
        kind = self.b[f] * self.stm
        if kind == KING and abs(t - f) == 2:
            return (f, t, t + 1, t - 1) if t > f else (f, t, t - 2, t + 1)
        if kind == PAWN and t == self.ep:
            return (f, t, t - 16 * self.stm)
        return (f, t)

    def is_capture(self, m: int) -> bool:
        t = (m >> 7) & 127
        return bool(self.b[t]) or (abs(self.b[m & 127]) == PAWN and t == self.ep)

    def outcome(self) -> tuple[str, int] | None:
        """(how it ended, the winning side or 0) or None while it goes on."""
        if not self.legal_moves():
            return ("checkmate", -self.stm) if self.in_check() else ("stalemate", 0)
        if self.half >= 100:
            return ("the fifty-move rule", 0)
        if self.keys.count(self.key) >= 3:
            return ("repetition", 0)
        if self._too_little():
            return ("too few pieces", 0)
        return None

    def _too_little(self) -> bool:
        minors = 0
        for s in SQUARES:
            kind = abs(self.b[s])
            if kind in (PAWN, ROOK, QUEEN):
                return False
            if kind in (KNIGHT, BISHOP):
                minors += 1
        return minors <= 1

    def san(self, m: int) -> str:
        """The move in standard notation, from this position: "Nf3", "exd5", "O-O", "e8=Q+"."""
        f, t, pr = m & 127, (m >> 7) & 127, m >> 14
        kind = abs(self.b[f])
        legal = self.legal_moves()
        if kind == KING and abs(t - f) == 2:
            text = "O-O" if t > f else "O-O-O"
        elif kind == PAWN:
            text = (square_name(f)[0] + "x" if self.is_capture(m) else "") + square_name(t)
            if pr:
                text += "=" + LETTERS[pr]
        else:
            rivals = [move_from(o) for o in legal if move_to(o) == t and move_from(o) != f and abs(self.b[move_from(o)]) == kind]
            which = ""
            if rivals:
                if all(r & 7 != f & 7 for r in rivals):
                    which = square_name(f)[0]
                elif all(r >> 4 != f >> 4 for r in rivals):
                    which = square_name(f)[1]
                else:
                    which = square_name(f)
            text = LETTERS[kind] + which + ("x" if self.b[t] else "") + square_name(t)
        self.push(m)
        if self.in_check():
            text += "#" if not self.legal_moves() else "+"
        self.pop()
        return text

    def spoken(self, m: int) -> str:
        """The move as said aloud: "Knight to c6.", "Bishop takes e5, check."."""
        f, t, pr = m & 127, (m >> 7) & 127, m >> 14
        kind = abs(self.b[f])
        if kind == KING and abs(t - f) == 2:
            text = "Castle " + ("kingside" if t > f else "queenside")
        else:
            text = f"{PIECE_NAMES[kind].capitalize()} {'takes' if self.is_capture(m) else 'to'} {square_name(t)}"
            if pr:
                text += f", make it a {PIECE_NAMES[pr]}"
        self.push(m)
        if self.in_check():
            text += ", checkmate" if not self.legal_moves() else ", check"
        self.pop()
        return text + "."


def perft(board: Board, depth: int) -> int:
    """How many move sequences of this length exist: the standard check of a move generator against published counts."""
    if depth == 0:
        return 1
    moves = board.legal_moves()
    if depth == 1:
        return len(moves)
    total = 0
    for m in moves:
        board.push(m)
        total += perft(board, depth - 1)
        board.pop()
    return total


# ---------------------------------------------------------------------------------------------------- the engine

MATE = 100_000
_INF = 1_000_000
_EXACT, _LOWER, _UPPER = 0, 1, 2
OPENINGS = (  # main lines, for variety and a sensible start; the search takes over where they end
    "e2e4 e7e5 g1f3 b8c6 f1b5 a7a6 b5a4 g8f6 e1g1 f8e7", "e2e4 e7e5 g1f3 b8c6 f1c4 f8c5 c2c3 g8f6 d2d4 e5d4",
    "e2e4 e7e5 g1f3 b8c6 d2d4 e5d4 f3d4 f8c5", "e2e4 e7e5 g1f3 g8f6 f3e5 d7d6 e5f3 f6e4", "e2e4 e7e5 b1c3 g8f6 g1f3 b8c6",
    "e2e4 e7e5 f1c4 g8f6 d2d3 c7c6", "e2e4 c7c5 g1f3 d7d6 d2d4 c5d4 f3d4 g8f6 b1c3 a7a6",
    "e2e4 c7c5 g1f3 b8c6 d2d4 c5d4 f3d4 g8f6 b1c3 e7e5", "e2e4 c7c5 b1c3 b8c6 g1f3 g7g6", "e2e4 e7e6 d2d4 d7d5 b1c3 g8f6",
    "e2e4 e7e6 d2d4 d7d5 e4e5 c7c5 c2c3 b8c6", "e2e4 c7c6 d2d4 d7d5 b1c3 d5e4 c3e4 c8f5", "e2e4 c7c6 d2d4 d7d5 e4e5 c8f5",
    "d2d4 d7d5 c2c4 e7e6 b1c3 g8f6 c1g5 f8e7", "d2d4 d7d5 c2c4 c7c6 g1f3 g8f6 b1c3 d5c4", "d2d4 d7d5 g1f3 g8f6 c2c4 e7e6",
    "d2d4 g8f6 c2c4 e7e6 g1f3 d7d5 b1c3 f8e7", "d2d4 g8f6 c2c4 g7g6 b1c3 f8g7 e2e4 d7d6", "d2d4 g8f6 c2c4 e7e6 b1c3 f8b4",
    "d2d4 g8f6 g1f3 g7g6 c2c4 f8g7", "c2c4 e7e5 b1c3 g8f6 g1f3 b8c6", "c2c4 g8f6 b1c3 e7e6 g1f3 d7d5", "c2c4 c7c5 g1f3 g8f6",
    "g1f3 d7d5 d2d4 g8f6 c2c4 e7e6", "g1f3 g8f6 c2c4 g7g6 b1c3 f8g7",
)


class _Stop(Exception):
    pass


class _Search:
    def __init__(self, board: Board, until: float, rng: random.Random, noise: int = 6):
        self.board, self.until, self.rng, self.noise = board, until, rng, noise
        self.nodes = 0
        self.table: dict[int, tuple] = {}
        self.killers = [[0, 0] for _ in range(128)]
        self.history: dict[int, int] = {}
        material = sum(VALUE[abs(board.b[s])] for s in SQUARES)
        self.endgame = material <= 2 * (VALUE[ROOK] + VALUE[KNIGHT] + 4 * VALUE[PAWN])

    def _eval(self) -> int:
        b = self.board
        score = b.score
        if self.endgame:  # kings walk to the middle
            k, q = b.kings[1], b.kings[-1]
            score += _KING_DELTA[k] - _KING_DELTA[(7 - (q >> 4)) * 16 + (q & 7)]
        return score * b.stm

    def _tick(self) -> None:
        self.nodes += 1
        if not self.nodes & 511 and time.monotonic() >= self.until:
            raise _Stop

    def _order(self, moves: list[int], best: int, ply: int) -> list[int]:
        b = self.board.b
        k0, k1 = self.killers[ply]
        hist = self.history
        scored = []
        for m in moves:
            cap = b[(m >> 7) & 127]
            if m == best:
                s = 1 << 24
            elif cap:
                s = (1 << 20) + VALUE[abs(cap)] * 16 - VALUE[abs(b[m & 127])] // 16
            elif m >> 14:
                s = 1 << 19
            elif m == k0 or m == k1:
                s = 1 << 18
            else:
                s = hist.get(m & 16383, 0)
            scored.append((s, m))
        scored.sort(reverse=True)
        return [m for _, m in scored]

    def _quiet(self, alpha: int, beta: int, ply: int) -> int:
        self._tick()
        stand = self._eval()
        if stand >= beta or ply >= 48:
            return stand
        alpha = max(alpha, stand)
        board = self.board
        us = board.stm
        for m in self._order(board.pseudo(captures_only=True), 0, ply):
            cap = board.b[(m >> 7) & 127]
            if stand + VALUE[abs(cap)] + 200 < alpha and not m >> 14:  # even winning this cannot help
                continue
            board.push(m)
            if board.attacked(board.kings[us], -us):
                board.pop()
                continue
            try:
                score = -self._quiet(-beta, -alpha, ply + 1)
            finally:  # out of time: the board still comes back as it was
                board.pop()
            if score >= beta:
                return score
            alpha = max(alpha, score)
        return alpha

    def _repeated(self) -> bool:
        b = self.board
        n = len(b.keys) - 1
        return any(b.keys[i] == b.key for i in range(n - 4, max(n - b.half - 1, -1), -2))

    def _node(self, depth: int, alpha: int, beta: int, ply: int, null_ok: bool = True) -> int:
        board = self.board
        if ply and (board.half >= 100 or (board.half >= 4 and self._repeated())):
            return 0
        us = board.stm
        checked = board.attacked(board.kings[us], -us)
        if checked:
            depth += 1
        if depth <= 0:
            return self._quiet(alpha, beta, ply)
        self._tick()
        start_alpha = alpha
        entry = self.table.get(board.key)
        best_move = 0
        if entry is not None:
            d, flag, value, best_move = entry
            if ply and d >= depth and abs(value) < MATE - 1000:
                if flag == _EXACT or (flag == _LOWER and value >= beta) or (flag == _UPPER and value <= alpha):
                    return value
        if null_ok and not checked and depth >= 3 and self._eval() >= beta and self._has_pieces(us):
            board.stm = -us  # pass: if that is still good enough, the real moves will be too
            saved_ep, board.ep = board.ep, -1
            key = board.key
            board.key ^= _Z_STM ^ (_Z_EP[saved_ep] if saved_ep >= 0 else 0)
            board.keys.append(board.key)
            try:
                score = -self._node(depth - 3, -beta, -beta + 1, ply + 1, False)
            finally:
                board.keys.pop()
                board.stm, board.ep, board.key = us, saved_ep, key
            if score >= beta:
                return score
        legal = 0
        best = -_INF
        for m in self._order(board.pseudo(), best_move, ply):
            quiet = not board.b[(m >> 7) & 127] and not m >> 14
            board.push(m)
            if board.attacked(board.kings[us], -us):
                board.pop()
                continue
            legal += 1
            try:
                if legal == 1:
                    score = -self._node(depth - 1, -beta, -alpha, ply + 1)
                else:
                    cut = 1 if legal > 4 and depth >= 3 and quiet and not checked else 0  # late, quiet: look less deep ...
                    score = -self._node(depth - 1 - cut, -alpha - 1, -alpha, ply + 1)
                    if alpha < score < beta or (cut and score > alpha):  # ... unless it turns out interesting
                        score = -self._node(depth - 1, -beta, -alpha, ply + 1)
            finally:
                board.pop()
            if score > best:
                best, best_move = score, m
            if score > alpha:
                alpha = score
            if alpha >= beta:
                if quiet:
                    killers = self.killers[ply]
                    if killers[0] != m:
                        killers[0], killers[1] = m, killers[0]
                    self.history[m & 16383] = self.history.get(m & 16383, 0) + depth * depth
                break
        if not legal:
            return -MATE + ply if checked else 0
        flag = _UPPER if best <= start_alpha else _LOWER if best >= beta else _EXACT
        if len(self.table) > 400_000:
            self.table.clear()
        self.table[board.key] = (depth, flag, best, best_move)
        return best

    def _has_pieces(self, side: int) -> bool:
        b = self.board.b
        return any(b[s] * side > PAWN and b[s] * side != KING for s in SQUARES)

    def run(self, moves: list[int], max_depth: int) -> tuple[int, int]:
        """(best move, depth finished)."""
        board = self.board
        best = moves[0]
        done = 0
        order = self.rng.sample(moves, len(moves))
        try:
            for depth in range(1, max_depth + 1):
                alpha, top = -_INF, order[0]
                top_score = -_INF
                for k, m in enumerate(order):
                    board.push(m)
                    try:
                        if k == 0:
                            score = -self._node(depth - 1, -_INF, -alpha, 1)
                        else:
                            score = -self._node(depth - 1, -alpha - 1, -alpha, 1)
                            if score > alpha:
                                score = -self._node(depth - 1, -_INF, -alpha, 1)
                    finally:
                        board.pop()
                    if abs(score) < MATE - 1000:
                        score += self.rng.randint(0, self.noise)  # games differ
                    if score > top_score:
                        top_score, top = score, m
                    alpha = max(alpha, score)
                    if depth > 1:  # the last depth's best was searched first: at worst this is still that one
                        best = top
                best, done = top, depth
                order = [top] + [m for m in order if m != top]
                if abs(top_score) >= MATE - 1000 or len(moves) == 1:
                    break
                if time.monotonic() > self.until - 0.5 * (self.until - self.started):
                    break  # the next depth would not finish
        except _Stop:
            pass
        return best, done

    started = 0.0


def book_reply(played: list[str], rng: random.Random) -> str | None:
    """A move from the opening lines that follow from the moves played so far, or None."""
    n = len(played)
    follow = {line.split()[n] for line in OPENINGS if len(line.split()) > n and line.split()[:n] == played}
    return rng.choice(sorted(follow)) if follow else None


def best_move(board: Board, think_s: float = 2.0, rng: random.Random | None = None, max_depth: int = 40) -> int:
    """The move to play from ``board``, which comes back as it was. It must have a legal move."""
    rng = rng or random.Random()
    moves = board.legal_moves()
    if len(moves) == 1:
        return moves[0]
    search = _Search(board, time.monotonic() + think_s, rng)
    search.started = time.monotonic()
    return search.run(moves, max_depth)[0]


def reply_to(played: list[str], think_s: float = 2.0, seed: int | None = None) -> str:
    """The move after these moves (UCI, from the start position)."""
    rng = random.Random(seed)
    board = Board()
    for text in played:
        board.push(parse_uci(text))
    known = book_reply(played, rng)
    if known is not None and parse_uci(known) in board.legal_moves():
        return known
    return uci(best_move(board, think_s, rng))


def main() -> None:
    p = argparse.ArgumentParser(description="The reply to a game so far: prints one move.")
    p.add_argument("--think", type=float, default=2.0)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("moves", nargs="*")
    args = p.parse_args()
    print(reply_to(args.moves, args.think, args.seed), flush=True)


if __name__ == "__main__":
    main()
