"""Hand games on the projected table (``lightwork play --table --game ...``): nothing to set up, two bare hands.

``catch``   Catch the light: glowing targets appear anywhere on the table; touch them before they fade. 45 s.
``follow``  Follow the lights: four pads light up in a sequence; touch them back in the same order. It grows.
``draw``    Draw and guess: pinch thumb and index finger together and draw in light (open them to lift the pen);
            take your hands away and Muse Spark guesses what it is.
``rps``     Rock paper scissors against the table: 3, 2, 1, SHOOT, then show rock, paper or scissors over the table;
            the camera reads the hand (3-D landmarks), the table shows its own move and keeps score. First to 5.
``slice``   Slice the fruit (:mod:`lightwork.fruit`): fruit is tossed up from the player's edge and falls back; a quick
            swipe of a fingertip through it cuts it in two. 60 s.
``chess``   Chess against the table with real pieces on a board of light (:mod:`lightwork.chesstable`): it reads your moves
            with the camera, animates them and replies; you make its moves for it.
``helper``  General helper (:mod:`lightwork.helper`): hold the mic bubble and say what you want: any job, or a game to start.
``menu``    Bubbles to pick one by holding a fingertip on it, and OBJECTS: back to the object board, where things put on
            the table are found, named (Muse Spark), outlined (SAM 3.1) and measured.

Hands come from the table's tracker (MediaPipe landmarks, local); everything is drawn in the playground's design space
(750 x 420, the page) and reads upright from the player's side with ``--rotate 180``. The games run inside
:class:`lightwork.tabletop.TableSession`, which pauses its object scans while one is on. Best scores are kept in
``.lightwork/play-games.json``.
"""

from __future__ import annotations

import json
import math
import random
import re
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from .timers import Timers

AREA = (82.0, 74.0, 738.0, 410.0)  # design px: where game things go (right of the button column, under the status bar)
GAMES = ("catch", "follow", "slice", "draw", "rps", "chess")
TITLES = {"catch": "CATCH THE LIGHT", "follow": "FOLLOW THE LIGHTS", "slice": "SLICE THE FRUIT",
          "draw": "DRAW AND GUESS", "rps": "ROCK PAPER SCISSORS", "chess": "CHESS"}
DWELL_S = 0.9  # a fingertip held this long on a bubble picks it
TOUCH_PAD = 20.0  # design px of slack around a target: hands are tracked, not exact
EM = 14.8  # design px per character of a label (bold capitals, about 24.6 design px high), for centring
WORDS = {  # spoken: game key
    "catch": ("catch", "catch the light", "light game"),
    "follow": ("follow", "follow the lights", "simon", "memory game", "memory"),
    "draw": ("draw", "drawing", "guess what i draw", "pictionary", "let's draw"),
    # never "rock" or "scissors" alone: "it's a rock" answers draw and guess
    "rps": ("rock paper scissors", "rock, paper, scissors", "rock-paper-scissors", "rock, paper and scissors",
            "rock paper and scissors"),
    "chess": ("chess",),
    # never "fruit" or "slice" alone: "it's a fruit" answers draw and guess
    "slice": ("slice the fruit", "fruit slice", "slice fruit", "slicing fruit", "fruit slicing", "cut the fruit",
              "fruit ninja", "fruit game", "slicing game"),
}
GUESS_PROMPT = ("Someone drew this on a table with a fingertip; the lines are the drawing (the light trail of the "
                "finger, a little shaky). What did they draw? Make your best guess even if it is rough. Reply with "
                "JSON: {\"guess\": \"1 to 3 words, a thing anyone would name\", \"say\": \"one short playful "
                "sentence to say out loud, at most 12 words, that names it\"}.")
GUESS_SCHEMA = {
    "type": "object",
    "properties": {"guess": {"type": "string"}, "say": {"type": "string"}},
    "required": ["guess", "say"],
    "additionalProperties": False,
}


def _near(points, c, r: float) -> bool:
    return any(math.dist(p, c) <= r for p in points)


def _seg_dist(c, a, b) -> float:
    """Distance from ``c`` to the segment a-b."""
    ax, ay = a
    bx, by = b
    dx, dy = bx - ax, by - ay
    n = dx * dx + dy * dy
    t = 0.0 if n == 0 else max(0.0, min(1.0, ((c[0] - ax) * dx + (c[1] - ay) * dy) / n))
    return math.dist(c, (ax + t * dx, ay + t * dy))


def _swept(segments, c, r: float) -> bool:
    """A hand passed over ``c`` between two frames (it moved too fast to be seen on it)."""
    return any(_seg_dist(c, a, b) <= r for a, b in segments)


@dataclass(frozen=True)
class Pen:
    """A hand as a pen: its index fingertip (design px, the cursor), whether thumb and index finger touch, and the
    shape it makes."""

    at: tuple[float, float]
    down: bool
    hand: int = 0  # the table's hand slot: a line stays with the hand that started it
    shape: str = ""  # "rock", "paper", "scissors" or "" (lightwork.vision.hands.ShapeVote, over the last frames)


def _unsaid(line: str) -> str:
    """``line`` without its spoken alternative, for when nothing listens: "HOLD PLAY AGAIN, OR SAY MENU" -> "HOLD PLAY
    AGAIN"; a line that is only one ("OR SAY ITS NAME, OR OBJECTS") -> ""."""
    if line.startswith("OR SAY"):
        return ""
    return re.sub(r",? OR SAY\b.*$", "", line)


def _fit(text: str, n: int = 44) -> str:
    return text if len(text) <= n else text[:n - 3] + "..."


class Game:
    key = ""

    def __init__(self, now: float, rng: random.Random):
        self.t0 = now
        self.rng = rng
        self.sfx: list[str] = []
        self.say: list[str] = []
        self.score = 0
        self.over = False

    draws_hands = False  # the game draws its own cursor for each hand (the table then draws none)
    END_BUBBLES = ((300.0, 300.0, 40.0), (520.0, 300.0, 40.0))  # PLAY AGAIN and MENU when it is over: (x, y, radius)
    exposure: int | None = None  # the camera exposure this tile needs (written to the camera), None: the games' own

    def update(self, now: float, points: list, tips: list, segments=(), pens=()) -> None:
        """``points``: hand landmarks, ``tips``: index fingertips, ``segments``: where points moved since the frame
        before, ``pens``: one :class:`Pen` per hand (design px)."""
        raise NotImplementedError

    def draw(self, c, now: float) -> None:
        raise NotImplementedError

    def hud(self, now: float, best: int) -> tuple[str, str, str]:
        raise NotImplementedError

    def hear(self, low: str, now: float) -> bool:
        return False


# ---------------------------------------------------------------------------------------------------- menu


@dataclass
class Bubble:
    key: str
    text: str
    c: tuple[float, float]
    r: float
    dwell: float = 0.0

    def step(self, points, dt: float) -> bool:
        if _near(points, self.c, self.r + 6):
            self.dwell += dt
            if self.dwell >= DWELL_S:
                self.dwell = 0.0
                return True
        else:
            self.dwell = max(0.0, self.dwell - 2 * dt)
        return False

    def draw(self, c, role: str = "info", sub: str = "") -> None:
        c.circle(f"bub-{self.key}", self.c, self.r, role, n=40)
        if self.dwell > 0:
            c.arc(f"bub-{self.key}-arc", self.c, self.r + 8, self.dwell / DWELL_S, "target")
        lines = self.text.split("\n")
        em = 14.8  # design px per character, bold capitals (projector_style label at about 24.6 design px)
        for k, line in enumerate(lines):
            y = self.c[1] + self.r + 22 + 26 * k
            c.label(f"bub-{self.key}-t{k}", (self.c[0] - em * len(line) / 2, y), line, role)
        if sub:
            c.label(f"bub-{self.key}-sub", (self.c[0] - em * len(sub) / 2, self.c[1] + self.r + 22 + 26 * len(lines)),
                    sub, "info")


class Menu(Game):
    key = "menu"

    def __init__(self, now, rng, best: dict):
        super().__init__(now, rng)
        self.best = best
        far, near = 110.0, 280.0  # two rows: eight two-line labels side by side would run into each other
        self.bubbles = [Bubble("helper", "GENERAL\nHELPER", (150.0, far), 36.0),
                        Bubble("catch", "CATCH\nTHE LIGHT", (310.0, far), 36.0),
                        Bubble("follow", "FOLLOW\nTHE LIGHTS", (470.0, far), 36.0),
                        Bubble("slice", "SLICE\nTHE FRUIT", (630.0, far), 36.0),
                        Bubble("draw", "DRAW\nAND GUESS", (150.0, near), 36.0),
                        Bubble("rps", "ROCK PAPER\nSCISSORS", (310.0, near), 36.0),
                        Bubble("chess", "CHESS", (470.0, near), 36.0),
                        Bubble("objects", "OBJECT\nBOARD", (630.0, near), 36.0)]
        self.choice: str | None = None
        self.last = now
        self.clocks_below = False  # timers are running along the near edge

    def update(self, now, points, tips, segments=(), pens=()):
        dt = min(max(now - self.last, 0.0), 0.1)
        self.last = now
        for b in self.bubbles:
            if b.step(points, dt):
                self.choice = b.key
                self.sfx.append("press")

    def draw(self, c, now):
        roles = {"helper": "target", "catch": "target", "follow": "source", "slice": "error", "draw": "warn",
                 "rps": "error", "chess": "source", "objects": "info"}
        for b in self.bubbles:
            best = None if self.clocks_below and b.c[1] > 200 else self.best.get(b.key)  # the near row: clocks go there
            b.draw(c, roles[b.key], f"BEST {best}" if best else "")
            self._icon(c, b, roles[b.key], now)

    @staticmethod
    def _icon(c, b: Bubble, role: str, now: float) -> None:
        x, y = b.c
        if b.key == "helper":
            mic_icon(c, "ico-helper", x, y, 22.0, role)
        elif b.key == "catch":  # a pulsing light
            r = 14 + 4 * math.sin(now * 5)
            c.circle("ico-catch", (x, y), r, role, n=20)
            c.dot("ico-catch-d", (x, y), role)
        elif b.key == "follow":  # four pads
            for k, (dx, dy) in enumerate(((-13, -13), (13, -13), (-13, 13), (13, 13))):
                lit = int(now * 2.5) % 4 == k
                c.circle(f"ico-follow-{k}", (x + dx, y + dy), 9 if lit else 6, role, n=14)
        elif b.key == "slice":  # an apple cut in two
            from .fruit import fruit_icon  # fruit builds on this module

            fruit_icon(c, "ico-slice", x, y, 20.0, role, now)
        elif b.key == "draw":  # a squiggle
            pts = [(x - 24 + 4 * i, y + 10 * math.sin(i * 0.9 + now * 3)) for i in range(13)]
            c.poly("ico-draw", pts, role)
        elif b.key == "rps":  # rock, then paper, then scissors
            move = MOVES[int(now / 0.9) % 3]
            move_icon(c, f"ico-rps-{move}", move, x, y, 22.0, role)
        elif b.key == "chess":  # a pawn
            c.circle("ico-chess-head", (x, y - 11), 7, role, n=14)
            c.poly("ico-chess-body", [(x - 5, y - 4), (x - 11, y + 13), (x + 11, y + 13), (x + 5, y - 4)], role)
            c.poly("ico-chess-foot", [(x - 15, y + 19), (x + 15, y + 19)], role)
        else:  # two things on a table, each outlined
            c.rect("ico-obj-a", x - 22, y - 14, x - 2, y + 6, role)
            c.circle("ico-obj-b", (x + 13, y + 6), 10, role, n=16)

    def hud(self, now, best):
        return "PICK ONE: HOLD A FINGER ON IT", "OR SAY ITS NAME, OR OBJECTS", "info"


# ---------------------------------------------------------------------------------------------------- catch


@dataclass
class Target:
    n: int
    c: tuple[float, float]
    r: float
    born: float
    life: float


class CatchLight(Game):
    key = "catch"
    ROUND_S = 45.0
    READY_S = 3.0

    def __init__(self, now, rng):
        super().__init__(now, rng)
        self.targets: list[Target] = []
        self.pops: list[tuple[tuple[float, float], float, float]] = []  # (centre, radius, time): caught, expanding
        self.missed = 0
        self.next_spawn = now + self.READY_S
        self.n = 0
        self.said_go = False
        self.end_t: float | None = None
        self.streak = 0
        self._t = 0.0

    def update(self, now, points, tips, segments=(), pens=()):
        self._t = now - self.t0
        if self.over:
            return
        if self._t < self.READY_S:
            return
        if not self.said_go:
            self.said_go = True
            self.sfx.append("answer")
            self.say.append("Go!")
        left = self.ROUND_S - (self._t - self.READY_S)
        if left <= 0:
            self.over = True
            self.end_t = now
            self.targets.clear()
            self.sfx.append("answer")
            return
        for t in list(self.targets):
            if _near(points, t.c, t.r + TOUCH_PAD) or _swept(segments, t.c, t.r + TOUCH_PAD):
                self.targets.remove(t)
                fast = now - t.born < 0.9
                self.score += 2 if fast else 1
                self.streak += 1
                self.pops.append((t.c, t.r, now))
                self.sfx.append("found")
                self.next_spawn = min(self.next_spawn, now + 0.25)
            elif now - t.born > t.life:
                self.targets.remove(t)
                self.missed += 1
                self.streak = 0
                self.sfx.append("gone")
        self.pops = [p for p in self.pops if now - p[2] < 0.45]
        want = 1 + min(2, self.score // 12)
        if len(self.targets) < want and now >= self.next_spawn:
            self._spawn(now, points)
            self.next_spawn = now + self.rng.uniform(0.25, 0.7)

    def _spawn(self, now, points) -> None:
        r = max(24.0, 38.0 - 0.4 * self.score)  # big and slow at first (3 Nov: "too difficult")
        life = max(2.2, 4.5 - 0.06 * self.score)
        x0, y0, x1, y1 = AREA
        for _ in range(30):
            c = (self.rng.uniform(x0 + r + 6, x1 - r - 6), self.rng.uniform(y0 + r + 6, y1 - r - 6))
            if any(math.dist(c, t.c) < t.r + r + 40 for t in self.targets):
                continue
            if _near(points, c, r + 70):  # not right under a hand: you have to move to it
                continue
            break
        self.n += 1
        self.targets.append(Target(self.n, c, r, now, life))

    def draw(self, c, now):
        t = now - self.t0
        if not self.over and t < self.READY_S:  # 3, 2, 1
            n = str(int(self.READY_S - t) + 1)
            c.number("catch-ready", n, 395.0, 190.0, 26.0, "warn")
            return
        for tg in self.targets:
            left = max(0.0, 1.0 - (now - tg.born) / tg.life)
            pulse = 1.0 + 0.08 * math.sin((now - tg.born) * 9)
            role = "target" if left > 0.35 else "warn"
            c.circle(f"tg-{tg.n}", tg.c, tg.r * pulse, role, n=32)
            c.circle(f"tg-{tg.n}-in", tg.c, 0.45 * tg.r * pulse, role, n=20)
            c.arc(f"tg-{tg.n}-life", tg.c, tg.r + 9, left, role)
        for k, (cc, r, t0) in enumerate(self.pops):
            a = (now - t0) / 0.45
            c.circle(f"pop-{k}-{int(t0 * 1000)}", cc, r * (1 + 1.6 * a), "done", n=28)
        if self.over:
            c.number("catch-score", str(self.score), 380.0 - 18 * (len(str(self.score)) - 1), 170.0, 26.0, "done")

    def hud(self, now, best):
        if self.over:
            rec = " · NEW BEST!" if self.score >= best and self.score > 0 else f" · BEST {best}"
            return f"TIME! YOU CAUGHT {self.score}{rec}", "HOLD PLAY AGAIN, OR SAY MENU", "done"
        t = now - self.t0
        if t < self.READY_S:
            return "CATCH THE LIGHT: TOUCH EVERY LIGHT", "BEFORE IT FADES. GET READY...", "warn"
        left = max(0.0, self.ROUND_S - (t - self.READY_S))
        return f"SCORE {self.score}" + (f" · STREAK {self.streak}" if self.streak >= 3 else ""), \
            f"{int(math.ceil(left))} S LEFT · BEST {best}", "info"


# ---------------------------------------------------------------------------------------------------- follow


class FollowLights(Game):
    key = "follow"
    PAD_R = 50.0
    ROLES = ("target", "source", "warn", "error")
    TONES = ("press", "found", "scan", "answer")

    def __init__(self, now, rng):
        super().__init__(now, rng)
        cx, cy = 410.0, 242.0
        self.pads = [(cx - 92, cy - 78), (cx + 92, cy - 78), (cx - 92, cy + 78), (cx + 92, cy + 78)]
        self.seq: list[int] = []
        self.state = "pause"
        self.state_t = now + 1.2
        self.show_k = 0
        self.input_k = 0
        self.lit: int | None = None
        self.lit_until = 0.0
        self.touching: int | None = None
        self.touch_t = 0.0
        self.released = True
        self._grow()

    def _grow(self) -> None:
        self.seq.append(self.rng.randrange(4))
        self.score = len(self.seq) - 1

    def _speed(self) -> tuple[float, float]:
        on = max(0.32, 0.62 - 0.03 * len(self.seq))
        return on, on * 0.45

    def update(self, now, points, tips, segments=(), pens=()):
        if self.over:
            return
        on, off = self._speed()
        if self.state == "pause":
            if now >= self.state_t:
                self.state, self.show_k, self.state_t = "show", 0, now
            return
        if self.state == "show":
            k = int((now - self.state_t) // (on + off))
            phase = (now - self.state_t) % (on + off)
            if k >= len(self.seq):
                self.state, self.input_k, self.lit = "input", 0, None
                self.released = True
                return
            pad = self.seq[k] if phase < on else None
            if pad is not None and (self.lit != pad or k != self.show_k):
                self.sfx.append(self.TONES[pad])
            self.lit, self.show_k = pad, k
            return
        # input: a pad counts once a hand has been on it a moment, and again only after the hand left it
        on_pad = next((i for i, p in enumerate(self.pads) if _near(points, p, self.PAD_R)), None)
        if on_pad is None:
            self.touching, self.released = None, True
            if now > self.lit_until:
                self.lit = None
            return
        if on_pad != self.touching:
            self.touching, self.touch_t = on_pad, now
        if not self.released or now - self.touch_t < 0.12:
            return
        self.released = False
        self.lit, self.lit_until = on_pad, now + 0.35
        if on_pad == self.seq[self.input_k]:
            self.sfx.append(self.TONES[on_pad])
            self.input_k += 1
            if self.input_k == len(self.seq):
                self._grow()
                self.say.append(f"{len(self.seq) - 1}!" if len(self.seq) - 1 not in (5, 10, 15) else
                                f"{len(self.seq) - 1} in a row. Nice.")
                self.state, self.state_t = "pause", now + 0.9
        else:
            self.sfx.append("error")
            self.over = True
            self.wrong = (on_pad, self.seq[self.input_k])
            self.say.append(f"Oops. You remembered {self.score} in a row.")

    def draw(self, c, now):
        for i, p in enumerate(self.pads):
            lit = self.lit == i
            role = self.ROLES[i]
            if self.over and getattr(self, "wrong", (None, None))[1] == i:
                lit = int(now * 4) % 2 == 0  # the one it should have been, blinking
            c.circle(f"pad-{i}", p, self.PAD_R, role, n=36)
            if lit:
                c.circle(f"pad-{i}-lit", p, self.PAD_R * 0.62, role, n=28)
                c.circle(f"pad-{i}-lit2", p, self.PAD_R * 0.28, role, n=18)
        if self.state == "input" and not self.over:
            c.arc("follow-progress", (410.0, 242.0), 26.0, self.input_k / max(1, len(self.seq)), "done")

    def hud(self, now, best):
        if self.over:
            rec = " · NEW BEST!" if self.score >= best and self.score > 0 else f" · BEST {best}"
            return f"GAME OVER: {self.score} IN A ROW{rec}", "HOLD PLAY AGAIN, OR SAY MENU", "error"
        if self.state in ("pause", "show"):
            return "FOLLOW THE LIGHTS: WATCH...", f"LEVEL {len(self.seq)} · BEST {best}", "warn"
        return "YOUR TURN: TOUCH THEM IN THE SAME ORDER", f"{self.input_k} OF {len(self.seq)} · BEST {best}", "info"


# ---------------------------------------------------------------------------------------------------- draw


class DrawGuess(Game):
    key = "draw"
    draws_hands = True
    LIFT_S = 0.6  # the drawing hand unseen this long ends its line (hands are found in only some frames)
    REJOIN = 80.0  # design px: a pinched hand found again this close to where the line stopped carries it on;
    # a pen that moves further than this between two frames slipped (tracking), and starts a new line there
    IDLE_S = 1.8  # hands away this long after drawing: guess
    STEP = 4.0  # design px between stroke points

    def __init__(self, now, rng, ask=None):
        super().__init__(now, rng)
        self.ask = ask  # callable(jpeg, not_these) -> None, starts a guess; ``answer`` gets (guess, say) or None
        self.wrong: list[str] = []  # guesses the player said no to, for this drawing
        self.strokes: list[list[tuple[float, float]]] = []
        self.cur: list[tuple[float, float]] | None = None
        self.cur_hand = -1  # the hand drawing ``cur``
        self.tip_t = -1e9
        self.hand_t = now
        self.pens: list = []  # this frame's, for the cursors
        self.guessing_since: float | None = None
        self.guess: str | None = None
        self.guessed_n = 0  # strokes the last guess saw
        self.n_guesses = 0

    def update(self, now, points, tips, segments=(), pens=()):
        """Thumb and index finger together put the pen down; the line follows the index fingertip until they open."""
        if points:
            self.hand_t = now
        self.pens = list(pens)
        if self.cur is not None:
            pen = next((p for p in pens if p.hand == self.cur_hand), None)
            if pen is None:  # its hand lost and found again, maybe as another hand slot: still pinched, close by
                back = [p for p in pens if p.down and math.dist(p.at, self.cur[-1]) <= self.REJOIN]
                pen = min(back, key=lambda p: math.dist(p.at, self.cur[-1]), default=None)
                if pen is not None:
                    self.cur_hand = pen.hand
            if pen is not None and not pen.down:  # fingers opened: the line ends here
                self._lift()
            elif pen is None:  # the hand is not seen: wait a moment before ending its line
                if now - self.tip_t > self.LIFT_S:
                    self._lift()
            else:
                self.tip_t = now
                d = math.dist(self.cur[-1], pen.at)
                if d > self.REJOIN or len(self.cur) > 400:  # a slip, or very long: a new line, same hand, from here
                    self._lift()
                    self.cur, self.cur_hand = [pen.at], pen.hand
                    self.strokes.append(self.cur)
                elif d >= self.STEP:
                    self.cur.append(pen.at)
        if self.cur is None:
            pen = next((p for p in pens if p.down), None)
            if pen is not None:
                if self.guess is not None:  # drawing again after a guess: a new picture
                    self.strokes, self.guess, self.guessed_n, self.wrong = [], None, 0, []
                self.cur, self.cur_hand, self.tip_t = [pen.at], pen.hand, now
                self.strokes.append(self.cur)
        ink = sum(len(s) for s in self.strokes)
        if (self.guessing_since is None and self.cur is None and ink >= 6 and len(self.strokes) != self.guessed_n
                and now - self.hand_t >= self.IDLE_S):
            self._guess(now)

    def _lift(self) -> None:
        if self.cur is not None and len(self.cur) < 2:
            self.strokes.remove(self.cur)
        self.cur, self.cur_hand = None, -1

    def _guess(self, now) -> None:
        if self.ask is None:
            self.say.append("I need a Meta model to guess. Start with --voice or --names meta.")
            self.guessed_n = len(self.strokes)
            return
        self.guessing_since = now
        self.guessed_n = len(self.strokes)
        self.sfx.append("listen")
        self.ask(self.picture(), list(self.wrong))

    def answer(self, guess: str | None, say: str | None, now: float) -> None:
        self.guessing_since = None
        if not guess:
            self.say.append("Hmm, I can't tell. Add a bit more?")
            self.sfx.append("error")
            return
        self.guess = guess.upper()
        self.n_guesses += 1
        self.score = self.n_guesses
        self.sfx.append("answer")
        self.say.append(say or f"Is it {guess}?")

    def picture(self) -> bytes:
        """The drawing as a JPEG: dark lines on white, cropped to the strokes with a margin."""
        pts = np.array([p for s in self.strokes for p in s], dtype=np.float64)
        x0, y0 = pts.min(axis=0) - 20
        x1, y1 = pts.max(axis=0) + 20
        side = max(x1 - x0, y1 - y0)
        f = 448.0 / side
        img = np.full((512, 512, 3), 255, np.uint8)
        ox, oy = (512 - (x1 - x0) * f) / 2, (512 - (y1 - y0) * f) / 2
        for s in self.strokes:
            q = np.array([((x - x0) * f + ox, (y - y0) * f + oy) for x, y in s], dtype=np.int32)
            if len(q) >= 2:
                cv2.polylines(img, [q], False, (20, 20, 20), 9, cv2.LINE_AA)
        _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 90])
        return buf.tobytes()

    def hear(self, low, now):
        words = re.sub(r"[^a-z' ]+", " ", low).split()
        if any(w in low for w in ("clear", "start over", "erase", "new drawing", "draw again")):
            self.strokes, self.cur, self.cur_hand, self.guess, self.guessed_n, self.wrong = [], None, -1, None, 0, []
            self.sfx.append("gone")
            return True
        if self.guess is not None:
            said = re.search(r"\b(?:it's|it is|its|that's|that is|this is) (?:a |an |the )?([a-z' ]{2,30})$", low.strip(" .!?"))
            neg = bool(words) and (words[0] in ("no", "nope", "nah", "wrong") or "not" in words[:4]
                                   or "guess again" in low or "try again" in low)
            if neg and said and "not" not in said.group(1).split()[:1]:
                neg = "not" in low.split(said.group(0))[0].split()[-2:] if said.group(0) in low else neg
            if said and not neg:  # "it's a cat": the answer
                thing = said.group(1).strip()
                right = thing.lower() == self.guess.lower()
                self.score += 1 if right else 0
                self.sfx.append("answer")
                self.say.append("Yes! I got it." if right else f"A {thing}! I'll remember that. Draw another one.")
                self.guess = thing.upper() if not right else self.guess
                return True
            if neg:
                self.wrong.append(self.guess.lower())
                self.say.append("Hmm, let me look again.")
                self.guess = None
                self._guess(now)
                return True
            if words and words[0] in ("yes", "yeah", "yep", "correct", "right", "exactly") or "you got it" in low:
                self.score += 1
                self.sfx.append("answer")
                self.say.append("Yes! Draw another one.")
                return True
        if any(w in low for w in ("guess", "what is it", "what did i draw", "what's this", "what is this")):
            if self.strokes and self.guessing_since is None:
                self._guess(now)
            return True
        return False

    def draw(self, c, now):
        for k, s in enumerate(self.strokes):
            if len(s) >= 2:
                c.poly(f"ink-{k}-{len(s) // 50}", s, "source", glow=True)
        if self.guess:
            pts = np.array([p for s in self.strokes for p in s]) if self.strokes else np.array([[410.0, 240.0]])
            x = float(np.clip(pts[:, 0].mean() - 7.4 * len(self.guess), AREA[0], AREA[2] - 15 * len(self.guess)))
            y = float(np.clip(pts[:, 1].max() + 30, AREA[1] + 20, AREA[3] - 10))
            c.label("guess", (x, y), self.guess + "?", "done", type_ms=40 * len(self.guess))
        for p in self.pens:  # the cursor: a ring while the fingers are open, a bright dot in a ring while drawing
            if p.down:
                c.dot(f"pen{p.hand}-dot", p.at, "target")
            c.circle(f"pen{p.hand}", p.at, 6 if p.down else 9, "target" if p.down else "info", n=16)

    def hud(self, now, best):
        if self.guessing_since is not None:
            return "HMM... MUSE SPARK IS LOOKING", "", "warn"
        if self.guess:
            return f"IT'S A {self.guess}!"[:44], "PINCH TO DRAW AGAIN, OR SAY MENU", "done"
        if self.cur is not None:
            return "DRAWING: OPEN YOUR FINGERS TO STOP", "HANDS AWAY WHEN DONE, AND I'LL GUESS", "target"
        if self.strokes:
            return "PINCH TO DRAW MORE, OR TAKE YOUR HANDS", "AWAY AND I'LL GUESS WHAT IT IS", "info"
        return "PINCH THUMB AND FINGER TOGETHER TO DRAW", "OPEN THEM TO STOP. HANDS AWAY: I GUESS", "info"


# ---------------------------------------------------------------------------------------------------- rock paper scissors

MOVES = ("rock", "paper", "scissors")
BEATS = {"rock": "scissors", "paper": "rock", "scissors": "paper"}  # what each move beats
VERBS = {"rock": "beats", "paper": "covers", "scissors": "cut"}  # "Paper covers rock."


def _ring(x: float, y: float, r: float, n: int = 24) -> list[tuple[float, float]]:
    return [(x + r * math.cos(a), y + r * math.sin(a)) for a in np.linspace(0, 2 * math.pi, n, endpoint=False)]


def move_icon(c, id: str, move: str, x: float, y: float, s: float, role: str, glow: bool = False) -> None:
    """Rock (a boulder), paper (a sheet with a folded corner) or scissors as a line drawing about ``2 s`` across,
    centred on (x, y)."""
    if move == "rock":  # a boulder, flat underneath, with facets
        outline = ((-1.0, 0.55), (-0.95, 0.05), (-0.7, -0.35), (-0.3, -0.52), (0.1, -0.62), (0.55, -0.48), (0.85, -0.1),
                   (1.0, 0.3), (0.9, 0.55))
        facets = (((-0.7, -0.35), (-0.25, 0.05), (-0.45, 0.55)), ((-0.25, 0.05), (0.4, -0.05)),
                  ((0.55, -0.48), (0.4, -0.05), (1.0, 0.3)))
        c.poly(id, [(x + s * u, y + s * v) for u, v in outline], role, closed=True, glow=glow)
        for k, f in enumerate(facets):
            c.poly(f"{id}-f{k}", [(x + s * u, y + s * v) for u, v in f], role)
    elif move == "paper":
        w, h, e = 0.7 * s, 0.95 * s, 0.34 * s  # half width, half height, the folded corner
        c.poly(id, [(x - w, y - h), (x + w - e, y - h), (x + w, y - h + e), (x + w, y + h), (x - w, y + h)], role,
               closed=True, glow=glow)
        c.poly(f"{id}-fold", [(x + w - e, y - h), (x + w - e, y - h + e), (x + w, y - h + e)], role)
        for k in range(3):  # lines of writing
            yy = y - 0.2 * s + 0.36 * s * k
            c.poly(f"{id}-l{k}", [(x - 0.42 * s, yy), (x + (0.42 if k < 2 else 0.05) * s, yy)], role)
    else:  # scissors: two blades crossed at a pivot, two finger loops
        for k, side in enumerate((-1.0, 1.0)):
            c.poly(f"{id}-loop{k}", _ring(x + side * 0.42 * s, y + 0.64 * s, 0.27 * s), role, closed=True, glow=glow)
            c.poly(f"{id}-blade{k}", [(x + side * 0.26 * s, y + 0.42 * s), (x - side * 0.5 * s, y - 0.95 * s)], role,
                   glow=glow)
        c.dot(f"{id}-pivot", (x, y - 0.05 * s), role)


def mic_icon(c, id: str, x: float, y: float, s: float, role: str) -> None:
    """A microphone about ``2 s`` tall, centred on (x, y): the capsule, its cradle and the stand."""
    w, top, bottom = 0.34 * s, y - 0.95 * s, y + 0.2 * s  # the capsule: half width, top and bottom
    caps = [(x - w * math.cos(a), top + w - w * math.sin(a)) for a in np.linspace(0, math.pi, 9)]
    caps += [(x + w * math.cos(a), bottom - w + w * math.sin(a)) for a in np.linspace(0, math.pi, 9)]
    c.poly(id, caps, role, closed=True)
    r = 0.62 * s
    c.poly(f"{id}-cradle", [(x + r * math.cos(a), y - 0.1 * s + r * math.sin(a)) for a in np.linspace(0, math.pi, 12)],
           role)
    c.poly(f"{id}-stand", [(x, y + 0.52 * s), (x, y + 0.95 * s)], role)
    c.poly(f"{id}-base", [(x - 0.34 * s, y + 0.95 * s), (x + 0.34 * s, y + 0.95 * s)], role)


def _question(c, id: str, x: float, y: float, s: float, role: str) -> None:
    """A question mark about ``2 s`` tall, centred on (x, y)."""
    r = 0.42 * s
    hook = [(x + r * math.cos(a), y - 0.48 * s + r * math.sin(a)) for a in np.radians(np.linspace(190, 430, 16))]
    c.poly(id, hook + [(x, y + 0.32 * s)], role, glow=True)
    c.dot(f"{id}-dot", (x, y + 0.8 * s), role)


def _label_c(c, id: str, x: float, y: float, text: str, role: str) -> None:
    """A label centred on ``x`` (``y`` is its vertical middle)."""
    c.label(id, (x - EM * len(text) / 2, y), text, role)


class RockPaperScissors(Game):
    """3, 2, 1, SHOOT against the table. The table picks its move when the count starts (before it sees the hand),
    then reads the player's: the shape most frames showed from just after SHOOT (``THROW_S``, the throw itself) to
    ``READ_S`` after it, over every hand on the page (``Pen.shape``). Nothing read: it says so and counts again, the
    score as it was. First to ``FIRST_TO`` wins the match; ``score`` is the player's points."""

    key = "rps"
    COUNT = (3, 2, 1)  # one number per beat, then SHOOT; (1, 2, 3) would count up
    BEAT_S = 0.8
    THROW_S = 0.2  # just after SHOOT the hand is still moving into its shape: not read yet ...
    READ_S = 1.0  # ... then read until this long after SHOOT
    HOLD_S = 0.3  # a hand over the table this long starts a count
    RESULT_S = 2.5  # the result stays up this long before the next count
    AGAIN_S = 1.5  # after a move it could not read
    FIRST_TO = 5
    # Design px. A hand is about 170 x 100 of them (the rig's page: 1.1 mm per px): the ring shows where to hold it,
    # in front of the player, and the count, SHOOT and the table's move go beyond it, between the scores.
    STAGE = (410.0, 165.0)
    SIDES = (172.0, 648.0)  # x of the player's score and the table's
    RING = (410.0, 330.0, 60.0)  # where to hold the hand (shown while waiting for it): centre, radius
    ROLES = {"win": "target", "lose": "error", "draw": "warn"}  # the result, for the player
    SOUNDS = {"win": "found", "lose": "gone", "draw": "answer"}

    def __init__(self, now, rng):
        super().__init__(now, rng)
        self.state = "wait"  # wait | count | shoot | result | noread
        self.t = now  # when the state began (shoot: the moment of SHOOT)
        self.beat = -1  # the count's last ticked beat
        self.hand_t: float | None = None  # a hand has been on the page since then
        self.you = self.me = 0  # points: the player's, the table's
        self.mine = self.yours = ""  # this round's moves: the table's, the player's as read
        self.outcome = ""  # win | lose | draw, for the player
        self.headline = ""  # the result for the status line
        self.votes: list[str] = []  # shapes read since SHOOT, one per hand per frame
        self.saw_hand = False  # a hand was on the page while reading
        self.played: tuple[int, tuple[float, float]] | None = None  # (hand slot, fingertip) of the move read
        self.pens: list = []
        self.log: list[str] = []  # for the terminal: what was read (the table session prints these)

    def update(self, now, points, tips, segments=(), pens=()):
        self.pens = list(pens)
        here = bool(pens or points)
        self.hand_t = (self.hand_t if self.hand_t is not None else now) if here else None
        if self.over:
            return
        held = here and now - self.hand_t >= self.HOLD_S
        if self.state == "wait":
            if held:
                self._count(now)
        elif self.state == "count":
            k = int((now - self.t) // self.BEAT_S)
            if k >= len(self.COUNT):  # SHOOT, on the beat (not on the frame that noticed it)
                self.state, self.t = "shoot", self.t + len(self.COUNT) * self.BEAT_S
                self.votes, self.saw_hand = [], False
                self.sfx.append("scan")
            elif k != self.beat:
                self.beat = k
                self.sfx.append("press")
        elif self.state == "shoot":
            if now - self.t >= self.THROW_S:
                self.saw_hand |= bool(pens)
                for p in pens:
                    if p.shape:
                        self.votes.append(p.shape)
                        self.played = (p.hand, p.at)
            if now - self.t >= self.READ_S:
                self._judge(now)
        elif now - self.t >= (self.RESULT_S if self.state == "result" else self.AGAIN_S):
            if max(self.you, self.me) >= self.FIRST_TO:
                # PLAY AGAIN and MENU come up beside the ring once the hand is away: a hand left there could pick one
                self.over = not here
            elif held:
                self._count(now)
            else:
                self.state, self.t = "wait", now

    def _count(self, now: float) -> None:
        """A new round: the table picks its move now, before it has seen the hand."""
        self.state, self.t, self.beat = "count", now, 0
        self.mine = self.rng.choice(MOVES)
        self.yours = self.outcome = ""
        self.played = None
        self.sfx.append("press")

    def _judge(self, now: float) -> None:
        self.t = now
        if not self.votes:
            self.state = "noread"
            self.sfx.append("error")
            self.say.append("I couldn't read your hand. Again!" if self.saw_hand else "I can't see your hand. Again!")
            self.log.append(f"rps: no move read ({'a hand, no shape' if self.saw_hand else 'no hand'}), table had "
                            f"{self.mine}")
            return
        counts = Counter(self.votes)
        top = max(counts.values())
        self.yours = yours = next(s for s in reversed(self.votes) if counts[s] == top)  # a tie: the shape shown last
        mine = self.mine
        if yours == mine:
            self.outcome, said, shown = "draw", f"Both {yours}. Draw!", f"BOTH {yours.upper()}: DRAW"
        elif BEATS[yours] == mine:
            self.you += 1
            self.outcome, said = "win", f"{yours.capitalize()} {VERBS[yours]} {mine}."
            shown = said[:-1].upper() + ": YOU WIN"
        else:
            self.me += 1
            self.outcome, said = "lose", f"{mine.capitalize()} {VERBS[mine]} {yours}."
            shown = said[:-1].upper() + ": TABLE WINS"
        if self.outcome != "draw":
            done = max(self.you, self.me) >= self.FIRST_TO
            a, b = (self.you, self.me) if self.outcome == "win" else (self.me, self.you)
            said += (f" {'You win' if self.outcome == 'win' else 'I win'} the match, {a} to {b}!" if done else
                     " You win!" if self.outcome == "win" else " I win!")
            shown += " THE MATCH!" if done else "!" if self.outcome == "win" else ""
        self.state, self.headline, self.score = "result", shown, self.you
        self.sfx.append(self.SOUNDS[self.outcome])
        self.say.append(said)
        self.log.append(f"rps: read {yours} ({', '.join(f'{s} {n}' for s, n in counts.most_common())} frames), table "
                        f"{mine}: {self.outcome}, {self.you}:{self.me}")

    def draw(self, c, now):
        x, y = self.STAGE
        if self.over:
            self._final(c)
            return
        self._scores(c)
        if self.state == "wait":  # the three moves, to show what to make, and where to hold the hand
            for k, move in enumerate(MOVES):
                mx = x + 112 * (k - 1)
                move_icon(c, f"rps-hint-{move}", move, mx, y - 14, 30.0, "info")
                _label_c(c, f"rps-hint-{move}-t", mx, y + 46, move.upper(), "info")
            rx, ry, rr = self.RING
            rr += 4 * math.sin(now * 4)
            for k in range(12):  # a dashed ring
                a0 = 2 * math.pi * k / 12
                c.poly(f"rps-ring-{k}", [(rx + rr * math.cos(a), ry + rr * math.sin(a))
                                         for a in np.linspace(a0, a0 + 0.3, 4)], "target")
            _label_c(c, "rps-ring-t", rx, ry, "HAND", "target")
        elif self.state == "count":
            k = min(int((now - self.t) // self.BEAT_S), len(self.COUNT) - 1)
            c.number("rps-count", str(self.COUNT[k]), x - 26, y - 52, 52.0, "warn")
            c.arc("rps-count-arc", (x, y), 80.0, (now - self.t) / (len(self.COUNT) * self.BEAT_S), "warn")
        elif self.state == "shoot":  # a burst around the word
            f = min(1.0, (now - self.t) / 0.3)
            for k in range(12):
                a = 2 * math.pi * (k + 0.5) / 12
                r0, r1 = 56 + 16 * f, 72 + 24 * f
                c.poly(f"rps-burst-{k}", [(x + r0 * math.cos(a), y + r0 * math.sin(a)),
                                          (x + r1 * math.cos(a), y + r1 * math.sin(a))], "target")
            _label_c(c, "rps-shoot", x, y, "SHOOT!", "target")
            p = next((p for p in self.pens if p.shape), None)
            if p is not None:  # what the camera reads, as it reads it
                self._at_hand(c, p.at, f"{p.shape.upper()}?", "info")
        elif self.state == "result":
            role = self.ROLES[self.outcome]
            move_icon(c, "rps-mine", self.mine, x, y - 10, 56.0, role, glow=True)
            _label_c(c, "rps-mine-t", x, y + 78, f"TABLE: {self.mine.upper()}", role)
            if self.played is not None:  # the player's move, at the hand that made it (or where it was)
                at = next((p.at for p in self.pens if p.hand == self.played[0]), self.played[1])
                self._at_hand(c, at, f"YOU: {self.yours.upper()}", role)
        else:  # noread
            _question(c, "rps-what", x, y, 52.0, "warn")

    def _scores(self, c) -> None:
        """YOU and the player's points at the left, TABLE and the table's at the right; the side that just scored lit
        in the result's colour."""
        lit = {"win": "you", "lose": "table"}.get(self.outcome) if self.state == "result" else None
        for name, x, n in (("you", self.SIDES[0], self.you), ("table", self.SIDES[1], self.me)):
            role = self.ROLES[self.outcome] if name == lit else "info"
            _label_c(c, f"rps-{name}", x, 108.0, name.upper(), role)
            c.number(f"rps-{name}-n", str(n), x - 16, 128.0, 32.0, role)

    def _final(self, c) -> None:
        """The match: YOU n : m TABLE, big, above the PLAY AGAIN and MENU bubbles."""
        x, y, w = 410.0, 168.0, 42.0
        role = "target" if self.you > self.me else "error"
        c.number("rps-final-you", str(self.you), x - 36 - w, y - w, w, role)
        c.number("rps-final-table", str(self.me), x + 36, y - w, w, role)
        c.dot("rps-final-c1", (x, y - 18), role)
        c.dot("rps-final-c2", (x, y + 18), role)
        _label_c(c, "rps-final-y", x - 36 - w - 50, y, "YOU", "info")
        _label_c(c, "rps-final-t", x + 36 + w + 62, y, "TABLE", "info")

    def _at_hand(self, c, at, text: str, role: str) -> None:
        """``text`` by the hand whose fingertip is at ``at``: beyond the fingertip (the hand and arm cover the near
        side), else beside the hand, else in front of the player; the first spot that covers neither the stage, a
        score nor the hand (else the one that covers least)."""
        w, h = EM * len(text), 26.0
        x, y = at
        sx, sy = self.STAGE
        taken = [(sx - 112, sy - 98, sx + 112, sy + 92), (x - 45, y - 8, x + 45, y + 160)]  # the stage, the hand
        taken += [(s - 40, 92.0, s + 40, 222.0) for s in self.SIDES]  # the scores: just under one, it reads as theirs
        best, least = (0.0, 0.0), math.inf
        for cx, cy in ((x, y - 36), (x - 53 - w / 2, y + 12), (x + 53 + w / 2, y + 12), (self.RING[0], AREA[3])):
            lx = min(max(cx - w / 2, AREA[0]), AREA[2] - w)
            ly = min(max(cy, AREA[1] + h / 2), AREA[3] - h / 2)
            cover = sum(max(0.0, min(lx + w, x1) - max(lx, x0)) * max(0.0, min(ly + h / 2, y1) - max(ly - h / 2, y0))
                        for x0, y0, x1, y1 in taken)
            if cover < least:
                best, least = (lx, ly), cover
            if cover == 0:
                break
        c.label("rps-yours", best, text, role)

    def hud(self, now, best):
        score = f"YOU {self.you} : {self.me} TABLE"
        if self.over:
            if self.you > self.me:
                return f"YOU WIN THE MATCH · {score}", "HOLD PLAY AGAIN, OR SAY MENU", "target"
            return f"TABLE WINS THE MATCH · {score}", "HOLD PLAY AGAIN, OR SAY MENU", "error"
        if self.state == "count":
            return "GET READY: SHOW YOUR MOVE ON SHOOT", f"{score} · FIRST TO {self.FIRST_TO}", "warn"
        if self.state == "shoot":
            return "SHOOT! HOLD YOUR MOVE STILL", f"{score} · FIRST TO {self.FIRST_TO}", "target"
        if self.state == "result":
            more = "TAKE YOUR HAND AWAY" if max(self.you, self.me) >= self.FIRST_TO else f"FIRST TO {self.FIRST_TO}"
            return self.headline, f"{score} · {more}", self.ROLES[self.outcome]
        if self.state == "noread":
            return "I COULDN'T SEE YOUR MOVE", "AGAIN: SHOW IT ON SHOOT AND HOLD IT", "warn"
        return "HOLD YOUR HAND OVER THE RING", f"FIRST TO {self.FIRST_TO} WINS · {score}", "info"


# ---------------------------------------------------------------------------------------------------- host


@dataclass
class GameHost:
    """The menu, the game being played and the best scores."""

    state_dir: Path | None = None
    seed: int | None = None
    ask_guess: object = None  # callable(jpeg) -> None (the session's brain job)
    timers: Timers = field(default_factory=Timers)  # kitchen timers: they outlive every tile
    voice: bool = True  # something always listens: hints may say "or say menu" (a gated mic does not count)
    listens: bool = False  # the helper tile can hear: voice is on (gated or not)
    thinks: bool = False  # the helper tile has a brain for jobs (games start without one)
    game: Game | None = None
    best: dict = field(default_factory=dict)
    again: Bubble | None = None
    home: Bubble | None = None
    last: float = 0.0

    def __post_init__(self) -> None:
        self.rng = random.Random(self.seed)
        if self.state_dir is not None:
            try:
                self.best = json.loads((Path(self.state_dir) / "play-games.json").read_text(encoding="utf-8"))
            except (OSError, ValueError):
                self.best = {}

    @property
    def active(self) -> bool:
        return self.game is not None

    def start(self, key: str, now: float) -> list[str]:
        self.again = self.home = None
        if key == "objects":  # the object board: no game; the table session learns the table and scans again
            self.game = None
            return ["Object board."]  # the session says what to do: learn the table, or put things down
        if key == "menu":
            self.game = Menu(now, self.rng, self.best)
            return ["Pick a game, or the object board."]
        if key == "helper":
            from .helper import Helper  # helper builds on this module

            self.game = Helper(now, self.rng, listens=self.listens, thinks=self.thinks, timers=self.timers)
            return ["Hold the microphone and tell me what you want to do." if self.listens else
                    "I can't hear you: voice is off."]
        if key == "catch":
            self.game = CatchLight(now, self.rng)
            return ["Catch the light! Touch every light before it fades."]
        if key == "follow":
            self.game = FollowLights(now, self.rng)
            return ["Follow the lights. Watch, then touch them in the same order."]
        if key == "slice":
            from .fruit import SliceFruit  # fruit builds on this module

            self.game = SliceFruit(now, self.rng)
            return ["Slice the fruit! Swipe a finger through it as it flies."]
        if key == "draw":
            self.game = DrawGuess(now, self.rng, ask=self.ask_guess)
            return [("Pinch your thumb and finger together to draw, and open them to stop. "
                     "Take your hands away and I'll guess.")]
        if key == "rps":
            self.game = RockPaperScissors(now, self.rng)
            return ["Rock, paper, scissors! First to five wins. Hold your hand over the ring."]
        if key == "chess":
            from .chesstable import ChessTable  # chesstable builds on this module

            self.game = ChessTable(now, self.rng, state_dir=self.state_dir)
            return ["Let's play chess. Set up the pieces. You are white, on your side."]
        raise ValueError(f"unknown game {key!r}")

    def stop(self) -> None:
        self.game = None

    def update(self, now: float, points: list, tips: list, segments=(), pens=()) -> tuple[list[str], list[str]]:
        """(lines to say, sounds) for this frame. ``segments``: where hand points moved since the frame before
        (a fast hand passes over a target between two frames); ``pens``: each hand as a :class:`Pen`."""
        g = self.game
        if g is None:
            return [], []
        dt = min(max(now - self.last, 0.0), 0.1)
        self.last = now
        g.update(now, points, tips, segments, pens)
        say, sfx = g.say, g.sfx
        g.say, g.sfx = [], []
        if isinstance(g, Menu) and g.choice:
            say += self.start(g.choice, now)
        elif g.over and not isinstance(g, Menu):
            if g.key not in self.best or g.score > self.best.get(g.key, 0):
                if g.score > 0:
                    self.best[g.key] = g.score
                    self._save()
            if self.again is None:
                (ax, ay, ar), (hx, hy, hr) = g.END_BUBBLES
                self.again = Bubble("again", "PLAY\nAGAIN", (ax, ay), ar)
                self.home = Bubble("home", "MENU", (hx, hy), hr)
                self.again.dwell = self.home.dwell = -1.0  # a moment before they can be picked
            elif self.again.step(points, dt):
                sfx.append("press")
                say += self.start(g.key, now)
            elif self.home.step(points, dt):
                sfx.append("press")
                say += self.start("menu", now)
        return say, sfx

    def _save(self) -> None:
        if self.state_dir is None:
            return
        try:
            (Path(self.state_dir) / "play-games.json").write_text(json.dumps(self.best) + "\n", encoding="utf-8")
        except OSError:
            pass

    def draw(self, c, now: float) -> None:
        if self.game is None:
            return
        menu = isinstance(self.game, Menu)
        if menu:
            self.game.clocks_below = self.timers.active
        self.game.draw(c, now)
        if self.again is not None:
            self.again.draw(c, "target")
            self.home.draw(c, "info")
        self.timers.draw(c, now, rows=1 if menu else 2)  # the menu's labels come down to the second line

    def hud(self, now: float) -> tuple[str, str, str]:
        g = self.game
        line1, line2, role = g.hud(now, int(self.best.get(g.key, 0)))
        if not self.voice:
            line1, line2 = _unsaid(line1), _unsaid(line2)
        return _fit(line1), _fit(line2), role

    def hear(self, text: str, now: float) -> list[str] | None:
        """Game words ("let's play", "catch the light", "menu", "stop playing"); None when it is not about games."""
        low = text.lower()
        if self.game is not None and any(w in low for w in ("object board", "objects", "scan the table", "map the table",
                                                            "recognize", "recognise")):
            return self.start("objects", now)
        if self.game is None and not any(w in low for w in ("play", "game", "let's draw", "lets draw")):
            return None  # not playing: only a sentence about playing starts a game ("what is this drawing?" does not)
        if any(w in low for w in ("stop playing", "stop the game", "exit game", "no more games", "back to the table")):
            self.stop()
            return ["Object board."]
        bare = low.strip(" .!?")
        if bare in ("play", "games", "game", "menu") or any(
                w in low for w in ("menu", "other game", "another game", "let's play", "play a game", "games")):
            return self.start("menu", now)
        for key, words in WORDS.items():
            if any(w in low for w in words) and (self.game is None or self.game.key != key or self.game.over):
                return self.start(key, now)
        if self.game is not None and self.game.hear(low, now):
            say, self.game.say = self.game.say, []
            return say
        if self.game is not None and any(w in low for w in ("again", "restart", "one more")) and self.game.over:
            return self.start(self.game.key, now)
        return None


def now_s() -> float:
    return time.monotonic()
