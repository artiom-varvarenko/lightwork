"""Hand games on the projected table (``lightwork play --table --game ...``): nothing to set up, two bare hands.

``catch``   Catch the light: glowing targets appear anywhere on the table; touch them before they fade. 45 s.
``follow``  Follow the lights: four pads light up in a sequence; touch them back in the same order. It grows.
``draw``    Draw and guess: pinch thumb and index finger together and draw in light (open them to lift the pen);
            take your hands away and Muse Spark guesses what it is.
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
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

AREA = (82.0, 74.0, 738.0, 410.0)  # design px: where game things go (right of the button column, under the status bar)
GAMES = ("catch", "follow", "draw")
TITLES = {"catch": "CATCH THE LIGHT", "follow": "FOLLOW THE LIGHTS", "draw": "DRAW AND GUESS"}
DWELL_S = 0.9  # a fingertip held this long on a bubble picks it
TOUCH_PAD = 20.0  # design px of slack around a target: hands are tracked, not exact
WORDS = {  # spoken: game key
    "catch": ("catch", "catch the light", "light game"),
    "follow": ("follow", "follow the lights", "simon", "memory game", "memory"),
    "draw": ("draw", "drawing", "guess what i draw", "pictionary", "let's draw"),
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
    """A hand as a pen: its index fingertip (design px, the cursor) and whether thumb and index finger touch."""

    at: tuple[float, float]
    down: bool
    hand: int = 0  # the table's hand slot: a line stays with the hand that started it


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
        y = 200.0
        self.bubbles = [Bubble("catch", "CATCH\nTHE LIGHT", (165.0, y), 46.0),
                        Bubble("follow", "FOLLOW\nTHE LIGHTS", (330.0, y), 46.0),
                        Bubble("draw", "DRAW\nAND GUESS", (495.0, y), 46.0),
                        Bubble("objects", "OBJECT\nBOARD", (660.0, y), 46.0)]
        self.choice: str | None = None
        self.last = now

    def update(self, now, points, tips, segments=(), pens=()):
        dt = min(max(now - self.last, 0.0), 0.1)
        self.last = now
        for b in self.bubbles:
            if b.step(points, dt):
                self.choice = b.key
                self.sfx.append("press")

    def draw(self, c, now):
        roles = {"catch": "target", "follow": "source", "draw": "warn", "objects": "info"}
        for b in self.bubbles:
            best = self.best.get(b.key)
            b.draw(c, roles[b.key], f"BEST {best}" if best else "")
            self._icon(c, b, roles[b.key], now)

    @staticmethod
    def _icon(c, b: Bubble, role: str, now: float) -> None:
        x, y = b.c
        if b.key == "catch":  # a pulsing light
            r = 14 + 4 * math.sin(now * 5)
            c.circle("ico-catch", (x, y), r, role, n=20)
            c.dot("ico-catch-d", (x, y), role)
        elif b.key == "follow":  # four pads
            for k, (dx, dy) in enumerate(((-13, -13), (13, -13), (-13, 13), (13, 13))):
                lit = int(now * 2.5) % 4 == k
                c.circle(f"ico-follow-{k}", (x + dx, y + dy), 9 if lit else 6, role, n=14)
        elif b.key == "draw":  # a squiggle
            pts = [(x - 24 + 4 * i, y + 10 * math.sin(i * 0.9 + now * 3)) for i in range(13)]
            c.poly("ico-draw", pts, role)
        else:  # two things on a table, each outlined
            c.rect("ico-obj-a", x - 22, y - 14, x - 2, y + 6, role)
            c.circle("ico-obj-b", (x + 13, y + 6), 10, role, n=16)

    def hud(self, now, best):
        return "PICK ONE: HOLD A FINGER ON IT", "OR SAY: CATCH, FOLLOW, DRAW OR OBJECTS", "info"


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


# ---------------------------------------------------------------------------------------------------- host


@dataclass
class GameHost:
    """The menu, the game being played and the best scores."""

    state_dir: Path | None = None
    seed: int | None = None
    ask_guess: object = None  # callable(jpeg) -> None (the session's brain job)
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
        if key == "catch":
            self.game = CatchLight(now, self.rng)
            return ["Catch the light! Touch every light before it fades."]
        if key == "follow":
            self.game = FollowLights(now, self.rng)
            return ["Follow the lights. Watch, then touch them in the same order."]
        if key == "draw":
            self.game = DrawGuess(now, self.rng, ask=self.ask_guess)
            return [("Pinch your thumb and finger together to draw, and open them to stop. "
                     "Take your hands away and I'll guess.")]
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
                self.again = Bubble("again", "PLAY\nAGAIN", (300.0, 300.0), 40.0)
                self.home = Bubble("home", "MENU", (520.0, 300.0), 40.0)
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
        self.game.draw(c, now)
        if self.again is not None:
            self.again.draw(c, "target")
            self.home.draw(c, "info")

    def hud(self, now: float) -> tuple[str, str, str]:
        g = self.game
        line1, line2, role = g.hud(now, int(self.best.get(g.key, 0)))
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
