"""Slice the fruit (``lightwork play --table --game slice``): the table tosses fruit up from the player's edge; each
piece flies up the table, slows, and falls back, and a quick swipe of a hand through it cuts it in two.

The blade is each hand's index fingertip (``Pen.at``, the games' cursor), and only a moving one cuts: faster than
``MIN_SPEED``, so a hand resting on the table, or held where the fruit will come, never does (the hint then says to
swipe). A swipe is fast enough to blur and the tracker loses the hand for a frame or two: the blade then runs straight
from where the fingertip was last seen to where it is found again, also when the table finds it again as its other
hand (``JOIN_S``). Fruit is line art in the projector's colours (no fills), unit shapes turned as they fly. A cut clips
every part of a fruit at the blade's line, so each half keeps its own outline, closed by a flat cut face, and the flesh
shows along it.

A fruit scores 1; in one swipe the second scores 2 and the third and later 3 each. A round is ``ROUND_S``. "Up" is away
from the player: with ``--rotate 180`` design space reads upright from the player's side, its y growing toward them.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import NamedTuple

from .games import AREA, EM, Game, _seg_dist
from .playground import DESIGN_H

ROUND_S = 60.0
READY_S = 3.0
MIN_SPEED = 260.0  # design px/s (about 0.3 m/s on the rig, 1.1 mm per px): slower is a hand reaching in or resting
MAX_JUMP = 450.0  # design px: the longest blade between two sightings of a hand (0.35 s of a fast swipe)
JOIN_S = 0.35  # a hand found again this soon after its last move, as the other hand, carries the swipe on
LOST_S = 0.5  # a blade whose hand is gone this long is forgotten
MIN_DT = 1 / 60  # s: two sightings are never closer than this (speeds stay finite)
HIT_PAD = 12.0  # design px around a fruit: tracking error, and the fingertip's own width
LAGS = (0.0, 0.05, 0.1, 0.15)  # s: a swipe also cuts where the fruit was this long ago. The player aims at the fruit
# they see, which the projector shows a moment late, and the camera sees the hand a moment late too: a fruit flying
# fast on its way up is a few of its own sizes further on by then
COMBO_S = 0.45  # cuts this close together by one blade are one swipe
TRAIL_S = 0.2  # the drawn blade: where the fingertip went in this long
HOT_S = 0.15  # the blade is drawn this long after it last moved fast enough to cut
SLOW_HINT_S = 1.5  # "swipe faster" stays up this long after a hand touched a fruit too slowly
MAX_AIR = 6  # whole fruit in the air at once
HALF_S = 1.4  # halves fall this long at most
SPLAT_S = 0.7
DROP_S = 0.35
LABEL_S = 0.8
NEAR_Y = DESIGN_H  # the player's edge of the page: fruit comes from just beyond it and falls back there


class Part(NamedTuple):
    pts: tuple  # ((x, y), ...) in fruit units: 1 is the fruit's radius, y toward the player
    role: str
    closed: bool = True
    glow: bool = False


@dataclass(frozen=True)
class Kind:
    name: str
    r: float  # design px: its size, and how near a blade must pass
    parts: tuple  # Part, ...; the first is the outline (its cut face shows the flesh)
    flesh: str  # role of the cut face
    juice: str  # role of the splash
    weight: int = 2  # how often it comes


# ---------------------------------------------------------------------------------------------------- shapes


def _wrap(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


def _bump(a: float, at: float, width: float) -> float:
    return math.exp(-(_wrap(a - at) / width) ** 2)


def _outline(radius, n: int, sx: float = 1.0, sy: float = 1.0) -> tuple:
    """A closed outline from the top: ``radius(a)`` at angle ``a`` (0 along +x, pi/2 toward the player)."""
    out = []
    for k in range(n):
        a = -math.pi / 2 + 2 * math.pi * k / n
        r = radius(a)
        out.append((round(sx * r * math.cos(a), 4), round(sy * r * math.sin(a), 4)))
    return tuple(out)


def _leaf(p0, p1, w: float, n: int = 6) -> tuple:
    """An almond from ``p0`` to ``p1``, ``w`` wide at its middle."""
    (x0, y0), (x1, y1) = p0, p1
    ux, uy = x1 - x0, y1 - y0
    length = math.hypot(ux, uy)
    nx, ny = -uy / length, ux / length
    one = [(x0 + ux * k / n + nx * w * math.sin(math.pi * k / n), y0 + uy * k / n + ny * w * math.sin(math.pi * k / n))
           for k in range(n + 1)]
    two = [(x0 + ux * k / n - nx * w * math.sin(math.pi * k / n), y0 + uy * k / n - ny * w * math.sin(math.pi * k / n))
           for k in range(n - 1, 0, -1)]
    return tuple(one + two)


def _apple() -> Kind:
    body = _outline(lambda a: 1 - 0.2 * _bump(a, -math.pi / 2, 0.3) - 0.06 * _bump(a, math.pi / 2, 0.4), 26, 1.04, 0.95)
    stem = ((0.0, -0.74), (0.05, -0.98), (0.14, -1.18))
    return Kind("apple", 30.0, (Part(body, "error", glow=True), Part(stem, "target", closed=False),
                                Part(_leaf((0.08, -0.96), (0.6, -1.16), 0.13), "target")), "warn", "warn", 3)


def _lemon() -> Kind:
    body = _outline(lambda a: 1 + 0.17 * _bump(a, 0.0, 0.2) + 0.17 * _bump(a, math.pi, 0.2), 26, 0.86, 0.66)
    return Kind("lemon", 27.0, (Part(body, "warn", glow=True),), "info", "warn")


def _watermelon() -> Kind:
    body = _outline(lambda a: 1.0, 30, 1.0, 0.84)
    stripes = []
    for xc in (-0.5, 0.0, 0.5):
        h = 0.84 * math.sqrt(1 - xc * xc) * 0.8
        ys = [-h + 2 * h * k / 6 for k in range(7)]
        stripes.append(Part(tuple((xc + 0.07 * math.sin(7 * y), y) for y in ys), "target", closed=False))
    return Kind("watermelon", 44.0, (Part(body, "target", glow=True), *stripes), "error", "error", 1)


def _banana() -> Kind:
    outer = [(1.55 * math.cos(math.radians(a)), -0.95 + 1.55 * math.sin(math.radians(a)) - 0.3)
             for a in (52, 66, 80, 94, 108, 122, 128)]
    inner = [(1.4 * math.cos(math.radians(a)), -1.3 + 1.4 * math.sin(math.radians(a)) - 0.3)
             for a in (126, 112, 98, 84, 70, 56)]
    body = tuple(outer + [(-1.04, -0.22)] + inner + [(1.05, -0.24)])
    return Kind("banana", 38.0, (Part(body, "warn", glow=True), Part(((-0.98, -0.3), (-1.14, -0.48)), "warn", False)),
                "info", "warn")


def _strawberry() -> Kind:
    def point(a):
        c, s = math.cos(a), math.sin(a)
        return 0.92 * c * (1 - 0.55 * max(0.0, s) ** 1.2), (0.95 * s if s > 0 else 0.7 * s) - 0.12

    body = tuple(point(-math.pi / 2 + 2 * math.pi * k / 22) for k in range(22))
    seeds = tuple(Part(((x, y - 0.05), (x + 0.03, y + 0.06)), "warn", closed=False)
                  for x, y in ((-0.42, -0.4), (0.05, -0.45), (0.45, -0.35), (-0.2, 0.0), (0.25, 0.05), (0.0, 0.4)))
    crown = ((-0.6, -0.78), (-0.42, -1.02), (-0.22, -0.82), (0.0, -1.1), (0.22, -0.82), (0.42, -1.02), (0.6, -0.78))
    return Kind("strawberry", 26.0, (Part(body, "error", glow=True), *seeds, Part(crown, "target", closed=False)),
                "info", "error")


KINDS = {k.name: k for k in (_apple(), _lemon(), _watermelon(), _banana(), _strawberry())}


# ---------------------------------------------------------------------------------------------------- cutting


def _side(q, p, n) -> float:
    return (q[0] - p[0]) * n[0] + (q[1] - p[1]) * n[1]


def _lerp(a, b, t: float) -> tuple[float, float]:
    return a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t


def clip_closed(pts, p, n) -> tuple:
    """The part of the closed outline ``pts`` on ``n``'s side of the line through ``p`` (Sutherland-Hodgman against
    one line): closed again along the line."""
    out = []
    for i, a in enumerate(pts):
        b = pts[(i + 1) % len(pts)]
        da, db = _side(a, p, n), _side(b, p, n)
        if da >= 0:
            out.append(a)
        if (da >= 0) != (db >= 0):
            out.append(_lerp(a, b, da / (da - db)))
    return tuple(out) if len(out) >= 3 else ()


def clip_open(pts, p, n) -> list[tuple]:
    """The runs of the polyline ``pts`` on ``n``'s side of the line through ``p``."""
    runs, cur = [], []
    for i, a in enumerate(pts):
        da = _side(a, p, n)
        if i:
            b = pts[i - 1]
            db = _side(b, p, n)
            if (da >= 0) != (db >= 0):
                cur.append(_lerp(b, a, db / (db - da)))
                if da < 0:
                    runs.append(cur)
                    cur = []
        if da >= 0:
            cur.append(a)
    runs.append(cur)
    return [tuple(r) for r in runs if len(r) >= 2]


def chord(pts, p, n) -> tuple | None:
    """Where the line through ``p`` square to ``n`` crosses the closed outline ``pts``: its two outermost crossings."""
    d = (-n[1], n[0])
    hits = []
    for i, a in enumerate(pts):
        b = pts[(i + 1) % len(pts)]
        da, db = _side(a, p, n), _side(b, p, n)
        if (da >= 0) != (db >= 0):
            hits.append(_lerp(a, b, da / (da - db)))
    if len(hits) < 2:
        return None
    hits.sort(key=lambda q: q[0] * d[0] + q[1] * d[1])
    return hits[0], hits[-1]


def split(kind: Kind, p, n) -> tuple[tuple, tuple]:
    """``kind`` cut along the line through ``p`` (fruit units) square to the unit vector ``n``: (the parts on n's side,
    the parts on the other), each with the flesh along its cut face."""
    halves = []
    for s in (1.0, -1.0):
        ns = (n[0] * s, n[1] * s)
        parts = []
        for part in kind.parts:
            if part.closed:
                pts = clip_closed(part.pts, p, ns)
                if pts:
                    parts.append(part._replace(pts=pts))
            else:
                parts += [part._replace(pts=run) for run in clip_open(part.pts, p, ns)]
        cut = chord(kind.parts[0].pts, p, n)
        if cut is not None:  # the flesh: the cut face a little shorter, just inside the half
            (ax, ay), (bx, by) = cut
            mx, my = (ax + bx) / 2, (ay + by) / 2
            ox, oy = 0.09 * ns[0], 0.09 * ns[1]
            parts.append(Part(((mx + 0.8 * (ax - mx) + ox, my + 0.8 * (ay - my) + oy),
                               (mx + 0.8 * (bx - mx) + ox, my + 0.8 * (by - my) + oy)), kind.flesh, closed=False))
        halves.append(tuple(parts))
    return halves[0], halves[1]


def _world(pts, c, a: float, r: float) -> list[tuple[float, float]]:
    ca, sa = math.cos(a), math.sin(a)
    return [(c[0] + r * (ca * x - sa * y), c[1] + r * (sa * x + ca * y)) for x, y in pts]


# ---------------------------------------------------------------------------------------------------- the game


@dataclass
class Flier:
    """A fruit or a half, thrown at ``t0`` from ``p0`` with ``v0`` (design px/s), falling toward the player at ``g``
    (design px/s^2) and turning ``w`` rad/s."""

    id: str
    kind: Kind
    parts: tuple
    p0: tuple[float, float]
    v0: tuple[float, float]
    g: float
    a0: float
    w: float
    t0: float

    def at(self, t: float) -> tuple[float, float]:
        dt = t - self.t0
        return self.p0[0] + self.v0[0] * dt, self.p0[1] + self.v0[1] * dt + 0.5 * self.g * dt * dt

    def vel(self, t: float) -> tuple[float, float]:
        return self.v0[0], self.v0[1] + self.g * (t - self.t0)

    def angle(self, t: float) -> float:
        return self.a0 + self.w * (t - self.t0)


@dataclass
class Blade:
    """One hand's index fingertip as a blade."""

    at: tuple[float, float]
    t: float  # when it was last seen somewhere new (or first seen): its next move's speed is over the time since
    moved_t: float = -1e9  # when it last moved (a hand just found has not)
    fast_t: float = -1e9  # when it last moved fast enough to cut
    speed: float = 0.0  # design px/s over its last move (0 on a frame it did not move)
    trail: list = field(default_factory=list)  # (t, at), for drawing
    streak: int = 0  # fruit cut in this swipe
    cut_t: float = -1e9


class SliceFruit(Game):
    key = "slice"
    draws_hands = True  # the fingertip is a blade: its own cursor and trail
    ROUND_S = ROUND_S
    READY_S = READY_S

    def __init__(self, now, rng):
        super().__init__(now, rng)
        self.fruit: list[Flier] = []
        self.halves: list[Flier] = []
        self.drops: list[tuple] = []  # (id, p0, v0, t0, role): juice flying off a cut
        self.splats: list[tuple] = []  # (id, centre, r, t0, role): where a fruit was cut
        self.labels: list[tuple] = []  # (id, text, x, y, t0): points for a swipe through more than one
        self.queue: list[tuple[float, Kind]] = []  # (when, kind): the rest of a volley
        self.next_volley = now + READY_S
        self.blades: dict[int, Blade] = {}
        self.pens: list = []
        self.n = 0
        self.cut = self.missed = 0
        self.best_swipe = 0
        self.combo = (0, -1e9)  # (fruit in the swipe, when): for the status line
        self.slow_t = -1e9  # a hand touched a fruit too slowly to cut it
        self.slow_ids: set[str] = set()
        self.said_go = False
        self.log: list[str] = []  # for the terminal (the table session prints these)

    # ---- hands

    def _blades(self, now: float, pens) -> list[tuple]:
        """This frame's cutting moves: (from, to, hand, speed), one per hand that moved fast since it was last seen."""
        moving = {p.hand for p in pens if p.hand in self.blades and p.at != self.blades[p.hand].at}
        segs = []
        for p in pens:
            b = self.blades.get(p.hand)
            if b is None:  # a hand that just showed up (a still one keeps its blade, however long it rests)
                b = self._found_again(p, now, moving) or Blade(p.at, now)
                self.blades[p.hand] = b
            if p.at == b.at:
                b.speed = 0.0
                continue
            d = math.dist(b.at, p.at)
            b.speed = d / max(now - b.t, MIN_DT)
            if b.speed >= MIN_SPEED and d <= MAX_JUMP:
                segs.append((b.at, p.at, p.hand, b.speed))
                b.fast_t = now
            b.at, b.t, b.moved_t = p.at, now, now
            b.trail.append((now, p.at))
        here = {p.hand for p in pens}
        for k, b in list(self.blades.items()):
            if k not in here and now - b.t > LOST_S:
                del self.blades[k]
            else:
                b.trail = [q for q in b.trail if now - q[0] <= TRAIL_S]
        return segs

    def _found_again(self, p, now: float, moving: set) -> Blade | None:
        """The blade of a hand that just showed up as ``p``, if it is a moving hand found again as the other hand:
        the other blade moved within ``JOIN_S``, has not moved since, and is near enough."""
        near = [(math.dist(b.at, p.at), k) for k, b in self.blades.items()
                if k != p.hand and k not in moving and now - b.moved_t <= JOIN_S and math.dist(b.at, p.at) <= MAX_JUMP]
        if not near:
            return None
        _, k = min(near)
        b = self.blades.pop(k)
        self.log.append(f"slice: hand {k} found again as hand {p.hand}, {math.dist(b.at, p.at):.0f} px on")
        return b

    # ---- the round

    def _level(self, now: float) -> float:
        """0 at GO, 1 at the end of the round."""
        return min(1.0, max(0.0, (now - self.t0 - READY_S) / ROUND_S))

    def update(self, now, points, tips, segments=(), pens=()):
        self.pens = list(pens)
        cuts = self._blades(now, self.pens)
        self._fade(now)
        if self.over or now - self.t0 < READY_S:
            return
        if not self.said_go:
            self.said_go = True
            self.sfx.append("answer")
            self.say.append("Go!")
        if now - self.t0 - READY_S >= ROUND_S:
            self._end(now)
            return
        while self.queue and self.queue[0][0] <= now:
            self.toss(self.queue.pop(0)[1], now)
        if not self.queue and now >= self.next_volley:
            self._volley(now)
        for a, b, hand, speed in cuts:
            for f in list(self.fruit):
                if any(_seg_dist(f.at(max(f.t0, now - lag)), a, b) <= f.kind.r + HIT_PAD for lag in LAGS):
                    self._cut(f, a, b, hand, speed, now)
        self._too_slow(now)
        for f in list(self.fruit):
            if f.at(now)[1] > NEAR_Y + f.kind.r and f.vel(now)[1] > 0:  # fell back past the player's edge
                self.fruit.remove(f)
                self.missed += 1

    def _end(self, now: float) -> None:
        self.over = True
        self.fruit.clear()
        self.queue.clear()
        self.sfx.append("answer")
        self.say.append(f"Time! You sliced {self.cut}." if self.cut else "Time!")
        self.log.append(f"slice: round over: {self.cut} cut, {self.missed} fell, score {self.score}, "
                        f"most in one swipe {self.best_swipe}")

    def _volley(self, now: float) -> None:
        """A few fruit, one after another; more, and quicker, as the round goes on."""
        f = self._level(now)
        sizes = (1, 1, 1, 2) if f < 0.2 else (1, 2, 2, 3) if f < 0.55 else (2, 2, 3, 4)
        n = min(self.rng.choice(sizes), MAX_AIR - len(self.fruit))
        if n <= 0:
            self.next_volley = now + 0.3
            return
        t = now
        for k in range(n):
            self.queue.append((t, self._pick()))
            t += self.rng.uniform(0.12, 0.35)
        self.next_volley = t + max(0.9, 2.0 - 0.9 * f) + self.rng.uniform(-0.2, 0.25)

    def _pick(self) -> Kind:
        kinds = list(KINDS.values())
        return self.rng.choices(kinds, weights=[k.weight for k in kinds])[0]

    def toss(self, kind: Kind, now: float, x0: float | None = None, x1: float | None = None, top: float | None = None,
             hang: float | None = None) -> Flier:
        """Throw ``kind`` up from the player's edge at ``x0``: it is highest (at ``top``, design px from the far edge)
        ``hang`` s later and comes back down at ``x1``. Unset: picked at random, quicker as the round goes on."""
        r, lo, hi = kind.r, AREA[0] + kind.r + 24, AREA[2] - kind.r - 24
        if x0 is None:
            recent = [fr.p0[0] for fr in self.fruit if now - fr.t0 < 0.6]
            for _ in range(10):
                x0 = self.rng.uniform(lo, hi)
                if all(abs(x0 - x) > 90 for x in recent):
                    break
        x1 = self.rng.uniform(max(lo, x0 - 260), min(hi, x0 + 260)) if x1 is None else x1
        top = self.rng.uniform(AREA[1] + r + 8, 230.0) if top is None else top
        hang = self.rng.uniform(1.15, 1.35) - 0.3 * self._level(now) if hang is None else hang
        y0 = NEAR_Y + 0.9 * r
        g = 2 * (y0 - top) / hang ** 2
        self.n += 1
        fr = Flier(f"fr{self.n}", kind, kind.parts, (x0, y0), ((x1 - x0) / (2 * hang), -g * hang), g,
                   self.rng.uniform(-0.5, 0.5), self.rng.choice((-1, 1)) * self.rng.uniform(0.8, 2.6), now)
        self.fruit.append(fr)
        self.sfx.append("toss")
        return fr

    def _cut(self, f: Flier, a, b, hand: int, speed: float, now: float) -> None:
        """Cut ``f`` along the blade's line a-b (kept within the middle of the fruit, so both halves show)."""
        self.fruit.remove(f)
        c, ang, r = f.at(now), f.angle(now), f.kind.r
        dl = math.dist(a, b)
        d = ((b[0] - a[0]) / dl, (b[1] - a[1]) / dl)
        n = (-d[1], d[0])
        off = max(-0.3 * r, min(0.3 * r, _side(c, a, n)))  # the line passes this far from the centre
        p = (c[0] - n[0] * off, c[1] - n[1] * off)
        ca, sa = math.cos(-ang), math.sin(-ang)  # into the fruit's own turned units
        px, py = (p[0] - c[0]) / r, (p[1] - c[1]) / r
        nl = (ca * n[0] - sa * n[1], sa * n[0] + ca * n[1])
        one, two = split(f.kind, (ca * px - sa * py, sa * px + ca * py), nl)
        v = f.vel(now)
        for s, parts in ((1.0, one), (-1.0, two)):
            push = self.rng.uniform(70.0, 110.0)
            self.halves.append(Flier(f"{f.id}{'ab'[s < 0]}", f.kind, parts, c,
                                     (v[0] + s * n[0] * push + 40 * d[0], v[1] + s * n[1] * push + 40 * d[1]),
                                     max(f.g, 300.0), ang, f.w + s * self.rng.uniform(1.5, 3.0), now))
        for k in range(7):  # juice
            t = 2 * math.pi * k / 7 + self.rng.uniform(-0.3, 0.3)
            sp = self.rng.uniform(120.0, 260.0)
            self.drops.append((f"{f.id}-j{k}", c, (v[0] * 0.3 + sp * math.cos(t), v[1] * 0.3 + sp * math.sin(t)), now,
                               f.kind.juice))
        self.splats.append((f"{f.id}-sp", c, r, now, f.kind.juice))
        blade = self.blades.get(hand)
        if blade is not None:
            blade.streak = blade.streak + 1 if now - blade.cut_t <= COMBO_S else 1
            blade.cut_t = now
            k = blade.streak
        else:
            k = 1
        pts = min(k, 3)
        self.score += pts
        self.cut += 1
        self.best_swipe = max(self.best_swipe, k)
        if k >= 2:
            self.combo = (k, now)
            self.labels.append((f"{f.id}-pts", f"+{pts}", c[0], c[1] - r - 14, now))
        self.sfx.append("slice")
        self.log.append(f"slice: cut a {f.kind.name} at {speed:.0f} px/s (hand {hand}" +
                        (f", {k} in one swipe)" if k >= 2 else ")"))

    def _too_slow(self, now: float) -> None:
        """A fingertip on a fruit that did not cut it: say to swipe (a fruit flying into a still hand stays whole)."""
        for p in self.pens:
            b = self.blades.get(p.hand)
            if b is None or now - b.fast_t <= HOT_S:
                continue
            for f in self.fruit:
                if math.dist(p.at, f.at(now)) <= f.kind.r:
                    self.slow_t = now
                    if f.id not in self.slow_ids:
                        self.slow_ids.add(f.id)
                        self.log.append(f"slice: hand {p.hand} on a {f.kind.name} at {b.speed:.0f} px/s: too slow "
                                        f"to cut (needs {MIN_SPEED:.0f})")

    def _fade(self, now: float) -> None:
        self.halves = [h for h in self.halves if now - h.t0 < HALF_S and h.at(now)[1] < NEAR_Y + 2 * h.kind.r]
        self.drops = [d for d in self.drops if now - d[3] < DROP_S]
        self.splats = [s for s in self.splats if now - s[3] < SPLAT_S]
        self.labels = [lb for lb in self.labels if now - lb[4] < LABEL_S]

    # ---- drawing

    def draw(self, c, now):
        t = now - self.t0
        if not self.over and t < READY_S:  # 3, 2, 1 (the blade already draws: try a swipe)
            c.number("slice-ready", str(int(READY_S - t) + 1), 395.0, 190.0, 26.0, "warn")
        for sid, (x, y), r, t0, role in self.splats:  # a burst where it was cut
            for k in range(8):
                a = math.pi * k / 4 + 0.3
                r1 = r * (0.95 + 0.35 * ((k * 7) % 3) / 2)
                c.poly(f"{sid}{k}", [(x + 0.6 * r * math.cos(a), y + 0.6 * r * math.sin(a)),
                                     (x + r1 * math.cos(a), y + r1 * math.sin(a))], role)
        for f in self.halves + self.fruit:
            self._draw_flier(c, f, now)
        for did, (x, y), (vx, vy), t0, role in self.drops:  # streaks of juice
            dt = now - t0
            head = (x + vx * dt, y + vy * dt + 150 * dt * dt)
            tail = (x + vx * max(0.0, dt - 0.05), y + vy * max(0.0, dt - 0.05) + 150 * max(0.0, dt - 0.05) ** 2)
            c.poly(did, [tail, head], role)
        for lid, text, x, y, t0 in self.labels:
            lx = min(max(x - EM * len(text) / 2, AREA[0]), AREA[2] - EM * len(text))
            c.label(lid, (lx, max(AREA[1], y - 30 * (now - t0) / LABEL_S)), text, "target")
        if self.over:
            s = str(self.score)
            c.number("slice-score", s, 380.0 - 18 * (len(s) - 1), 170.0, 26.0, "done")
        for k, b in self.blades.items():
            pts = [q for tq, q in b.trail if now - tq <= TRAIL_S]
            if now - b.fast_t <= HOT_S and len(pts) >= 2:
                c.poly(f"blade-{k}", pts, "info", glow=True)
        for p in self.pens:
            c.circle(f"pen{p.hand}", p.at, 8.0, "info", n=16)

    @staticmethod
    def _draw_flier(c, f: Flier, now: float) -> None:
        at, a = f.at(now), f.angle(now)
        for k, part in enumerate(f.parts):
            c.poly(f"{f.id}-{k}", _world(part.pts, at, a, f.kind.r), part.role, closed=part.closed, glow=part.glow)

    def hud(self, now, best):
        if self.over:
            rec = " · NEW BEST!" if self.score >= best and self.score > 0 else f" · BEST {best}"
            return f"TIME! SCORE {self.score}{rec}", "HOLD PLAY AGAIN, OR SAY MENU", "done"
        t = now - self.t0
        if t < READY_S:
            return "SLICE THE FRUIT: SWIPE THROUGH IT", "QUICKLY, WITH A FINGER. GET READY...", "warn"
        k, kt = self.combo
        line1 = f"SCORE {self.score} · {k} IN ONE SWIPE!" if now - kt < 1.2 else f"SCORE {self.score}"
        if now - self.slow_t < SLOW_HINT_S:
            return line1, "SWIPE FASTER: A QUICK FLICK CUTS", "warn"
        left = max(0.0, ROUND_S - (t - READY_S))
        return line1, f"{math.ceil(left)} S LEFT · BEST {best}", "info"


_ICON_N = (0.26, -0.97)  # the icon's cut: nearly flat, the top half (stem and leaf) lifting off like a lid
_ICON_CUT = tuple(half[:-1] for half in split(KINDS["apple"], (0.0, 0.0), _ICON_N))  # no flesh line: too small


def fruit_icon(c, id: str, x: float, y: float, s: float, role: str, now: float) -> None:
    """An apple about ``2 s`` across, cut in two, its halves opening and closing, centred on (x, y)."""
    gap = 0.16 + 0.24 * (0.5 + 0.5 * math.sin(now * 4))
    for h, (sign, parts) in enumerate(((1.0, _ICON_CUT[0]), (-1.0, _ICON_CUT[1]))):
        at = (x + sign * _ICON_N[0] * gap * s, y + 0.12 * s + sign * _ICON_N[1] * gap * s)
        for k, part in enumerate(parts):
            c.poly(f"{id}-{h}{k}", _world(part.pts, at, 0.0, 0.8 * s), role, closed=part.closed)
