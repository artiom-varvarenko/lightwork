"""Timers and stopwatches on the table: they keep running while you get on with something else, whatever tile is up.

"Set a timer for 30 minutes for the oven", "ten minute timer", "start a stopwatch", "how long is left on the oven?", "add
5 minutes", "cancel the timer": understood here, with no model call, so they act as soon as they are heard. A job's oven
step starts one too (:mod:`lightwork.helper`). Each clock is one line along the near edge of the page, with a ring that
runs down. A timer that is up flashes and beeps until it is stopped: by saying so, or by holding a fingertip on it. A
stopwatch stops under a held fingertip, and goes under the next hold. A running timer ignores fingers: an arm resting on
the table edge must not cancel the oven.

Pure logic and drawing (design px), like the games: :class:`lightwork.games.GameHost` owns one :class:`Timers`, which
outlives the tiles, and :class:`lightwork.tabletop.TableSession` feeds it time and fingertips.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass

MAX_CLOCKS = 4
MAX_SECONDS = 24 * 3600.0
ROWS_Y = (397.0, 371.0)  # design px: two lines along the near edge; lightwork.helper keeps its text above them
X0, X1 = 76.0, 620.0  # right of the button column, left of the helper's controls
RING_R = 10.0
RING_W = 28.0  # design px the ring takes before the text
CHAR_W = 15.7  # design px per character: what bold capitals took on the real page
GAP = 22.0  # between two clocks on a line
HOLD_S = 0.9  # a fingertip held this long on a clock that takes touches
LAST_S = 30.0  # the last seconds of a timer are drawn yellow
BEEP_S = 6.0  # a timer that is up beeps this often until it is stopped ...
SAY_AGAIN_S = 60.0  # ... and says so once more after this long
NAME_CHARS = 9
UNITS = {"second": 1.0, "seconds": 1.0, "sec": 1.0, "secs": 1.0, "minute": 60.0, "minutes": 60.0, "min": 60.0,
         "mins": 60.0, "hour": 3600.0, "hours": 3600.0, "hr": 3600.0, "hrs": 3600.0}
ONES = ("zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve",
        "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen")
TENS = {"twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90}
ABOUT_TIMERS = ("timer", "timers", "stopwatch", "stop watch", "countdown", "remind me", "time left", "time is left",
                "how long is left", "how long left", "how much longer", "time remaining")
ASK_WORDS = ("how long", "how much", "time left", "time is left", "time remaining")
STOP_WORDS = ("cancel", "stop", "clear", "dismiss", "delete", "remove", "turn off", "end ", "no more", "reset")
ADDING = re.compile(r"\b(?:add|another|extend\b.*?\bby)\s+(?:\d|an?\s+(?:second|sec|minute|min|hour|hr|half|quarter)|half|quarter)"
                    r"|\d\s+(?:more|extra)\b")  # "add 5 minutes", "another 5 minutes", "5 more minutes"; not "another timer"
QUIET_WORDS = ("stop", "okay", "ok", "dismiss", "thanks", "thank you", "got it", "turn it off", "quiet", "enough", "done")
NOT_NAMES = {"a", "an", "the", "my", "our", "this", "that", "new", "another", "set", "start", "cancel", "stop", "clear",
             "add", "put", "make", "me", "kitchen", "first", "second", "all", "up", "it", "one", "how", "much", "long",
             "left", "time", "is", "on", "in", "for", "to", "of", "and", "timer", "timers", "countdown", "reset", "delete",
             "remove", "dismiss", "turn", "off", "more", "extra", "please", "extend", "what", "whats", "what's"} | set(UNITS)


def _normal(text: str) -> str:
    """Lower case, number words as digits, "half an hour" as seconds: what the patterns below read."""
    t = re.sub(r"[^a-z0-9:.' ]+", " ", text.lower().replace("-", " "))
    t = " ".join(w.strip(".") for w in t.split())
    for word, tens in TENS.items():
        for k in range(1, 10):
            t = re.sub(rf"\b{word} {ONES[k]}\b", str(tens + k), t)
        t = re.sub(rf"\b{word}\b", str(tens), t)
    for k in range(len(ONES) - 1, -1, -1):
        t = re.sub(rf"\b{ONES[k]}\b", str(k), t)
    return t


def parse_duration(text: str) -> float | None:
    """Seconds a sentence names ("10 minutes", "an hour and a half", "half an hour", "1:30"), or None."""
    t = re.sub(r"\b(?:more|extra)\b", " ", _normal(text))  # "5 more minutes"
    total, found = 0.0, False
    for phrase, seconds in (("3 quarters of an hour", 2700.0), ("quarter of an hour", 900.0), ("half an hour", 1800.0),
                            ("half hour", 1800.0)):
        if phrase in t:
            total, found = total + seconds, True
            t = t.replace(phrase, " ")
    clock = re.search(r"\b(\d{1,2}):(\d{2})\b", t)
    if clock:
        total, found = total + 60.0 * int(clock.group(1)) + int(clock.group(2)), True
        t = t.replace(clock.group(0), " ")
    t = re.sub(r"\b(a|an)\s+(?=(?:second|sec|minute|min|hour|hr)s?\b)", "1 ", t)
    units = "|".join(sorted(UNITS, key=len, reverse=True))
    last_unit = None
    for m in re.finditer(rf"(\d+(?:\.\d+)?)(\s+and\s+a\s+half)?\s*({units})\b(\s+and\s+a\s+half)?", t):
        n = float(m.group(1)) + (0.5 if m.group(2) or m.group(4) else 0.0)
        last_unit = UNITS[m.group(3)]
        total, found = total + n * last_unit, True
    if found and last_unit == 3600.0:  # "1 hour 20": the minutes go without saying
        rest = re.search(r"(?:hour|hours|hr|hrs)\s+(?:and\s+)?(\d{1,2})\s*$", t)
        if rest:
            total += 60.0 * int(rest.group(1))
    return total if found and 0 < total <= MAX_SECONDS else None


def spoken_time(seconds: float) -> str:
    s = int(math.ceil(max(seconds, 0.0)))
    h, rest = divmod(s, 3600)
    m, r = divmod(rest, 60)
    parts = []
    if h:
        parts.append(f"{h} hour{'' if h == 1 else 's'}")
    if m:
        parts.append(f"{m} minute{'' if m == 1 else 's'}")
    if r or not parts:
        parts.append(f"{r} second{'' if r == 1 else 's'}")
    return " ".join(parts)


def clock_text(seconds: float) -> str:
    s = int(math.ceil(max(seconds, 0.0)))
    h, rest = divmod(s, 3600)
    m, r = divmod(rest, 60)
    return f"{h}:{m:02d}:{r:02d}" if h else f"{m}:{r:02d}"


def name_in(text: str) -> str:
    """What a timer is for: "oven timer", "timer for the pasta", "remind me to check the cake". "" when it does not say."""
    t = _normal(text)
    for pattern in (r"\b([a-z']+) timer\b", r"\b(?:for|on|to check|to turn|to take)\s+(?:out\s+)?(?:the\s+|my\s+|our\s+)?([a-z']+)",
                    r"\b(?:the|my|our)\s+([a-z']+)"):
        for m in re.finditer(pattern, t):
            word = m.group(1)
            if word not in NOT_NAMES and not word.isdigit():
                return word[:NAME_CHARS].upper()
    return ""


@dataclass
class Clock:
    uid: int
    name: str
    kind: str  # timer | stopwatch
    started: float
    seconds: float = 0.0  # timer: its length; stopwatch: what it had counted when it was last stopped
    running: bool = True
    rang: float | None = None  # a timer's: when it went off
    beeped: float = -1e9
    said_again: bool = False
    held: float = 0.0  # how long a fingertip has been on it
    spent: bool = False  # a hold acted: the finger has to leave before the next one counts

    @property
    def end(self) -> float:
        return self.started + self.seconds

    def counted(self, now: float) -> float:
        """A stopwatch's time."""
        return self.seconds + (now - self.started if self.running else 0.0)

    def text(self, now: float) -> str:
        if self.kind == "stopwatch":
            return f"{self.name} {clock_text(self.counted(now))}"
        return f"{self.name} DONE" if self.rang is not None else f"{self.name} {clock_text(self.end - now)}"


class Timers:
    def __init__(self) -> None:
        self.clocks: list[Clock] = []
        self.sfx: list[str] = []
        self.wanted: str | None = None  # "set a timer" with no time: its name, until the answer to "how long?"
        self._uid = 0

    @property
    def active(self) -> bool:
        return bool(self.clocks)

    # ---- starting and stopping

    def start_timer(self, name: str, seconds: float, now: float) -> str:
        """A countdown; returns what to say. ``name`` as it is shown; a name already running gets a number."""
        if len(self.clocks) >= MAX_CLOCKS:
            return "That's too many timers. Cancel one first."
        name = (name or "TIMER")[:NAME_CHARS].upper()
        taken = {c.name for c in self.clocks}
        shown, k = name, 1
        while shown in taken:
            k += 1
            shown = f"{name[:NAME_CHARS - 2]} {k}"
        self._uid += 1
        self.clocks.append(Clock(self._uid, shown, "timer", now, seconds=float(seconds)))
        self.sfx.append("press")
        about = "Timer set" if shown == "TIMER" else f"{shown.capitalize()} timer set"
        return f"{about}: {spoken_time(seconds)}."

    def _stopwatch(self, now: float) -> str:
        c = next((c for c in self.clocks if c.kind == "stopwatch"), None)
        if c is None:
            if len(self.clocks) >= MAX_CLOCKS:
                return "That's too many timers. Cancel one first."
            self._uid += 1
            self.clocks.append(Clock(self._uid, "STOPWATCH", "stopwatch", now))
            self.sfx.append("press")
            return "Stopwatch started."
        if not c.running:
            c.started, c.running = now, True
            self.sfx.append("press")
            return "Stopwatch running again."
        return f"The stopwatch is at {spoken_time(c.counted(now))}."

    def _stop_watch(self, c: Clock, now: float) -> str:
        c.seconds, c.running = c.counted(now), False
        self.sfx.append("press")
        return f"Stopped at {spoken_time(c.seconds)}."

    def _remove(self, clocks: list[Clock]) -> None:
        self.clocks = [c for c in self.clocks if c not in clocks]
        self.sfx.append("gone")

    def _matching(self, t: str, kinds=("timer", "stopwatch")) -> list[Clock]:
        """The clocks a sentence names; all of them when it names none."""
        named = [c for c in self.clocks if c.kind in kinds and c.name.split()[0].lower() in t.split()]
        return named or [c for c in self.clocks if c.kind in kinds]

    # ---- what was said

    def hear(self, text: str, now: float) -> list[str] | None:
        """Lines to say, or None when the sentence is not about timers."""
        low = " ".join(re.sub(r"[^a-z0-9':.\- ]+", " ", text.lower()).split()).strip(" .")
        ringing = [c for c in self.clocks if c.rang is not None]
        if ringing and low in QUIET_WORDS:  # "stop" while it rings is about the ringing, not the job
            self._remove(ringing)
            return ["Okay."]
        if self.wanted is not None:  # the answer to "how long?"
            name, self.wanted = self.wanted, None
            seconds = parse_duration(text)
            if seconds is not None:
                return [self.start_timer(name, seconds, now)]
        if not any(w in low for w in ABOUT_TIMERS):
            return None
        t = _normal(text)
        if "stopwatch" in low or "stop watch" in low:
            return self._hear_stopwatch(low, now)
        if any(w in low for w in ASK_WORDS):
            return self._hear_ask(t, now)
        seconds = parse_duration(text)
        if seconds is not None and ADDING.search(t):
            targets = self._matching(t, ("timer",))
            if targets:
                c = targets[-1]
                c.seconds += seconds
                c.rang, c.said_again = None, False
                self.sfx.append("press")
                return [f"{spoken_time(c.end - now).capitalize()} left."]
        if any(w in low for w in STOP_WORDS):
            targets = self._matching(t, ("timer",))
            if not targets:
                return ["No timer is running."]
            self._remove(targets)
            return ["Timer cancelled." if len(targets) == 1 else "Timers cancelled."]
        if seconds is not None:
            return [self.start_timer(name_in(text), seconds, now)]
        self.wanted = name_in(text)
        return ["How long?"]

    def _hear_stopwatch(self, low: str, now: float) -> list[str]:
        c = next((c for c in self.clocks if c.kind == "stopwatch"), None)
        if any(w in low for w in ("reset", "clear", "cancel", "delete", "remove", "dismiss")):
            if c is None:
                return ["No stopwatch is running."]
            self._remove([c])
            return ["Stopwatch cleared."]
        if re.search(r"\b(stop|pause|hold)\b", low.replace("stop watch", "stopwatch").replace("stopwatch", "")):
            if c is None:
                return ["No stopwatch is running."]
            return [self._stop_watch(c, now) if c.running else f"It stopped at {spoken_time(c.seconds)}."]
        return [self._stopwatch(now)]

    def _hear_ask(self, t: str, now: float) -> list[str]:
        timers = self._matching(t, ("timer",))
        if not timers:
            return ["No timer is running."]
        out = []
        for c in timers:
            left = "is done" if c.rang is not None else f"has {spoken_time(c.end - now)} left"
            out.append(f"{self._spoken(c)} {left}.")
        return [" ".join(out)]

    # ---- per frame

    def _spoken(self, c: Clock) -> str:
        return "The timer" if c.name == "TIMER" else f"The {c.name.lower()} timer"

    def update(self, now: float, tips=(), dt: float = 0.0) -> list[str]:
        """Lines to say this frame. ``tips``: fingertips (design px), for holding a clock."""
        say: list[str] = []
        for c in list(self.clocks):
            if c.kind == "timer" and c.rang is None and now >= c.end:
                c.rang, c.beeped = now, now
                self.sfx.append("answer")
                say.append(f"{self._spoken(c)} is done.")
            elif c.rang is not None:
                if now - c.beeped >= BEEP_S:
                    c.beeped = now
                    self.sfx.append("answer")
                if not c.said_again and now - c.rang >= SAY_AGAIN_S:
                    c.said_again = True
                    say.append(f"{self._spoken(c)} is still done.")
            say += self._touch(c, now, tips, dt)
        return say

    def _touch(self, c: Clock, now: float, tips, dt: float) -> list[str]:
        if c.kind == "timer" and c.rang is None:  # a running timer takes no touches
            c.held = 0.0
            return []
        x, y, w = self._where(now).get(c.uid, (X0, ROWS_Y[0], 0.0))
        on = any(x - 8 <= p[0] <= x + w + 8 and y - 22 <= p[1] <= y + 22 for p in tips)
        if not on:
            c.held, c.spent = 0.0, False
            return []
        if c.spent:
            return []
        c.held += dt
        if c.held < HOLD_S:
            return []
        c.held, c.spent = 0.0, True
        if c.kind == "stopwatch" and c.running:
            return [self._stop_watch(c, now)]
        self._remove([c])
        return []

    # ---- display

    def _where(self, now: float) -> dict[int, tuple[float, float, float]]:
        """uid -> (x, y, width): along the near edge, then on a second line above it."""
        out = {}
        x, row = X0, 0
        for c in self.clocks:
            w = RING_W + CHAR_W * len(c.text(now))
            if x + w > X1 and x > X0 and row + 1 < len(ROWS_Y):
                x, row = X0, row + 1
            out[c.uid] = (x, ROWS_Y[row], w)
            x += w + GAP
        return out

    def draw(self, c, now: float, *, rows: int = len(ROWS_Y)) -> None:
        """``rows``: how many of the lines may be used; a clock beyond them is left out but keeps running."""
        where = self._where(now)
        for k in self.clocks:
            x, y, _ = where[k.uid]
            if y not in ROWS_Y[:rows]:
                continue
            at = (x + RING_R, y)
            if k.kind == "stopwatch":
                role = "info" if k.running else "warn"
                a = 2 * math.pi * (k.counted(now) % 60.0) / 60.0 - math.pi / 2
                c.circle(f"tm-{k.uid}-ring", at, RING_R, role, n=20)
                c.poly(f"tm-{k.uid}-hand", [at, (at[0] + 8 * math.cos(a), at[1] + 8 * math.sin(a))], role)
            elif k.rang is not None:
                role = "error" if int((now - k.rang) * 2) % 2 == 0 else "warn"
                c.circle(f"tm-{k.uid}-ring", at, RING_R, role, n=20)
                c.dot(f"tm-{k.uid}-dot", at, role)
            else:
                left = k.end - now
                role = "warn" if left <= LAST_S else "info"
                c.arc(f"tm-{k.uid}-arc", at, RING_R, left / max(k.seconds, 1.0), role)
            c.label(f"tm-{k.uid}", (x + RING_W, y), k.text(now), role)
            if k.held > 0:
                c.arc(f"tm-{k.uid}-held", at, RING_R + 7, k.held / HOLD_S, "target")
