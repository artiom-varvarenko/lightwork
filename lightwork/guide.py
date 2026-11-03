"""Guided tasks for ``lightwork play --table --do ...``: any job on the table as a short plan the table walks you through.

A plan is a small, domain-neutral JSON: the **items** the job needs on the table (materials, containers, tools and
measuring tools, with amounts) and **steps** from a fixed vocabulary the camera can follow:

``transfer``  move something from one item into another, N times (scoops of flour into a bowl, eggs, 10 ml from cup 1
              to cup 2, one part into each box). Counted from the hands: a trip from the source to the destination is
              one. ``into`` may name a range of numbered containers (``cups#2-6``): one trip into each, any order.
``use``       pick up an item (the whisk, the screwdriver).
``timer``     wait or work for N seconds (whisk 45 s, let it rest): a countdown ring.
``check``     the camera looks and a Meta model judges a question ("is the batter smooth?").
``confirm``   something the camera cannot follow (pour the milk): done when you say so.

The same steps run a recipe, a lab dilution or a kit check. Plans come from ``lightwork/plans/*.json`` or are written
for a spoken request by Muse Spark (:mod:`lightwork.brain`). The model proposes the plan; the numbers are computed here:
grams to millilitres with a density, millilitres to scoops from the measuring tool's capacity, which comes from its
name (a tablespoon is 15 ml) or from its size on the table (a round scoop of width d holds about a hemisphere,
pi d^3 / 12). Mistakes the table catches: the wrong item picked up, a trip into the wrong container, one too many.

Everything here is pure logic over what :class:`lightwork.tabletop.TableSession` already tracks (objects with a
plan item each, their outlines and the hands, all in design px); it draws with the playground's ``Canvas``.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field, replace
from pathlib import Path

import cv2
import numpy as np

from .benchtask import BenchTask, spoken_list

PLANS_DIR = Path(__file__).parent / "plans"
KINDS = ("material", "container", "tool", "measure")
STEP_KINDS = ("transfer", "use", "timer", "check", "confirm")
MAX_ITEMS = 12
MAX_STEPS = 14
MAX_COUNT = 40
UNITS = {  # unit word -> (canonical unit, factor)
    "g": ("g", 1.0), "gram": ("g", 1.0), "grams": ("g", 1.0), "kg": ("g", 1000.0),
    "ml": ("ml", 1.0), "millilitre": ("ml", 1.0), "milliliter": ("ml", 1.0), "millilitres": ("ml", 1.0),
    "milliliters": ("ml", 1.0), "l": ("ml", 1000.0), "litre": ("ml", 1000.0), "liter": ("ml", 1000.0),
    "tsp": ("ml", 5.0), "teaspoon": ("ml", 5.0), "teaspoons": ("ml", 5.0),
    "tbsp": ("ml", 15.0), "tablespoon": ("ml", 15.0), "tablespoons": ("ml", 15.0),
    "cup": ("ml", 240.0), "cups": ("ml", 240.0),
    "piece": ("piece", 1.0), "pieces": ("piece", 1.0), "pcs": ("piece", 1.0), "x": ("piece", 1.0),
    "pinch": ("pinch", 1.0), "pinches": ("pinch", 1.0),
}
DENSITY = (  # g/ml, first match wins: what a level scoop of it weighs
    ("powdered sugar", 0.5), ("icing sugar", 0.5), ("brown sugar", 0.9), ("sugar", 0.85), ("flour", 0.53),
    ("baking powder", 0.9), ("baking soda", 0.9), ("salt", 1.2), ("cocoa", 0.42), ("oat", 0.41), ("rice", 0.85),
    ("coffee", 0.4), ("tea", 0.4), ("milk", 1.03), ("water", 1.0), ("oil", 0.92), ("butter", 0.96),
    ("honey", 1.42), ("yogurt", 1.03), ("yoghurt", 1.03), ("cream", 1.0), ("semolina", 0.65), ("polenta", 0.65),
)
DEFAULT_DENSITY = 0.6
NUMBER_WORDS = ("zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven",
                "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen", "twenty")

# runtime (design px: about 1.25 mm each on the 3 Nov table; seconds)
NEAR_SOURCE = 34.0  # a hand point this close to the source's outline is at the source
NEAR_DEST = 30.0  # ... to a destination's outline: a trip lands there (the nearest container wins)
MIN_TRIP_S = 0.7  # two counts at least this far apart
LEAVE_S = 0.5  # with no source in view, the hand must leave the destination this long between counts
WRONG_HOLD_S = 0.5  # a wrong item held this long is called out
WRONG_DEST_S = 0.35  # a hand this long at a wrong container after the source is called out
WRONG_SHOW_S = 4.0  # a called-out object stays red this long
ADVANCE_S = 1.6  # a finished step moves on after this (time to see the full count)
USE_HOLD_S = 0.6
CHECK_AGAIN_S = 8.0
HUD_CHARS = 46


# ---------------------------------------------------------------------------------------------------- plan


@dataclass(frozen=True)
class Item:
    id: str
    name: str
    kind: str = "material"  # material | container | tool | measure
    amount: float | None = None
    unit: str | None = None  # canonical: g | ml | piece | pinch
    count: int = 1  # how many of it the job needs on the table (6 cups)
    density: float | None = None  # g/ml, for grams -> scoops
    capacity_ml: float | None = None  # a measuring tool's, when known (a 10 ml syringe)
    optional: bool = False

    def amount_text(self) -> str:
        if self.amount is None or self.unit is None:
            return ""
        a = self.amount
        num = f"{a:.0f}" if abs(a - round(a)) < 0.05 or a >= 20 else f"{a:.1f}"
        if self.unit == "piece":
            return f"{num} {plural(self.name, a)}"
        if self.unit == "pinch":
            return "a pinch" if a <= 1 else f"{num} pinches"
        if self.unit == "ml" and a <= 60:  # spoons read better than millilitres for small amounts
            for size, word in ((15.0, "tbsp"), (5.0, "tsp")):
                k = a / size
                if k >= 1 and abs(k - round(k)) < 0.05:
                    return f"{round(k)} {word}"
        return f"{num} {self.unit}"


@dataclass(frozen=True)
class Step:
    do: str
    say: str
    item: str | None = None  # source / item used: an item id, or "id#k" for the k-th of a counted item
    into: str | None = None  # destination: an item id, "id#k" or a range "id#a-b" (one into each)
    count: int | None = None
    measure: str | None = None  # item id of the measuring tool that counts this transfer
    seconds: float | None = None
    question: str | None = None  # check: what the camera should judge
    target: str | None = None  # check: the item to look at (else the whole table)


@dataclass(frozen=True)
class Plan:
    title: str
    items: tuple[Item, ...]
    steps: tuple[Step, ...]
    note: str = ""
    serves: float | None = None  # a recipe's base servings (scaling)
    source: str = "built-in"  # built-in | <model id> | file

    def item(self, ref: str | None) -> Item | None:
        if not ref:
            return None
        base = ref.split("#", 1)[0]
        return next((it for it in self.items if it.id == base), None)

    def index(self, ref: str | None) -> int:
        """1-based number of the item a reference names (0: none)."""
        it = self.item(ref)
        return 0 if it is None else self.items.index(it) + 1

    @property
    def bench(self) -> BenchTask:
        """The checklist view the object namer matches against (one entry per item, in order)."""
        return BenchTask(self.title, tuple(it.name for it in self.items))


def plural(name: str, n: float) -> str:
    if abs(n - 1) < 1e-9 or name.endswith("s"):
        return name
    if name.endswith(("sh", "ch", "x")):
        return name + "es"
    return name + "s"


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")[:24] or "item"


def _unit(raw) -> tuple[str | None, float]:
    if raw is None:
        return None, 1.0
    key = str(raw).strip().lower().rstrip(".")
    if key in UNITS:
        return UNITS[key]
    if key in ("egg", "eggs", "each", "whole", "unit", "units", "item", "items", ""):
        return "piece", 1.0
    return None, 1.0


def _num(v) -> float | None:
    if isinstance(v, bool):
        return None
    if isinstance(v, (int, float)) and math.isfinite(v) and v > 0:
        return float(v)
    if isinstance(v, str):
        m = re.fullmatch(r"\s*(\d+(?:\.\d+)?)\s*(?:/\s*(\d+))?\s*", v)
        if m:
            a = float(m.group(1))
            return a / float(m.group(2)) if m.group(2) else (a if a > 0 else None)
    return None


def _ref_ok(ref: str | None, ids: set[str], counts: dict[str, int]) -> bool:
    if ref is None:
        return True
    m = re.fullmatch(r"([a-z0-9-]+)(?:#(\d+)(?:-(\d+))?)?", ref)
    if not m or m.group(1) not in ids:
        return False
    n = counts[m.group(1)]
    a = int(m.group(2)) if m.group(2) else 1
    b = int(m.group(3)) if m.group(3) else a
    return 1 <= a <= b <= n


def parse_plan(data: dict, *, source: str = "built-in") -> Plan:
    """A validated plan from JSON (a model's or a file's). Unknown fields are ignored; bad references raise
    ``ValueError`` (a model's plan that does not hold together is refused, never half-run)."""
    if not isinstance(data, dict):
        raise ValueError("a plan is a JSON object")
    title = " ".join(str(data.get("title") or "").split())[:40]
    if not title:
        raise ValueError("the plan has no title")
    raw_items = data.get("items")
    if not isinstance(raw_items, list) or not raw_items:
        raise ValueError("the plan lists no items")
    items: list[Item] = []
    ids: dict[str, str] = {}
    for raw in raw_items[:MAX_ITEMS]:
        if not isinstance(raw, dict):
            continue
        name = " ".join(str(raw.get("name") or "").lower().split())[:28]
        if not name:
            continue
        iid = _slug(str(raw.get("id") or name))
        while iid in ids:
            iid += "-2"
        kind = str(raw.get("kind") or "material").lower()
        kind = kind if kind in KINDS else "material"
        unit, factor = _unit(raw.get("unit"))
        amount = _num(raw.get("amount"))
        amount = amount * factor if amount is not None and unit is not None else None
        if amount is None:
            unit = None
        count = raw.get("count")
        count = int(count) if isinstance(count, (int, float)) and not isinstance(count, bool) and 1 <= count <= 12 else 1
        items.append(Item(id=iid, name=name, kind=kind, amount=amount, unit=unit, count=count,
                          density=_num(raw.get("density")), capacity_ml=_num(raw.get("capacity_ml")),
                          optional=bool(raw.get("optional", False))))
        ids[iid] = name
    if not items:
        raise ValueError("the plan lists no usable items")
    counts = {it.id: it.count for it in items}
    by_name = {it.name: it.id for it in items}

    def ref(v) -> str | None:  # tolerate a model naming the item instead of giving its id
        if v is None or v == "":
            return None
        s = str(v).strip().lower()
        base, sep, rest = s.partition("#")
        base = by_name.get(base, _slug(base))
        return base + (sep + rest if sep else "")

    steps: list[Step] = []
    for raw in data.get("steps") or []:
        if not isinstance(raw, dict) or len(steps) >= MAX_STEPS:
            continue
        do = str(raw.get("do") or raw.get("kind") or "").lower()
        if do not in STEP_KINDS:
            do = "confirm"
        say = " ".join(str(raw.get("say") or "").split())[:90]
        if not say:
            continue
        st = Step(do=do, say=say, item=ref(raw.get("item")), into=ref(raw.get("into")),
                  count=int(raw["count"]) if isinstance(raw.get("count"), (int, float))
                  and not isinstance(raw.get("count"), bool) and 1 <= raw["count"] <= MAX_COUNT else None,
                  measure=ref(raw.get("measure")), seconds=_num(raw.get("seconds")),
                  question=(" ".join(str(raw.get("question") or "").split())[:200] or None),
                  target=ref(raw.get("target")))
        for r in (st.item, st.into, st.measure, st.target):
            if not _ref_ok(r, set(ids), counts):
                raise ValueError(f"step {say!r} names {r!r}, which is not one of the plan's items")
        if st.do == "transfer" and st.into is None:
            st = replace(st, do="confirm")
        src = next((it for it in items if it.id == (st.item or "").split("#")[0]), None)
        if st.do == "transfer" and st.measure and src is not None and src.unit in ("g", "ml"):
            st = replace(st, count=None)  # the table counts scoops from the tool it measures
        if st.do == "timer" and not st.seconds:
            st = replace(st, do="confirm")
        if st.do == "check" and not st.question:
            st = replace(st, question=st.say)
        if st.seconds:
            st = replace(st, seconds=min(st.seconds, 3600.0))
        steps.append(st)
    if not steps:
        raise ValueError("the plan has no steps")
    serves = _num(data.get("serves"))
    note = " ".join(str(data.get("note") or "").split())[:80]
    return Plan(title=title, items=tuple(items), steps=tuple(steps), note=note, serves=serves, source=source)


def builtin_plans() -> dict[str, Path]:
    return {p.stem: p for p in sorted(PLANS_DIR.glob("*.json"))}


def load_plan(spec: str) -> Plan:
    """A plan file, or a built-in plan by name (``pancakes``, ``serial-dilution``, ...)."""
    path = Path(spec)
    if not path.is_file():
        found = builtin_plans().get(_slug(spec))
        if found is None:
            raise ValueError(f"no plan file or built-in plan {spec!r} (built-in: {', '.join(builtin_plans())})")
        path = found
    return parse_plan(json.loads(path.read_text(encoding="utf-8")), source="built-in" if path.parent == PLANS_DIR
                      else "file")


def find_builtin(request: str) -> tuple[Plan, float | None] | None:
    """A built-in plan a spoken request names ("pancakes for four"), and the servings asked for."""
    low = request.lower()
    for stem, path in builtin_plans().items():
        words = [w for w in stem.split("-") if len(w) > 3]
        if words and all(w in low or w.rstrip("s") in low for w in words[:1]):
            plan = parse_plan(json.loads(path.read_text(encoding="utf-8")))
            return plan, servings(request)
    return None


def servings(text: str) -> float | None:
    """"for four people" -> 4.0."""
    low = text.lower()
    m = re.search(r"\bfor\s+(\d+|" + "|".join(NUMBER_WORDS[1:13]) + r")\b", low)
    if not m:
        return None
    w = m.group(1)
    return float(w) if w.isdigit() else float(NUMBER_WORDS.index(w))


def scale_plan(plan: Plan, serves: float | None) -> Plan:
    """The plan's amounts for ``serves`` (a recipe with a base ``serves``); pieces round up (half an egg is an egg)."""
    if not serves or not plan.serves or abs(serves - plan.serves) < 1e-9:
        return plan
    f = serves / plan.serves
    items = []
    for it in plan.items:
        if it.amount is None or it.unit == "pinch":
            items.append(it)
            continue
        a = it.amount * f
        a = float(max(1, math.ceil(a - 0.25))) if it.unit == "piece" else round(a, 1)
        items.append(replace(it, amount=a))
    return replace(plan, items=tuple(items), serves=serves,
                   note=f"for {serves:.0f}" + (f" · {plan.note}" if plan.note else ""))


def density_of(item: Item) -> float:
    if item.density:
        return item.density
    return next((d for key, d in DENSITY if key in item.name), DEFAULT_DENSITY)


def amount_ml(item: Item) -> float | None:
    if item.amount is None:
        return None
    if item.unit == "ml":
        return item.amount
    if item.unit == "g":
        return item.amount / density_of(item)
    return None


def capacity_ml(name: str, sides_mm: tuple[float, float] | None, known: float | None = None) -> float | None:
    """What a measuring tool holds: given, from its name, or from its size (long, short sides in mm)."""
    if known:
        return known
    low = name.lower()
    m = re.search(r"(\d+(?:\.\d+)?)\s*ml\b", low)
    if m:
        return float(m.group(1))
    if "teaspoon" in low or "tsp" in low:
        return 5.0
    if "tablespoon" in low or "tbsp" in low:
        return 15.0
    for frac, ml in (("1/4 cup", 60.0), ("quarter cup", 60.0), ("1/3 cup", 80.0), ("1/2 cup", 120.0), ("half cup", 120.0)):
        if frac in low:
            return ml
    if sides_mm is None:
        return 15.0 if "spoon" in low else None
    long, short = max(sides_mm), min(sides_mm)
    if "spoon" in low:
        return 15.0 if long >= 150 else 5.0
    if any(w in low for w in ("scoop", "cup", "measur", "ladle", "jug")):
        d_cm = short / 10.0  # a round scoop's bowl: its width; the handle makes the long side
        return float(min(max(math.pi * d_cm ** 3 / 12.0, 1.0), 500.0))
    return None


def scoops_for(item: Item, cap_ml: float | None) -> tuple[int, float] | None:
    """(scoops, grams or ml per scoop) to measure ``item`` with a tool holding ``cap_ml``; None when it does not fit
    (no amount, or less than half a scoop)."""
    need = amount_ml(item)
    if need is None or not cap_ml:
        return None
    n = need / cap_ml
    if n < 0.5:
        return None
    per = cap_ml * density_of(item) if item.unit == "g" else cap_ml
    return max(1, int(round(n))), per


def short_name(name: str, limit: int = 10) -> str:
    """"measuring scoop" -> "scoop": a label on the table must not cover its neighbours."""
    return name if len(name) <= limit else name.split()[-1]


def number_word(n: int) -> str:
    return NUMBER_WORDS[n] if 0 <= n < len(NUMBER_WORDS) else str(n)


# ---------------------------------------------------------------------------------------------------- runtime


@dataclass(eq=False)
class Seen:
    """One object on the table, as the guide sees it (design px). Equal by track: views of the same object taken in
    different frames compare equal."""

    uid: int
    item: int  # 1-based plan item, 0 = not in the plan, -1 = not known yet
    name: str | None
    poly: np.ndarray  # outline, design px
    held: bool = False
    sides_mm: tuple[float, float] | None = None

    def __eq__(self, other) -> bool:
        return isinstance(other, Seen) and other.uid == self.uid

    def __hash__(self) -> int:
        return hash(self.uid)

    @property
    def centre(self) -> tuple[float, float]:
        c = self.poly.mean(axis=0)
        return float(c[0]), float(c[1])

    @property
    def box(self) -> tuple[float, float, float, float]:
        return (float(self.poly[:, 0].min()), float(self.poly[:, 1].min()), float(self.poly[:, 0].max()),
                float(self.poly[:, 1].max()))


def outside_by(poly: np.ndarray, pt) -> float:
    """Distance from ``pt`` to the outline (0 inside)."""
    d = cv2.pointPolygonTest(poly.astype(np.float32).reshape(-1, 1, 2), (float(pt[0]), float(pt[1])), True)
    return max(0.0, -float(d))


@dataclass
class _StepState:
    started: float
    count: int = 0
    target: int | None = None
    per: float | None = None  # grams/ml per scoop
    unit_word: str = ""
    src_t: float = -1e9  # last time a hand was at the source
    count_t: float = -1e9  # last count
    left_dest_t: float = -1e9  # last time no hand was at the destination
    filled: set = field(default_factory=set)  # "each" transfers: destination instance numbers done
    done_t: float | None = None  # finished: moves on at done_t + ADVANCE_S
    timer_end: float | None = None
    held_t: dict = field(default_factory=dict)  # uid -> since when a wrong item is held
    at_wrong: dict = field(default_factory=dict)  # uid -> since when a hand is at a wrong container
    warned: set = field(default_factory=set)
    over_said: int = 0
    use_t: float | None = None
    check_sent: float | None = None
    check_reply: str | None = None


class Guide:
    """Walks a :class:`Plan` on the table: what to put out, what to do next, counts, timers, checks, mistakes.

    Per frame: :meth:`update` with the objects and hand points (design px); it returns lines to speak, fills ``sfx``,
    and sets ``want_check`` when a check step needs the camera judged (the session sends it to the brain and answers
    with :meth:`check_result`). :meth:`hud` gives the two status lines, :meth:`view` an object's colour and label,
    :meth:`draw` the counter and timer."""

    def __init__(self, plan: Plan, now: float = 0.0):
        self.plan = plan
        self.phase = "gather"  # gather | steps | done
        self.k = 0
        self.st = _StepState(started=now)
        self.sfx: list[str] = []
        self.want_check: tuple[int, str | None] | None = None  # (step number, target item ref)
        self.red: dict[int, float] = {}  # uid -> red until
        self.said_ready = False
        self._seen: list[Seen] = []
        self.started = now
        self.log: list[str] = []

    # ---- what is on the table

    def instances(self, item_no: int, seen: list[Seen] | None = None) -> list[Seen]:
        """The objects that are plan item ``item_no``, numbered left to right as the person sees them."""
        objs = [s for s in (self._seen if seen is None else seen) if s.item == item_no]
        return sorted(objs, key=lambda s: (s.centre[0], s.centre[1]))

    def resolve(self, ref: str | None, seen: list[Seen] | None = None) -> list[Seen]:
        """Objects a reference names: ``bowl`` (all of them), ``cups#2``, ``cups#2-6``."""
        if not ref:
            return []
        n = self.plan.index(ref)
        if not n:
            return []
        inst = self.instances(n, seen)
        _, sep, rng = ref.partition("#")
        if not sep:
            return inst
        a, _, b = rng.partition("-")
        lo, hi = int(a), int(b or a)
        return [s for k, s in enumerate(inst, 1) if lo <= k <= hi]

    def ref_name(self, ref: str | None) -> str:
        it = self.plan.item(ref)
        if it is None:
            return "it"
        _, sep, rng = (ref or "").partition("#")
        if not sep:
            return it.name
        base = it.name.removesuffix("s")
        word = base.split()[-1] if base.split() else base
        return f"{word} {rng.replace('-', ' to ')}"

    def missing(self) -> list[str]:
        out = []
        for n, it in enumerate(self.plan.items, 1):
            have = len(self.instances(n))
            if have < it.count and not it.optional:
                out.append(it.name if it.count - have == 1 else f"{it.count - have} {plural(it.name, 2)}")
        return out

    # ---- per frame

    def update(self, now: float, seen: list[Seen], hands: list[list[tuple[float, float]]]) -> list[str]:
        self._seen = seen
        self.red = {u: t for u, t in self.red.items() if t > now}
        if self.phase == "gather":
            missing = self.missing()
            if not missing and not self.said_ready and seen:
                self.said_ready = True
                self.sfx.append("answer")
                return ["Everything is here."] + self._begin(0, now)
            return []
        if self.phase != "steps":
            return []
        step = self.plan.steps[self.k]
        st = self.st
        if st.done_t is not None:
            if now - st.done_t >= ADVANCE_S:
                return self._begin(self.k + 1, now)
            return []
        if step.do == "transfer":
            return self._transfer(step, now, seen, hands)
        if step.do == "use":
            return self._use(step, now, seen, hands)
        if step.do == "timer":
            if st.timer_end is not None and now >= st.timer_end:
                st.done_t = now
                self.sfx.append("answer")
                return ["Time."]
        return []

    def _points(self, hands) -> list[tuple[float, float]]:
        return [p for h in hands for p in h]

    def _at(self, objs: list[Seen], pts, r: float) -> Seen | None:
        """The object of ``objs`` nearest to any hand point, within ``r`` of its outline."""
        best, best_d = None, r
        for s in objs:
            for p in pts:
                d = outside_by(s.poly, p)
                if d < best_d or (d == 0 and best_d > 0):
                    best, best_d = s, d
        return best

    def _wrong_items(self, step: Step, now: float, seen: list[Seen]) -> list[str]:
        """Another material picked up during a step that uses one: called out once."""
        st = self.st
        src = self.plan.index(step.item)
        out = []
        for s in seen:
            it = self.plan.items[s.item - 1] if s.item > 0 else None
            if it is None or it.kind != "material" or s.item == src or s.uid in st.warned:
                st.held_t.pop(s.uid, None)
                continue
            if not s.held:
                st.held_t.pop(s.uid, None)
                continue
            t0 = st.held_t.setdefault(s.uid, now)
            if now - t0 >= WRONG_HOLD_S:
                st.warned.add(s.uid)
                self.red[s.uid] = now + WRONG_SHOW_S
                self.sfx.append("error")
                want = self.ref_name(step.item)
                out.append(f"Careful, that's the {it.name}. This step needs the {want}.")
                self.log.append(f"wrong item: {it.name} (step {self.k + 1} needs {want})")
        return out

    def _transfer(self, step: Step, now: float, seen: list[Seen], hands) -> list[str]:
        st = self.st
        out = self._wrong_items(step, now, seen)
        pts = self._points(hands)
        sources = self.resolve(step.item, seen)
        dests = self.resolve(step.into, seen)
        each = "#" in (step.into or "") and "-" in (step.into or "").partition("#")[2]
        containers = [s for s in seen if s.item > 0 and self.plan.items[s.item - 1].kind == "container"
                      and s not in sources]
        if pts and sources and self._at(sources, pts, NEAR_SOURCE) is not None:
            st.src_t = now
        at = self._at(containers + [d for d in dests if d not in containers], pts, NEAR_DEST) if pts else None
        if at is None or at not in dests:
            st.left_dest_t = now
        if at is not None and at not in dests:  # a hand at another container right after the source
            if sources and st.src_t > st.count_t and at.uid not in st.warned:
                t0 = st.at_wrong.setdefault(at.uid, now)
                if now - t0 >= WRONG_DEST_S:
                    st.warned.add(at.uid)
                    self.red[at.uid] = now + WRONG_SHOW_S
                    self.sfx.append("error")
                    name = self._instance_name(at)
                    out.append(f"Stop: that's {name}. This goes into {self.ref_name(step.into)}.")
                    self.log.append(f"wrong destination: {name} (step {self.k + 1})")
            return out
        st.at_wrong.clear()
        if at is None or now - st.count_t < MIN_TRIP_S:
            return out
        if sources:
            if not st.src_t > st.count_t:
                return out
        elif now - st.left_dest_t > 0.05 and st.count_t > st.left_dest_t - LEAVE_S:
            return out  # no source in view: the hand must have left the destination since the last count
        if each:
            n = self.instances(self.plan.index(step.into), seen).index(at) + 1
            if n in st.filled:
                if at.uid not in st.warned:
                    st.warned.add(at.uid)
                    self.red[at.uid] = now + WRONG_SHOW_S
                    self.sfx.append("error")
                    out.append(f"{self._instance_name(at).capitalize()} already has it.")
                st.count_t = now
                return out
            st.filled.add(n)
        st.count += 1
        st.count_t = now
        self.sfx.append("found")
        if st.target is None:
            out.append(number_word(st.count).capitalize() + ".")
            return out
        if st.count < st.target:
            out.append(number_word(st.count).capitalize() + ".")
        elif st.count == st.target:
            st.done_t = now
            self.sfx.append("answer")
            out.append(f"{number_word(st.count).capitalize()}. That's all {st.target}.")
        else:
            extra = st.count - st.target
            self.sfx.append("error")
            for d in dests:
                self.red[d.uid] = now + WRONG_SHOW_S
            out.append(f"That's {st.count}: {number_word(extra)} too many. Take {'it' if extra == 1 else 'them'} out.")
            self.log.append(f"over count: {st.count}/{st.target} (step {self.k + 1})")
        return out

    def _use(self, step: Step, now: float, seen: list[Seen], hands) -> list[str]:
        st = self.st
        out = self._wrong_items(step, now, seen)
        mine = self.resolve(step.item, seen)
        if any(s.held for s in mine):
            st.use_t = st.use_t or now
        if st.use_t is not None and now - st.use_t >= USE_HOLD_S:
            st.done_t = now
            self.sfx.append("found")
        return out

    def _instance_name(self, s: Seen) -> str:
        if s.item <= 0:
            return f"the {s.name or 'object'}"
        it = self.plan.items[s.item - 1]
        if it.count > 1:
            k = self.instances(s.item).index(s) + 1
            base = it.name.removesuffix("s")
            return f"{base.split()[-1]} {k}"
        return f"the {it.name}"

    # ---- steps

    def start(self, now: float) -> list[str]:
        """What to say when the plan is first shown."""
        self.started = now
        names = [it.amount_text() + " " + it.name if it.amount_text() and it.unit != "piece" else
                 (it.amount_text() or (f"{it.count} {plural(it.name, it.count)}" if it.count > 1 else it.name))
                 for it in self.plan.items if not it.optional]
        self.sfx.append("answer")
        note = self.plan.note.rstrip(". ")
        title = self.plan.title.rstrip(". ")
        if note and note.lower() in title.lower():
            note = ""
        intro = f"{title}{', ' + note[0].lower() + note[1:] if note else ''}."
        return [f"{intro} You need {spoken_list([n.strip() for n in names])}. Put them on the table."]

    def _begin(self, k: int, now: float) -> list[str]:
        self.want_check = None
        if k >= len(self.plan.steps):
            self.phase, self.k = "done", len(self.plan.steps)
            self.sfx.append("answer")
            return [f"{self.plan.title}: done."]
        self.phase, self.k = "steps", k
        step = self.plan.steps[k]
        self.st = st = _StepState(started=now)
        extra = ""
        if step.do == "transfer":
            st.target, st.per, st.unit_word, extra = self._target(step)
        elif step.do == "timer":
            st.timer_end = now + float(step.seconds or 0)
            extra = f" {fmt_time(step.seconds or 0, spoken=True)}."
        elif step.do == "check":
            self.want_check = (k, step.target)
            st.check_sent = now
        elif step.do == "confirm":
            extra = " Say done when it's done."
        return [f"Step {k + 1}. {step.say}.{extra}".replace("..", ".")]

    def _target(self, step: Step) -> tuple[int | None, float | None, str, str]:
        """(count, per scoop, unit word, what to add to the spoken step) for a transfer."""
        src = self.plan.item(step.item)
        if "#" in (step.into or "") and "-" in (step.into or "").partition("#")[2]:
            n = len(self.resolve(step.into)) or _range_len(step.into)
            return n, None, "", f" One into each, {n} in all."
        if step.count:
            word = "scoop" if step.measure else (plural(src.name, 2) if src and src.unit == "piece" else "time")
            return step.count, None, word, ""
        if step.measure and src is not None:
            tool = self.plan.item(step.measure)
            inst = self.resolve(step.measure)
            sides = inst[0].sides_mm if inst else None
            cap = capacity_ml(inst[0].name or tool.name if inst else tool.name, sides, tool.capacity_ml) if tool else None
            got = scoops_for(src, cap)
            if got is not None:
                n, per = got
                unit = "g" if src.unit == "g" else "ml"
                word = (tool.name.split()[-1] if tool else "scoop")
                return n, per, word, (f" That's {n} {plural(word, n)} of your {tool.name if tool else 'scoop'},"
                                      f" about {per:.0f} {'grams' if unit == 'g' else 'millilitres'} each.")
        if src is not None and src.unit == "piece" and src.amount:
            n = int(round(src.amount))
            return n, None, plural(src.name, n), ""
        if src is not None and src.amount_text():
            return None, None, "", f" {src.amount_text().capitalize()}. Say done when it's in."
        return None, None, "", " Say done when it's done."

    def command(self, verb: str, now: float) -> list[str]:
        """A spoken command: next | back | repeat | start | stop | undo | check | how many."""
        if verb == "stop":
            self.phase = "done"
            return [f"Stopped {self.plan.title}."]
        if verb == "start" or (verb == "next" and self.phase == "gather"):
            missing = self.missing()
            pre = [f"Still missing {spoken_list(missing)}, starting anyway."] if missing else []
            return pre + self._begin(0, now)
        if self.phase != "steps":
            if verb == "repeat":
                return [f"Put these on the table: {spoken_list(self.missing())}."] if self.phase == "gather" else []
            return []
        if verb == "next":
            return self._begin(self.k + 1, now)
        if verb == "back":
            return self._begin(max(0, self.k - 1), now)
        if verb == "repeat":
            return self._begin(self.k, now)
        if verb == "undo" and self.st.count > 0:
            self.st.count -= 1
            self.st.done_t = None
            return [f"Back to {self.st.count}."]
        if verb == "how many":
            t = f" of {self.st.target}" if self.st.target else ""
            return [f"{self.st.count}{t}."]
        if verb == "check" and self.plan.steps[self.k].do == "check":
            self.want_check = (self.k, self.plan.steps[self.k].target)
            self.st.check_sent = now
            return ["Looking."]
        return []

    def check_result(self, k: int, ok: bool | None, say: str, now: float) -> list[str]:
        """The brain's verdict for check step ``k``."""
        if self.phase != "steps" or k != self.k:
            return []
        self.st.check_reply = say
        if ok:
            self.st.done_t = now
            self.sfx.append("answer")
            return [say or "Looks right."]
        self.sfx.append("error")
        for s in self.resolve(self.plan.steps[k].target):
            self.red[s.uid] = now + WRONG_SHOW_S
        return [(say or "Not yet.") + " Say check when you want me to look again."]

    # ---- display

    def hud(self, now: float) -> tuple[str, str, str]:
        """(first line, second line, role) for the status area."""
        p = self.plan
        if self.phase == "gather":
            missing = self.missing()
            line2 = ("NEED: " + ", ".join(m.upper() for m in missing)) if missing else "ALL HERE: SAY START"
            return _fit(f"{p.title.upper()}: PUT THESE ON THE TABLE"), _fit(line2), "info"
        if self.phase == "done":
            return _fit(f"{p.title.upper()}: DONE"), "SAY WHAT YOU WANT TO DO NEXT", "done"
        step = p.steps[self.k]
        st = self.st
        line1 = _fit(f"{self.k + 1}/{len(p.steps)} {step.say.upper()}")
        role = "done" if st.done_t is not None else "info"
        if step.do == "transfer":
            if st.target:
                line2 = f"{st.count} OF {st.target} {plural(st.unit_word, st.target).upper()}".strip()
                if st.per:
                    src = p.item(step.item)
                    line2 += f" · ~{st.per:.0f} {'G' if src and src.unit == 'g' else 'ML'} EACH"
                if st.count > st.target:
                    role = "error"
            else:
                src = p.item(step.item)
                amt = src.amount_text().upper() if src else ""
                line2 = (amt + " · " if amt else "") + "SAY DONE WHEN IT'S IN"
        elif step.do == "timer":
            left = max(0.0, (st.timer_end or now) - now)
            line2 = fmt_time(left)
        elif step.do == "check":
            line2 = (st.check_reply or "LOOKING...").upper()
        elif step.do == "use":
            line2 = f"PICK UP THE {self.ref_name(step.item).upper()}"
        else:
            line2 = "SAY DONE WHEN IT'S DONE"
        return line1, _fit(line2), role

    def view(self, s: Seen, now: float) -> tuple[str, list[str]] | None:
        """(role, label lines) for an object, or None to leave it as it is."""
        if s.uid in self.red:
            return "error", [(s.name or "?").upper(), "NOT THIS ONE"]
        if s.item <= 0:
            return None
        it = self.plan.items[s.item - 1]
        name = short_name(it.name).upper() if it.count == 1 else self._instance_name(s).upper()
        if self.phase == "gather":
            amt = it.amount_text().upper()
            if it.kind == "measure":
                cap = capacity_ml(s.name or it.name, s.sides_mm, it.capacity_ml)
                amt = f"~{cap:.0f} ML" if cap else ""
            return "done", [name, amt] if amt else [name]
        if self.phase != "steps":
            return "info", [name]
        step = self.plan.steps[self.k]
        if s in self.resolve(step.item):
            return "source", [name, "USE THIS"]
        if step.do == "transfer" and s in self.resolve(step.into):
            each = "-" in (step.into or "").partition("#")[2]
            if each:
                n = self.instances(s.item).index(s) + 1
                return ("done", [name, "DONE"]) if n in self.st.filled else ("target", [name, "INTO THIS"])
            return "target", [name, "INTO THIS"]
        if step.do == "check" and s in self.resolve(step.target):
            return "target", [name, "CHECKING"]
        if it.kind == "measure":
            sides = s.sides_mm
            cap = capacity_ml(s.name or it.name, sides, it.capacity_ml)
            return "info", [name, f"~{cap:.0f} ML" if cap else ""]
        return "info", [name]

    def draw(self, c, now: float) -> None:
        """The counter beside the destination and the timer ring (design px, on a playground ``Canvas``)."""
        if self.phase != "steps":
            return
        step = self.plan.steps[self.k]
        st = self.st
        if step.do == "transfer":
            dests = self.resolve(step.into)
            if not dests or "-" in (step.into or "").partition("#")[2]:
                return
            d = dests[0]
            x0, y0, x1, y1 = d.box
            cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
            r = 0.5 * math.hypot(x1 - x0, y1 - y0) + 10
            over = st.target is not None and st.count > st.target
            role = "error" if over else ("done" if st.done_t is not None else "target")
            if st.target:
                c.arc("guide-arc", (cx, cy), r, min(st.count / st.target, 1.0), role)
            text = str(st.count)
            w = 19.0  # digit width, design px (twice as tall): readable across the table and on video
            digits = w * 1.45 * len(text)
            of = f"/{st.target}" if st.target else ""
            room = digits + (16.0 * len(of) + 8 if of else 0)
            tx, ty = self._counter_spot(d, (cx, cy), r, room, 2 * w)
            c.number("guide-count", text, tx, ty, w, role)
            if of:
                c.label("guide-count-of", (tx + digits + 4, ty + w), of, role)
        elif step.do == "timer":
            left = max(0.0, (st.timer_end or now) - now)
            total = float(step.seconds or 1)
            secs = str(int(math.ceil(left)))
            items = self.resolve(step.item)
            if items:
                x0, y0, x1, y1 = items[0].box
                cx, cy, r = (x0 + x1) / 2, (y0 + y1) / 2, 0.5 * math.hypot(x1 - x0, y1 - y0) + 12
            else:
                cx, cy, r = 420.0, 230.0, 60.0
            c.arc("guide-timer", (cx, cy), r, left / total, "done" if left <= 0 else "target")
            w = 17.0
            tx, ty = self._counter_spot(items[0] if items else None, (cx, cy), r, w * 1.45 * len(secs), 2 * w)
            c.number("guide-timer-s", secs, tx, ty, w, "done" if left <= 0 else "target")

    def _counter_spot(self, around: Seen | None, c, r: float, w: float, h: float) -> tuple[float, float]:
        """Top-left of a w x h counter beside ``around`` (outside its ring of radius r): right, left, below or above,
        the first that covers no other object and stays on the table part of the page."""
        cx, cy = c
        boxes = [o.box for o in self._seen if around is None or o.uid != around.uid]
        options = [(cx + r + 10, cy - h / 2), (cx - r - 10 - w, cy - h / 2), (cx - w / 2, cy + r + 8),
                   (cx - w / 2, cy - r - 8 - h)]
        best, best_cost = options[0], math.inf
        for x, y in options:
            x = min(max(x, 72.0), 744.0 - w)
            y = min(max(y, 68.0), 416.0 - h)
            cost = sum(max(0.0, min(x + w, b[2] + 6) - max(x, b[0] - 6)) * max(0.0, min(y + h, b[3] + 6) - max(y, b[1] - 6))
                       for b in boxes)
            if cost < best_cost:
                best, best_cost = (x, y), cost
            if cost == 0:
                break
        return best


def _range_len(ref: str | None) -> int:
    a, _, b = (ref or "").partition("#")[2].partition("-")
    try:
        return int(b) - int(a) + 1
    except ValueError:
        return 0


def fmt_time(seconds: float, spoken: bool = False) -> str:
    s = int(math.ceil(max(0.0, seconds)))
    if spoken:
        m, r = divmod(s, 60)
        parts = ([f"{m} minute{'s' if m != 1 else ''}"] if m else []) + ([f"{r} seconds"] if r or not m else [])
        return " ".join(parts)
    return f"{s // 60}:{s % 60:02d}"


def _fit(text: str, n: int = HUD_CHARS) -> str:
    text = " ".join(text.split())
    return text if len(text) <= n else text[:n - 3].rstrip(" ,·") + "..."


# ---------------------------------------------------------------------------------------------------- requests


PLAN_SCHEMA = {
    "type": "object",
    "properties": {
        "title": {"type": "string"},
        "note": {"type": "string"},
        "items": {"type": "array", "items": {
            "type": "object",
            "properties": {
                "id": {"type": "string"}, "name": {"type": "string"},
                "kind": {"type": "string", "enum": list(KINDS)},
                "amount": {"type": ["number", "null"]}, "unit": {"type": ["string", "null"]},
                "count": {"type": "integer"}, "density": {"type": ["number", "null"]},
                "capacity_ml": {"type": ["number", "null"]}, "optional": {"type": "boolean"},
            },
            "required": ["id", "name", "kind", "amount", "unit", "count", "density", "capacity_ml", "optional"],
            "additionalProperties": False,
        }},
        "steps": {"type": "array", "items": {
            "type": "object",
            "properties": {
                "do": {"type": "string", "enum": list(STEP_KINDS)}, "say": {"type": "string"},
                "item": {"type": ["string", "null"]}, "into": {"type": ["string", "null"]},
                "count": {"type": ["integer", "null"]}, "measure": {"type": ["string", "null"]},
                "seconds": {"type": ["number", "null"]}, "question": {"type": ["string", "null"]},
                "target": {"type": ["string", "null"]},
            },
            "required": ["do", "say", "item", "into", "count", "measure", "seconds", "question", "target"],
            "additionalProperties": False,
        }},
    },
    "required": ["title", "note", "items", "steps"],
    "additionalProperties": False,
}

PLAN_PROMPT = """You are Lightwork, a helper projected onto a work table by a small projector, with a camera looking down.
The person at the table said: "{request}"
{context}
Write the job as a short plan the table can walk them through. Rules:
- items: everything the job needs, including things not on the table yet (at most 10). kind "material" (gets used
  up: flour, milk, dye, screws), "container" (things go into it: bowl, cup, box, bag), "tool" (whisk, screwdriver,
  camera) or "measure" (measuring scoop, spoon, syringe, pipette). Amounts with a unit (g, ml, piece, pinch, tsp,
  tbsp, cup) for what they asked for; a density in g/ml for powders. count > 1 only for several identical
  containers (6 cups). Names: short and plain, what a camera sees ("milk", "egg", "camera").
- steps (at most 8), each one of:
  "transfer": put an item into another item. For powders and liquids give the material's amount in g or ml and set
     "measure" to the measuring tool's id, and leave "count" null: the table works out the number of scoops itself.
     Set "count" only for whole pieces (2 eggs, 4 screws). Numbered containers: "cups#2", or "cups#2-6" for one into
     each.
  "use": pick up a tool.  "timer": "seconds" to wait or mix.  "confirm": something a camera cannot see.
  "check": a yes/no "question" a camera looking down can answer about the "target" item (null: the whole table).
- "say": a short natural instruction, 3 to 7 words, like "Add the flour" or "Crack the eggs into the bowl". No
  amounts in it: the table shows them. Refer to items by their id in item/into/measure/target.
- "note": at most 5 words ("for four people").
- Safe for a home or lab bench: no heat, knives or chemicals beyond food colouring unless they ask.
Answer with JSON only."""


def plan_request_context(seen: list[tuple[str, tuple[float, float] | None]]) -> str:
    if not seen:
        return "The table is empty right now."
    parts = [f"{name} ({a:.0f} x {b:.0f} mm)" if size else name for name, size in seen
             for a, b in [size or (0, 0)]]
    return "Objects on the table now: " + "; ".join(parts) + "."


CHECK_PROMPT = """You are Lightwork, a helper projected onto a work table, with a camera looking down.
The job: {title}. The current step: "{step}".
Look at the image(s) and answer the question: {question}
Reply with JSON: {{"ok": true or false, "say": "one short sentence, at most 14 words, to say out loud"}}.
Say ok only if the image clearly shows it. If you cannot tell from the image, ok is false and say what you'd need."""

CHECK_SCHEMA = {
    "type": "object",
    "properties": {"ok": {"type": "boolean"}, "say": {"type": "string"}},
    "required": ["ok", "say"],
    "additionalProperties": False,
}


def plan_dict(plan: Plan) -> dict:
    """A plan as JSON (what ``parse_plan`` reads back): for ``lightwork plan --save``."""
    def clean(d: dict) -> dict:
        return {k: v for k, v in d.items() if v not in (None, False, "") and not (k == "count" and v == 1)}

    items = [clean({"id": it.id, "name": it.name, "kind": it.kind, "amount": it.amount, "unit": it.unit,
                    "count": it.count, "density": it.density, "capacity_ml": it.capacity_ml, "optional": it.optional})
             for it in plan.items]
    steps = [clean({"do": st.do, "say": st.say, "item": st.item, "into": st.into, "count": st.count,
                    "measure": st.measure, "seconds": st.seconds, "question": st.question, "target": st.target})
             for st in plan.steps]
    return clean({"title": plan.title, "serves": plan.serves, "note": plan.note, "source": plan.source,
                  "items": items, "steps": steps})
