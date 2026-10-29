"""C3 Step graph (``procedures/<slug>/stepgraph.json``, ``schema_version: 1``) and its semantic validator.

References:
- ``check.object`` / ``targets[]`` / ``arrow.from|to`` / ``check.region`` name an object id, or a grid cell as
  ``<grid_object>:<cell>`` (e.g. ``plate:A6``).
- ``check.cells`` are cell names of the grid named by ``check.object`` (e.g. ``["A1", "A2"]``).
- ``recovery.on`` is the id of a failed check in the same step, ``wrong_cell:<cell>`` (the reducer saw the target
  cell unchanged and ``<cell>`` of the same grid changed instead), ``timeout``, or ``*`` (any failure). All entries
  whose ``on`` matches are applied in list order; their ``say`` lines are spoken in that order.

Meta-reading rule (PLAN.md §4.3 invariants): a step that can reach ``passed`` needs at least one Meta-reading check
(SAM-based: present/absent/moved_to/count, or ``judge``). Steps whose checks are all ``confirm`` are allowed; they end
``unverified`` or ``skipped`` and never ``passed``.
"""

from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

CheckKind = Literal[
    "present", "absent", "moved_to", "count", "colour_change", "series", "activity_at", "judge", "confirm"
]
SAM_CHECKS = frozenset({"present", "absent", "moved_to", "count"})
LOCAL_CHECKS = frozenset({"colour_change", "series", "activity_at"})
META_CHECKS = SAM_CHECKS | {"judge"}
RecoveryAction = Literal["exclude_cells", "skip_steps", "redo_step", "insert_step", "abort"]

ID_RE = re.compile(r"^[a-z0-9][a-z0-9_\-]*$")


class _M(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class Source(_M):
    kind: Literal["text", "url", "pdf"]
    ref: str
    licence: str
    attribution: str


class Grid(_M):
    rows: int = Field(ge=1, le=32)
    cols: int = Field(ge=1, le=48)
    naming: Literal["A1", "r1c1"]
    channel: Literal["r", "g", "b"] = "r"  # OD channel: complementary to the dye (blue dye -> r, tea/coffee -> b)


class Object(_M):
    id: str
    noun: str  # SAM concept, <= 3 words
    kind: Literal["item", "container", "tool", "grid", "zone"]
    grid: Grid | None = None


class Arrow(_M):
    from_: str = Field(alias="from")
    to: str

    model_config = ConfigDict(extra="forbid", frozen=True, populate_by_name=True, serialize_by_alias=True)


class Check(_M):
    id: str
    kind: CheckKind
    object: str | None = None
    region: str | None = None
    cells: list[str] | None = None
    n: int | None = None
    direction: Literal["darker", "lighter", "hue", "none"] | None = None
    min_delta: float | None = None
    ratio: float | None = None
    tol: float | None = None
    question: str | None = None
    required: bool = True


class Recovery(_M):
    on: str
    action: RecoveryAction
    cells: list[str] | None = None
    steps: list[str] | None = None
    step: Step | None = None  # insert_step: the inline step
    say: str = ""


class Step(_M):
    id: str
    say: str
    detail: str = ""
    targets: list[str] = []
    arrow: Arrow | None = None
    checks: list[Check]
    recovery: list[Recovery] = []


Recovery.model_rebuild()


class StepGraph(_M):
    schema_version: Literal[1] = 1
    id: str
    title: str
    source: Source
    objects: list[Object]
    steps: list[Step]
    vocabulary: list[str] = []  # extra voice keywords (<= 20 nouns)


# ---------------------------------------------------------------- semantics


def cell_names(grid: Grid) -> list[str]:
    if grid.naming == "A1":
        if grid.rows > 26:
            raise ValueError("A1 naming supports at most 26 rows")
        return [f"{chr(65 + r)}{c + 1}" for r in range(grid.rows) for c in range(grid.cols)]
    return [f"r{r + 1}c{c + 1}" for r in range(grid.rows) for c in range(grid.cols)]


def _all_steps(steps: list[Step]) -> list[Step]:
    out = []
    for s in steps:
        out.append(s)
        for r in s.recovery:
            if r.step is not None:
                out.extend(_all_steps([r.step]))
    return out


def validate_semantics(g: StepGraph) -> list[str]:
    """Return human-readable errors (empty = valid)."""
    errs: list[str] = []
    objs = {o.id: o for o in g.objects}
    if len(objs) != len(g.objects):
        errs.append("duplicate object ids")
    cells: dict[str, set[str]] = {}
    for o in g.objects:
        if not ID_RE.match(o.id):
            errs.append(f"object {o.id!r}: id must match {ID_RE.pattern}")
        if len(o.noun.split()) > 3 or not o.noun.strip():
            errs.append(f"object {o.id}: noun {o.noun!r} must be 1-3 words")
        if (o.kind == "grid") != (o.grid is not None):
            errs.append(f"object {o.id}: grid spec required exactly when kind == 'grid'")
        if o.grid is not None:
            try:
                cells[o.id] = set(cell_names(o.grid))
            except ValueError as e:
                errs.append(f"object {o.id}: {e}")

    def ref_ok(ref: str) -> bool:
        if ":" in ref:
            obj, cell = ref.split(":", 1)
            return obj in cells and cell in cells[obj]
        return ref in objs

    steps = _all_steps(g.steps)
    step_ids = [s.id for s in steps]
    if len(set(step_ids)) != len(step_ids):
        errs.append("duplicate step ids")
    all_ids = set(step_ids)

    for s in steps:
        where = f"step {s.id}"
        if not ID_RE.match(s.id):
            errs.append(f"{where}: id must match {ID_RE.pattern}")
        if len(s.say.split()) > 30:
            errs.append(f"{where}: say has more than 30 words")
        if not s.checks:
            errs.append(f"{where}: no checks")
        for t in s.targets:
            if not ref_ok(t):
                errs.append(f"{where}: target {t!r} does not resolve")
        if s.arrow is not None:
            for end in (s.arrow.from_, s.arrow.to):
                if not ref_ok(end):
                    errs.append(f"{where}: arrow end {end!r} does not resolve")
        kinds = {c.kind for c in s.checks}
        if kinds != {"confirm"} and not (kinds & META_CHECKS):
            errs.append(f"{where}: no Meta-reading check (SAM check or judge)")
        check_ids = [c.id for c in s.checks]
        if len(set(check_ids)) != len(check_ids):
            errs.append(f"{where}: duplicate check ids")
        for c in s.checks:
            errs.extend(f"{where} check {c.id}: {e}" for e in _check_errors(c, objs, cells, ref_ok))
        for r in s.recovery:
            if not (r.on in ("*", "timeout") or r.on in check_ids or r.on.startswith("wrong_cell:")):
                errs.append(f"{where}: recovery on {r.on!r} is not a check id, wrong_cell:<cell>, timeout or *")
            if r.action == "exclude_cells" and not r.cells:
                errs.append(f"{where}: exclude_cells needs cells")
            if r.action == "skip_steps":
                for sid in r.steps or []:
                    if sid not in all_ids:
                        errs.append(f"{where}: skip_steps names unknown step {sid!r}")
                if not r.steps:
                    errs.append(f"{where}: skip_steps needs steps")
            if r.action == "insert_step" and r.step is None:
                errs.append(f"{where}: insert_step needs step")
            for cell in r.cells or []:
                if not ref_ok(cell):
                    errs.append(f"{where}: recovery cell {cell!r} must be <grid>:<cell>")
    if len(g.vocabulary) > 20:
        errs.append("vocabulary has more than 20 entries")
    return errs


def _check_errors(c: Check, objs, cells, ref_ok) -> list[str]:
    e: list[str] = []
    k = c.kind
    if c.region is not None and not ref_ok(c.region):
        e.append(f"region {c.region!r} does not resolve")
    if k in ("present", "absent", "moved_to", "count"):
        if c.object not in objs:
            e.append("needs an existing object")
        if k in ("moved_to", "count") and c.region is None:
            e.append("needs region")
        if k == "count" and (c.n is None or c.n < 0):
            e.append("needs n >= 0")
    elif k in ("colour_change", "series"):
        if c.object not in cells:
            e.append("needs a grid object")
        if not c.cells:
            e.append("needs cells")
        else:
            bad = [x for x in c.cells if x not in cells.get(c.object or "", set())]
            if bad:
                e.append(f"unknown cells {bad}")
        if k == "colour_change" and c.direction is None:
            e.append("needs direction")
        if k == "series" and (c.ratio is None or c.tol is None):
            e.append("needs ratio and tol")
    elif k == "activity_at":
        if c.region is None:
            e.append("needs region")
    elif k == "judge":
        if not c.question:
            e.append("needs question")
    return e
