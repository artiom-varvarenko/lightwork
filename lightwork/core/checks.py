"""Check evaluators for the C3 vocabulary (PLAN.md §6.1). Pure functions; the reducer adds evidence refs.

Statuses: ``pass``; ``pending`` (not yet, may still pass); ``fail`` (definitive); ``inconclusive`` (the reading
cannot decide: unreadable cell, colour below the camera's floor, model error/timeout, verdict ``unsure``).

Colour semantics, per cell against the step's pre-step reference keyframe:

- ``darker`` / ``lighter``: OD change >= need; ``none``: |OD change| < need, a change is a definitive ``fail``;
  ``hue``: CIEDE2000 >= need. ``need = min_delta`` (defaults in :class:`~lightwork.core.params.Params`).
- Transfer steps (arrow between two cells of the check's grid): need = ``max(min_delta, 0.25 x OD(source) before)``
  and the colour is unreadable (``inconclusive``) when OD(source) before is below ``transfer_min_source_od``.
  ``ratio``/``tol`` on the target's ``darker`` check add the equal-OD band: OD(target)/OD(source) after must lie in
  ``ratio x (1 +- tol)`` when both are above the series floor, else a definitive ``fail``.
- Wrong cell: the target did not darken but a neighbour within ``wrong_cell_reach`` did, by the same need.

SAM semantics (masks in camera px): ``present`` = some mask has >= ``present_min_overlap`` of its area in the
region (anywhere without a region); ``absent`` = no such mask; ``count`` = number of such masks == n, with Spark's
count of grounded points in the region as a cross-check (a disagreement is ``inconclusive``); ``moved_to`` = a mask
centroid is outside the region before the step and inside afterwards. A missing before mask is inconclusive;
an object already inside does not prove movement.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from ..contracts.stepgraph import Check, Grid, Step
from ..geometry import Regions, cell_index
from ..measure.colour import delta_e, series_ok
from ..measure.colour import wrong_cell as _wrong_cell
from ..measure.grid import neighbours
from .graph import region_hit, transfer_of
from .masks import CamMask, overlap
from .params import Params
from .readings import CellRead


@dataclass(frozen=True)
class Mark:
    """A failure highlight for the scene: a region ref, or a circle in camera px (where SAM saw the object)."""

    role: str  # error | warn
    ref: str | None = None
    circle: tuple[float, float, float] | None = None


@dataclass(frozen=True)
class Outcome:
    status: str
    reason: str
    evidence: tuple[str, ...] = ()
    wrong: str | None = None  # wrong-cell recovery key: the cell that changed instead of the target
    marks: tuple[Mark, ...] = ()


def fmt_cells(cells: Sequence[str], naming: str) -> str:
    """``A1, A2, A3, A5`` -> ``"A1-A3, A5"`` (runs of consecutive cells in one row)."""
    runs: list[list[str]] = []
    prev: tuple[int, int] | None = None
    for c in cells:
        rc = cell_index(c, naming)
        if prev is not None and rc == (prev[0], prev[1] + 1):
            runs[-1].append(c)
        else:
            runs.append([c])
        prev = rc
    return ", ".join(r[0] if len(r) == 1 else f"{r[0]}-{r[-1]}" for r in runs)


# ---------------------------------------------------------------- colour


def colour_need(c: Check, step: Step, before: Mapping[str, CellRead], p: Params) -> tuple[float | None, str]:
    """Threshold for a colour_change check, or ``(None, reason)`` when the camera cannot read it."""
    base = c.min_delta if c.min_delta is not None else (p.hue_min_delta if c.direction == "hue" else p.od_min_delta)
    t = transfer_of(step)
    if t is None or t[0] != c.object or c.direction not in ("darker", "lighter") or not set(c.cells) <= {t[1], t[2]}:
        return base, ""
    src = before.get(f"{t[0]}:{t[1]}")
    if src is None:
        return None, f"{t[1]} unreadable before the step"
    if src.od < p.transfer_min_source_od:
        return None, f"{t[1]} OD {src.od:.4f} is below the readable floor {p.transfer_min_source_od:g}"
    return max(base, p.transfer_drop_frac * src.od), ""


def colour_change(c: Check, step: Step, now: Mapping[str, CellRead], before: Mapping[str, CellRead],
                  p: Params) -> Outcome:
    g = c.object
    need, why = colour_need(c, step, before, p)
    if need is None:
        return Outcome("inconclusive", why)
    parts = []
    for cell in c.cells:
        a, b = before.get(f"{g}:{cell}"), now.get(f"{g}:{cell}")
        if a is None or b is None:
            return Outcome("inconclusive", f"{cell} unreadable")
        if c.direction == "hue":
            de = delta_e(a.lab, b.lab)
            if de < need:
                return Outcome("pending", f"{cell} colour unchanged (delta E {de:.2f}, need {need:.2f})")
            parts.append(f"{cell} delta E {de:.2f}")
            continue
        if c.direction == "lighter" and a.od < need:
            return Outcome("inconclusive", f"{cell} OD {a.od:.4f} cannot resolve a drop of {need:.4f}")
        d = b.od - a.od
        if c.direction == "darker" and d < need:
            return Outcome("pending", f"{cell} did not darken (OD change {d:+.4f}, need +{need:.4f})")
        if c.direction == "lighter" and -d < need:
            return Outcome("pending", f"{cell} did not get lighter (OD change {d:+.4f}, need -{need:.4f})")
        if c.direction == "none" and abs(d) >= need:
            return Outcome("fail", f"{cell} changed (OD change {d:+.4f}, expected none)",
                           marks=(Mark("error", ref=f"{g}:{cell}"),))
        parts.append(f"{cell} OD {a.od:.4f} -> {b.od:.4f}")
    if c.ratio is not None:
        _, s_cell, t_cell = transfer_of(step)
        s, t = now.get(f"{g}:{s_cell}"), now.get(f"{g}:{t_cell}")
        if s is None:
            return Outcome("inconclusive", f"{s_cell} unreadable")
        if s.od > p.series_floor and t.od > p.series_floor:
            tol = c.tol if c.tol is not None else p.ratio_tol
            lo, hi = c.ratio * (1 - tol), c.ratio * (1 + tol)
            r = t.od / s.od
            if not lo <= r <= hi:
                return Outcome("fail", f"{t_cell}/{s_cell} OD ratio {r:.2f} outside {lo:.2f}-{hi:.2f}",
                               marks=(Mark("warn", ref=f"{g}:{s_cell}"), Mark("warn", ref=f"{g}:{t_cell}")))
            parts.append(f"ratio {r:.2f}")
    return Outcome("pass", ", ".join(parts))


def wrong_cell(c: Check, step: Step, grid: Grid, now: Mapping[str, CellRead], before: Mapping[str, CellRead],
               need: float, p: Params) -> Outcome | None:
    """Definitive failure of a pending ``darker`` check when liquid went into a neighbour instead."""
    g = c.object
    own = {x for k in step.checks if k.kind == "colour_change" and k.object == g for x in k.cells}
    for target in c.cells:
        cands = [target, *neighbours(target, grid.rows, grid.cols, grid.naming, p.wrong_cell_reach)]
        deltas = {}
        for cell in cands:
            a, b = before.get(f"{g}:{cell}"), now.get(f"{g}:{cell}")
            if a is not None and b is not None:
                deltas[cell] = b.od - a.od
        if target not in deltas:
            continue
        hit = _wrong_cell(target, deltas, exclude=sorted(own - {target}), min_delta=need)
        if hit is not None:
            return Outcome("fail", f"liquid went into {hit}, expected {target}", wrong=hit,
                           marks=(Mark("error", ref=f"{g}:{hit}"), Mark("warn", ref=f"{g}:{target}")))
    return None


def series(c: Check, grid: Grid, now: Mapping[str, CellRead], excluded: set[str], p: Params) -> Outcome:
    g = c.object
    cells = [x for x in c.cells if f"{g}:{x}" not in excluded]
    dropped = [x for x in c.cells if f"{g}:{x}" in excluded]
    missing = [x for x in cells if f"{g}:{x}" not in now]
    if missing:
        return Outcome("inconclusive", f"{fmt_cells(missing, grid.naming)} unreadable")
    ok, why = series_ok([now[f"{g}:{x}"].od for x in cells], c.ratio, c.tol if c.tol is not None else p.series_tol,
                        floor=p.series_floor, names=cells)
    label = f"series over {fmt_cells(cells, grid.naming)}"
    if dropped:
        label += f" ({fmt_cells(dropped, grid.naming)} excluded)"
    if ok:
        return Outcome("pass", f"{label}: {why}")
    return Outcome("fail", f"{label}: {why}", marks=(Mark("warn", ref=g),))


# ---------------------------------------------------------------- activity


def activity_at(c: Check, motion: set[str], tips: set[str], hands_tracked: bool) -> Outcome:
    """Motion and fingertip reached the same region; motion-only fallback when hand tracking is unavailable."""
    ref = c.region
    if not region_hit(ref, motion):
        suffix = " (suspect: motion elsewhere)" if motion else ""
        return Outcome("pending", f"no motion at {ref} yet{suffix}")
    if hands_tracked:
        if not region_hit(ref, tips):
            return Outcome("pending", f"motion at {ref}, no fingertip there yet")
        return Outcome("pass", f"motion and fingertip at {ref}")
    return Outcome("pass", f"motion at {ref} (motion-only mode; no hand tracking)")


# ---------------------------------------------------------------- SAM


def _inside(masks: Sequence[CamMask], regions: Regions, ref: str | None, p: Params) -> list[CamMask]:
    if ref is None:
        return list(masks)
    return [m for m in masks if overlap(m, regions, ref) >= p.present_min_overlap]


def _rings(masks: Sequence[CamMask], role: str, p: Params) -> tuple[Mark, ...]:
    return tuple(Mark(role, circle=(m.centroid[0], m.centroid[1], m.radius + p.mark_pad_px)) for m in masks[:3])


def present(c: Check, noun: str, masks: Sequence[CamMask], regions: Regions, p: Params) -> Outcome:
    where = f" in {c.region}" if c.region else ""
    if _inside(masks, regions, c.region, p):
        return Outcome("pass", f"{noun} seen{where}")
    if not masks:
        return Outcome("fail", f"no {noun} found", marks=(Mark("warn", ref=c.region),) if c.region else ())
    return Outcome("fail", f"{noun} not{where}", marks=_rings(masks, "error", p) + ((Mark("warn", ref=c.region),)
                                                                                   if c.region else ()))


def absent(c: Check, noun: str, masks: Sequence[CamMask], regions: Regions, p: Params) -> Outcome:
    where = f" in {c.region}" if c.region else ""
    hits = _inside(masks, regions, c.region, p)
    if not hits:
        return Outcome("pass", f"no {noun}{where}")
    return Outcome("fail", f"{noun} still{where}" if c.region else f"{noun} still there", marks=_rings(hits, "error", p))


def moved_to(c: Check, noun: str, before: Sequence[CamMask], after: Sequence[CamMask], regions: Regions,
             p: Params) -> Outcome:
    def inside(ms: Sequence[CamMask]) -> bool:
        return any(regions.contains(c.region, m.centroid) for m in ms)

    if inside(after):
        if not before:
            return Outcome("inconclusive", f"no {noun} located before the step")
        if inside(before):
            return Outcome("fail", f"{noun} was already in {c.region}; no movement proved",
                           marks=(Mark("warn", ref=c.region),))
        return Outcome("pass", f"{noun} moved into {c.region}")
    if not after:
        return Outcome("fail", f"no {noun} found", marks=(Mark("warn", ref=c.region),))
    return Outcome("fail", f"{noun} not in {c.region}", marks=_rings(after, "error", p) + (Mark("warn", ref=c.region),))


def count(c: Check, noun: str, masks: Sequence[CamMask], spark_points: Sequence[tuple[float, float]] | None,
          regions: Regions, p: Params) -> Outcome:
    n = len(_inside(masks, regions, c.region, p))
    where = f" in {c.region}" if c.region else ""
    if spark_points is not None:
        k = sum(1 for xy in spark_points if c.region is None or regions.contains(c.region, xy))
        if k != n:
            return Outcome("inconclusive", f"SAM counts {n} {noun}{where}, Spark counts {k}")
    if n == c.n:
        return Outcome("pass", f"{n} {noun}{where}")
    return Outcome("fail", f"{n} {noun}{where}, expected {c.n}", marks=(Mark("warn", ref=c.region),) if c.region else ())


# ---------------------------------------------------------------- judge


def judge(verdict: Mapping) -> Outcome:
    evidence = verdict.get("evidence") or ""
    if verdict.get("contradiction"):
        return Outcome("fail", str(verdict["contradiction"]))
    done = verdict.get("done")
    if done == "yes":
        return Outcome("pass", f"verdict yes: {evidence}" if evidence else "verdict yes")
    if done == "no":
        return Outcome("fail", evidence or "verdict no")
    return Outcome("inconclusive", f"verdict unsure: {evidence}" if evidence else "verdict unsure")
