"""Step-graph helpers for the reducer: step walking, region references, transfers, reducer-level validation."""

from __future__ import annotations

from collections.abc import Iterable, Iterator

from ..contracts.layout import Layout
from ..contracts.stepgraph import Step, StepGraph, validate_semantics


def iter_steps(steps: Iterable[Step]) -> Iterator[Step]:
    """Every step, including inline ``insert_step`` recovery steps, depth-first in list order."""
    for s in steps:
        yield s
        for r in s.recovery:
            if r.step is not None:
                yield from iter_steps([r.step])


def split_ref(ref: str) -> tuple[str, str | None]:
    """``"plate:A6"`` -> ``("plate", "A6")``; ``"remote"`` -> ``("remote", None)``."""
    if ":" in ref:
        obj, cell = ref.split(":", 1)
        return obj, cell
    return ref, None


def transfer_of(step: Step) -> tuple[str, str, str] | None:
    """``(grid, source cell, target cell)`` when the step's arrow joins two cells of one grid (a liquid transfer)."""
    if step.arrow is None:
        return None
    ga, ca = split_ref(step.arrow.from_)
    gb, cb = split_ref(step.arrow.to)
    if ca is None or cb is None or ga != gb:
        return None
    return ga, ca, cb


def confirm_only(step: Step) -> bool:
    return all(c.kind == "confirm" for c in step.checks)


def region_hit(ref: str, hits: set[str]) -> bool:
    """``ref`` is in ``hits``; a grid object also counts when any of its cells is."""
    if ref in hits:
        return True
    obj, cell = split_ref(ref)
    return cell is None and any(h.startswith(obj + ":") for h in hits)


def validate(graph: StepGraph, layout: Layout) -> list[str]:
    """C3 semantics plus what the reducer needs from the graph and the layout (empty list = valid)."""
    errs = list(validate_semantics(graph))
    objects = {o.id: o for o in graph.objects}
    laid_grids = {oid for oid, o in layout.objects.items() if oid in objects and objects[oid].grid is not None
                  and o.corners_cam_px is not None}

    def placed(ref: str) -> bool:
        obj, cell = split_ref(ref)
        return obj in laid_grids if cell is not None else obj in layout.objects

    for s in iter_steps(graph.steps):
        where = f"step {s.id}"
        refs = list(s.targets) + ([s.arrow.from_, s.arrow.to] if s.arrow is not None else [])
        refs += [c.region for c in s.checks if c.region is not None]
        for ref in refs:
            if not placed(ref):
                errs.append(f"{where}: {ref!r} has no geometry in the layout")
        transfer = transfer_of(s)
        for c in s.checks:
            what = f"{where} check {c.id}"
            if c.kind in ("colour_change", "series") and c.object not in laid_grids:
                errs.append(f"{what}: grid {c.object!r} has no corners in the layout")
            if c.min_delta is not None and c.min_delta <= 0:
                errs.append(f"{what}: min_delta must be > 0")
            if c.tol is not None and c.tol <= 0:
                errs.append(f"{what}: tol must be > 0")
            if c.kind == "series":
                if c.ratio is not None and not 0 < c.ratio < 1:
                    errs.append(f"{what}: series ratio must be in (0, 1)")
                if c.cells is not None and len(c.cells) < 2:
                    errs.append(f"{what}: series needs at least 2 cells")
            if c.kind == "colour_change" and c.ratio is not None:
                ok = (c.direction == "darker" and transfer is not None and transfer[0] == c.object
                      and c.cells == [transfer[2]] and c.ratio > 0)
                if not ok:
                    errs.append(f"{what}: ratio needs direction darker on the target cell of a cell->cell arrow")
    return errs
