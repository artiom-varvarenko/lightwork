"""Keyframe measurements (C1 ``measurement``) through ``lightwork.measure`` (workstream E).

What is measured: per grid object, every cell named by a ``colour_change`` or ``series`` check, plus the neighbours
(``Params.wrong_cell_reach``) of each ``darker`` target, which are the wrong-cell candidates. One ``measurement``
event per grid and verified keyframe, on the grid's ``channel``; cells are keyed ``<grid>:<cell>``; unreadable cells
(``n == 0``) are left out.

Reference rule (owned here): a cell's OD is relative to its linear intensity in the first verified keyframe of the
run in which it was readable. The orchestrator captures that keyframe before any dye arrives, so the OD is absolute
for the dilution physics. Checks compare ODs of two keyframes, which does not depend on the reference. A flat field
is not applied: the per-cell reference already cancels illumination non-uniformity.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from ..contracts.stepgraph import StepGraph
from ..geometry import Regions
from ..measure.colour import cell_stats, decode, od, od_sd
from ..measure.grid import cell_rois, neighbours
from .graph import iter_steps
from .params import Params


@dataclass(frozen=True)
class CellRead:
    od: float
    lab: tuple[float, float, float]
    n: int
    sd: float


@dataclass(frozen=True)
class _GridPlan:
    grid: str
    channel: str
    centres: dict[str, tuple[float, float]]  # cell name -> camera px
    radius: float


class Readings:
    def __init__(self, graph: StepGraph, regions: Regions, params: Params):
        wanted: dict[str, set[str]] = {}
        grids = {o.id: o.grid for o in graph.objects if o.grid is not None}
        for s in iter_steps(graph.steps):
            for c in s.checks:
                if c.kind not in ("colour_change", "series"):
                    continue
                cells = wanted.setdefault(c.object, set())
                cells.update(c.cells)
                if c.kind == "colour_change" and c.direction == "darker":
                    g = grids[c.object]
                    for cell in c.cells:
                        cells.update(neighbours(cell, g.rows, g.cols, g.naming, params.wrong_cell_reach))
        self._plans: list[_GridPlan] = []
        for gid in sorted(wanted):
            geom = regions.grids[gid]
            names = [n for n in geom.names() if n in wanted[gid]]
            centres, radius = cell_rois(geom, names)
            self._plans.append(_GridPlan(gid, grids[gid].channel, centres, radius))
        self._i_ref: dict[str, float] = {}
        self._frames: dict[str, dict[str, CellRead]] = {}

    def measure(self, sha: str, jpeg: bytes, crop_xywh: Sequence[int]) -> list[dict]:
        """Measure a verified keyframe once; returns the ``measurement`` payloads to emit."""
        if sha in self._frames or not self._plans:
            self._frames.setdefault(sha, {})
            return []
        img = decode(jpeg)
        origin = (float(crop_xywh[0]), float(crop_xywh[1]))
        frame: dict[str, CellRead] = {}
        out = []
        for plan in self._plans:
            stats = cell_stats(img, plan.centres, plan.radius, plan.channel, origin=origin)
            cells = {}
            for name, st in stats.items():
                ref = f"{plan.grid}:{name}"
                if st.n == 0:
                    continue
                if ref not in self._i_ref:
                    if st.i_lin <= 0:
                        continue
                    self._i_ref[ref] = st.i_lin
                read = CellRead(od=od(st.i_lin, self._i_ref[ref]), lab=st.lab, n=st.n, sd=od_sd(st))
                frame[ref] = read
                cells[ref] = {"od": read.od, "lab": list(read.lab), "n": read.n, "sd": read.sd}
            out.append({"frame_sha": sha, "channel": plan.channel, "cells": cells})
        self._frames[sha] = frame
        return out

    def at(self, sha: str | None) -> dict[str, CellRead]:
        return self._frames.get(sha, {}) if sha is not None else {}
