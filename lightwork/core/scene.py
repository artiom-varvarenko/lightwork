"""Scene builder: C2 primitives in camera px from the reducer state (D5).

Layers, bottom to top: provisional targets (``info``, pulsing), active targets and source arrow, failure marks
(``error`` red / ``warn`` yellow), labels outside the work area. Failure rings win over coincident guidance
(the wrong A7 is also the recovery source). Cells are rings; objects and zones are their outlines.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from ..contracts.canonical import canonical_sha, q_px, q_xy
from ..contracts.layout import Layout
from ..contracts.render import Arrow, Label, Poly, Ring, SceneMsg
from ..contracts.stepgraph import Step
from ..geometry import Regions
from .checks import Mark
from .graph import split_ref
from .params import Params

ARROW_MIN_PX = 12.0


def scene_hash(scene: SceneMsg) -> str:
    """``canonical_sha`` of the scene without ``seq`` (the C1 ``scene`` event)."""
    body = scene.model_dump(mode="json")
    body.pop("seq")
    return canonical_sha(body)


class SceneBuilder:
    def __init__(self, regions: Regions, layout: Layout, params: Params):
        self.regions = regions
        self.p = params
        x, y, w, h = layout.work_area_xywh
        self.label_at = [q_px(x), q_px(y - 16 if y >= 40 else y + h + 32)]

    def _ring_r(self, ref: str) -> float:
        obj, cell = split_ref(ref)
        return self.p.cell_ring_frac * self.regions.grids[obj].pitch(cell)

    def _shape(self, ref: str, pid: str, role: str, pulse: bool = False):
        obj, cell = split_ref(ref)
        if cell is not None:
            return Ring(id=pid, c=q_xy(self.regions.centre(ref)), r=q_px(self._ring_r(ref)), role=role, pulse=pulse)
        pts = [q_xy(pt) for pt in self.regions.polygon(ref)]
        return Poly(id=pid, pts=pts, closed=True, role=role)

    def _arrow(self, step: Step) -> list:
        a_ref, b_ref = step.arrow.from_, step.arrow.to
        a, b = np.array(self.regions.centre(a_ref)), np.array(self.regions.centre(b_ref))
        length = float(np.linalg.norm(b - a))
        if length == 0:
            return [self._shape(a_ref, f"s:{a_ref}", "source")]

        def inset(ref: str) -> float:
            if split_ref(ref)[1] is not None:
                return self._ring_r(ref) + 4
            return 0.25 * length

        da, db = inset(a_ref), inset(b_ref)
        prims = [self._shape(a_ref, f"s:{a_ref}", "source")]
        if length - da - db >= ARROW_MIN_PX:
            u = (b - a) / length
            prims.append(Arrow(id=f"a:{a_ref}>{b_ref}", a=q_xy(a + u * da), b=q_xy(b - u * db), role="target"))
        return prims

    def build(self, *, active: Step | None, provisional: Step | None, marks: Sequence[Mark],
              label: str | None, status: str | None = None) -> list:
        prims: list = []
        busy: set[str] = set()
        if active is not None:
            busy = set(active.targets) | ({active.arrow.from_, active.arrow.to} if active.arrow else set())
        if provisional is not None:
            for ref in provisional.targets:
                if ref not in busy:
                    prims.append(self._shape(ref, f"p:{ref}", "info", pulse=True))
        if active is not None:
            if active.arrow is not None:
                prims.extend(self._arrow(active))
            for ref in active.targets:
                prims.append(self._shape(ref, f"t:{ref}", "target"))
        for i, m in enumerate(marks):
            if m.circle is not None:
                cx, cy, r = m.circle
                prims.append(Ring(id=f"m{i}", c=q_xy((cx, cy)), r=q_px(r), role=m.role))
            else:
                prims.append(self._shape(m.ref, f"m{i}", m.role))
        if label is not None:
            prims.append(Label(id="step", at=self.label_at, text=label, role="info"))
        if status:
            prims.append(Label(id="status", at=[q_px(self.label_at[0] + 120), self.label_at[1]],
                               text=status, role="warn"))
        return prims
