"""A Muse Spark sweep adds what the scans cannot tell from the table (a white egg on a white table)."""

from __future__ import annotations

from types import SimpleNamespace

from lightwork.guide import load_plan, scale_plan

from .test_table_scans import Rig
from .test_tabletop import FH, FW, put, table, page_rect


class FakeBrain:
    label = "fake brain"
    run_dir = None

    def __init__(self, objects):
        self.objects = objects
        self.calls = 0

    def check(self):
        return None

    def ask(self, role, prompt, jpegs=(), **kw):
        self.calls += 1
        return SimpleNamespace(text="", data={"objects": self.objects}, provider="meta", model="muse-spark-1.3",
                               ms=1200.0, cost_usd=0.0)

    def summary(self):
        return f"fake: {self.calls}"

    def close(self):
        pass


def roi_box(rig, page_box):
    """A page-px box as the 0-1000 grid of the table view the session sends."""
    from lightwork.geometry import apply_h

    x, y, w, h = rig.ts.zone.roi
    (x0, y0), (x1, y1) = apply_h(rig.ts.geo.Hi, [page_box[:2], page_box[2:]])
    return {"x_min": round((x0 - x) / w * 1000), "y_min": round((y0 - y) / h * 1000),
            "x_max": round((x1 - x) / w * 1000), "y_max": round((y1 - y) / h * 1000)}


def wait_jobs(rig, scene, seconds=1.0):
    rig.run(seconds, scene)


def test_a_pale_object_the_scans_miss_is_added_by_a_sweep_and_kept():
    scene = put(table(), page_rect(250, 200, 100, 60, 0), (40, 60, 200))  # a red box the scans see
    scene = put(scene, page_rect(520, 260, 50, 40, 0), (205, 210, 215))  # a white egg they don't
    brain = FakeBrain([])
    rig = Rig(brain=brain)
    rig.run(2.0, scene)
    assert len(rig.ts.tracks.confirmed) == 1
    brain.objects = [dict(name="red box", **roi_box(rig, (200, 170, 300, 230))),
                     dict(name="white egg", **roi_box(rig, (495, 240, 545, 280)))]
    said = rig.ts.hear("look again", rig.t)
    assert said == ["Looking."]
    wait_jobs(rig, scene)
    names = sorted(tr.name or "" for tr in rig.ts.tracks.confirmed)
    assert names[-1] == "white egg" and len(names) == 2  # the red box was not added twice
    egg = next(tr for tr in rig.ts.tracks.confirmed if tr.name == "white egg")
    assert egg.sticky
    assert any("white egg" in s for s in rig.said)
    other = put(scene, page_rect(250, 340, 60, 60, 0), (180, 90, 30))  # something put down elsewhere: a scan
    rig.run(2.0, other)
    assert egg in rig.ts.tracks.confirmed  # nothing moved over the egg: still there


def test_a_job_that_misses_things_looks_by_itself():
    scene = put(table(), page_rect(560, 270, 120, 120, 0), (40, 90, 40))
    brain = FakeBrain([])
    rig = Rig(brain=brain, plan=scale_plan(load_plan("pancakes"), 2))
    rig.run(8.0, scene)
    assert brain.calls >= 1  # the bowl is there, the rest is missing: it looked over the whole table
    calls = brain.calls
    rig.run(10.0, scene)
    assert brain.calls <= calls + 1  # and it backs off when nothing new turns up
