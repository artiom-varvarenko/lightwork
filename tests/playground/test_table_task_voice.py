"""lightwork play --table: the bench check (--task), spoken questions (--voice) and SAM outlines (--sam)."""

from __future__ import annotations

import json
import random
import re

import cv2
import httpx
import numpy as np
import pytest

from lightwork.benchtask import load_task, match_item, spoken_list
from lightwork.config import Settings
from lightwork.geometry import apply_h
from lightwork.models.budget import Budget
from lightwork.models.client import ModelClient
from lightwork.models.synthetic import SamObject, responses_sse, sam_sse

from .test_table_scans import Rig
from .test_tabletop import HI, page_rect, pose, put, table


def test_a_task_comes_from_a_json_or_text_file_or_inline(tmp_path):
    (tmp_path / "t.json").write_text(json.dumps({"title": "Serial dilution", "items": ["96-well plate", "Pipette"]}))
    (tmp_path / "t.txt").write_text("Kit check\n# comment\n- Remote\n- Charger\n")
    assert load_task(str(tmp_path / "t.json")).items == ("96-well plate", "Pipette")
    t = load_task(str(tmp_path / "t.txt"))
    assert (t.title, t.items) == ("Kit check", ("Remote", "Charger"))
    t = load_task("Serial dilution: well plate, pipette tips; food dye")
    assert (t.title, t.items) == ("Serial dilution", ("well plate", "pipette tips", "food dye"))
    with pytest.raises(ValueError):
        load_task("Nothing:")
    assert match_item("multichannel pipette", t.items) == 2
    assert match_item("blue food dye bottle", t.items) == 3
    assert match_item("claw hammer", t.items) == 0
    assert spoken_list(["a", "b", "c"]) == "a, b and c"


class TaskNamer:
    """Names in the order asked, each with the task item the model would have matched."""

    label, error = "fake", None

    def __init__(self, answers):
        self.answers = list(answers)
        self.out: dict = {}
        self.matches: dict = {}

    def request(self, key, jpeg):
        name, item = self.answers.pop(0)
        self.out[key], self.matches[key] = name, item
        return True

    def result(self, key):
        return (True, self.out.pop(key)) if key in self.out else (False, None)

    def match(self, key):
        return self.matches.pop(key, None)

    def summary(self):
        return "fake"


def test_the_bench_check_ticks_items_off_and_flags_what_is_not_needed():
    task = load_task("Serial dilution: well plate, pipette, food dye")
    namer = TaskNamer([("pipette", 2), ("claw hammer", 0), ("well plate", 1), ("dye bottle", 3)])
    rig = Rig(namer=namer, task=task)
    scene = table()
    for rect, bgr in ((page_rect(250, 150, 140, 30, 0), (40, 60, 200)), (page_rect(560, 320, 180, 40, 0), (30, 30, 40)),
                      (page_rect(450, 160, 120, 80, 0), (180, 90, 30)), (page_rect(250, 320, 50, 50, 0), (60, 170, 60))):
        scene = put(scene, rect, bgr)
        rig.run(1.5, scene)
    said = " | ".join(rig.said)
    assert "Pipette, check. 2 to go: well plate and food dye." in said
    assert "Claw hammer is not on the list for Serial dilution." in said
    assert "Everything for Serial dilution is on the table. Ready to start." in said
    assert rig.ts.dash.task_line == "SERIAL DILUTION 3/3 READY"
    roles = {tr.name: rig.ts.dash._role(tr, rig.t) for tr in rig.ts.tracks.confirmed}
    assert roles == {"pipette": "target", "claw hammer": "error", "well plate": "target", "dye bottle": "target"}
    assert "error" in rig.ts.sfx and "answer" in rig.ts.sfx


class FakeAsker:
    label, error = "fake", None

    def __init__(self):
        self.asked = []
        self.ready = False

    def ask(self, key, question, jpegs, context=""):
        self.asked.append((key, question, jpegs, context))
        return True

    def result(self, key):
        if self.ready and self.asked and self.asked[-1][0] == key:
            return True, "That is a stapler: it joins sheets of paper."
        return False, None

    def summary(self):
        return "fake"


def test_what_is_this_asks_about_the_object_pointed_at():
    asker = FakeAsker()
    rig = Rig(asker=asker)
    obj = put(table(), page_rect(300, 200, 110, 50, 0), (40, 60, 200))
    rig.run(1.5, obj)
    (tr,) = rig.ts.tracks.confirmed
    # a hand whose index fingertip rests just beside the object, then is gone when the sentence is heard
    w, k = apply_h(HI, [(300, 330), (300, 270)])
    p = pose(w, k)
    tip = p.img[8]
    shift = apply_h(HI, [(300, 236)])[0] - tip  # move the hand so its fingertip is 11 page px below the object
    p = p.shifted(*shift)
    rig.run(0.3, obj, poses=lambda t: [p])
    rig.run(0.5, obj)
    assert rig.ts.hear("What is this?", rig.t) == []
    ((_, question, jpegs, context),) = asker.asked
    assert question == "What is this?" and len(jpegs) == 2  # the object, then the whole table
    assert all(j[:3] == b"\xff\xd8\xff" for j in jpegs)
    assert f"pointing at object {tr.id}" in context
    assert rig.ts.dash._role(tr, rig.t) == "target" and "listen" in rig.ts.sfx
    asker.ready = True
    rig.run(0.1, obj)
    assert rig.said[-1] == "That is a stapler: it joins sheets of paper." and "answer" in rig.ts.sfx


def test_what_is_missing_is_answered_from_the_bench_check_without_a_model():
    asker = FakeAsker()
    rig = Rig(asker=asker, namer=TaskNamer([("pipette", 2)]), task=load_task("Dilution: plate, pipette, dye"))
    rig.run(1.5, put(table(), page_rect(300, 200, 140, 30, 0), (40, 60, 200)))
    assert rig.ts.hear("What's missing?", rig.t) == ["Still missing: plate and dye."]
    assert asker.asked == []


def _sam_handler(seen):
    """A SAM stand-in: for every box prompt, an ellipse filling the box (sent-image px)."""

    def handler(req):
        body = json.loads(req.content)
        seen.append(body)
        text = body["input"][0]["content"][0]["text"]
        objs = []
        size = None
        for oid, fields in re.findall(r"(\d+)<\|box;([^|]*)\|>", text):
            f = dict(kv.split("=") for kv in fields.split(";"))
            x0, y0, x1, y1 = int(f["x1"]), int(f["y1"]), int(f["x2"]) + 1, int(f["y2"]) + 1
            size = (int(f["w"]), int(f["h"]))
            mask = np.zeros((y1 - y0, x1 - x0), np.uint8)
            cv2.ellipse(mask, ((x1 - x0) // 2, (y1 - y0) // 2), ((x1 - x0) // 2 - 6, (y1 - y0) // 2 - 6), 0, 0, 360, 1, -1)
            objs.append(SamObject(int(oid), (x0, y0, x1, y1), mask.astype(bool), 0.9))
        return httpx.Response(200, content=sam_sse(objs, size), headers={"content-type": "text/event-stream"})

    return handler


def test_sam_outlines_come_back_as_camera_contours(tmp_path, no_network):
    from lightwork.outliner import SamOutliner

    seen = []
    settings = Settings(model_api_key="secret-never-record", root=tmp_path)
    client = ModelClient(settings, run_dir=tmp_path / "sam", provenance="meta", budget=Budget(tmp_path / "spend.json", 5.0),
                         cache=None, transport=httpx.MockTransport(_sam_handler(seen)), rng=random.Random(0))
    out = SamOutliner(settings, run_dir=tmp_path / "sam", client=client)
    try:
        img = np.full((540, 960, 3), 200, np.uint8)
        assert out.request(img, [(7, (300, 200, 420, 260)), (9, (600, 100, 660, 180))])
        got = {}
        for _ in range(300):
            for key in (7, 9):
                done, contour = out.result(key)
                if done:
                    got[key] = contour
            if len(got) == 2:
                break
            import time

            time.sleep(0.01)
        assert set(got) == {7, 9}
        c7 = got[7]
        assert c7.min(axis=0)[0] >= 300 and c7.max(axis=0)[0] <= 420  # camera px, inside its box
        assert abs(c7.mean(axis=0)[0] - 360) < 3 and abs(c7.mean(axis=0)[1] - 230) < 3
        (body,) = seen
        assert body["model"] == "sam-3.1" and "7<|box" in body["input"][0]["content"][0]["text"]
        assert out.calls == 1 and out.cost_usd == pytest.approx(0.0025)
    finally:
        out.close()


def test_a_session_outline_snaps_to_the_sam_mask():
    class Outliner:
        def __init__(self):
            self.asked = {}

        def request(self, img, objects):
            for key, box in objects:
                x0, y0, x1, y1 = box
                c = np.array([[x0 + 10, y0 + 6], [x1 - 10, y0 + 6], [x1 - 10, y1 - 6], [x0 + 10, y1 - 6]], float)
                self.asked[key] = c
            return True

        def result(self, key):
            return (True, self.asked.pop(key)) if key in self.asked else (False, None)

        def summary(self):
            return "fake"

    rig = Rig(outliner=Outliner())
    rig.run(1.5, put(table(), page_rect(300, 200, 110, 50, 0), (40, 60, 200)))
    (tr,) = rig.ts.tracks.confirmed
    assert tr.refined and tr.sam_state == "done"
    assert any(p.id == f"obj-{tr.uid}-s" for p in rig.ts.scene(rig.t))  # a new prim: it draws itself in place


def test_the_asker_records_its_call_and_keeps_answers_short(tmp_path, no_network):
    from lightwork.asker import MAX_CHARS, MetaAsker

    long_answer = "This is a claw hammer. " + "It is used to drive nails into wood and pull them out again. " * 10

    def handler(req):
        body = json.loads(req.content)
        return httpx.Response(200, content=responses_sse(long_answer, model=body["model"]),
                              headers={"content-type": "text/event-stream"})

    settings = Settings(model_api_key="secret-never-record", root=tmp_path)
    client = ModelClient(settings, run_dir=tmp_path / "ask", provenance="meta", budget=Budget(tmp_path / "spend.json", 5.0),
                         cache=None, transport=httpx.MockTransport(handler), rng=random.Random(0))
    asker = MetaAsker(settings, run_dir=tmp_path / "ask", client=client)
    try:
        assert asker.ask(1, "what is this?", [b"\xff\xd8\xff" + bytes(50)], "Objects: 1: hammer.")
        for _ in range(300):
            done, answer = asker.result(1)
            if done:
                break
            import time

            time.sleep(0.01)
        assert done and answer.startswith("This is a claw hammer.") and len(answer) <= MAX_CHARS
        assert answer.endswith(".")
        assert len(list((tmp_path / "ask" / "calls").iterdir())) == 1
    finally:
        asker.close()
