"""Studying at the table: a question gets a spoken answer, notes, and a Muse Image picture served by URL."""

from __future__ import annotations

import base64
import json
import random
from types import SimpleNamespace

import cv2
import httpx
import numpy as np
from fastapi.testclient import TestClient

from lightwork.contracts import render
from lightwork.helper import Helper, MIC_C, STRIP_X, WORK, free_spot
from lightwork.painter import BLACK_POINT, Painted, Painter, for_the_table
from lightwork.playground import DESIGN_H, DESIGN_W, Canvas, Geo
from lightwork.server.app import create_app
from lightwork.vision.overlay_mask import rasterize

from .test_helper import FakeBrain, FakeGate, finger, until
from .test_table_scans import Rig
from .test_tabletop import table

ANSWER = {"intent": "question", "game": "none", "job": "", "manual": "", "question": "", "say": "", "subject": "",
          "answer": "This book is about plant biology. A plant cell has a wall and chloroplasts.",
          "notes": ["PLANT CELLS HAVE WALLS", "CHLOROPLASTS MAKE FOOD FROM LIGHT"],
          "picture": "A labelled diagram of a plant cell.",
          "things": [{"name": "book", "x_min": 100, "y_min": 200, "x_max": 300, "y_max": 800}]}
SRC = "/pictures/" + "ab" * 16 + ".png"
PAINTED = Painted(src=SRC, width=1024, height=683, model="muse-image-1.0", provider="meta", ms=11000.0, cost_usd=0.01)


class FakePainter:
    def __init__(self, result=PAINTED):
        self.prompts = []
        self.result = result

    def paint(self, prompt):
        self.prompts.append(prompt)
        if isinstance(self.result, Exception):
            raise self.result
        return self.result


def answered(painted=PAINTED) -> Helper:
    g = Helper(0.0, random.Random(1))
    g._new_job("what is this book about", 0.0)
    ask, g.want = g.want, None  # the session takes it
    g.answered(ask, ANSWER, None, 1.0)
    return g


def test_a_question_is_answered_out_loud_at_once_and_the_picture_follows():
    g = answered()
    assert g.state == "answer" and g.job is None
    assert g.say == [ANSWER["answer"]]  # said before the picture exists: the wait is not empty
    assert g.want_picture == ANSWER["picture"] and g.painting
    assert all(len(line) <= 22 for line in g.notes) and len(g.notes) <= 3 and g.notes[0] == "PLANT CELLS HAVE WALLS"
    assert g.want is None  # no plan, no steps
    g.pictured(PAINTED, 12.0)
    assert g.picture is PAINTED and not g.painting


def test_a_failed_picture_leaves_the_answer_standing():
    g = answered()
    g.pictured(None, 12.0)
    assert g.state == "answer" and g.picture is None and not g.painting and g.notes


def test_the_picture_comes_first_in_the_scene_keeps_clear_of_the_book_and_the_controls():
    geo = Geo(np.eye(3), DESIGN_W, DESIGN_H)
    g = answered()
    g.pictured(PAINTED, 12.0)
    c = Canvas(geo)
    g.draw(c, 13.0)
    assert c.prims[0].kind == "picture" and c.prims[0].src == SRC  # black in a picture would hide what lies under it
    (ax, ay), (bx, by), (cx, cy), (dx, dy) = c.prims[0].corners
    assert ax < bx and ay == by and dy > ay and dx == ax  # top-left first, so it reads upright for the person
    assert bx <= STRIP_X and ay >= WORK[1] and dy <= WORK[3]  # not under NEXT, MIC, BACK, and above the clocks
    book = g.things[0].box
    assert ax >= book[2] or bx <= book[0]
    labels = [p for p in c.prims if p.kind == "label" and p.id.startswith("hp-l")]
    assert labels


def test_while_it_is_being_drawn_its_place_is_marked_and_nothing_of_a_picture_is_sent():
    geo = Geo(np.eye(3), DESIGN_W, DESIGN_H)
    g = answered()
    c = Canvas(geo)
    g.draw(c, 3.0)
    assert not [p for p in c.prims if p.kind == "picture"] and [p for p in c.prims if p.id == "hp-pic-sweep"]
    assert "DRAWING" in g.hud(3.0, 0)[0]


def test_a_new_question_clears_the_answer_and_a_late_picture_for_the_old_one_is_dropped(tmp_path):
    brain = FakeBrain(see=[dict(ANSWER), dict(ANSWER)])
    painter = FakePainter()
    rig = Rig(game="helper", voice=True, brain=brain, state_dir=tmp_path, mic_gate=FakeGate(), painter=painter)
    rig.run(0.5, table())
    g = rig.ts.games.game
    rig.ts.hear("what is this book about", rig.t)
    until(rig, table(), lambda: g.state == "answer" and painter.prompts)
    until(rig, table(), lambda: g.picture is not None)
    assert not g.painting
    g._new_job("something else", rig.t)
    assert g.picture is None and not g.notes and g._pic_rect is None


def test_the_picture_goes_dark_during_a_look(tmp_path):
    brain = FakeBrain(see=[dict(ANSWER), {**ANSWER, "picture": ""}])
    rig = Rig(game="helper", voice=True, brain=brain, state_dir=tmp_path, mic_gate=FakeGate(), painter=FakePainter())
    rig.run(0.5, table())
    g = rig.ts.games.game
    rig.ts.hear("what is this book about", rig.t)
    until(rig, table(), lambda: g.picture is not None)
    assert [p for p in rig.ts.scene(rig.t) if p.kind == "picture"]
    rig.run(3.5, table())  # the last look is too old to use again
    rect = g._pic_rect
    g._new_job("and this one", rig.t)
    g.state, g.picture, g._pic_rect = "answer", PAINTED, rect  # even with the picture still up when the look starts
    rig.run(0.05, table())
    assert rig.ts._look is not None and rig.ts.scene(rig.t) == []


def test_a_session_without_a_painter_answers_without_a_picture(tmp_path):
    brain = FakeBrain(see=dict(ANSWER))
    rig = Rig(game="helper", voice=True, brain=brain, state_dir=tmp_path, mic_gate=FakeGate())
    rig.run(0.5, table())
    g = rig.ts.games.game
    rig.ts.hear("what is this book about", rig.t)
    until(rig, table(), lambda: g.state == "answer" and not g.painting)
    assert g.picture is None and g.notes


def test_a_job_is_still_a_job_when_the_model_says_so_or_says_nothing():
    for extra in ({"intent": "job"}, {}):
        g = Helper(0.0, random.Random(1))
        g._new_job("cook something", 0.0)
        ask, g.want = g.want, None
        g.answered(ask, {"game": "none", "job": "soup", "things": [], "manual": "", "question": "", "say": "",
                            **extra}, None, 1.0)
        assert g.state == "planning"


# ---- the primitive, on its way to the page


def test_the_prim_refuses_anything_but_a_served_picture():
    ok = {"kind": "picture", "id": "p", "src": SRC, "corners": [[0, 0], [1, 0], [1, 1], [0, 1]]}
    assert render.Picture(**ok).src == SRC
    for bad in ("https://example.com/a.png", "/pictures/../secret.png", "data:image/png;base64,AAAA"):
        try:
            render.Picture(**{**ok, "src": bad})
        except ValueError:
            continue
        raise AssertionError(f"accepted {bad}")


def test_the_overlay_mask_covers_a_picture():
    prim = render.Picture(id="p", src=SRC, corners=[[100, 100], [300, 100], [300, 200], [100, 200]])
    mask = rasterize([prim], np.eye(3), (750, 420), (750, 420))
    assert mask[150, 200] == 255 and mask[50, 50] == 0


def test_the_route_serves_a_picture_and_nothing_else(tmp_path):
    name = "ab" * 16 + ".png"
    (tmp_path / name).write_bytes(b"png")
    (tmp_path / "secret.txt").write_text("no")
    client = TestClient(create_app(mode="play", pictures_dir=tmp_path))
    r = client.get(f"/pictures/{name}")
    assert r.status_code == 200 and r.content == b"png"
    assert client.get("/pictures/secret.txt").status_code == 404
    assert client.get("/pictures/" + "cd" * 16 + ".png").status_code == 404


# ---- the model call


def png_of(img) -> bytes:
    return cv2.imencode(".png", img)[1].tobytes()


def settings(tmp_path):
    return SimpleNamespace(model_api_key="k", meta_api_base="https://api.meta.ai/v1", state_dir=tmp_path, budget_usd=5.0)


def test_a_painted_picture_is_served_recorded_and_counted(tmp_path):
    img = np.zeros((1024, 1536, 3), np.uint8)
    img[100:200, 100:800] = (0, 255, 255)
    img[500:600, 100:800] = (BLACK_POINT - 1,) * 3  # a dark grey "black": no light on the table
    seen = []

    def reply(request: httpx.Request) -> httpx.Response:
        seen.append(json.loads(request.content))
        return httpx.Response(200, json={"created": 1, "data": [{"b64_json": base64.b64encode(png_of(img)).decode()}]})

    p = Painter(settings(tmp_path), transport=httpx.MockTransport(reply), run_dir=tmp_path / "run")
    out = p.paint("A diagram.")
    assert seen[0]["model"] == "muse-image-1.0" and "pure black" in seen[0]["prompt"]
    assert out.src.startswith("/pictures/") and out.cost_usd == 0.01 and out.width == 1024
    served = cv2.imread(str(tmp_path / "pictures" / out.src.rsplit("/", 1)[1]))
    assert served.shape[1] == 1024 and served[:, :, :].max() > 200
    assert served[int(550 * 683 / 1024), 300].max() == 0  # the grey is gone
    calls = list((tmp_path / "run" / "calls").iterdir())
    assert len(calls) == 1 and calls[0].name.endswith("image-muse-image-1.0")
    response = json.loads((calls[0] / "response.json").read_text())
    assert "$frame" in response["data"][0]["b64_json"]  # the recording holds its hash, the picture is beside it
    assert len(list((tmp_path / "run" / "frames").glob("*.png"))) == 1
    meta = json.loads((calls[0] / "meta.json").read_text())
    assert meta["cost_usd"] == 0.01 and meta["provenance"] == "meta" and meta["status"] == "ok"
    assert json.loads((tmp_path / "spend.json").read_text())["spent_usd"] == 0.01
    assert "1 call" in p.summary()


def test_a_refused_picture_raises_is_recorded_and_costs_nothing(tmp_path):
    p = Painter(settings(tmp_path), transport=httpx.MockTransport(lambda r: httpx.Response(400, json={"error": "no"})),
                run_dir=tmp_path / "run")
    try:
        p.paint("A diagram.")
    except RuntimeError as exc:
        assert "400" in str(exc)
    else:
        raise AssertionError("no error")
    assert json.loads((tmp_path / "spend.json").read_text())["spent_usd"] == 0.0
    assert len(list((tmp_path / "run" / "calls").iterdir())) == 1


def test_black_stays_black_and_a_big_picture_shrinks():
    img = np.zeros((800, 2000, 3), np.uint8)
    png, w, h = for_the_table(png_of(img))
    assert (w, h) == (1024, 410) and cv2.imdecode(np.frombuffer(png, np.uint8), cv2.IMREAD_COLOR).max() == 0


def test_free_spot_is_used_for_the_picture_in_an_empty_work_area():
    assert free_spot(345.0, 230.0, []) == (WORK[0], WORK[1])
