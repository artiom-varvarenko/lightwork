"""Exploring: "show me a ship" becomes a 3D model from Muse Wonder that you turn, slice open and move through with your
hands, and whose parts you can ask for by name."""

from __future__ import annotations

import json
import random

import numpy as np

from lightwork.contracts import render
from lightwork.games import Pen
from lightwork.helper import (
    IDLE_SPIN,
    MIC_C,
    MODEL_RECT,
    SLICE_X,
    SLICE_Y,
    STRIP_X,
    TILT_RANGE,
    TILT_TALL,
    TILT_WIDE,
    ZOOM_RANGE,
    Helper,
    asked_to_show,
)
from lightwork.playground import DESIGN_H, DESIGN_W, Canvas, Geo
from lightwork.vision.overlay_mask import rasterize

from .test_helper import FakeBrain, FakeGate, until
from .test_table_scans import Rig
from .test_tabletop import table

SRC = "/pictures/" + "ab" * 16 + ".json"
SHIP = {"name": "Ship", "src": SRC, "half": [1.0, 0.2, 0.3], "parts": ["hull", "engine room", "propeller"],
        "boxes": {"hull": [0, 0, 0, 1.0, 0.2, 0.3], "engine room": [-0.5, 0, -0.1, 0.1, 0.05, 0.05],
                  "propeller": [-0.95, 0, -0.2, 0.05, 0.05, 0.05]}, "triangles": 100, "lines": 3}
ON_LEFT = (300.0, 200.0)


def showing(data=SHIP) -> Helper:
    g = Helper(0.0, random.Random(1))
    g.show_model("ship", 0.0)
    ask, g.want = g.want, None
    g.answered(ask, data, None, 1.0)
    return g


def test_what_asks_for_a_model_and_what_does_not():
    assert asked_to_show("Show me a bicycle.") == "bicycle"
    assert asked_to_show("Can you show me the Eiffel Tower in 3D?") == "eiffel tower"
    assert asked_to_show("Can you just show me a ship?") == "ship"
    assert asked_to_show("please show me an old steam engine") == "old steam engine"
    for not_one in ("Show me how to fold this", "show me the book", "show me what this is", "help me build a bicycle",
                    "show me", "show me the map of europe please and then more of it and even more", "what is a bicycle",
                    "show me the inside"):
        assert asked_to_show(not_one) is None, not_one


def test_it_comes_up_from_the_side_that_suits_it():
    assert showing().tilt == TILT_WIDE
    assert showing({**SHIP, "half": [0.4, 0.4, 1.0]}).tilt == TILT_TALL
    assert showing({**SHIP, "half": [1.0, 0.2, 0.3]}).fit_dist < showing({**SHIP, "half": [1.0, 1.0, 1.0]}).fit_dist


def test_one_pinched_hand_turns_it_and_tilts_it_within_limits():
    g = showing()
    yaw, tilt = g.yaw, g.tilt
    g.update(2.0, [], [], (), [Pen(ON_LEFT, True, 0)])
    g.update(2.05, [], [], (), [Pen((340.0, 230.0), True, 0)])
    assert g.yaw > yaw and g.tilt > tilt
    for k in range(200):
        g.update(3.0 + k * 0.05, [], [], (), [Pen((340.0, 230.0 + 20 * k), True, 0)])
    assert g.tilt <= TILT_RANGE[1]


def test_an_open_hand_does_nothing_and_a_tracker_jump_does_not_throw_it():
    g = showing()
    g.update(1.5, [], [], (), [])
    yaw = g.yaw
    g.update(2.0, [], [], (), [Pen(ON_LEFT, False, 0)])  # moving, fingers open: only the idle spin
    assert abs(g.yaw - yaw) < 0.1
    g.update(2.1, [], [], (), [Pen((150.0, 100.0), True, 0)])
    yaw = g.yaw
    g.update(2.15, [], [], (), [Pen((500.0, 300.0), True, 0)])  # 400 px in a frame
    assert g.yaw == yaw


def test_a_pinch_on_the_bar_slices_and_its_hand_can_leave_the_bar():
    g = showing()
    assert g.cut_at is None
    g.update(2.0, [], [], (), [Pen((SLICE_X, SLICE_Y[1]), True, 0)])
    yaw = g.yaw
    g.update(2.05, [], [], (), [Pen((SLICE_X + 5, (SLICE_Y[0] + SLICE_Y[1]) / 2), True, 0)])
    assert g.cut == 0.5 and abs(g.cut_at) < 1e-6 and g.cut_axis == 0 and g.yaw == yaw  # the longest side, and no turning
    g.update(2.1, [], [], (), [Pen((SLICE_X + 90, SLICE_Y[0]), True, 0)])  # it stays with the hand, wandering off the bar
    assert g.cut == 1.0
    g.update(2.15, [], [], (), [])
    assert g._slicer is None and g.cut == 1.0  # let go: the cut stays where it is


def test_a_hand_turning_the_model_passing_over_the_bar_does_not_take_it():
    g = showing()
    g.update(2.0, [], [], (), [Pen((SLICE_X + 70, 200.0), True, 0)])
    g.update(2.05, [], [], (), [Pen((SLICE_X + 10, 200.0), True, 0)])
    assert g.cut == 0.0 and g._slicer is None


def test_one_hand_slices_while_the_other_turns():
    g = showing()
    yaw = g.yaw
    pens = [Pen((SLICE_X, 300.0), True, 0), Pen(ON_LEFT, True, 1)]
    g.update(2.0, [], [], (), pens)
    g.update(2.05, [], [], (), [Pen((SLICE_X, 250.0), True, 0), Pen((340.0, 200.0), True, 1)])
    assert g.cut > 0.0 and g.yaw > yaw


def test_two_pinched_hands_zoom_in_and_move_through_it():
    g = showing()
    g.update(2.0, [], [], (), [Pen((300.0, 200.0), True, 0), Pen((400.0, 200.0), True, 1)])
    assert g.zoom == 1.0
    g.update(2.05, [], [], (), [Pen((250.0, 200.0), True, 0), Pen((450.0, 200.0), True, 1)])
    assert g.zoom == 2.0 and g.target == [0.0, 0.0, 0.0]
    g.update(2.1, [], [], (), [Pen((300.0, 200.0), True, 0), Pen((500.0, 200.0), True, 1)])  # both to the right
    assert g.zoom == 2.0 and g.target != [0.0, 0.0, 0.0]
    g.update(2.15, [], [], (), [Pen((130.0, 200.0), True, 0), Pen((615.0, 200.0), True, 1)])
    assert g.zoom <= ZOOM_RANGE[1]
    assert g.fit_dist / ZOOM_RANGE[1] < 0.5  # close enough to be inside it


def test_left_alone_it_settles_into_a_slow_turn_but_not_when_cut_open_or_close_up():
    g = showing()
    g.update(2.0, [], [], (), [Pen(ON_LEFT, True, 0)])
    g.update(2.05, [], [], (), [Pen((360.0, 200.0), True, 0)])
    assert g.spin > IDLE_SPIN
    t = 2.05
    for _ in range(200):
        t += 0.05
        g.update(t, [], [], (), [])
    assert abs(g.spin - IDLE_SPIN) < 0.05
    g.cut = 0.5
    for _ in range(200):
        t += 0.05
        g.update(t, [], [], (), [])
    assert abs(g.spin) < 0.05


def test_the_mic_side_is_no_handle():
    g = showing()
    yaw = g.yaw
    g.update(2.0, [], [], (), [Pen((STRIP_X + 40, 200.0), True, 0)])
    g.update(2.05, [], [], (), [Pen((STRIP_X + 80, 240.0), True, 0)])
    assert g._drag is None and abs(g.yaw - yaw) < 0.2


def test_it_goes_out_as_one_prim_with_no_line_of_it_in_the_scene():
    g = showing()
    g.cut = 0.25
    c = Canvas(Geo(np.eye(3), DESIGN_W, DESIGN_H))
    g.draw(c, 2.0)
    solid = next(p for p in c.prims if p.kind == "solid")
    assert solid.src == SRC and solid.cut == g.cut_at and solid.cut_axis == 0 and solid.only == ""
    assert solid.corners[0] == [MODEL_RECT[0], MODEL_RECT[1]] and max(q[0] for q in solid.corners) < STRIP_X
    assert solid.dist == round(g.fit_dist / g.zoom, 4)
    assert [p for p in c.prims if p.id == "hp-slice-h"]
    assert "SLICES" in g.hud(2.0, 0)[1]
    assert len(solid.model_dump_json()) < 600


def test_the_prim_refuses_anything_but_a_served_model_and_is_masked():
    prim = render.Solid3D(id="m", src=SRC, corners=[[100, 100], [300, 100], [300, 200], [100, 200]], yaw=0, tilt=0, dist=2, target=[0, 0, 0])
    assert rasterize([prim], np.eye(3), (750, 420), (750, 420))[150, 200] == 255
    try:
        render.Solid3D(id="m", src="https://example.com/a.json", corners=prim.corners, yaw=0, tilt=0, dist=2, target=[0, 0, 0])
    except ValueError:
        return
    raise AssertionError("accepted a url from elsewhere")


def test_saying_slice_and_asking_for_a_part_need_no_model():
    g = showing()
    assert g.model_command("Slice it") and g.cut == 0.5
    assert g.model_command("show me the engine room") and g.only == "engine room" and g.cut == 0.0
    t = 2.0
    for _ in range(100):  # the camera goes to it
        t += 0.05
        g.update(t, [], [], (), [])
    assert g.target == SHIP["boxes"]["engine room"][:3] and g.zoom > 1.0 and "ENGINE ROOM" in g.hud(t, 0)[0]
    c = Canvas(Geo(np.eye(3), DESIGN_W, DESIGN_H))
    g.draw(c, t)
    assert next(p for p in c.prims if p.kind == "solid").only == "engine room"
    assert g.model_command("show everything") and g.only == "" and g.cut == 0.0
    for _ in range(100):
        t += 0.05
        g.update(t, [], [], (), [])
    assert g.zoom == 1.0 and g.target == [0.0, 0.0, 0.0]
    assert not g.model_command("what is the weather")
    assert not g.model_command("show me a bicycle")  # not a part of it: a new model


def test_the_hands_take_over_from_the_camera_on_its_way():
    g = showing()
    g.model_command("show me the propeller")
    g.update(2.0, [], [], (), [])
    assert g._glide is not None
    g.update(2.05, [], [], (), [Pen((300.0, 200.0), True, 0), Pen((400.0, 200.0), True, 1)])
    assert g._glide is None


def test_asking_again_while_it_builds_does_not_start_it_over():
    g = Helper(0.0, random.Random(1))
    g.show_model("ship", 0.0)
    ask, epoch = g.want, g.epoch
    g.show_model("ship", 5.0)
    assert g.want is ask and g.epoch == epoch and g.say[-1] == "Still building it."
    g.show_model("bicycle", 6.0)
    assert g.subject == "bicycle" and g.epoch == epoch + 1


def test_a_failed_model_says_so_and_leaves_the_tile_ready():
    g = Helper(0.0, random.Random(1))
    g.show_model("cube", 0.0)
    ask, g.want = g.want, None
    g.answered(ask, None, "boom", 1.0)
    assert g.state == "idle" and "couldn't build a cube" in g.say[-1]
    g.show_model("cube", 2.0)
    ask, g.want = g.want, None
    g.answered(ask, {"name": "x"}, None, 3.0)
    assert g.state == "idle"


CUBE_PROGRAM = {"name": "Cube", "code": """
part("outside", "#ff0000")
box((0, 0, 0), (2, 2, 2))
part("core", "#00ff00")
sphere((0, 0, 0), 0.5, n=6)
"""}


def test_saying_it_asks_wonder_without_a_look_and_a_retake_comes_from_the_cache(tmp_path):
    brain = FakeBrain(model=CUBE_PROGRAM)
    rig = Rig(game="helper", voice=True, brain=brain, state_dir=tmp_path, mic_gate=FakeGate())
    rig.run(0.5, table())
    g = rig.ts.games.game
    rig.ts.hear("Show me a cube.", rig.t)
    assert g.state == "modelling" and rig.ts._look is None
    until(rig, table(), lambda: g.state == "model")
    assert rig.ts._look is None and rig.ts._look_jpeg is None
    assert len(brain.calls) == 1 and brain.calls[0].jpegs == [] and "a cube" in brain.calls[0].prompt
    kept = json.loads((tmp_path / "play-models" / "solid-cube.json").read_text())
    assert kept["name"] == "Cube" and kept["parts"] == ["outside", "core"]
    assert (tmp_path / "pictures" / g.model["src"].rsplit("/", 1)[1]).is_file()
    prims = rig.ts.scene(rig.t)
    assert [p for p in prims if p.kind == "solid"]
    said = rig.ts.hear("Show me the core", rig.t)  # a part of what is on show: no model call
    assert g.only == "core" and len(brain.calls) == 1 and said == ["The core."]
    rig.ts.hear("stop", rig.t)
    assert g.state == "idle" and g.model is None
    rig.ts.hear("show me a cube", rig.t)
    until(rig, table(), lambda: g.state == "model", 1.0)
    assert len(brain.calls) == 1  # no second call


def test_a_kept_model_whose_files_are_gone_is_built_again(tmp_path):
    brain = FakeBrain(model=CUBE_PROGRAM)
    rig = Rig(game="helper", voice=True, brain=brain, state_dir=tmp_path, mic_gate=FakeGate())
    rig.run(0.5, table())
    g = rig.ts.games.game
    rig.ts.hear("show me a cube", rig.t)
    until(rig, table(), lambda: g.state == "model")
    (tmp_path / "pictures" / g.model["src"].rsplit("/", 1)[1]).unlink()
    rig.ts.hear("stop", rig.t)
    rig.ts.hear("show me a cube", rig.t)
    until(rig, table(), lambda: g.state == "model")
    assert len(brain.calls) == 2


def test_the_model_can_name_a_thing_too(tmp_path):
    see = {"intent": "model", "subject": "cube", "game": "none", "job": "", "things": [], "manual": "", "question": "",
           "say": "", "answer": "", "notes": [], "picture": ""}
    brain = FakeBrain(see=see, model=CUBE_PROGRAM)
    rig = Rig(game="helper", voice=True, brain=brain, state_dir=tmp_path, mic_gate=FakeGate())
    rig.run(0.5, table())
    g = rig.ts.games.game
    g._new_job("let's have a look at a cube in three dimensions", rig.t)
    until(rig, table(), lambda: g.state == "model")
    assert [c.role for c in brain.calls] == ["helper-see", "helper-model"]


def test_holding_the_mic_still_works_while_a_model_shows():
    g = showing()
    for k in range(10):
        g.update(2.0 + k * 0.05, [MIC_C], [MIC_C], (), [])
    assert g.mic_open and g.state == "model"
