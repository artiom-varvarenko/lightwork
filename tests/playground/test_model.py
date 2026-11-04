"""Exploring: "show me a bicycle" becomes a 3D wireframe from Muse Wonder that pinching turns and two hands zoom."""

from __future__ import annotations

import json
import math
import random

import numpy as np

from lightwork.games import Pen
from lightwork.helper import (
    IDLE_SPIN,
    MAX_WIRE_POINTS,
    MIC_C,
    MODEL_C,
    STRIP_X,
    TILT_RANGE,
    TILT_TALL,
    TILT_WIDE,
    ZOOM_RANGE,
    Helper,
    asked_to_show,
    parse_wire,
    project,
)
from lightwork.playground import DESIGN_H, DESIGN_W, Canvas, Geo

from .test_helper import FakeBrain, FakeGate, until
from .test_table_scans import Rig
from .test_tabletop import table

CUBE = {"name": "Cube", "lines": [[[0, 0, 0], [2, 0, 0], [2, 2, 0], [0, 2, 0], [0, 0, 0]],
                                  [[0, 0, 2], [2, 0, 2], [2, 2, 2], [0, 2, 2], [0, 0, 2]],
                                  [[0, 0, 0], [0, 0, 2]], [[2, 0, 0], [2, 0, 2]], [[2, 2, 0], [2, 2, 2]], [[0, 2, 0], [0, 2, 2]]]}


def showing() -> Helper:
    g = Helper(0.0, random.Random(1))
    g.show_model("cube", 0.0)
    ask, g.want = g.want, None
    g.answered(ask, CUBE, None, 1.0)
    return g


def test_what_asks_for_a_model_and_what_does_not():
    assert asked_to_show("Show me a bicycle.") == "bicycle"
    assert asked_to_show("Can you show me the Eiffel Tower in 3D?") == "eiffel tower"
    assert asked_to_show("please show me an old steam engine") == "old steam engine"
    for not_one in ("Show me how to fold this", "show me the book", "show me what this is", "help me build a bicycle",
                    "show me", "show me the map of europe please and then more of it and even more", "what is a bicycle"):
        assert asked_to_show(not_one) is None, not_one


def test_a_wireframe_is_centred_and_scaled_and_junk_is_refused():
    w = parse_wire(CUBE)
    every = np.concatenate(w.lines)
    assert w.name == "Cube" and every.min() == -1.0 and every.max() == 1.0
    assert parse_wire({"name": "x", "lines": [[[0, 0, 0], [1, 1, 1]]]}) is None  # too few lines
    assert parse_wire({"name": "x", "lines": [[[1, 1, 1], [1, 1, 1]]] * 4}) is None  # one spot
    assert parse_wire({"name": "x", "lines": [[["a", 0, 0], [1, 1]]] * 4}) is None
    assert parse_wire("nonsense") is None


def test_too_many_points_are_thinned_keeping_each_lines_ends():
    long = [[[math.cos(a) , math.sin(a), k / 10] for a in np.linspace(0, 6, 80)] for k in range(20)]
    w = parse_wire({"name": "coil", "lines": long})
    assert w.points <= MAX_WIRE_POINTS + len(w.lines)
    assert np.allclose(w.lines[0][0][:2], parse_wire({"name": "c", "lines": long[:1] * 3}).lines[0][0][:2], atol=1.0)


def test_turning_moves_near_points_with_the_finger():
    near = np.array([[0.0, -1.0, 0.0], [0.0, -1.0, 0.1]])  # towards the person (-y)
    a, _ = project(near, 0.0, 0.0, 1.0)
    b, _ = project(near, 0.3, 0.0, 1.0)
    assert b[0, 0] > a[0, 0] and a[0, 0] == MODEL_C[0]


def test_one_pinched_hand_turns_it_and_tilts_it_within_limits():
    g = showing()
    yaw, tilt = g.yaw, g.tilt
    g.update(2.0, [], [], (), [Pen((300.0, 200.0), True, 0)])
    g.update(2.05, [], [], (), [Pen((340.0, 230.0), True, 0)])
    assert g.yaw > yaw and g.tilt > tilt
    for k in range(200):
        g.update(3.0 + k * 0.05, [], [], (), [Pen((340.0, 230.0 + 20 * k), True, 0)])
    assert g.tilt <= TILT_RANGE[1]


def test_an_open_hand_does_nothing_and_a_tracker_jump_does_not_throw_it():
    g = showing()
    g.update(1.5, [], [], (), [])
    yaw = g.yaw
    g.update(2.0, [], [], (), [Pen((300.0, 200.0), False, 0)])  # moving, fingers open: only the idle spin
    assert abs(g.yaw - yaw) < 0.1
    g.update(2.1, [], [], (), [Pen((100.0, 100.0), True, 0)])
    yaw = g.yaw
    g.update(2.15, [], [], (), [Pen((500.0, 300.0), True, 0)])  # 400 px in a frame
    assert g.yaw == yaw


def test_two_pinched_hands_zoom_it_within_limits():
    g = showing()
    g.update(2.0, [], [], (), [Pen((300.0, 200.0), True, 0), Pen((400.0, 200.0), True, 1)])
    assert g.zoom == 1.0
    g.update(2.05, [], [], (), [Pen((250.0, 200.0), True, 0), Pen((450.0, 200.0), True, 1)])
    assert g.zoom == 2.0
    g.update(2.1, [], [], (), [Pen((100.0, 200.0), True, 0), Pen((600.0, 200.0), True, 1)])
    assert g.zoom == ZOOM_RANGE[1]


def test_left_alone_it_settles_into_a_slow_turn():
    g = showing()
    g.update(2.0, [], [], (), [Pen((300.0, 200.0), True, 0)])
    g.update(2.05, [], [], (), [Pen((360.0, 200.0), True, 0)])
    assert g.spin > IDLE_SPIN
    t = 2.05
    for _ in range(200):
        t += 0.05
        g.update(t, [], [], (), [])
    assert abs(g.spin - IDLE_SPIN) < 0.05


def test_the_mic_side_is_no_handle():
    g = showing()
    yaw = g.yaw
    g.update(2.0, [], [], (), [Pen((STRIP_X + 40, 200.0), True, 0)])
    g.update(2.05, [], [], (), [Pen((STRIP_X + 80, 240.0), True, 0)])
    assert g._drag is None and abs(g.yaw - yaw) < 0.2


def test_it_draws_as_lines_inside_the_work_area_near_and_far_apart():
    g = showing()
    c = Canvas(Geo(np.eye(3), DESIGN_W, DESIGN_H))
    g.draw(c, 2.0)
    wires = [p for p in c.prims if p.id.startswith("hp-w")]
    assert len(wires) == len(g.wire.lines) and {p.role for p in wires} == {"source", "done"}
    assert max(q[0] for p in wires for q in p.pts) < STRIP_X
    assert "PINCH" in g.hud(2.0, 0)[1]


def test_a_failed_model_says_so_and_leaves_the_tile_ready():
    g = Helper(0.0, random.Random(1))
    g.show_model("cube", 0.0)
    ask, g.want = g.want, None
    g.answered(ask, None, "boom", 1.0)
    assert g.state == "idle" and "couldn't build a cube" in g.say[-1]
    g.show_model("cube", 2.0)
    ask, g.want = g.want, None
    g.answered(ask, {"name": "x", "lines": []}, None, 3.0)
    assert g.state == "idle"


CUBE_PROGRAM = {"name": "Cube", "code": """
for a in (0, 2):
    line([(0, 0, a), (2, 0, a), (2, 2, a), (0, 2, a), (0, 0, a)])
for x, y in ((0, 0), (2, 0), (2, 2), (0, 2)):
    line([(x, y, 0), (x, y, 2)])
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
    assert len(brain.calls) == 1 and brain.calls[0].jpegs == [] and "cube" in brain.calls[0].prompt
    kept = json.loads((tmp_path / "play-models" / "wire-cube.json").read_text())
    assert kept["name"] == "Cube" and len(kept["lines"]) == 6
    assert g.wire_src.startswith("/pictures/") and (tmp_path / "pictures" / g.wire_src.rsplit("/", 1)[1]).is_file()
    c = Canvas(Geo(np.eye(3), DESIGN_W, DESIGN_H))
    g.draw(c, rig.t)
    served = next(p for p in c.prims if p.kind == "wire")  # the page draws it: no line of it in the scene
    assert served.src == g.wire_src and not [p for p in c.prims if p.id.startswith("hp-w") and p.kind == "poly"]
    rig.ts.hear("stop", rig.t)
    assert g.state == "idle" and g.wire is None
    rig.ts.hear("show me a cube", rig.t)
    until(rig, table(), lambda: g.state == "model", 1.0)
    assert len(brain.calls) == 1  # no second call


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


def test_a_tower_is_first_seen_from_nearly_level_and_a_wide_thing_from_above():
    g = Helper(0.0, random.Random(1))
    g.show_model("mast", 0.0)
    ask, g.want = g.want, None
    g.answered(ask, {"name": "Mast", "lines": [[[0, 0, 0], [0, 0, 9]], [[0, 0, 9], [0.1, 0, 0]], [[1, 0, 0], [0, 1, 0]]]}, None, 1.0)
    assert g.tilt == TILT_TALL
    g = showing()
    g2 = Helper(0.0, random.Random(1))
    g2.show_model("plate", 0.0)
    ask, g2.want = g2.want, None
    g2.answered(ask, {"name": "Plate", "lines": [[[0, 0, 0], [9, 0, 0]], [[9, 0, 0], [9, 9, 0]], [[9, 9, 0], [0, 9, 0.1]]]}, None, 1.0)
    assert g2.tilt == TILT_WIDE


def test_holding_the_mic_still_works_while_a_model_shows():
    g = showing()
    for k in range(10):
        g.update(2.0 + k * 0.05, [MIC_C], [MIC_C], (), [])
    assert g.mic_open and g.state == "model"
