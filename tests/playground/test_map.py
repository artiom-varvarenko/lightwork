"""Exploring: "take me to Paris" flies a 3D map to it; hands drag, turn and zoom it. The page draws it; the scene says where."""

from __future__ import annotations

import math
import random

import numpy as np

from lightwork.contracts import render
from lightwork.games import Pen
from lightwork.helper import (
    FLY_S,
    MAP_PITCH,
    MAP_RECT,
    PLACE_ZOOM,
    STRIP_X,
    Helper,
    _from_world,
    _to_world,
    asked_to_go,
    parse_place,
)
from lightwork.playground import DESIGN_H, DESIGN_W, Canvas, Geo
from lightwork.vision.overlay_mask import rasterize

from .test_helper import FakeBrain, FakeGate, until
from .test_table_scans import Rig
from .test_tabletop import table

PARIS = {"name": "Paris", "lat": 48.8584, "lon": 2.2945, "zoom": 15, "bearing": 300, "say": "The Eiffel Tower is 330 metres tall."}


def flown() -> Helper:
    g = Helper(0.0, random.Random(1))
    g.go_to("paris", 0.0)
    ask, g.want = g.want, None
    g.answered(ask, PARIS, None, 1.0)
    return g


def after_landing(g: Helper) -> float:
    t = 1.0 + FLY_S + 0.1
    g.update(t, [], [], (), [])
    return t


def test_what_asks_for_a_place():
    assert asked_to_go("Take me to Paris.") == "paris"
    assert asked_to_go("Can you take me to the Eiffel Tower?") == "eiffel tower"
    assert asked_to_go("let's go to Mount Fuji") == "mount fuji"
    for not_one in ("what is Paris", "take me", "take me to a place far far away from all of this right now", "go to the next step"):
        assert asked_to_go(not_one) is None, not_one


def test_a_place_must_be_on_the_map():
    p = parse_place(PARIS)
    assert (p.name, p.lon, p.lat, p.zoom, p.bearing) == ("Paris", 2.2945, 48.8584, 15.0, 300.0)
    assert parse_place({**PARIS, "lat": 95}) is None
    assert parse_place({**PARIS, "lon": "east"}) is None
    assert parse_place({"name": "x"}) is None
    assert parse_place(None) is None
    assert parse_place({**PARIS, "zoom": 99}).zoom == PLACE_ZOOM[1]
    assert parse_place({**PARIS, "bearing": 400}).bearing == 40.0


def test_arriving_flies_there_says_something_and_the_map_keeps_out_of_the_controls():
    g = flown()
    assert g.state == "map" and g.flight == 1 and g.say[-1] == PARIS["say"]
    c = Canvas(Geo(np.eye(3), DESIGN_W, DESIGN_H))
    g.draw(c, 2.0)
    m = next(p for p in c.prims if p.kind == "map")
    assert m.center == [2.2945, 48.8584] and m.pitch == MAP_PITCH and m.bearing == 300.0 and m.fly_ms == FLY_S * 1000
    assert max(q[0] for q in m.corners) <= STRIP_X and min(q[1] for q in m.corners) == MAP_RECT[1]
    assert "OPENFREEMAP" in g.hud(2.0, 0)[0]  # the attribution, also large enough to read on the table


def test_hands_do_not_stop_the_flight():
    g = flown()
    at = g.map_at
    g.update(2.0, [], [], (), [Pen((300.0, 200.0), True, 0)])
    g.update(2.1, [], [], (), [Pen((400.0, 200.0), True, 0)])
    assert g.map_at == at and g.map_bearing == 300.0


def test_a_drag_moves_the_ground_with_the_finger():
    g = flown()
    t = after_landing(g)
    g.map_bearing = 0.0
    lon, lat = g.map_at
    g.update(t + 0.05, [], [], (), [Pen((300.0, 200.0), True, 0)])
    g.update(t + 0.10, [], [], (), [Pen((340.0, 200.0), True, 0)])  # to the right: what lay left of the middle comes there
    assert g.map_at[0] < lon and abs(g.map_at[1] - lat) < 1e-6
    lon, lat = g.map_at
    g.update(t + 0.15, [], [], (), [Pen((340.0, 240.0), True, 0)])  # towards the person: what lay north comes nearer
    assert g.map_at[1] > lat and abs(g.map_at[0] - lon) < 1e-6
    g.map_bearing = 90.0  # facing east, a finger going right brings up what lies south of the middle
    lon, lat = g.map_at
    g.update(t + 0.20, [], [], (), [Pen((380.0, 240.0), True, 0)])
    assert g.map_at[1] > lat


def test_two_hands_zoom_and_turn_it_with_them():
    g = flown()
    t = after_landing(g)
    bearing, zoom = g.map_bearing, g.map_zoom
    g.update(t + 0.05, [], [], (), [Pen((300.0, 200.0), True, 0), Pen((400.0, 200.0), True, 1)])
    g.update(t + 0.10, [], [], (), [Pen((250.0, 200.0), True, 0), Pen((450.0, 200.0), True, 1)])
    assert math.isclose(g.map_zoom, zoom + 1.0) and math.isclose(g.map_bearing, bearing)
    g.update(t + 0.15, [], [], (), [Pen((350.0, 100.0), True, 0), Pen((350.0, 300.0), True, 1)])  # a quarter turn clockwise
    assert math.isclose(g.map_bearing, (bearing - 90) % 360)


def test_left_alone_the_view_slowly_turns():
    g = flown()
    t = after_landing(g)
    bearing = g.map_bearing
    for k in range(20):
        g.update(t + k * 0.05, [], [], (), [])
    assert g.map_bearing != bearing and g.map_at == (PARIS["lon"], PARIS["lat"])


def test_the_world_coordinates_go_both_ways():
    for lon, lat in ((2.2945, 48.8584), (-122.4, 37.8), (139.7, -33.9)):
        back = _from_world(*_to_world(lon, lat))
        assert math.isclose(back[0], lon, abs_tol=1e-9) and math.isclose(back[1], lat, abs_tol=1e-9)


def test_the_prim_is_masked_and_a_failed_place_says_so():
    prim = render.MapView(id="m", corners=[[100, 100], [300, 100], [300, 200], [100, 200]], center=[2.0, 48.0], zoom=15, bearing=0, pitch=60)
    assert rasterize([prim], np.eye(3), (750, 420), (750, 420))[150, 200] == 255
    g = Helper(0.0, random.Random(1))
    g.go_to("atlantis", 0.0)
    ask, g.want = g.want, None
    g.answered(ask, {"name": "Atlantis", "lat": 200, "lon": 0, "zoom": 3, "bearing": 0, "say": ""}, None, 1.0)
    assert g.state == "idle" and "couldn't find atlantis" in g.say[-1]


def test_saying_it_needs_no_look_and_a_retake_comes_from_the_cache(tmp_path):
    brain = FakeBrain(place=PARIS)
    rig = Rig(game="helper", voice=True, brain=brain, state_dir=tmp_path, mic_gate=FakeGate())
    rig.run(0.5, table())
    g = rig.ts.games.game
    rig.ts.hear("Take me to Paris.", rig.t)
    assert g.state == "placing" and rig.ts._look is None
    until(rig, table(), lambda: g.state == "map")
    assert rig.ts._look is None and brain.calls[0].role == "helper-place" and brain.calls[0].jpegs == []
    assert (tmp_path / "play-models" / "place-paris.json").is_file()
    prims = rig.ts.scene(rig.t)
    assert [p for p in prims if p.kind == "map"]
    rig.ts.hear("stop", rig.t)
    assert g.state == "idle" and g.map_place is None
    rig.ts.hear("take me to paris", rig.t)
    until(rig, table(), lambda: g.state == "map", 1.0)
    assert len(brain.calls) == 1
