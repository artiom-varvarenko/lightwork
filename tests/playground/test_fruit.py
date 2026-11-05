"""Slice the fruit (lightwork.fruit): fruit tossed up the table, cut by a quick swipe of a fingertip."""

from __future__ import annotations

import math
import random

import numpy as np
import pytest

from lightwork.fruit import HIT_PAD, KINDS, MIN_SPEED, ROUND_S, SliceFruit, clip_closed, split
from lightwork.games import DWELL_S, GameHost, Menu, Pen
from lightwork.helper import asked_for_game
from lightwork.playground import DESIGN_H, DESIGN_W, Canvas, Geo

DT = 1 / 30


def area(pts) -> float:
    return 0.5 * abs(sum(a[0] * b[1] - b[0] * a[1] for a, b in zip(pts, pts[1:] + pts[:1], strict=True)))


def playing(seed: int = 1) -> tuple[SliceFruit, float]:
    """A game past its 3-2-1 that throws nothing by itself."""
    g = SliceFruit(0.0, random.Random(seed))
    g.next_volley = math.inf
    t = g.READY_S + 0.05
    g.update(t, [], [])
    assert g.said_go and not g.fruit
    return g, t


def run(g, t0: float, t1: float, pens=lambda t: ()) -> float:
    """Frames from ``t0`` to ``t1``; ``pens(t)``: the hands then."""
    t = t0
    while t < t1 - 1e-9:
        t += DT
        g.update(t, [], [], (), list(pens(t)))
    return t


def swipe(a, b, t0: float, seconds: float, hand: int = 0):
    """A fingertip from ``a`` to ``b`` over ``seconds`` from ``t0``, held at ``b`` after."""
    def pens(t):
        f = min(1.0, max(0.0, (t - t0) / seconds))
        return [Pen((a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f), False, hand)]
    return pens


def test_every_fruit_cuts_into_two_halves_that_make_the_whole():
    for kind in KINDS.values():
        body = kind.parts[0].pts
        for ang in np.linspace(0, math.pi, 7):
            n = (math.cos(ang), math.sin(ang))
            one, two = split(kind, (0.1, -0.05), n)
            assert area(list(one[0].pts)) + area(list(two[0].pts)) == pytest.approx(area(list(body)), rel=1e-9)
            for half, s in ((one, 1), (two, -1)):  # each on its own side of the cut
                assert all(s * ((x - 0.1) * n[0] + (y + 0.05) * n[1]) >= -1e-9 for p in half for x, y in p.pts), kind
                flesh = half[-1]
                assert flesh.role == kind.flesh and not flesh.closed and len(flesh.pts) == 2
    assert clip_closed(((0, 0), (1, 0), (1, 1)), (5, 0), (1, 0)) == ()  # all on the other side: nothing


def test_a_tossed_fruit_comes_up_from_the_players_edge_and_falls_back():
    g, t = playing()
    f = g.toss(KINDS["apple"], t, x0=300, x1=360, top=150, hang=1.2)
    assert f.at(t)[1] > DESIGN_H and "toss" in g.sfx  # thrown from beyond the near edge
    ys = []
    while f in g.fruit:
        t = run(g, t, t + DT)
        ys.append(f.at(t)[1])
    assert min(ys) == pytest.approx(150, abs=3) and ys[-1] > DESIGN_H
    assert 2.3 < t - g.READY_S < 2.6 and g.missed == 1 and g.cut == 0


def test_a_quick_swipe_through_a_fruit_cuts_it_in_two():
    g, t = playing()
    f = g.toss(KINDS["watermelon"], t, x0=400, x1=400, top=200, hang=1.0)
    t = run(g, t, t + 1.0)  # at the top
    at = f.at(t)
    t = run(g, t, t + 0.4, swipe((at[0] - 150, at[1] + 20), (at[0] + 150, at[1] - 10), t, 0.25))
    assert f not in g.fruit and g.cut == 1 and g.score == 1 and g.missed == 0
    assert sorted(h.id for h in g.halves) == [f"{f.id}a", f"{f.id}b"]
    assert "slice" in g.sfx and any(line.startswith("slice: cut a watermelon at") for line in g.log)


def test_a_swipe_at_where_the_fruit_was_shown_a_moment_ago_still_cuts_it():
    """The projector shows the fruit late and the camera sees the hand late: a swipe aimed at a fast fruit lands behind
    it."""
    g, t = playing()
    f = g.toss(KINDS["apple"], t, x0=400, x1=400, top=150, hang=1.0)
    t = run(g, t, t + 0.3)  # on its way up, fast
    seen = f.at(t - 0.1)
    # where the fruit really is while the swipe goes by: too far for the swipe line, without the allowance
    assert all(abs(f.at(t + k * DT)[1] - seen[1]) > f.kind.r + HIT_PAD for k in (1, 2, 3))
    t = run(g, t, t + 0.1, swipe((seen[0] - 120, seen[1]), (seen[0] + 120, seen[1]), t, 0.1))
    assert g.cut == 1


def test_a_hand_resting_where_the_fruit_flies_never_cuts_it_and_says_to_swipe():
    g, t = playing()
    f = g.toss(KINDS["apple"], t, x0=400, x1=400, top=200, hang=1.0)
    held = [Pen(f.at(t + 1.0), False, 0)]
    t = run(g, t, t + 1.1, lambda _: held)
    assert f in g.fruit and g.cut == 0
    assert g.hud(t, 0)[1] == "SWIPE FASTER: A QUICK FLICK CUTS"
    assert any("too slow to cut" in line for line in g.log)


def test_a_slow_hand_does_not_cut():
    g, t = playing()
    f = g.toss(KINDS["lemon"], t, x0=400, x1=400, top=200, hang=1.0)
    t = run(g, t, t + 0.9)
    at = f.at(t + 0.1)
    speed = 0.6 * MIN_SPEED
    t = run(g, t, t + 0.5, swipe((at[0] - speed * 0.25, at[1]), (at[0] + speed * 0.25, at[1]), t, 0.5))
    assert g.cut == 0


def test_a_hand_lost_mid_swipe_and_found_again_as_the_other_hand_still_cuts():
    """A fast hand blurs: the tracker loses it, then finds it again further on, often in its other slot."""
    g, t = playing()
    f = g.toss(KINDS["strawberry"], t, x0=400, x1=400, top=200, hang=1.0)
    t = run(g, t, t + 0.9)
    at = f.at(t + 0.2)
    t = run(g, t, t + 0.1, swipe((at[0] - 200, at[1]), (at[0] - 120, at[1]), t, 0.1))  # hand 0, fast
    frozen = Pen((at[0] - 120, at[1]), False, 0)  # the table holds its last place a moment ...
    t = run(g, t, t + 0.1, lambda _: [frozen])
    t = run(g, t, t + DT, lambda _: [frozen, Pen((at[0] + 120, at[1]), False, 1)])  # ... found again as hand 1
    assert f not in g.fruit and g.cut == 1
    assert any("found again as hand 1" in line for line in g.log)


def test_a_still_hand_is_never_joined_to_a_new_one():
    g, t = playing()
    f = g.toss(KINDS["apple"], t, x0=400, x1=400, top=200, hang=1.0)
    t = run(g, t, t + 0.9)
    at = f.at(t + 0.1)
    rest = Pen((at[0] - 120, at[1]), False, 0)
    t = run(g, t, t + 0.1, lambda _: [rest])  # a hand resting left of the fruit ...
    t = run(g, t, t + 0.1, lambda _: [rest, Pen((at[0] + 120, at[1]), False, 1)])  # ... and another comes in
    assert f in g.fruit and g.cut == 0


def test_one_swipe_through_three_fruit_scores_more_for_each():
    g, t = playing()
    row = [g.toss(KINDS["apple"], t, x0=x, x1=x, top=200, hang=1.0) for x in (250, 400, 550)]
    t = run(g, t, t + 1.0)
    t = run(g, t, t + 0.5, swipe((150, 200), (650, 200), t, 0.3))
    assert all(f not in g.fruit for f in row)
    assert g.cut == 3 and g.score == 1 + 2 + 3 and g.best_swipe == 3
    c = Canvas(Geo(np.eye(3), DESIGN_W, DESIGN_H))
    g.draw(c, t - 0.3)
    assert {p.text for p in c.prims if p.kind == "label"} >= {"+2", "+3"}
    assert g.hud(t - 0.3, 0)[0] == "SCORE 6 · 3 IN ONE SWIPE!"


def test_it_throws_by_itself_more_as_the_round_goes_on():
    g = SliceFruit(0.0, random.Random(4))
    thrown = []
    t = 0.0
    while t < g.READY_S + ROUND_S - 0.1:
        t += DT
        g.update(t, [], [])
        thrown += [t] * g.sfx.count("toss")
        g.sfx = []
        assert len(g.fruit) <= 6
    first = sum(1 for x in thrown if x < g.READY_S + 20)
    last = sum(1 for x in thrown if x > g.READY_S + ROUND_S - 20)
    assert thrown and min(thrown) >= g.READY_S and first >= 8 and last > 1.4 * first
    assert g.missed > 0 and not g.over


def test_every_moment_draws_unique_prim_ids_on_the_page():
    geo = Geo(np.eye(3), DESIGN_W, DESIGN_H)
    g = SliceFruit(0.0, random.Random(7))
    t = 0.0
    k = 0
    while t < g.READY_S + 12:
        t += DT
        k += 1
        x = 100 + (k * 37) % 560  # a hand sweeping back and forth, fast
        g.update(t, [], [], (), [Pen((x, 160 + 60 * math.sin(k / 5)), False, 0)])
        c = Canvas(geo)
        g.draw(c, t)
        ids = [p.id for p in c.prims]
        assert len(ids) == len(set(ids)), t
        for p in c.prims:
            if p.kind == "label":
                assert 0 <= p.at[0] and p.at[0] + 14.8 * len(p.text) <= DESIGN_W + 1, p
        for line in g.hud(t, 0)[:2]:
            assert len(line) <= 44
    assert g.cut > 0


def test_the_round_ends_keeps_the_best_and_plays_again(tmp_path):
    host = GameHost(state_dir=tmp_path, seed=2)
    host.start("slice", 0.0)
    g = host.game
    assert isinstance(g, SliceFruit)
    g.score = 7  # as if sliced
    t = 0.0
    said = []
    while t < g.READY_S + ROUND_S + 0.2:
        t += DT
        said += host.update(t, [], [])[0]
    assert g.over and not g.fruit and host.best["slice"] == 7 and said[-1].startswith("Time!")
    assert host.hud(t)[0] == "TIME! SCORE 7 · NEW BEST!"
    assert any(line.startswith("slice: round over") for line in g.log)
    again = host.again
    while t < g.READY_S + ROUND_S + 3.0:
        t += DT
        host.update(t, [again.c], [])
    assert isinstance(host.game, SliceFruit) and host.game is not g and not host.game.over


def test_the_menu_starts_it_and_so_do_its_names(tmp_path):
    host = GameHost(state_dir=tmp_path, seed=1)
    host.start("menu", 0.0)
    b = next(b for b in host.game.bubbles if b.key == "slice")
    t, said = 0.0, []
    while t < DWELL_S + 0.3:
        t += 0.05
        said += host.update(t, [b.c], [])[0]
    assert isinstance(host.game, SliceFruit) and said[-1].startswith("Slice the fruit!")
    host.start("menu", t)
    assert isinstance(host.game, Menu)
    assert host.hear("fruit ninja", t)[0].startswith("Slice the fruit!")
    host.start("draw", t)
    host.game.guess = "BALL"
    host.hear("No, it's a fruit.", t)  # an answer in draw and guess, not a game
    assert host.game.key == "draw"
    for said, key in (("let's play slice the fruit", "slice"), ("Start fruit ninja", "slice"),
                      ("play the fruit game", "slice"), ("let's play the slicing game", "slice"),
                      ("slice it open", None), ("cut the onion into slices", None), ("what fruit is this?", None)):
        assert asked_for_game(said) == key, said


def test_on_the_table_a_swiping_hand_cuts_a_fruit(tmp_path):
    """Through the table session: camera landmarks, the hand slots, the steadied fingertip, the page."""
    from lightwork.vision.hands import INDEX_TIP

    from .test_table_scans import Rig
    from .test_tabletop import GEO, pose, table

    rig = Rig(game="slice", state_dir=tmp_path)
    rig.run(0.2, table())
    g = rig.ts.games.game
    assert isinstance(g, SliceFruit)
    g.next_volley = math.inf
    rig.run(g.READY_S - 0.1, table())
    assert g.said_go
    f = g.toss(KINDS["watermelon"], rig.t, x0=400, x1=400, top=210, hang=1.0)
    rig.run(0.9, table())
    at = f.at(rig.t + 0.15)

    def hand(t, t0=rig.t):  # a hand, fingers toward the far edge, its index fingertip swiping across
        x = at[0] - 220 + 440 * min(1.0, (t - t0) / 0.35)
        tip = GEO.d2c([(x, at[1])])[0]
        p = pose((tip[0], tip[1] + 160), (tip[0], tip[1] + 70))
        shift = tip - p.img[INDEX_TIP]
        return [pose((tip[0] + shift[0], tip[1] + 160 + shift[1]), (tip[0] + shift[0], tip[1] + 70 + shift[1]))]

    rig.run(0.5, table(), poses=hand)
    assert f not in g.fruit and g.cut == 1
    assert any(line.startswith("slice: cut a watermelon") for line in rig.ts.log)  # the session prints them
    ids = {p.id for p in rig.page.history[-1][2]}
    assert f"{f.id}a-0" in ids and f"{f.id}b-0" in ids
    assert rig.ts.scans == 0  # no object scans while it plays
