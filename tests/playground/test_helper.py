"""General helper tile (lightwork.helper): what was said, the mic, the questions, the steps; then on a synthetic table."""

from __future__ import annotations

import random
import threading
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from lightwork.games import DWELL_S, Bubble, CatchLight, GameHost, Menu, RockPaperScissors
from lightwork.helper import (
    ADVANCE_S,
    BACK_C,
    CHECK_AFTER_S,
    LINE_H,
    MAX_QUESTIONS,
    MIC_C,
    MIC_TAIL_S,
    NEXT_C,
    NO_WORDS_S,
    TEXT_W,
    UTTER_END_S,
    Ask,
    Helper,
    asked_for_game,
    box_of,
    free_spot,
    parse_job,
)
from lightwork.playground import DESIGN_H, DESIGN_W, Canvas, Geo
from lightwork.tabletop import LOOK_SCALE, PARTIAL_AFTER_S, snap_box
from lightwork.vision.hands import HandPose
from lightwork.voice.capture import MicGate

from .test_table_scans import Rig
from .test_tabletop import GEO, HI, page_rect, pose, put, table

SEEN = {"game": "none", "job": "cottage cheese pancakes", "manual": "", "question": "How many grams of cottage cheese?",
        "say": "I see a bowl of cottage cheese and three eggs.",
        "things": [{"name": "bowl of cottage cheese", "x_min": 200, "y_min": 300, "x_max": 400, "y_max": 700},
                   {"name": "egg", "x_min": 500, "y_min": 400, "x_max": 560, "y_max": 500},
                   {"name": "egg", "x_min": 600, "y_min": 450, "x_max": 660, "y_max": 550},
                   {"name": "egg", "x_min": 540, "y_min": 600, "x_max": 600, "y_max": 700}]}
READY = {**SEEN, "question": "", "say": ""}
PLAN = {"title": "Syrniki", "steps": [
    {"kind": "do", "say": "Crack the eggs into the bowl", "parts": ["egg", "bowl of cottage cheese"], "count": 0,
     "seconds": 0, "check": "Are the eggs in the bowl?"},
    {"kind": "add", "say": "Add 3 spoons of flour", "parts": ["flour", "bowl of cottage cheese"], "count": 3,
     "seconds": 0, "check": ""},
    {"kind": "timer", "say": "Mix well", "parts": ["bowl of cottage cheese"], "count": 0, "seconds": 30, "check": ""},
    {"kind": "do", "say": "Fry on the stove", "parts": [], "count": 0, "seconds": 0, "check": ""},
]}


def tick(g, t0, t1, points=(), tips=(), dt=1 / 20):
    t = t0
    while t < t1 - 1e-9:
        t += dt
        g.update(t, list(points), list(tips))
    return t


def tile(**kw) -> Helper:
    return Helper(0.0, random.Random(0), **kw)


def asking() -> Helper:
    """A tile that heard a job and got the model's question."""
    g = tile()
    g.heard("Help me cook with this", 0.0)
    tick(g, 0.0, UTTER_END_S + 0.1)
    assert g.want.role == "see"
    ask, g.want = g.want, None
    g.answered(ask, SEEN, None, 2.0)
    return g


def stepping() -> Helper:
    """A tile on the first step of ``PLAN``."""
    g = asking()
    g.heard("about 250 grams", 2.0)
    tick(g, 2.0, 2.0 + UTTER_END_S + 0.1)
    ask, g.want = g.want, None
    g.answered(ask, READY, None, 4.0)
    plan, g.want = g.want, None
    assert plan.role == "plan"
    g.answered(plan, PLAN, None, 5.0)
    g.last = 5.0
    assert g.state == "steps" and g.k == 0
    return g


# ---- what was said


@pytest.mark.parametrize("said, tile_key", [
    ("Start Rock Paper Scissors", "rps"),
    ("Let's start the Catch Light game.", "catch"),
    ("let's start the catch the light game", "catch"),
    ("Play rock, paper, scissors!", "rps"),
    ("rock-paper-scissors please", "rps"),
    ("Start follow the lights", "follow"),
    ("let's play draw and guess", "draw"),
    ("I want to play the drawing game", "draw"),
    ("let's play catch", "catch"),
    ("let's play a game", "menu"),
    ("back to the menu", "menu"),
    ("show me the object board", "objects"),
    ("Help me follow the manual", None),  # no game: "follow" and "draw" are ordinary words at a bench
    ("help me draw a circuit on this board", None),
    ("I want to catch the drips with a tray", None),
    ("help me build this Lego", None),
    ("It's a rock", None),
    ("help me play chess", None),
    ("what can I cook with this?", None),
])
def test_games_start_by_name_and_ordinary_words_stay_a_job(said, tile_key):
    assert asked_for_game(said) == tile_key


def test_a_box_on_the_models_grid_is_a_box_in_design_px():
    assert box_of({"x_min": 100, "y_min": 500, "x_max": 200, "y_max": 1000}) == pytest.approx((75.0, 210.0, 150.0, 420.0))
    assert box_of({"x_min": 200, "y_min": 900, "x_max": 100, "y_max": 500}) == pytest.approx((75.0, 210.0, 150.0, 378.0))
    assert box_of({"x_min": 100, "y_min": 100, "x_max": 101, "y_max": 900}) is None  # no area
    assert box_of({"x_min": 0, "y_min": 0, "x_max": 1000, "y_max": 1000}) is None  # the table itself
    assert box_of({"x_min": 0}) is None


def test_a_model_plan_never_gets_a_step_that_cannot_finish():
    job = parse_job({"title": "X", "steps": [
        {"kind": "add", "say": "Add it", "count": 0}, {"kind": "timer", "say": "Wait", "seconds": 0},
        {"kind": "fly", "say": "Fly"}, {"kind": "add", "say": "Spoons", "count": 3, "seconds": 9, "check": "?"},
        {"kind": "do", "say": ""}]})
    assert [s.kind for s in job.steps] == ["do", "do", "do", "add"]
    assert parse_job({"steps": [{"say": "Put the one red brick on the plate."}]}).steps[0].say == "Put the red brick on the plate"
    assert job.steps[3].count == 3 and job.steps[3].check == "" and job.steps[3].seconds == 0
    assert parse_job({"title": "X", "steps": []}) is None and parse_job(None) is None
    assert len(parse_job({"steps": [{"say": f"s{k}"} for k in range(30)]}).steps) == 8


def test_the_text_block_goes_where_nothing_lies():
    obj = (72.0, 70.0, 400.0, 200.0)
    x, y = free_spot(300.0, 60.0, [obj])
    assert not (x < obj[2] and x + 300 > obj[0] and y < obj[3] and y + 60 > obj[1])


# ---- the mic


def test_the_mic_opens_only_while_a_fingertip_is_on_the_bubble():
    g = tile()
    t = tick(g, 0.0, 1.0, points=[MIC_C])  # a wrist or knuckle passing over it is no fingertip
    assert not g.mic_open
    t = tick(g, t, t + 0.1, points=[MIC_C], tips=[MIC_C])  # a fingertip passing over it
    assert not g.mic_open
    t = tick(g, t, t + 0.5, points=[MIC_C], tips=[MIC_C])
    assert g.mic_open and g.hush
    t = tick(g, t, t + 0.2)  # the tracker lost it for a moment
    assert g.mic_open
    tick(g, t, t + MIC_TAIL_S)
    assert not g.mic_open


def test_what_is_said_while_it_is_held_is_one_request():
    g = tile()
    t = tick(g, 0.0, 0.5, tips=[MIC_C])
    g.heard("Hey, I build Lego.", t)
    t = tick(g, t, t + 3.0, tips=[MIC_C])  # a long pause, still holding: not finished
    assert g.want is None
    g.heard("Help me build this together.", t)
    t = tick(g, t, t + MIC_TAIL_S + 0.1)  # let go: the last words may still be on their way
    assert g.want is None
    tick(g, t, t + UTTER_END_S)
    assert g.want.role == "see" and g.state == "seeing"
    assert "Hey, I build Lego. Help me build this together." in g.want.prompt


def test_letting_go_with_no_words_says_so_and_nothing_is_sent():
    g = tile()
    t = tick(g, 0.0, 0.5, tips=[MIC_C])
    tick(g, t, t + MIC_TAIL_S + NO_WORDS_S + 0.2)
    assert g.want is None and g.state == "idle" and "didn't catch" in g.say[-1]


def test_a_tile_that_cannot_hear_never_opens_the_mic_and_says_why():
    g = tile(listens=False)
    tick(g, 0.0, 1.0, tips=[MIC_C])
    assert not g.mic_open and "VOICE IS OFF" in g.hud(1.0, 0)[0]


def test_the_gate_passes_frames_only_while_open_and_hands_on_the_moment_before():
    got = []
    gate = MicGate(got.append, pre_roll=2)
    gate(b"a")
    gate(b"b")
    gate(b"c")
    assert got == []
    gate.set_open(True)
    assert got == [b"b", b"c"]  # the words started just before the finger landed
    gate(b"d")
    assert got == [b"b", b"c", b"d"]
    gate.set_open(False)
    gate(b"e")
    assert got == [b"b", b"c", b"d"] and not gate.is_open
    gate.set_open(True)
    assert got[-1] == b"e" and gate.opened == 2


# ---- it asks only what it cannot see


def test_it_says_what_it_sees_asks_its_question_and_plans_with_the_answer():
    g = asking()
    assert g.state == "asking" and g.question == "How many grams of cottage cheese?"
    assert g.say[-1] == "I see a bowl of cottage cheese and three eggs. How many grams of cottage cheese?"
    assert [t.name for t in g.things] == ["bowl of cottage cheese", "egg", "egg", "egg"]  # it counted them itself
    assert g.things[0].box == pytest.approx((150.0, 126.0, 300.0, 294.0))
    assert [t.uid for t in g.want_outlines] == [t.uid for t in g.things]
    g.heard("about 250 grams", 2.0)
    tick(g, 2.0, 2.0 + UTTER_END_S + 0.1)
    assert g.want.role == "see" and g.want.look  # it looks again: the answer may have been to put something down
    assert 'You asked: "How many grams of cottage cheese?" They answered: "about 250 grams"' in g.want.prompt
    ask, g.want = g.want, None
    g.answered(ask, READY, None, 4.0)
    assert g.state == "planning" and g.want.role == "plan" and not g.want.look
    assert "They answered: \"about 250 grams\"" in g.want.prompt
    assert g.say[-1] == "Okay, let me plan that."  # and not "I see ..." all over again
    assert "On the table: bowl of cottage cheese; egg (3 of them)." in g.want.prompt
    assert "The job: cottage cheese pancakes." in g.want.prompt  # the dish it named is the dish it plans


def test_it_stops_asking_after_a_few_questions():
    g = asking()
    t = 2.0
    for k in range(MAX_QUESTIONS):
        assert g.state == "asking"
        g.heard(f"answer {k}", t)
        t = tick(g, t, t + UTTER_END_S + 0.1)
        ask, g.want = g.want, None
        g.answered(ask, SEEN, None, t)
    assert g.state == "planning" and g.want.role == "plan"


def test_a_game_the_model_reads_out_of_what_was_said_starts():
    g = tile()
    g.heard("that game with the moving lights", 0.0)
    tick(g, 0.0, UTTER_END_S + 0.1)
    ask, g.want = g.want, None
    g.answered(ask, {**READY, "game": "catch"}, None, 3.0)
    assert g.start_game == "catch"


def test_a_failed_look_says_so_and_a_question_stays_open():
    g = tile()
    g.heard("help me", 0.0)
    tick(g, 0.0, UTTER_END_S + 0.1)
    ask, g.want = g.want, None
    g.answered(ask, None, "timeout", 3.0)
    assert g.state == "idle" and "couldn't work that out" in g.say[-1]
    g = asking()
    g.heard("250", 2.0)
    tick(g, 2.0, 2.0 + UTTER_END_S + 0.1)
    ask, g.want = g.want, None
    g.answered(ask, None, "timeout", 3.0)
    assert g.state == "asking" and g.question


def test_an_answer_to_an_earlier_request_is_dropped():
    g = asking()
    old = Ask("see", "", {}, epoch=g.epoch - 1)
    g.answered(old, {**READY, "game": "rps"}, None, 3.0)
    assert g.start_game is None and g.state == "asking"


def test_without_a_model_a_job_is_refused_but_nothing_breaks():
    g = tile(thinks=False)
    g.heard("help me build this", 0.0)
    tick(g, 0.0, UTTER_END_S + 0.1)
    assert g.want is None and g.state == "idle" and "only start games" in g.say[-1]


# ---- the steps


def test_steps_show_what_to_assemble_with_what_and_hold_next_moves_once():
    g = stepping()
    assert g.say[-1] == "Step 1. Crack the eggs into the bowl."
    assert [[t.name for t in part] for part in g.parts] == [["egg", "egg", "egg"], ["bowl of cottage cheese"]]
    assert g.want is None  # the first step uses what was just seen: no second look
    t = tick(g, 5.0, 5.0 + DWELL_S + 0.1, tips=[NEXT_C])
    assert g.k == 1 and g.want.role == "locate" and g.hush
    t = tick(g, t, t + 3.0, tips=[NEXT_C])  # the finger stays: it does not run through the steps
    assert g.k == 1
    t = tick(g, t, t + 0.2)
    t = tick(g, t, t + DWELL_S + 0.1, tips=[BACK_C])
    assert g.k == 0
    t = tick(g, t, t + 3.0, tips=[BACK_C])  # no BACK on the first step, and it stays there
    assert g.k == 0


def test_every_step_looks_again_because_things_move():
    g = stepping()
    t = tick(g, 5.0, 5.0 + DWELL_S + 0.1, tips=[NEXT_C])
    ask, g.want = g.want, None
    g.want_outlines = []  # the session took those
    assert ask.look and "flour" in ask.prompt
    bowl = next(t_ for t_ in g.things if "bowl" in t_.name)
    g.outlined(bowl.uid, [(150.0, 126.0), (300.0, 126.0), (300.0, 294.0), (150.0, 294.0)])
    assert bowl.outline is not None
    found = [{"name": "flour", "visible": True, "x_min": 700, "y_min": 200, "x_max": 800, "y_max": 400},
             {"name": "bowl of cottage cheese", "visible": True, "x_min": 205, "y_min": 305, "x_max": 405, "y_max": 705}]
    g.answered(ask, {"found": found}, None, t)
    assert [x.name for x in g.lit] == ["flour", "bowl of cottage cheese"]
    assert bowl.outline is not None and next(x for x in g.lit if "bowl" in x.name) is bowl  # a box varying a little
    assert [x.name for x in g.want_outlines] == ["flour"]
    moved = [{"name": "bowl of cottage cheese", "visible": True, "x_min": 600, "y_min": 305, "x_max": 800, "y_max": 705},
             {"name": "flour", "visible": False, "x_min": 0, "y_min": 0, "x_max": 0, "y_max": 0}]
    g.answered(Ask("locate", "", {}, epoch=g.epoch), {"found": moved}, None, t)
    assert bowl.outline is None and bowl.box[0] == pytest.approx(450.0)  # really moved: outlined again


def test_a_spoon_step_counts_trips_to_the_target():
    g = stepping()
    t = tick(g, 5.0, 5.0 + DWELL_S + 0.1, tips=[NEXT_C])
    g.want = None
    t = tick(g, t, t + 0.3)
    bowl = g.things[0]
    g.parts = [[bowl]]
    for n in (1, 2):
        t = tick(g, t, t + 0.5, points=[bowl.centre])
        assert g.trips == n and g.advance_at is None
        t = tick(g, t, t + 0.7, points=[(100.0, 400.0)])  # back to the flour
    t = tick(g, t, t + 0.5, points=[bowl.centre])
    assert g.trips == 3 and g.advance_at is not None and g.say[-1] == "3. That's all 3."
    tick(g, t, t + ADVANCE_S + 0.1)
    assert g.k == 2


def test_a_hand_lost_for_a_frame_is_not_a_second_trip():
    g = stepping()
    t = tick(g, 5.0, 5.0 + DWELL_S + 0.1, tips=[NEXT_C])
    g.want = None
    g.parts = [[g.things[0]]]
    c = g.things[0].centre
    t = tick(g, t, t + 0.3, points=[c])
    assert g.trips == 1
    for _ in range(5):
        t = tick(g, t, t + 0.1)
        t = tick(g, t, t + 0.3, points=[c])
    assert g.trips == 1


def test_a_timer_runs_and_moves_on_by_itself():
    g = stepping()
    t = tick(g, 5.0, 5.0 + DWELL_S + 0.1, tips=[NEXT_C])
    t = tick(g, t, t + 0.2)
    t = tick(g, t, t + DWELL_S + 0.1, tips=[NEXT_C])
    assert g.k == 2 and g.timer_end - t == pytest.approx(30.0, abs=0.3)
    t = tick(g, t, g.timer_end + 0.1)
    assert g.say[-1] == "Time." and g.k == 2
    tick(g, t, t + ADVANCE_S + 0.1)
    assert g.k == 3


def test_the_camera_checks_once_the_hands_are_away_and_a_yes_moves_on():
    g = stepping()
    t = tick(g, 5.0, 8.0)
    assert g.want is None  # no hand has been at work yet: nothing to look at
    t = tick(g, t, t + 1.0, points=[(200.0, 200.0)])
    t = tick(g, t, t + CHECK_AFTER_S - 0.3)
    assert g.want is None
    t = tick(g, t, t + 0.4)
    assert g.want.role == "check" and "Are the eggs in the bowl?" in g.want.prompt
    ask, g.want = g.want, None
    t = tick(g, t, t + 1.0)
    g.answered(ask, {"ok": False, "say": "I still see two whole eggs."}, None, t)
    assert g.k == 0 and g.reply == "I still see two whole eggs." and g.say[-1] == g.reply
    t = tick(g, t, t + 5.0)
    assert g.want is None  # no hand since: it does not look again by itself
    t = tick(g, t, t + 1.0, points=[(200.0, 200.0)])
    t = tick(g, t, t + CHECK_AFTER_S + 0.1)
    ask, g.want = g.want, None
    g.answered(ask, {"ok": True, "say": "Both eggs are in."}, None, t)
    assert g.k == 0 and g.reply == "Both eggs are in."
    tick(g, t, t + ADVANCE_S + 0.1)
    assert g.k == 1


def test_a_hand_on_the_controls_is_no_work_for_the_camera():
    g = stepping()
    t = tick(g, 5.0, 6.0, points=[NEXT_C])
    tick(g, t, t + CHECK_AFTER_S + 1.0)
    assert g.want is None


def test_commands_while_stepping():
    g = stepping()
    assert g.command("next", 6.0) and g.k == 1
    g.want = None
    assert g.command("back", 6.1) and g.k == 0
    g.want = None
    assert g.command("repeat", 6.2) and g.say[-1] == "Step 1. Crack the eggs into the bowl."
    assert g.command("check", 6.3) and g.want.role == "check"
    g.want = None
    g.command("stop", 6.4)
    assert g.state == "idle" and g.job is None
    assert not g.command("next", 6.5)


def test_a_question_during_a_step_is_answered_and_the_step_stays():
    g = stepping()
    g.heard("how long do I crack them for?", 5.0)
    t = tick(g, 5.0, 5.0 + UTTER_END_S + 0.1)
    assert g.want.role == "answer" and g.k == 0
    ask, g.want = g.want, None
    g.answered(ask, {"say": "Just until they are open."}, None, t)
    assert g.k == 0 and g.reply == "Just until they are open."
    g.heard("help me build something else", t)
    tick(g, t, t + UTTER_END_S + 0.1)
    assert g.state == "seeing" and g.want.role == "see" and g.job is None


def test_the_last_step_ends_the_job_and_the_mic_takes_the_next_one():
    g = stepping()
    g._go(3, 6.0)
    g.want = None
    assert g.k == 3
    t = tick(g, 6.0, 6.0 + DWELL_S + 0.1, tips=[NEXT_C])
    assert g.state == "done" and "done" in g.say[-1] and "NEXT JOB" in g.hud(t, 0)[1]
    g.heard("help me with something else", t)
    tick(g, t, t + UTTER_END_S + 0.1)
    assert g.state == "seeing"


def test_a_sam_outline_that_does_not_fit_the_box_is_not_used():
    g = asking()
    t = g.things[0]  # 150 x 168 design px at (150, 126)
    g.outlined(t.uid, [(600.0, 300.0), (640.0, 300.0), (640.0, 340.0)])  # far away
    assert t.outline is None
    g.outlined(t.uid, [(0.0, 60.0), (750.0, 60.0), (750.0, 420.0), (0.0, 420.0)])  # the whole table
    assert t.outline is None
    g.outlined(t.uid, None)
    assert t.outline is None
    g.outlined(t.uid, [(160.0, 130.0), (290.0, 130.0), (290.0, 290.0), (160.0, 290.0)])
    assert t.outline is not None


def test_every_state_draws_on_the_page_with_unique_prim_ids():
    geo = Geo(np.eye(3), DESIGN_W, DESIGN_H)

    def check(g, now):
        c = Canvas(geo)
        g.draw(c, now)
        ids = [p.id for p in c.prims]
        assert len(ids) == len(set(ids)), (g.state, ids)
        for p in c.prims:
            if p.kind == "label":
                assert 0 <= p.at[0] and p.at[0] + 14.8 * len(p.text) <= DESIGN_W + 1, p
                assert 0 <= p.at[1] <= DESIGN_H, p
        for line in g.hud(now, 0)[:2]:
            assert isinstance(line, str)
        return c.prims

    g = tile()
    assert any(p.id == "hp-l0" for p in check(g, 0.5))
    g.heard("help me", 0.0)
    tick(g, 0.0, UTTER_END_S + 0.1)
    check(g, 3.0)
    assert g.state == "seeing"
    g = asking()
    assert any(p.id.startswith("hp-t") for p in check(g, 3.0))
    t = tick(g, 3.0, 3.5, tips=[MIC_C])
    assert any(p.id.startswith("hp-mic-ring") for p in check(g, t))
    g = stepping()
    prims = check(g, 5.5)
    assert any(p.id == "hp-arrow" for p in prims) and any(p.id == "hp-k-0b" for p in prims)
    for k in (1, 2, 3):
        g._go(k, 6.0)
        check(g, 6.5)
    g._go(4, 7.0)
    assert g.state == "done"
    check(g, 7.5)


def test_a_models_box_snaps_to_the_object_the_picture_shows():
    view = np.full((840, 1500, 3), 225, np.uint8)
    cv2.circle(view, (900, 400), 50, (40, 110, 180), -1)  # an egg, with a design-px box of (425, 175, 475, 225)
    near = snap_box(view, (440.0, 190.0, 490.0, 235.0), 2.0)  # the model's box is 15 px off, as it often is
    assert near == pytest.approx((425.0, 175.0, 475.5, 225.5), abs=1.5)
    assert snap_box(view, (100.0, 100.0, 150.0, 150.0), 2.0) is None  # nothing there: the model's box stands
    white = np.full((840, 1500, 3), 225, np.uint8)
    cv2.circle(white, (900, 400), 50, (222, 224, 226), -1)  # white on white: the picture does not say
    assert snap_box(white, (440.0, 190.0, 490.0, 235.0), 2.0) is None
    big = np.full((840, 1500, 3), 225, np.uint8)
    cv2.rectangle(big, (700, 300), (1400, 700), (60, 60, 60), -1)  # reaches out of the search area: not this box's thing
    assert snap_box(big, (440.0, 190.0, 490.0, 235.0), 2.0) is None


def test_a_small_things_outline_is_not_refused_for_its_margin():
    """4 Nov: a 2x4 brick's good mask, pushed out by the page margin, was twice the model's tight box."""
    g = tile()
    g.heard("help", 0.0)
    tick(g, 0.0, UTTER_END_S + 0.1)
    ask, g.want = g.want, None
    g.answered(ask, {**READY, "things": [{"name": "red brick", "x_min": 368, "y_min": 365, "x_max": 420, "y_max": 400}]},
               None, 2.0)
    t = g.things[0]
    g.outlined(t.uid, [(265.0, 146.0), (308.0, 146.0), (308.0, 175.0), (265.0, 175.0)])
    assert t.outline is not None and t.box == pytest.approx((265.0, 146.0, 308.0, 175.0))


def test_the_text_keeps_clear_of_everything_on_the_table():
    g = stepping()
    g._layout()
    x, y = g._block
    lines = g._block_lines()
    w, h = TEXT_W * max(len(t) for t, _ in lines) + g._digit_w(), LINE_H * len(lines)
    for t in g.things:  # lit or not
        assert not (x < t.box[2] and x + w > t.box[0] and y < t.box[3] and y + h > t.box[1]), t.name


def test_the_text_keeps_clear_of_a_timers_ring():
    g = stepping()
    g._go(2, 6.0)
    g.want = None
    assert g.parts  # the timer goes around the bowl
    g._layout()
    cx, cy, r = g._ring()
    x, y = g._block
    lines = g._block_lines()
    w, h = TEXT_W * max(len(t) for t, _ in lines) + g._digit_w(), LINE_H * len(lines)
    assert x >= cx + r or x + w <= cx - r or y >= cy + r or y + h <= cy - r


def test_with_a_game_the_microphone_goes_through_the_gate_and_without_one_it_does_not(monkeypatch):
    """The wiring ``lightwork play`` does on the rig; the camera and the microphone are not there in a test."""
    import lightwork.voice as voice_pkg
    import lightwork.voice.capture as capture
    from lightwork.commands import play

    class FakeVoice:
        def __init__(self, settings, keywords, on_event, on_hud=None):
            self.keywords, self.pcm = keywords, []

        def start(self):
            pass

        def feed_pcm(self, frame):
            self.pcm.append(frame)

        def feed_held(self, frame):
            self.pcm.append(frame)

        def phone_quiet(self):
            return True

        def browser_active(self):
            return False

    class FakeMic:
        def __init__(self, sink, source, yield_to=None):
            self.sink = sink

        def start(self):
            return None

    monkeypatch.setattr(voice_pkg, "VoiceService", FakeVoice)
    monkeypatch.setattr(capture, "ServerMic", FakeMic)
    monkeypatch.setattr(capture, "find_source", lambda pattern: "alsa_input.C920")
    args = SimpleNamespace(table=True, voice=True, game="menu", mic="auto")
    voice, mic, gate = play._voice(args, None, None, None)
    assert isinstance(gate, MicGate) and "rock paper scissors" in voice.keywords
    mic.sink(b"table talk")
    assert voice.pcm == []
    gate.set_open(True)
    gate.set_open(False)
    voice.pcm.clear()
    mic.sink(b"more talk")
    assert voice.pcm == []
    voice, mic, gate = play._voice(SimpleNamespace(table=True, voice=True, game=None, mic="auto"), None, None, None)
    assert gate is None
    mic.sink(b"x")
    assert voice.pcm == [b"x"]
    assert play._voice(SimpleNamespace(table=True, voice=False, game="menu", mic="auto"), None, None, None) == (
        None, None, None)


def test_three_eggs_are_three_outlines_and_one_label():
    g = asking()
    geo = Geo(np.eye(3), DESIGN_W, DESIGN_H)
    c = Canvas(geo)
    g.draw(c, 3.0)
    assert sum(p.id.startswith("hp-t") and p.kind == "poly" for p in c.prims) == 4
    assert sorted(p.text for p in c.prims if p.id.endswith("-txt")) == ["3 EGGS", "BOWL OF COTTAGE CHEESE"]


def test_the_menu_has_the_tile_and_it_starts(tmp_path):
    host = GameHost(state_dir=tmp_path, seed=1, listens=True, thinks=True)
    host.start("menu", 0.0)
    assert isinstance(host.game, Menu)
    b: Bubble = host.game.bubbles[0]
    assert b.key == "helper"
    t = 0.0
    said = []
    while t < DWELL_S + 0.3:
        t += 0.05
        said += host.update(t, [b.c], [])[0]
    assert isinstance(host.game, Helper) and said[-1].startswith("Hold the microphone")


# ---- on the table: the session's look and calls


class FakeBrain:
    label = "fake brain"
    run_dir = None

    def __init__(self, **answers):
        self.answers = answers
        self.calls: list = []

    def check(self):
        return None

    def ask(self, role, prompt, jpegs=(), **kw):
        self.calls.append(SimpleNamespace(role=role, prompt=prompt, jpegs=list(jpegs), kw=kw))
        a = self.answers[role.removeprefix("helper-")]
        data = a.pop(0) if isinstance(a, list) else a
        if isinstance(data, Exception):
            raise data
        return SimpleNamespace(text="", data=data, provider="baseten", model="meta-models/muse-spark-open-1026",
                               ms=2100.0, cost_usd=0.0)

    def summary(self):
        return f"fake: {len(self.calls)}"

    def close(self):
        pass


class FakeOutliner:
    label, error = "fake sam", None

    def __init__(self):
        self.requests: list = []
        self.contours: dict = {}

    def request(self, img, objects):
        self.requests.append((img, objects))
        return True

    def result(self, key):
        if key in self.contours:
            return True, self.contours.pop(key)
        return False, None

    def summary(self):
        return "fake sam"


class FakeGate:
    def __init__(self):
        self.states: list[bool] = []
        self.recent = True

    def set_open(self, on):
        if not self.states or self.states[-1] != on:
            self.states.append(on)

    def recently_open(self, within_s):
        return self.recent


def finger(at_design):
    """A hand whose index fingertip is at a design-px spot, its wrist towards the person."""
    tip = GEO.d2c([at_design])[0]
    p = pose((tip[0], tip[1] + 110), (tip[0], tip[1]))
    return HandPose(p.img + (tip - p.img[8]), p.world, p.side, p.score)


def until(rig, scene, cond, seconds=6.0):
    end = rig.t + seconds
    while rig.t < end and not cond():
        rig.run(0.1, scene)
    assert cond(), f"state {getattr(rig.ts.games.game, 'state', '?')}: {rig.ts.log[-5:]}"


def test_a_transcript_out_of_silence_with_the_mic_shut_is_no_job(tmp_path):
    brain = FakeBrain()
    gate = FakeGate()
    gate.recent = False
    rig = Rig(game="helper", voice=True, brain=brain, state_dir=tmp_path, mic_gate=gate)
    rig.run(0.5, table())
    assert rig.ts.hear("by", rig.t) == []  # 4 Nov: the transcriber said this with nobody at the mic
    rig.run(3.0, table())
    g = rig.ts.games.game
    assert g.state == "idle" and g.want is None and not g.utter and brain.calls == []


def test_the_last_partial_stands_in_when_no_final_transcript_comes(tmp_path):
    brain = FakeBrain()
    rig = Rig(game="helper", voice=True, brain=brain, state_dir=tmp_path, mic_gate=FakeGate())
    rig.run(0.5, table())
    g = rig.ts.games.game
    rig.run(0.5, table(), poses=lambda t: [finger(MIC_C)])
    assert g.mic_open
    partial_t = rig.t
    assert rig.ts.heard_partly("start rock paper scissors", partial_t, rig.t) == []  # still holding it
    rig.run(1.0, table())
    assert not g.mic_open and g.released_t is not None
    assert rig.ts.heard_partly("start rock paper scissors", partial_t, rig.t) == []  # the final may still come
    assert rig.ts.heard_partly("from before this hold", g.opened_t - 1.0, rig.t + 5.0) == []
    rig.run(PARTIAL_AFTER_S, table())
    said = rig.ts.heard_partly("start rock paper scissors", partial_t, rig.t)
    assert isinstance(rig.ts.games.game, RockPaperScissors) and said
    assert rig.ts.hear("Start rock paper scissors.", rig.t) == []  # its final, late: no second start
    assert brain.calls == []


def test_saying_a_game_at_the_tile_starts_it_with_no_model_call_and_shuts_the_mic(tmp_path):
    brain = FakeBrain()
    gate = FakeGate()
    rig = Rig(game="helper", voice=True, brain=brain, state_dir=tmp_path, mic_gate=gate)
    rig.run(0.5, table())
    assert isinstance(rig.ts.games.game, Helper)
    rig.run(0.5, table(), poses=lambda t: [finger(MIC_C)])
    assert gate.states[-1] is True  # a fingertip on the bubble opens it
    said = rig.ts.hear("Let's start the Catch Light game.", rig.t)
    assert isinstance(rig.ts.games.game, CatchLight) and said[0].startswith("Catch the light")
    rig.run(0.3, table(), poses=lambda t: [finger(MIC_C)])
    assert gate.states[-1] is False  # no more listening in the game, finger or not
    assert brain.calls == []
    rig.ts.start_game("helper", rig.t)
    said = rig.ts.hear("Start Rock Paper Scissors", rig.t)
    assert isinstance(rig.ts.games.game, RockPaperScissors) and said[0].startswith("Rock, paper, scissors")
    assert brain.calls == []


def test_the_mic_is_shut_on_the_rest_of_the_table(tmp_path):
    gate = FakeGate()
    rig = Rig(game="menu", voice=True, state_dir=tmp_path, mic_gate=gate)
    rig.run(0.5, table(), poses=lambda t: [finger(MIC_C)])
    assert gate.states == [False]  # a finger there on the menu is nothing
    assert rig.ts.games.voice is False and rig.ts.dash.voice is False  # and no hint says "or say menu"


def test_a_job_on_the_table_looks_with_our_light_off_and_highlights_what_the_model_found(tmp_path):
    bowl = page_rect(250, 210, 150, 170, 0)
    scene = put(table(), bowl, (40, 60, 200))  # a red bowl, BGR
    brain = FakeBrain(see=[SEEN, READY], plan=PLAN, locate={"found": []})
    sam = FakeOutliner()
    rig = Rig(game="helper", voice=True, brain=brain, outliner=sam, state_dir=tmp_path, mic_gate=FakeGate())
    rig.run(0.5, scene)
    g = rig.ts.games.game
    lit_before = np.array(rig.page.seen(rig.t), dtype=object)
    assert len(lit_before) > 0  # the mic and the status bar are on the table
    assert rig.ts.hear("Help me cook with this", rig.t) == []
    until(rig, scene, lambda: bool(brain.calls))
    assert any(prims == [] for _, _, prims in rig.page.history)  # everything went dark for the picture
    call = brain.calls[0]
    assert call.role == "helper-see" and "Help me cook with this" in call.prompt and call.kw["schema"]["type"] == "object"

    # the picture is the page, rectified: the bowl lies where design px say, in its own colour again
    view = cv2.imdecode(np.frombuffer(call.jpegs[0], np.uint8), cv2.IMREAD_COLOR)
    assert view.shape[:2] == (int(DESIGN_H * LOOK_SCALE), int(DESIGN_W * LOOK_SCALE))
    red = (view[..., 2].astype(int) - view[..., 1].astype(int)) > 90
    ys, xs = np.nonzero(red)
    cx, cy = GEO.p2d((250.0, 210.0))
    assert xs.mean() / LOOK_SCALE == pytest.approx(cx, abs=5.0) and ys.mean() / LOOK_SCALE == pytest.approx(cy, abs=5.0)
    table_px = view[int(0.8 * view.shape[0]), int(0.7 * view.shape[1])]
    assert table_px.max() - table_px.min() < 45  # the table white, not orange

    until(rig, scene, lambda: g.state == "asking")
    assert rig.said[-1].endswith("How many grams of cottage cheese?")
    assert "BY MUSE-SPARK-OPEN-1026 @ BASETEN" in rig.ts.dash.note
    assert all(t.asked for t in g.things)
    assert sam.requests and len(sam.requests[-1][1]) == 4  # SAM is asked for each thing, with grown boxes
    img, objects = sam.requests[-1]
    assert img.shape[:2] == (rig.ts.frame_size[1], rig.ts.frame_size[0])
    uid, (x0, y0, x1, y1) = objects[0]
    assert uid == g.things[0].uid
    box = GEO.d2c([g.things[0].box[:2], g.things[0].box[2:]])
    assert x0 < box[0][0] and y0 < box[0][1] and x1 > box[1][0] and y1 > box[1][1]
    box_before = g.things[0].box
    sam.contours[uid] = apply_page_rect(page_rect(225, 210, 140, 170, 0))
    until(rig, scene, lambda: g.things[0].outline is not None)
    assert any(p.id == f"hp-t{uid}-s" for p in rig.ts.scene(rig.t))
    assert g.things[0].box != box_before  # the box is now where SAM says, not where the model guessed
    assert (g.things[0].box[0] + g.things[0].box[2]) / 2 == pytest.approx(GEO.p2d((225.0, 210.0))[0], abs=3.0)

    assert rig.ts.hear("About 250 grams.", rig.t) == []
    until(rig, scene, lambda: g.state == "steps")
    assert [c.role for c in brain.calls] == ["helper-see", "helper-see", "helper-plan"]
    assert 'They answered: "About 250 grams."' in brain.calls[1].prompt
    assert brain.calls[2].jpegs == brain.calls[1].jpegs  # the plan sees the picture of the last look
    assert rig.said[-1] == "Step 1. Crack the eggs into the bowl."
    ids = {p.id for p in rig.ts.scene(rig.t)}
    assert "hp-l0" in ids and "hp-k-0b" in ids and "hp-arrow" in ids
    assert sum(i.endswith("-txt") for i in ids) == 2  # one label each: "3 EGGS" and the bowl
    assert any(p.kind == "label" and p.text == "3 EGGS" for p in rig.ts.scene(rig.t))


def apply_page_rect(rect):
    """Camera px of a rectangle given in page px (what SAM returns)."""
    from lightwork.geometry import apply_h

    return apply_h(HI, rect)


def test_things_that_did_not_move_cost_no_model_call_and_keep_their_place(tmp_path):
    """4 Nov, real model: a bowl that never moved jumped 25 px between two calls, and the outline jumped with it."""
    scene = put(table(), page_rect(250, 210, 150, 170, 0), (40, 60, 200))
    plan = {"title": "T", "steps": [
        {"kind": "do", "say": "One", "parts": ["egg", "bowl of cottage cheese"], "count": 0, "seconds": 0, "check": ""},
        {"kind": "do", "say": "Two", "parts": ["bowl of cottage cheese"], "count": 0, "seconds": 0, "check": ""},
        {"kind": "do", "say": "Three", "parts": ["egg"], "count": 0, "seconds": 0, "check": ""}]}
    brain = FakeBrain(see=READY, plan=plan, locate={"found": [
        {"name": "egg", "visible": True, "x_min": 100, "y_min": 100, "x_max": 200, "y_max": 200}]})
    rig = Rig(game="helper", voice=True, brain=brain, state_dir=tmp_path)
    rig.run(0.5, scene)
    g = rig.ts.games.game
    rig.ts.hear("help me", rig.t)
    until(rig, scene, lambda: g.state == "steps")
    box = g.things[0].box
    rig.run(0.5, scene, poses=lambda t: [finger(NEXT_C)])  # a hand somewhere: the next step takes a new look
    rig.run(0.3, scene)
    rig.ts.hear("next", rig.t)
    until(rig, scene, lambda: g.want is None and rig.ts._look is None)
    rig.run(0.5, scene)
    assert g.k == 1 and not any(c.role == "helper-locate" for c in brain.calls)  # the picture says nothing moved
    assert g.things[0].box == box
    moved = put(scene, page_rect(397, 189, 60, 60, 0), (200, 90, 40))  # something changed where the first egg lay
    rig.run(0.5, moved, poses=lambda t: [finger(NEXT_C)])
    rig.run(0.3, moved)
    rig.ts.hear("next", rig.t)
    until(rig, moved, lambda: any(c.role == "helper-locate" for c in brain.calls))


def test_a_brick_moved_before_a_check_is_looked_for_again_on_the_next_step(tmp_path):
    """4 Nov, real model: the check's own look showed the moved brick, so the next step's look found "nothing moved"
    since then, and the arrow still pointed at where the brick had lain when the model last gave its box."""
    scene = put(table(), page_rect(250, 210, 150, 170, 0), (40, 60, 200))
    plan = {"title": "T", "steps": [
        {"kind": "do", "say": "One", "parts": ["egg"], "count": 0, "seconds": 0, "check": "Done?"},
        {"kind": "do", "say": "Two", "parts": ["egg"], "count": 0, "seconds": 0, "check": ""}]}
    brain = FakeBrain(see=READY, plan=plan, check={"ok": True, "say": "Yes."},
                      locate={"found": [{"name": "egg", "visible": True, "x_min": 100, "y_min": 100, "x_max": 160,
                                         "y_max": 200}]})
    rig = Rig(game="helper", voice=True, brain=brain, state_dir=tmp_path)
    rig.run(0.5, scene)
    g = rig.ts.games.game
    rig.ts.hear("help me", rig.t)
    until(rig, scene, lambda: g.state == "steps")
    moved = put(scene, page_rect(397, 189, 60, 60, 0), (200, 90, 40))
    rig.run(0.3, moved)
    assert g.command("check", rig.t)
    until(rig, moved, lambda: g.k == 1)
    until(rig, moved, lambda: any(c.role == "helper-locate" for c in brain.calls))
    until(rig, moved, lambda: g.things[1].box[0] < 100)


def test_a_frayed_sam_mask_is_the_table_and_is_not_drawn(tmp_path):
    """4 Nov, a bowl: with too much table in the box SAM returned the table around it, a frayed square that fits."""
    scene = table()
    sam = FakeOutliner()
    brain = FakeBrain(see=SEEN)
    rig = Rig(game="helper", voice=True, brain=brain, outliner=sam, state_dir=tmp_path)
    rig.run(0.5, scene)
    g = rig.ts.games.game
    rig.ts.hear("help me", rig.t)
    until(rig, scene, lambda: g.state == "asking" and all(t.asked for t in g.things))
    t = g.things[0]
    a = np.linspace(0, 2 * np.pi, 400, endpoint=False)
    frayed = (120 + 40 * (np.arange(400) % 2))[:, None] * np.column_stack([np.cos(a), np.sin(a)])
    sam.contours[t.uid] = GEO.d2c([t.centre])[0] + frayed
    until(rig, scene, lambda: any("frayed" in line for line in rig.ts.log))
    assert t.outline is None


def test_spoken_next_works_and_a_look_right_after_a_check_is_not_repeated(tmp_path):
    scene = table()
    brain = FakeBrain(see=READY, plan=PLAN, locate={"found": []}, check={"ok": True, "say": "Yes."})
    rig = Rig(game="helper", voice=True, brain=brain, state_dir=tmp_path)
    rig.run(0.5, scene)
    g = rig.ts.games.game
    rig.ts.hear("help me", rig.t)
    until(rig, scene, lambda: g.state == "steps")
    darks = sum(1 for _, _, prims in rig.page.history if prims == [])
    assert rig.ts.hear("Next.", rig.t)[0].startswith("Step 2")
    assert g.k == 1
    until(rig, scene, lambda: any(c.role == "helper-locate" for c in brain.calls))
    assert sum(1 for _, _, prims in rig.page.history if prims == []) <= darks + 1


def test_a_failed_model_call_is_said_and_the_tile_stays_usable(tmp_path):
    scene = table()
    brain = FakeBrain(see=[RuntimeError("HTTP 500"), READY], plan=PLAN)
    rig = Rig(game="helper", voice=True, brain=brain, state_dir=tmp_path)
    rig.run(0.5, scene)
    g = rig.ts.games.game
    rig.ts.hear("help me", rig.t)
    until(rig, scene, lambda: g.state == "idle" and len(brain.calls) == 1)
    assert "couldn't work that out" in rig.said[-1]
    rig.ts.hear("help me again", rig.t)
    until(rig, scene, lambda: g.state == "steps")


def test_leaving_the_tile_drops_what_was_on_its_way(tmp_path):
    scene = table()
    answer = threading.Event()

    class SlowBrain(FakeBrain):
        def ask(self, role, prompt, jpegs=(), **kw):
            answer.wait(5.0)
            return super().ask(role, prompt, jpegs, **kw)

    brain = SlowBrain(see=SEEN)
    rig = Rig(game="helper", voice=True, brain=brain, state_dir=tmp_path)
    rig.run(0.5, scene)
    g = rig.ts.games.game
    rig.ts.hear("help me", rig.t)
    until(rig, scene, lambda: bool(rig.ts._helper_jobs))
    rig.ts.hear("back to the menu", rig.t)
    assert isinstance(rig.ts.games.game, Menu)
    answer.set()
    until(rig, scene, lambda: bool(brain.calls))
    rig.run(0.5, scene)
    assert g.state == "seeing" and isinstance(rig.ts.games.game, Menu)  # nothing raised, nothing said
    assert not any("cottage cheese" in line for line in rig.said)
    rig.ts.start_game("helper", rig.t)  # and a new tile is not answered with the old one's words
    rig.run(0.3, scene)
    assert rig.ts.games.game.state == "idle" and rig.ts._helper_jobs == []


def test_without_voice_the_tile_says_how_to_turn_it_on(tmp_path):
    rig = Rig(game="helper", voice=False, state_dir=tmp_path)
    rig.run(0.5, table())
    assert rig.ts.games.game.listens is False
    assert any("voice is off" in line.lower() for line in rig.said)
