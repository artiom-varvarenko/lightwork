"""Timers and stopwatches on the table (lightwork.timers), and how the helper tile and the session use them."""

from __future__ import annotations

import random

import numpy as np
import pytest

from lightwork.games import GameHost, Menu
from lightwork.helper import Helper, MIC_C, MIC_TAIL_S, NO_WORDS_S, WORK, parse_job
from lightwork.playground import DESIGN_H, DESIGN_W, Canvas, Geo
from lightwork.timers import BEEP_S, HOLD_S, ROWS_Y, SAY_AGAIN_S, X0, X1, Timers, clock_text, name_in, parse_duration, spoken_time


def drawn(t: Timers, now: float) -> list:
    c = Canvas(Geo(np.eye(3), DESIGN_W, DESIGN_H))
    t.draw(c, now)
    return c.prims


def labels(t: Timers, now: float) -> list[str]:
    return [p.text for p in drawn(t, now) if p.kind == "label"]


@pytest.mark.parametrize("said, seconds", [
    ("set a timer for 10 minutes", 600), ("ten minute timer", 600), ("Timer, 30 seconds.", 30), ("90 seconds", 90),
    ("an hour and a half", 5400), ("half an hour", 1800), ("a minute and a half", 90), ("2 and a half minutes", 150),
    ("1 hour 20 minutes", 4800), ("1 hour and 20 minutes", 4800), ("one hour 20", 4800), ("twenty-five minutes", 1500),
    ("a quarter of an hour", 900), ("three quarters of an hour", 2700), ("1:30", 90), ("1.5 hours", 5400),
    ("forty five mins", 2700), ("thirty secs", 30), ("an hour", 3600),
    ("set a timer", None), ("for the oven", None), ("0 minutes", None), ("300 hours", None),
])
def test_durations_as_people_say_them(said, seconds):
    assert parse_duration(said) == seconds


@pytest.mark.parametrize("said, name", [
    ("set a timer for 30 minutes for the oven", "OVEN"), ("oven timer 30 minutes", "OVEN"),
    ("start a 10 minute timer for the pasta", "PASTA"), ("remind me to check the cake in 10 minutes", "CAKE"),
    ("set a timer for 10 minutes", ""), ("ten minute timer", ""), ("start a timer", ""), ("put a timer on for 5 minutes", ""),
    ("timer for my rice 20 minutes", "RICE"),
])
def test_what_a_timer_is_for(said, name):
    assert name_in(said) == name


def test_clock_and_speech_formats():
    assert clock_text(65) == "1:05" and clock_text(3725) == "1:02:05" and clock_text(0.2) == "0:01" and clock_text(-3) == "0:00"
    assert spoken_time(90) == "1 minute 30 seconds" and spoken_time(3600) == "1 hour" and spoken_time(0) == "0 seconds"


def test_a_timer_counts_down_rings_and_stops_when_told():
    t = Timers()
    assert t.hear("set a timer for 30 minutes for the oven", 0.0) == ["Oven timer set: 30 minutes."]
    assert labels(t, 0.0) == ["OVEN 30:00"]
    assert labels(t, 65.0) == ["OVEN 28:55"]
    assert t.update(1799.0) == []
    assert t.update(1800.1) == ["The oven timer is done."] and "answer" in t.sfx
    assert labels(t, 1800.5) == ["OVEN DONE"]
    t.sfx.clear()
    t.update(1800.1 + BEEP_S + 0.1)
    assert t.sfx == ["answer"]  # it keeps beeping ...
    assert t.update(1800.1 + SAY_AGAIN_S + 0.1) == ["The oven timer is still done."]  # ... and says it once more
    assert t.update(1800.1 + 2 * SAY_AGAIN_S) == []
    assert t.hear("stop", 2000.0) == ["Okay."] and not t.active


def test_stop_means_the_job_when_nothing_rings():
    t = Timers()
    t.hear("timer 10 minutes", 0.0)
    assert t.hear("stop", 5.0) is None  # the helper's own "stop"
    assert t.hear("help me bake it for 10 minutes", 5.0) is None  # no timer asked for: a job
    assert len(t.clocks) == 1


def test_asking_adding_and_cancelling():
    t = Timers()
    t.hear("oven timer 30 minutes", 0.0)
    t.hear("pasta timer 10 minutes", 0.0)
    assert t.hear("how long is left on the oven?", 600.0) == ["The oven timer has 20 minutes left."]
    assert t.hear("how much time is left?", 300.0) == ["The oven timer has 25 minutes left. The pasta timer has 5 minutes left."]
    assert t.hear("add 5 minutes to the oven timer", 600.0) == ["25 minutes left."]
    assert t.hear("timer 5 more minutes on the oven", 600.0) == ["30 minutes left."]
    assert t.hear("another timer for 1 minute", 600.0) == ["Timer set: 1 minute."]  # a new one, not an extension
    t.clocks.pop()
    assert t.hear("cancel the pasta timer", 601.0) == ["Timer cancelled."]
    assert [c.name for c in t.clocks] == ["OVEN"]
    t.hear("cancel the timer", 602.0)
    assert t.hear("cancel the timer", 603.0) == ["No timer is running."]


def test_a_timer_with_no_time_asks_how_long_and_the_answer_sets_it():
    t = Timers()
    assert t.hear("set a timer for the oven", 0.0) == ["How long?"]
    assert t.hear("Twenty minutes.", 1.0) == ["Oven timer set: 20 minutes."]
    assert t.clocks[0].seconds == 1200
    assert t.hear("start a timer", 2.0) == ["How long?"]
    assert t.hear("help me build this", 3.0) is None and t.wanted is None  # not an answer: it lets go


def test_two_timers_of_one_name_and_a_limit():
    t = Timers()
    t.hear("timer 1 minute", 0.0)
    t.hear("timer 2 minutes", 0.0)
    assert [c.name for c in t.clocks] == ["TIMER", "TIMER 2"]
    t.hear("timer 3 minutes", 0.0)
    t.hear("timer 4 minutes", 0.0)
    assert t.hear("timer 5 minutes", 0.0) == ["That's too many timers. Cancel one first."] and len(t.clocks) == 4


def test_a_stopwatch_by_voice_and_by_a_held_finger():
    t = Timers()
    assert t.hear("start a stopwatch", 0.0) == ["Stopwatch started."]
    assert labels(t, 83.0) == ["STOPWATCH 1:23"]
    assert t.hear("stop the stopwatch", 83.0) == ["Stopped at 1 minute 23 seconds."]
    assert labels(t, 100.0) == ["STOPWATCH 1:23"]  # frozen
    assert t.hear("start the stopwatch", 100.0) == ["Stopwatch running again."]
    assert labels(t, 110.0) == ["STOPWATCH 1:33"]
    assert t.hear("how long on the stop watch", 110.0)[0].startswith("The stopwatch is at 1 minute 33")
    x, y, w = t._where(110.0)[t.clocks[0].uid]
    finger = (x + w / 2, y)
    now, said = 110.0, []
    for _ in range(int(HOLD_S / 0.05) + 2):
        now += 0.05
        said += t.update(now, [finger], 0.05)
    assert said and said[0].startswith("Stopped at") and not t.clocks[0].running
    for _ in range(40):  # the finger stays: the next hold needs it to leave first
        now += 0.05
        t.update(now, [finger], 0.05)
    assert t.active
    t.update(now + 0.05, [], 0.05)
    now += 0.05
    for _ in range(int(HOLD_S / 0.05) + 2):
        now += 0.05
        t.update(now, [finger], 0.05)
    assert not t.active
    t.hear("start a stopwatch", 0.0)
    assert t.hear("reset the stopwatch", 1.0) == ["Stopwatch cleared."] and not t.active


def test_a_running_timer_ignores_a_resting_arm_and_one_that_rings_stops_under_a_finger():
    t = Timers()
    t.hear("timer 1 minute", 0.0)
    x, y, w = t._where(0.0)[t.clocks[0].uid]
    finger = (x + w / 2, y)
    now = 0.0
    for _ in range(100):
        now += 0.05
        t.update(now, [finger], 0.05)
    assert t.active and t.clocks[0].rang is None
    t.update(60.1)
    now = 60.1
    for _ in range(int(HOLD_S / 0.05) + 2):
        now += 0.05
        t.update(now, [finger], 0.05)
    assert not t.active


def test_clocks_stay_on_the_page_in_two_lines_with_unique_prim_ids():
    t = Timers()
    t.hear("stopwatch start", 0.0)
    t.hear("lasagna timer 2 hours", 0.0)
    t.hear("potatoes timer 1 hour 30 minutes", 0.0)
    t.hear("dough timer 45 minutes", 0.0)
    t.clocks[-1].held = 0.3
    prims = drawn(t, 5.0)
    ids = [p.id for p in prims]
    assert len(ids) == len(set(ids))
    for x, y, w in t._where(5.0).values():
        assert X0 <= x and x + w <= X1 and y in ROWS_Y
    t.update(2700.5)  # one rings: it is drawn too, and the ids stay unique
    ids = [p.id for p in drawn(t, 2701.0)]
    assert len(ids) == len(set(ids))


def test_clocks_outlive_the_tiles(tmp_path):
    host = GameHost(state_dir=tmp_path, seed=1, listens=True, thinks=True)
    host.start("helper", 0.0)
    assert host.game.timers is host.timers
    host.timers.hear("oven timer 30 minutes", 0.0)
    for key in ("menu", "catch", "rps", "helper"):
        host.start(key, 1.0)
        c = Canvas(Geo(np.eye(3), DESIGN_W, DESIGN_H))
        host.draw(c, 2.0)
        assert any(p.kind == "label" and p.text.startswith("OVEN 29:") for p in c.prims), key
        ids = [p.id for p in c.prims]
        assert len(ids) == len(set(ids)), key
    assert isinstance(host.start("menu", 1.0), list) and isinstance(host.game, Menu)


def test_on_the_menu_clocks_keep_to_one_line_and_clear_of_the_labels(tmp_path):
    host = GameHost(state_dir=tmp_path, seed=1, listens=True, thinks=True)
    host.best = {"rps": 5, "catch": 9}
    host.start("menu", 0.0)
    for words in ("oven timer 30 minutes", "eggs timer 10 minutes", "start a stopwatch"):
        host.timers.hear(words, 0.0)
    c = Canvas(Geo(np.eye(3), DESIGN_W, DESIGN_H))
    host.draw(c, 1.0)
    texts = [p.text for p in c.prims if p.kind == "label"]
    assert any(t.startswith("OVEN") for t in texts) and any(t.startswith("EGGS") for t in texts)
    assert not any(t.startswith("STOPWATCH") for t in texts)  # beyond the line, and still running
    assert "BEST 9" in texts and "BEST 5" not in texts  # the far row's best stays, the near row's would hit the clocks
    assert host.timers.clocks[2].running
    host.start("helper", 1.0)
    c = Canvas(Geo(np.eye(3), DESIGN_W, DESIGN_H))
    host.draw(c, 2.0)
    assert any(p.kind == "label" and p.text.startswith("STOPWATCH") for p in c.prims)


# ---- a job's oven step


OVEN = {"title": "Bake", "steps": [
    {"kind": "background", "say": "Put the dish in the oven for 30 minutes", "parts": [], "count": 0, "seconds": 1800,
     "check": "", "label": "oven"},
    {"kind": "do", "say": "Wash the bowl while it bakes", "parts": [], "count": 0, "seconds": 0, "check": "", "label": ""},
]}


def job_tile() -> Helper:
    g = Helper(0.0, random.Random(0))
    g.request = "bake"
    g.job = parse_job(OVEN)
    g.state = "steps"
    g._enter(0, 1.0)
    g.want = None
    return g


def test_a_plans_background_step_needs_a_time_and_keeps_its_label():
    job = parse_job(OVEN)
    assert job.steps[0].kind == "background" and job.steps[0].seconds == 1800 and job.steps[0].label == "oven"
    assert job.steps[1].label == ""
    bad = parse_job({"steps": [{"kind": "background", "say": "Bake it", "seconds": 0, "label": "oven"}]})
    assert bad.steps[0].kind == "do" and bad.steps[0].label == ""


def test_the_oven_step_starts_its_clock_when_they_move_on_and_the_job_carries_on():
    g = job_tile()
    assert not g.timers.active  # it isn't in the oven yet
    assert ("OVEN 30:00 WHEN YOU HOLD NEXT", "source") in g._block_lines()
    assert g._hint().startswith("HOLD NEXT")
    g.command("next", 5.0)
    assert g.k == 1 and [c.name for c in g.timers.clocks] == ["OVEN"]
    assert "Oven timer set: 30 minutes." in g.say
    assert g.timers.clocks[0].started == 5.0
    g.command("back", 6.0)  # looking back does not start it over
    g.want = None
    g.command("next", 7.0)
    assert len(g.timers.clocks) == 1 and g.timers.clocks[0].started == 5.0
    g.command("next", 8.0)  # the job can finish with the oven still on
    assert g.state == "done" and g.timers.clocks[0].rang is None


def test_going_back_from_the_oven_step_starts_nothing():
    g = job_tile()
    g.job = parse_job({"steps": [OVEN["steps"][1], OVEN["steps"][0]]})
    g._enter(1, 1.0)
    g.want = None
    g.command("back", 2.0)
    assert g.k == 0 and not g.timers.active


def test_text_keeps_clear_of_the_clocks():
    assert WORK[3] <= min(ROWS_Y) - 13  # the helper's text and labels stop above the two lines of clocks
    assert X0 >= WORK[0] and X1 <= WORK[2]


# ---- on the table


def test_a_spoken_timer_on_the_table_is_shown_rings_and_costs_no_model_call(tmp_path):
    from .test_helper import FakeBrain, FakeGate, until
    from .test_table_scans import Rig
    from .test_tabletop import table

    brain = FakeBrain()
    rig = Rig(game="helper", voice=True, brain=brain, state_dir=tmp_path, mic_gate=FakeGate())
    rig.run(0.5, table())
    g = rig.ts.games.game
    assert rig.ts.hear("Set a timer for 2 seconds for the oven", rig.t) == ["Oven timer set: 2 seconds."]
    rig.run(0.2, table())
    assert any(p.kind == "label" and p.text.startswith("OVEN 0:0") for p in rig.ts.scene(rig.t))
    assert brain.calls == [] and g.state == "idle"
    until(rig, table(), lambda: "The oven timer is done." in rig.said)
    assert any(p.kind == "label" and p.text == "OVEN DONE" for p in rig.ts.scene(rig.t))
    assert rig.ts.hear("stop", rig.t) == ["Okay."] and not rig.ts.games.timers.active
    assert g.state == "idle"  # the "stop" was about the ringing
    assert brain.calls == []


def test_words_acted_on_are_not_called_unheard_after_letting_go(tmp_path):
    """What was said is taken as soon as it is heard ("next", a timer); the mic closes after it. That is not silence."""
    g = Helper(0.0, random.Random(0))
    t = 0.0
    for _ in range(10):
        t += 0.05
        g.update(t, [MIC_C], [MIC_C])
    assert g.mic_open
    g.dealt_with()
    for _ in range(int((MIC_TAIL_S + NO_WORDS_S + 1.0) / 0.05)):
        t += 0.05
        g.update(t, [], [])
    assert not g.mic_open and not any("didn't catch" in line for line in g.say) and g.released_t is None


def test_a_look_at_the_table_has_no_clock_in_the_picture(tmp_path):
    from .test_helper import FakeBrain, READY, until
    from .test_table_scans import Rig
    from .test_tabletop import table

    brain = FakeBrain(see=READY, plan={"title": "T", "steps": [{"kind": "do", "say": "One", "parts": [], "count": 0,
                                                                "seconds": 0, "check": "", "label": ""}]})
    rig = Rig(game="helper", voice=True, brain=brain, state_dir=tmp_path)
    rig.run(0.5, table())
    rig.ts.hear("oven timer 10 minutes", rig.t)
    rig.ts.hear("help me", rig.t)
    until(rig, table(), lambda: bool(brain.calls))
    assert any(prims == [] for _, _, prims in rig.page.history)  # still all dark for the picture
    rig.run(0.3, table())
    assert any(p.kind == "label" and p.text.startswith("OVEN ") for p in rig.ts.scene(rig.t))


def test_on_the_object_board_a_clock_runs_and_never_becomes_an_object(tmp_path):
    from .test_table_scans import Rig
    from .test_tabletop import table

    rig = Rig(state_dir=tmp_path)
    rig.ts.games.timers.start_timer("OVEN", 600.0, rig.t)
    rig.run(6.0, table())
    assert rig.ts.tracks.confirmed == []
    assert any(p.kind == "label" and p.text.startswith("OVEN 9:") for p in rig.ts.scene(rig.t))
