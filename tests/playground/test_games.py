"""Hand games on the table (lightwork.games), driven with synthetic hand points."""

from __future__ import annotations

from lightwork.games import DWELL_S, CatchLight, DrawGuess, FollowLights, GameHost, Menu, Pen


def frames(host_or_game, t0, t1, points=(), tips=(), dt=1 / 20):
    t = t0
    out = []
    while t < t1:
        t += dt
        if isinstance(host_or_game, GameHost):
            say, _ = host_or_game.update(t, list(points), list(tips))
            out += say
        else:
            host_or_game.update(t, list(points), list(tips))
    return t, out


def test_menu_picks_by_dwell(tmp_path):
    host = GameHost(state_dir=tmp_path, seed=1)
    host.start("menu", 0.0)
    menu = host.game
    assert isinstance(menu, Menu)
    catch = next(b for b in menu.bubbles if b.key == "catch")
    t, said = frames(host, 0.0, DWELL_S + 0.3, points=[catch.c])
    assert isinstance(host.game, CatchLight)
    assert said and said[-1].startswith("Catch the light")


def test_catch_scores_touches_and_ends(tmp_path):
    host = GameHost(state_dir=tmp_path, seed=3)
    host.start("catch", 0.0)
    g = host.game
    t, _ = frames(host, 0.0, g.READY_S + 1.0)
    assert g.targets
    for _ in range(5):  # touch whatever is lit
        tg = g.targets[0]
        t, _ = frames(host, t, t + 0.2, points=[tg.c])
        t, _ = frames(host, t, t + 0.6)
    assert g.score >= 5
    t, said = frames(host, t, g.READY_S + g.ROUND_S + 0.5)
    assert g.over and host.best["catch"] == g.score
    assert (tmp_path / "play-games.json").read_text().strip().startswith("{")
    again = host.again
    t, said = frames(host, t, t + 2.5, points=[again.c])  # hold PLAY AGAIN
    assert isinstance(host.game, CatchLight) and not host.game.over


def test_targets_expire_without_hands():
    g = CatchLight(0.0, __import__("random").Random(0))
    t, _ = frames(g, 0.0, g.READY_S + 6.0)
    assert g.missed >= 1 and g.score == 0


def test_follow_grows_on_the_right_order_and_ends_on_a_wrong_pad():
    import random

    g = FollowLights(0.0, random.Random(5))
    t = 0.0
    for level in range(1, 4):
        while g.state != "input":
            t += 0.05
            g.update(t, [], [])
        for pad in list(g.seq):
            t, _ = frames(g, t, t + 0.3, points=[g.pads[pad]])
            t, _ = frames(g, t, t + 0.3)  # hand off the pad between touches
        assert len(g.seq) == level + 1 and not g.over
    while g.state != "input":
        t += 0.05
        g.update(t, [], [])
    wrong = (g.seq[0] + 1) % 4
    t, _ = frames(g, t, t + 0.3, points=[g.pads[wrong]])
    assert g.over and g.score == 3 and "in a row" in g.say[-1]


def test_draw_then_hands_away_asks_for_a_guess():
    import random

    asked = []
    g = DrawGuess(0.0, random.Random(0), ask=lambda jpeg, not_these=(): asked.append((jpeg, not_these)))
    t = 0.0
    for k in range(30):  # a stroke across the table, thumb and finger together
        t += 0.05
        p = (200 + 6 * k, 200.0)
        g.update(t, [p], [p], (), [Pen(p, True)])
    t, _ = frames(g, t, t + 2.5)
    assert len(asked) == 1 and asked[0][0][:3] == b"\xff\xd8\xff"
    g.answer("cat", "That's a cat!", t)
    assert g.guess == "CAT" and g.say[-1] == "That's a cat!"
    p = (300.0, 300.0)
    g.update(t + 1, [p], [p], (), [Pen(p, True)])  # drawing again: a new picture
    assert g.guess is None and len(g.strokes) == 1


def draw_frames(g, t, xs, y, down, hand=0, dt=0.05):
    for x in xs:
        t += dt
        p = (float(x), float(y))
        g.update(t, [p], [p], (), [Pen(p, down, hand)])
    return t


def test_a_line_is_drawn_only_while_thumb_and_finger_touch():
    """3 Nov: "it draws as we move our finger around and never stops, so it's hard to draw when you can't stop"."""
    import random

    g = DrawGuess(0.0, random.Random(0), ask=lambda jpeg, not_these=(): None)
    t = draw_frames(g, 0.0, range(200, 260, 6), 200, down=False)  # fingers apart: the cursor moves, no ink
    assert g.strokes == [] and g.pens and not g.pens[0].down
    t = draw_frames(g, t, range(260, 320, 6), 200, down=True)  # together: a line
    assert len(g.strokes) == 1 and len(g.strokes[0]) == 10 and g.cur is not None
    t = draw_frames(g, t, [326], 200, down=False)  # apart again: the line ends at once
    assert g.cur is None and len(g.strokes[0]) == 10
    t = draw_frames(g, t, range(326, 400, 6), 230, down=False)  # moved away with the pen up: nothing joins them
    t = draw_frames(g, t, range(260, 290, 6), 260, down=True)
    assert len(g.strokes) == 2 and g.strokes[1][0] == (260.0, 260.0)
    assert g.hud(t, 0)[0].startswith("DRAWING")


def test_a_short_tracking_dropout_does_not_break_the_line_and_the_other_hand_cannot_take_it():
    import random

    g = DrawGuess(0.0, random.Random(0))
    t = draw_frames(g, 0.0, range(200, 240, 6), 200, down=True, hand=0)
    g.update(t + 0.2, [], [], (), [])  # hand 0 not seen for a moment
    t = draw_frames(g, t + 0.2, range(240, 270, 6), 200, down=True, hand=0)
    assert len(g.strokes) == 1
    other = Pen((500.0, 300.0), True, 1)
    g.update(t + 0.05, [(500.0, 300.0)], [], (), [other, Pen((276.0, 200.0), True, 0)])
    assert len(g.strokes) == 1 and g.strokes[0][-1] == (276.0, 200.0)  # the other hand's pinch is not this line's
    g.update(t + 0.1, [(500.0, 300.0)], [], (), [other])  # hand 0 gone; hand 1 is far from the line: not it ...
    assert g.cur_hand == 0
    g.update(t + 0.1 + g.LIFT_S + 0.05, [(500.0, 300.0)], [], (), [other])  # ... gone longer than LIFT_S: line ends
    assert g.cur is not None and g.cur_hand == 1 and len(g.strokes) == 2  # and hand 1, still pinched, draws its own


def test_a_hand_found_again_in_another_slot_carries_its_line_on():
    """The table's tracker may find a lost hand again as its other hand slot: same pinch, near where the line stopped."""
    import random

    g = DrawGuess(0.0, random.Random(0))
    t = draw_frames(g, 0.0, range(200, 240, 6), 200, down=True, hand=0)
    g.update(t + 0.3, [], [], (), [])
    t = draw_frames(g, t + 0.3, range(250, 280, 6), 205, down=True, hand=1)
    assert len(g.strokes) == 1 and g.cur_hand == 1 and g.strokes[0][-1] == (274.0, 205.0)


def test_voice_routing(tmp_path):
    host = GameHost(state_dir=tmp_path)
    assert host.hear("what is this drawing?", 0.0) is None  # not playing: not a game sentence
    assert host.hear("let's play a game", 0.0)[0].startswith("Pick a game")
    assert isinstance(host.game, Menu)
    assert host.hear("follow the lights", 1.0)[0].startswith("Follow")
    assert isinstance(host.game, FollowLights)
    assert host.hear("stop playing", 2.0) == ["Object board."] and not host.active


def test_table_session_runs_a_game_without_scanning(tmp_path):
    from .test_table_scans import Rig
    from .test_tabletop import page_rect, put, table

    rig = Rig(game="catch", state_dir=tmp_path)
    scene = put(table(), page_rect(300, 200, 120, 70, 0), (40, 60, 200))
    rig.run(4.0, scene)
    assert rig.ts.scans == 0 and rig.ts.games.active
    ids = {p.id for p in rig.page.history[-1][2]}
    assert "btn-home" in ids and not any(i.startswith("btn-learn") for i in ids)
    assert any(i.startswith("tg-") for i in ids)  # a target is up
    rig.ts.hear("stop playing", rig.t)
    rig.run(3.0, scene)
    assert not rig.ts.games.active and rig.ts.scans >= 1 and len(rig.ts.tracks.confirmed) == 1


def test_a_fast_hand_catches_a_light_it_crossed_between_frames():
    import random

    g = CatchLight(0.0, random.Random(2))
    g.update(g.READY_S + 0.5, [], [])
    tg = g.targets[0]
    x, y = tg.c
    a, b = (x - 150.0, y), (x + 150.0, y)  # seen left of it, then right of it: never on it
    g.update(g.READY_S + 0.6, [a], [], [])
    assert g.score == 0
    g.update(g.READY_S + 0.65, [b], [], [(a, b)])
    assert g.score >= 1


def test_spoken_corrections_in_draw_and_guess():
    import random

    asked = []
    g = DrawGuess(0.0, random.Random(0), ask=lambda jpeg, not_these=(): asked.append(list(not_these)))
    g.strokes = [[(200.0, 200.0), (260.0, 220.0), (300.0, 260.0)]]
    g._guess(1.0)
    g.answer("sleeping dog", "A sleepy dog!", 2.0)
    assert g.hear("no, it's not a sleeping dog", 3.0)
    assert asked[-1] == ["sleeping dog"] and g.say[-1] == "Hmm, let me look again."
    g.answer("cat", "A cat!", 4.0)
    assert g.hear("no, it's a horse", 5.0)
    assert g.say[-1].startswith("A horse!") and g.guess == "HORSE"
    g.answer("horse", "A horse!", 6.0)
    assert g.hear("yes", 7.0) and g.say[-1] == "Yes! Draw another one."


def test_recal_button_and_smooth_projection_with_games(tmp_path):
    from .test_table_scans import Rig

    rig = Rig(state_dir=tmp_path)
    calls = []
    rig.page.set_smooth = calls.append  # the render hub's switch for the page's easing and fades
    assert {b.key for b in rig.ts.dash.buttons} == {"home", "learn", "scale", "recal"}
    rig.ts.start_game("catch", rig.t)
    assert {b.key for b in rig.ts.dash.buttons} == {"home", "recal"} and calls == [True]
    rig.ts.hear("stop playing", rig.t)
    assert calls == [True, False] and "recal" in {b.key for b in rig.ts.dash.buttons}
    recal = next(b for b in rig.ts.dash.buttons if b.key == "recal")
    fired = rig.ts.dash.step([recal.c], 0.0, 0.05)
    t, got = 0.0, []
    while t < 1.5 and not got:
        t += 0.05
        got = rig.ts.dash.step([recal.c], t, 0.05)
    assert got == ["recal"] and fired == []  # held 1.2 s: passed up to the play loop, which recalibrates


def test_hub_smooth_keeps_the_rotation():
    from lightwork.contracts.render import View
    from lightwork.server.hub import RenderHub

    hub = RenderHub()
    hub.set_view(View(rotate=180))
    hub.set_smooth(True)
    assert hub._view.rotate == 180 and hub._view.smooth
    assert '"smooth":true' in hub._view.model_dump_json()
    hub.set_smooth(False)
    assert '"smooth"' not in hub._view.model_dump_json()  # off: the message is as before smoothing existed


def test_object_board_tile_learns_the_table_when_it_never_was(tmp_path):
    from .test_table_scans import Rig
    from .test_tabletop import table

    rig = Rig(game="menu", state_dir=tmp_path)
    rig.ts.zone.ref = None  # started in a game: no empty table learned yet
    rig.run(0.5, table())
    menu = rig.ts.games.game
    tile = next(b for b in menu.bubbles if b.key == "objects")
    t = rig.t
    said = []
    while rig.ts.games.active and rig.t < t + 2.0:
        rig.t += 0.05
        rig.page.now = rig.t
        said += rig.ts.games.update(rig.t, [tile.c], [])[0]
        if not rig.ts.games.active:
            said += rig.ts._leave_games(rig.t)
    assert not rig.ts.games.active and rig.ts.learn_at is not None
    assert said[-1].startswith("Object board. Clear the table")
    assert "home" in {b.key for b in rig.ts.dash.buttons} and "learn" in {b.key for b in rig.ts.dash.buttons}


def pinched(p):
    """``p`` with the thumb tip next to the index fingertip (a tenth of the palm away), in the image and in 3-D."""
    import numpy as np

    from lightwork.vision.hands import HandPose

    img = p.img.copy()
    img[4] = img[8] + (img[9] - img[0]) * 0.1
    scale = np.linalg.norm(p.world[9] - p.world[0]) / np.linalg.norm(p.img[9] - p.img[0])
    return HandPose(img, np.column_stack([(img - img.mean(axis=0)) * scale, np.zeros(21)]), p.side, p.score)


def test_table_draw_game_draws_with_a_pinch_and_shows_the_pen(tmp_path):
    from .test_table_scans import Rig
    from .test_tabletop import pose, table

    rig = Rig(game="draw", state_dir=tmp_path)

    def hand(t):
        return pose((560 + 60 * t, 640), (590 + 60 * t, 540))

    t0 = rig.t
    rig.run(1.0, table(), poses=lambda t: [hand(t - t0)])
    g = rig.ts.games.game
    assert isinstance(g, DrawGuess) and g.strokes == []
    ids = {p.id for p in rig.page.history[-1][2]}
    assert "pen0" in ids and "pen0-dot" not in ids and not any(i.endswith("-tip") for i in ids)
    rig.run(1.0, table(), poses=lambda t: [pinched(hand(t - t0))])
    assert len(g.strokes) == 1 and len(g.strokes[0]) >= 3 and g.cur is not None
    assert "pen0-dot" in {p.id for p in rig.page.history[-1][2]}
    rig.run(0.5, table(), poses=lambda t: [hand(t - t0)])
    assert g.cur is None and len(g.strokes) == 1
    assert any(line.startswith("pen down") for line in rig.ts.log) and any(
        line.startswith("pen up") for line in rig.ts.log)


def test_questions_during_a_game_are_answered_and_table_talk_is_not_sent(tmp_path):
    """3 Nov: "um" went to the asker as a question; its answer was never collected during the game, so everything said
    after it got "One moment, I am still answering the last question."."""
    from .test_table_scans import Rig
    from .test_table_task_voice import FakeAsker
    from .test_tabletop import table

    asker = FakeAsker()
    rig = Rig(game="draw", asker=asker, state_dir=tmp_path)
    rig.run(0.5, table())
    assert rig.ts.hear("um", rig.t) == [] and rig.ts.hear("Hmm...", rig.t) == []
    assert rig.ts.hear("Piano.", rig.t) == [] and rig.ts.hear("тяжело вот", rig.t) == []
    assert asker.asked == []
    assert rig.ts.hear("What is on my left?", rig.t) == [] and len(asker.asked) == 1
    asker.ready = True
    rig.run(0.2, table())
    assert rig.said[-1].startswith("That is a stapler") and rig.ts.games.active
    rig.ts.hear("how far is it?", rig.t)
    assert len(asker.asked) == 2  # not stuck on the first question


def test_a_hand_off_the_page_plays_no_part(tmp_path):
    """3 Nov: a plush toy beside the table was found as a hand (0.5-0.99 confidence); draw and guess, seeing a hand
    all along, never guessed."""
    from lightwork.geometry import apply_h

    from .test_table_scans import Rig
    from .test_tabletop import HI, pose, table

    rig = Rig(game="draw", state_dir=tmp_path)
    rig.run(0.1, table())
    g = rig.ts.games.game
    g.strokes = [[(200.0 + 20 * k, 200.0 + 8 * k) for k in range(8)]]
    w, k = apply_h(HI, [(300, -150), (300, -210)])  # well beyond the page's far edge
    toy = pose(w, k)
    rig.run(2.5, table(), poses=lambda t: [toy])
    assert rig.ts.slots.active and rig.ts._game_slots() == []  # tracked, but not playing
    assert g.guessed_n == 1  # no hand on the page: it guessed (no brain in this rig, so it says it needs one)
    assert "pen0" not in {p.id for p in rig.page.history[-1][2]}


def test_a_tracking_slip_starts_a_new_line_instead_of_a_long_straight_one():
    import random

    g = DrawGuess(0.0, random.Random(0))
    t = draw_frames(g, 0.0, range(200, 240, 6), 200, down=True)
    t = draw_frames(g, t, [240 + g.REJOIN + 5], 200, down=True)  # one frame later, a hand's width away
    assert len(g.strokes) == 2 and g.strokes[1] == [(240 + g.REJOIN + 5, 200.0)]


def test_letting_go_leaves_no_hook_and_a_blip_does_not_break_the_line(tmp_path):
    """The pen holds still while fingers that were together read apart: a blip of a frame or two carries on, and
    letting go ends the line where the fingers were last together."""
    from lightwork.vision.hands import PINCH_UP_FRAMES

    from .test_table_scans import Rig
    from .test_tabletop import pose, table

    rig = Rig(game="draw", state_dir=tmp_path)
    t0 = rig.t

    def hand(t):
        return pose((560 + 80 * (t - t0), 640), (590 + 80 * (t - t0), 540))

    rig.run(0.8, table(), poses=lambda t: [pinched(hand(t))])
    g = rig.ts.games.game
    assert len(g.strokes) == 1
    n = 0

    def blip(t):
        nonlocal n
        n += 1
        return [hand(t) if n < PINCH_UP_FRAMES else pinched(hand(t))]  # apart for PINCH_UP_FRAMES - 1 frames

    rig.run(0.5, table(), poses=blip)
    assert len(g.strokes) == 1 and g.cur is not None
    last = g.cur[-1]
    rig.run(0.5, table(), poses=lambda t: [hand(t)])  # let go and keep moving
    assert g.cur is None and len(g.strokes) == 1 and g.strokes[0][-1] == last
