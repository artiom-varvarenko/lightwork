"""Hand games on the table (lightwork.games), driven with synthetic hand points."""

from __future__ import annotations

from lightwork.games import (
    AREA,
    BEATS,
    DWELL_S,
    EM,
    MOVES,
    CatchLight,
    DrawGuess,
    FollowLights,
    GameHost,
    Menu,
    Pen,
    RockPaperScissors,
)


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


# ---------------------------------------------------------------------------------------------------- rock paper scissors


VERB = {"rock": "beats", "paper": "covers", "scissors": "cut"}


def beating(move: str) -> str:
    return next(m for m in MOVES if BEATS[m] == move)


def rps_frames(host_or_game, t0, t1, shape=None, at=(410.0, 320.0), dt=1 / 20):
    """Frames with one hand on the page making ``shape`` ("": a hand it cannot read; None: no hand). Returns (time,
    lines said, sounds) when driving a :class:`GameHost`."""
    t, said, sounds = t0, [], []
    while t < t1 - 1e-9:
        t += dt
        pens = [] if shape is None else [Pen(at, False, 0, shape)]
        if isinstance(host_or_game, GameHost):
            say, sfx = host_or_game.update(t, [p.at for p in pens], [p.at for p in pens], (), pens)
            said, sounds = said + say, sounds + sfx
        else:
            host_or_game.update(t, [p.at for p in pens], [p.at for p in pens], (), pens)
    return t, said, sounds


def rps_round(host_or_game, t, shape_for, before="rock"):
    """A hand (making ``before``) on the page until a count starts, then making ``shape_for(the table's move)`` until
    the round is decided. Returns (time, lines said)."""
    g = host_or_game.game if isinstance(host_or_game, GameHost) else host_or_game
    said = []
    while g.state != "count":
        t, s, _ = rps_frames(host_or_game, t, t + 0.05, before)
        said += s
    shape = shape_for(g.mine)
    while g.state in ("count", "shoot"):
        t, s, _ = rps_frames(host_or_game, t, t + 0.05, shape)
        said += s
    return t, said


def test_rps_counts_down_reads_the_hand_and_keeps_score():
    import random

    g = RockPaperScissors(0.0, random.Random(3))
    t, _, _ = rps_frames(g, 0.0, 1.0)  # nobody there: it waits
    assert g.state == "wait" and g.sfx == [] and g.hud(t, 0)[0] == "HOLD YOUR HAND OVER THE RING"
    t, _, _ = rps_frames(g, t, t + g.HOLD_S + 0.1, "rock")
    assert g.state == "count" and g.mine in MOVES  # the table picked its move as the count began
    start, mine = g.t, g.mine
    digits = []
    while g.state == "count":
        t, _, _ = rps_frames(g, t, t + 0.05, beating(mine))
        digits.append(g.COUNT[min(int((t - start) // g.BEAT_S), len(g.COUNT) - 1)])
    assert sorted(set(digits), reverse=True) == [3, 2, 1] and digits[0] == 3
    assert g.state == "shoot" and g.t == start + 3 * g.BEAT_S and g.sfx == ["press"] * 3 + ["scan"]
    t, _, _ = rps_frames(g, t, g.t + g.READ_S + 0.05, beating(mine))
    assert g.state == "result" and g.mine == mine and g.yours == beating(mine) and g.outcome == "win"
    assert (g.you, g.me, g.score) == (1, 0, 1) and g.sfx[-1] == "found"
    assert g.say == [f"{beating(mine).capitalize()} {VERB[beating(mine)]} {mine}. You win!"]
    assert g.hud(t, 0) == (f"{beating(mine).upper()} {VERB[beating(mine)].upper()} {mine.upper()}: YOU WIN!",
                           "YOU 1 : 0 TABLE · FIRST TO 5", "target")
    t, _, _ = rps_frames(g, t, t + g.RESULT_S - 0.1, beating(mine))
    assert g.state == "result"  # the result stays up a while ...
    t, _, _ = rps_frames(g, t, t + 0.2, beating(mine))
    assert g.state == "count"  # ... then the next count, the hand still there
    t, _ = rps_round(g, t, lambda m: BEATS[m])  # the table's move beats the hand's
    assert g.outcome == "lose" and (g.you, g.me, g.score) == (1, 1, 1) and g.say[-1].endswith(". I win!")
    assert g.sfx[-1] == "gone" and g.hud(t, 0)[1:] == ("YOU 1 : 1 TABLE · FIRST TO 5", "error")
    t, _ = rps_round(g, t, lambda m: m)
    assert g.outcome == "draw" and (g.you, g.me) == (1, 1) and g.say[-1] == f"Both {g.mine}. Draw!"
    assert g.hud(t, 0)[0] == f"BOTH {g.mine.upper()}: DRAW" and g.sfx[-1] == "answer"
    assert any(line.startswith(f"rps: read {g.mine}") for line in g.log)


def test_rps_nothing_read_says_so_and_counts_again_with_the_score_unchanged():
    import random

    g = RockPaperScissors(0.0, random.Random(2))
    t, _ = rps_round(g, 0.0, lambda m: "")  # a hand on the page, in no shape it can read
    assert g.state == "noread" and g.say[-1] == "I couldn't read your hand. Again!" and g.sfx[-1] == "error"
    assert (g.you, g.me, g.score) == (0, 0, 0) and g.hud(t, 0)[0] == "I COULDN'T SEE YOUR MOVE"
    t, _, _ = rps_frames(g, t, t + g.AGAIN_S + 0.1, "rock")
    assert g.state == "count"  # again, the hand still there
    t, _ = rps_round(g, t, lambda m: None)  # the hand went away before SHOOT
    assert g.state == "noread" and g.say[-1] == "I can't see your hand. Again!" and (g.you, g.me) == (0, 0)
    t, _, _ = rps_frames(g, t, t + g.AGAIN_S + 0.5)
    assert g.state == "wait"  # nobody there: it waits for a hand
    t, _ = rps_round(g, t, beating)
    assert g.outcome == "win" and (g.you, g.me) == (1, 0)


def test_rps_the_table_moves_the_same_whatever_the_hand_shows():
    """Fair: the table's moves come from the count starting, never from the hand."""
    import random

    picked = {}
    for shape in MOVES:
        g = RockPaperScissors(0.0, random.Random(11))
        t, moves = 0.0, []
        for _ in range(4):
            t, _ = rps_round(g, t, lambda m: shape, before=shape)
            moves.append(g.mine)
        picked[shape] = moves
    assert picked["rock"] == picked["paper"] == picked["scissors"]
    assert len(set(picked["rock"])) > 1


def test_rps_reads_the_shape_held_after_the_throw():
    """A fist pumped through the count and the throw is not the move: shapes count from ``THROW_S`` after SHOOT, and
    the one most frames showed wins (a tie: the one shown last)."""
    import random

    g = RockPaperScissors(0.0, random.Random(4))
    t, _, _ = rps_frames(g, 0.0, 0.5, "rock")
    shoot = g.t + len(g.COUNT) * g.BEAT_S
    t, _, _ = rps_frames(g, t, shoot + 0.15, "rock")
    t, _, _ = rps_frames(g, t, shoot + 0.8, "paper")
    t, _, _ = rps_frames(g, t, shoot + 1.05, "scissors")
    assert g.state == "result" and g.yours == "paper" and "rock" not in g.votes and "scissors" in g.votes
    g.votes, g.state = ["rock", "paper", "paper", "rock"], "shoot"
    g._judge(t)
    assert g.yours == "rock"


def test_rps_first_to_five_ends_the_match_and_play_again_starts_over(tmp_path):
    host = GameHost(state_dir=tmp_path, seed=5)
    host.start("rps", 0.0)
    g = host.game
    t, said, huds = 0.0, [], set()
    for _ in range(2):
        t, s = rps_round(host, t, beating)
        said += s
        huds |= {*g.hud(t, 0)[:2]}
    for _ in range(5):
        t, s = rps_round(host, t, lambda m: BEATS[m])
        said += s
        huds |= {*g.hud(t, 0)[:2]}
    assert (g.you, g.me) == (2, 5) and not g.over  # the last result stays up first
    assert said[-1].endswith(". I win the match, 5 to 2!")
    assert host.hud(t)[0].endswith(": TABLE WINS THE MATCH!")
    t, _, _ = rps_frames(host, t, t + g.RESULT_S + 1.0, "rock")
    assert not g.over and host.again is None  # the hand still there: no PLAY AGAIN or MENU next to it yet
    assert host.hud(t)[1] == "YOU 2 : 5 TABLE · TAKE YOUR HAND AWAY"
    huds |= {*g.hud(t, 0)[:2]}
    t, _, _ = rps_frames(host, t, t + 0.1)
    assert g.over and g.score == 2 and host.best["rps"] == 2 and host.again is not None
    assert host.hud(t) == ("TABLE WINS THE MATCH · YOU 2 : 5 TABLE", "HOLD PLAY AGAIN, OR SAY MENU", "error")
    huds |= {*g.hud(t, 0)[:2]}
    assert all(len(line) <= 44 for line in huds), [line for line in huds if len(line) > 44]  # never cut short
    t, _, _ = rps_frames(host, t, t + 2.5, "rock", at=host.again.c)  # hold PLAY AGAIN
    assert isinstance(host.game, RockPaperScissors) and host.game is not g and (host.game.you, host.game.me) == (0, 0)
    g = host.game
    for _ in range(5):
        t, s = rps_round(host, t, beating)
    assert s[-1].endswith(". You win the match, 5 to 0!")
    t, _, _ = rps_frames(host, t, t + g.RESULT_S + 0.1)
    assert g.over and host.best["rps"] == 5 and host.hud(t)[0] == "YOU WIN THE MATCH · YOU 5 : 0 TABLE"


def test_rps_draws_every_state_with_unique_prim_ids():
    import random

    import numpy as np

    from lightwork.playground import Canvas, Geo

    geo = Geo(np.eye(3), 750.0, 420.0)
    g = RockPaperScissors(0.0, random.Random(1))
    t, seen = 0.0, set()

    def play(shape, seconds):
        nonlocal t
        end = t + seconds
        while t < end - 1e-9:
            t, _, _ = rps_frames(g, t, t + 0.05, shape)
            c = Canvas(geo)
            g.draw(c, t)
            ids = [p.id for p in c.prims]
            assert len(ids) == len(set(ids)), (g.state, ids)
            seen.add("over" if g.over else g.state)

    play(None, 1.0)  # waiting
    play("", 4.0)  # a count, SHOOT, nothing read
    play("paper", 8.0)  # a count, SHOOT, a result
    g.you = g.me = g.FIRST_TO - 1  # the next win or loss ends the match
    while max(g.you, g.me) < g.FIRST_TO and t < 200:
        play("paper", 1.0)
    play(None, g.RESULT_S + 0.5)  # the hand taken away: the match is over
    assert seen == {"wait", "count", "shoot", "noread", "result", "over"}


def test_menu_has_seven_bubbles_whose_labels_do_not_overlap_and_starts_rock_paper_scissors(tmp_path):
    from itertools import combinations

    host = GameHost(state_dir=tmp_path, seed=1)
    host.start("menu", 0.0)
    menu = host.game
    assert [b.key for b in menu.bubbles] == ["helper", "catch", "follow", "draw", "rps", "chess", "objects"]  # OBJECT BOARD last

    def boxes(b):  # the ring with its dwell arc, then each label line (10 % wider than Bubble lays them out)
        x, y = b.c
        out = [(x - b.r - 8, y - b.r - 8, x + b.r + 8, y + b.r + 8)]
        for k, line in enumerate(b.text.split("\n") + ["BEST 99"]):
            half, ly = 1.1 * EM * len(line) / 2, y + b.r + 22 + 26 * k
            out.append((x - half, ly - 13, x + half, ly + 13))
        return out

    for a, b in combinations(menu.bubbles, 2):
        for p in boxes(a):
            for q in boxes(b):
                assert min(p[2], q[2]) <= max(p[0], q[0]) or min(p[3], q[3]) <= max(p[1], q[1]), (a.key, b.key)
    for b in menu.bubbles:
        for x0, y0, x1, y1 in boxes(b):
            assert AREA[0] - 16 <= x0 and x1 <= AREA[2] + 8 and AREA[1] - 8 <= y0 and y1 <= AREA[3]
    rps = next(b for b in menu.bubbles if b.key == "rps")
    t, said = frames(host, 0.0, DWELL_S + 0.3, points=[rps.c])
    assert isinstance(host.game, RockPaperScissors) and said[-1].startswith("Rock, paper, scissors!")


def test_rps_by_voice_and_never_by_one_word(tmp_path):
    host = GameHost(state_dir=tmp_path)
    assert host.hear("Play rock, paper, scissors!", 0.0)[0].startswith("Rock, paper, scissors!")
    assert isinstance(host.game, RockPaperScissors)
    host.start("draw", 1.0)
    host.game.guess = "STONE"
    host.hear("No, it's a rock.", 2.0)  # an answer in draw and guess, not a game
    assert isinstance(host.game, DrawGuess) and host.game.guess == "ROCK"


def shaped(p, shape: str):
    """``p`` (a flat hand at its place on the page) with 3-D landmarks making ``shape``."""
    from lightwork.vision.hands import HandPose

    from ..vision_hands.test_shape import hand_world

    return HandPose(p.img, hand_world(shape), p.side, p.score)


def test_table_rps_reads_the_hand_over_the_table_and_scores(tmp_path):
    from .test_table_scans import Rig
    from .test_tabletop import pose, table

    rig = Rig(game="rps", state_dir=tmp_path)
    flat = pose((560, 640), (590, 540))
    rig.run(0.3, table())
    g = rig.ts.games.game
    assert isinstance(g, RockPaperScissors) and g.state == "wait"
    assert "rps-ring-t" in {p.id for p in rig.page.history[-1][2]}
    rig.run(0.8, table(), poses=lambda t: [shaped(flat, "rock")])
    assert g.state == "count"
    win = beating(g.mine)
    while g.state in ("count", "shoot"):
        rig.run(0.2, table(), poses=lambda t: [shaped(flat, win)])
    assert g.state == "result" and g.yours == win and (g.you, g.me) == (1, 0)
    prims = rig.page.history[-1][2]
    assert any(p.id == "rps-mine" or p.id.startswith("rps-mine-") for p in prims)
    assert next(p for p in prims if p.id == "rps-yours").text == f"YOU: {win.upper()}"
    assert any(line.endswith("You win!") for line in rig.said)
    assert any(line.startswith(f"rps: read {win}") for line in rig.ts.log)
    assert any(line.startswith("hand ") and f": {win} (fingers" in line for line in rig.ts.log)
    assert rig.ts.scans == 0  # no object scans while it plays


def test_no_spoken_hints_when_nothing_listens(tmp_path):
    """3 Nov: voice recognition is off on the rig for now; the hints must not ask for words."""
    host = GameHost(state_dir=tmp_path, voice=False)
    host.start("menu", 0.0)
    line1, line2, _ = host.hud(0.0)
    assert line1 == "PICK ONE: HOLD A FINGER ON IT" and line2 == ""
    host.start("catch", 0.0)
    host.game.over = True
    assert host.hud(1.0)[1] == "HOLD PLAY AGAIN"
    talking = GameHost(state_dir=tmp_path)
    talking.start("catch", 0.0)
    talking.game.over = True
    assert talking.hud(1.0)[1] == "HOLD PLAY AGAIN, OR SAY MENU"
