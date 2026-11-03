"""Hand games on the table (lightwork.games), driven with synthetic hand points."""

from __future__ import annotations

from lightwork.games import DWELL_S, CatchLight, DrawGuess, FollowLights, GameHost, Menu


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
    g = DrawGuess(0.0, random.Random(0), ask=asked.append)
    t = 0.0
    for k in range(30):  # a stroke across the table
        t += 0.05
        g.update(t, [(200 + 6 * k, 200.0)], [(200 + 6 * k, 200.0)])
    t, _ = frames(g, t, t + 2.5)
    assert len(asked) == 1 and asked[0][:3] == b"\xff\xd8\xff"
    g.answer("cat", "That's a cat!", t)
    assert g.guess == "CAT" and g.say[-1] == "That's a cat!"
    g.update(t + 1, [(300.0, 300.0)], [(300.0, 300.0)])  # drawing again: a new picture
    assert g.guess is None and len(g.strokes) == 1


def test_voice_routing(tmp_path):
    host = GameHost(state_dir=tmp_path)
    assert host.hear("what is this drawing?", 0.0) is None  # not playing: not a game sentence
    assert host.hear("let's play a game", 0.0)[0].startswith("Pick a game")
    assert isinstance(host.game, Menu)
    assert host.hear("follow the lights", 1.0)[0].startswith("Follow")
    assert isinstance(host.game, FollowLights)
    assert host.hear("stop playing", 2.0) == ["Okay, back to the table."] and not host.active


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
