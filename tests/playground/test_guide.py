"""Guided jobs (lightwork.guide): plans, quantities and the step machine, on synthetic objects and hands."""

from __future__ import annotations

import numpy as np
import pytest

from lightwork.guide import (
    ADVANCE_S,
    Guide,
    Seen,
    builtin_plans,
    capacity_ml,
    find_builtin,
    load_plan,
    parse_plan,
    scale_plan,
    scoops_for,
    servings,
)
from lightwork.tabletop import command_of, is_plan_request


def box(cx, cy, w=60, h=60):
    return np.array([(cx - w / 2, cy - h / 2), (cx + w / 2, cy - h / 2), (cx + w / 2, cy + h / 2),
                     (cx - w / 2, cy + h / 2)], dtype=np.float64)


def seen(uid, item, name, cx, cy, held=False, w=60, h=60, mm=None):
    return Seen(uid=uid, item=item, name=name, poly=box(cx, cy, w, h), held=held, sides_mm=mm)


def hand_at(x, y):
    return [[(x, y), (x + 5, y), (x, y + 5), (x - 5, y), (x, y - 5)]]


SIMPLE = {
    "title": "Test job",
    "items": [
        {"id": "flour", "name": "flour", "kind": "material", "amount": 120, "unit": "g"},
        {"id": "salt", "name": "salt", "kind": "material", "amount": 1, "unit": "pinch"},
        {"id": "bowl", "name": "bowl", "kind": "container"},
        {"id": "scoop", "name": "measuring scoop", "kind": "measure"},
    ],
    "steps": [
        {"do": "transfer", "say": "Add the flour", "item": "flour", "into": "bowl", "count": 3},
        {"do": "timer", "say": "Wait", "seconds": 5},
        {"do": "check", "say": "Look", "target": "bowl", "question": "Is it smooth?"},
        {"do": "confirm", "say": "Clean up"},
    ],
}


def table():
    return [seen(1, 1, "flour", 200, 200), seen(2, 2, "salt", 300, 200), seen(3, 3, "bowl", 500, 220, w=90, h=90),
            seen(4, 4, "measuring scoop", 380, 330, mm=(120.0, 62.0))]


# ---- plans


def test_builtin_plans_load_and_validate():
    names = set(builtin_plans())
    assert {"pancakes", "serial-dilution", "seedlings", "kit-return"} <= names
    for n in names:
        plan = load_plan(n)
        assert plan.items and plan.steps


def test_bad_reference_is_refused():
    bad = dict(SIMPLE, steps=[{"do": "transfer", "say": "x", "item": "sugar", "into": "bowl"}])
    with pytest.raises(ValueError, match="sugar"):
        parse_plan(bad)


def test_transfer_without_destination_becomes_confirm_and_names_resolve_to_ids():
    plan = parse_plan(dict(SIMPLE, steps=[{"do": "transfer", "say": "Add", "item": "flour"},
                                          {"do": "transfer", "say": "Add", "item": "flour", "into": "bowl"}]))
    assert plan.steps[0].do == "confirm"
    assert plan.steps[1].into == "bowl"


def test_scaling_rounds_pieces_up_and_keeps_pinches():
    plan = scale_plan(load_plan("pancakes"), 4)
    by = {it.id: it for it in plan.items}
    assert by["flour"].amount == 250
    assert by["egg"].amount == 2
    assert by["salt"].amount == 1 and by["salt"].unit == "pinch"
    assert servings("pancakes for four people please") == 4.0
    found = find_builtin("can you help me make pancakes for 6")
    assert found is not None and found[0].title == "Pancakes" and found[1] == 6.0


def test_capacity_and_scoops():
    assert capacity_ml("tablespoon", None) == 15.0
    assert capacity_ml("syringe", None, known=10.0) == 10.0
    cap = capacity_ml("measuring scoop", (120.0, 62.0))
    assert 55 < cap < 70  # a hemisphere 62 mm across
    plan = scale_plan(load_plan("pancakes"), 4)
    flour = next(it for it in plan.items if it.id == "flour")
    n, per = scoops_for(flour, cap)
    assert n == round(250 / 0.53 / cap) and 25 < per < 40
    assert scoops_for(flour, 2000.0) is None  # less than half a scoop: not measured that way


# ---- the step machine


def run_until_steps(g: Guide, objs, t=0.0):
    out = g.update(t, objs, [])
    assert g.phase == "steps", out
    return out


def test_gather_moves_on_when_everything_is_there():
    g = Guide(parse_plan(SIMPLE))
    assert g.update(0.0, table()[:2], []) == []
    assert g.phase == "gather"
    assert any("BOWL" in line for line in g.hud(0.0)[:2])
    said = g.update(1.0, table(), [])
    assert g.phase == "steps" and g.k == 0
    assert said[0] == "Everything is here." and said[1].startswith("Step 1.")


def test_trips_from_source_to_bowl_are_counted_and_the_step_finishes():
    g = Guide(parse_plan(SIMPLE))
    objs = table()
    run_until_steps(g, objs)
    t = 1.0
    said = []
    for _ in range(3):
        said += g.update(t, objs, hand_at(200, 200))  # at the flour
        t += 0.5
        said += g.update(t, objs, hand_at(500, 220))  # into the bowl
        t += 0.5
    assert g.st.count == 3
    assert said[-1] == "Three. That's all 3."
    assert g.update(t + ADVANCE_S + 0.01, objs, [])[0].startswith("Step 2.")
    assert g.k == 1


def test_hovering_over_the_bowl_without_the_source_does_not_count():
    g = Guide(parse_plan(SIMPLE))
    objs = table()
    run_until_steps(g, objs)
    for k in range(6):
        g.update(1.0 + k * 0.4, objs, hand_at(500, 220))
    assert g.st.count == 0


def test_one_too_many_is_called_out():
    g = Guide(parse_plan(SIMPLE))
    objs = table()
    run_until_steps(g, objs)
    t = 1.0
    g.st.target = 2
    said = []
    for _ in range(3):
        said += g.update(t, objs, hand_at(200, 200))
        said += g.update(t + 0.5, objs, hand_at(500, 220))
        t += 1.0
        if g.st.done_t is not None:
            g.st.done_t = None  # keep the step open, as if the extra scoop came at once
    assert g.st.count == 3
    assert any("one too many" in s for s in said)
    assert "error" in g.sfx
    assert g.hud(t)[2] == "error"


def test_wrong_item_picked_up_is_called_out_once():
    g = Guide(parse_plan(SIMPLE))
    objs = table()
    run_until_steps(g, objs)
    held = [seen(1, 1, "flour", 200, 200), seen(2, 2, "salt", 300, 200, held=True)] + objs[2:]
    said = g.update(1.0, held, []) + g.update(1.7, held, []) + g.update(2.5, held, [])
    assert said == ["Careful, that's the salt. This step needs the flour."]
    assert g.view(held[1], 2.6)[0] == "error"


def test_wrong_cup_in_a_dilution_is_caught():
    g = Guide(load_plan("serial-dilution"))
    cups = [seen(10 + k, 1, "clear cup", 150 + 90 * k, 250) for k in range(6)]
    objs = cups + [seen(30, 2, "water bottle", 150, 120), seen(31, 3, "blue water", 300, 120),
                   seen(32, 4, "syringe", 450, 120)]
    g.update(0.0, objs, [])
    assert g.phase == "steps"
    g.command("next", 0.1)  # water
    g.command("next", 0.2)  # dye into cup 1
    assert g.plan.steps[g.k].say == "Move 10 ml from cup 1 to cup 2"
    said = g.update(1.0, objs, hand_at(150, 250))  # at cup 1, the source
    said += g.update(1.5, objs, hand_at(330, 250))  # cup 3, not cup 2
    said += g.update(2.0, objs, hand_at(330, 250))
    assert said == ["Stop: that's cup 3. This goes into cup 2."]
    assert g.st.count == 0
    said = g.update(3.0, objs, hand_at(240, 250))  # cup 2
    assert g.st.count == 1 and g.st.done_t is not None


def test_one_into_each_counts_each_destination_once():
    g = Guide(load_plan("serial-dilution"))
    cups = [seen(10 + k, 1, "clear cup", 150 + 90 * k, 250) for k in range(6)]
    objs = cups + [seen(30, 2, "water bottle", 150, 120), seen(31, 3, "blue water", 300, 120),
                   seen(32, 4, "syringe", 450, 120)]
    g.update(0.0, objs, [])
    assert g.st.target == 5
    t = 1.0
    said = []
    for x in (240, 330, 330):  # cups 2, 3, then 3 again
        said += g.update(t, objs, hand_at(150, 120))  # the water bottle
        said += g.update(t + 0.4, objs, hand_at(x, 250))
        t += 1.0
    assert g.st.count == 2
    assert said[-1] == "Cup 3 already has it."
    for x in (420, 510, 600):
        g.update(t, objs, hand_at(150, 120))
        g.update(t + 0.4, objs, hand_at(x, 250))
        t += 1.0
    assert g.st.count == 5 and g.st.done_t is not None


def test_timer_and_check_and_commands():
    g = Guide(parse_plan(SIMPLE))
    objs = table()
    run_until_steps(g, objs)
    said = g.command("next", 1.0)
    assert said == ["Step 2. Wait. 5 seconds."]
    assert g.hud(2.0)[1] == "0:04"
    g.update(6.1, objs, [])
    assert g.st.done_t is not None
    said = g.update(6.11 + ADVANCE_S, objs, [])
    assert said[0].startswith("Step 3.") and g.want_check == (2, "bowl")
    assert g.check_result(2, False, "Still lumpy.", 9.0) == ["Still lumpy. Say check when you want me to look again."]
    assert g.command("check", 10.0) == ["Looking."] and g.want_check == (2, "bowl")
    assert g.check_result(2, True, "Smooth.", 11.0) == ["Smooth."]
    g.update(11.01 + ADVANCE_S, objs, [])
    assert g.k == 3
    assert g.command("back", 12.0)[0].startswith("Step 3.")
    assert g.command("stop", 13.0) == ["Stopped Test job."] and g.phase == "done"


def test_undo_and_how_many():
    g = Guide(parse_plan(SIMPLE))
    objs = table()
    run_until_steps(g, objs)
    g.update(1.0, objs, hand_at(200, 200))
    g.update(1.5, objs, hand_at(500, 220))
    assert g.command("how many", 2.0) == ["1 of 3."]
    assert g.command("undo", 2.1) == ["Back to 0."]


def test_voice_routing_words():
    assert command_of("next") == "next"
    assert command_of("okay, next step please") == "next"
    assert command_of("go back") == "back"
    assert command_of("that didn't count") == "undo"
    assert command_of("what is this thing over here") is None
    assert is_plan_request("can you help me make pancakes for four")
    assert is_plan_request("I want to pack my camera bag")
    assert not is_plan_request("what is this")
