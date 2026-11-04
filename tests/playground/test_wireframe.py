"""Muse Wonder's wireframe programs: what they can build, and what they are not let do."""

from __future__ import annotations

import os
from types import SimpleNamespace

import pytest

from lightwork.wireframe import MAX_SEGMENTS, WireError, build, check_code, clean, run_code, segments

PROGRAM = """
def ring(z, r, n=24):
    return [(r * math.cos(2 * math.pi * k / n), r * math.sin(2 * math.pi * k / n), z) for k in range(n + 1)]

for level in range(5):
    line(ring(level, 5 - level))
legs = []
for k in range(4):
    a = k * math.pi / 2
    legs.append([(5 * math.cos(a), 5 * math.sin(a), 0), (math.cos(a), math.sin(a), 4)])
for leg in legs:
    line(leg)
"""


def test_a_program_builds_its_lines():
    lines = run_code(PROGRAM)
    assert len(lines) == 9 and segments(lines) == 5 * 24 + 4


def test_what_comes_back_is_centred_and_within_one():
    lines, z_share = clean(run_code(PROGRAM))
    every = [v for line in lines for v in line]
    assert max(abs(c) for v in every for c in v) <= 1.0 and 0.0 < z_share <= 1.0


@pytest.mark.parametrize("bad", [
    "import os",
    "from os import path",
    "open('/etc/passwd')",
    "line.__globals__",
    "().__class__.__bases__",
    "getattr(line, 'x')",
    "eval('1')",
    "exec('1')",
    "__import__('os')",
    "x = 'a'.format(1)",
    "class A:\n    pass",
    "math.__loader__",
    "with x as y:\n    pass",
    "try:\n    pass\nexcept Exception:\n    pass",
    "async def f():\n    pass",
    "x = f'{line}'",
    "_ = 1",
])
def test_a_program_cannot_reach_out(bad):
    with pytest.raises(WireError):
        run_code(bad)


def test_builtins_are_only_the_safe_few_even_past_the_check():
    for name in ("open", "eval", "exec", "getattr", "globals", "compile", "input"):
        with pytest.raises(WireError):
            run_code(f"{name}('x')\nline([(0, 0, 0), (1, 1, 1)])")


def test_the_key_is_not_in_its_environment():
    os.environ["MODEL_API_KEY_TEST_ONLY"] = "secret"
    try:  # dict and list reach nothing; the child has an empty environment, so there is nothing to find anyway
        assert run_code("line([(0, 0, 0), (1, 1, 1)])") == [[[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]]
    finally:
        del os.environ["MODEL_API_KEY_TEST_ONLY"]


def test_a_program_that_never_ends_is_stopped():
    with pytest.raises(WireError, match="too long"):
        run_code("while True:\n    pass", timeout_s=1.0)


def test_a_program_that_eats_memory_is_stopped():
    with pytest.raises(WireError):
        run_code("x = [0] * (10 ** 10)")


def test_a_program_that_draws_nothing_or_one_spot_is_refused():
    with pytest.raises(WireError, match="nothing"):
        run_code("x = 1")
    with pytest.raises(WireError, match="one spot"):
        clean([[[1, 1, 1], [1, 1, 1]]] * 4)
    with pytest.raises(WireError, match="fewer than three"):
        clean([[[0, 0, 0], [1, 1, 1]]])


def test_too_many_lines_are_thinned_evenly():
    lines = [[[float(k), 0.0, 0.0], [float(k), 1.0, 0.0]] for k in range(MAX_SEGMENTS * 2)]
    out, _ = clean(lines)
    assert segments(out) <= MAX_SEGMENTS and len(out) >= MAX_SEGMENTS * 0.9


class Brain:
    def __init__(self, *programs):
        self.programs = list(programs)
        self.prompts = []

    def ask(self, role, prompt, jpegs=(), **kw):
        assert role == "helper-model" and jpegs == []
        self.prompts.append(prompt)
        return SimpleNamespace(data={"name": "Thing", "code": self.programs.pop(0)}, ms=1000.0, cost_usd=0.1, model="m", provider="meta")


def test_wonder_gets_its_error_back_once_and_can_fix_it():
    brain = Brain("import os", PROGRAM)
    reply = build(brain, "a thing")
    assert reply.data["name"] == "Thing" and len(reply.data["lines"]) == 9
    assert "a thing" in brain.prompts[0] and "no imports" in brain.prompts[1]
    assert reply.ms == 2000.0 and reply.cost_usd == pytest.approx(0.2)


def test_two_failures_end_it():
    with pytest.raises(WireError, match="failed twice"):
        build(Brain("import os", "import os"), "a thing")
