"""Muse Wonder's model programs: what they can build, and what they are not let do."""

from __future__ import annotations

import json
import os
from types import SimpleNamespace

import numpy as np
import pytest

from lightwork.wireframe import MAX_TRIANGLES, WireError, build, check_code, fit, run_code, serve

PROGRAM = """
part("hull", "#ff3030")
box((0, 0, 0), (10, 2, 2))
part("engine", "#ffff00")
cylinder((-1, 0, 0), (1, 0, 0), 0.5)
sphere((3, 0, 0), 0.4, n=8)
torus((-3, 0, 0), 0.6, 0.1, axis=(1, 0, 0))
part("pipes", "#00ffff")
tube([(0, 0, 0), (1, 1, 1), (2, 1, 1)], 0.05)
beam((0, 0, 0), (0, 0, 3), 0.2)
revolve([(0.5, 0), (0.2, 1)], center=(4, 0, 0))
loft([[(0, 0, 0), (1, 0, 0), (1, 1, 0)], [(0, 0, 1), (1, 0, 1), (1, 1, 1)]])
extrude([(0, 0), (1, 0), (1, 1)], 0, 1)
quad((0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0))
for k in range(5):
    line([(k, 0, 0), (k, 1, 1), (k, 2, 0)])
"""


def test_a_program_builds_its_parts_in_order():
    parts = run_code(PROGRAM)
    assert [p.name for p in parts] == ["hull", "engine", "pipes"]
    assert parts[0].color == "#ff3030"
    assert len(parts[0].tris) == 16  # a box: two per side, four per end
    assert parts[2].lines.shape == (10, 6) and parts[1].lines.size == 0
    assert all(p.tris.shape[1] == 9 for p in parts)


def test_drawing_with_no_part_named_goes_in_one():
    assert [p.name for p in run_code("line([(0, 0, 0), (1, 1, 1)])")] == ["model"]


def test_what_comes_back_is_centred_and_within_one():
    parts, half = fit(run_code(PROGRAM))
    every = np.concatenate([np.concatenate([p.tris.reshape(-1, 3), p.lines.reshape(-1, 3)]) for p in parts])
    assert np.abs(every).max() <= 1.0 + 1e-5 and max(half) == pytest.approx(1.0, abs=1e-3)
    assert np.allclose((every.max(axis=0) + every.min(axis=0)) / 2, 0.0, atol=1e-4)


def test_it_is_served_as_a_binary_and_a_manifest_the_page_can_slice_up(tmp_path):
    parts, half = fit(run_code(PROGRAM))
    info = serve(parts, half, "Ship", tmp_path)
    assert info["parts"] == ["hull", "engine", "pipes"] and info["src"].startswith("/pictures/")
    manifest = json.loads((tmp_path / info["src"].rsplit("/", 1)[1]).read_text())
    floats = np.frombuffer((tmp_path / manifest["bin"].rsplit("/", 1)[1]).read_bytes(), dtype=np.float32)
    assert floats.size == sum(p["tri_n"] * 9 + p["line_n"] * 6 for p in manifest["parts"])
    pipes = manifest["parts"][2]
    assert np.allclose(floats[pipes["line_at"]:pipes["line_at"] + pipes["line_n"] * 6], parts[2].lines.reshape(-1))
    box = info["boxes"]["hull"]
    assert box[3] == pytest.approx(1.0, abs=0.2) and len(box) == 6


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
    "def _f():\n    pass",
    "global x",
    "cylinder.__code__",
])
def test_a_program_cannot_reach_out(bad):
    with pytest.raises(WireError):
        run_code(bad)


def test_builtins_are_only_the_safe_few_even_past_the_check():
    for name in ("open", "eval", "exec", "getattr", "globals", "compile", "input", "vars", "dir", "type"):
        with pytest.raises(WireError):
            run_code(f"{name}('x')\nline([(0, 0, 0), (1, 1, 1)])")


def test_the_drawing_functions_helpers_are_out_of_reach():
    for name in ("resource", "sys", "json", "array", "code", "parts", "safe", "env"):
        with pytest.raises(WireError):
            run_code(f"print({name})\nline([(0, 0, 0), (1, 1, 1)])")


def test_it_has_nothing_in_its_environment():
    os.environ["LW_TEST_ONLY"] = "secret"
    try:
        check_code("line([(0, 0, 0), (1, 1, 1)])")
        assert len(run_code("line([(0, 0, 0), (1, 1, 1)])")) == 1
    finally:
        del os.environ["LW_TEST_ONLY"]


def test_a_program_that_never_ends_is_stopped():
    with pytest.raises(WireError, match="too long"):
        run_code("while True:\n    pass", timeout_s=1.0)


def test_a_program_that_eats_memory_is_stopped():
    with pytest.raises(WireError):
        run_code("x = [0] * (10 ** 10)")


def test_too_many_triangles_are_refused_with_a_reason_it_can_act_on():
    code = f"for k in range({MAX_TRIANGLES + 5}):\n    tri((0, 0, 0), (1, 0, 0), (0, 1, 0))"
    with pytest.raises(WireError, match="too many triangles"):
        run_code(code, timeout_s=30.0)


def test_a_program_that_builds_nothing_or_one_spot_is_refused():
    with pytest.raises(WireError, match="nothing"):
        run_code("x = 1")
    with pytest.raises(WireError, match="one spot"):
        fit(run_code("line([(1, 1, 1), (1, 1, 1)])"))


class Brain:
    def __init__(self, *programs):
        self.programs = list(programs)
        self.prompts = []

    def ask(self, role, prompt, jpegs=(), **kw):
        assert role == "helper-model" and jpegs == []
        self.prompts.append(prompt)
        return SimpleNamespace(data={"name": "Thing", "code": self.programs.pop(0)}, ms=1000.0, cost_usd=0.1, model="m", provider="meta")


def test_wonder_gets_its_error_back_once_and_can_fix_it(tmp_path):
    brain = Brain("import os", PROGRAM)
    reply = build(brain, "a thing", tmp_path)
    assert reply.data["name"] == "Thing" and reply.data["parts"] == ["hull", "engine", "pipes"]
    assert "a thing" in brain.prompts[0] and "no imports" in brain.prompts[1]
    assert reply.ms == 2000.0 and reply.cost_usd == pytest.approx(0.2)


def test_two_failures_end_it(tmp_path):
    with pytest.raises(WireError, match="failed twice"):
        build(Brain("import os", "import os"), "a thing", tmp_path)
