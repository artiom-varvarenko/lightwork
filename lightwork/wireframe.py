"""Muse Wonder's 3D models: it writes a short program that builds the wireframe, so a tower gets its real lattice from a
loop rather than the 150 points anyone would write out by hand.

The program comes from a model, so it runs like untrusted input. Its syntax tree is checked against a short list first:
arithmetic, loops, functions and lists, the ``math`` module and nothing else, so no imports, no attributes beyond a few
list methods, no names with underscores. Then it runs in a child Python of its own (``-I -S``, no site packages) with no
builtins but a safe few and limits on CPU time, memory and files. All it can do is call ``line(points)``.
"""

from __future__ import annotations

import ast
import json
import math
import subprocess
import sys

MODEL_NAME_CHARS = 28
CPU_S = 4  # a program this long is stuck
MEMORY_BYTES = 768 * 1024 * 1024
MAX_SEGMENTS = 12000  # what comes back beyond this is thinned: the projector has 480 lines
MAX_CODE_CHARS = 20000

PROMPT = """Write a short Python program that builds {what} as a detailed 3D wireframe, as close to the real thing as you \
can: true proportions and its real structure, every part it is known for. A small projector shows it on a table and a \
person turns it with their fingers. Only lines: no surfaces, no text.
Rules. Plain Python only, with `math` already there: no imports, no classes, no files. Call `line(points)` once per \
polyline, points being a list of (x, y, z) tuples, z up, in any units: it gets centred and scaled afterwards. Use \
functions, loops and symmetry to get the detail cheaply: build one leg and repeat it four times, mirror one side to the \
other, sweep circles with 24 or more segments, fill a lattice with cross bracing in a loop. Aim for 1500 to 5000 line \
segments in all, and do not write out coordinate lists by hand beyond what the shape needs. At most 150 lines of code.
Reply with JSON: "name" (what it is, 1 to 3 words) and "code" (the program)."""
RETRY = """

Your last program for this failed with: {error}
Fix that: the same rules apply, so reply with the whole program again."""
SCHEMA = {
    "type": "object",
    "properties": {"name": {"type": "string"}, "code": {"type": "string"}},
    "required": ["name", "code"],
    "additionalProperties": False,
}

_NODES = (
    ast.Module, ast.FunctionDef, ast.arguments, ast.arg, ast.Return, ast.Assign, ast.AugAssign, ast.AnnAssign, ast.For,
    ast.While, ast.If, ast.IfExp, ast.Expr, ast.Pass, ast.Break, ast.Continue, ast.Call, ast.keyword, ast.Name, ast.Load,
    ast.Store, ast.Del, ast.Delete, ast.Constant, ast.BinOp, ast.UnaryOp, ast.BoolOp, ast.Compare, ast.List, ast.Tuple,
    ast.Dict, ast.Set, ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp, ast.comprehension, ast.Subscript,
    ast.Slice, ast.Starred, ast.Lambda, ast.Attribute, ast.operator, ast.unaryop, ast.boolop, ast.cmpop, ast.Import,
    ast.alias,
)
_METHODS = frozenset({"append", "extend", "insert", "pop", "reverse", "copy", "sort", "index", "count", "items", "keys",
                      "values", "get"})
_BUILTINS = ("abs", "all", "any", "bool", "dict", "divmod", "enumerate", "float", "int", "len", "list", "map", "max",
             "min", "pow", "print", "range", "reversed", "round", "set", "sorted", "sum", "tuple", "zip")
_RUNNER = """
import json, math, resource, sys
resource.setrlimit(resource.RLIMIT_CPU, ({cpu}, {cpu}))
resource.setrlimit(resource.RLIMIT_AS, ({mem}, {mem}))
resource.setrlimit(resource.RLIMIT_NOFILE, (16, 16))
resource.setrlimit(resource.RLIMIT_FSIZE, (0, 0))
code = sys.stdin.read()
lines = []
def line(points):
    lines.append([[float(v) for v in p[:3]] for p in points])
    if len(lines) > 40000:
        raise RuntimeError("too many lines")
safe = {{n: __builtins__.__dict__[n] for n in {names!r}}}
safe["print"] = lambda *a, **k: None
exec(compile(code, "model", "exec"), {{"__builtins__": safe, "math": math, "line": line}})
out = json.dumps(lines)
sys.__stdout__.write(out)
"""


class WireError(RuntimeError):
    """The program is refused, or it failed: the message is what the model gets to fix it."""


def check_code(code: str) -> None:
    """Raises :class:`WireError` unless the program stays within the list."""
    if len(code) > MAX_CODE_CHARS:
        raise WireError(f"the program is too long: {len(code)} characters")
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        raise WireError(f"syntax error on line {exc.lineno}: {exc.msg}") from exc
    for node in ast.walk(tree):
        if not isinstance(node, _NODES):
            raise WireError(f"{type(node).__name__} is not allowed")
        if isinstance(node, ast.Import):
            if any(a.name != "math" or a.asname for a in node.names):
                raise WireError("no imports: math is already there")
        if isinstance(node, ast.Name) and node.id.startswith("_"):
            raise WireError(f"the name {node.id} is not allowed")
        if isinstance(node, ast.Attribute):
            on_math = isinstance(node.value, ast.Name) and node.value.id == "math"
            if node.attr.startswith("_") or not (on_math or node.attr in _METHODS):
                raise WireError(f"the attribute .{node.attr} is not allowed")
        if isinstance(node, (ast.FunctionDef, ast.arg, ast.keyword)):
            name = getattr(node, "name", None) or getattr(node, "arg", None) or ""
            if name.startswith("_"):
                raise WireError(f"the name {name} is not allowed")
        if isinstance(node, ast.Constant) and isinstance(node.value, str) and len(node.value) > 200:
            raise WireError("long strings are not allowed")


def run_code(code: str, timeout_s: float = CPU_S + 3.0) -> list:
    """The polylines the program draws. Raises :class:`WireError`."""
    check_code(code)
    code = "\n".join(line for line in code.splitlines() if line.strip() not in ("import math",))
    runner = _RUNNER.format(cpu=CPU_S, mem=MEMORY_BYTES, names=[n for n in _BUILTINS])
    try:
        done = subprocess.run([sys.executable, "-I", "-S", "-c", runner], input=code.encode(), capture_output=True,
                              timeout=timeout_s, close_fds=True, cwd="/", env={})
    except subprocess.TimeoutExpired as exc:
        raise WireError("the program ran too long") from exc
    if done.returncode != 0:
        tail = done.stderr.decode("utf-8", "replace").strip().splitlines()
        raise WireError(tail[-1][:300] if tail else f"the program ended with code {done.returncode}")
    try:
        lines = json.loads(done.stdout)
    except ValueError as exc:
        raise WireError("the program printed something else") from exc
    if not lines:
        raise WireError("the program drew nothing: it never called line(points)")
    return lines


def segments(lines: list) -> int:
    return sum(max(len(p) - 1, 0) for p in lines)


def clean(lines: list) -> tuple[list, float]:
    """Finite polylines of at least two points, centred, within -1 to 1, thinned to ``MAX_SEGMENTS`` by dropping whole
    polylines evenly (a lattice with fewer bars still reads as one). Returns them and how much of the size is z."""
    good = [[p for p in line if len(p) == 3 and all(math.isfinite(v) for v in p)] for line in lines]
    good = [p for p in good if len(p) >= 2]
    if len(good) < 3:
        raise WireError("the program drew fewer than three lines")
    every = [v for line in good for v in line]
    low = [min(v[k] for v in every) for k in range(3)]
    high = [max(v[k] for v in every) for k in range(3)]
    size = max(high[k] - low[k] for k in range(3))
    if size <= 0:
        raise WireError("everything it drew is in one spot")
    centre = [(low[k] + high[k]) / 2 for k in range(3)]
    scale = 2.0 / size
    out = [[[round((v[k] - centre[k]) * scale, 4) for k in range(3)] for v in line] for line in good]
    total = segments(out)
    if total > MAX_SEGMENTS:
        keep = MAX_SEGMENTS / total
        out = [line for i, line in enumerate(out) if int((i + 1) * keep) != int(i * keep)]
    return out, (high[2] - low[2]) / size


def build(brain, what: str, *, timeout_s: float = 150.0, tries: int = 2):
    """Muse Wonder writes the program; it runs. A program that fails goes back to it once, with the error. Returns the
    brain's reply with ``data`` = {"name", "lines"} and ``cost_usd``, ``ms`` summed over the tries."""
    prompt = PROMPT.format(what=what)
    error, ms, cost = "", 0.0, 0.0
    for _ in range(tries):
        reply = brain.ask("helper-model", prompt + (RETRY.format(error=error) if error else ""), [], schema=SCHEMA,
                          effort="low", max_out=12000, timeout_s=timeout_s)
        ms += reply.ms
        cost += reply.cost_usd
        try:
            lines, _ = clean(run_code(str((reply.data or {}).get("code") or "")))
        except WireError as exc:
            error = str(exc)
            continue
        reply.data = {"name": str(reply.data.get("name") or what)[:MODEL_NAME_CHARS], "lines": lines}
        reply.ms, reply.cost_usd = ms, cost
        return reply
    raise WireError(f"the model's program failed twice: {error}")
