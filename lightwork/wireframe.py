"""Muse Wonder's 3D models: it writes a short program that builds the thing part by part, inside included, so a ship has
an engine room to slice open and a tower its real lattice from a loop.

The program comes from a model, so it runs like untrusted input. Its syntax tree is checked against a short list first:
arithmetic, loops, functions and lists, the ``math`` module and nothing else, so no imports, no attributes beyond a few
list methods, no names with underscores. Then it runs in a child Python of its own (``-I -S``, no site packages, an empty
environment) with no builtins but a safe few and limits on CPU time, memory and files. All it can do is call the drawing
functions below. What it builds goes to the page as one binary file plus a small manifest; web/js/solid.js draws it.
"""

from __future__ import annotations

import ast
import hashlib
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

MODEL_NAME_CHARS = 28
CPU_S = 6  # a program this long is stuck
MEMORY_BYTES = 1024 * 1024 * 1024
MAX_TRIANGLES = 300_000  # far more than 854 x 480 can show
MAX_CODE_CHARS = 30000

PROMPT = """Write a short Python program that builds {what} as a detailed 3D model made of solid parts, as close to the \
real thing as you can: true proportions and its real structure. Model what is inside as well as the outside, as separate \
parts, because a person will slice it open with their hands and move through it: a ship has its hull and decks and, inside \
them, an engine room with engines, a shaft to the propeller, cabins and holds; a car its body and, inside, engine, seats \
and gearbox. If a thing has no inside (a bicycle, a lattice tower), give it every part it is known for instead.
These functions are already there. Points are (x, y, z) tuples, z up, x along its length, in any units: it gets centred \
and scaled afterwards. Angles are in radians.
- part(name, color): what follows belongs to this part until the next part(). The name is a plain word a person would say \
("engine", "hull", "front wheel"). The color is a bright "#rrggbb": a projector shows dark colors as nothing.
- box(center, size, yaw=0): size is (dx, dy, dz). beam(a, b, w, h=None): a square bar from a to b.
- cylinder(a, b, r, r2=None, n=24): from a to b, a cone if r2 differs. sphere(center, r, n=16, size=None): size (rx, ry, rz) \
makes an ellipsoid. torus(center, r, tube, axis=(0, 0, 1), n=32): a ring around axis. tube(points, r, n=8): a pipe along a path.
- revolve(profile, center=(0, 0, 0), axis=(0, 0, 1), n=32): a lathe; profile is a list of (radius, height).
- loft(rings): a skin through closed rings of equal length, each a list of points: for hulls, fuselages, bodies.
- extrude(polygon, z0, z1): an outline of (x, y) points raised from z0 to z1. tri(a, b, c), quad(a, b, c, d) for anything else.
- line(points): thin lines, for cables, spokes, lattice bracing: far cheaper than solids when there are thousands.
Rules. Plain Python only, with `math` already there: no imports, no classes, no files. Use functions, loops and symmetry to \
get the detail cheaply: build one leg and repeat it, mirror one side to the other, loop over frames, spokes and lattice \
bars. 6 to 25 parts. At most 200 lines of code. No text.
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
_API = ("part", "line", "tri", "quad", "box", "beam", "cylinder", "sphere", "torus", "tube", "revolve", "loft", "extrude")
# Runs in the child, before the program; the check above does not apply to it. Writes a JSON header line, then the floats.
_RUNNER = r"""
import array, json, math, resource, sys
resource.setrlimit(resource.RLIMIT_CPU, (%(cpu)d, %(cpu)d))
resource.setrlimit(resource.RLIMIT_AS, (%(mem)d, %(mem)d))
resource.setrlimit(resource.RLIMIT_NOFILE, (16, 16))
resource.setrlimit(resource.RLIMIT_FSIZE, (0, 0))
code = sys.stdin.read()
parts, cur, n_tris = [], None, 0

def part(name="model", color="#8fd3ff"):
    global cur
    cur = {"name": str(name)[:30], "color": str(color)[:9], "tris": array.array("f"), "lines": array.array("f")}
    parts.append(cur)

def _now():
    if cur is None:
        part()
    return cur

def _v(p):
    return (float(p[0]), float(p[1]), float(p[2]))

def _sub(a, b):
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])

def _add(a, b, s=1.0):
    return (a[0] + b[0] * s, a[1] + b[1] * s, a[2] + b[2] * s)

def _cross(a, b):
    return (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0])

def _unit(a):
    d = math.sqrt(a[0] * a[0] + a[1] * a[1] + a[2] * a[2]) or 1.0
    return (a[0] / d, a[1] / d, a[2] / d)

def _frame(axis):
    w = _unit(axis)
    u = _unit(_cross(w, (0.0, 0.0, 1.0) if abs(w[2]) < 0.9 else (1.0, 0.0, 0.0)))
    return u, _cross(w, u)

def _ring(c, u, v, r, n):
    return [(c[0] + r * (math.cos(2 * math.pi * k / n) * u[0] + math.sin(2 * math.pi * k / n) * v[0]),
             c[1] + r * (math.cos(2 * math.pi * k / n) * u[1] + math.sin(2 * math.pi * k / n) * v[1]),
             c[2] + r * (math.cos(2 * math.pi * k / n) * u[2] + math.sin(2 * math.pi * k / n) * v[2])) for k in range(n)]

def tri(a, b, c):
    global n_tris
    n_tris += 1
    if n_tris > %(max_tris)d:
        raise RuntimeError("too many triangles: use fewer segments, or line() for the fine detail")
    t = _now()["tris"]
    t.extend(_v(a)); t.extend(_v(b)); t.extend(_v(c))

def quad(a, b, c, d):
    tri(a, b, c); tri(a, c, d)

def line(points):
    pts = [_v(p) for p in points]
    out = _now()["lines"]
    for a, b in zip(pts, pts[1:]):
        out.extend(a); out.extend(b)

def _band(ra, rb):
    n = len(ra)
    for k in range(n):
        quad(ra[k], ra[(k + 1) %% n], rb[(k + 1) %% n], rb[k])

def _cap(ring):
    n = len(ring)
    c = (sum(p[0] for p in ring) / n, sum(p[1] for p in ring) / n, sum(p[2] for p in ring) / n)
    for k in range(n):
        tri(c, ring[k], ring[(k + 1) %% n])

def box(center, size, yaw=0.0):
    c, s = _v(center), _v(size)
    cy, sy = math.cos(yaw), math.sin(yaw)
    def at(dx, dy, dz):
        x, y = dx * s[0] / 2, dy * s[1] / 2
        return (c[0] + x * cy - y * sy, c[1] + x * sy + y * cy, c[2] + dz * s[2] / 2)
    low = [at(-1, -1, -1), at(1, -1, -1), at(1, 1, -1), at(-1, 1, -1)]
    high = [at(-1, -1, 1), at(1, -1, 1), at(1, 1, 1), at(-1, 1, 1)]
    _band(low, high); _cap(low); _cap(high)

def beam(a, b, w, h=None):
    a, b = _v(a), _v(b)
    u, v = _frame(_sub(b, a))
    hw, hh = w / 2, (h if h is not None else w) / 2
    def corners(c):
        return [_add(_add(c, u, sx * hw), v, sy * hh) for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1))]
    ra, rb = corners(a), corners(b)
    _band(ra, rb); _cap(ra); _cap(rb)

def cylinder(a, b, r, r2=None, n=24, caps=True):
    a, b = _v(a), _v(b)
    n = max(3, int(n))
    u, v = _frame(_sub(b, a))
    ra, rb = _ring(a, u, v, r, n), _ring(b, u, v, r if r2 is None else r2, n)
    _band(ra, rb)
    if caps:
        _cap(ra); _cap(rb)

def sphere(center, r=1.0, n=16, size=None):
    c = _v(center)
    rx, ry, rz = _v(size) if size is not None else (r, r, r)
    n = max(4, int(n))
    rings = []
    for i in range(1, n):
        lat = math.pi * i / n - math.pi / 2
        rings.append([(c[0] + rx * math.cos(lat) * math.cos(2 * math.pi * k / (2 * n)),
                       c[1] + ry * math.cos(lat) * math.sin(2 * math.pi * k / (2 * n)),
                       c[2] + rz * math.sin(lat)) for k in range(2 * n)])
    for ra, rb in zip(rings, rings[1:]):
        _band(ra, rb)
    south, north = (c[0], c[1], c[2] - rz), (c[0], c[1], c[2] + rz)
    m = len(rings[0])
    for k in range(m):
        tri(south, rings[0][k], rings[0][(k + 1) %% m])
        tri(north, rings[-1][k], rings[-1][(k + 1) %% m])

def torus(center, r, tube=0.1, axis=(0.0, 0.0, 1.0), n=32, m=10):
    c, w = _v(center), _unit(_v(axis))
    u, v = _frame(w)
    n, m = max(6, int(n)), max(3, int(m))
    rings = []
    for k in range(n):
        a = 2 * math.pi * k / n
        out = _add(_add((0.0, 0.0, 0.0), u, math.cos(a)), v, math.sin(a))
        rings.append(_ring(_add(c, out, r), out, w, tube, m))
    for k in range(n):
        _band(rings[k], rings[(k + 1) %% n])

def tube(points, r=0.05, n=8):
    pts = [_v(p) for p in points]
    for a, b in zip(pts, pts[1:]):
        cylinder(a, b, r, n=n, caps=False)

def revolve(profile, center=(0.0, 0.0, 0.0), axis=(0.0, 0.0, 1.0), n=32):
    c, w = _v(center), _unit(_v(axis))
    u, v = _frame(w)
    n = max(6, int(n))
    rings = [_ring(_add(c, w, float(h)), u, v, float(rad), n) for rad, h in profile]
    for ra, rb in zip(rings, rings[1:]):
        _band(ra, rb)

def loft(rings, caps=True):
    rs = [[_v(p) for p in ring] for ring in rings]
    for ra, rb in zip(rs, rs[1:]):
        if len(ra) != len(rb):
            raise RuntimeError("loft: all rings need the same number of points")
        _band(ra, rb)
    if caps and rs:
        _cap(rs[0]); _cap(rs[-1])

def extrude(polygon, z0, z1):
    low = [(float(p[0]), float(p[1]), float(z0)) for p in polygon]
    high = [(float(p[0]), float(p[1]), float(z1)) for p in polygon]
    _band(low, high); _cap(low); _cap(high)

safe = {n: __builtins__.__dict__[n] for n in %(names)r}
safe["print"] = lambda *a, **k: None
env = {"__builtins__": safe, "math": math}
for name in %(api)r:
    env[name] = globals()[name]
exec(compile(code, "model", "exec"), env)
head = [{"name": p["name"], "color": p["color"], "tris": len(p["tris"]) // 9, "lines": len(p["lines"]) // 6} for p in parts]
out = sys.__stdout__.buffer
out.write(json.dumps(head).encode() + b"\n")
for p in parts:
    out.write(p["tris"].tobytes())
    out.write(p["lines"].tobytes())
"""


class WireError(RuntimeError):
    """The program is refused, or it failed: the message is what the model gets to fix it."""


@dataclass
class Part:
    name: str
    color: str
    tris: np.ndarray  # N x 9, float32
    lines: np.ndarray  # M x 6, float32


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


def run_code(code: str, timeout_s: float = CPU_S + 4.0) -> list[Part]:
    """The parts the program builds, in its order. Raises :class:`WireError`."""
    check_code(code)
    code = "\n".join(line for line in code.splitlines() if line.strip() != "import math")
    runner = _RUNNER % {"cpu": CPU_S, "mem": MEMORY_BYTES, "max_tris": MAX_TRIANGLES, "names": _BUILTINS, "api": _API}
    try:
        done = subprocess.run([sys.executable, "-I", "-S", "-c", runner], input=code.encode(), capture_output=True,
                              timeout=timeout_s, close_fds=True, cwd="/", env={})
    except subprocess.TimeoutExpired as exc:
        raise WireError("the program ran too long") from exc
    if done.returncode != 0:
        tail = done.stderr.decode("utf-8", "replace").strip().splitlines()
        raise WireError(tail[-1][:300] if tail else f"the program ended with code {done.returncode}")
    head, _, body = done.stdout.partition(b"\n")
    try:
        heads = json.loads(head)
    except ValueError as exc:
        raise WireError("the program wrote something else") from exc
    floats = np.frombuffer(body, dtype=np.float32)
    parts, at = [], 0
    for h in heads:
        t, m = h["tris"] * 9, h["lines"] * 6
        parts.append(Part(h["name"], h["color"], floats[at:at + t].reshape(-1, 9), floats[at + t:at + t + m].reshape(-1, 6)))
        at += t + m
    parts = [p for p in parts if len(p.tris) or len(p.lines)]
    if not parts:
        raise WireError("the program built nothing: it never called a drawing function")
    return parts


def fit(parts: list[Part]) -> tuple[list[Part], list[float]]:
    """Centred and within -1 to 1; returns the parts and the half size along x, y, z. Things that are not finite are
    dropped; a model all in one spot is refused."""
    kept = []
    for p in parts:
        tris = p.tris[np.isfinite(p.tris).all(axis=1)]
        lines = p.lines[np.isfinite(p.lines).all(axis=1)]
        if len(tris) or len(lines):
            kept.append(Part(p.name, p.color, tris, lines))
    if not kept:
        raise WireError("the program built nothing finite")
    every = np.concatenate([np.concatenate([p.tris.reshape(-1, 3), p.lines.reshape(-1, 3)]) for p in kept])
    low, high = every.min(axis=0), every.max(axis=0)
    size = float((high - low).max())
    if size <= 1e-9:
        raise WireError("everything it built is in one spot")
    centre, scale = (low + high) / 2, 2.0 / size
    fitted = [Part(p.name, p.color, ((p.tris.reshape(-1, 3) - centre) * scale).reshape(-1, 9).astype(np.float32),
                   ((p.lines.reshape(-1, 3) - centre) * scale).reshape(-1, 6).astype(np.float32)) for p in kept]
    half = ((high - low) * scale / 2).tolist()
    return fitted, [round(v, 4) for v in half]


def _box_of(p: Part) -> list[float]:
    """A part's middle and half size, model units: what the camera goes to when it is asked for."""
    pts = np.concatenate([p.tris.reshape(-1, 3), p.lines.reshape(-1, 3)])
    low, high = pts.min(axis=0), pts.max(axis=0)
    return [round(float(v), 4) for v in (*((low + high) / 2), *((high - low) / 2))]


def serve(parts: list[Part], half: list[float], name: str, folder: Path) -> dict:
    """Writes the binary and its manifest where the page loads them from. Returns what the helper needs to know."""
    body = b"".join(p.tris.tobytes() + p.lines.tobytes() for p in parts)
    bin_name = f"{hashlib.sha256(body).hexdigest()[:32]}.bin"
    entries, at = [], 0
    for p in parts:
        entries.append({"name": p.name, "color": p.color, "tri_at": at, "tri_n": len(p.tris),
                        "line_at": at + p.tris.size, "line_n": len(p.lines)})
        at += p.tris.size + p.lines.size
    manifest = {"name": name, "bin": f"/pictures/{bin_name}", "half": half, "parts": entries}
    text = json.dumps(manifest, separators=(",", ":")).encode()
    manifest_name = f"{hashlib.sha256(text).hexdigest()[:32]}.json"
    folder.mkdir(parents=True, exist_ok=True)
    (folder / bin_name).write_bytes(body)
    (folder / manifest_name).write_bytes(text)
    return {"name": name, "src": f"/pictures/{manifest_name}", "half": half, "parts": [p.name for p in parts],
            "boxes": {p.name: _box_of(p) for p in parts}, "triangles": sum(len(p.tris) for p in parts),
            "lines": sum(len(p.lines) for p in parts)}


def build(brain, what: str, folder: Path, *, timeout_s: float = 170.0, tries: int = 2):
    """Muse Wonder writes the program; it runs. A program that fails goes back to it once, with the error. Returns the
    brain's reply with ``data`` as from :func:`serve` and ``cost_usd``, ``ms`` summed over the tries."""
    prompt = PROMPT.format(what=what)
    error, ms, cost = "", 0.0, 0.0
    for _ in range(tries):
        reply = brain.ask("helper-model", prompt + (RETRY.format(error=error) if error else ""), [], schema=SCHEMA,
                          effort="low", max_out=16000, timeout_s=timeout_s)
        ms += reply.ms
        cost += reply.cost_usd
        try:
            parts, half = fit(run_code(str((reply.data or {}).get("code") or "")))
        except WireError as exc:
            error = str(exc)
            continue
        reply.data = serve(parts, half, str(reply.data.get("name") or what)[:MODEL_NAME_CHARS], Path(folder))
        reply.ms, reply.cost_usd = ms, cost
        return reply
    raise WireError(f"the model's program failed twice: {error}")
