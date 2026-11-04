"""Pure parser of recorded responses (D2): raw bytes -> role result. Replay and verify call it on recorded bytes, so
it must be deterministic and never touch the network.

``parse_response`` returns ``{"status", "result", "error", "usage"}``:
- ``ok``: verdict = validated C9 ``Verdict`` dict (``""``/``"none"``/``"null"`` contradiction -> null);
  ground = ``{"items": [{name, x, y} | {name, bbox: [x0, y0, x1, y1]}]}`` on the 0-1000 grid of the sent
  image (ints, clamped), plus ``"counts": {name: n}`` for count replies; why = ``{"text"}``; segment =
  ``{"masks": [C9 Mask dicts in sent-image px], "size": [w, h] | None}``; author = ``{"stepgraph": C3 dict
  without id/source}``.
- ``invalid``: the reply does not fit the role; ``result = {"text": reply}``, ``error`` says why (for the repair retry).
- ``incomplete``: the output budget ran out (``error`` ``max_output_tokens`` or ``length``); ``result = {"text"}``.
- ``error``: HTTP error body, failed/refused/cut stream; ``result = {}``.
"""

from __future__ import annotations

import json
import math
import re
from typing import Any

import numpy as np
from meta_sam_parser import CompletedOutcome, decode_mask_to_raster, image_segmentation_format
from pydantic import ValidationError

from lightwork.contracts.backend import GroundBox, GroundPoint, Verdict
from lightwork.contracts.canonical import q_px

from .errors import ResultError
from .prompts import author, why
from .requests import ROLES
from .wire import CHAT, RESPONSES, read_body

GRID_MAX = 1000
MASK_LONG_SIDE = 1024  # SAM caps a mask's long side (4 Nov: 1024x616 for a box of 1028x618)
_NULLISH = ("", "none", "null")
_FENCE = re.compile(r"^```[A-Za-z0-9_-]*[ \t]*\r?\n(.*?)\r?\n?```$", re.S)
_BOX_TOKEN = re.compile(r"<\|box;([^|]*)\|>")


def parse_response(role: str, raw: bytes, kind: str, endpoint: str) -> dict[str, Any]:
    if role not in ROLES:
        raise ValueError(f"unknown role {role!r}")
    if role == "segment" and endpoint != RESPONSES:
        raise ValueError("segment responses come from the Responses API")
    if endpoint not in (RESPONSES, CHAT):
        raise ValueError(f"unknown endpoint {endpoint!r}")
    reading = read_body(raw, kind, endpoint)
    if reading.status == "error":
        return {"status": "error", "result": {}, "error": reading.error, "usage": reading.usage}
    if reading.status == "incomplete":
        return {"status": "incomplete", "result": {"text": reading.text}, "error": reading.error, "usage": reading.usage}
    try:
        result = _ROLE[role](reading.text)
    except ResultError as e:
        return {"status": "invalid", "result": {"text": reading.text}, "error": str(e), "usage": reading.usage}
    return {"status": "ok", "result": result, "error": None, "usage": reading.usage}


# ---------------------------------------------------------------- JSON roles


def json_reply(text: str) -> Any:
    """The JSON value of a reply; one surrounding Markdown code fence is tolerated (Claude stand-ins, Glimmer)."""
    s = text.strip()
    if not s:
        raise ResultError("empty reply")
    m = _FENCE.match(s)
    if m:
        s = m.group(1).strip()
    try:
        return json.loads(s, parse_constant=_bad_constant)
    except json.JSONDecodeError as e:
        raise ResultError(f"reply is not JSON ({e.msg} at char {e.pos})") from None


def _bad_constant(value: str):
    raise ResultError(f"non-finite JSON number {value}")


def _verdict(text: str) -> dict[str, Any]:
    obj = json_reply(text)
    if not isinstance(obj, dict) or set(obj) != {"done", "contradiction", "evidence"}:
        raise ResultError("reply must contain exactly done, contradiction, evidence")
    done = obj.get("done")
    done = done.strip().lower() if isinstance(done, str) else done
    if done not in ("yes", "no", "unsure"):
        raise ResultError(f'"done" must be "yes", "no" or "unsure", got {obj.get("done")!r}')
    contradiction = obj.get("contradiction")
    if contradiction is not None and not isinstance(contradiction, str):
        raise ResultError('"contradiction" must be a string or null')
    if contradiction is not None:
        contradiction = contradiction.strip()
        contradiction = None if contradiction.lower() in _NULLISH else contradiction
    evidence = obj.get("evidence", "")
    if not isinstance(evidence, str):
        raise ResultError('"evidence" must be a string')
    try:
        return Verdict(done=done, contradiction=contradiction, evidence=evidence.strip()).model_dump(mode="json")
    except ValidationError as e:
        problems = "; ".join(f"{err['loc'][0]}: {err['msg']}" for err in e.errors())
        raise ResultError(problems) from None


def _grid(v: Any, what: str) -> int:
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
        raise ResultError(f"{what} must be a number, got {v!r}")
    return min(GRID_MAX, max(0, int(round(v))))


def _ground(text: str) -> dict[str, Any]:
    obj = json_reply(text)
    items_in = obj.get("items") if isinstance(obj, dict) else obj
    if not isinstance(items_in, list):
        raise ResultError('reply must be {"items": [...]}')
    items: list[dict[str, Any]] = []
    counts: dict[str, int] = {}
    for i, it in enumerate(items_in):
        name = it.get("object_name") if isinstance(it, dict) else None
        if not isinstance(name, str) or not name.strip():
            raise ResultError(f"item {i}: needs object_name")
        name = name.strip()
        if "points" in it:
            pts = it["points"]
            if not isinstance(pts, list) or not all(isinstance(p, dict) for p in pts):
                raise ResultError(f"item {i}: points must be a list of {{x, y}}")
            items += [{"name": name, "x": _grid(p.get("x"), "x"), "y": _grid(p.get("y"), "y")} for p in pts]
            n = it.get("count")
            counts[name] = n if isinstance(n, int) and not isinstance(n, bool) and n >= 0 else len(pts)
        elif "bbox" in it:
            boxes = it["bbox"] if isinstance(it["bbox"], list) else None
            if boxes is None or not all(isinstance(b, dict) for b in boxes):
                raise ResultError(f"item {i}: bbox must be a list of {{x_min, y_min, x_max, y_max}}")
            for b in boxes:
                x0, x1 = sorted((_grid(b.get("x_min"), "x_min"), _grid(b.get("x_max"), "x_max")))
                y0, y1 = sorted((_grid(b.get("y_min"), "y_min"), _grid(b.get("y_max"), "y_max")))
                items.append({"name": name, "bbox": [x0, y0, x1, y1]})
        else:
            items.append({"name": name, "x": _grid(it.get("x"), "x"), "y": _grid(it.get("y"), "y")})
    out: dict[str, Any] = {"items": items}
    if counts:
        out["counts"] = counts
    try:
        for item in items:
            (GroundBox if "bbox" in item else GroundPoint).model_validate(item)
    except ValidationError as e:
        raise ResultError(str(e)) from None
    return out


def _why(text: str) -> dict[str, Any]:
    s = " ".join(text.split())
    if not s:
        raise ResultError("empty reply")
    return {"text": s[: why.MAX_CHARS]}


def _author(text: str) -> dict[str, Any]:
    return {"stepgraph": author.draft_to_stepgraph(json_reply(text))}


# ---------------------------------------------------------------- SAM


def _segment(text: str) -> dict[str, Any]:
    parser = image_segmentation_format().create_parser()
    parser.push(text)
    result = parser.finish(CompletedOutcome()).result
    problems = [f"{d.code}: {d.message}" for d in result.diagnostics if d.severity == "error"]
    problems += [f"unexpected text {r.text[:60]!r}" for r in result.records if r.kind == "text" and r.text.strip()]
    if problems:
        raise ResultError("; ".join(problems)[:500])
    box_conf = {r.object_id: r.confidence for r in result.records if r.kind == "box"}
    latest = {r.object_id: r for r in result.records if r.kind == "mask"}
    masks = []
    for oid in sorted(latest, key=int):
        m = _mask(latest[oid], box_conf.get(oid))
        if m is not None:
            masks.append(m)
    return {"masks": masks, "size": _frame_size(text) if latest else None}


def _mask(rec, box_confidence: float | None) -> dict[str, Any] | None:
    b = rec.bounds
    x0, y0, x1, y1 = int(b.left), int(b.top), int(b.right), int(b.bottom)
    w, h = rec.mask.width, rec.mask.height
    bw, bh = x1 - x0, y1 - y0
    if (bw, bh) != (w, h) and not (max(w, h) == MASK_LONG_SIDE and w <= bw and h <= bh):
        raise ResultError(f"object {rec.object_id}: mask {w}x{h} does not cover its box {bw}x{bh}")
    raster = np.frombuffer(decode_mask_to_raster(rec.mask), dtype=np.uint8).reshape(h, w)
    if (bw, bh) != (w, h):  # a capped mask, scaled back to its box (nearest: still exact 0s and 1s)
        raster = raster[np.ix_(np.arange(bh) * h // bh, np.arange(bw) * w // bw)]
        h, w = bh, bw
    ys, xs = np.nonzero(raster)
    if xs.size == 0:
        return None
    conf = rec.confidence if rec.confidence is not None else box_confidence
    return {
        "object_id": int(rec.object_id),
        "box": [x0, y0, x1, y1],
        "area_px": int(xs.size),
        "centroid": [q_px(x0 + xs.mean()), q_px(y0 + ys.mean())],
        "confidence": None if conf is None else float(conf),
        "rle": {"size": [h, w], "counts": coco_rle_counts(raster)},
    }


def _frame_size(text: str) -> list[int] | None:
    m = _BOX_TOKEN.search(text)
    if m is None:
        return None
    fields = dict(f.split("=", 1) for f in m.group(1).split(";") if "=" in f)
    try:
        return [int(fields["w"]), int(fields["h"])]
    except (KeyError, ValueError):
        return None


def coco_rle_counts(mask: np.ndarray) -> str:
    """COCO compressed RLE string of a binary (h, w) mask, column-major (pycocotools ``rleToString``)."""
    flat = np.asarray(mask, dtype=bool).T.ravel()
    edges = np.flatnonzero(flat[1:] != flat[:-1]) + 1
    runs = np.diff(np.concatenate(([0], edges, [flat.size]))).tolist()
    if flat.size and flat[0]:
        runs.insert(0, 0)
    out: list[str] = []
    for i, x in enumerate(runs):
        if i > 2:
            x -= runs[i - 2]
        more = True
        while more:
            c = x & 0x1F
            x >>= 5
            more = x != -1 if c & 0x10 else x != 0
            if more:
                c |= 0x20
            out.append(chr(c + 48))
    return "".join(out)


_ROLE = {"verdict": _verdict, "ground": _ground, "why": _why, "segment": _segment, "author": _author}
