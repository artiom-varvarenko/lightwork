"""Offline input scenarios for reducer, recorder/verify and replay-as-live tests.

These are generated fixtures, NOT model evidence: ``platform='synthetic'`` and ``resp_synthetic*`` response ids.
Every image comes from ``testing.synth``; every reply is encoded by ``models.synthetic`` and decoded by the real
model parser. No API, environment, clock or filesystem is used here. Never publish these as Meta inference runs.

``build(name)`` gives artefacts, frame bytes, a scripted input timeline and a deterministic responder. ``drive``
merges the timeline with model results scheduled at request time + latency (results win equal-time ties, then call
index). Its ingest callback may return C1 Events (Recorder.ingest) or derived (type, payload) pairs. A supplied
``on_call(record)`` writes the complete call before its result is ingested; it may return an allocated call_dir.
The caller initializes and feeds run_started separately; the timeline begins with the three *_loaded events.

Dilution includes setup, water, dye, ten cell-to-cell transfers, A11->waste and the final read. With OD0=1.2 the
last transfers are below the camera's floor: their required colour checks remain inconclusive, and explicit user
confirms finish them as UNVERIFIED, never passed. The final series checks readable ratios and monotonic low ODs.
"""

from __future__ import annotations

import heapq
from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Any

import numpy as np

from ..contracts.calls import call_dirname
from ..contracts.canonical import canonical, canonical_sha, q_s, sha256_hex
from ..contracts.events import Event, genesis_hash, seal
from ..contracts.layout import Calibration, Layout
from ..contracts.stepgraph import StepGraph
from ..geometry import Regions
from ..models.synthetic import SamObject, chat_sse, responses_sse, sam_sse
from ..models.wire import read_body
from ..testing.synth import Item, encode_jpeg, kit_scene, well_plate
from .reducer import Reducer
from .store import MemoryStore

SPARK = "muse-spark-1.3-contributor"
MODELS = {"verdict": SPARK, "ground": SPARK, "segment": "sam-3.1"}
CREATED = "2026-10-29T12:00:00Z"
SOURCE = {"kind": "text", "ref": "lightwork/core/scenarios.py", "licence": "Apache-2.0",
          "attribution": "Lightwork synthetic tests; not real model evidence"}


@dataclass(frozen=True)
class Reply:
    status: str
    raw: bytes | None
    kind: str = "sse"
    latency_ms: int = 200
    http_status: int | None = None


@dataclass(frozen=True)
class CallRecord:
    index: int
    intent_id: str
    role: str
    model: str
    endpoint: str
    request: dict
    call_dir: str
    status: str
    raw: bytes | None
    kind: str
    latency_ms: int
    usage: dict | None
    t_request: float
    http_status: int | None = None


@dataclass
class Scenario:
    name: str
    pack: str
    stepgraph: StepGraph
    layout: Layout
    calib: Calibration
    run_started: dict
    frames: dict[str, bytes]
    timeline: list[tuple[float, str, str, dict]]
    respond: Callable[[str, str, str, str, dict], Reply]


def calibration() -> Calibration:
    return Calibration.model_validate({
        "H_cam_to_page": [[0.6, 0., 0.], [0., 0.6, 0.], [0., 0., 1.]],
        "page": {"w": 854., "h": 480., "dpr": 1.},
        "camera": {"device_id": "synthetic", "mode": "MJPG 1280x720@30", "controls": {}},
        "residuals": {"rms_px": 0., "p95_px": 0., "max_px": 0., "n_markers": 15, "n_inliers": 60},
        "mm": {"source": "nominal"}, "measurement_mode": "grey", "created": CREATED,
    })


def _shape(kind: str, noun: str, rect, *, mode="noun") -> dict:
    x, y, w, h = rect
    return {"kind": kind, "corners_cam_px": [[x, y], [x+w, y], [x+w, y+h], [x, y+h]],
            "sam": {"mode": mode, "noun": noun}}


def _judge(question: str) -> dict:
    return {"id": "judge", "kind": "judge", "question": question}


def _frame_sha(request: Any) -> str:
    if isinstance(request, dict):
        if "$frame" in request:
            return request["$frame"]
        children = request.values()
    elif isinstance(request, list):
        children = request
    else:
        return ""
    return next((sha for x in children if (sha := _frame_sha(x))), "")


def _text_reply(value: dict | str, model: str, endpoint: str, *, latency=400, terminal="completed") -> Reply:
    text = value if isinstance(value, str) else canonical(value)
    raw = (responses_sse(text, model=model, terminal=terminal) if endpoint == "/v1/responses"
           else chat_sse(text, model=model, finish="length" if terminal == "incomplete" else "stop"))
    return Reply("ok", raw, latency_ms=latency)


class _Script:
    def __init__(self, name: str, graph: dict, layout: dict):
        self.name = name
        self.graph = StepGraph.model_validate(graph)
        self.layout = Layout.model_validate(layout)
        self.calib = calibration()
        self.frames: dict[str, bytes] = {}
        self.truth: dict[str, dict[str, list[SamObject]]] = {}
        self.timeline = [
            (0., "procedure_loaded", "orchestrator", {"stepgraph_sha": canonical_sha(self.graph)}),
            (0., "calibration_loaded", "orchestrator", {"calib_sha": canonical_sha(self.calib)}),
            (0., "layout_loaded", "orchestrator", {"layout_sha": canonical_sha(self.layout)}),
        ]
        self.regions = Regions(self.layout, self.graph)
        self.verdicts: dict[str, dict | str] = {}
        self.delays: dict[str, int] = {}
        self.timeouts: set[str] = set()
        self.incomplete: set[str] = set()

    def event(self, t: float, type_: str, src: str, **payload) -> None:
        self.timeline.append((q_s(t), type_, src, payload))

    def frame(self, t: float, img: np.ndarray, boxes: dict[str, list[tuple]] | None = None) -> str:
        x, y, w, h = self.layout.work_area_xywh
        jpeg = encode_jpeg(img[y:y+h, x:x+w])
        sha = sha256_hex(jpeg)
        self.frames[sha] = jpeg
        self.truth[sha] = {obj: [SamObject(i+1, (a-x, b-y, c-x, d-y)) for i, (a, b, c, d) in enumerate(rects)]
                           for obj, rects in (boxes or {}).items()}
        self.event(t, "keyframe", "camera", frame_sha=sha, crop_xywh=[x, y, w, h], mode="grey", verified=True,
                   t_capture=q_s(t))
        return sha

    def action(self, t: float, refs: list[str], img, boxes=None) -> str:
        self.event(t, "motion", "gate", state="moving", active_regions=refs)
        for i, ref in enumerate(refs):
            self.event(t + .05 * (i+1), "hands", "hands", over_regions=[ref],
                       tip_xy=list(self.regions.centre(ref)))
        self.event(t+.65, "hands", "hands", over_regions=[], tip_xy=None)
        self.event(t+1., "motion", "gate", state="settled", active_regions=refs)
        return self.frame(t+1.2, img, boxes)

    def done(self, t: float) -> None:
        self.event(t, "voice", "voice", command="done", transcript="Done.", session_id="synthetic-session",
                   turn_id=f"turn-{t:g}")

    def respond(self, iid: str, role: str, model: str, endpoint: str, request: dict) -> Reply:
        if iid in self.timeouts:
            return Reply("timeout", None, latency_ms=20_000)
        if role == "verdict":
            verdict = self.verdicts.get(iid, {"done": "yes", "contradiction": None,
                                            "evidence": "Synthetic scripted observation."})
            return _text_reply(verdict, model, endpoint, latency=self.delays.get(iid, 400),
                               terminal="incomplete" if iid in self.incomplete else "completed")
        obj = iid.split("/")[2]
        sha = _frame_sha(request)
        objects = self.truth[sha].get(obj, [])
        w, h = self.layout.work_area_xywh[2:]
        if role == "segment":
            return Reply("ok", sam_sse(objects, size=(w, h)), latency_ms=self.delays.get(iid, 200))
        noun = self.layout.objects[obj].sam.noun
        if "/count/" in iid:
            points = [{"x": round((o.box[0]+o.box[2])/2/w*1000),
                       "y": round((o.box[1]+o.box[3])/2/h*1000)} for o in objects]
            value = {"items": [{"object_name": noun, "points": points, "count": len(points)}]}
        else:
            boxes = [{"x_min": round(o.box[0]/w*1000), "y_min": round(o.box[1]/h*1000),
                      "x_max": round(o.box[2]/w*1000), "y_max": round(o.box[3]/h*1000)} for o in objects]
            value = {"items": [{"object_name": noun, "bbox": boxes}]}
        return _text_reply(value, model, endpoint, latency=self.delays.get(iid, 100))

    def finish(self, end: float, *, models=None, backend="meta") -> Scenario:
        self.event(end, "run_stopped", "orchestrator", reason="synthetic scenario complete")
        self.timeline.sort(key=lambda e: e[0])
        started = {"run_id": f"20261029T120000Z_{self.graph.id}", "pack": self.graph.id, "backend": backend,
                   "models": models or dict(MODELS), "code_commit": "0000000", "dirty": False,
                   "platform": "synthetic", "version": "0.1.0"}
        return Scenario(self.name, self.graph.id, self.graph, self.layout, self.calib, started, self.frames,
                        self.timeline, self.respond)


def kit(*, missing_remote=False, contradiction=False) -> Scenario:
    name = "kit-remote-missing" if missing_remote else "contradiction-next-active" if contradiction else "kit-clean"
    slots = {"slot-p": (730, 150, 230, 180), "slot-c": (730, 370, 230, 180), "slot-r": (1000, 150, 180, 390)}
    objects = [
        {"id": "projector", "noun": "projector", "kind": "item"},
        {"id": "cable", "noun": "power cable", "kind": "item"},
        {"id": "remote", "noun": "remote control", "kind": "item"},
        {"id": "battery", "noun": "battery", "kind": "item"},
        {"id": "case", "noun": "case", "kind": "container"},
        *[{"id": s, "noun": "slot", "kind": "zone"} for s in slots],
        {"id": "bench", "noun": "bench", "kind": "zone"},
    ]
    steps = [
        {"id": "projector-in", "say": "Put the projector in its slot.", "detail": "Keep its lens pointing up.",
         "targets": ["slot-p"], "checks": [
             {"id": "present", "kind": "present", "object": "projector", "region": "slot-p"},
             _judge("Is the projector seated with the lens pointing up?")],
         "recovery": [{"on": "*", "action": "redo_step", "say": "Correct the projector position."}]},
        {"id": "cable-in", "say": "Put the power cable in the lower slot.", "targets": ["slot-c"], "checks": [
             {"id": "moved", "kind": "moved_to", "object": "cable", "region": "slot-c"}]},
        {"id": "remote-in", "say": "Put the remote and two batteries in the right slot.", "targets": ["slot-r"],
         "arrow": {"from": "remote", "to": "slot-r"}, "checks": [
             {"id": "moved", "kind": "moved_to", "object": "remote", "region": "slot-r"},
             {"id": "two", "kind": "count", "object": "battery", "region": "slot-r", "n": 2},
             {"id": "activity", "kind": "activity_at", "region": "slot-r"},
             _judge("Are the remote and batteries in the right-hand slot?")],
         "recovery": [{"on": "*", "action": "redo_step", "say": "Put the missing remote in its slot."}]},
        {"id": "close", "say": "Check the bench, then close the case and say done.", "targets": ["case"],
         "checks": [{"id": "clear", "kind": "absent", "object": "remote", "region": "bench"},
                    _judge("Are all kit items packed?"), {"id": "confirm", "kind": "confirm"}]},
    ]
    graph = {"id": "synthetic-kit", "title": "Synthetic kit packing", "source": SOURCE, "objects": objects,
             "steps": steps, "vocabulary": ["projector", "cable", "remote", "battery"]}
    layout_objs = {s: _shape("zone", "slot", rect) for s, rect in slots.items()}
    layout_objs.update({"projector": _shape("item", "projector", (180, 170, 160, 100)),
                        "cable": _shape("item", "power cable", (350, 360, 120, 100)),
                        "remote": _shape("item", "remote control", (200, 420, 50, 120), mode="grounded"),
                        "battery": _shape("item", "battery", (460, 180, 70, 50)),
                        "case": _shape("container", "case", (700, 120, 520, 480)),
                        "bench": _shape("zone", "bench", (100, 100, 540, 500))})
    script = _Script(name, graph, {"pack": graph["id"], "created": CREATED, "work_area_xywh": [80, 60, 1160, 600],
                                  "objects": layout_objs})
    packed: set[str] = set()

    def frame():
        specs = [
            ("projector", (845, 240) if "projector" in packed else (260, 220), (160, 100), (80, 80, 80)),
            ("cable", (845, 460) if "cable" in packed else (410, 410), (120, 100), (40, 40, 150)),
            ("remote", (1090, 230) if "remote" in packed else (225, 480), (50, 120), (40, 40, 40)),
            ("battery", (1050, 420) if "battery" in packed else (475, 205), (30, 50), (40, 170, 40)),
            ("battery", (1120, 420) if "battery" in packed else (525, 205), (30, 50), (40, 170, 40)),
        ]
        items, boxes = [], {}
        for obj, xy, size, colour in specs:
            items.append(Item(obj, "rect", colour, xy, size))
            x, y = xy
            w, h = size
            boxes.setdefault(obj, []).append((x-w//2, y-h//2, x+w//2+1, y+h//2+1))
        return kit_scene(items, slots=slots), boxes

    script.frame(.1, *frame())
    packed.add("projector")
    script.action(2., ["slot-p"], *frame())
    shift = 0.
    if contradiction:
        script.verdicts["projector-in/judge/1"] = {"done": "yes", "contradiction": "the projector is upside down",
                                                   "evidence": "Synthetic contradiction after provisional."}
        script.delays["projector-in/judge/1"] = 2000
        script.action(6., ["slot-p"], *frame())
        shift = 4.
    packed.add("cable")
    script.action(6.+shift, ["slot-c"], *frame())
    packed.add("battery")
    if not missing_remote:
        packed.add("remote")
    script.action(10.+shift, ["slot-r"], *frame())
    if missing_remote:
        packed.add("remote")
        script.action(14.+shift, ["slot-r"], *frame())
        shift += 4.
    script.action(14.+shift, ["case"], *frame())
    script.done(17.+shift)
    return script.finish(19.+shift)


def dilution(*, planted=False) -> Scenario:
    name = "dilution-planted-a7" if planted else "dilution-clean"
    objects = [{"id": "plate", "noun": "well plate", "kind": "grid",
                "grid": {"rows": 8, "cols": 12, "naming": "A1", "channel": "r"}},
               {"id": "waste", "noun": "waste cup", "kind": "container"}]
    activity = lambda c: {"id": f"at-{c.lower()}", "kind": "activity_at", "region": f"plate:{c}"}
    colour = lambda cid, cells, direction, **kw: {"id": cid, "kind": "colour_change", "object": "plate",
                                                  "cells": cells, "direction": direction, **kw}
    steps = [
        {"id": "setup", "say": "Check that A1 is at the top-left.", "targets": ["plate"], "checks": [
            {"id": "plate", "kind": "present", "object": "plate", "region": "plate"},
            _judge("Is the plate landscape with A1 at the top-left?"), {"id": "confirm", "kind": "confirm"}]},
        {"id": "water", "say": "Add water to A2 through A12.", "targets": [f"plate:A{k}" for k in range(2, 13)],
         "checks": [*[activity(f"A{k}") for k in range(2, 13)], _judge("Are A2 through A12 prepared with diluent?")]},
        {"id": "dye", "say": "Add dye to A1 only.", "targets": ["plate:A1"], "checks": [
            colour("darker", ["A1"], "darker", min_delta=.5),
            colour("elsewhere", [f"A{k}" for k in range(2, 13)], "none"), _judge("Is dye in A1 only?")]},
    ]
    for k in range(1, 11):
        source, target = f"A{k}", f"A{k+1}"
        step = {"id": f"transfer-{k}", "say": f"Mix {source}, then transfer half into {target}.",
                "targets": [f"plate:{target}"], "arrow": {"from": f"plate:{source}", "to": f"plate:{target}"},
                "checks": [activity(source), activity(target), colour("lighter", [source], "lighter"),
                           colour("darker", [target], "darker", ratio=1., tol=.25),
                           _judge(f"Was liquid transferred from {source} into {target}?")]}
        if k == 5:
            step["recovery"] = [
                {"on": "wrong_cell:A7", "action": "exclude_cells", "cells": ["plate:A6"],
                 "say": "A6 is now an extra blank."},
                {"on": "wrong_cell:A7", "action": "skip_steps", "steps": ["transfer-6"],
                 "say": "Continue from A7."},
            ]
        steps.append(step)
    steps += [
        {"id": "waste", "say": "Mix A11 and transfer half to waste.", "targets": ["waste"],
         "arrow": {"from": "plate:A11", "to": "waste"}, "checks": [activity("A11"),
             {"id": "waste", "kind": "activity_at", "region": "waste"}, colour("lighter", ["A11"], "lighter"),
             colour("blank", ["A12"], "none"), _judge("Did liquid go to waste, leaving A12 blank?")]},
        {"id": "read", "say": "Check the final dilution series and say done.", "targets": ["plate"], "checks": [
             {"id": "series", "kind": "series", "object": "plate", "cells": [f"A{k}" for k in range(1, 12)],
              "ratio": .5, "tol": .15}, colour("blank", ["A12"], "none"),
             _judge("Is the dilution series monotonic with the designated blank cells clear?"),
             {"id": "confirm", "kind": "confirm"}]},
    ]
    graph = {"id": "synthetic-dilution", "title": "Synthetic row A dilution", "source": SOURCE,
             "objects": objects, "steps": steps}
    ods: dict[str, float] = {}
    img, geom = well_plate(ods)
    script = _Script(name, graph, {"pack": graph["id"], "created": CREATED, "work_area_xywh": [160, 80, 960, 580],
        "objects": {"plate": {"kind": "grid", "corners_cam_px": geom.corners.tolist(),
                               "sam": {"mode": "noun", "noun": "well plate"}},
                    "waste": _shape("container", "waste cup", (1000, 220, 90, 100))}})
    truth = {"plate": [(200, 110, 920, 590)]}
    script.frame(.1, img, truth)
    script.action(2., ["plate"], img, truth)
    script.done(4.)
    script.action(6., [f"plate:A{k}" for k in range(2, 13)], img, truth)
    ods["A1"] = 1.2
    script.action(10., ["plate:A1"], well_plate(ods)[0], truth)
    t = 14.
    for k in range(1, 11):
        if planted and k == 6:
            continue
        target = 7 if planted and k == 5 else k+1
        source_before = ods[f"A{k}"]
        ods[f"A{k}"] /= 2
        ods[f"A{target}"] = ods[f"A{k}"]
        script.action(t, [f"plate:A{k}", f"plate:A{target}"], well_plate(ods)[0], truth)
        if source_before < .06:
            script.done(t+2.)
        t += 4.
    ods["A11"] /= 2
    script.action(t, ["plate:A11", "waste"], well_plate(ods)[0], truth)
    script.done(t+2.)  # the last half-OD is below the camera's required colour-change threshold
    script.done(t+3.)  # final read has now received its Meta verdict
    return script.finish(t+5.)


def timeout() -> Scenario:
    graph = {"id": "synthetic-timeout", "title": "Synthetic verdict timeout", "source": SOURCE,
             "objects": [{"id": "slot", "kind": "zone", "noun": "slot"}], "steps": [
                 {"id": "place", "say": "Put the item in the slot.", "targets": ["slot"], "checks": [
                     {"id": "activity", "kind": "activity_at", "region": "slot"}, _judge("Is the item in place?")]}]}
    script = _Script("verdict-timeout", graph, {"pack": graph["id"], "created": CREATED,
        "work_area_xywh": [80, 60, 1160, 600], "objects": {"slot": _shape("zone", "slot", (700, 150, 200, 200))}})
    img = kit_scene([])
    script.frame(.1, img)
    script.action(2., ["slot"], img)
    script.timeouts.add("place/judge/1")
    script.event(23.3, "tick", "orchestrator")
    script.done(24.)
    return script.finish(25.)


SCENARIOS: dict[str, Callable[[], Scenario]] = {
    "kit-clean": kit,
    "kit-remote-missing": lambda: kit(missing_remote=True),
    "dilution-clean": dilution,
    "dilution-planted-a7": lambda: dilution(planted=True),
    "verdict-timeout": timeout,
    "contradiction-next-active": lambda: kit(contradiction=True),
}


def build(name: str) -> Scenario:
    return SCENARIOS[name]()


def drive(scenario: Scenario, ingest, request_for, on_call=None) -> list[CallRecord]:
    """Run the scripted input timeline; persist each response via on_call BEFORE ingesting its model_result."""
    pending: list[tuple[float, int, CallRecord]] = []
    calls: list[CallRecord] = []
    index = 0

    def send(t, type_, src, payload):
        nonlocal index
        for item in ingest(type_, src, payload, t):
            et, p = (item.type, item.payload) if isinstance(item, Event) else item
            if et != "model_request":
                continue
            endpoint, request = request_for(p["intent_id"])
            reply = scenario.respond(p["intent_id"], p["role"], p["model"], endpoint, request)
            index += 1
            usage = read_body(reply.raw, reply.kind, endpoint).usage if reply.raw is not None else None
            rec = CallRecord(index, p["intent_id"], p["role"], p["model"], endpoint, request,
                             "calls/" + call_dirname(index, p["role"], p["model"]), reply.status, reply.raw,
                             reply.kind, reply.latency_ms, usage, t,
                             reply.http_status if reply.http_status is not None else (200 if reply.status == "ok" else None))
            heapq.heappush(pending, (q_s(t + reply.latency_ms/1000), index, rec))

    def flush(until):
        while pending and pending[0][0] <= until:
            t, _, rec = heapq.heappop(pending)
            allocated = on_call(rec) if on_call is not None else None
            if allocated is not None:
                rec = replace(rec, call_dir=allocated)
            calls.append(rec)
            send(t, "model_result", "model", {"intent_id": rec.intent_id, "call_dir": rec.call_dir,
                 "status": rec.status, "latency_ms": rec.latency_ms, "usage": rec.usage, "http_status": rec.http_status})

    for t, type_, src, payload in scenario.timeline:
        flush(t)
        send(t, type_, src, payload)
    flush(float("inf"))
    return sorted(calls, key=lambda c: c.index)


def run_memory(scenario: Scenario, *, params=None, model_fns=None, on_scene=None) -> tuple[list[Event], list[CallRecord]]:
    """Record a scenario in RAM with the same hash chain and sealing order as Recorder.ingest (no network)."""
    store = MemoryStore(scenario.frames)
    reducer = Reducer(scenario.stepgraph, scenario.layout, scenario.calib, store,
                      backend=scenario.run_started["backend"], models=scenario.run_started["models"],
                      params=params, model_fns=model_fns)
    events: list[Event] = []
    prev = genesis_hash(scenario.run_started["run_id"])

    def append(type_, src, payload, t):
        nonlocal prev
        ev = seal(prev, len(events)+1, q_s(t), type_, src, payload)
        events.append(ev)
        prev = ev.h
        return ev

    def ingest(type_, src, payload, t):
        inp = append(type_, src, payload, t)
        out = reducer.feed(inp)
        for et, p in out:
            append(et, "reducer", p, t)
        if on_scene is not None and any(et == "scene" for et, _ in out):
            on_scene(reducer.scene)
        return out

    def on_call(rec):
        if rec.raw is not None:
            store.put_call(rec.call_dir, rec.raw, rec.kind)

    ingest("run_started", "orchestrator", scenario.run_started, 0.)
    calls = drive(scenario, ingest, reducer.request_for, on_call)
    return events, calls
