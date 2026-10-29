"""Core reducer (workstream H): a pure function of the input event sequence (PLAN.md D1, D20, §4.2-§4.3, §6).

``feed(input event)`` returns the derived ``(type, payload)`` list for that input, in this order: causal events
(``measurement``, ``parsed``, ``cost``, ``check``, ``step``, ``model_request``) as they arise, then ``scene`` if the
scene changed, then ``say`` lines. Time comes only from event ``t``; there is no randomness; the only I/O is
``store.frame_bytes`` (verified keyframes) and ``store.call_response`` (model results with status ``ok``).

Rules (the parts PLAN.md leaves to H):

- **Start.** The first step activates once ``procedure_loaded``, ``calibration_loaded`` and ``layout_loaded`` have
  arrived with the SHA-256 of the artefacts this reducer was built with (else :class:`ReducerError`).
- **Gate.** A keyframe is *current* while no motion started after it. On a current verified keyframe the active
  step's local checks run against its pre-step reference (the latest verified keyframe when the attempt began). The
  step turns provisional when every required local check passes and there is evidence of action: a passed local
  check or motion at a target (at a check region without targets; any motion without either). At most one step is
  provisional; a ready active step freezes its evidence frame until the slot frees. Definitive failures need a
  keyframe newer than the attempt's start, so a redo never fails on the frame that failed it. Unreadable required
  checks never silently pass: a user override ends the step ``unverified``.
- **Meta readings** are requested when a step turns provisional: ``judge`` -> Spark verdict on the provisional
  keyframe; SAM checks -> ``segment`` of the object (noun prompt, or Spark grounding boxes -> box prompts when the
  layout says ``grounded``); ``moved_to`` also segments the pre-step keyframe; ``count`` adds a Spark count. Intent
  ids: ``<step>/<check>/<n>``, ``<step>/<segment|segment-before|ground|ground-before|count>/<object>/<n>``, and one
  retry ``<intent>/retry`` when a parse is ``invalid`` or ``incomplete``.
  A final contributor HTTP 429 derives ``<intent>/fallback`` on standard Spark, only before the step deadline.
- **Resolution.** Failed when a required check fails. Passed when every required check passes and at least one
  Meta check passed. Otherwise the step stays provisional and needs a confirm. The verification deadline
  (``Params.verify_timeout_*``, event time) turns missing readings inconclusive; ``timeout`` recovery entries then
  apply when the step has them.
- **Confirm** (voice ``done``, key ``space``, ui ``done``): resolves a step that needs confirm (``passed`` only on
  the Meta readings, else ``unverified``); on the active step: confirm-only -> ``unverified``; otherwise it passes
  ``confirm`` checks and triggers the gate, and if the local checks still do not pass on a current keyframe the step
  ends ``unverified`` (user override).
- **Failure.** The reason is spoken (interrupt) and marked in the scene until the next provisional or passed step.
  An active step started during the failed step's verification is suspended (``waiting``). Recovery entries whose
  ``on`` is a failed check id, ``wrong_cell:<cell>`` or ``*`` apply in list order; with none, the step is redone.
  ``insert_step`` places the inline step before the failed step, so a redo runs after it.
- **Commands.** ``next`` skips a confirm-only active step; ``repeat``; ``why`` (last failure reason since the last
  pass, else the step detail); ``back`` re-activates the previous step unless it passed.
- **Cost.** ``models.pricing.cost_usd(model, model_result.usage)`` over aggregate billed usage inputs, falling back
  to parsed-final usage when the input is null. Retry-attempt costs therefore re-derive without reading spend state.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from ..contracts.calls import request_fingerprint
from ..contracts.canonical import canonical_sha, q_s
from ..contracts.events import INPUT_TYPES, Event, make_payload
from ..contracts.layout import Calibration, Layout, SamPrompt
from ..contracts.render import SceneMsg
from ..contracts.stepgraph import LOCAL_CHECKS, META_CHECKS, SAM_CHECKS, Check, Recovery, Step, StepGraph
from ..geometry import Regions
from . import checks as ck
from . import params as params_mod
from .graph import confirm_only, region_hit, validate
from .masks import to_camera
from .modelio import ModelFns, default_model_fns
from .readings import Readings
from .scene import SceneBuilder, scene_hash
from .store import RunStore

TERMINAL = frozenset({"passed", "failed", "unverified", "skipped", "excluded"})
COMMANDS = frozenset({"next", "done", "repeat", "why", "back"})
KEY_COMMANDS = {"space": "done", " ": "done", "r": "repeat", "b": "back"}
MAX_PROMPT_OBJECTS = 16


class ReducerError(ValueError):
    """The input stream does not fit this reducer (wrong artefact, unknown or repeated intent, derived input)."""


@dataclass(frozen=True)
class Failure:
    """A Meta reading that will not arrive or cannot be used."""

    reason: str


@dataclass(frozen=True)
class Keyframe:
    seq: int
    sha: str
    crop: tuple[int, int, int, int]


@dataclass
class Intent:
    id: str
    role: str
    model: str
    endpoint: str
    request: dict[str, Any]
    ctx: dict[str, Any]
    frame: str | None
    crop: tuple[int, int, int, int] | None
    step: str
    attempt: int
    key: str  # reading key it serves: verdict:<check> | seg:<object>:<before|after> | ground:... | count:<object>
    retry: bool = False
    answered: bool = False


@dataclass
class Attempt:
    n: int
    start_kf: int  # seq of the latest verified keyframe when the attempt began (0: none)
    ref: str | None  # pre-step reference keyframe
    motion: set[str] = field(default_factory=set)
    motion_seqs: list[int] = field(default_factory=list)
    tips: set[str] = field(default_factory=set)
    hands_seqs: list[int] = field(default_factory=list)
    confirm: str | None = None  # evidence ref of a confirm while active
    outcomes: dict[str, ck.Outcome] = field(default_factory=dict)
    shown: dict[str, tuple[str, str]] = field(default_factory=dict)  # check id -> last status/reason
    frame: Keyframe | None = None  # provisional keyframe
    ready: bool = False
    deadline: float | None = None
    timed_out: bool = False
    deadline_expired: bool = False
    needs_confirm: bool = False
    requested: dict[str, str] = field(default_factory=dict)  # reading key -> intent id
    readings: dict[str, Any] = field(default_factory=dict)  # reading key -> result | Failure
    by: dict[str, str] = field(default_factory=dict)  # reading key -> evidence intent


@dataclass
class StepRun:
    step: Step
    state: str = "waiting"
    attempts: int = 0
    att: Attempt | None = None


def _sentence(text: str) -> str:
    text = text.strip().rstrip(".")
    return (text[:1].upper() + text[1:] + ".") if text else ""


class Reducer:
    def __init__(self, stepgraph: StepGraph, layout: Layout, calib: Calibration, store: RunStore, *, backend: str,
                 models: dict[str, str], params: params_mod.Params | None = None,
                 model_fns: ModelFns | None = None):
        errs = validate(stepgraph, layout)
        if errs:
            raise ReducerError("step graph/layout rejected: " + "; ".join(errs))
        self.stepgraph = stepgraph
        self.layout = layout
        self.calib = calib
        self.store = store
        self.backend = backend
        self.models = dict(models)
        self.p = params if params is not None else params_mod.DEFAULT
        self.fns = model_fns if model_fns is not None else default_model_fns()
        self.regions = Regions(layout, stepgraph)
        self.objects = {o.id: o for o in stepgraph.objects}
        self.readings = Readings(stepgraph, self.regions, self.p)
        self.scenes = SceneBuilder(self.regions, layout, self.p)
        self._artefacts = {"procedure_loaded": canonical_sha(stepgraph), "calibration_loaded": canonical_sha(calib),
                           "layout_loaded": canonical_sha(layout)}
        self._loaded: set[str] = set()
        self._started = False
        self._aborted = False
        self._complete = False
        self._steps = {s.id: StepRun(s) for s in stepgraph.steps}
        self._order = [s.id for s in stepgraph.steps]
        self._active: StepRun | None = None
        self._prov: StepRun | None = None
        self._kf: Keyframe | None = None
        self._frames_by_sha: dict[str, Keyframe] = {}
        self._kf_current = False
        self._moving = False
        self._hands_tracked = False
        self._excluded: set[str] = set()
        self._marks: tuple[ck.Mark, ...] = ()
        self._last_reason = ""
        self._intents: dict[str, Intent] = {}
        self._cost = 0.0
        self._timeout = self.p.verify_timeout_meta_s if backend == "meta" else self.p.verify_timeout_chat_s
        self._scene = SceneMsg(seq=0, mode="normal", prims=[])
        self._scene_hash = scene_hash(self._scene)
        self._t = 0.0
        self._out: list[tuple[str, dict]] = []
        self._says: list[tuple[str, dict]] = []

    # ------------------------------------------------------------ public API

    def feed(self, ev: Event) -> list[tuple[str, dict]]:
        if ev.derived or ev.type not in INPUT_TYPES:
            raise ReducerError(f"seq {ev.seq}: {ev.type!r} is not an input event")
        payload = ev.typed_payload()
        self._t = ev.t
        self._out, self._says = [], []
        self._expire()
        getattr(self, f"_on_{ev.type}")(ev, payload)
        self._expire()
        self._refresh_scene()
        out = self._out + self._says
        self._out, self._says = [], []
        return out

    def request_for(self, intent_id: str) -> tuple[str, dict]:
        try:
            it = self._intents[intent_id]
        except KeyError:
            raise ReducerError(f"unknown intent {intent_id!r}") from None
        return it.endpoint, it.request

    @property
    def scene(self) -> SceneMsg:
        return self._scene

    # ------------------------------------------------------------ emitters

    def _emit(self, type_: str, **payload: Any) -> None:
        self._out.append((type_, make_payload(type_, payload)))

    def _say(self, text: str, priority: str = "normal") -> None:
        if text:
            self._says.append(("say", make_payload("say", {"text": text, "priority": priority})))

    def _set_state(self, s: StepRun, state: str, reason: str = "") -> None:
        s.state = state
        self._emit("step", step=s.step.id, state=state, reason=reason)

    def _show(self, s: StepRun, c: Check, o: ck.Outcome) -> None:
        """Emit first observations and changes of status/reason (including suspect motion still pending)."""
        att = s.att
        att.outcomes[c.id] = o
        signature = (o.status, o.reason)
        if att.shown.get(c.id) != signature:
            att.shown[c.id] = signature
            self._emit("check", step=s.step.id, check=c.id, status=o.status, reason=o.reason,
                       evidence=list(o.evidence))

    def _refresh_scene(self) -> None:
        label = None
        if self._active is not None:
            label = f"{self._order.index(self._active.step.id) + 1}/{len(self._order)}"
        warnings = []
        unverified = sum(s.state == "unverified" for s in self._steps.values())
        if unverified:
            warnings.append(f"UNVERIFIED: {unverified}")
        if not self._hands_tracked and self._active is not None and any(
                c.kind == "activity_at" for c in self._active.step.checks):
            warnings.append("MOTION ONLY: no hand tracking")
        prims = self.scenes.build(active=self._active.step if self._active else None,
                                  provisional=self._prov.step if self._prov else None, marks=self._marks, label=label,
                                  status=" | ".join(warnings))
        candidate = SceneMsg(seq=self._scene.seq + 1, mode="normal", prims=prims)
        h = scene_hash(candidate)
        if h != self._scene_hash:
            self._scene, self._scene_hash = candidate, h
            self._emit("scene", hash=h)

    # ------------------------------------------------------------ inputs

    def _on_run_started(self, ev: Event, p) -> None:
        if p.backend != self.backend or dict(p.models) != self.models:
            raise ReducerError(f"seq {ev.seq}: run_started backend/models differ from the reducer's")

    def _loaded_artefact(self, ev: Event, sha: str) -> None:
        want = self._artefacts[ev.type]
        if sha != want:
            raise ReducerError(f"seq {ev.seq}: {ev.type} sha {sha[:12]} does not match {want[:12]}")
        self._loaded.add(ev.type)
        if not self._started and self._loaded == set(self._artefacts):
            self._started = True
            self._advance()

    def _on_procedure_loaded(self, ev: Event, p) -> None:
        self._loaded_artefact(ev, p.stepgraph_sha)

    def _on_calibration_loaded(self, ev: Event, p) -> None:
        self._loaded_artefact(ev, p.calib_sha)

    def _on_layout_loaded(self, ev: Event, p) -> None:
        self._loaded_artefact(ev, p.layout_sha)

    def _on_keyframe(self, ev: Event, p) -> None:
        if not p.verified:
            return
        for m in self.readings.measure(p.frame_sha, self.store.frame_bytes(p.frame_sha), p.crop_xywh):
            self._emit("measurement", **m)
        self._kf = Keyframe(ev.seq, p.frame_sha, tuple(p.crop_xywh))
        self._frames_by_sha[p.frame_sha] = self._kf
        self._kf_current = not self._moving
        a = self._active
        if a is not None and self._kf_current:
            if a.att.ref is None:
                a.att.ref = p.frame_sha
                a.att.start_kf = ev.seq
            self._gate(a)

    def _on_motion(self, ev: Event, p) -> None:
        self._moving = p.state == "moving"
        if self._moving:
            self._kf_current = False
        a = self._active
        if a is not None and p.active_regions:
            a.att.motion.update(p.active_regions)
            a.att.motion_seqs.append(ev.seq)

    def _on_hands(self, ev: Event, p) -> None:
        self._hands_tracked = True
        a = self._active
        if a is None:
            return
        hits = set()
        if p.tip_xy is not None:
            hits.update(ref for ref in self.regions.all_refs() if self.regions.contains(ref, p.tip_xy))
        if hits:
            a.att.tips.update(hits)
            a.att.hands_seqs.append(ev.seq)

    def _on_voice(self, ev: Event, p) -> None:
        self._command(p.command, f"voice:{ev.seq}")

    def _on_key(self, ev: Event, p) -> None:
        if p.key in KEY_COMMANDS:
            self._command(KEY_COMMANDS[p.key], f"seq:{ev.seq}")

    def _on_ui(self, ev: Event, p) -> None:
        if p.action in COMMANDS:
            self._command(p.action, f"seq:{ev.seq}")

    def _on_voice_rejected(self, ev: Event, p) -> None:
        pass

    def _on_tick(self, ev: Event, p) -> None:
        pass  # time advanced; deadlines are checked after every input

    def _on_run_stopped(self, ev: Event, p) -> None:
        pass

    def _on_model_result(self, ev: Event, p) -> None:
        it = self._intents.get(p.intent_id)
        if it is None:
            raise ReducerError(f"seq {ev.seq}: model_result for unknown intent {p.intent_id!r}")
        if it.answered:
            raise ReducerError(f"seq {ev.seq}: second model_result for {p.intent_id!r}")
        it.answered = True
        result: Any
        if p.status == "ok":
            raw, kind = self.store.call_response(p.call_dir)
            res = self.fns.parse_response(it.role, raw, kind, it.endpoint)
            ok = res["status"] == "ok"
            self._emit("parsed", intent_id=it.id, kind=it.role if ok else "error",
                       result=res["result"] if ok else {"status": res["status"], "error": res["error"],
                                                        "detail": res["result"]})
            self._bill(it.model, p.usage if p.usage is not None else res.get("usage"))
            if ok:
                result = res["result"]
            elif res["status"] in ("invalid", "incomplete") and not it.retry and it.role in ("verdict", "ground"):
                if self._live(it) and self._t < self._steps[it.step].att.deadline:
                    self._retry(it, res)
                return
            else:
                result = Failure(f"{it.role} reply {res['status']}: {res['error'] or ''}".rstrip(": "))
        else:
            self._bill(it.model, p.usage)
            if (p.http_status == 429 and it.model == "muse-spark-1.3-contributor" and self._live(it)
                    and self._t < self._steps[it.step].att.deadline):
                s = self._steps[it.step]
                frame = Keyframe(0, it.frame, it.crop) if it.frame is not None else None
                self._intent(s, f"{it.id}/fallback", it.role, "muse-spark-1.3", frame, it.ctx, it.key, retry=it.retry)
                return
            result = Failure(f"{it.role} call {p.status}")
        if self._live(it):
            if p.status == "timeout":
                self._steps[it.step].att.timed_out = True
            self._deliver(self._steps[it.step], it, result)

    def _bill(self, model: str, usage: dict | None) -> None:
        cost = self.fns.cost_usd(model, usage)
        if cost:
            self._cost = round(self._cost + cost, 6)
            self._emit("cost", usd_total=self._cost)

    # ------------------------------------------------------------ commands

    def _command(self, cmd: str, ev_ref: str) -> None:
        if not self._started or self._aborted:
            return
        a = self._active
        if cmd == "done":
            self._confirm(ev_ref)
        elif cmd == "repeat":
            self._say(a.step.say if a else "All steps done.", "interrupt")
        elif cmd == "why":
            text = self._last_reason or (a.step.detail or a.step.say if a else "All steps done.")
            self._say(_sentence(text), "interrupt")
        elif cmd == "next":
            if a is not None and confirm_only(a.step):
                self._end(a, "skipped", "skipped by user")
            elif a is not None:
                self._say("This step is checked by the camera; say done if it is finished.", "interrupt")
        elif cmd == "back":
            self._back()

    def _confirm(self, ev_ref: str) -> None:
        pv = self._prov
        if pv is not None and pv.att.needs_confirm:
            self._confirm_provisional(pv, ev_ref)
            return
        a = self._active
        if a is None:
            return
        if confirm_only(a.step):
            for c in a.step.checks:
                self._show(a, c, ck.Outcome("pass", "confirmed by user", (ev_ref,)))
            self._end(a, "unverified", "confirmed by user")
            return
        for c in a.step.checks:
            if c.kind == "confirm":
                self._show(a, c, ck.Outcome("pass", "confirmed by user", (ev_ref,)))
        a.att.confirm = ev_ref
        if self._kf is not None and self._kf_current:
            self._gate(a)

    def _confirm_provisional(self, s: StepRun, ev_ref: str) -> None:
        for c in s.step.checks:
            if c.kind == "confirm":
                self._show(s, c, ck.Outcome("pass", "confirmed by user", (ev_ref,)))
        if self._verified(s):
            self._pass(s)
        else:
            self._end(s, "unverified", "confirmed by user; " + self._unverified_reason(s))

    def _back(self) -> None:
        a = self._active
        idx = self._order.index(a.step.id) if a is not None else len(self._order)
        if idx == 0:
            return
        prev = self._steps[self._order[idx - 1]]
        if prev.state == "passed":
            self._say("The previous step already passed.", "interrupt")
            return
        if a is not None:
            self._set_state(a, "waiting", "back")
            self._active = None
        if self._prov is prev:
            self._prov = None
        self._activate(prev, "back")

    # ------------------------------------------------------------ lifecycle

    def _advance(self) -> None:
        if not self._started or self._aborted:
            return
        if self._active is None:
            nxt = next((self._steps[i] for i in self._order if self._steps[i].state in ("waiting", "recovery")), None)
            if nxt is not None:
                self._activate(nxt, "")
            elif self._prov is None and not self._complete:
                self._complete = True
                n = sum(s.state == "unverified" for s in self._steps.values())
                self._say(f"All steps done. {n} unverified." if n else "All steps done.")
        elif self._prov is None:
            if self._active.att.ready:
                self._provisional(self._active, "local gate passed; verification slot available")
            elif self._kf is not None and self._kf_current:
                self._gate(self._active)

    def _activate(self, s: StepRun, reason: str) -> None:
        s.attempts += 1
        kf = self._kf
        s.att = Attempt(n=s.attempts, start_kf=kf.seq if kf else 0, ref=kf.sha if kf else None)
        self._active = s
        self._complete = False
        self._set_state(s, "active", reason)
        self._say(s.step.say)
        if confirm_only(s.step):
            return
        if kf is not None and self._kf_current:
            self._gate(s)

    def _gate(self, s: StepRun) -> None:
        """Evaluate the active step's local checks on the current keyframe and move it on if they allow."""
        if confirm_only(s.step):
            return
        att, kf = s.att, self._kf
        if att.ready:
            if self._prov is None:
                self._provisional(s, "local gate passed")
            return
        if att.ref is None:
            att.ref = kf.sha
            att.start_kf = kf.seq
        now, before = self.readings.at(kf.sha), self.readings.at(att.ref)
        fresh = kf.seq > att.start_kf
        frames = (f"frame:{kf.sha}", f"frame:{att.ref}")
        local = [c for c in s.step.checks if c.kind in LOCAL_CHECKS]
        out: dict[str, ck.Outcome] = {}
        for c in local:
            out[c.id] = self._eval_local(s, c, now, before, frames)
        required = [c for c in local if c.required]
        for c in required:
            o = out[c.id]
            if c.kind == "colour_change" and c.direction == "darker" and o.status == "pending" and fresh:
                need, _ = ck.colour_need(c, s.step, before, self.p)
                w = ck.wrong_cell(c, s.step, self.objects[c.object].grid, now, before, need, self.p)
                if w is not None and region_hit(f"{c.object}:{w.wrong}", att.motion):
                    out[c.id] = ck.Outcome(w.status, w.reason, frames, w.wrong, w.marks)
        for c in local:
            o = out[c.id]
            others_ok = all(out[k.id].status == "pass" for k in required if k.id != c.id)
            if o.status == "fail" and (not fresh or (c.kind == "series" and not others_ok)):
                out[c.id] = ck.Outcome("pending", o.reason, o.evidence)
            self._show(s, c, out[c.id])
        failed = [c for c in required if out[c.id].status == "fail"]
        if failed:
            keys = [c.id for c in failed] + [f"wrong_cell:{out[c.id].wrong}" for c in failed if out[c.id].wrong]
            marks = tuple(m for c in failed for m in out[c.id].marks)
            self._fail(s, out[failed[0].id].reason, keys, marks)
            return
        if all(out[c.id].status == "pass" for c in required):
            acted = any(out[c.id].status == "pass" for c in required) or self._activity_seen(s)
            if acted:
                att.frame = kf
                att.ready = True
                if self._prov is None:
                    self._provisional(s, "local gate passed")
                return
        if att.confirm is not None:
            pending = next((out[c.id].reason for c in required if out[c.id].status != "pass"), "no action seen")
            self._end(s, "unverified", f"confirmed by user; camera: {pending}")

    def _eval_local(self, s: StepRun, c: Check, now, before, frames) -> ck.Outcome:
        att = s.att
        if c.kind == "colour_change":
            o = ck.colour_change(c, s.step, now, before, self.p)
            return ck.Outcome(o.status, o.reason, frames, o.wrong, o.marks)
        if c.kind == "series":
            o = ck.series(c, self.objects[c.object].grid, now, self._excluded, self.p)
            return ck.Outcome(o.status, o.reason, frames[:1], o.wrong, o.marks)
        o = ck.activity_at(c, att.motion, att.tips, self._hands_tracked)
        evidence = tuple(f"seq:{n}" for n in att.motion_seqs) + tuple(f"hands:{n}" for n in att.hands_seqs)
        return ck.Outcome(o.status, o.reason, evidence if o.status == "pass" else ())

    def _activity_seen(self, s: StepRun) -> bool:
        refs = list(s.step.targets) or [c.region for c in s.step.checks if c.region is not None]
        if not refs:
            return bool(s.att.motion)
        return any(region_hit(r, s.att.motion) for r in refs)

    def _provisional(self, s: StepRun, reason: str) -> None:
        att = s.att
        att.ready = False
        att.deadline = q_s(self._t + self._timeout)
        self._prov = s
        if self._active is s:
            self._active = None
        self._set_state(s, "provisional", reason)
        self._marks = ()
        self._request_readings(s)
        self._settle(s)
        if s.state == "provisional":
            self._advance()

    def _verified(self, s: StepRun) -> bool:
        """Every required check passed and at least one Meta check passed with recorded evidence."""
        out = s.att.outcomes
        meta_pass = any(c.kind in META_CHECKS and out.get(c.id) and out[c.id].status == "pass" for c in s.step.checks)
        if not meta_pass:
            return False
        for c in s.step.checks:
            if not c.required:
                continue
            st = out[c.id].status if c.id in out else "pending"
            if st != "pass":
                return False
        return True

    def _unverified_reason(self, s: StepRun) -> str:
        out = s.att.outcomes
        for c in s.step.checks:
            if c.required and c.kind != "confirm" and c.id in out and out[c.id].status != "pass":
                return out[c.id].reason
        if not any(c.kind in META_CHECKS and c.id in out and out[c.id].status == "pass" for c in s.step.checks):
            return "no Meta reading confirmed the step"
        return "confirm missing"

    def _resolve(self, s: StepRun) -> None:
        """Decide a provisional step once its Meta readings are in (or timed out)."""
        out = s.att.outcomes
        contradictions = {
            c.id for c in s.step.checks if c.kind == "judge"
            and isinstance(s.att.readings.get(f"verdict:{c.id}"), dict)
            and s.att.readings[f"verdict:{c.id}"].get("contradiction")
        }
        failed = [c for c in s.step.checks if (c.required or c.id in contradictions)
                  and c.id in out and out[c.id].status == "fail"]
        if failed:
            marks = tuple(m for c in failed for m in out[c.id].marks)
            if not marks:
                marks = tuple(ck.Mark("warn", ref=r) for r in s.step.targets)
            self._fail(s, out[failed[0].id].reason, [c.id for c in failed], marks)
            return
        if any(c.kind in META_CHECKS and (c.id not in out or out[c.id].status == "pending") for c in s.step.checks):
            return
        if self._verified(s):
            self._pass(s)
            return
        if s.att.timed_out and any(r.on == "timeout" for r in s.step.recovery):
            self._fail(s, "verification timed out", ["timeout"], tuple(ck.Mark("warn", ref=r) for r in s.step.targets))
            return
        if s.att.needs_confirm:
            return
        if not s.att.timed_out and self._unverified_reason(s) != "confirm missing":
            return  # Inconclusive readings stay provisional until the event-time deadline.
        s.att.needs_confirm = True
        self._set_state(s, "provisional", "needs confirm: " + self._unverified_reason(s))
        self._say("Say done to confirm." if self._unverified_reason(s) == "confirm missing"
                  else "I could not verify that step. Say done if it is right.")

    def _pass(self, s: StepRun) -> None:
        if self._prov is s:
            self._prov = None
        if self._active is s:
            self._active = None
        self._set_state(s, "passed")
        self._marks = ()
        self._last_reason = ""
        self._advance()

    def _end(self, s: StepRun, state: str, reason: str) -> None:
        """Terminal states other than passed/failed: unverified, skipped."""
        if self._prov is s:
            self._prov = None
        if self._active is s:
            self._active = None
        self._set_state(s, state, reason)
        self._advance()

    def _fail(self, s: StepRun, reason: str, keys: list[str], marks: tuple[ck.Mark, ...]) -> None:
        was_provisional = self._prov is s
        if was_provisional:
            self._prov = None
        if self._active is s:
            self._active = None
        self._set_state(s, "failed", reason)
        self._last_reason = reason
        self._marks = marks
        self._say(_sentence(reason), "interrupt")
        if was_provisional and self._active is not None:
            self._set_state(self._active, "waiting", f"suspended: {s.step.id} failed")
            self._active = None
        entries = [r for r in s.step.recovery if r.on in keys or r.on == "*"] or [Recovery(on="*", action="redo_step")]
        for r in entries:
            self._recover(s, r)
            self._say(r.say)
            if self._aborted:
                return
        self._advance()

    def _recover(self, s: StepRun, r: Recovery) -> None:
        if r.action == "exclude_cells":
            self._excluded.update(r.cells)
        elif r.action == "skip_steps":
            for sid in r.steps:
                t = self._steps[sid] if sid in self._steps else None
                if t is None or t.state in TERMINAL:
                    continue
                if self._active is t:
                    self._active = None
                if self._prov is t:
                    self._prov = None
                self._set_state(t, "skipped", f"recovery of {s.step.id}")
        elif r.action == "redo_step":
            if s.state == "failed":
                self._set_state(s, "recovery", "redo")
        elif r.action == "insert_step":
            sid = r.step.id
            if sid not in self._steps:
                self._steps[sid] = StepRun(r.step)
                self._order.insert(self._order.index(s.step.id), sid)
                self._set_state(self._steps[sid], "waiting", f"inserted by {s.step.id}")
            elif self._steps[sid].state in TERMINAL:
                self._set_state(self._steps[sid], "waiting", f"inserted by {s.step.id}")
        elif r.action == "abort":
            self._aborted = True
            self._active = self._prov = None
            for sid in self._order:
                t = self._steps[sid]
                if t.state not in TERMINAL:
                    self._set_state(t, "skipped", "aborted")

    # ------------------------------------------------------------ Meta readings

    def _model(self, role: str) -> str | None:
        return self.models.get(role)

    def _sam_prompt(self, obj: str) -> SamPrompt:
        lo = self.layout.objects.get(obj)
        return lo.sam if lo is not None else SamPrompt(mode="noun", noun=self.objects[obj].noun)

    def _request_readings(self, s: StepRun) -> None:
        for c in s.step.checks:
            if c.kind == "judge":
                self._need(s, f"verdict:{c.id}", c)
            elif c.kind in SAM_CHECKS:
                if c.kind == "moved_to":
                    self._need(s, f"seg:{c.object}:before", c)
                self._need(s, f"seg:{c.object}:after", c)
                if c.kind == "count":
                    self._need(s, f"count:{c.object}", c)

    def _need(self, s: StepRun, key: str, c: Check) -> None:
        att = s.att
        if key in att.requested or key in att.readings:
            return
        kind, _, rest = key.partition(":")
        sid, n = s.step.id, att.n
        if kind == "verdict":
            model = self._model("verdict")
            if model is None:
                att.readings[key] = Failure("no verdict model configured")
                return
            ctx = {"step_id": sid, "instruction": s.step.say, "question": c.question, "expected": s.step.detail}
            self._intent(s, f"{sid}/{c.id}/{n}", "verdict", model, att.frame, ctx, key)
            return
        if kind == "count":
            model = self._model("ground")
            if model is None:
                att.readings[key] = Failure("no grounding model configured")
                return
            noun = self._sam_prompt(rest).noun
            self._intent(s, f"{sid}/count/{rest}/{n}", "ground", model, att.frame, {"nouns": [noun], "shape": "count"},
                         key)
            return
        obj, _, which = rest.partition(":")
        frame = att.frame if which == "after" else self._frame_record(att.ref)
        if frame is None:
            att.readings[key] = Failure("no keyframe before the step")
            return
        if self.backend != "meta" or self._model("segment") is None:
            att.readings[key] = Failure("no SAM segmenter on this backend")
            return
        prompt = self._sam_prompt(obj)
        suffix = "" if which == "after" else "-before"
        if prompt.mode == "grounded" and self._model("ground") is not None:
            self._intent(s, f"{sid}/ground{suffix}/{obj}/{n}", "ground", self._model("ground"), frame,
                         {"nouns": [prompt.noun], "shape": "box"}, key)
        else:
            self._intent(s, f"{sid}/segment{suffix}/{obj}/{n}", "segment", self._model("segment"), frame,
                         {"concept": prompt.noun}, key)

    def _frame_record(self, sha: str | None) -> Keyframe | None:
        if sha is None:
            return None
        return self._frames_by_sha.get(sha)

    def _intent(self, s: StepRun, iid: str, role: str, model: str, frame: Keyframe | None, ctx: dict, key: str,
                retry: bool = False) -> None:
        endpoint, request = self.fns.build_request(role, model=model, backend=self.backend,
                                                   frame_sha=frame.sha if frame else None, ctx=ctx)
        it = Intent(id=iid, role=role, model=model, endpoint=endpoint, request=request, ctx=ctx,
                    frame=frame.sha if frame else None, crop=frame.crop if frame else None, step=s.step.id,
                    attempt=s.att.n, key=key, retry=retry)
        if iid in self._intents:
            raise ReducerError(f"intent {iid!r} derived twice")
        self._intents[iid] = it
        s.att.requested[key] = iid
        self._emit("model_request", intent_id=iid, role=role, model=model, fingerprint=request_fingerprint(request),
                   frame_sha=it.frame, prompt_version=self.fns.prompt_versions[role])

    def _retry(self, it: Intent, res: dict) -> None:
        s = self._steps[it.step]
        if res["status"] == "incomplete":
            retry = {"reason": "length"}
        else:
            text = res["result"].get("text", "") if isinstance(res["result"], dict) else ""
            retry = {"reason": "invalid", "previous": text, "error": res["error"] or ""}
        frame = Keyframe(0, it.frame, it.crop) if it.frame is not None else None
        self._intent(s, f"{it.id}/retry", it.role, it.model, frame, {**it.ctx, "retry": retry}, it.key, retry=True)

    def _live(self, it: Intent) -> bool:
        """The intent belongs to the current attempt of a step that is still being verified."""
        s = self._steps[it.step]
        return (s.state == "provisional" and s.att is not None and s.att.n == it.attempt
                and not s.att.timed_out)

    def _deliver(self, s: StepRun, it: Intent, result: Any) -> None:
        att = s.att
        if it.key.startswith("seg:") and it.role == "ground":
            self._ground_to_segment(s, it, result)
        elif it.key.startswith("seg:"):
            att.readings[it.key] = self._masks(it, result)
        elif it.key.startswith("count:"):
            att.readings[it.key] = self._points(it, result)
        else:
            att.readings[it.key] = result
        att.by[it.key] = it.id
        self._settle(s)

    def _ground_to_segment(self, s: StepRun, it: Intent, result: Any) -> None:
        """Spark boxes -> SAM box prompts on the same frame, respecting the session's selected prompt mode."""
        if isinstance(result, Failure) or s.att.timed_out:
            s.att.readings[it.key] = result if isinstance(result, Failure) else Failure("grounding arrived too late")
            return
        x, y, w, h = it.crop
        prompts = []
        if not isinstance(result, Failure):
            for item in result.get("items", [])[:MAX_PROMPT_OBJECTS]:
                if "bbox" not in item:
                    continue
                x0, y0, x1, y1 = item["bbox"]
                bx0, by0 = math.floor(x0 * w / 1000), math.floor(y0 * h / 1000)
                bx1, by1 = min(math.ceil(x1 * w / 1000), w), min(math.ceil(y1 * h / 1000), h)
                if bx1 > bx0 and by1 > by0:
                    prompts.append({"id": len(prompts) + 1, "box": [bx0, by0, bx1, by1]})
        obj, which = it.key.split(":")[1:]
        if not prompts:
            s.att.readings[it.key] = Failure(f"no grounding box for {self._sam_prompt(obj).noun}")
            return
        ctx = {"object_prompts": prompts, "w": w, "h": h}
        suffix = "" if which == "after" else "-before"
        frame = Keyframe(0, it.frame, it.crop)
        s.att.requested.pop(it.key, None)
        self._intent(s, f"{s.step.id}/segment{suffix}/{obj}/{s.att.n}", "segment", self._model("segment"), frame, ctx,
                     it.key)

    def _masks(self, it: Intent, result: Any) -> Any:
        if isinstance(result, Failure):
            return result
        size = result.get("size")
        w, h = it.crop[2], it.crop[3]
        if size is not None and list(size) != [w, h]:
            return Failure(f"SAM answered for a {size[0]}x{size[1]} image, the keyframe is {w}x{h}")
        return [to_camera(m, it.crop) for m in result.get("masks", [])]

    def _points(self, it: Intent, result: Any) -> Any:
        if isinstance(result, Failure):
            return result
        for name, count in result.get("counts", {}).items():
            n_points = sum(1 for item in result.get("items", []) if item["name"] == name and "x" in item)
            if count != n_points:
                return Failure(f"Spark reports {count} {name}, but locates {n_points}")
        x, y, w, h = it.crop
        return [(x + item["x"] * w / 1000, y + item["y"] * h / 1000) for item in result.get("items", [])
                if "x" in item and "y" in item]

    def _settle(self, s: StepRun) -> None:
        """Evaluate every Meta check whose readings are in, then resolve the step."""
        att = s.att
        by = att.by
        for c in s.step.checks:
            if c.kind not in META_CHECKS:
                continue
            o = self._eval_meta(s, c)
            if o is None:
                continue
            keys = self._keys(c)
            evidence = tuple(f"intent:{by[k]}" for k in keys if k in by and not isinstance(att.readings[k], Failure))
            if evidence:
                evidence += (f"frame:{att.frame.sha}",)
            self._show(s, c, ck.Outcome(o.status, o.reason, evidence, o.wrong, o.marks))
        if s.state == "provisional":
            self._resolve(s)

    def _keys(self, c: Check) -> list[str]:
        if c.kind == "judge":
            return [f"verdict:{c.id}"]
        keys = ([f"seg:{c.object}:before"] if c.kind == "moved_to" else []) + [f"seg:{c.object}:after"]
        return keys + ([f"count:{c.object}"] if c.kind == "count" else [])

    def _eval_meta(self, s: StepRun, c: Check) -> ck.Outcome | None:
        r = s.att.readings
        keys = self._keys(c)
        if any(k not in r for k in keys):
            return None
        sam = [r[k] for k in keys if k.startswith("seg:")]
        fails = [x for x in sam if isinstance(x, Failure)]
        if c.kind == "judge":
            v = r[keys[0]]
            return ck.Outcome("inconclusive", v.reason) if isinstance(v, Failure) else ck.judge(v)
        if fails:
            return ck.Outcome("inconclusive", fails[0].reason)
        noun = self._sam_prompt(c.object).noun
        if c.kind == "present":
            return ck.present(c, noun, sam[0], self.regions, self.p)
        if c.kind == "absent":
            return ck.absent(c, noun, sam[0], self.regions, self.p)
        if c.kind == "moved_to":
            return ck.moved_to(c, noun, sam[0], sam[1], self.regions, self.p)
        spark = r[f"count:{c.object}"]
        if isinstance(spark, Failure):
            return ck.Outcome("inconclusive", spark.reason)
        return ck.count(c, noun, sam[0], spark, self.regions, self.p)

    def _expire(self) -> None:
        s = self._prov
        if s is None or s.att.deadline_expired or s.att.deadline is None or self._t < s.att.deadline:
            return
        if s.att.needs_confirm and self._unverified_reason(s) == "confirm missing":
            return  # The model verification already succeeded; waiting for a human is not a model timeout.
        s.att.timed_out = True
        s.att.deadline_expired = True
        missing = sorted(k for k in s.att.requested if k not in s.att.readings)
        for k in missing:
            s.att.readings[k] = Failure(f"no reading within {self._timeout:g} s")
        self._settle(s)
