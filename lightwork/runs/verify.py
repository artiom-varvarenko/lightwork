"""Offline verification of a recorded run (PLAN.md §4.2; D1, D2). Reads only the run directory and the code: no key,
no network.

1. The event log: canonical lines, payload schemas, sequence and hash chain.
2. The files against ``manifest.json`` (hashes, size, identity, totals, provenance); frames against their content
   address. A run without ``manifest.json`` is incomplete (the recorder never closed it).
3. ``stepgraph.json``, ``layout.json`` and ``calib.json`` against their contracts and the logged ``*_loaded`` hashes.
4. Re-derivation: a fresh ``core.Reducer`` is fed every input event in log order. What it returns for an input must
   equal the derived events logged right after that input, in canonical JSON with the input's ``t`` and
   ``src="reducer"``.
5. Calls: ``request_fingerprint(request.json)`` against ``meta.json`` and against the reducer's re-derived intent.
   Optional ``model_result.usage`` and ``http_status`` must match C5 metadata; aggregate usage must match the final
   response plus every retained raw retry attempt.

Diff classes (``contracts.run.Diff.cls``):

- ``chain``: a sequence gap or hash mismatch in ``events.jsonl`` (an edited, removed or reordered event).
- ``schema``: a line, payload or file that fails its contract, or a reducer output that does.
- ``manifest``: files disagree with ``manifest.json`` or with the logged ``*_loaded`` hashes, or a frame's bytes do
  not match its content address.
- ``derived``: the re-derived stream differs from the logged one: an event differs at the same position, the reducer
  emits an event the log lacks (``seq`` = the input that causes it) or no longer emits a logged one, or the reducer
  fails on the logged inputs.
- ``fingerprint``: a recorded request differs from the request the reducer builds now (``model_request`` events and
  ``calls/*/request.json``), or ``meta.json`` disagrees with ``request.json``.
- ``missing``: the run lacks a file named by the manifest or by an event, or ``manifest.json`` itself (incomplete).
- ``extra``: the run has a file the manifest does not list, or a call no ``model_result`` references.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any, Protocol

from .. import __version__
from ..contracts.calls import CallMeta, request_fingerprint
from ..contracts.canonical import canonical, canonical_sha
from ..contracts.events import DERIVED_TYPES, Event, event_hash, genesis_hash, make_payload, validate_event
from ..contracts.layout import Calibration, Layout
from ..contracts.run import Diff, Manifest, VerifyReport
from ..contracts.stepgraph import StepGraph, validate_semantics
from ..models.pricing import aggregate_usage
from ..models.wire import read_body
from .manifest import files_size, hashed_files, run_totals
from .store import CALIB, EVENTS, LAYOUT, MANIFEST, RESPONSE_FILES, RUN_INPUT_FILES, STEPGRAPH, VERIFY, DirStore, write_atomic

ReducerFactory = Callable[..., Any]  # core.Reducer(stepgraph, layout, calib, store, *, backend, models)

LOADED = {
    "procedure_loaded": ("stepgraph_sha", STEPGRAPH),
    "calibration_loaded": ("calib_sha", CALIB),
    "layout_loaded": ("layout_sha", LAYOUT),
}
IDENTITY = ("code_commit", "dirty", "version", "platform", "backend", "models")
_ABSENT = object()


class Observer(Protocol):
    """Hooks of a paced replay: ``start`` after the static checks, then around each re-derived input."""

    def start(self, v: Verifier) -> None: ...

    def before_input(self, ev: Event) -> None: ...

    def after_input(self, ev: Event, derived: list[tuple[str, dict]], reducer: Any, last_seq: int) -> None: ...


def default_reducer_factory() -> ReducerFactory:
    from ..core import Reducer

    return Reducer


def verify_run(
    run_dir: Path,
    *,
    reducer_factory: ReducerFactory | None = None,
    write: bool = True,
    observer: Observer | None = None,
) -> VerifyReport:
    """Verify one run directory and (by default) write its ``verify.json``."""
    run_dir = Path(run_dir)
    v = Verifier(run_dir)
    v.check_log()
    v.check_files()
    v.load_inputs()
    if observer is not None:
        observer.start(v)
    v.rederive(reducer_factory or default_reducer_factory(), observer)
    v.check_remaining_calls()
    v.check_unreferenced_calls()
    report = v.report()
    if write:
        write_atomic(run_dir / VERIFY, (canonical(report) + "\n").encode("utf-8"))
    return report


class Verifier:
    """The individual checks; ``verify_run`` runs them all, ``audit`` reuses the static ones."""

    def __init__(self, run_dir: Path) -> None:
        self.run_dir = run_dir
        self.store = DirStore(run_dir)
        self.diffs: list[Diff] = []
        self.events: list[Event] = []
        self.invalid: set[int] = set()  # seqs whose payload fails its schema
        self.started: dict[str, Any] | None = None
        self.manifest: Manifest | None = None
        self.inputs: tuple[StepGraph, Layout, Calibration] | None = None
        self._checked_calls: set[int] = set()

    def add(self, cls: str, detail: str, *, seq: int | None = None, path: str | None = None) -> None:
        self.diffs.append(Diff(cls=cls, seq=seq, path=path, detail=detail))

    def valid(self, types: tuple[str, ...]) -> Iterator[Event]:
        return (ev for ev in self.events if ev.type in types and ev.seq not in self.invalid)

    # ---------------------------------------------------------------- 1. log

    def check_log(self) -> None:
        try:
            raw = (self.run_dir / EVENTS).read_bytes()
        except FileNotFoundError:
            self.add("missing", "no events.jsonl", path=EVENTS)
            return
        for n, line in enumerate(raw.splitlines(), 1):
            try:
                ev = Event.model_validate(json.loads(line))
            except ValueError as e:
                self.add("schema", f"line {n} is not a C1 event: {_short(e)}", path=EVENTS)
                continue
            if line != ev.line().encode("utf-8"):
                self.add("schema", f"line {n} is not canonical JSON", seq=ev.seq, path=EVENTS)
            self.events.append(ev)
        if not self.events:
            return
        first = self.events[0]
        run_id = first.payload.get("run_id", "") if first.type == "run_started" else ""
        prev, expected = genesis_hash(str(run_id)), 1
        for ev in self.events:
            if ev.seq != expected:
                self.add("chain", f"expected seq {expected}, found {ev.seq} (event removed, added or moved)", seq=ev.seq)
            try:
                validate_event(ev)
            except (ValueError, TypeError) as e:
                self.invalid.add(ev.seq)
                self.add("schema", f"{ev.type}: {_short(e)}", seq=ev.seq, path=EVENTS)
            if event_hash(prev, ev.body()) != ev.h:
                self.add("chain", f"{ev.type}: hash mismatch (event edited)", seq=ev.seq, path=EVENTS)
            prev, expected = ev.h, ev.seq + 1
        if first.type == "run_started" and first.seq not in self.invalid:
            self.started = first.payload
        else:
            self.add("schema", "the first event is not a valid run_started", seq=first.seq, path=EVENTS)

    # ---------------------------------------------------------------- 2. files

    def check_files(self) -> None:
        actual = hashed_files(self.run_dir)
        bad_address = {rel for rel, sha in actual.items() if rel.startswith("frames/") and rel != f"frames/{sha}.jpg"}
        try:
            self.manifest = Manifest.model_validate_json((self.run_dir / MANIFEST).read_bytes())
        except FileNotFoundError:
            self.add("missing", "incomplete run: no manifest.json (the recorder never closed it)", path=MANIFEST)
        except ValueError as e:
            self.add("schema", _short(e), path=MANIFEST)
        m = self.manifest
        if m is None:
            for rel in sorted(bad_address):
                self.add("manifest", "bytes do not match the frame's content address", path=rel)
            return
        for rel in sorted(set(m.files) | set(actual)):
            if rel not in actual:
                self.add("missing", "listed in manifest.json, not in the run", path=rel)
            elif rel not in m.files:
                self.add("extra", "not listed in manifest.json", path=rel)
            elif m.files[rel] != actual[rel]:
                self.add("manifest", f"sha256 {actual[rel][:12]}… differs from manifest {m.files[rel][:12]}…", path=rel)
            elif rel in bad_address:
                self.add("manifest", "bytes do not match the frame's content address", path=rel)
        size = files_size(self.run_dir, actual)
        if size != m.size_bytes:
            self.add("manifest", f"size_bytes {m.size_bytes} but the files hold {size}", path=MANIFEST)
        if self.started is not None:
            if m.run_id != self.started["run_id"]:
                self.add("manifest", f"run_id {m.run_id} but run_started says {self.started['run_id']}", path=MANIFEST)
            fields = [k for k in IDENTITY if canonical(getattr(m, k)) != canonical(self.started[k])]
            if fields:
                self.add("manifest", f"{', '.join(fields)} differ from run_started", path=MANIFEST)
        if self.events and not self.invalid:
            totals = run_totals(self.events)
            fields = [k for k, v in totals.model_dump().items() if canonical(v) != canonical(getattr(m.totals, k))]
            if fields:
                self.add("manifest", f"totals {', '.join(fields)} differ from the log", path=MANIFEST)
        if m.provenance == "meta":
            for rel, prov in sorted(call_provenances(self.run_dir).items()):
                if prov != "meta":
                    self.add("manifest", f"provenance is meta but the call says {prov!r}", path=f"{rel}/meta.json")

    # ---------------------------------------------------------------- 3. run inputs

    def load_inputs(self) -> None:
        objs: dict[str, Any] = {}
        for name, model in RUN_INPUT_FILES.items():
            try:
                objs[name] = model.model_validate_json((self.run_dir / name).read_bytes())
            except FileNotFoundError:
                if self.manifest is None or name not in self.manifest.files:  # else check_files reported it
                    self.add("missing", f"no {name}", path=name)
            except ValueError as e:
                self.add("schema", _short(e), path=name)
        if STEPGRAPH in objs:
            errs = validate_semantics(objs[STEPGRAPH])
            if errs:
                more = f" (+{len(errs) - 1} more)" if len(errs) > 1 else ""
                self.add("schema", errs[0] + more, path=STEPGRAPH)
        for ev in self.valid(tuple(LOADED)):
            field, name = LOADED[ev.type]
            if name in objs and canonical_sha(objs[name]) != ev.payload[field]:
                self.add("manifest", f"{name} is not the one loaded at seq {ev.seq}", seq=ev.seq, path=name)
        for ev in self.valid(("keyframe",)):
            sha = ev.payload["frame_sha"]
            rel = f"frames/{sha}.jpg"
            try:
                exists = self.store.frame_path(sha).is_file()
            except ValueError as e:
                self.add("schema", _short(e), seq=ev.seq, path=rel)
                continue
            if not exists and (self.manifest is None or rel not in self.manifest.files):
                self.add("missing", "keyframe referenced by the event is absent", seq=ev.seq, path=rel)
        if len(objs) == len(RUN_INPUT_FILES):
            self.inputs = (objs[STEPGRAPH], objs[LAYOUT], objs[CALIB])

    # ---------------------------------------------------------------- 4. re-derivation

    def rederive(self, factory: ReducerFactory, observer: Observer | None = None) -> None:
        if self.started is None or self.inputs is None:
            self.add("derived", "not re-derived: run_started or a run input file is missing or invalid")
            return
        try:
            reducer = factory(*self.inputs, self.store, backend=self.started["backend"], models=self.started["models"])
        except Exception as e:  # noqa: BLE001 - any failure to rebuild the reducer fails the run
            self.add("derived", f"reducer failed to start: {type(e).__name__}: {_short(e)}")
            return
        intents: dict[str, str] = {}
        for inp, logged in _groups(self.events):
            if inp is None:
                for ev in logged:
                    self.add("derived", f"derived {ev.type} before any input", seq=ev.seq)
                continue
            if observer is not None:
                observer.before_input(inp)
            try:
                out = list(reducer.feed(inp))
            except Exception as e:  # noqa: BLE001 - reported as a diff; re-derivation cannot continue
                self.add("derived", f"reducer raised on {inp.type}: {type(e).__name__}: {_short(e)}", seq=inp.seq)
                return
            new = [self._canonical_output(inp, type_, payload) for type_, payload in out]
            self._compare(inp, logged, new)
            for type_, payload in new:
                if type_ == "model_request" and payload is not None:
                    intents[payload["intent_id"]] = payload["fingerprint"]
            if inp.type == "model_result" and inp.seq not in self.invalid:
                self.check_call(inp, intents.get(inp.payload["intent_id"]), "re-derived")
            if observer is not None:
                done = [(t, p) for t, p in new if p is not None]
                observer.after_input(inp, done, reducer, logged[-1].seq if logged else inp.seq)

    def _canonical_output(self, inp: Event, type_: str, payload: Any) -> tuple[str, dict | None]:
        if type_ not in DERIVED_TYPES:
            self.add("schema", f"reducer emitted non-derived type {type_!r}", seq=inp.seq)
            return type_, None
        try:
            return type_, make_payload(type_, payload)
        except (ValueError, TypeError) as e:
            self.add("schema", f"reducer emitted an invalid {type_}: {_short(e)}", seq=inp.seq)
            return type_, None

    def _compare(self, inp: Event, logged: list[Event], new: list[tuple[str, dict | None]]) -> None:
        for k in range(max(len(logged), len(new))):
            if k >= len(logged):
                type_, payload = new[k]
                self.add("derived", f"re-derived {type_} is not in the log: {_brief(payload)}", seq=inp.seq)
                continue
            ev = logged[k]
            if k >= len(new):
                self.add("derived", f"logged {ev.type} is not re-derived", seq=ev.seq)
                continue
            type_, payload = new[k]
            if payload is None:
                continue
            want = {"payload": payload, "src": "reducer", "t": inp.t, "type": type_}
            got = {"payload": ev.payload, "src": ev.src, "t": ev.t, "type": ev.type}
            if canonical(want) != canonical(got):
                same_intent = type_ == ev.type == "model_request" and payload["intent_id"] == ev.payload.get("intent_id")
                self.add("fingerprint" if same_intent else "derived", f"{ev.type} {_first_diff(got, want)}", seq=ev.seq)

    # ---------------------------------------------------------------- 5. calls

    def check_call(self, ev: Event, intent_fp: str | None, source: str) -> None:
        self._checked_calls.add(ev.seq)
        rel, intent = ev.payload["call_dir"], ev.payload["intent_id"]
        try:
            d = self.store.call_path(rel)
        except ValueError as e:
            self.add("schema", str(e), seq=ev.seq)
            return
        if not d.is_dir():
            self.add("missing", f"call directory of {intent}", seq=ev.seq, path=rel)
            return
        req_fp = meta = None
        try:
            req = json.loads((d / "request.json").read_bytes())
            if not isinstance(req, dict):
                raise ValueError("not a JSON object")
            req_fp = request_fingerprint(req)
        except FileNotFoundError:
            self.add("missing", f"request of {intent}", seq=ev.seq, path=f"{rel}/request.json")
        except ValueError as e:
            self.add("schema", f"request.json: {_short(e)}", seq=ev.seq, path=f"{rel}/request.json")
        try:
            meta = CallMeta.model_validate_json((d / "meta.json").read_bytes())
        except FileNotFoundError:
            self.add("missing", f"meta of {intent}", seq=ev.seq, path=f"{rel}/meta.json")
        except ValueError as e:
            self.add("schema", f"meta.json: {_short(e)}", seq=ev.seq, path=f"{rel}/meta.json")
        if not any((d / name).is_file() for name, _ in RESPONSE_FILES):
            self.add("missing", f"response of {intent}", seq=ev.seq, path=rel)
        if meta is not None:
            self.check_usage(ev, d, meta)
            if ev.payload.get("http_status") is not None and ev.payload["http_status"] != meta.http_status:
                self.add("schema", "model_result.http_status differs from meta.http_status", seq=ev.seq, path=f"{rel}/meta.json")
        if req_fp is None:
            return
        if meta is not None and meta.fingerprint != req_fp:
            self.add("fingerprint", f"meta.json says {meta.fingerprint[:12]}…, request.json is {req_fp[:12]}…", seq=ev.seq, path=f"{rel}/meta.json")
        if intent_fp is None:
            self.add("fingerprint", f"no {source} model_request for {intent}", seq=ev.seq, path=f"{rel}/request.json")
        elif intent_fp != req_fp:
            self.add(
                "fingerprint",
                f"request.json {req_fp[:12]}… differs from the {source} intent {intent} {intent_fp[:12]}…",
                seq=ev.seq,
                path=f"{rel}/request.json",
            )

    def check_usage(self, ev: Event, call: Path, meta: CallMeta) -> None:
        """Billed retry counters must be backed by the retained raw attempt responses, not just asserted in C1."""
        rel = ev.payload["call_dir"]
        supplied = ev.payload.get("usage")
        if supplied is not None and canonical(supplied) != canonical(meta.usage):
            self.add("schema", "model_result.usage differs from meta.usage", seq=ev.seq, path=f"{rel}/meta.json")
        paths: list[tuple[Path, str]] = []
        complete = True
        expected = set()
        for n in range(1, len(meta.attempts)):
            choices = [(call / "attempts" / f"{n:04d}.{kind}", kind) for kind in ("sse", "json")]
            present = [(p, kind) for p, kind in choices if p.is_file()]
            expected.update(p for p, _ in present)
            if not present:
                self.add("missing", f"raw evidence of billed attempt {n}", seq=ev.seq, path=f"{rel}/attempts/{n:04d}.*")
                complete = False
            elif len(present) != 1:
                self.add("schema", f"multiple responses for attempt {n}", seq=ev.seq, path=f"{rel}/attempts")
                complete = False
            else:
                paths.extend(present)
        attempts = call / "attempts"
        if attempts.is_dir():
            for path in sorted(attempts.iterdir()):
                if path.is_file() and path not in expected:
                    self.add("extra", "response has no matching attempt", seq=ev.seq, path=path.relative_to(self.run_dir).as_posix())
        final = [(call / name, kind) for name, kind in RESPONSE_FILES if (call / name).is_file()]
        if len(final) != 1:
            if final:
                self.add("schema", "call has multiple final response files", seq=ev.seq, path=rel)
            return  # the missing-final-response diff is emitted by check_call
        paths.extend(final)
        usages = []
        for path, kind in paths:
            try:
                usages.append(read_body(path.read_bytes(), kind, meta.endpoint).usage)
            except (OSError, ValueError, TypeError) as e:
                self.add("schema", f"cannot read usage: {_short(e)}", seq=ev.seq, path=path.relative_to(self.run_dir).as_posix())
                complete = False
        if complete and canonical(aggregate_usage(usages)) != canonical(meta.usage):
            self.add("schema", "meta.usage differs from aggregate raw response usage", seq=ev.seq, path=f"{rel}/meta.json")

    def check_remaining_calls(self) -> None:
        """Calls that re-derivation did not reach, checked against the logged intents."""
        logged: dict[str, str] = {}
        for ev in self.valid(("model_request", "model_result")):
            if ev.type == "model_request":
                logged[ev.payload["intent_id"]] = ev.payload["fingerprint"]
            elif ev.seq not in self._checked_calls:
                self.check_call(ev, logged.get(ev.payload["intent_id"]), "logged")

    def check_unreferenced_calls(self) -> None:
        referenced = {ev.payload["call_dir"] for ev in self.valid(("model_result",))}
        calls = self.run_dir / "calls"
        if calls.is_dir():
            for d in sorted(p for p in calls.iterdir() if p.is_dir()):
                if f"calls/{d.name}" not in referenced:
                    self.add("extra", "no model_result references this call", path=f"calls/{d.name}")

    def report(self) -> VerifyReport:
        n_derived = sum(ev.derived for ev in self.events)
        return VerifyReport(
            run_id=self.started["run_id"] if self.started else self.run_dir.name,
            ok=not self.diffs,
            n_events=len(self.events),
            n_inputs=len(self.events) - n_derived,
            n_derived=n_derived,
            n_calls=sum(ev.type == "model_result" for ev in self.events),
            diffs=self.diffs,
            code_version=__version__,
        )


def format_diff(d: Diff) -> str:
    where = ", ".join(x for x in (f"seq {d.seq}" if d.seq is not None else "", d.path or "") if x)
    return f"{d.cls:<11} {where + ': ' if where else ''}{d.detail}"


def report_lines(report: VerifyReport, limit: int = 20) -> list[str]:
    """``<run_id>: 0 diffs (24 events, 1 call)`` followed by the first ``limit`` diffs."""
    n, calls = len(report.diffs), report.n_calls
    head = f"{report.run_id}: {n} diff{'' if n == 1 else 's'} ({report.n_events} events, {calls} call{'' if calls == 1 else 's'})"
    lines = [head, *("  " + format_diff(d) for d in report.diffs[:limit])]
    if n > limit:
        lines.append(f"  ... and {n - limit} more")
    return lines


def call_provenances(run_dir: Path) -> dict[str, str]:
    """``{calls/<dir>: provenance}`` from each call's ``meta.json`` ("unreadable" when it cannot be read)."""
    out = {}
    for meta in sorted(run_dir.glob("calls/*/meta.json")):
        try:
            prov = json.loads(meta.read_bytes()).get("provenance")
        except (OSError, ValueError, AttributeError):
            prov = None
        out[meta.parent.relative_to(run_dir).as_posix()] = prov if isinstance(prov, str) else "unreadable"
    return out


def _groups(events: list[Event]) -> Iterator[tuple[Event | None, list[Event]]]:
    """Each input event with the derived events logged right after it (a leading derived run has no input)."""
    inp: Event | None = None
    logged: list[Event] = []
    for ev in events:
        if ev.derived:
            logged.append(ev)
            continue
        if inp is not None or logged:
            yield inp, logged
        inp, logged = ev, []
    if inp is not None or logged:
        yield inp, logged


def _first_diff(a: Any, b: Any, path: str = "") -> str:
    """Path and values of the first difference between a logged (a) and a re-derived (b) JSON value."""
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b)):
            x, y = a.get(k, _ABSENT), b.get(k, _ABSENT)
            if x is _ABSENT or y is _ABSENT or canonical(x) != canonical(y):
                return _first_diff(x, y, f"{path}.{k}" if path else k)
    elif isinstance(a, list) and isinstance(b, list) and len(a) == len(b):
        for i, (x, y) in enumerate(zip(a, b, strict=True)):
            if canonical(x) != canonical(y):
                return _first_diff(x, y, f"{path}[{i}]")
    return f"{path or 'value'}: logged {_brief(a)}, re-derived {_brief(b)}"


def _brief(value: Any, limit: int = 80) -> str:
    text = "absent" if value is _ABSENT else canonical(value)
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _short(e: Exception, limit: int = 160) -> str:
    text = " ".join(str(e).split())
    return text if len(text) <= limit else text[: limit - 1] + "…"
