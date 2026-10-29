"""C1 Event envelope (``events.jsonl``, one event per line) and per-type payloads.

Envelope: ``{seq, t, type, src, derived, payload, h}``.

- ``seq`` is contiguous from 1. ``t`` is seconds since run start (laptop monotonic clock), quantized to 1e-3.
- ``h = sha256(prev_h + canonical(event without h))``; ``prev_h`` for seq 1 is ``sha256(run_id)``.
- Input events are facts the reducer cannot recompute; derived events are recomputed by the reducer (PLAN.md §4.2).
  ``derived`` must agree with the type's class.

Payload fields are always present (``None`` is written as ``null``) so the canonical form is unambiguous.
Evidence references in ``check.evidence`` are strings with a prefix: ``frame:<sha>``, ``intent:<intent_id>``,
``seq:<n>`` (another event), ``hands:<seq>``, ``voice:<seq>``.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from .canonical import canonical, sha256_hex


class Payload(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


# ---------------------------------------------------------------- input payloads


class RunStarted(Payload):
    run_id: str
    pack: str
    backend: Literal["meta", "chat"]
    models: dict[str, str]  # role -> model id
    code_commit: str
    dirty: bool
    platform: str
    version: str


class ProcedureLoaded(Payload):
    stepgraph_sha: str


class CalibrationLoaded(Payload):
    calib_sha: str


class LayoutLoaded(Payload):
    layout_sha: str


class Keyframe(Payload):
    frame_sha: str
    crop_xywh: list[int] = Field(min_length=4, max_length=4)  # work-area crop in camera px
    mode: Literal["grey", "dark", "normal"]
    verified: bool
    t_capture: float


class Motion(Payload):
    state: Literal["moving", "settled"]
    active_regions: list[str]


class Hands(Payload):
    over_regions: list[str]
    tip_xy: Annotated[list[float], Field(min_length=2, max_length=2)] | None  # camera px


class Voice(Payload):
    command: Literal["next", "done", "repeat", "why", "back"]
    transcript: str
    session_id: str
    turn_id: str


class VoiceRejected(Payload):
    transcript: str


class Key(Payload):
    key: str


class Ui(Payload):
    action: str
    args: dict[str, Any]


class ModelResult(Payload):
    intent_id: str
    call_dir: str  # relative to the run dir, e.g. "calls/0003-verdict-muse-spark-1.3-contributor"
    status: Literal["ok", "error", "timeout"]
    latency_ms: int


class Tick(Payload):
    pass


class RunStopped(Payload):
    reason: str


# ---------------------------------------------------------------- derived payloads

ModelRole = Literal["verdict", "ground", "author", "why", "segment"]


class ModelRequest(Payload):
    intent_id: str
    role: ModelRole
    model: str
    fingerprint: str
    frame_sha: str | None
    prompt_version: str


class CellReading(Payload):
    od: float
    lab: list[float] = Field(min_length=3, max_length=3)
    n: int
    sd: float


class Measurement(Payload):
    frame_sha: str
    channel: Literal["r", "g", "b", "lab"]
    cells: dict[str, CellReading]


class Parsed(Payload):
    intent_id: str
    kind: str  # verdict | ground | segment | why | author | error
    result: dict[str, Any]


CheckStatus = Literal["pass", "fail", "pending", "inconclusive"]


class Check(Payload):
    step: str
    check: str
    status: CheckStatus
    reason: str
    evidence: list[str]


StepState = Literal[
    "waiting", "active", "provisional", "passed", "failed", "recovery", "unverified", "skipped", "excluded"
]


class Step(Payload):
    step: str
    state: StepState
    reason: str


class Scene(Payload):
    hash: str


class Say(Payload):
    text: str
    priority: Literal["normal", "interrupt"]


class Cost(Payload):
    usd_total: float


INPUT_PAYLOADS: dict[str, type[Payload]] = {
    "run_started": RunStarted,
    "procedure_loaded": ProcedureLoaded,
    "calibration_loaded": CalibrationLoaded,
    "layout_loaded": LayoutLoaded,
    "keyframe": Keyframe,
    "motion": Motion,
    "hands": Hands,
    "voice": Voice,
    "voice_rejected": VoiceRejected,
    "key": Key,
    "ui": Ui,
    "model_result": ModelResult,
    "tick": Tick,
    "run_stopped": RunStopped,
}

DERIVED_PAYLOADS: dict[str, type[Payload]] = {
    "model_request": ModelRequest,
    "measurement": Measurement,
    "parsed": Parsed,
    "check": Check,
    "step": Step,
    "scene": Scene,
    "say": Say,
    "cost": Cost,
}

PAYLOADS: dict[str, type[Payload]] = {**INPUT_PAYLOADS, **DERIVED_PAYLOADS}
INPUT_TYPES = frozenset(INPUT_PAYLOADS)
DERIVED_TYPES = frozenset(DERIVED_PAYLOADS)


class Event(BaseModel):
    """One line of ``events.jsonl``. ``payload`` is the canonical dict of the type's payload model."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    seq: int = Field(ge=1)
    t: float = Field(ge=0)
    type: str
    src: str  # producer: camera | gate | hands | voice | ui | key | model | reducer | orchestrator
    derived: bool
    payload: dict[str, Any]
    h: str = ""

    def body(self) -> dict[str, Any]:
        """The event without ``h`` (what the hash covers)."""
        d = self.model_dump(mode="json")
        d.pop("h")
        return d

    def typed_payload(self) -> Payload:
        return PAYLOADS[self.type].model_validate(self.payload)

    def line(self) -> str:
        return canonical(self.model_dump(mode="json"))


def make_payload(type_: str, payload: Payload | dict[str, Any]) -> dict[str, Any]:
    """Validate a payload against its type and return its canonical dict form (all fields present)."""
    model = PAYLOADS[type_]
    if isinstance(payload, Payload):
        if not isinstance(payload, model):
            raise TypeError(f"{type_} expects {model.__name__}, got {type(payload).__name__}")
        obj = payload
    else:
        obj = model.model_validate(payload)
    return obj.model_dump(mode="json")


def genesis_hash(run_id: str) -> str:
    return sha256_hex(run_id)


def event_hash(prev_h: str, body: dict[str, Any]) -> str:
    return sha256_hex(prev_h + canonical(body))


def seal(prev_h: str, seq: int, t: float, type_: str, src: str, payload: Payload | dict[str, Any]) -> Event:
    """Build a hashed event. ``t`` must already be quantized (``q_s``)."""
    if type_ not in PAYLOADS:
        raise ValueError(f"unknown event type {type_!r}")
    ev = Event(seq=seq, t=t, type=type_, src=src, derived=type_ in DERIVED_TYPES, payload=make_payload(type_, payload))
    return ev.model_copy(update={"h": event_hash(prev_h, ev.body())})


class ChainError(ValueError):
    pass


def validate_event(ev: Event) -> None:
    if ev.type not in PAYLOADS:
        raise ChainError(f"seq {ev.seq}: unknown type {ev.type!r}")
    if ev.derived != (ev.type in DERIVED_TYPES):
        raise ChainError(f"seq {ev.seq}: derived flag {ev.derived} wrong for {ev.type}")
    canon = make_payload(ev.type, ev.payload)
    if canonical(canon) != canonical(ev.payload):
        raise ChainError(f"seq {ev.seq}: payload not in canonical form")


def check_chain(run_id: str, events: Iterable[Event]) -> None:
    """Raise ChainError on the first broken link, gap, or invalid payload."""
    prev = genesis_hash(run_id)
    expected_seq = 1
    for ev in events:
        if ev.seq != expected_seq:
            raise ChainError(f"seq gap: expected {expected_seq}, got {ev.seq}")
        validate_event(ev)
        h = event_hash(prev, ev.body())
        if h != ev.h:
            raise ChainError(f"seq {ev.seq}: hash mismatch")
        prev = h
        expected_seq += 1


def read_events(path) -> list[Event]:
    import json

    out = []
    with open(path, encoding="utf-8") as f:
        for n, line in enumerate(f, 1):
            line = line.rstrip("\n")
            if not line:
                continue
            try:
                out.append(Event.model_validate(json.loads(line)))
            except Exception as e:  # noqa: BLE001 - report line number
                raise ChainError(f"{path}:{n}: {e}") from e
    return out
