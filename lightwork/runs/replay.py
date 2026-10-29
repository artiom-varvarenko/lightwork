"""Replay engine (PLAN.md §4.2, D16): ``verify`` paced by the recorded ``t`` and shown on the screen page.

Inputs are fed to a fresh reducer at ``t / speed`` after the start. Every scene carries a provenance-aware replay
banner; synthetic and stand-in recordings are explicitly labelled. ``preview/<seq>.jpg`` is pushed after event ``<seq>``,
and the screen state (steps, chips, trace) is folded from the same events as in live mode. Model responses come from
``calls/`` only; differences from the log are reported at the end exactly as ``verify`` reports them.
"""

from __future__ import annotations

import re
import time
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from ..contracts.calls import CallMeta
from ..contracts.events import Event
from ..contracts.run import VerifyReport
from ..server.screen_state import ScreenState
from .manifest import RUN_ID_RE, created_utc
from .store import DirStore
from .verify import ReducerFactory, Verifier, verify_run

_CAMERA_MODE_RE = re.compile(r"(\d+)x(\d+)")


def replay_banner(when_utc: str | None, provenance: str = "unknown") -> str:
    """Label test and stand-in data distinctly from recorded Meta runs."""
    label = {
        "meta": "REPLAY of recorded run",
        "synthetic": "SYNTHETIC TEST REPLAY — not model evidence",
        "dev-standin": "DEV STAND-IN REPLAY — not submission evidence",
    }.get(provenance, "UNVERIFIED PROVENANCE REPLAY")
    when = f"{when_utc[:10]} {when_utc[11:16]} UTC" if when_utc else "(date unknown)"
    return f"{label} {when}"


def event_line(ev_t: float, type_: str, payload: dict[str, Any]) -> str | None:
    """One terminal line for the events a viewer follows (steps, checks, speech, calls, commands)."""
    p = payload
    if type_ == "step":
        text = f"step   {p['step']} -> {p['state']}" + (f" ({p['reason']})" if p["reason"] else "")
    elif type_ == "check" and p["status"] != "pending":
        text = f"check  {p['step']}/{p['check']} {p['status']}" + (f": {p['reason']}" if p["reason"] else "")
    elif type_ == "say":
        text = f'say    "{p["text"]}"'
    elif type_ == "model_result":
        text = f"model  {p['intent_id']} {p['status']} {p['latency_ms']} ms"
    elif type_ == "voice":
        text = f'voice  "{p["command"]}" ({p["transcript"]})'
    elif type_ in ("key", "ui"):
        text = f"{type_:<6} {p.get('key') or p.get('action')}"
    else:
        return None
    return f"{ev_t:9.3f}  {text}"


class ReplayEngine:
    """Replays one run directory. ``hubs`` are the server's ``render_hub``/``preview_hub``/``control_hub``; without
    them the replay is headless (terminal lines only)."""

    def __init__(
        self,
        run_dir: Path,
        *,
        speed: float = 1.0,
        hubs: Mapping[str, Any] | None = None,
        wait_for_screen: bool = False,
        reducer_factory: ReducerFactory | None = None,
        echo: Callable[[str], None] = print,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if not speed > 0:
            raise ValueError("speed must be > 0")
        self.run_dir = Path(run_dir)
        self.speed = speed
        hubs = hubs or {}
        self.render_hub = hubs.get("render_hub")
        self.preview_hub = hubs.get("preview_hub")
        self.control_hub = hubs.get("control_hub")
        self.wait_for_screen = wait_for_screen
        self.reducer_factory = reducer_factory
        self.echo = echo
        self.clock = clock
        self.sleep = sleep
        self.banner = replay_banner(None)
        self._screen: ScreenState | None = None
        self._previews: list[tuple[int, Path]] = []
        self._t_start = 0.0

    def run(self) -> VerifyReport:
        return verify_run(self.run_dir, reducer_factory=self.reducer_factory, write=False, observer=self)

    # ---------------------------------------------------------------- verify observer

    def start(self, v: Verifier) -> None:
        run_id = v.started["run_id"] if v.started else self.run_dir.name
        if v.manifest is not None:
            when = v.manifest.created_utc
        else:
            when = created_utc(run_id) if RUN_ID_RE.match(run_id) else None
        self.banner = replay_banner(when, v.manifest.provenance if v.manifest else "unknown")
        self._previews = DirStore(self.run_dir).previews()
        self.echo(f"{self.banner}: {run_id} at {self.speed:g}x")
        if self.control_hub is not None:
            state: dict[str, Any] = {
                "run": {
                    "id": run_id,
                    "pack": v.started["pack"] if v.started else "",
                    "mode": "replay",
                    "backend": v.started["backend"] if v.started else "",
                    "provenance": v.manifest.provenance if v.manifest else "unknown",
                },
                "banner": self.banner,
            }
            if v.inputs is not None:
                stepgraph, layout, calib = v.inputs
                self._screen = ScreenState(stepgraph)
                state.update(self._screen.snapshot())
                state["preview_xywh"] = list(layout.work_area_xywh)
                m = _CAMERA_MODE_RE.search(calib.camera.mode)
                if m:
                    state["camera"] = {"w": int(m.group(1)), "h": int(m.group(2))}
            self.control_hub.push_state(state)
        if self.wait_for_screen and self.render_hub is not None:
            while self.render_hub.hello("screen") is None:
                self.sleep(0.1)
        self._t_start = self.clock()

    def before_input(self, ev: Event) -> None:
        delay = self._t_start + ev.t / self.speed - self.clock()
        if delay > 0:
            self.sleep(delay)

    def after_input(self, ev: Event, derived: list[tuple[str, dict]], reducer: Any, last_seq: int) -> None:
        for type_, payload in [(ev.type, ev.payload), *derived]:
            line = event_line(ev.t, type_, payload)
            if line is not None:
                self.echo(line)
            if self._screen is not None and self.control_hub is not None:
                patch = self._screen.feed(type_, payload)
                if type_ == "model_result":
                    patch.update(self._call_cost_patch(payload))
                if patch:
                    self.control_hub.push_state(patch)
        if self.render_hub is not None and any(type_ == "scene" for type_, _ in derived):
            self.render_hub.publish_scene(reducer.scene.model_copy(update={"banner": self.banner}))
        due = [p for p in self._previews if p[0] <= last_seq]
        if due:
            self._previews = self._previews[len(due) :]
            if self.preview_hub is not None:
                seq, path = due[-1]
                self.preview_hub.push(seq, ev.t, path.read_bytes())

    def _call_cost_patch(self, payload: dict[str, Any]) -> dict[str, Any]:
        try:
            call = DirStore(self.run_dir).call_path(payload["call_dir"])
            meta = CallMeta.model_validate_json((call / "meta.json").read_bytes())
        except (OSError, ValueError):
            return {}  # verify reports the unreadable call
        return self._screen.set_call_cost(payload["intent_id"], meta.cost_usd)
