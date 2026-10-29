"""Folds C1 events into the ``screen.html`` state pushed through ``ControlHub.push_state`` (keys documented in
``hub.py``). Live and replay feed the same events, so both show the same step list, evidence chips and trace.
"""

from __future__ import annotations

from typing import Any

from ..contracts.stepgraph import StepGraph

TRACE_MAX = 50


class ScreenState:
    def __init__(self, stepgraph: StepGraph):
        self._steps: list[dict[str, Any]] = [{"id": s.id, "say": s.say, "state": "waiting", "reason": ""} for s in stepgraph.steps]
        self._active: str | None = None
        self._chips: dict[tuple[str, str], dict[str, Any]] = {}
        self._trace: dict[str, dict[str, Any]] = {}
        self._cost = 0.0
        self._last_say = ""

    def snapshot(self) -> dict[str, Any]:
        return {
            "steps": [dict(s) for s in self._steps],
            "active": self._active,
            "chips": [dict(c) for c in self._chips.values()],
            "trace": [dict(r) for r in list(self._trace.values())[-TRACE_MAX:]],
            "cost_usd": self._cost,
            "last_say": self._last_say,
        }

    def feed(self, type_: str, payload: dict[str, Any]) -> dict[str, Any]:
        """Apply one event; returns the state patch to push (empty when the event does not show on the screen)."""
        snap = self.snapshot
        if type_ == "step":
            row = next((s for s in self._steps if s["id"] == payload["step"]), None)
            if row is None:  # inserted by a recovery
                row = {"id": payload["step"], "say": "", "state": "waiting", "reason": ""}
                self._steps.append(row)
            row["state"] = payload["state"]
            row["reason"] = payload.get("reason", "")
            if payload["state"] == "active":
                self._active = payload["step"]
            return {k: snap()[k] for k in ("steps", "active")}
        if type_ == "check":
            key = (payload["step"], payload["check"])
            self._chips[key] = {k: payload[k] for k in ("step", "check", "status", "reason")}
            return {"chips": snap()["chips"]}
        if type_ == "model_request":
            self._trace[payload["intent_id"]] = {
                "intent_id": payload["intent_id"],
                "role": payload["role"],
                "model": payload["model"],
                "status": "pending",
                "latency_ms": None,
                "cost_usd": None,
            }
            return {"trace": snap()["trace"]}
        if type_ == "model_result":
            row = self._trace.get(payload["intent_id"])
            if row is None:
                return {}
            row["status"] = payload["status"]
            row["latency_ms"] = payload["latency_ms"]
            return {"trace": snap()["trace"]}
        if type_ == "cost":
            self._cost = payload["usd_total"]
            return {"cost_usd": self._cost}
        if type_ == "say":
            self._last_say = payload["text"]
            return {"last_say": self._last_say}
        return {}

    def set_call_cost(self, intent_id: str, cost_usd: float) -> dict[str, Any]:
        """Per-call cost from the call's ``meta.json`` (C5), which the event stream does not carry."""
        row = self._trace.get(intent_id)
        if row is None:
            return {}
        row["cost_usd"] = cost_usd
        return {"trace": self.snapshot()["trace"]}
