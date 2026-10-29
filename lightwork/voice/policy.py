"""Upstream session failure policy (briefs/webvoice.md §1.2 close-code table, D3 backoff)."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Literal

ActionKind = Literal["done", "reconnect", "backoff", "off", "fallback"]

MAX_TRIES = 4  # D3: at most 4 tries for server-side failures
NETWORK_MAX_DELAY_S = 8.0
REPEAT_1008_WINDOW_S = 300.0


@dataclass(frozen=True)
class Action:
    kind: ActionKind
    why: str
    base_s: float = 0.0  # backoff start
    limited: bool = True  # False: network drops retry forever (capped delay); voice must survive a Wi-Fi blip


def close_action(code: int | None, reason: str, *, acked: bool, ended_by_us: bool, repeat_1008: bool) -> Action:
    """What to do after an upstream session closed.

    ``acked``: the handshake was acknowledged; ``ended_by_us``: we sent ``endStream``; ``repeat_1008``: another
    mid-stream 1008 happened within the last 5 min.
    """
    if code == 1000:
        return Action("done", "normal close") if ended_by_us else Action("reconnect", "unsolicited 1000")
    if code == 1008:
        if not acked:
            return Action("off", f"handshake rejected (1008): {reason}")
        if repeat_1008:
            return Action("off", f"second pacing failure (1008) within 5 min: {reason}")
        return Action("reconnect", f"pacing failure (1008): {reason}")
    if code == 1011:
        if "max session duration" in reason.lower():
            return Action("reconnect", "max session duration reached")
        return Action("backoff", f"upstream internal error (1011): {reason}", 0.5)
    if code == 1013:
        return Action("backoff", "rate limited (1013)", 2.0)
    return Action("backoff", f"connection lost ({code}): {reason}", 0.5, limited=False)


def status_action(status: int) -> Action:
    """WebSocket upgrade rejected with an HTTP status."""
    if status == 401:
        return Action("off", "key rejected (401)")
    if status in (403, 404):
        return Action("fallback", f"realtime access denied ({status})")
    if status == 429:
        return Action("backoff", "rate limited (429)", 2.0)
    if status >= 500:
        return Action("backoff", f"upstream error ({status})", 0.5)
    return Action("off", f"upgrade rejected ({status})")


class Backoff:
    """D3 schedule: base, doubling, ±20 % jitter. ``next`` returns None once a limited action is exhausted."""

    def __init__(self, rng: random.Random | None = None):
        self._rng = rng or random.Random()
        self.tries = 0

    def reset(self) -> None:
        self.tries = 0

    def next(self, action: Action) -> float | None:
        if action.limited and self.tries >= MAX_TRIES:
            return None
        delay = action.base_s * (2 ** self.tries)
        if not action.limited:
            delay = min(delay, NETWORK_MAX_DELAY_S)
        self.tries += 1
        return delay * self._rng.uniform(0.8, 1.2)
