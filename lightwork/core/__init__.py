"""Deterministic procedure engine. Feed recorded inputs; emit derived events without network or wall-clock I/O.

See :mod:`lightwork.core.reducer` for lifecycle and cost rules, :mod:`lightwork.core.readings` for colour references,
and :mod:`lightwork.core.scenarios` for reproducible synthetic runs (never evidence of real Meta inference).
"""

from .reducer import Reducer, ReducerError
from .store import MemoryStore, RunStore

__all__ = ["Reducer", "ReducerError", "RunStore", "MemoryStore"]
