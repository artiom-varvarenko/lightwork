"""Runs (C6): the live recorder, run-directory access, offline ``verify``, the provenance ``audit`` and the replay
engine. Conventions beyond C6 are documented where they are produced: ``preview/<seq>.jpg`` in
:meth:`Recorder.put_preview`, ``manifest.totals`` in :func:`manifest.run_totals`, diff classes in :mod:`.verify`.
"""

from .audit import audit_run
from .recorder import Recorder
from .replay import ReplayEngine
from .store import DirStore, find_runs
from .verify import verify_run

__all__ = ["DirStore", "Recorder", "ReplayEngine", "audit_run", "find_runs", "verify_run"]
