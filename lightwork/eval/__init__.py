"""Collection, verified offline metrics and immutable model evaluations."""

from .evidence import EvidenceError, load_runs
from .metrics import compare_saved

__all__ = ['EvidenceError', 'load_runs', 'compare_saved']
