"""Exceptions of the model layer.

``ResultError`` belongs to the pure side (a model reply that does not fit its schema). The others are raised by the
live transport (``ModelClient``) and stop a run: they mean every further call would fail or must not be made.
"""

from __future__ import annotations


class ResultError(ValueError):
    """A model reply that cannot be turned into its role's result (``parse_response`` status ``invalid``)."""


class ModelError(RuntimeError):
    """Base class for transport-side errors that stop the caller."""


class ModelConfigError(ModelError):
    """Missing key or base URL for the backend a request needs."""


class ProvenanceError(ModelError):
    """A call that would put a non-Meta model into a Meta-only run, or a stand-in without LW_DEV_ALLOW_NON_META=1."""


class BudgetExceeded(ModelError):
    """The persisted Meta spend plus this call's estimate would pass LW_BUDGET_USD (D21)."""


class ModelAccessError(ModelError):
    """401/402/403/404 from the backend. The call was recorded; ``result`` is its C1 ``model_result`` payload."""

    def __init__(self, message: str, result: dict):
        super().__init__(message)
        self.result = result
