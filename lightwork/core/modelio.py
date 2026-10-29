"""The pure side of the model layer the reducer depends on (``lightwork.models``, workstream F).

Requests are derived (the reducer builds them and fingerprints them); responses are inputs that only
``models.parse.parse_response`` turns into results (D2). :class:`ModelFns` bundles those functions so unit tests can
substitute doubles; the reducer uses :func:`default_model_fns` otherwise.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ModelFns:
    build_request: Callable[..., tuple[str, dict[str, Any]]]  # (role, *, model, backend, frame_sha, ctx)
    parse_response: Callable[[str, bytes, str, str], dict[str, Any]]  # (role, raw, kind, endpoint)
    cost_usd: Callable[[str, dict[str, Any] | None], float]  # (model, usage)
    prompt_versions: Mapping[str, str]


def default_model_fns() -> ModelFns:
    from ..models.parse import parse_response
    from ..models.pricing import cost_usd
    from ..models.requests import PROMPT_VERSIONS, build_request

    return ModelFns(build_request=build_request, parse_response=parse_response, cost_usd=cost_usd,
                    prompt_versions=PROMPT_VERSIONS)
