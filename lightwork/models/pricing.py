"""Meta Model API prices (context/meta-docs/pricing-rate-limits.md, 29 Oct 2026) and pure cost functions (D21).

``cost_usd`` is pure and quantized to 1e-6 USD so the reducer's ``cost`` events re-derive byte-identically. Models
outside the table (Glimmer on own hardware, dev stand-ins) cost 0 here: they are not billed by Meta.
"""

from __future__ import annotations

from typing import Any

# USD per 1M tokens: (input, cached input, output). Cached tokens are a subset of input tokens.
_SPARK_STANDARD = (1.25, 0.15, 4.25)
_SPARK_CONTRIBUTOR = (0.10, 0.002, 0.20)
TOKEN_PRICES: dict[str, tuple[float, float, float]] = {
    "muse-wonder-1.0": (4.00, 0.20, 20.00),
    "muse-spark-1.3": _SPARK_STANDARD,
    "muse-spark-1.2": _SPARK_STANDARD,
    "muse-spark-1.1": _SPARK_STANDARD,
    "muse-spark-1.3-contributor": _SPARK_CONTRIBUTOR,
    "muse-spark-1.2-contributor": _SPARK_CONTRIBUTOR,
}
SAM_MODELS = frozenset({"sam-3.1"})
SAM_IMAGE_USD = 2.50 / 1000
SAM_VIDEO_FRAME_USD = 0.20 / 1000

# A hard cap must reserve a bound, not a pixels-to-tokens guess. Spark/Wonder have a 1,048,576-token
# context (models.md). Reserving unused input capacity is conservative; unused money is released on settlement.
CONTEXT_TOKENS = 1_048_576


def _q(usd: float) -> float:
    v = round(usd, 6)
    return 0.0 if v == 0 else v


def _int(v: Any) -> int:
    return v if isinstance(v, int) and not isinstance(v, bool) and v > 0 else 0


def cost_usd(model: str, usage: dict[str, Any] | None) -> float:
    """Billed cost of one call from its response ``usage`` (0 when there is no usage or the model is not billed)."""
    if not usage:
        return 0.0
    if model in SAM_MODELS:
        return _q(_int(usage.get("images_processed")) * SAM_IMAGE_USD
                  + _int(usage.get("video_frames_processed")) * SAM_VIDEO_FRAME_USD)
    prices = TOKEN_PRICES.get(model)
    if prices is None:
        return 0.0
    p_in, p_cached, p_out = prices
    details = usage.get("input_tokens_details")
    cached = _int(details.get("cached_tokens")) if isinstance(details, dict) else 0
    inp = _int(usage.get("input_tokens"))
    cached = min(cached, inp)
    out = _int(usage.get("output_tokens"))
    return _q(((inp - cached) * p_in + cached * p_cached + out * p_out) / 1e6)


def estimate_usd(model: str, request: dict[str, Any]) -> float:
    """Upper-bound cost of a canonical (media-redacted) request, for the budget reservation before sending it."""
    if model in SAM_MODELS:
        return SAM_IMAGE_USD
    prices = TOKEN_PRICES.get(model)
    if prices is None:
        return 0.0
    tokens_out = _int(request.get("max_output_tokens"))
    if not 16 <= tokens_out <= CONTEXT_TOKENS:
        raise ValueError("a billed text request needs an explicit valid max_output_tokens")
    tokens_in = CONTEXT_TOKENS - tokens_out
    return _q((tokens_in * prices[0] + tokens_out * prices[2]) / 1e6)


def aggregate_usage(usages: list[dict[str, Any] | None]) -> dict[str, Any] | None:
    """Sum billed usage counters across attempts, including nested cached/reasoning details.

    Absent usage stays absent, never estimated. Non-counter fields, if a provider adds them, retain the last
    supplied value. Feed attempts in order, followed by the final response; pure and shared with run verification.
    """
    out: dict[str, Any] | None = None
    for usage in usages:
        if usage is None:
            continue
        if out is None:
            out = {}
        for key, value in usage.items():
            prior = out.get(key)
            if isinstance(value, dict):
                out[key] = aggregate_usage([prior if isinstance(prior, dict) else None, value])
            elif isinstance(value, (int, float)) and not isinstance(value, bool):
                out[key] = (prior if isinstance(prior, (int, float)) and not isinstance(prior, bool) else 0) + value
            else:
                out[key] = value
    return out
