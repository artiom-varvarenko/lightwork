"""Canonical JSON, hashing and quantization shared by every contract.

Canonical JSON: sorted keys, UTF-8, no insignificant whitespace, no NaN/Infinity, ``-0.0`` written as ``0.0``.
Floats are written with Python's shortest round-trip repr, so the same double always yields the same bytes.
Producers quantize numeric values (``q_px``, ``q_od``, ...) before they reach the log; canonicalization does not
quantize on its own because the quantum depends on the field.
"""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any

from pydantic import BaseModel

# Quanta (PLAN.md §4.2): coordinates 0.1 px, OD 1e-4, Lab 0.01, seconds 1e-3.
PX_DIGITS = 1
OD_DIGITS = 4
LAB_DIGITS = 2
SEC_DIGITS = 3


def _q(x: float, digits: int) -> float:
    v = round(float(x), digits)
    if not math.isfinite(v):
        raise ValueError(f"non-finite value {x!r}")
    return 0.0 if v == 0 else v


def q_px(x: float) -> float:
    return _q(x, PX_DIGITS)


def q_od(x: float) -> float:
    return _q(x, OD_DIGITS)


def q_lab(x: float) -> float:
    return _q(x, LAB_DIGITS)


def q_s(x: float) -> float:
    return _q(x, SEC_DIGITS)


def q_xy(xy: tuple[float, float] | list[float]) -> list[float]:
    return [q_px(xy[0]), q_px(xy[1])]


def _normalize(obj: Any) -> Any:
    if isinstance(obj, BaseModel):
        return _normalize(obj.model_dump(mode="json"))
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            if not isinstance(k, str):
                raise TypeError(f"canonical JSON keys must be str, got {type(k).__name__}")
            out[k] = _normalize(v)
        return out
    if isinstance(obj, (list, tuple)):
        return [_normalize(v) for v in obj]
    if isinstance(obj, bool) or obj is None or isinstance(obj, (int, str)):
        return obj
    if isinstance(obj, float):
        if not math.isfinite(obj):
            raise ValueError(f"non-finite float {obj!r} in canonical JSON")
        return 0.0 if obj == 0 else obj
    # numpy scalars and similar
    if hasattr(obj, "item"):
        return _normalize(obj.item())
    raise TypeError(f"not JSON-serializable: {type(obj).__name__}")


def canonical(obj: Any) -> str:
    """Canonical JSON text for a dict/list/scalar or a pydantic model."""
    return json.dumps(_normalize(obj), sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def canonical_bytes(obj: Any) -> bytes:
    return canonical(obj).encode("utf-8")


def sha256_hex(data: bytes | str) -> str:
    if isinstance(data, str):
        data = data.encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def canonical_sha(obj: Any) -> str:
    return sha256_hex(canonical_bytes(obj))


def file_sha(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()
