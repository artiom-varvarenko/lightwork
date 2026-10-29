"""JSON-schema builders for the strict structured-output subset (structured-output.md "Enforce strict mode")."""

from __future__ import annotations

from typing import Any


def obj(props: dict[str, Any]) -> dict[str, Any]:
    """Object schema with every property required and no extra keys."""
    return {"type": "object", "properties": props, "required": list(props), "additionalProperties": False}


def nullable(schema: dict[str, Any]) -> dict[str, Any]:
    if set(schema) == {"type"} and isinstance(schema["type"], str):
        return {"type": [schema["type"], "null"]}
    return {"anyOf": [schema, {"type": "null"}]}


STR = {"type": "string"}
INT = {"type": "integer"}
NUM = {"type": "number"}
BOOL = {"type": "boolean"}


def array(items: dict[str, Any]) -> dict[str, Any]:
    return {"type": "array", "items": items}


def enum(*values: str) -> dict[str, Any]:
    return {"type": "string", "enum": list(values)}
