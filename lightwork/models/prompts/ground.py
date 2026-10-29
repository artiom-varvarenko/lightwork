"""Grounding on Spark's perception grid (image-understanding.md "Localize objects with perception grounding").

Coordinates are on the 0-1000 grid of the image actually sent (the keyframe crop): (0, 0) top-left, x right, y down.
The user prompts are Meta's documented point/box/count prompts; Meta advises no system prompt for grounding. The
reply is wrapped in ``{"items": [...]}`` because the strict structured-output subset needs an object at the root.
"""

from __future__ import annotations

import json

from .schema import INT, STR, array, obj

VERSION = "ground-v1"
SHAPES = ("point", "box", "count")

_XY = obj({"x": INT, "y": INT})
_BOX = obj({"x_min": INT, "y_min": INT, "x_max": INT, "y_max": INT})

SCHEMAS = {
    "point": ("GroundPoints", obj({"items": array(obj({"object_name": STR, "x": INT, "y": INT}))})),
    "box": ("GroundBoxes", obj({"items": array(obj({"object_name": STR, "bbox": array(_BOX)}))})),
    "count": ("GroundCounts", obj({"items": array(obj({"object_name": STR, "points": array(_XY), "count": INT}))})),
}

_TAIL = (
    'The coordinates should be in the 0-1000 range. Return a JSON object {{"items": [...]}} with these results. '
    "If you cannot find {what}, omit it from the results."
)


def user_text(nouns: list[str], shape: str) -> str:
    names = ", ".join(json.dumps(n, ensure_ascii=False) for n in nouns)
    if shape == "point":
        return (
            f"Point to each of the following objects in the image: {names}. For each object, answer in the format "
            '{"object_name": "<name>", "x": <int>, "y": <int>}. ' + _TAIL.format(what="an object")
        )
    if shape == "box":
        return (
            "You are an object grounding expert. Provide the bounding box coordinates of the objects in the image: "
            f"{names}. Ensure the objects accurately match the request and do not miss any objects. For each object, "
            'answer in the format {"object_name": "<name>", "bbox": [{"x_min": <int>, "y_min": <int>, "x_max": '
            '<int>, "y_max": <int>}]}. ' + _TAIL.format(what="an object")
        )
    if shape == "count":
        return (
            f"Point to every instance of each of the following objects in the image and count them: {names}. For "
            'each object, answer in the format {"object_name": "<name>", "points": [{"x": <int>, "y": <int>}, ...], '
            '"count": <int>}. ' + _TAIL.format(what="any instances of an object")
        )
    raise ValueError(f"unknown grounding shape {shape!r}")
