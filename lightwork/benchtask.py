"""A bench check for ``lightwork play --table --task``: what a job needs on the table, ticked off as it is put down.

``--task`` takes a JSON file ``{"title": "...", "items": ["96-well plate", ...]}``, a text file (first line the
title, then one item per line), or an inline ``"Serial dilution: plate, pipette, tips"``. Muse Spark names each
object and says which listed item it is, in the same call (:mod:`lightwork.namer`); :func:`match_item` is the fallback
when the namer cannot (a local model): shared words between the object's name and an item.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

MAX_ITEMS = 12
_STOP = {"a", "an", "the", "of", "for", "and", "with", "small", "large", "big", "empty", "full", "set", "pair"}


@dataclass(frozen=True)
class BenchTask:
    title: str
    items: tuple[str, ...]

    def missing(self, present: set[int]) -> list[str]:
        """Item names not yet present (``present``: 1-based item numbers)."""
        return [name for k, name in enumerate(self.items, 1) if k not in present]


def load_task(spec: str) -> BenchTask:
    path = Path(spec)
    if path.is_file():
        text = path.read_text(encoding="utf-8")
        if path.suffix.lower() == ".json":
            data = json.loads(text)
            title, items = str(data.get("title") or path.stem), [str(i) for i in data.get("items") or []]
        else:
            lines = [ln.strip() for ln in text.splitlines() if ln.strip() and not ln.lstrip().startswith("#")]
            title, items = (lines[0], lines[1:]) if lines else (path.stem, [])
    else:
        title, colon, rest = spec.partition(":")
        if not colon:  # no title given: the whole text is the list
            title, rest = "Bench check", spec
        items = [i.strip() for i in re.split(r"[,;]", rest)]
    items = [i for i in (re.sub(r"\s+", " ", i).strip(" -*.") for i in items) if i][:MAX_ITEMS]
    if not items:
        raise ValueError(f"task {spec!r} lists no items")
    return BenchTask(title.strip().rstrip(":") or "Bench check", tuple(items))


def _words(text: str) -> set[str]:
    words = set(re.findall(r"[a-z0-9]+", text.lower()))
    return {w[:-1] if w.endswith("s") and len(w) > 3 else w for w in words} - _STOP


def match_item(name: str | None, items: tuple[str, ...]) -> int:
    """1-based number of the item ``name`` shares the most words with, or 0."""
    if not name:
        return 0
    have = _words(name)
    best, score = 0, 0
    for k, item in enumerate(items, 1):
        s = len(have & _words(item))
        if s > score:
            best, score = k, s
    return best


def spoken_list(names: list[str]) -> str:
    if not names:
        return ""
    if len(names) == 1:
        return names[0]
    return ", ".join(names[:-1]) + " and " + names[-1]
