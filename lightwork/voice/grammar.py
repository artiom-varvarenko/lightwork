"""Voice command grammar (PLAN.md §6.3, voice-chess pattern).

Exact full match after normalisation; nothing is repaired or lifted out of a sentence, so "I'm done with that" is
rejected. Normalisation: trim, lowercase, strip one trailing ``.``, ``!`` or ``?``.
"""

from __future__ import annotations

from ..contracts.stepgraph import StepGraph

COMMANDS: tuple[str, ...] = ("next", "done", "repeat", "why", "back")
MAX_NOUNS = 20


def normalise(transcript: str) -> str:
    text = transcript.strip().lower()
    if text and text[-1] in ".!?":
        text = text[:-1]
    return text


def parse_command(transcript: str) -> str | None:
    text = normalise(transcript)
    return text if text in COMMANDS else None


def build_keywords(nouns: list[str]) -> list[str]:
    """Commands first, then at most ``MAX_NOUNS`` distinct nouns (case-insensitive dedup, order kept)."""
    out = list(COMMANDS)
    seen = set(COMMANDS)
    extra = 0
    for noun in nouns:
        key = noun.strip().lower()
        if not key or key in seen:
            continue
        if extra == MAX_NOUNS:
            break
        seen.add(key)
        out.append(noun.strip())
        extra += 1
    return out


def stepgraph_nouns(graph: StepGraph) -> list[str]:
    """Voice nouns of a procedure: its explicit ``vocabulary`` first, then object nouns."""
    return [*graph.vocabulary, *(o.noun for o in graph.objects)]
