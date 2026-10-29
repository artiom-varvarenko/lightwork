"""Spoken "why" explanation (Spark or Glimmer), text only. Plain-text reply, no schema."""

from __future__ import annotations

from lightwork.contracts.backend import WhyCtx

VERSION = "why-v1"
MAX_CHARS = 300

INSTRUCTIONS = """\
You help a person doing a hands-on procedure at a workbench. Answer in at most 40 words of plain spoken English: \
why the current step matters or, if a problem is given, what went wrong and how to fix it. No lists, no markdown, \
no preamble."""


def user_text(ctx: WhyCtx) -> str:
    return f"Step: {ctx.instruction}\nDetail: {ctx.detail or 'none'}\nProblem: {ctx.last_reason or 'none'}"
