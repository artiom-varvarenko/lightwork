"""C9 Backend capabilities (Meta backend and chat backend).

Grounding coordinates are on the 0-1000 grid of the *sent crop*, clamped. SAM masks are in camera px with half-open
boxes ``[x0, y0, x1, y1)``. Prompts and schemas live in ``lightwork/models/prompts/`` with a ``prompt_version``.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field

from .stepgraph import StepGraph


class _M(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class Verdict(_M):
    done: Literal["yes", "no", "unsure"]
    contradiction: Annotated[str, Field(max_length=120)] | None
    evidence: Annotated[str, Field(max_length=160)]


class GroundPoint(_M):
    name: str
    x: Annotated[int, Field(ge=0, le=1000)]
    y: Annotated[int, Field(ge=0, le=1000)]


class GroundBox(_M):
    name: str
    bbox: Annotated[list[Annotated[int, Field(ge=0, le=1000)]], Field(min_length=4, max_length=4)]  # x0,y0,x1,y1


class Mask(_M):
    object_id: int
    box: Annotated[list[int], Field(min_length=4, max_length=4)]  # half-open, camera px
    area_px: int
    centroid: Annotated[list[float], Field(min_length=2, max_length=2)]
    confidence: float | None = None
    rle: dict[str, Any]  # COCO RLE {size:[h,w], counts}, column-major, camera-px crop frame


class VerdictCtx(_M):
    step_id: str
    instruction: str
    question: str
    expected: str = ""


class WhyCtx(_M):
    step_id: str
    instruction: str
    detail: str
    last_reason: str = ""


GroundShape = Literal["point", "box", "count"]


@runtime_checkable
class Backend(Protocol):
    """Pure-request side of a backend. Transport (live/replay) is separate (D2)."""

    name: Literal["meta", "chat"]

    def verdict(self, frame_jpeg: bytes, ctx: VerdictCtx) -> Verdict: ...

    def ground(self, frame_jpeg: bytes, nouns: list[str], shape: GroundShape) -> list[GroundPoint | GroundBox]: ...

    def author(self, source: str) -> StepGraph: ...

    def why(self, ctx: WhyCtx) -> str: ...


@runtime_checkable
class Segmenter(Protocol):
    """Meta backend only."""

    def segment(self, frame_jpeg: bytes, concept: str | None = None, object_prompts: list | None = None) -> list[Mask]: ...
