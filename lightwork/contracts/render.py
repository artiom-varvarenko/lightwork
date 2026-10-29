"""C2 Render protocol over ``/ws/render``.

All primitive coordinates are camera px (D5); each page maps them with ``H_cam_to_page`` from the ``calib`` message.
``scene`` is always a full snapshot; the latest wins and the server keeps at most 2 unacknowledged.
Projector renderers map ``role`` to saturated primaries on pure black, lines >= 4 projector px, no fills, text >= 28
projector px and never inside the work area.
"""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

Role = Literal["target", "source", "error", "warn", "done", "info"]
XY = Annotated[list[float], Field(min_length=2, max_length=2)]


class _M(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


# ---------------------------------------------------------------- primitives


class Ring(_M):
    kind: Literal["ring"] = "ring"
    id: str
    c: XY
    r: float
    role: Role
    pulse: bool = False


class Arrow(_M):
    kind: Literal["arrow"] = "arrow"
    id: str
    a: XY
    b: XY
    role: Role


class Poly(_M):
    kind: Literal["poly"] = "poly"
    id: str
    pts: list[XY]
    closed: bool
    role: Role


class Dot(_M):
    kind: Literal["dot"] = "dot"
    id: str
    c: XY
    role: Role


class Label(_M):
    kind: Literal["label"] = "label"
    id: str
    at: XY
    text: str
    role: Role


Prim = Annotated[Ring | Arrow | Poly | Dot | Label, Field(discriminator="kind")]


# ---------------------------------------------------------------- page -> server


class Hello(_M):
    type: Literal["hello"] = "hello"
    role: Literal["projector", "screen", "viewer"]
    innerWidth: int
    innerHeight: int
    dpr: float
    screen: dict[str, float | int | str]  # {width, height, availWidth, availHeight, ...}
    orientation: str
    fullscreen: bool
    secure: bool
    wakelock: bool
    ua: str


class MarkerRect(_M):
    id: int
    x: float
    y: float
    size: float


class Ack(_M):
    type: Literal["ack"] = "ack"
    seq: int
    drawn: list[MarkerRect] | None = None  # pattern mode only: marker rects as drawn, page CSS px


class Tts(_M):
    type: Literal["tts"] = "tts"
    id: str
    state: Literal["start", "end", "error"]


# ---------------------------------------------------------------- server -> page


class Calib(_M):
    type: Literal["calib"] = "calib"
    H_cam_to_page: list[list[float]]  # 3x3
    page_size: XY  # CSS px [w, h]


class Pattern(_M):
    aruco_dict: Literal["DICT_4X4_50"] = "DICT_4X4_50"  # PLAN.md C2 "dict"; renamed: `dict` shadows BaseModel.dict
    markers: list[MarkerRect]  # page CSS px, as requested


SceneMode = Literal["normal", "pattern", "dark", "grey", "white"]


class SceneMsg(_M):
    type: Literal["scene"] = "scene"
    seq: int
    mode: SceneMode
    prims: list[Prim]
    pattern: Pattern | None = None
    banner: str | None = None


class SayMsg(_M):
    type: Literal["say"] = "say"
    id: str
    text: str
    priority: Literal["normal", "interrupt"] = "normal"


PageToServer = Annotated[Hello | Ack | Tts, Field(discriminator="type")]
ServerToPage = Annotated[Calib | SceneMsg | SayMsg, Field(discriminator="type")]
