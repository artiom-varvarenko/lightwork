"""C2 Render protocol over ``/ws/render``.

All primitive coordinates are camera px (D5); each page maps them with ``H_cam_to_page`` from the ``calib`` message.
``scene`` is always a full snapshot; the latest wins and the server keeps at most 2 unacknowledged.
Projector renderers map ``role`` to saturated primaries on pure black, lines >= 4 projector px, no fills, text >= 28
projector px and never inside the work area. Effects change how a prim is drawn, never where: ``glow`` adds a faint
wide halo under the stroke; ``draw_ms`` / ``type_ms`` reveal an outline / label text when its id first appears (the
overlay mask covers them as fully drawn). Effects at their default are left out of dumps, so a scene without effects
serialises, and hashes (C1 ``scene``), exactly as before they existed.
"""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

Role = Literal["target", "source", "error", "warn", "done", "info"]
XY = Annotated[list[float], Field(min_length=2, max_length=2)]


class _M(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


def _unset(v: object) -> bool:
    return not v


# ---------------------------------------------------------------- primitives


class Ring(_M):
    kind: Literal["ring"] = "ring"
    id: str
    c: XY
    r: float
    role: Role
    pulse: bool = False
    glow: bool = Field(False, exclude_if=_unset)  # faint halo under the stroke (projector_style.GLOW_*)


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
    glow: bool = Field(False, exclude_if=_unset)
    draw_ms: int = Field(0, ge=0, le=5000, exclude_if=_unset)  # > 0: draws itself from pts[0] when its id appears


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
    type_ms: int = Field(0, ge=0, le=5000, exclude_if=_unset)  # > 0: typed out when its id appears or text changes


class Picture(_M):
    """A picture the page loads by URL from this server. Scenes go out up to 30 times a second, so they never carry its
    bytes. Black in it is no light on the table. Drawn under the other prims of its scene if it comes first."""

    kind: Literal["picture"] = "picture"
    id: str
    src: str = Field(pattern=r"^/pictures/[0-9a-f]{16,64}\.(png|jpg)$")
    corners: Annotated[list[XY], Field(min_length=4, max_length=4)]  # the picture's top-left, top-right, bottom-right, bottom-left
    draw_ms: int = Field(0, ge=0, le=5000, exclude_if=_unset)  # > 0: wipes on from its top edge when its id appears


Prim = Annotated[Ring | Arrow | Poly | Dot | Label | Picture, Field(discriminator="kind")]


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


class Sfx(_M):
    """A short synthesized UI sound on the speaking page (projector, else screen); pages without sound ignore it."""

    type: Literal["sfx"] = "sfx"
    name: Literal["scan", "found", "press", "gone", "error", "listen", "answer"]
    gain: float = Field(1.0, ge=0.0, le=1.0)


class View(_M):
    """How the projector page turns everything it draws, pattern included: ``rotate: 180`` reads right from the far side
    of a table. Page CSS px (``Calib.page_size``, ``Ack.drawn``) stay in the page's own, unturned frame, so a mapping
    solved from a turned pattern already contains the turn. Sent on connect and on change; screen pages ignore it."""

    type: Literal["view"] = "view"
    rotate: Literal[0, 180] = 0
    # The page eases prims that moved between two scenes and fades ones that came or went (hand games, 3 Nov: "make
    # the projector smoother"). Off whenever the table is scanned: a fading outline would be taken for an object.
    smooth: bool = Field(False, exclude_if=_unset)


PageToServer = Annotated[Hello | Ack | Tts, Field(discriminator="type")]
ServerToPage = Annotated[Calib | SceneMsg | SayMsg | Sfx | View, Field(discriminator="type")]
