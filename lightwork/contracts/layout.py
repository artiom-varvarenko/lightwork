"""C4 Layout (per session, copied into each run) and C7 Calibration (``calib.json``)."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

XY = Annotated[list[float], Field(min_length=2, max_length=2)]
H3 = Annotated[list[Annotated[list[float], Field(min_length=3, max_length=3)]], Field(min_length=3, max_length=3)]


class _M(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


# ---------------------------------------------------------------- C4


class SamPrompt(_M):
    mode: Literal["noun", "grounded"]  # which SAM prompt kind worked for this object during layout (D4)
    noun: str


class LayoutObject(_M):
    kind: Literal["item", "container", "tool", "grid", "zone"]
    corners_cam_px: Annotated[list[XY], Field(min_length=4, max_length=4)] | None = None  # TL, TR, BR, BL
    polygon_cam_px: list[XY] | None = None
    sam: SamPrompt

    @model_validator(mode="after")
    def _one_shape(self):
        if (self.corners_cam_px is None) == (self.polygon_cam_px is None):
            raise ValueError("exactly one of corners_cam_px / polygon_cam_px")
        if self.kind == "grid" and self.corners_cam_px is None:
            raise ValueError("grid objects need corners_cam_px")
        return self


class Layout(_M):
    pack: str
    created: str  # ISO-8601 UTC
    work_area_xywh: Annotated[list[int], Field(min_length=4, max_length=4)]
    objects: dict[str, LayoutObject]


# ---------------------------------------------------------------- C7


class PageInfo(_M):
    w: float
    h: float
    dpr: float


class CameraInfo(_M):
    device_id: str
    mode: str  # e.g. "MJPG 1920x1080@30"
    controls: dict[str, int]


class Residuals(_M):
    rms_px: float
    p95_px: float
    max_px: float
    n_markers: int
    n_inliers: int


class MmScale(_M):
    source: Literal["card", "plate", "idcard", "nominal"]
    H_cam_to_mm: H3 | None = None


class Calibration(_M):
    H_cam_to_page: H3
    page: PageInfo
    camera: CameraInfo
    residuals: Residuals
    mm: MmScale
    flat_field_sha: str | None = None
    measurement_mode: Literal["grey", "dark"]
    created: str
    H_plate_cam_to_page: H3 | None = None  # optional second homography at plate height (rig:C4 > 3 mm)
