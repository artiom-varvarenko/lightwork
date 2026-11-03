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
    mode: Literal["noun", "grounded"]  # selected SAM prompt; validated records whether layout tested it (D4)
    noun: str
    validated: bool = False  # True only when this prompt mode was tested and accepted during layout


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
    """``measurement_mode`` "normal" marks a screen-only calibration (``lightwork calibrate screen``): no projector
    was calibrated, so ``H_cam_to_page``, ``page`` and ``residuals`` are null and nothing may be projected with it.
    Grey/dark calibrations always carry the projector mapping they were solved with."""

    H_cam_to_page: H3 | None
    page: PageInfo | None
    camera: CameraInfo
    residuals: Residuals | None
    mm: MmScale
    flat_field_sha: str | None = None
    flat_field_crop_xywh: Annotated[list[int], Field(min_length=4, max_length=4)] | None = None
    measurement_mode: Literal["grey", "dark", "normal"]
    created: str
    H_plate_cam_to_page: H3 | None = None  # optional second homography at plate height (rig:C4 > 3 mm)

    @property
    def screen_only(self) -> bool:
        return self.measurement_mode == "normal"

    @model_validator(mode="after")
    def _projector_or_screen(self):
        mapping = (self.H_cam_to_page, self.page, self.residuals)
        if not self.screen_only:
            if any(v is None for v in mapping):
                raise ValueError("a grey/dark (projector) calibration needs H_cam_to_page, page and residuals")
            return self
        projector_only = (*mapping, self.H_plate_cam_to_page, self.flat_field_sha, self.flat_field_crop_xywh,
                          self.mm.H_cam_to_mm)
        if any(v is not None for v in projector_only) or self.mm.source != "nominal":
            raise ValueError("a screen-only calibration (measurement_mode normal) has no projector mapping, "
                             "residuals, plate homography, flat field or mm scale")
        return self
