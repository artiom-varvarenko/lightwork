from __future__ import annotations

import numpy as np
import pytest

from lightwork.contracts.layout import Calibration, CameraInfo, MmScale, PageInfo, Residuals
from lightwork.geometry import apply_h
from lightwork.vision.calib.sim import SimCamera, SimLink, SimPage, SimRig


@pytest.fixture(autouse=True)
def offline(no_network):
    pass


@pytest.fixture
def make_rig():
    def make(*, dpr=1.0, **kw):
        rig = SimRig(**kw)
        page = SimPage(rig, dpr=dpr)
        return rig, SimCamera(rig), page, SimLink(page)

    return make


@pytest.fixture
def truth_calib():
    def make(rig, page, *, mm=None):
        return Calibration(
            H_cam_to_page=rig.H_cam_to_page.tolist(),
            page=PageInfo(w=page.w, h=page.h, dpr=page.dpr),
            camera=CameraInfo(device_id="sim", mode="sim 1920x1080@30", controls={}),
            residuals=Residuals(rms_px=0.0, p95_px=0.0, max_px=0.0, n_markers=15, n_inliers=60),
            mm=mm or MmScale(source="plate", H_cam_to_mm=np.linalg.inv(rig.bench_to_cam).tolist()),
            measurement_mode="grey", created="2026-10-29T20:00:00Z",
        )

    return make


@pytest.fixture
def truth_error():
    def error(H, rig, w=854, h=480):
        pts = np.array([[x, y] for y in np.linspace(0, h, 7) for x in np.linspace(0, w, 11)])
        return np.linalg.norm(apply_h(np.asarray(H), apply_h(rig.H_page_to_cam, pts)) - pts, axis=1)

    return error
