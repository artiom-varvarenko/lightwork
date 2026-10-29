from __future__ import annotations

import re

import cv2

from lightwork.server import projector_style as ps
from lightwork.server.app import WEB_DIR


def _const(js: str, name: str) -> float:
    m = re.search(rf"^const {name} = ([0-9.]+);", js, re.M)
    assert m, name
    return float(m.group(1))


def test_python_mirror_matches_the_page():
    scene = (WEB_DIR / "js" / "scene.js").read_text()
    proj = (WEB_DIR / "js" / "projector.js").read_text()
    for name in ("RING_SEGMENTS", "PULSE_HZ", "PULSE_GAIN", "DOT_RADIUS_LINES", "ARROW_HEAD_LINES", "ARROW_HEAD_RAD"):
        assert _const(scene, name) == getattr(ps, name), name
    for name in ("PROJ_W", "PROJ_H", "LINE_PROJ_PX", "TEXT_PROJ_PX"):
        assert _const(proj, name) == getattr(ps, name), name
    assert ps.unit_css_px(915, 412) == 915 / 854
    assert ps.max_stroke_css_px(854, 480) == 8.0


def test_embedded_aruco_codes_match_opencv():
    js = (WEB_DIR / "js" / "aruco.js").read_text()
    codes = [int(h, 16) for h in re.findall(r"0x([0-9a-f]{4})", js.split("const CODES")[1].split("];")[0])]
    d = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    expected = []
    for i in range(15):
        img = cv2.aruco.generateImageMarker(d, i, 6)
        bits = 0
        for v in img[1:5, 1:5].ravel():
            bits = (bits << 1) | int(v > 127)
        expected.append(bits)
    assert codes == expected
