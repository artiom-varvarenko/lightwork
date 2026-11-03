"""lightwork play session calibration: the projected border watch and the hand check's cursor correction."""

from __future__ import annotations

from itertools import pairwise

import cv2
import numpy as np

from lightwork.commands.play import Play
from lightwork.geometry import apply_h
from lightwork.playground import Geo
from lightwork.sessioncal import (
    FRAME_CONFIRM,
    BorderWatch,
    evaluate,
    frame_corners_page,
    frame_inset,
)

PAGE_W, PAGE_H = 749.0, 421.0
# page -> camera px: a wall page about 1256 px wide in a 1920x1080 frame, slightly keystoned
HI = np.array([[1.65, 0.06, 500.0], [0.02, 1.62, 200.0], [0.0, 0.00004, 1.0]])
GEO = Geo(np.linalg.inv(HI), PAGE_W, PAGE_H)
STROKE = 4.4  # page px, LINE_PROJ_PX at this page size


def _line(img, pts_page, value: int) -> None:
    """A projected stroke along page-px points, sampled densely so it follows the keystone."""
    pts = np.asarray(pts_page, dtype=np.float64)
    dense = np.concatenate([np.linspace(a, b, 60) for a, b in pairwise(pts)])
    cam = np.round(apply_h(HI, dense)).astype(np.int32)
    cv2.polylines(img, [cam], False, (value, value, value), round(STROKE * 1.63), cv2.LINE_AA)


def dashboard_frame(dx: float = 0.0) -> np.ndarray:
    """The border, the drop zone 13.8 px inside its right side drawn brighter than it, and a title under the top line."""
    img = np.full((1080, 1920, 3), 70, np.uint8)
    c = frame_corners_page(PAGE_W, PAGE_H)
    _line(img, [*c, c[0]], 170)
    zx0, zy0, zx1, zy1 = GEO.d2p([(440.0, 52.0), (732.0, 330.0)]).ravel()
    _line(img, [(zx0, zy0), (zx1, zy0), (zx1, zy1), (zx0, zy1), (zx0, zy0)], 215)
    at = apply_h(HI, [(18.0, 24.75 + 8.6)])[0]  # label: left edge, vertical middle; cap tops ~12 px under the line
    cv2.putText(img, "LIGHTWORK 14:22:45", (round(at[0]), round(at[1])), cv2.FONT_HERSHEY_SIMPLEX, 1.25, (215,) * 3, 4)
    img = cv2.GaussianBlur(img, (7, 7), 1.5)
    return cv2.warpAffine(img, np.float32([[1, 0, dx], [0, 1, 0]]), (1920, 1080)) if dx else img


def test_zone_outline_inside_the_reach_is_not_taken_for_the_border():
    assert abs(PAGE_W - frame_inset(PAGE_W, PAGE_H) - GEO.d2p([(732.0, 0.0)])[0][0] - 13.8) < 0.2
    r = BorderWatch(GEO).read(dashboard_frame())
    assert r.n_seen == 4 and all(abs(o) < 1.0 for o in r.sides), r


def test_still_dashboard_never_reports_a_move_and_a_bump_does():
    w = BorderWatch(GEO)
    frame = dashboard_frame()
    assert [w.update(frame) for _ in range(10)] == [None] * 10
    assert w.baseline is not None
    bumped = dashboard_frame(dx=6.0)  # ~3.6 page px to the right
    events = [w.update(bumped) for _ in range(FRAME_CONFIRM)]
    assert events[-1] == "moved"


class _Stub:
    def __init__(self, **kw):
        self.__dict__.update(kw)


def _final_failure(errors):
    play = Play.__new__(Play)
    play.args = _Stub(quiet=True)
    play.check = _Stub(attempt=2)
    play.dash = _Stub(offset=(0.0, 0.0))
    play.zone = None
    play.border = None
    play.say = ""
    res = evaluate(errors, "touch")
    assert res.status == "fail"
    play.finish_check(res, 100.0)
    return play.dash.offset, res


def test_scattered_failed_check_leaves_the_cursor_uncorrected():
    offset, res = _final_failure([(-1.3, -8.5), (17.3, -42.9), (-37.0, 2.7)])  # 1 Nov, wall: scatter 30 px
    assert res.residual > 15 and offset == (0.0, 0.0)


def test_consistent_failed_check_still_corrects_the_cursor():
    offset, res = _final_failure([(50.0, 10.0), (52.0, 12.0), (48.0, 9.0)])  # agrees, but past PASS_OFFSET
    assert res.residual < 5 and offset == res.offset == (50.0, 10.3)


def _pending_play(kind: str, flash_ok):
    """A Play with a mapping and a pending recalibration; ``flash_ok()`` decides whether the next flash solves."""
    from lightwork.vision.calib.solve import CalibrationError

    play = Play.__new__(Play)
    play.args = _Stub(quiet=True)
    play.render = _Stub(hello=lambda role: None, say=lambda text: None)
    play.page = _Stub(innerWidth=749, innerHeight=421)
    play.check, play.pending, play.pending_since, play.say = None, kind, 0.0, ""
    play.calib_failures, play.calib_t = 0, 0.0
    play.checks = []

    def calibrate(page, *, gain):
        if not flash_ok():
            raise CalibrationError("found 5 of 15 pattern markers, need >= 6")

    play.calibrate = calibrate
    play.start_check = lambda attempt=1: play.checks.append(attempt)
    return play


def test_failed_recalibration_mid_session_keeps_the_mapping_and_retries():
    play = _pending_play("retry", lambda: False)
    for n in range(1, 3):  # CALIB_RETRIES: tried again, the session goes on
        play._run_pending()
        assert play.pending == "retry" and play.calib_failures == n
    play._run_pending()  # then the mapping is kept and nothing is pending
    assert play.pending is None and play.calib_failures == 0 and play.checks == []


def test_recalibration_that_succeeds_on_retry_runs_the_hand_check():
    results = iter([False, True])
    play = _pending_play("retry", lambda: next(results))
    play._run_pending()
    play._run_pending()
    assert play.pending is None and play.calib_failures == 0 and play.checks == [2]
