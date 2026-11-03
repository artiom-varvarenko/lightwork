"""TiledHandTracker: tile layout, coordinates, search order and cost, with a fake landmarker per window.

The fake finds a "hand" (a bright square) only when the square is whole in its input and spans at least ``MIN_FRAC`` of
the input's long side, like MediaPipe's palm detector, which sees every input at 192 px.
"""

from __future__ import annotations

import numpy as np

from lightwork.vision.hands import HandObs, TiledHandTracker, tile_rects
from lightwork.vision.source import Frame

W, H = 1920, 1080
REGION = (374, 74, 1882, 982)  # a wall page seen from 2 m, plus margin


class FakeTracker:
    MIN_FRAC = 0.03

    def __init__(self, calls: list):
        self.calls = calls
        self.closed = False

    def process(self, frame: Frame) -> HandObs:
        self.calls.append(self)
        img = frame.image
        ys, xs = np.nonzero(img[..., 0] > 128)
        h, w = img.shape[:2]
        if xs.size == 0 or xs.min() == 0 or ys.min() == 0 or xs.max() == w - 1 or ys.max() == h - 1:
            return HandObs(present=False, tip_xy=None)
        if max(np.ptp(xs) + 1, np.ptp(ys) + 1) / max(w, h) < self.MIN_FRAC:
            return HandObs(present=False, tip_xy=None)
        tip = (float(xs.mean()), float(ys.mean()))
        return HandObs(present=True, tip_xy=tip, tips=(tip,))

    def close(self) -> None:
        self.closed = True


def frame_with_hand(cx: float, cy: float, size: int = 40, idx: int = 0) -> Frame:
    img = np.zeros((H, W, 3), np.uint8)
    h = size // 2
    img[round(cy) - h:round(cy) + h, round(cx) - h:round(cx) + h] = 255
    return Frame(img, 10.0 + idx / 25, idx)


def empty_frame(idx: int = 0) -> Frame:
    return Frame(np.zeros((H, W, 3), np.uint8), 10.0 + idx / 25, idx)


def tracker(region=REGION):
    calls: list = []
    return TiledHandTracker(region, factory=lambda: FakeTracker(calls)), calls


def test_tiles_are_square_ish_overlap_and_cover_the_region():
    tiles = tile_rects(REGION)
    assert len(tiles) == 2
    (a0, ay0, a1, ay1), (b0, by0, b1, by1) = tiles
    assert (a0, ay0, ay1, b1) == (374, 74, 982, 1882) and (by0, by1) == (74, 982)
    assert a1 - b0 >= 0.25 * (a1 - a0)  # a hand on the seam is whole in one tile
    assert 0.9 < (a1 - a0) / (ay1 - ay0) < 1.1
    assert tile_rects((0, 0, 1000, 1000)) == [(0, 0, 1000, 1000)]
    assert len(tile_rects((100, 0, 500, 1080))) == 3  # tall regions split vertically, at most max_tiles
    assert tile_rects((0, 0, 0, 10)) == []


def test_small_hand_is_found_by_a_tile_not_by_the_whole_frame():
    whole, _ = tracker(region=None)
    assert all(not whole.process(frame_with_hand(700, 500, idx=k)).present for k in range(6))
    tiled, _ = tracker()
    seen = [tiled.process(frame_with_hand(700, 500, idx=k)) for k in range(3)]  # whole frame, tile 1, tile 2 in turn
    hit = next(o for o in seen if o.present)
    assert np.allclose(hit.tip_xy, (699.5, 499.5)) and hit.tips == (hit.tip_xy,)  # tile px mapped back to camera px


def test_one_mediapipe_call_per_frame_when_idle_or_tracking():
    t, calls = tracker()
    for k in range(9):
        before = len(calls)
        assert not t.process(empty_frame(k)).present
        assert len(calls) - before == 1
    for k in range(9, 30):
        before = len(calls)
        t.process(frame_with_hand(700, 500, idx=k))
        if k >= 12:  # found within one round of the three windows
            assert t.active == 1 and len(calls) - before == 1


def test_hand_crossing_into_the_next_tile_is_handed_over_in_the_same_frame():
    t, _ = tracker()
    for k in range(3):
        t.process(frame_with_hand(700, 500, idx=k))
    assert t.active == 1
    obs = t.process(frame_with_hand(1500, 500, idx=3))  # outside tile 1, inside tile 2
    assert obs.present and t.active == 2 and np.allclose(obs.tip_xy, (1499.5, 499.5))


def test_large_hand_tracked_by_the_whole_frame_moves_to_the_tile_under_it():
    t, _ = tracker()
    first = t.process(frame_with_hand(700, 500, size=100, idx=0))  # large enough for the whole-frame window
    assert first.present and t.active == 0
    for k in range(1, TiledHandTracker.PROBE_EVERY + 1):
        obs = t.process(frame_with_hand(700, 500, size=100, idx=k))
        assert obs.present
    assert t.active == 1


def test_set_region_reuses_landmarkers_and_closes_extras():
    t, _ = tracker()
    trackers = list(t._trackers)
    assert len(trackers) == 3
    t.set_region((374, 76, 1882, 984))  # a 2 px drift: same tiles count, same landmarkers
    assert t._trackers == trackers
    assert t.set_region(None) == [] and t._trackers == trackers[:1]
    assert trackers[1].closed and trackers[2].closed and not trackers[0].closed
    t.close()
    assert trackers[0].closed
