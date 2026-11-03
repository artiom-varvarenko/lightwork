"""CropHandTracker: hands too small for a whole-page search are found in full-resolution tiles, then each is followed
in its own crop; fake landmarkers stand in for MediaPipe."""

from __future__ import annotations

import cv2
import numpy as np

from lightwork.vision.hands import CropHandTracker, HandPose, grid_tiles
from lightwork.vision.source import Frame

W, H = 1920, 1080
REGION = (240, 90, 1895, 1012)  # the 2 Nov table page plus margin: 1655 x 922 camera px


class Fake:
    """Finds each bright square that is whole in its input and spans at least ``MIN_FRAC`` of the input's long side,
    like MediaPipe's palm detector, which sees every input at 192 px. A found square is a hand: wrist below its centre,
    middle knuckle above it (palm length 0.6 of the side), index fingertip at the top."""

    MIN_FRAC = 0.15

    def __init__(self, log: list, kind: str):
        self.log, self.kind = log, kind

    def poses(self, frame: Frame) -> list[HandPose]:
        img = frame.image
        h, w = img.shape[:2]
        self.log.append((self.kind, (w, h)))
        n, _, stats, cents = cv2.connectedComponentsWithStats((img[..., 0] > 128).astype(np.uint8))
        out = []
        for i in range(1, n):
            x, y, bw, bh = (int(v) for v in stats[i, :4])
            if x == 0 or y == 0 or x + bw == w or y + bh == h or max(bw, bh) < self.MIN_FRAC * max(w, h):
                continue
            cx, cy = cents[i]
            lm = np.tile([cx, cy], (21, 1)).astype(np.float64)
            lm[0] = (cx, cy + 0.3 * bh)
            lm[9] = (cx, cy - 0.3 * bh)
            lm[8] = (cx + 0.1 * bw, cy - 0.45 * bh)
            lm[4] = (cx - 0.45 * bw, cy)
            out.append(HandPose(lm, np.zeros((21, 3)), "Left", 0.9))
        return out

    def close(self) -> None:
        pass


def tracker(**kw) -> tuple[CropHandTracker, list]:
    log: list = []
    t = CropHandTracker(REGION, factory=lambda: Fake(log, "follow"), search_factory=lambda: Fake(log, "search"), **kw)
    return t, log


def frame(*hands: tuple[float, float], side: int = 120, idx: int = 0) -> Frame:
    img = np.zeros((H, W, 3), np.uint8)
    for cx, cy in hands:
        img[round(cy) - side // 2:round(cy) + side // 2, round(cx) - side // 2:round(cx) + side // 2] = 255
    return Frame(img, 10.0 + idx / 25, idx)


def test_tiles_cover_the_page_in_a_grid_and_overlap():
    tiles = grid_tiles(REGION)
    assert len(tiles) == 8  # 4 across, 2 down
    assert all(500 <= x1 - x0 <= 560 and 500 <= y1 - y0 <= 560 for x0, y0, x1, y1 in tiles)
    covered = np.zeros((H, W), bool)
    for x0, y0, x1, y1 in tiles:
        covered[y0:y1, x0:x1] = True
    rx0, ry0, rx1, ry1 = REGION
    assert covered[ry0:ry1, rx0:rx1].all()
    (a, b) = tiles[:2]
    assert a[2] - b[0] >= 0.25 * (a[2] - a[0])  # neighbours share a palm's width and more
    assert grid_tiles((0, 0, 300, 200)) == [(0, 0, 300, 200)]


def test_a_palm_too_small_for_the_page_is_found_in_a_tile_then_followed_in_its_crop():
    t, log = tracker()
    assert Fake.MIN_FRAC * (REGION[2] - REGION[0]) > 120  # the whole page would never show this hand
    found_at = None
    for i in range(8):
        poses = t.poses(frame((1300, 600), idx=i))
        if poses:
            found_at = i
            break
    assert found_at is not None and found_at <= 3  # 8 tiles, 2 a frame
    (p,) = poses
    assert np.allclose(p.img[0], (1299.5, 635.5), atol=1.0)  # camera px, not tile px
    log.clear()
    for i in range(found_at + 1, found_at + 6):
        (p,) = t.poses(frame((1300, 600), idx=i))
    follows = [size for kind, size in log if kind == "follow"]
    assert len(follows) == 5 and all(w == h and w < 400 for w, h in follows)  # a crop around the hand, every frame
    assert sum(kind == "search" for kind, _ in log) == 5  # one tile a frame keeps looking for the second hand


def test_a_fast_hand_stays_in_its_crop():
    t, _ = tracker()
    i = 0
    while not t.poses(frame((500, 500), idx=i)):
        i += 1
    seen = []
    for k in range(1, 20):  # 60 px a frame: 1.5 m/s on the 2 Nov table, up to the right edge of the page
        x = 500 + 60 * k
        poses = t.poses(frame((x, 500), idx=i + k))
        seen.append(bool(poses) and abs(poses[0].img[0][0] - x) < 2)
    assert all(seen)
    assert t.following == 1


def test_two_hands_get_two_followers_and_a_lost_hand_is_searched_for_where_it_left():
    t, log = tracker()
    i = 0
    while len(t.poses(frame((500, 300), (1500, 800), idx=i))) < 2:
        i += 1
        assert i < 10
    assert t.following == 2
    for _ in range(4):  # the right hand leaves
        i += 1
        t.poses(frame((500, 300), idx=i))
    assert t.following == 1
    log.clear()
    i += 1
    poses = t.poses(frame((500, 300), (1480, 790), idx=i))  # ... and comes back near where it left
    first_search = next(size for kind, size in log if kind == "search")
    assert len(poses) == 2 and first_search[0] < 600  # found again at once, in the tile it left from


def test_two_followers_on_one_hand_collapse_to_one():
    t, _ = tracker()
    i = 0
    while not t.poses(frame((800, 500), idx=i)):
        i += 1
    a = next(f for f in t._followers if f.active)
    b = next(f for f in t._followers if not f.active)
    b.active, b.centre, b.side = True, a.centre.copy(), a.side  # as if it had jumped onto the same hand
    poses = t.poses(frame((800, 500), idx=i + 1))
    assert len(poses) == 1 and t.following == 1 and a.active  # the older follower keeps it


def test_no_region_tiles_the_whole_frame():
    log: list = []
    t = CropHandTracker(None, factory=lambda: Fake(log, "follow"), search_factory=lambda: Fake(log, "search"))
    for i in range(12):
        if t.poses(frame((100, 100), idx=i)):
            break
    assert t.following == 1 and len(t.tiles) == 15  # 5 x 3 over 1920 x 1080


def test_motion_hints_search_where_something_moved_and_idle_frames_search_rarely():
    t, log = tracker(search_per_frame=2)
    t.IDLE_SEARCH_EVERY = 5
    for i in range(10):  # an empty table, nothing moving: a search turn only every 5th frame
        t.poses(frame(idx=i), motion=[])
    assert sum(1 for kind, _ in log if kind == "search") == 2 * 2
    log.clear()
    hand = (1500.0, 700.0)
    tile_k = next(k for k, (x0, y0, x1, y1) in enumerate(t.tiles) if x0 + 100 < hand[0] < x1 - 100
                  and y0 + 100 < hand[1] < y1 - 100)
    x0, y0, x1, y1 = t.tiles[tile_k]
    poses = t.poses(frame(hand, idx=11), motion=[(int(hand[0]) - 80, int(hand[1]) - 80, int(hand[0]) + 80,
                                                   int(hand[1]) + 80)])
    assert poses and abs(poses[0].img[9][1] - (hand[1] - 36)) < 2  # found in the first frame it moved in
    assert log[0][0] == "search"


def test_motion_boxes_mark_what_changed():
    from lightwork.vision.hands import MotionBoxes

    mb = MotionBoxes()
    a = np.zeros((1080, 1920, 3), np.uint8)
    assert mb(a) is None
    b = a.copy()
    b[400:520, 800:960] = 200
    boxes = mb(b)
    assert len(boxes) == 1
    x0, y0, x1, y1 = boxes[0]
    assert x0 <= 800 and y0 <= 400 and x1 >= 960 and y1 >= 520
    assert mb(b) == []
