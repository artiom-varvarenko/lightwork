"""RegionHandTracker: two hands searched in a crop around the projected page, landmarks back in camera px."""

from __future__ import annotations

import cv2
import numpy as np

from lightwork.vision.hands import HandPose, RegionHandTracker
from lightwork.vision.source import Frame


class FakeLandmarker:
    """Finds every bright square in its input as a "hand" whose landmarks all sit on the square's centre."""

    def __init__(self):
        self.inputs: list[tuple[int, int]] = []

    def poses(self, frame: Frame) -> list[HandPose]:
        img = frame.image
        self.inputs.append(img.shape[:2])
        n, _, _, cents = cv2.connectedComponentsWithStats((img[..., 0] > 128).astype(np.uint8))
        return [HandPose(np.tile(cents[i], (21, 1)), np.zeros((21, 3)), "Left", 0.9) for i in range(1, n)]

    def close(self):
        pass


def frame(*centres) -> Frame:
    img = np.zeros((1080, 1920, 3), np.uint8)
    for cx, cy in centres:
        img[cy - 10:cy + 10, cx - 10:cx + 10] = 255
    return Frame(img, 1.0, 0)


def test_two_hands_in_the_crop_come_back_in_camera_px():
    fake = FakeLandmarker()
    t = RegionHandTracker((400, 200, 1500, 900), factory=lambda: fake)
    poses = t.poses(frame((600, 500), (1300, 450)))
    assert fake.inputs == [(700, 1100)]
    assert sorted(p.tip for p in poses) == [(599.5, 499.5), (1299.5, 449.5)]
    obs = t.process(frame((600, 500), (1300, 450)))
    assert obs.present and len(obs.tips) == 2


def test_no_region_searches_the_whole_frame_and_a_tiny_region_nothing():
    fake = FakeLandmarker()
    t = RegionHandTracker(None, factory=lambda: fake)
    assert len(t.poses(frame((100, 100)))) == 1 and fake.inputs == [(1080, 1920)]
    assert t.set_region((1900, 1060, 2400, 1400)) == [(1900, 1060, 2400, 1400)]
    assert t.poses(frame((100, 100))) == [] and len(fake.inputs) == 1
