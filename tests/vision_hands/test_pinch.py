"""Pinch (thumb and index finger together): the pen of draw and guess."""

from __future__ import annotations

import math

import numpy as np

from lightwork.vision.hands import PINCH_DOWN, PINCH_UP, PINCH_UP_FRAMES, HandPose, Pinch, pinch_ratio, pinch_ratio_3d


def hand(ratio: float, palm: float = 0.09, img_ratio: float | None = None) -> HandPose:
    """A hand whose thumb tip is ``ratio`` palm lengths from its index fingertip in 3-D (``img_ratio`` in the image)."""
    world = np.zeros((21, 3))
    world[9] = (0.0, -palm, 0.0)
    world[8] = (0.0, -1.9 * palm, 0.0)
    world[4] = world[8] + (ratio * palm, 0.0, 0.0)
    img = np.zeros((21, 2))
    img[9] = (0.0, -100.0)
    img[8] = (0.0, -190.0)
    img[4] = img[8] + ((ratio if img_ratio is None else img_ratio) * 100.0, 0.0)
    return HandPose(img, world, "Left", 0.9)


def test_ratio_is_in_palm_lengths_in_the_image():
    assert math.isclose(pinch_ratio(hand(0.2)), 0.2)
    big = hand(0.2)
    assert math.isclose(pinch_ratio(HandPose(big.img * 1.7, big.world, "Left", 0.9)), 0.2)  # nearer the camera
    # 3 Nov rig: in 3-D (MediaPipe's world landmarks) pinches and open hands both read 0.25-0.5; in the image the
    # touching tips are one spot. The image decides; 3-D is only logged.
    seen_from_above = hand(0.38, img_ratio=0.12)
    assert pinch_ratio(seen_from_above) < PINCH_DOWN and math.isclose(pinch_ratio_3d(seen_from_above), 0.38)
    flat = HandPose(np.zeros((21, 2)), seen_from_above.world, "Left", 0.9)  # no palm in the image: never a pinch
    assert pinch_ratio(flat) == math.inf


def test_pinch_has_hysteresis_and_ignores_short_blips():
    p = Pinch()
    assert not p.update(hand(0.6))
    assert not p.update(hand((PINCH_DOWN + PINCH_UP) / 2))  # in the band, coming from apart: still up
    assert p.update(hand(0.1)) and not p.apart  # together: down at once
    assert p.update(hand((PINCH_DOWN + PINCH_UP) / 2)) and not p.apart  # in the band: stays down
    for _ in range(PINCH_UP_FRAMES - 1):  # apart for fewer frames than it takes (a blip): still down, holding
        assert p.update(hand(0.6)) and p.apart
    assert p.update(hand(0.1)) and not p.apart
    for _ in range(PINCH_UP_FRAMES - 1):
        p.update(hand(0.6))
    assert not p.update(hand(0.6)) and not p.apart  # apart long enough: let go
    assert not p.update(hand((PINCH_DOWN + PINCH_UP) / 2))  # in the band: stays up
    p.update(hand(0.1))
    p.reset()
    assert not p.down and p.ratio == math.inf
