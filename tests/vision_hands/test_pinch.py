"""Pinch (thumb and index finger together): the pen of draw and guess."""

from __future__ import annotations

import math

import numpy as np

from lightwork.vision.hands import PINCH_DOWN, PINCH_UP, HandPose, Pinch, pinch_ratio


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


def test_ratio_is_in_palm_lengths_and_in_3d():
    assert math.isclose(pinch_ratio(hand(0.2)), 0.2)
    assert math.isclose(pinch_ratio(hand(0.2, palm=0.12)), 0.2)  # a bigger hand, the same pinch
    tucked = hand(0.38, img_ratio=0.15)  # seen from above a tucked thumb overlaps the finger; in 3-D it is 3 cm off
    assert pinch_ratio(tucked) > PINCH_DOWN  # so it starts no line
    flat = HandPose(tucked.img, np.zeros((21, 3)), "Left", 0.9)  # no world landmarks (synthetic): the image
    assert math.isclose(pinch_ratio(flat), 0.15) and pinch_ratio(flat) < PINCH_DOWN


def test_pinch_has_hysteresis_and_ignores_one_bad_frame():
    p = Pinch()
    assert not p.update(hand(0.6))
    assert not p.update(hand((PINCH_DOWN + PINCH_UP) / 2))  # in the band, coming from apart: still up
    assert p.update(hand(0.1))  # together: down at once (hands are not found in every frame)
    assert p.update(hand((PINCH_DOWN + PINCH_UP) / 2))  # in the band: stays down
    assert p.update(hand(0.6))  # one frame apart (a blurred frame): still down
    assert p.update(hand(0.1))
    assert p.update(hand(0.6)) and not p.update(hand(0.6))  # two frames apart: up
    assert not p.update(hand((PINCH_DOWN + PINCH_UP) / 2))  # in the band: stays up
    assert not p.update(hand(0.38))  # a thumb tucked beside a pointing finger (3 Nov rig: 0.35-0.41) is no pinch
    p.update(hand(0.1))
    p.reset()
    assert not p.down and p.ratio == math.inf
