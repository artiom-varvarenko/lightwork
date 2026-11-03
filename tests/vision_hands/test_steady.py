"""PoseFilter: a still hand's landmarks drawn without shaking, a moving hand still followed closely; JitterMeter."""

from __future__ import annotations

import numpy as np

from lightwork.vision.hands import JitterMeter, PoseFilter, palm_px

FPS = 25.0
PALM = 160.0  # px: a hand across the table from a 1080p camera (2 Nov rig)


def hand(centre=(900.0, 600.0)) -> np.ndarray:
    """21 landmarks of a flat hand, wrist to middle knuckle PALM px, fingers up the image."""
    rng = np.random.default_rng(7)
    img = rng.uniform(-0.5, 0.5, (21, 2)) * PALM * 1.6
    img[0] = (0.0, PALM / 2)
    img[9] = (0.0, -PALM / 2)
    return img + centre


def frames(base: np.ndarray, n: int, noise_px: float, seed: int = 1, velocity=(0.0, 0.0)):
    rng = np.random.default_rng(seed)
    for i in range(n):
        t = i / FPS
        yield t, base + np.asarray(velocity) * t + rng.normal(0.0, noise_px, base.shape)


def test_a_still_hand_is_drawn_still():
    for noise in (3.0, 5.0):  # px per landmark: MediaPipe-like
        f, base = PoseFilter(), hand()
        seen = list(frames(base, 100, noise_px=noise))  # 4 s
        drawn = np.array([f.update(img, t) for t, img in seen])[25:]
        raw = np.array([img for _, img in seen])[25:]
        rms = lambda a: float(np.sqrt(np.mean(np.sum(np.diff(a, axis=0) ** 2, axis=2))))  # noqa: E731
        assert rms(drawn) < rms(raw) / 20  # frame to frame: raw 6-10 px, drawn well under a px
        assert float(np.max(np.linalg.norm(drawn[-1] - base, axis=1))) < 0.06 * PALM  # and drawn where the hand is


def test_a_moving_hand_is_followed_closely():
    f, base = PoseFilter(), hand()
    speed = 3.0 * PALM  # px/s: a deliberate move across the table
    for t, img in frames(base, 50, noise_px=3.0, velocity=(speed, 0.0)):
        drawn = f.update(img, t)
    lag = float(np.mean(base[:, 0] + speed * t - drawn[:, 0]))
    assert 0.0 < lag < 0.15 * PALM  # behind by less than 15 % of a palm (raw MediaPipe noise is about 2 %)


def test_a_jump_is_caught_up_with_and_a_gap_restarts():
    f, base = PoseFilter(), hand()
    for t, img in frames(base, 25, noise_px=0.0):
        f.update(img, t)
    moved = base + (2 * PALM, 0.0)
    for i in range(1, 9):
        drawn = f.update(moved, 1.0 + i / FPS)
    assert float(np.max(np.linalg.norm(drawn - moved, axis=1))) <= (PoseFilter.DEADBAND + 0.01) * PALM  # 0.3 s
    far = base + (-5 * PALM, 0.0)
    assert np.array_equal(f.update(far, 3.0), far)  # after a gap: drawn where it is at once


def test_jitter_meter_counts_only_a_still_hand():
    m, base = JitterMeter(), hand()
    f = PoseFilter()
    for t, img in frames(base, 50, noise_px=2.0):
        m.add(img, f.update(img, t))
    n, raw, drawn, redrawn = m.take()
    assert n >= 40 and 3.5 < raw < 4.5 and drawn < 0.2 and redrawn < 0.5  # raw: 2 sigma for 2-D steps
    m.clear()
    for t, img in frames(base, 50, noise_px=2.0, velocity=(3 * PALM, 0.0)):
        m.add(img, img)
    assert m.take()[0] == 0  # a moving hand is no jitter
    assert palm_px(base) == PALM
