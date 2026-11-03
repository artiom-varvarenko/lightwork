"""lightwork play drop zone: light compensation must follow the room, not creep on a static scene."""

from __future__ import annotations

import cv2
import numpy as np

from lightwork.playground import RELEARN_AFTER_S, Dashboard, DropZone, Geo

FW, FH = 960, 540
HI = np.array([[1.1, 0.0, 50.0], [0.0, 1.1, 40.0], [0.0, 0.0, 1.0]])  # page -> camera px
GEO = Geo(np.linalg.inv(HI), 749.0, 421.0)
HZ = 12.0  # zone updates per second (every second frame)


def wall(seed: int = 0) -> np.ndarray:
    """A textured wall with a vignette and a darker mark in the zone: an uneven, skewed brightness spread."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:FH, 0:FW].astype(np.float32)
    vig = 1.0 - 1.4 * (((xx - FW / 2) / FW) ** 2 + ((yy - FH / 2) / FH) ** 2)
    img = np.stack([95 * vig, 108 * vig, 118 * vig], axis=2)
    img += cv2.GaussianBlur(rng.normal(0, 12, img.shape).astype(np.float32), (0, 0), 3)
    img[220:320, 640:740] *= 0.55
    return img


class Feed:
    def __init__(self, scene: np.ndarray, noise: float = 4.0, seed: int = 1):
        self.scene, self.noise, self.rng = scene, noise, np.random.default_rng(seed)

    def __call__(self, scene: np.ndarray | None = None) -> np.ndarray:
        s = self.scene if scene is None else scene
        return np.clip(s + self.rng.normal(0, self.noise, s.shape), 0, 255).astype(np.uint8)


def zone_on(feed: Feed) -> DropZone:
    z = DropZone(GEO, Dashboard(GEO).zone_rect, (FW, FH))
    z.learn(feed())
    return z


def test_static_wall_keeps_its_light_gain_and_is_never_relearned():
    """1 Nov, wall: nothing moved, yet the gain crept 1.00 -> 0.88 within a minute and the zone relearned every 1-2
    min, because slow adaptation and the gain estimate fed each other."""
    feed = Feed(wall())
    z = zone_on(feed)
    worst = 0.0
    for i in range(int(90 * HZ)):
        z.update(feed(), i / HZ, overlay=None, hand_near=False)
        worst = max(worst, float(np.max(np.abs(z.gain - 1.0))))
    assert z.relearns == 0
    assert z.status == "ready"
    assert worst < 0.03


def test_even_dimming_is_compensated_then_relearned():
    scene = wall()
    feed = Feed(scene)
    z = zone_on(feed)
    t = 0.0
    for _ in range(int(5 * HZ)):
        z.update(feed(), t, overlay=None, hand_near=False)
        t += 1 / HZ
    dim = scene * 0.8
    z.update(feed(dim), t, overlay=None, hand_near=False)
    assert z.status == "light"
    assert np.allclose(z.gain, 0.8, atol=0.03)
    assert z.objects == ()  # compensated: the dimmed wall is no object
    for _ in range(int((RELEARN_AFTER_S + 2) * HZ)):
        t += 1 / HZ
        z.update(feed(dim), t, overlay=None, hand_near=False)
    assert z.relearns == 1
    assert z.status == "ready"
    assert np.allclose(z.gain, 1.0, atol=0.03)


def test_object_put_down_after_a_long_quiet_spell_is_found():
    scene = wall()
    feed = Feed(scene)
    z = zone_on(feed)
    t = 0.0
    for _ in range(int(60 * HZ)):
        z.update(feed(), t, overlay=None, hand_near=False)
        t += 1 / HZ
    with_obj = scene.copy()
    with_obj[150:200, 600:660] = (40, 60, 200)  # red block, BGR
    changed = False
    for _ in range(int(2 * HZ)):
        t += 1 / HZ
        changed |= z.update(feed(with_obj), t, overlay=None, hand_near=False)
    assert changed
    assert [o.colour for o in z.objects] == ["red"]
