"""Rock, paper or scissors from a hand's 3-D landmarks (the table's rock-paper-scissors game)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from lightwork.vision.hands import HandPose, ShapeVote, finger_bends, finger_states, hand_shape

KNUCKLES = ((0.026, -0.082), (0.006, -0.088), (-0.012, -0.083), (-0.029, -0.072))  # index..little MCP (x, y) metres
BONES = ((0.040, 0.024, 0.021), (0.044, 0.028, 0.023), (0.041, 0.027, 0.022), (0.032, 0.019, 0.019))  # to PIP, DIP, tip
BENDS = {  # degrees bent toward the palm at the MCP, PIP and DIP joints
    "s": (5.0, 8.0, 5.0),  # straight (relaxed)
    "c": (80.0, 100.0, 60.0),  # curled into a fist
    "l": (55.0, 85.0, 40.0),  # a loose fist
    "h": (25.0, 50.0, 30.0),  # half curled: neither
    "d": (75.0, 5.0, 5.0),  # straight, pointing down at the table (bent at the knuckle only)
}
SHAPES = {"rock": "cccc", "paper": "ssss", "scissors": "sscc"}


def hand_world(fingers: str, turn: np.ndarray | None = None) -> np.ndarray:
    """21 x 3 world landmarks like MediaPipe's (metres, centred on the hand). Built with the wrist at the origin, the
    fingers along -y and the back of the hand toward the camera (+z); each finger bends toward the palm (-z) by
    ``BENDS[letter]``. ``fingers``: one letter per finger, index first, or a shape name; ``turn``: a rotation."""
    fingers = SHAPES.get(fingers, fingers)
    w = np.zeros((21, 3))
    tucked = fingers[0] in "cl"  # the thumb across a fist, else out to the side (the classifier ignores it)
    w[1:5] = ([(0.022, -0.025, -0.01), (0.034, -0.045, -0.025), (0.024, -0.062, -0.035), (0.008, -0.072, -0.04)]
              if tucked else [(0.022, -0.025, -0.008), (0.042, -0.042, -0.014), (0.052, -0.06, -0.018),
                              (0.058, -0.078, -0.02)])
    for (x, y), bones, letter, mcp in zip(KNUCKLES, BONES, fingers, (5, 9, 13, 17), strict=True):
        p = np.array([x, y, 0.0])
        w[mcp] = p
        a = 0.0
        for j, (bone, bend) in enumerate(zip(bones, BENDS[letter], strict=True)):
            a += math.radians(bend)
            p = p + bone * np.array([0.0, -math.cos(a), -math.sin(a)])
            w[mcp + 1 + j] = p
    if turn is not None:
        w = w @ turn.T
    return w - w.mean(axis=0)


def hand(fingers: str, turn: np.ndarray | None = None, at=(480.0, 300.0), noise_m: float = 0.0, seed: int = 0,
         side: str = "Left") -> HandPose:
    """A hand under a camera looking straight down: world landmarks as :func:`hand_world` (plus ``noise_m`` of
    jitter), image landmarks their view from above (1.8 px per mm, around ``at``)."""
    w = hand_world(fingers, turn)
    if noise_m:
        w = w + np.random.default_rng(seed).normal(0.0, noise_m, w.shape)
    return HandPose(w[:, :2] * 1800.0 + np.asarray(at, float), w, side, 0.9)


def turns(n: int, seed: int = 0) -> list[np.ndarray]:
    """``n`` random rotations."""
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        q, r = np.linalg.qr(rng.normal(size=(3, 3)))
        q = q @ np.diag(np.sign(np.diag(r)))
        out.append(q if np.linalg.det(q) > 0 else -q)
    return out


def test_each_shape_is_read_from_the_fingers():
    assert hand_shape(hand("rock")) == "rock"
    assert hand_shape(hand("paper")) == "paper"
    assert hand_shape(hand("scissors")) == "scissors"
    assert hand_shape(hand("llll")) == "rock"  # a loose fist
    assert hand_shape(hand("sssc")) == hand_shape(hand("sssh")) == "paper"  # little finger down: still paper
    assert finger_states(hand("paper")) == [1, 1, 1, 1] and finger_states(hand("rock")) == [-1, -1, -1, -1]
    (bend, reach), = finger_bends(hand("ssss"))[:1]
    assert bend < 15 and reach > 1.3


@pytest.mark.parametrize("fingers", ["scss", "sccc", "hhhh", "cchh", "shcc", "hscc", "ssch"])
def test_unclear_hands_read_nothing(fingers):
    """One finger pointing, a half-curled hand, a finger in between: no move (the game asks again)."""
    assert hand_shape(hand(fingers)) == ""


def test_shapes_hold_however_the_hand_is_turned():
    for k, turn in enumerate(turns(24)):
        for shape in ("rock", "paper", "scissors"):
            assert hand_shape(hand(shape, turn)) == shape, (k, shape)


def test_shapes_hold_through_landmark_jitter():
    """MediaPipe's world landmarks shake by a few mm from frame to frame."""
    for seed in range(40):
        for shape in ("rock", "paper", "scissors"):
            assert hand_shape(hand(shape, noise_m=0.002, seed=seed)) == shape, (seed, shape)


def test_read_in_3d_not_in_the_camera_image():
    """Fingers pointing straight down at the table look like stubs from above (as a fist does); in 3-D they are
    straight: two of them are scissors, not rock."""
    down = hand("ddcc")
    assert hand_shape(down) == "scissors"
    flat = HandPose(down.img, np.zeros((21, 3)), "Left", 0.9)  # no world landmarks: only the image to go by
    assert hand_shape(flat) != "scissors"
    assert hand_shape(HandPose(hand("paper").img, np.zeros((21, 3)), "Left", 0.9)) == "paper"  # a flat hand still reads


def test_the_vote_needs_most_of_the_last_frames():
    v = ShapeVote()  # 3 of the last 5
    assert v.update(hand("rock")) == "" and v.update(hand("rock")) == ""
    assert v.update(hand("rock")) == "rock" and v.raw == "rock"
    assert v.update(hand("hhhh")) == "rock" and v.raw == ""  # one unclear frame: still rock
    assert v.update(hand("paper")) == "rock"  # rock 3 of the last 5
    assert v.update(hand("paper")) == ""  # rock 2, paper 2: neither yet
    assert v.update(hand("paper")) == "paper"
    v.reset()
    assert v.name == "" and v.raw == "" and v.update(hand("paper")) == ""


def test_a_tie_goes_to_the_latest_shape():
    v = ShapeVote(n=4, need=2)
    for shape in ("rock", "rock", "scissors", "scissors"):
        v.update(hand(shape))
    assert v.name == "scissors"
    v.update(hand("hhhh"))  # rock 1, scissors 2
    assert v.name == "scissors"
