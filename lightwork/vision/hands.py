"""Hands (D12, advisory only): MediaPipe HandLandmarker -> index fingertip in camera px -> C1 ``hands`` events.

``HandTracker`` runs MediaPipe 1.x HandLandmarker in VIDEO mode at 640 px wide (640x360 for 16:9 frames), or at full
resolution / in IMAGE mode when asked. MediaPipe is in the optional ``live`` extra and is imported lazily; replay,
verify and export never touch this class.
``TiledHandTracker`` also searches overlapping tiles over a region, for hands far from the camera (``lightwork play``).
``RegionHandTracker`` follows up to two hands, with all landmarks, in a crop around the projected page.
``CropHandTracker`` (``play --table``) finds up to two hands in full-resolution tiles and follows each in its own crop.
``hand_shape`` reads rock, paper or scissors from a hand's 3-D landmarks; ``ShapeVote`` steadies it over frames.

``HandEvents`` turns observations into C1 ``hands`` payloads ``{over_regions, tip_xy}``:

- ``over_regions``: every region ref (``geometry.Regions.all_refs()``) that contains the index fingertip of any
  detected hand (rig:H1 is a fingertip test). A region is entered when the tip is inside its polygon and left only
  when the tip is more than ``exit_margin_px`` outside it, so tracking jitter on a cell boundary does not flicker.
- ``tip_xy``: the primary hand's index fingertip (highest handedness score), quantized with ``q_px``; ``None`` = no
  hand. A dropout shorter than ``absent_s`` holds the last state.
- Events are emitted only when the state differs from the last emitted one (regions, presence, or a tip move of at
  least ``min_move_px``), and never closer than ``1 / max_hz`` seconds apart. A change that arrives inside that
  window is emitted on the first ``update`` after it, if it still holds.

``python -m lightwork.vision.hands`` runs the rig:H1 fingertip-registration test against the live camera.
"""

from __future__ import annotations

import argparse
import hashlib
import math
import os
import sys
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import cv2
import httpx
import numpy as np

from ..config import STATE_DIR_NAME
from ..contracts.canonical import q_px
from ..geometry import Regions
from .source import Frame

MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
MODEL_SHA256 = "fbc2a30080c3c557093b5ddfc334698132eb341044ccee322ccf8bcf3607cde1"
MODEL_SIZE = 7_819_105
MODEL_FILE = "hand_landmarker.task"
INFER_W = 640
INDEX_TIP = 8  # MediaPipe hand landmark index of the index fingertip
THUMB_TIP = 4
# Thumb tip to index fingertip in the camera image, in palm lengths. 3 Nov rig, 1074 readings while someone wrote
# "ABCD" and drew a heart pinching: pinches 0.05-0.25, open hands 0.4-0.8, few in between. (MediaPipe's 3-D world
# landmarks put both at 0.25-0.5: the pen went down only when a twist of the arm happened to bring it under 0.30.)
PINCH_DOWN = 0.25  # closer than this is a pinch (the pen goes down) ...
PINCH_UP = 0.35  # ... and only past this is it let go: in between nothing changes, so a shaky pinch does not flicker
PINCH_UP_FRAMES = 3  # frames in a row apart before it is let go: 1- and 2-frame blips came 14 times in that session
# Rock, paper, scissors: each finger (MCP, PIP, DIP, tip landmarks) out or curled, from the 3-D world landmarks.
# Synthetic hands (tests/vision_hands/test_shape.py): straight fingers bend 10 degrees, relaxed open ones 30-35, half
# curled ones about 64 (reach 1.1), a fist 100-130 (reach 0.6-0.8). Not yet measured on the rig.
FINGERS = ((5, 6, 7, 8), (9, 10, 11, 12), (13, 14, 15, 16), (17, 18, 19, 20))  # index, middle, ring, little finger
OUT_BEND = 55.0  # degrees: a finger bent less than this at its middle joint (MCP->PIP against PIP->tip) ...
OUT_REACH = 1.15  # ... whose tip is this many times further from the wrist than that joint is out (straight: 1.3-1.4)
CURL_BEND = 75.0  # bent more than this, ...
CURL_REACH = 1.0  # ... or its tip no further from the wrist than that joint: curled. In between: unsure
SHAPE_FRAMES = 5  # a hand's shape is the one most of its last 5 frames read ...
SHAPE_NEED = 3  # ... at least 3 of them


class HandsUnavailable(RuntimeError):
    """MediaPipe or the pinned model is missing; callers fall back to motion-only ``activity_at`` (PLAN §13)."""


def default_model_path(root: Path | None = None) -> Path:
    return (root or Path.cwd()) / STATE_DIR_NAME / "models" / MODEL_FILE


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def ensure_model(path: Path) -> Path:
    """Return ``path`` holding the pinned model, downloading it first if absent. Never overwrites a mismatching file."""
    if path.is_file():
        if _sha256(path) != MODEL_SHA256:
            raise HandsUnavailable(f"{path}: SHA-256 does not match the pin {MODEL_SHA256}; delete it to re-download")
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_name(path.name + ".part")
    h = hashlib.sha256()
    n = 0
    try:
        with httpx.stream("GET", MODEL_URL, follow_redirects=True, timeout=60.0) as r, part.open("wb") as f:
            r.raise_for_status()
            for chunk in r.iter_bytes(1 << 16):
                f.write(chunk)
                h.update(chunk)
                n += len(chunk)
    except (httpx.HTTPError, OSError) as e:
        part.unlink(missing_ok=True)
        raise HandsUnavailable(f"downloading {MODEL_URL} failed: {e}") from e
    if n != MODEL_SIZE or h.hexdigest() != MODEL_SHA256:
        part.unlink(missing_ok=True)
        raise HandsUnavailable(f"{MODEL_URL}: got {n} bytes sha256 {h.hexdigest()}, pinned {MODEL_SIZE} bytes {MODEL_SHA256}")
    os.replace(part, path)
    return path


def _import_mediapipe():
    """Import MediaPipe without its audio stack.

    ``import mediapipe`` imports ``sounddevice``, whose PortAudio init opens a PipeWire-JACK client. On this laptop
    (Arch, PipeWire) that sets RLIMIT_RTTIME to 0 us and asks the Realtime portal to promote a thread; the kernel
    then SIGKILLs the process as soon as HandLandmarker starts its workers (measured 29 Oct: exit 137 in
    ``create_from_options``). Hands never use audio, so ``sounddevice`` is hidden while MediaPipe imports.
    """
    hide = "sounddevice" not in sys.modules
    if hide:
        sys.modules["sounddevice"] = None  # makes `import sounddevice` raise ImportError, which MediaPipe tolerates
    try:
        import mediapipe as mp
        from mediapipe.tasks.python import vision
        from mediapipe.tasks.python.core.base_options import BaseOptions
    except ImportError as e:
        raise HandsUnavailable("mediapipe is not installed; install the live extra: uv sync --extra live") from e
    finally:
        if hide and sys.modules.get("sounddevice", False) is None:
            del sys.modules["sounddevice"]
    if sys.platform.startswith("linux"):
        import resource  # POSIX-only module; RLIMIT_RTTIME is Linux-only

        if resource.getrlimit(resource.RLIMIT_RTTIME)[1] == 0:
            raise HandsUnavailable("RLIMIT_RTTIME is 0 in this process (a PortAudio/JACK client was opened before MediaPipe); HandLandmarker would be SIGKILLed")
    return mp, vision, BaseOptions


@dataclass(frozen=True)
class HandObs:
    present: bool
    tip_xy: tuple[float, float] | None  # primary hand's index fingertip, camera px
    tips: tuple[tuple[float, float], ...] = ()  # index fingertips of all detected hands, primary first


@dataclass(frozen=True, eq=False)
class HandPose:
    img: np.ndarray  # 21 x 2 camera px
    world: np.ndarray  # 21 x 3 metres, hand-centred (MediaPipe world landmarks)
    side: str  # MediaPipe handedness label ("Left"/"Right"; it assumes a mirrored, selfie-style image)
    score: float

    @property
    def tip(self) -> tuple[float, float]:
        return float(self.img[INDEX_TIP][0]), float(self.img[INDEX_TIP][1])

    def shifted(self, dx: float, dy: float) -> HandPose:
        return HandPose(self.img + (dx, dy), self.world, self.side, self.score)


def palm_px(img: np.ndarray) -> float:
    """Palm length in image px: wrist to middle-finger knuckle (landmarks 0 and 9)."""
    return float(np.linalg.norm(img[9] - img[0]))


def pinch_ratio(pose: HandPose) -> float:
    """Thumb tip to index fingertip over the palm length (wrist to middle knuckle), in the camera image. Seen from
    above, touching tips are one spot in the image whatever the hand's angle. (A thumb tucked beside a pointing finger
    reads 0.2-0.3 too.)"""
    palm = float(np.linalg.norm(pose.img[9] - pose.img[0]))
    return float(np.linalg.norm(pose.img[THUMB_TIP] - pose.img[INDEX_TIP])) / palm if palm > 1e-6 else math.inf


def pinch_ratio_3d(pose: HandPose) -> float:
    """The same from MediaPipe's 3-D world landmarks (for the logs: it does not tell a pinch here, see PINCH_DOWN)."""
    palm = float(np.linalg.norm(pose.world[9] - pose.world[0]))
    return float(np.linalg.norm(pose.world[THUMB_TIP] - pose.world[INDEX_TIP])) / palm if palm > 1e-6 else math.inf


class Pinch:
    """Thumb and index finger held together, for one hand: down at once below ``down`` palm lengths apart, up again
    only above ``up`` and only once ``up_frames`` frames in a row agree (a blip does not break a line). ``apart``:
    down, but this frame reads apart (it may be letting go: hold the line still)."""

    def __init__(self, down: float = PINCH_DOWN, up: float = PINCH_UP, up_frames: int = PINCH_UP_FRAMES):
        self.down_at, self.up_at, self.up_frames = down, up, up_frames
        self.down = False
        self.ratio = math.inf  # the last frame's ``pinch_ratio``
        self._run = 0

    def update(self, pose: HandPose) -> bool:
        self.ratio = r = pinch_ratio(pose)
        if not self.down:
            self.down, self._run = r < self.down_at, 0
            return self.down
        self._run = self._run + 1 if r > self.up_at else 0
        if self._run >= self.up_frames:
            self.down, self._run = False, 0
        return self.down

    @property
    def apart(self) -> bool:
        return self.down and self._run > 0

    def reset(self) -> None:
        self.down, self.ratio, self._run = False, math.inf, 0


def finger_bends(pose: HandPose) -> list[tuple[float, float]]:
    """(bend in degrees, reach) of the index, middle, ring and little finger: the bend is the angle at the middle
    joint between the finger's first bone (MCP to PIP) and the rest of it (PIP to tip), the reach is how much further
    the tip is from the wrist than the middle joint. From the 3-D world landmarks (synthetic poses without them: image
    px), because the camera looks down on the back of the hand: a finger curled under the palm and one pointing down
    at the table are both short in the image, and the thumb tucked over a fist lies across the fingers there."""
    pts = pose.world if float(np.linalg.norm(pose.world[9] - pose.world[0])) > 1e-6 else pose.img
    out = []
    for mcp, pip, _, tip in FINGERS:
        a, b = pts[pip] - pts[mcp], pts[tip] - pts[pip]
        na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
        bend = math.degrees(math.acos(max(-1.0, min(1.0, float(a @ b) / (na * nb))))) if na * nb > 1e-12 else 0.0
        near = float(np.linalg.norm(pts[pip] - pts[0]))
        out.append((bend, float(np.linalg.norm(pts[tip] - pts[0])) / near if near > 1e-9 else 0.0))
    return out


def finger_states(pose: HandPose) -> list[int]:
    """Index, middle, ring and little finger: 1 out (straight and reaching), -1 curled, 0 in between (unsure)."""
    return [1 if bend < OUT_BEND and reach > OUT_REACH else -1 if bend > CURL_BEND or reach < CURL_REACH else 0
            for bend, reach in finger_bends(pose)]


def hand_shape(pose: HandPose) -> str:
    """"rock", "paper" or "scissors" from the four fingers (the thumb is ignored: it may be anywhere in all three), or
    "" when unsure. Rock: no finger out, three or four curled. Scissors: index and middle finger out, ring and little
    finger curled. Paper: index and middle finger out and at least one more. Anything else (one finger pointing, a
    half curled hand) is ""."""
    s = finger_states(pose)
    out = sum(v == 1 for v in s)
    if out == 0:
        return "rock" if sum(v == -1 for v in s) >= 3 else ""
    if s[0] == 1 and s[1] == 1:
        if out >= 3:
            return "paper"
        if s[2] == -1 and s[3] == -1:
            return "scissors"
    return ""


class ShapeVote:
    """One hand's rock, paper or scissors: the shape most of its last ``n`` frames read, once at least ``need`` of them
    agree (landmarks flicker, and a hand moving into a shape passes others), else ""."""

    def __init__(self, n: int = SHAPE_FRAMES, need: int = SHAPE_NEED):
        self.need = need
        self.recent: deque[str] = deque(maxlen=n)
        self.name = ""  # the voted shape
        self.raw = ""  # the last frame's reading

    def update(self, pose: HandPose) -> str:
        self.raw = hand_shape(pose)
        self.recent.append(self.raw)
        best, top = "", 0
        for s in reversed(self.recent):  # most votes; a tie goes to the latest
            n = self.recent.count(s)
            if s and n > top:
                best, top = s, n
        self.name = best if top >= self.need else ""
        return self.name

    def reset(self) -> None:
        self.recent.clear()
        self.name = self.raw = ""


def _alpha(cutoff_hz, dt: float):
    return 1.0 / (1.0 + 1.0 / (2.0 * math.pi * np.asarray(cutoff_hz) * dt))


class PoseFilter:
    """Steadies one hand's 21 landmarks for drawing: a One Euro filter per landmark, then a dead band.

    MediaPipe's landmarks move by a few px every frame even when the hand is still, and drawn raw onto the hand they
    shake (2 Nov, table: "the projection keeps shaking ... even when they're still"). The One Euro filter (Casiez,
    Roussel and Vogel, CHI 2012) smooths strongly at rest and hardly at all in fast moves: its cutoff rises with each
    landmark's speed. Speeds are measured in palm lengths per second and the dead band in palm lengths, so the same
    settings hold for a hand near the camera and one across the table. A drawn landmark then moves only once its
    filtered position is more than ``deadband`` away, and then only by the excess (a leash, never a jump): what is left
    of the noise never reaches the projection, and a small tremble between the projected lines and the hand they light
    cannot feed itself.
    """

    # Tuned on simulated MediaPipe noise (3-5 px per landmark on a 160 px palm, 25 fps, 2 Nov): a still hand is drawn
    # 40-100x steadier than raw; at 3 palm lengths per second the drawing trails by about a tenth of a palm. 3 Nov
    # ("the hand tracking is slow"): a 0.5 -> 1.0 Hz floor only cut the trail of a slow hand from 155 to 131 ms; the
    # dead band sets most of it, and it is what keeps a still hand still. Kept as tuned.
    MIN_CUTOFF_HZ = 0.5  # at rest
    BETA = 2.0  # cutoff added per palm length per second of landmark speed
    D_CUTOFF_HZ = 2.0  # speed estimate
    DEADBAND = 0.03  # palm lengths (about 3 mm on a hand): the leash between filtered and drawn landmarks
    RESET_S = 0.3  # a longer gap restarts the filter at the new position

    def __init__(self, *, min_cutoff: float = MIN_CUTOFF_HZ, beta: float = BETA, d_cutoff: float = D_CUTOFF_HZ,
                 deadband: float = DEADBAND):
        self.min_cutoff, self.beta, self.d_cutoff, self.deadband = min_cutoff, beta, d_cutoff, deadband
        self.reset()

    def reset(self) -> None:
        self.raw: np.ndarray | None = None  # last input, image px
        self.x: np.ndarray | None = None  # filtered
        self.dx: np.ndarray | None = None  # filtered velocity, px/s
        self.shown: np.ndarray | None = None  # what to draw: filtered, behind the dead band
        self.palm = 0.0  # smoothed palm length, px
        self.t: float | None = None

    def update(self, img: np.ndarray, t: float) -> np.ndarray:
        """Feed this frame's landmarks (21 x 2 px, frame time ``t`` in s); returns the landmarks to draw (a copy)."""
        img = np.asarray(img, dtype=np.float64)
        palm = max(palm_px(img), 1.0)
        dt = None if self.t is None else t - self.t
        if dt is None or dt <= 0 or dt > self.RESET_S:
            self.raw, self.x, self.shown = img.copy(), img.copy(), img.copy()
            self.dx = np.zeros_like(img)
            self.palm, self.t = palm, t
            return self.shown.copy()
        self.palm += 0.2 * (palm - self.palm)
        self.dx += _alpha(self.d_cutoff, dt) * ((img - self.raw) / dt - self.dx)
        speed = np.linalg.norm(self.dx, axis=1) / self.palm  # palm lengths per second, per landmark
        self.x += _alpha(self.min_cutoff + self.beta * speed, dt)[:, None] * (img - self.x)
        gap = self.x - self.shown
        dist = np.linalg.norm(gap, axis=1)
        excess = dist - self.deadband * self.palm
        moved = excess > 0  # dragged on a leash: never a jump, and still while the filter stays within the band
        self.shown[moved] += gap[moved] * (excess[moved] / dist[moved])[:, None]
        self.raw, self.t = img.copy(), t
        return self.shown.copy()


class JitterMeter:
    """Frame-to-frame landmark movement of a hand held still, raw and as drawn (``PoseFilter.shown``), in image px.

    A frame counts when the raw palm centre stayed within ``STILL`` palm lengths of its mean over the last ``WINDOW``
    frames. ``take()`` returns (still frames, raw RMS px, drawn RMS px, share of still frames that redrew) and starts
    over."""

    WINDOW = 8
    STILL = 0.04
    PALM = (0, 5, 9, 13, 17)

    def __init__(self) -> None:
        self._centres: list[np.ndarray] = []
        self._prev: tuple[np.ndarray, np.ndarray] | None = None
        self._n = self._moved = 0
        self._raw2 = self._shown2 = 0.0

    def clear(self) -> None:
        self._centres, self._prev = [], None

    def add(self, raw: np.ndarray, shown: np.ndarray) -> None:
        self._centres = (self._centres + [raw[list(self.PALM)].mean(axis=0)])[-self.WINDOW:]
        prev, self._prev = self._prev, (raw.copy(), shown.copy())
        if prev is None or len(self._centres) < self.WINDOW:
            return
        c = np.array(self._centres)
        if float(np.max(np.linalg.norm(c - c.mean(axis=0), axis=1))) > self.STILL * max(palm_px(raw), 1.0):
            return
        d_raw = np.linalg.norm(raw - prev[0], axis=1)
        d_shown = np.linalg.norm(shown - prev[1], axis=1)
        self._n += 1
        self._raw2 += float(np.mean(d_raw**2))
        self._shown2 += float(np.mean(d_shown**2))
        self._moved += bool(np.any(d_shown > 0))

    def take(self) -> tuple[int, float, float, float]:
        n = self._n
        out = (n, math.sqrt(self._raw2 / n), math.sqrt(self._shown2 / n), self._moved / n) if n else (0, 0.0, 0.0, 0.0)
        self._n = self._moved = 0
        self._raw2 = self._shown2 = 0.0
        return out


class HandTracker:
    """MediaPipe HandLandmarker (VIDEO mode). Not thread-safe: call ``process`` from one worker thread.

    ``num_hands=1`` keeps a tracked hand on the cheap landmark path (8.5 ms p50 / 9.7 ms p95 per 1080p frame incl.
    the resize, Ryzen 7 7840HS, 29 Oct). With ``num_hands=2`` and one hand in view MediaPipe re-runs palm detection
    on every frame looking for the second hand: 17 ms p50 / 25 ms p95, over the 15 ms budget.
    """

    def __init__(self, model_path: Path | None = None, *, num_hands: int = 1, min_confidence: float = 0.5,
                 infer_w: int | None = INFER_W, image_mode: bool = False, gpu: bool = False,
                 track_confidence: float | None = None):
        """``infer_w``: width every input is resized to first (None: as given, at full resolution). ``image_mode``:
        each call stands alone (MediaPipe IMAGE mode, palm detection every time) instead of tracking (VIDEO mode).
        ``gpu``: MediaPipe's GPU delegate (OpenGL ES through Mesa on this laptop's Radeon: a 540 px palm search in
        4.2 ms instead of 9.7 ms on the CPU, 3 Nov); falls back to the CPU when it cannot start.
        ``min_confidence`` is what it takes to find a hand; ``track_confidence`` (default the same) what it takes to
        keep one: lower, so a hand already found is not lost, without taking a toy for a new hand."""
        mp, vision, BaseOptions = _import_mediapipe()
        path = ensure_model(Path(model_path) if model_path is not None else default_model_path())

        def options(delegate):
            return vision.HandLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=str(path), delegate=delegate),
                running_mode=vision.RunningMode.IMAGE if image_mode else vision.RunningMode.VIDEO,
                num_hands=num_hands,
                min_hand_detection_confidence=min_confidence,
                min_hand_presence_confidence=min_confidence if track_confidence is None else track_confidence,
                min_tracking_confidence=min_confidence if track_confidence is None else track_confidence,
            )

        self._mp = mp
        self.delegate = "CPU"
        self.gpu_error: str | None = None
        if gpu:
            try:
                self._lm = vision.HandLandmarker.create_from_options(options(BaseOptions.Delegate.GPU))
                self.delegate = "GPU"
            except Exception as exc:  # noqa: BLE001 - no GPU context: the CPU path always works
                self.gpu_error = f"{type(exc).__name__}: {exc}"
        if self.delegate == "CPU":
            self._lm = vision.HandLandmarker.create_from_options(options(BaseOptions.Delegate.CPU))
        self._last_ms = -1
        self._infer_w = infer_w
        self._image_mode = image_mode

    def _detect(self, frame: Frame):
        img = frame.image
        h, w = img.shape[:2]
        if self._infer_w is None or w == self._infer_w:
            small = img
        else:
            small = cv2.resize(img, (self._infer_w, max(1, round(h * self._infer_w / w))), interpolation=cv2.INTER_AREA)
        rgb = np.ascontiguousarray(cv2.cvtColor(small, cv2.COLOR_BGR2RGB))
        image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
        if self._image_mode:
            return self._lm.detect(image)
        # VIDEO mode requires strictly increasing integer-ms timestamps.
        ts = max(round(frame.t_mono * 1000.0), self._last_ms + 1)
        self._last_ms = ts
        return self._lm.detect_for_video(image, ts)

    def poses(self, frame: Frame) -> list[HandPose]:
        """Every detected hand with all 21 landmarks, most confident first."""
        h, w = frame.image.shape[:2]
        res = self._detect(frame)
        out = []
        for i, lms in enumerate(res.hand_landmarks):
            cat = res.handedness[i][0] if i < len(res.handedness) and res.handedness[i] else None
            img = np.array([[p.x * w, p.y * h] for p in lms], dtype=np.float64)
            world = np.array([[p.x, p.y, p.z] for p in res.hand_world_landmarks[i]], dtype=np.float64)
            out.append(HandPose(img, world, cat.category_name if cat else "?", cat.score if cat else 0.0))
        out.sort(key=lambda p: -p.score)
        return out

    def process(self, frame: Frame) -> HandObs:
        h, w = frame.image.shape[:2]
        res = self._detect(frame)
        hands = []
        for i, lms in enumerate(res.hand_landmarks):
            score = res.handedness[i][0].score if i < len(res.handedness) and res.handedness[i] else 0.0
            tip = lms[INDEX_TIP]
            # Normalized coords refer to the whole (resized) image, so they scale straight to camera px.
            xy = (min(max(tip.x, 0.0), 1.0) * w, min(max(tip.y, 0.0), 1.0) * h)
            hands.append((-score, i, xy))
        if not hands:
            return HandObs(present=False, tip_xy=None)
        tips = tuple(xy for _, _, xy in sorted(hands))
        return HandObs(present=True, tip_xy=tips[0], tips=tips)

    def close(self) -> None:
        self._lm.close()

    def __enter__(self) -> HandTracker:
        return self

    def __exit__(self, *exc) -> None:
        self.close()


Rect = tuple[int, int, int, int]  # x0, y0, x1, y1 camera px, end-exclusive


def tile_rects(region: Rect, *, max_tiles: int = 3, overlap: float = 0.3) -> list[Rect]:
    """Split ``region`` along its long side into at most ``max_tiles`` roughly square tiles sharing ``overlap`` of their
    length, so a hand crossing a seam is whole in one of them. A tile's long side sets how large a palm looks to the
    palm detector, which sees every input at 192 px."""
    x0, y0, x1, y1 = region
    w, h = x1 - x0, y1 - y0
    long, short = max(w, h), min(w, h)
    if short <= 0 or max_tiles < 1:
        return []
    n = max(1, min(max_tiles, round((long / short - overlap) / (1 - overlap))))
    size = long / (n - (n - 1) * overlap)
    step = size * (1 - overlap)
    out = []
    for i in range(n):
        a, b = round(i * step), (long if i == n - 1 else round(i * step + size))
        out.append((x0 + a, y0, x0 + b, y1) if w >= h else (x0, y0 + a, x1, y0 + b))
    return out


class TiledHandTracker:
    """One hand's index fingertip, searched in the whole frame and in overlapping tiles over a region of interest.

    MediaPipe finds palms on a 192 px copy of its input. In a 1920x1080 frame a hand on a wall 2 m from the camera then
    has a palm about 5 px wide and is mostly missed (1 Nov, wall: a hand in 15-50 % of the frames while someone used
    it, in one session almost never; ceiling mode, with the hand near the camera, tracked well). Tiles over ``region``
    (the projected page plus a margin) show the detector the palm about twice as large, and the landmarks come from a
    sharper crop. The whole frame stays a search window for hands too large for a tile.

    Every window has its own VIDEO-mode landmarker, because MediaPipe tracks a hand from the previous frame of the same
    input. That keeps the cost at one MediaPipe call per frame in the steady state: the window tracking a hand runs
    alone, and with no hand one window is searched per frame, in turn. When its window loses the hand (it moved into
    the next tile), the other windows are searched in the same frame. While the whole frame tracks a hand that is over
    a tile, that tile is tried every ``PROBE_EVERY`` frames and takes over once it sees the hand (sharper landmarks).
    """

    PROBE_EVERY = 3

    def __init__(self, region: Rect | None = None, *, max_tiles: int = 3, overlap: float = 0.3,
                 model_path: Path | None = None, min_confidence: float = 0.5, factory=None):
        self._factory = factory or (lambda: HandTracker(model_path, min_confidence=min_confidence))
        self.max_tiles, self.overlap = max_tiles, overlap
        self.tiles: list[Rect] = []
        self._trackers = [self._factory()]  # [0] searches the whole frame, [1 + k] tile k
        self.active: int | None = None  # window index tracking a hand
        self._next = 0
        self._frames = 0
        self._last_tip: tuple[float, float] | None = None
        self.set_region(region)

    def set_region(self, region: Rect | None) -> list[Rect]:
        """Tile ``region`` (None: whole frame only). Landmarkers are reused; returns the tiles."""
        self.tiles = tile_rects(region, max_tiles=self.max_tiles, overlap=self.overlap) if region is not None else []
        while len(self._trackers) < 1 + len(self.tiles):
            self._trackers.append(self._factory())
        while len(self._trackers) > 1 + len(self.tiles):
            self._trackers.pop().close()
        if self.active is not None and self.active >= len(self._trackers):
            self.active = None
        return self.tiles

    def _run(self, i: int, frame: Frame) -> HandObs:
        if i == 0:
            return self._trackers[0].process(frame)
        h, w = frame.image.shape[:2]
        x0, y0, x1, y1 = self.tiles[i - 1]
        x0, y0, x1, y1 = max(0, x0), max(0, y0), min(w, x1), min(h, y1)
        if x1 - x0 < 32 or y1 - y0 < 32:
            return HandObs(present=False, tip_xy=None)
        obs = self._trackers[i].process(Frame(frame.image[y0:y1, x0:x1], frame.t_mono, frame.idx))
        if not obs.present:
            return obs
        tips = tuple((x + x0, y + y0) for x, y in (obs.tips or (obs.tip_xy,)))
        return HandObs(present=True, tip_xy=tips[0], tips=tips)

    def _tile_at(self, xy: tuple[float, float]) -> int | None:
        """Window index of the tile holding ``xy`` farthest from its edges, or None."""
        best, depth = None, 0.0
        for k, (x0, y0, x1, y1) in enumerate(self.tiles):
            d = min(xy[0] - x0, x1 - xy[0], xy[1] - y0, y1 - xy[1])
            if d > depth:
                best, depth = 1 + k, d
        return best

    def _centre_dist(self, i: int, xy: tuple[float, float]) -> float:
        x0, y0, x1, y1 = self.tiles[i - 1]
        return math.dist(((x0 + x1) / 2, (y0 + y1) / 2), xy)

    def _found(self, i: int, obs: HandObs) -> HandObs:
        self.active = i
        self._last_tip = obs.tip_xy
        return obs

    def process(self, frame: Frame) -> HandObs:
        self._frames += 1
        n = len(self._trackers)
        if self.active is not None:
            obs = self._run(self.active, frame)
            if obs.present:
                if self.active == 0 and self._frames % self.PROBE_EVERY == 0:
                    k = self._tile_at(obs.tip_xy)
                    if k is not None:
                        tile_obs = self._run(k, frame)
                        if tile_obs.present:
                            return self._found(k, tile_obs)
                return self._found(self.active, obs)
            lost, self.active = self.active, None
            order = [i for i in range(1, n) if i != lost]
            if self._last_tip is not None:  # the tile around the last fingertip first, the whole frame last
                order.sort(key=lambda i: self._centre_dist(i, self._last_tip))
            for i in order + ([0] if lost != 0 else []):
                obs = self._run(i, frame)
                if obs.present:
                    return self._found(i, obs)
            return HandObs(present=False, tip_xy=None)
        i = self._next % n
        self._next = i + 1
        obs = self._run(i, frame)
        return self._found(i, obs) if obs.present else obs

    def close(self) -> None:
        for t in self._trackers:
            t.close()
        self._trackers = []


class RegionHandTracker:
    """Up to ``num_hands`` hands (default two) with all landmarks, searched in a crop of the frame (``lightwork play
    --table``: the projected page plus a margin).

    One VIDEO-mode landmarker with ``num_hands=2``: while fewer hands than that are tracked, MediaPipe runs its palm
    detector on every frame (17 ms p50 / 25 ms p95 at 1080p, see :class:`HandTracker`); tracked hands take the cheap
    landmark path. The palm detector sees every input at 192 px, so cropping the frame to the page shows it larger
    palms: a page half the frame wide doubles them. The crop is not reset when the region moves (a recalibration):
    MediaPipe re-finds a hand whose tracked position no longer fits within a frame.
    """

    def __init__(self, region: Rect | None = None, *, num_hands: int = 2, model_path: Path | None = None,
                 min_confidence: float = 0.5, factory=None):
        make = factory or (lambda: HandTracker(model_path, num_hands=num_hands, min_confidence=min_confidence))
        self._tracker = make()
        self.region: Rect | None = None
        self.set_region(region)

    def set_region(self, region: Rect | None) -> list[Rect]:
        """Crop to ``region`` (None: the whole frame). Returns the search windows, for reporting."""
        self.region = region
        return [region] if region is not None else []

    def poses(self, frame: Frame) -> list[HandPose]:
        """Every hand found, camera px, most confident first."""
        if self.region is None:
            return self._tracker.poses(frame)
        h, w = frame.image.shape[:2]
        x0, y0, x1, y1 = self.region
        x0, y0, x1, y1 = max(0, x0), max(0, y0), min(w, x1), min(h, y1)
        if x1 - x0 < 32 or y1 - y0 < 32:
            return []
        crop = Frame(frame.image[y0:y1, x0:x1], frame.t_mono, frame.idx)
        return [p.shifted(x0, y0) for p in self._tracker.poses(crop)]

    def process(self, frame: Frame) -> HandObs:
        poses = self.poses(frame)
        if not poses:
            return HandObs(present=False, tip_xy=None)
        tips = tuple(p.tip for p in poses)
        return HandObs(present=True, tip_xy=tips[0], tips=tips)

    def close(self) -> None:
        self._tracker.close()


TILE_PX = 560  # search tile side, camera px: a 140 px palm shows about 48 px wide to the 192 px palm detector


def grid_tiles(region: Rect, *, tile: int = TILE_PX, overlap: float = 0.3) -> list[Rect]:
    """Tiles of about ``tile`` px in a grid over ``region``, neighbours sharing ``overlap`` of their side, so a palm
    crossing a seam is whole in one of them."""
    x0, y0, x1, y1 = region
    w, h = x1 - x0, y1 - y0
    if w <= 0 or h <= 0:
        return []

    def split(length: int) -> list[tuple[int, int]]:
        n = max(1, math.ceil((length - overlap * tile) / ((1 - overlap) * tile)))
        if n == 1:
            return [(0, length)]
        size = length / (n - (n - 1) * overlap)
        step = size * (1 - overlap)
        return [(round(i * step), length if i == n - 1 else round(i * step + size)) for i in range(n)]

    return [(x0 + a, y0 + b, x0 + c, y0 + d) for b, d in split(h) for a, c in split(w)]


def hand_centre(img: np.ndarray) -> np.ndarray:
    """Centre of the landmarks' bounding box, image px."""
    return (img.min(axis=0) + img.max(axis=0)) / 2


@dataclass(eq=False)
class _Follower:
    lm: object  # VIDEO-mode landmarker, one hand
    active: bool = False
    centre: np.ndarray | None = None  # where the hand is expected in the next frame (crop centre), camera px
    last: np.ndarray | None = None  # hand centre last seen, camera px
    vel: np.ndarray | None = None  # hand centre motion, px per frame (smoothed)
    side: int = 0  # crop side, camera px
    misses: int = 0
    age: int = 0  # frames followed


def _hits(tile, boxes) -> bool:
    x0, y0, x1, y1 = tile
    return any(bx0 < x1 and bx1 > x0 and by0 < y1 and by1 > y0 for bx0, by0, bx1, by1 in boxes)


def _hit_area(tile, boxes) -> float:
    x0, y0, x1, y1 = tile
    return sum(max(0, min(x1, bx1) - max(x0, bx0)) * max(0, min(y1, by1) - max(y0, by0)) for bx0, by0, bx1, by1 in boxes)


class MotionBoxes:
    """Where the camera image changed since the last frame: boxes in camera px from a tiny grey difference image
    (1/8 scale, under a millisecond at 1080p). Hints for :meth:`CropHandTracker.poses`."""

    def __init__(self, down: int = 8, threshold: int = 14, min_px: int = 3):
        self.down, self.threshold, self.min_px = down, threshold, min_px
        self.prev: np.ndarray | None = None

    def __call__(self, image: np.ndarray) -> list[tuple[int, int, int, int]] | None:
        d = self.down
        small = cv2.resize(image, (image.shape[1] // d, image.shape[0] // d), interpolation=cv2.INTER_AREA)
        g = cv2.GaussianBlur(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY), (3, 3), 0)
        prev, self.prev = self.prev, g
        if prev is None or prev.shape != g.shape:
            return None
        m = (cv2.absdiff(g, prev) > self.threshold).astype(np.uint8)
        m = cv2.dilate(m, np.ones((3, 3), np.uint8))
        n, _, stats, _ = cv2.connectedComponentsWithStats(m)
        return [(int(x * d), int(y * d), int((x + w) * d), int((y + h) * d))
                for x, y, w, h, area in stats[1:] if area >= self.min_px]


class CropHandTracker:
    """Up to ``num_hands`` hands at full camera resolution (``lightwork play --table``).

    MediaPipe finds a hand by first spotting its palm in a 192 x 192 copy of its input. A whole table seen by a 1080p
    camera shows a palm about 16 px wide there (2 Nov, :class:`RegionHandTracker` on the page crop: "palms 1.2x larger
    to the detector", a hand found in 5-20 % of the frames while objects were put down). So:

    - Search: overlapping full-resolution tiles of about ``TILE_PX`` over the region (the projected page plus a
      margin), in MediaPipe IMAGE mode, ``search_per_frame`` tiles per frame in turn while a hand slot is free. A palm
      shows about three times larger to the detector than in the page crop. Tiles near a hand lost a moment ago go
      first.
    - Follow: each hand found gets its own VIDEO-mode landmarker, fed a square crop ``CROP_PALMS`` palm lengths wide,
      centred where the hand is expected (last centre plus last motion). The hand fills the crop, so its landmarks
      come from full-resolution pixels, and MediaPipe's own tracking carries over: the crop moves with the hand, so
      the hand stays where MediaPipe last saw it in crop coordinates. ``LOST_FRAMES`` misses in a row free the slot.

    Steady state: one landmark pass per hand (MediaPipe skips palm detection while it tracks), plus the search while a
    slot is free. Not thread-safe; call from one thread.
    """

    CROP_PALMS = 4.5
    CROP_MIN, CROP_MAX = 288, 720
    LOST_FRAMES = 3
    GROW_ON_MISS = 1.25  # the follow crop grows this much per missed frame
    IDLE_SEARCH_EVERY = 6  # with motion hints and nothing moving: one round-robin search turn every this many frames
    motion_aware = True  # poses() takes ``motion`` boxes
    SAME_HAND = 0.6  # palm lengths: wrists (or hand centres) closer than this are one hand
    LOST_FIRST_S = 1.0
    keep = None  # callable(HandPose) -> bool: hands it refuses are neither followed nor returned (they would take a
    # hand slot: 3 Nov, a plush toy beside the table was "a hand" at 0.5-0.99 confidence, and only one real hand fit)

    def __init__(self, region: Rect | None = None, *, num_hands: int = 2, search_per_frame: int = 2,
                 tile: int = TILE_PX, model_path: Path | None = None, min_confidence: float = 0.5, factory=None,
                 search_factory=None, gpu: bool = False, upside_down: bool = False,
                 track_confidence: float | None = None):
        """``upside_down``: the person sits opposite the camera, so hands come in fingers down: the models are shown
        every crop turned 180 degrees, hands upright (4 Nov, real footage: the longest dropout 11 -> 7 frames)."""
        follow = factory or (lambda: HandTracker(model_path, num_hands=1, min_confidence=min_confidence, infer_w=None,
                                                 gpu=gpu, track_confidence=track_confidence))
        search = search_factory or (lambda: HandTracker(model_path, num_hands=2, min_confidence=min_confidence,
                                                         infer_w=None, image_mode=True, gpu=gpu))
        self.upside_down = upside_down
        self._followers = [_Follower(follow()) for _ in range(num_hands)]
        self._search = search()
        self.search_per_frame = search_per_frame
        self.tile = tile
        self.region: Rect | None = None
        self.tiles: list[Rect] = []
        self._tiles_for: tuple[int, int] | None = None  # frame size the whole-frame tiles were made for
        self._next = 0
        self._lost: list[tuple[float, np.ndarray]] = []  # (t, centre) of hands lost lately
        self.searched = 0  # tile searches, for status lines
        self.set_region(region)

    def set_region(self, region: Rect | None) -> list[Rect]:
        """Search tiles over ``region`` (None: the whole frame). Followers keep their hands. Returns the tiles."""
        self.region = region
        self.tiles = grid_tiles(region, tile=self.tile) if region is not None else []
        self._tiles_for = None
        self._next = 0
        return self.tiles

    @property
    def delegate(self) -> str:
        """"GPU", "CPU" or "GPU+CPU" (what the search and the followers got)."""
        kinds = {getattr(t, "delegate", "CPU") for t in [self._search] + [f.lm for f in self._followers]}
        return "+".join(sorted(kinds, reverse=True))

    @property
    def following(self) -> int:
        return sum(f.active for f in self._followers)

    def poses(self, frame: Frame, motion: list | None = None) -> list[HandPose]:
        """Every hand, camera px, most confident first. ``motion``: camera-px boxes where the image changed since the
        last frame; a new hand is searched for there first, and while nothing moves only now and then (3 Nov games:
        searching 4 tiles every frame cost 5 fps with no hand in view)."""
        h, w = frame.image.shape[:2]
        if self.region is None and self._tiles_for != (w, h):
            self.tiles, self._tiles_for = grid_tiles((0, 0, w, h), tile=self.tile), (w, h)
        out: list[tuple[_Follower, HandPose]] = []
        for f in self._followers:
            if not f.active:
                continue
            p = self._follow(f, frame)
            if p is not None and self.keep is not None and not self.keep(p):
                self._release(f, frame.t_mono, lost=False)
                continue
            if p is not None:
                out.append((f, p))
            else:
                f.misses += 1
                if f.vel is not None and f.centre is not None:  # a fast, blurred hand: look where it was going,
                    f.centre = f.centre + f.vel  # in a wider crop (3 Nov games: a moving hand was found in 5-35 %
                if f.side:  # of frames, a still one in 100 %)
                    f.side = int(min(f.side * self.GROW_ON_MISS, self.CROP_MAX))
                if f.misses >= self.LOST_FRAMES:
                    self._release(f, frame.t_mono, lost=True)
        out = self._dedupe(out)
        free = [f for f in self._followers if not f.active]
        if free and self.tiles:
            order, rr = self._search_order(frame.t_mono)
            if motion is not None:
                lost_first = [k for k in order if k not in rr]
                moving = sorted((k for k in range(len(self.tiles)) if _hits(self.tiles[k], motion)),
                                key=lambda k: -_hit_area(self.tiles[k], motion))
                self._idle = getattr(self, "_idle", 0) + 1
                if moving or lost_first or self._idle % self.IDLE_SEARCH_EVERY == 0:
                    first = lost_first + [k for k in moving if k not in lost_first]
                    order = first + [k for k in order if k not in first]
                else:
                    order = []  # nothing moved and no hand was just lost: no new hand to look for
            searched_rr = 0
            budget = max(1, self.search_per_frame - len(out))  # each followed hand already costs a landmark pass
            for k in order[:budget]:
                searched_rr += k in rr
                for p in self._search_tile(self.tiles[k], frame):
                    if self.keep is not None and not self.keep(p):
                        continue
                    if free and not any(self._same(p, q) for _, q in out):
                        f = free.pop(0)
                        self._start(f, p)
                        out.append((f, p))
            self._next = (self._next + searched_rr) % len(self.tiles)
        return [p for _, p in sorted(out, key=lambda fp: -fp[1].score)]

    def process(self, frame: Frame) -> HandObs:
        poses = self.poses(frame)
        if not poses:
            return HandObs(present=False, tip_xy=None)
        tips = tuple(p.tip for p in poses)
        return HandObs(present=True, tip_xy=tips[0], tips=tips)

    def close(self) -> None:
        for f in self._followers:
            f.lm.close()
        self._search.close()
        self._followers = []

    # ---- internals

    @staticmethod
    def _square(centre: np.ndarray, side: int, w: int, h: int) -> tuple[int, int, int]:
        s = max(1, min(side, w, h))
        x0 = round(min(max(float(centre[0]) - s / 2, 0.0), w - s))
        y0 = round(min(max(float(centre[1]) - s / 2, 0.0), h - s))
        return x0, y0, s

    def _follow(self, f: _Follower, frame: Frame) -> HandPose | None:
        h, w = frame.image.shape[:2]
        x0, y0, s = self._square(f.centre, f.side, w, h)
        poses = self._look(f.lm, frame.image[y0:y0 + s, x0:x0 + s], frame)
        if not poses:
            return None
        p = poses[0].shifted(x0, y0)
        self._seen(f, p)
        return p

    def _look(self, tracker, crop: np.ndarray, frame: Frame) -> list[HandPose]:
        """Poses in ``crop``, crop px; the model sees it upright when hands come in upside down."""
        if not self.upside_down:
            return tracker.poses(Frame(crop, frame.t_mono, frame.idx))
        h, w = crop.shape[:2]
        found = tracker.poses(Frame(cv2.rotate(crop, cv2.ROTATE_180), frame.t_mono, frame.idx))
        turn = np.array([-1.0, -1.0, 1.0])
        return [HandPose((w, h) - p.img, p.world * turn, p.side, p.score) for p in found]

    def _seen(self, f: _Follower, p: HandPose) -> None:
        c = hand_centre(p.img)
        step = np.zeros(2) if f.last is None else c - f.last
        f.vel = step if f.vel is None else 0.5 * f.vel + 0.5 * step
        f.last, f.centre = c, c + f.vel
        want = int(min(max(self.CROP_PALMS * palm_px(p.img), self.CROP_MIN), self.CROP_MAX))
        if f.side == 0 or abs(want - f.side) > 0.3 * f.side:  # steady crop size: MediaPipe's ROI is relative to it
            f.side = want
        f.misses = 0
        f.age += 1

    def _start(self, f: _Follower, p: HandPose) -> None:
        f.active, f.last, f.vel, f.side, f.age = True, None, None, 0, 0
        self._seen(f, p)

    def _release(self, f: _Follower, t: float, *, lost: bool) -> None:
        if lost and f.last is not None:
            self._lost.append((t, f.last.copy()))
        f.active, f.centre, f.last, f.vel, f.misses, f.age = False, None, None, None, 0, 0

    def _same(self, p: HandPose, q: HandPose) -> bool:
        palm = max(palm_px(p.img), palm_px(q.img), 1.0)
        near = self.SAME_HAND * palm
        return (math.dist(p.img[0], q.img[0]) < near
                or float(np.linalg.norm(hand_centre(p.img) - hand_centre(q.img))) < near)

    def _dedupe(self, out: list[tuple[_Follower, HandPose]]) -> list[tuple[_Follower, HandPose]]:
        """One hand followed twice (hands crossed, or a follower jumped to the other hand): the older follower keeps it."""
        keep: list[tuple[_Follower, HandPose]] = []
        for f, p in sorted(out, key=lambda fp: -fp[0].age):
            if any(self._same(p, q) for _, q in keep):
                self._release(f, 0.0, lost=False)
            else:
                keep.append((f, p))
        return keep

    def _search_order(self, t: float) -> tuple[list[int], set[int]]:
        """Tile indices to search, best first, and which of them are round-robin turns."""
        n = len(self.tiles)
        self._lost = [(tl, c) for tl, c in self._lost if t - tl <= self.LOST_FIRST_S]
        first: list[int] = []
        for _, c in reversed(self._lost):
            best, depth = None, 0.0
            for k, (x0, y0, x1, y1) in enumerate(self.tiles):
                d = min(c[0] - x0, x1 - c[0], c[1] - y0, y1 - c[1])
                if d > depth:
                    best, depth = k, d
            if best is not None and best not in first:
                first.append(best)
        rr = [(self._next + i) % n for i in range(n)]
        rr = [k for k in rr if k not in first]
        return first + rr, set(rr)

    def _search_tile(self, tile: Rect, frame: Frame) -> list[HandPose]:
        h, w = frame.image.shape[:2]
        x0, y0, x1, y1 = max(0, tile[0]), max(0, tile[1]), min(w, tile[2]), min(h, tile[3])
        if x1 - x0 < 32 or y1 - y0 < 32:
            return []
        side = max(x1 - x0, y1 - y0)  # square: MediaPipe's rotated hand box is only right on a square input
        x0, y0, s = self._square(np.array([(x0 + x1) / 2, (y0 + y1) / 2]), side, w, h)
        self.searched += 1
        found = self._look(self._search, frame.image[y0:y0 + s, x0:x0 + s], frame)
        return [p.shifted(x0, y0) for p in found]


class HandEvents:
    """Observations -> C1 ``hands`` events, only on change and at most ``max_hz`` per second (see module doc)."""

    def __init__(self, regions: Regions, *, max_hz: float = 5.0, min_move_px: float = 3.0, exit_margin_px: float = 6.0, absent_s: float = 0.2):
        self._polys = {ref: regions.polygon(ref).astype(np.float32) for ref in regions.all_refs()}
        self._min_dt = 1.0 / max_hz
        self._min_move = min_move_px
        self._exit_margin = exit_margin_px
        self._absent_s = absent_s
        self._over: frozenset[str] = frozenset()  # current state (hysteresis applies to it)
        self._tip: tuple[float, float] | None = None
        self._last_seen: float | None = None
        self._sent_over: frozenset[str] = frozenset()
        self._sent_tip: tuple[float, float] | None = None
        self._sent_t: float | None = None

    def _regions_for(self, tips: tuple[tuple[float, float], ...]) -> frozenset[str]:
        out = set()
        for ref, poly in self._polys.items():
            held = ref in self._over
            for x, y in tips:
                if held:
                    inside = cv2.pointPolygonTest(poly, (float(x), float(y)), True) >= -self._exit_margin
                else:
                    inside = cv2.pointPolygonTest(poly, (float(x), float(y)), False) >= 0
                if inside:
                    out.add(ref)
                    break
        return frozenset(out)

    def update(self, obs: HandObs, t: float) -> list[tuple[str, dict]]:
        if obs.present and obs.tip_xy is not None:
            self._last_seen = t
            self._over = self._regions_for(obs.tips or (obs.tip_xy,))
            self._tip = (q_px(obs.tip_xy[0]), q_px(obs.tip_xy[1]))
        elif self._tip is not None and self._last_seen is not None and t - self._last_seen < self._absent_s:
            pass  # brief tracking dropout: hold the last state
        else:
            self._over = frozenset()
            self._tip = None

        if self._sent_t is not None and t - self._sent_t < self._min_dt:
            return []
        if self._over == self._sent_over and (self._tip is None) == (self._sent_tip is None):
            if self._tip is None or math.dist(self._tip, self._sent_tip) < self._min_move:
                return []
        self._sent_over, self._sent_tip, self._sent_t = self._over, self._tip, t
        return [("hands", {"over_regions": sorted(self._over), "tip_xy": list(self._tip) if self._tip is not None else None})]


def _h1(argv: list[str] | None = None) -> int:
    """rig:H1: click the N target centres (projected rings or stickers), then touch each in order; >= 90 % to pass."""
    from ..contracts.layout import Layout, LayoutObject, SamPrompt
    from ..contracts.stepgraph import Source, StepGraph
    from .capture import open_camera

    ap = argparse.ArgumentParser(prog="python -m lightwork.vision.hands", description=_h1.__doc__)
    ap.add_argument("--camera", default=None, help="camera spec for open_camera (default: C920 by-id)")
    ap.add_argument("--targets", type=int, default=10)
    ap.add_argument("--radius", type=float, default=30.0, help="target radius, camera px")
    ap.add_argument("--dwell", type=float, default=0.5, help="seconds the tip must rest before a touch counts")
    args = ap.parse_args(argv)

    tracker = HandTracker()
    cam = open_camera(args.camera)
    win = "rig:H1"
    clicks: list[tuple[float, float]] = []
    scale = 1.0

    def on_mouse(ev, x, y, *_):
        if ev == cv2.EVENT_LBUTTONDOWN and len(clicks) < args.targets:
            clicks.append((x / scale, y / scale))

    def show(img, lines, tip=None, ring=None):
        disp = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        for k, (cx, cy) in enumerate(clicks):
            colour = (0, 255, 255) if ring == k else (0, 200, 0)
            cv2.circle(disp, (round(cx * scale), round(cy * scale)), round(args.radius * scale), colour, 2)
            cv2.putText(disp, f"t{k + 1}", (round(cx * scale) + 6, round(cy * scale) - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, colour, 2)
        if tip is not None:
            cv2.circle(disp, (round(tip[0] * scale), round(tip[1] * scale)), 6, (0, 0, 255), -1)
        for j, line in enumerate(lines):
            cv2.putText(disp, line, (10, 30 + 28 * j), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.imshow(win, disp)
        return cv2.waitKey(1) & 0xFF

    try:
        cv2.namedWindow(win)
        cv2.setMouseCallback(win, on_mouse)
        while len(clicks) < args.targets:
            f = cam.read()
            if f is None:
                return 2
            scale = min(1.0, 960.0 / f.image.shape[1])
            if show(f.image, [f"click target centres {len(clicks)}/{args.targets}  (q quits)"]) == ord("q"):
                return 2
        circle = [(math.cos(2 * math.pi * k / 24), math.sin(2 * math.pi * k / 24)) for k in range(24)]
        layout = Layout(
            pack="rig-h1",
            created="1970-01-01T00:00:00Z",
            work_area_xywh=[0, 0, f.image.shape[1], f.image.shape[0]],
            objects={
                f"t{k + 1}": LayoutObject(kind="zone", polygon_cam_px=[[cx + args.radius * a, cy + args.radius * b] for a, b in circle], sam=SamPrompt(mode="noun", noun="target"))
                for k, (cx, cy) in enumerate(clicks)
            },
        )
        sg = StepGraph(id="rig-h1", title="rig:H1", source=Source(kind="text", ref="rig:H1", licence="Apache-2.0", attribution=""), objects=[], steps=[])
        events = HandEvents(Regions(layout, sg))
        hits = 0
        for k in range(args.targets):
            want = f"t{k + 1}"
            over: list[str] = []
            tip = None
            since: float | None = None
            need_clear = True  # the previous touch must end before this one counts
            while True:
                f = cam.read()
                if f is None:
                    return 2
                for _, p in events.update(tracker.process(f), f.t_mono):
                    over, tip = p["over_regions"], p["tip_xy"]
                    since = f.t_mono if over else None
                if need_clear:
                    need_clear = bool(over)
                elif over and since is not None and f.t_mono - since >= args.dwell:
                    break
                key = show(f.image, [f"touch {want}  ({k}/{args.targets} done, {hits} correct)", f"over: {', '.join(over) or '-'}"], tip, ring=k)
                if key == ord("q"):
                    return 2
            ok = want in over
            hits += ok
            print(f"{want}: over={over} {'OK' if ok else 'WRONG'}", flush=True)
        passed = hits >= math.ceil(0.9 * args.targets)
        print(f"rig:H1 {hits}/{args.targets} {'PASS' if passed else 'FAIL'}")
        return 0 if passed else 1
    finally:
        tracker.close()
        cam.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    sys.exit(_h1())
