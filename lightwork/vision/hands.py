"""Hands (D12, advisory only): MediaPipe HandLandmarker -> index fingertip in camera px -> C1 ``hands`` events.

``HandTracker`` runs MediaPipe 1.x HandLandmarker in VIDEO mode at 640 px wide (640x360 for 16:9 frames). MediaPipe
is in the optional ``live`` extra and is imported lazily; replay, verify and export never touch this class.

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


class HandTracker:
    """MediaPipe HandLandmarker (VIDEO mode). Not thread-safe: call ``process`` from one worker thread.

    ``num_hands=1`` keeps a tracked hand on the cheap landmark path (8.5 ms p50 / 9.7 ms p95 per 1080p frame incl.
    the resize, Ryzen 7 7840HS, 29 Oct). With ``num_hands=2`` and one hand in view MediaPipe re-runs palm detection
    on every frame looking for the second hand: 17 ms p50 / 25 ms p95, over the 15 ms budget.
    """

    def __init__(self, model_path: Path | None = None, *, num_hands: int = 1, min_confidence: float = 0.5):
        mp, vision, BaseOptions = _import_mediapipe()
        path = ensure_model(Path(model_path) if model_path is not None else default_model_path())
        opts = vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(path)),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=num_hands,
            min_hand_detection_confidence=min_confidence,
            min_hand_presence_confidence=min_confidence,
            min_tracking_confidence=min_confidence,
        )
        self._mp = mp
        self._lm = vision.HandLandmarker.create_from_options(opts)
        self._last_ms = -1

    def process(self, frame: Frame) -> HandObs:
        img = frame.image
        h, w = img.shape[:2]
        ih = max(1, round(h * INFER_W / w))
        small = img if w == INFER_W else cv2.resize(img, (INFER_W, ih), interpolation=cv2.INTER_AREA)
        rgb = np.ascontiguousarray(cv2.cvtColor(small, cv2.COLOR_BGR2RGB))
        # VIDEO mode requires strictly increasing integer-ms timestamps.
        ts = max(round(frame.t_mono * 1000.0), self._last_ms + 1)
        self._last_ms = ts
        res = self._lm.detect_for_video(self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb), ts)
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
