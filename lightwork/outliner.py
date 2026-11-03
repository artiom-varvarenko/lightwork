"""SAM 3.1 outlines for ``lightwork play --table`` (``--sam``): the projected outline follows the object's mask.

After a scan finds objects, the clean view (our light off) goes to SAM 3.1 once, cropped to the objects, with one box
prompt per object (``models.requests`` "segment" role, at most ``MAX_OBJECTS`` per request). Each returned mask's
outer contour, in camera px, replaces the outline drawn from the change blob, which also holds shadows and misses
low-contrast parts. Calls go through :class:`lightwork.models.client.ModelClient`: recorded under
``.lightwork/play-calls/<session>-sam/`` and held to the ``LW_BUDGET_USD`` cap ($2.50 per 1,000 images).

``request`` and ``result`` never block; a worker thread makes the calls.
"""

from __future__ import annotations

import queue
import threading
import time
from collections import deque
from datetime import UTC, datetime
from pathlib import Path

import cv2
import numpy as np

SAM_MODEL = "sam-3.1"
TIMEOUT_S = 8.0
MARGIN = 0.15  # the crop sent reaches this share of its size past the objects' boxes
JPEG_QUALITY = 90


class SamOutliner:
    label = SAM_MODEL

    def __init__(self, settings, *, run_dir: Path | None = None, client=None, model: str = SAM_MODEL):
        from .models.budget import Budget
        from .models.client import ModelClient

        self.model = model
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        self.run_dir = Path(run_dir) if run_dir is not None else settings.state_dir / "play-calls" / f"{stamp}-sam"
        self._settings = settings
        self._client = client or ModelClient(settings, run_dir=self.run_dir, provenance="meta",
                                             budget=Budget.from_settings(settings), cache=None)
        self._q: queue.Queue = queue.Queue(maxsize=4)
        self._done: dict[int, np.ndarray | None] = {}
        self._lock = threading.Lock()
        self._index = 0
        self.error: str | None = None
        self.calls = 0
        self.cost_usd = 0.0
        self.ms: deque[float] = deque(maxlen=20)
        self._thread = threading.Thread(target=self._loop, name="sam-outliner", daemon=True)
        self._thread.start()

    def check(self) -> str | None:
        return None if self._settings.model_api_key else "MODEL_API_KEY is not set (.env)"

    def request(self, img: np.ndarray, objects: list[tuple[int, tuple[int, int, int, int]]]) -> bool:
        """Queue one SAM call for ``objects`` [(key, half-open camera-px box)] on the clean view ``img``."""
        from .models.prompts.segment import MAX_OBJECTS

        objects = objects[:MAX_OBJECTS]
        if not objects:
            return False
        h, w = img.shape[:2]
        boxes = np.array([b for _, b in objects], dtype=np.float64)
        x0, y0 = boxes[:, :2].min(axis=0)
        x1, y1 = boxes[:, 2:].max(axis=0)
        mx, my = MARGIN * (x1 - x0) + 16, MARGIN * (y1 - y0) + 16
        cx0, cy0 = int(max(0, x0 - mx)), int(max(0, y0 - my))
        cx1, cy1 = int(min(w, x1 + mx)), int(min(h, y1 + my))
        if cx1 - cx0 < 16 or cy1 - cy0 < 16:
            return False
        ok, buf = cv2.imencode(".jpg", img[cy0:cy1, cx0:cx1], [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
        if not ok:
            return False
        prompts = []
        for key, (bx0, by0, bx1, by1) in objects:
            box = [max(0, bx0 - cx0), max(0, by0 - cy0), min(cx1 - cx0, bx1 - cx0), min(cy1 - cy0, by1 - cy0)]
            if box[2] > box[0] and box[3] > box[1]:
                prompts.append({"id": int(key), "box": [int(v) for v in box]})
        try:
            self._q.put_nowait((buf.tobytes(), (cx0, cy0, cx1 - cx0, cy1 - cy0), prompts))
            return True
        except queue.Full:
            return False

    def result(self, key: int) -> tuple[bool, np.ndarray | None]:
        """(finished, outer contour in camera px or None)."""
        with self._lock:
            if key in self._done:
                return True, self._done.pop(key)
        return False, None

    def summary(self) -> str:
        ms = f", {sum(self.ms) / len(self.ms) / 1000:.1f} s per call" if self.ms else ""
        return f"{self.label}: {self.calls} call{'' if self.calls == 1 else 's'}{ms}, ${self.cost_usd:.4f} this session"

    def _segment(self, jpeg: bytes, crop_xywh, prompts: list[dict]) -> dict[int, np.ndarray]:
        from .contracts.canonical import sha256_hex
        from .core.masks import to_camera
        from .models.parse import parse_response
        from .models.pricing import cost_usd
        from .models.requests import build_request

        sha = sha256_hex(jpeg)
        frames = self.run_dir / "frames"
        frames.mkdir(parents=True, exist_ok=True)
        (frames / f"{sha}.jpg").write_bytes(jpeg)
        endpoint, request = build_request("segment", model=self.model, backend="meta", frame_sha=sha,
                                          ctx={"object_prompts": prompts, "w": crop_xywh[2], "h": crop_xywh[3]})
        self._index += 1
        result = self._client.execute(intent_id=f"play-outline-{self._index}", role="segment", endpoint=endpoint,
                                      request=request, frames={sha: jpeg}, index=self._index, timeout_s=TIMEOUT_S)
        self.cost_usd += cost_usd(self.model, result.get("usage"))
        body = next((self.run_dir / result["call_dir"]).glob("response.*"))
        parsed = parse_response("segment", body.read_bytes(), body.suffix[1:], endpoint)
        if parsed["status"] != "ok":
            raise RuntimeError(f"{parsed['status']}: {parsed['error']}")
        out: dict[int, np.ndarray] = {}
        for m in parsed["result"]["masks"]:
            cam = to_camera(m, crop_xywh)
            cs, _ = cv2.findContours(cam.bits.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not cs:
                continue
            big = max(cs, key=cv2.contourArea).reshape(-1, 2).astype(np.float64)
            out[cam.object_id] = big + (cam.box[0], cam.box[1])
        return out

    def _loop(self) -> None:
        while True:
            item = self._q.get()
            if item is None:
                return
            jpeg, crop, prompts = item
            t = time.perf_counter()
            try:
                found = self._segment(jpeg, crop, prompts)
                self.error = None
            except Exception as exc:  # noqa: BLE001 - a failed outline keeps the blob's; never stop the worker
                found, self.error = {}, f"{type(exc).__name__}: {exc}"
            self.calls += 1
            self.ms.append((time.perf_counter() - t) * 1000.0)
            with self._lock:
                for p in prompts:
                    self._done[p["id"]] = found.get(p["id"])

    def close(self) -> None:
        try:
            self._q.put_nowait(None)
        except queue.Full:
            pass
        self._thread.join(timeout=2.0)
        self._client.close()
