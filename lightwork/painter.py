"""Muse Image: a picture for the table.

On the table black is no light, so a picture is asked for in bright lines on black and what comes back is tidied for a
100 lumen projector: a dark grey "black" still lights the table, so the black point is pulled down. Nothing else is done
to it. Every call is recorded like the other models' (``.lightwork/play-calls/<session>-image/calls/<n>-image-.../``:
request, response with the picture replaced by its hash, meta) with the picture as sent back under ``frames/``, is held
to the ``LW_BUDGET_USD`` cap, and counts in ``spend.json``.
"""

from __future__ import annotations

import base64
import hashlib
import json
import threading
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urlsplit

import cv2
import httpx
import numpy as np

MODEL = "muse-image-1.0"
ENDPOINT = "/v1/images/generations"
SIZE = "1536x1024"  # a shape, not a resolution: the model keeps its own
SERVED_W = 1024  # px wide on the page: the projector has 854
BLACK_POINT = 28  # levels below this are "black" that would light the table
STYLE = (" Draw it in bright, thick lines and bright flat colours on a pure black background, like a chalk diagram on a "
         "blackboard, with short English labels in large letters. The background must be pure black everywhere: no "
         "border, no frame, no paper, no shading, no title bar.")


class PainterError(RuntimeError):
    pass


@dataclass(frozen=True)
class Painted:
    src: str  # the URL a C2 ``picture`` prim loads it from
    width: int
    height: int
    model: str
    provider: str
    ms: float
    cost_usd: float


def for_the_table(png: bytes) -> tuple[bytes, int, int]:
    """The picture as the page gets it: at most SERVED_W wide, its black no light at all."""
    img = cv2.imdecode(np.frombuffer(png, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise PainterError("the model's picture cannot be decoded")
    h, w = img.shape[:2]
    if w > SERVED_W:
        img = cv2.resize(img, (SERVED_W, round(h * SERVED_W / w)), interpolation=cv2.INTER_AREA)
    lifted = (img.astype(np.float32) - BLACK_POINT) * (255.0 / (255 - BLACK_POINT))
    img = np.clip(lifted, 0, 255).astype(np.uint8)
    ok, buf = cv2.imencode(".png", img)
    if not ok:
        raise PainterError("the picture cannot be encoded")
    return buf.tobytes(), img.shape[1], img.shape[0]


class Painter:
    label = MODEL

    def __init__(self, settings, *, run_dir: Path | None = None, pictures_dir: Path | None = None,
                 transport: httpx.BaseTransport | None = None):
        from .models.budget import Budget

        self.settings = settings
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        self.run_dir = Path(run_dir) if run_dir is not None else settings.state_dir / "play-calls" / f"{stamp}-image"
        self.pictures_dir = Path(pictures_dir) if pictures_dir is not None else settings.state_dir / "pictures"
        self.budget = Budget.from_settings(settings)
        self.calls = 0
        self.cost_usd = 0.0
        self.ms: list[float] = []
        self._http = httpx.Client(transport=transport, trust_env=False)
        self._lock = threading.Lock()

    def check(self) -> str | None:
        return None if self.settings.model_api_key else "MODEL_API_KEY is not set (.env)"

    def paint(self, prompt: str, *, timeout_s: float = 90.0) -> Painted:
        from .contracts.calls import CallMeta, frame_ref, request_fingerprint
        from .models.pricing import IMAGE_USD
        from .models.record import write_call

        request = {"model": MODEL, "prompt": " ".join(prompt.split()) + STYLE, "n": 1, "size": SIZE,
                   "response_format": "b64_json", "output_format": "png", "reasoning_strength": "low",
                   "tool_enablement": {"enable_image_search": False, "enable_web_search": False, "enable_shell": False}}
        reserved = self.budget.reserve(MODEL, request)
        base = self.settings.meta_api_base
        t0 = time.perf_counter()
        status, code, raw, kept, cost, picture = "error", None, b"{}", None, 0.0, None
        try:
            r = self._http.post(f"{base}/images/generations", json=request, timeout=timeout_s,
                                headers={"Authorization": f"Bearer {self.settings.model_api_key}"})
            code = r.status_code
            raw = r.content
            if code == 200:
                body = r.json()
                item = (body.get("data") or [{}])[0]
                if item.get("b64_json"):
                    picture = base64.b64decode(item["b64_json"])
                    status, cost = "ok", IMAGE_USD
                    sha = hashlib.sha256(picture).hexdigest()
                    frames = self.run_dir / "frames"
                    frames.mkdir(parents=True, exist_ok=True)
                    (frames / f"{sha}.png").write_bytes(picture)
                    kept = {**body, "data": [{**item, "b64_json": frame_ref(sha)}]}
                    raw = json.dumps(kept).encode()
        except httpx.TimeoutException:
            status = "timeout"
        except httpx.HTTPError as exc:
            raw = json.dumps({"error": str(exc)}).encode()
        finally:
            self.budget.settle(reserved, cost if status == "ok" else (None if status == "timeout" else 0.0))
        ms = (time.perf_counter() - t0) * 1000.0
        with self._lock:
            self.calls += 1
            index = self.calls
            self.cost_usd += cost
            self.ms.append(ms)
        meta = CallMeta(backend="meta", base_host=urlsplit(base).netloc, endpoint=ENDPOINT, model=MODEL, status=status,
                        http_status=code, attempts=[], headers={}, response_id=None, ttfb_ms=None, total_ms=round(ms),
                        usage=(kept or {}).get("usage"), cost_usd=cost, fingerprint=request_fingerprint(request),
                        provenance="meta")
        write_call(self.run_dir, index=index, role="image", request=request, raw=raw, kind="json", meta=meta)
        if picture is None:
            raise PainterError(f"{MODEL}: {status} (HTTP {code}): {raw[:200].decode('utf-8', 'replace')}")
        served, w, h = for_the_table(picture)
        name = f"{hashlib.sha256(served).hexdigest()[:32]}.png"
        self.pictures_dir.mkdir(parents=True, exist_ok=True)
        (self.pictures_dir / name).write_bytes(served)
        return Painted(src=f"/pictures/{name}", width=w, height=h, model=MODEL, provider="meta", ms=ms, cost_usd=cost)

    def summary(self) -> str:
        each = f", {sum(self.ms) / len(self.ms) / 1000:.1f} s per picture" if self.ms else ""
        return f"pictures {MODEL}: {self.calls} call{'' if self.calls == 1 else 's'}{each}, ${self.cost_usd:.2f}"

    def close(self) -> None:
        self._http.close()
