"""Spoken questions about the table for ``lightwork play --table --voice``: Muse Voice Transcribe hears the question,
Muse Spark answers it from what the camera sees, and the projector page speaks the answer.

Point at an object and ask "what is this?" or "what is it for?": the question, a clean crop of the object pointed at
(from the last scan, our light off) and the names already known go to Spark in one call. A question about the whole
table ("what can I build with these?") gets the clean view of the table instead. Calls are recorded under
``.lightwork/play-calls/<session>-ask/`` and held to the ``LW_BUDGET_USD`` cap, like the names (:mod:`lightwork.namer`).
``ask`` and ``result`` never block; a worker thread makes the calls.
"""

from __future__ import annotations

import queue
import threading
import time
from collections import deque
from datetime import UTC, datetime
from pathlib import Path

META_MODEL = "muse-spark-1.3-contributor"
PROMPT = ("You are Lightwork, an assistant projected onto a work table by a small projector, with a camera looking "
          "down at the table. Someone at the table just asked you, out loud: \"{question}\"\n{context}\n"
          "The images are what the camera sees. Answer in one or two short sentences, the way you would say it "
          "aloud: no lists, no markdown, no more than 40 words. If the images do not show enough to answer, say "
          "what you can see and what you would need.")
MAX_OUTPUT = 1500
TIMEOUT_S = 30.0
MAX_CHARS = 320


class MetaAsker:
    label = META_MODEL

    def __init__(self, settings, model: str = META_MODEL, *, run_dir: Path | None = None, client=None):
        from .models.budget import Budget
        from .models.client import ModelClient

        self.model = model
        self.label = model
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        self.run_dir = Path(run_dir) if run_dir is not None else settings.state_dir / "play-calls" / f"{stamp}-ask"
        self._settings = settings
        self._client = client or ModelClient(settings, run_dir=self.run_dir, provenance="meta",
                                             budget=Budget.from_settings(settings), cache=None)
        self._q: queue.Queue = queue.Queue(maxsize=4)
        self._done: dict[object, str | None] = {}
        self._lock = threading.Lock()
        self._index = 0
        self.error: str | None = None
        self.calls = 0
        self.cost_usd = 0.0
        self.ms: deque[float] = deque(maxlen=20)
        self._thread = threading.Thread(target=self._loop, name="table-asker", daemon=True)
        self._thread.start()

    def check(self) -> str | None:
        return None if self._settings.model_api_key else "MODEL_API_KEY is not set (.env)"

    def ask(self, key, question: str, jpegs: list[bytes], context: str = "") -> bool:
        try:
            self._q.put_nowait((key, question, list(jpegs), context))
            return True
        except queue.Full:
            return False

    def result(self, key) -> tuple[bool, str | None]:
        with self._lock:
            if key in self._done:
                return True, self._done.pop(key)
        return False, None

    def summary(self) -> str:
        ms = f", {sum(self.ms) / len(self.ms) / 1000:.1f} s per answer" if self.ms else ""
        return f"{self.label} answers: {self.calls}{ms}, ${self.cost_usd:.4f} this session"

    def _answer(self, question: str, jpegs: list[bytes], context: str) -> str | None:
        from .contracts.calls import frame_ref
        from .contracts.canonical import sha256_hex
        from .models.pricing import cost_usd
        from .models.wire import RESPONSES, read_body

        frames_dir = self.run_dir / "frames"
        frames_dir.mkdir(parents=True, exist_ok=True)
        frames = {}
        content = [{"type": "input_text", "text": PROMPT.format(question=question.strip(), context=context.strip())}]
        for jpeg in jpegs:
            sha = sha256_hex(jpeg)
            (frames_dir / f"{sha}.jpg").write_bytes(jpeg)
            frames[sha] = jpeg
            content.append({"type": "input_image", "image_url": frame_ref(sha)})
        request = {
            "model": self.model,
            "input": [{"type": "message", "role": "user", "content": content}],
            "reasoning": {"effort": "minimal"},
            "max_output_tokens": MAX_OUTPUT,
            "store": False,
            "stream": True,
        }
        self._index += 1
        result = self._client.execute(intent_id=f"play-ask-{self._index}", role="ask", endpoint=RESPONSES,
                                      request=request, frames=frames, index=self._index, timeout_s=TIMEOUT_S)
        self.cost_usd += cost_usd(self.model, result.get("usage"))
        body = next((self.run_dir / result["call_dir"]).glob("response.*"))
        reading = read_body(body.read_bytes(), body.suffix[1:], RESPONSES)
        if result["status"] != "ok" or reading.status != "ok":
            raise RuntimeError(f"{result['status']}: {reading.error or 'no answer'}")
        text = " ".join(reading.text.replace("*", "").split())
        if len(text) > MAX_CHARS:  # spoken: cut at the last sentence end that fits
            cut = max(text.rfind(c, 0, MAX_CHARS) for c in ".!?")
            text = text[:cut + 1] if cut > 40 else text[:MAX_CHARS]
        return text or None

    def _loop(self) -> None:
        while True:
            item = self._q.get()
            if item is None:
                return
            key, question, jpegs, context = item
            t = time.perf_counter()
            try:
                answer = self._answer(question, jpegs, context)
                self.error = None
            except Exception as exc:  # noqa: BLE001 - a failed answer is said as such; never stop the worker
                answer, self.error = None, f"{type(exc).__name__}: {exc}"
            self.calls += 1
            self.ms.append((time.perf_counter() - t) * 1000.0)
            with self._lock:
                self._done[key] = answer

    def close(self) -> None:
        try:
            self._q.put_nowait(None)
        except queue.Full:
            pass
        self._thread.join(timeout=2.0)
        self._client.close()
