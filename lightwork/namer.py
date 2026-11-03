"""Object names for ``lightwork play --table`` (``--names meta|ollama``): one short name per shown object.

Each new object's camera crop is named once, on worker threads, so the frame loop never waits. Crops queued together
(a scan that found several objects) go out as one request with one image per object (``BATCH_MAX``), and two workers
run side by side, so a slow call never holds up the next scan's objects (2 Nov: one call at a time, 7-11 s each; the
hammer waited 54 s for its name behind phantom objects).

``MetaNamer`` (``--names meta``)
    Muse Spark through :class:`lightwork.models.client.ModelClient`, by default the cheapest model,
    ``muse-spark-1.3-contributor`` ($0.10 / $0.20 per 1M input / output tokens; its inputs may train Meta models: only
    object crops are sent, never a hand or the room). Every call is recorded under ``.lightwork/play-calls/<session>/``
    (request, raw response, meta with usage and cost) and goes through the persisted ``LW_BUDGET_USD`` cap. The
    session's calls and spend are counted for the screen and the terminal.
``OllamaNamer`` (``--names ollama``)
    A local vision-language model served by Ollama, free and offline, but not Meta (NOTES §3: small on-device non-Meta
    helpers are allowed next to Meta models doing the core work). qwen3.5:4b on this laptop's GPU, 2 Nov: 0.4-0.6 s
    per crop warm, 1.7 s cold; it named a hammer, a water bottle and two phones from top-down crops in warm light,
    where an ImageNet classifier got only the hammer.
"""

from __future__ import annotations

import base64
import queue
import re
import threading
import time
from collections import deque
from datetime import UTC, datetime
from pathlib import Path

import httpx

OLLAMA_MODEL = "qwen3.5:4b"
OLLAMA_URL = "http://127.0.0.1:11434"
META_MODEL = "muse-spark-1.3-contributor"
PROMPT = ("This is a top-down photo of one object lying on a white table under warm light. What is the object? "
          "Answer with 1 to 3 words, no punctuation.")
BATCH_PROMPT = ("Each numbered image is a top-down photo of one object lying on a white table. Look at each one closely "
                "and name it in 1 to 3 plain words, the way someone at the table would say it: a generic name, not a "
                "brand. Answer with JSON, one name per image, in image order.")
BATCH_SCHEMA = {
    "type": "object",
    "properties": {"names": {"type": "array", "items": {"type": "string"}}},
    "required": ["names"],
    "additionalProperties": False,
}
TASK_PROMPT = ("Each numbered image is a top-down photo of one object lying on a white table, where someone is getting "
               "ready for: {title}. The job needs these items:\n{items}\nFor every image, first look closely and name "
               "what the object really is in 1 to 3 plain words (a generic name, not a brand). Then give the number of "
               "the listed item only if the object clearly is that item; most everyday objects are none of them, so "
               "give 0. Never rename an object to fit the list. Answer with JSON, one entry per image, in image order.")
TASK_SCHEMA = {
    "type": "object",
    "properties": {"objects": {"type": "array", "items": {
        "type": "object",
        "properties": {"name": {"type": "string"}, "item": {"type": "integer"}},
        "required": ["name", "item"],
        "additionalProperties": False,
    }}},
    "required": ["objects"],
    "additionalProperties": False,
}
BATCH_MAX = 8  # crops per request
META_WORKERS = 2
MAX_WORDS = 3
MAX_CHARS = 18
META_MAX_OUTPUT = 1000  # reasoning at "minimal" still spends some output tokens before the answer
META_TIMEOUT_S = 30.0


def clean_name(text: str) -> str | None:
    """A short lowercase name from a model answer: letters, digits, spaces and hyphens, at most ``MAX_WORDS`` words."""
    words = re.sub(r"[^a-z0-9\- ]+", " ", text.lower()).split()
    words = [w for w in words if w not in ("a", "an", "the")][:MAX_WORDS]
    while words and len(" ".join(words)) > MAX_CHARS:
        words.pop()
    return " ".join(words) or None


class _Namer:
    """Worker threads: ``request(key, jpeg)`` queues a crop, ``result(key)`` collects its name; neither blocks."""

    label = "namer"
    batch_max = 1  # crops one call can name

    def __init__(self, workers: int = 1) -> None:
        self._q: queue.Queue[tuple[object, bytes] | None] = queue.Queue(maxsize=32)
        self._done: dict[object, str | None] = {}
        self._matches: dict[object, int] = {}  # key -> 1-based task item (0 = none), when the namer knows the task
        self._lock = threading.Lock()
        self.error: str | None = None
        self.calls = 0
        self.cost_usd = 0.0
        self.ms: deque[float] = deque(maxlen=20)
        self._threads = [threading.Thread(target=self._loop, name=f"object-namer-{i}", daemon=True)
                         for i in range(max(1, workers))]
        for t in self._threads:
            t.start()

    def check(self) -> str | None:
        return None

    def request(self, key, jpeg: bytes) -> bool:
        """Queue a crop; False when the queue is full (ask again later)."""
        try:
            self._q.put_nowait((key, jpeg))
            return True
        except queue.Full:
            return False

    def result(self, key) -> tuple[bool, str | None]:
        """(finished, name). A finished request without a usable answer gives (True, None)."""
        with self._lock:
            if key in self._done:
                return True, self._done.pop(key)
        return False, None

    def match(self, key) -> int | None:
        """Which task item (1-based, 0 = none) the named object is, if the namer was told the task; else None."""
        with self._lock:
            return self._matches.pop(key, None)

    def summary(self) -> str:
        ms = f", {sum(self.ms) / len(self.ms) / 1000:.1f} s per call" if self.ms else ""
        cost = f", ${self.cost_usd:.4f} this session" if self.cost_usd or self.label.startswith("muse") else ""
        return f"{self.label}: {self.calls} call{'' if self.calls == 1 else 's'}{ms}{cost}"

    def _name(self, jpeg: bytes) -> str | None:
        raise NotImplementedError

    def _name_batch(self, jpegs: list[bytes]) -> list[str | None]:
        """Names for several crops, in order; by default one call each."""
        return [self._name(j) for j in jpegs]

    def _loop(self) -> None:
        while True:
            item = self._q.get()
            if item is None:
                return
            batch = [item]
            stop = False
            while len(batch) < self.batch_max:  # what queued up meanwhile goes in the same call
                try:
                    nxt = self._q.get_nowait()
                except queue.Empty:
                    break
                if nxt is None:
                    stop = True
                    break
                batch.append(nxt)
            t = time.perf_counter()
            try:
                names = list(self._name_batch([j for _, j in batch]))
                names += [None] * (len(batch) - len(names))
                self.error = None
            except Exception as exc:  # noqa: BLE001 - a failed name must never stop the worker; it is retried later
                names, self.error = [None] * len(batch), f"{type(exc).__name__}: {exc}"
            with self._lock:
                self.calls += 1
                self.ms.append((time.perf_counter() - t) * 1000.0)
                for (key, _), name in zip(batch, names, strict=False):
                    if isinstance(name, tuple):  # (name, task item)
                        name, self._matches[key] = name
                    self._done[key] = name
            if stop:
                return

    def close(self) -> None:
        for _ in self._threads:
            try:
                self._q.put_nowait(None)
            except queue.Full:
                break
        for t in self._threads:
            t.join(timeout=2.0)


class OllamaNamer(_Namer):
    def __init__(self, model: str = OLLAMA_MODEL, url: str = OLLAMA_URL, *, timeout: float = 20.0, client=None):
        self.model, self.url = model, url.rstrip("/")
        self.label = f"{model} via Ollama (local, not Meta)"
        self._http = client or httpx.Client(timeout=timeout)
        super().__init__(workers=1)

    def check(self) -> str | None:
        try:
            tags = self._http.get(f"{self.url}/api/tags", timeout=3.0).json()
        except (httpx.HTTPError, ValueError) as exc:
            return f"no Ollama server at {self.url} ({exc})"
        names = {m.get("name") for m in tags.get("models", [])} | {m.get("model") for m in tags.get("models", [])}
        return None if self.model in names else f"Ollama has no model {self.model!r} (ollama pull {self.model})"

    def _name(self, jpeg: bytes) -> str | None:
        body = {"model": self.model, "stream": False, "think": False, "keep_alive": "30m",
                "options": {"temperature": 0, "num_predict": 12},
                "messages": [{"role": "user", "content": PROMPT, "images": [base64.b64encode(jpeg).decode()]}]}
        r = self._http.post(f"{self.url}/api/chat", json=body)
        r.raise_for_status()
        return clean_name(r.json()["message"]["content"])

    def close(self) -> None:
        super().close()
        self._http.close()


class MetaNamer(_Namer):
    """Muse Spark, recorded and budget-capped through the project's model client (see the module doc)."""

    batch_max = BATCH_MAX

    def __init__(self, settings, model: str = META_MODEL, *, run_dir: Path | None = None, client=None,
                 workers: int = META_WORKERS):
        from .models.budget import Budget
        from .models.client import ModelClient

        self.model = model
        self.label = model
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        self.run_dir = Path(run_dir) if run_dir is not None else settings.state_dir / "play-calls" / stamp
        self._settings = settings
        if client is not None:  # one injected client: one worker
            self._clients = [client]
            workers = 1
        else:  # a client per worker: calls in flight side by side, numbered from one counter
            self._clients = [ModelClient(settings, run_dir=self.run_dir, provenance="meta",
                                         budget=Budget.from_settings(settings), cache=None) for _ in range(workers)]
        self._client_of: dict[int, object] = {}
        self.task = None  # lightwork.benchtask.BenchTask: each name also says which listed item the object is
        self._index = 0
        self._index_lock = threading.Lock()
        super().__init__(workers=workers)

    def check(self) -> str | None:
        return None if self._settings.model_api_key else "MODEL_API_KEY is not set (.env)"

    def _client(self):
        ident = threading.get_ident()
        with self._index_lock:
            if ident not in self._client_of:
                self._client_of[ident] = self._clients[len(self._client_of) % len(self._clients)]
            return self._client_of[ident]

    def _next_index(self) -> int:
        with self._index_lock:
            self._index += 1
            return self._index

    def _frames(self, jpegs: list[bytes]) -> list[str]:
        from .contracts.canonical import sha256_hex

        frames = self.run_dir / "frames"
        frames.mkdir(parents=True, exist_ok=True)
        shas = []
        for jpeg in jpegs:
            sha = sha256_hex(jpeg)
            (frames / f"{sha}.jpg").write_bytes(jpeg)  # the recording holds what was sent
            shas.append(sha)
        return shas

    def _send(self, request: dict, frames: dict[str, bytes]):
        from .models.pricing import cost_usd
        from .models.wire import RESPONSES, read_body

        index = self._next_index()
        result = self._client().execute(intent_id=f"play-name-{index}", role="name", endpoint=RESPONSES,
                                        request=request, frames=frames, index=index, timeout_s=META_TIMEOUT_S)
        with self._lock:
            self.cost_usd += cost_usd(self.model, result.get("usage"))
        call = self.run_dir / result["call_dir"]
        body = next(call.glob("response.*"))
        reading = read_body(body.read_bytes(), body.suffix[1:], RESPONSES)
        if result["status"] != "ok" or reading.status != "ok":
            raise RuntimeError(f"{result['status']}: {reading.error or 'no answer'}")
        return reading.text

    def _name(self, jpeg: bytes) -> str | None:
        from .contracts.calls import frame_ref

        (sha,) = self._frames([jpeg])
        request = {
            "model": self.model,
            "input": [{"type": "message", "role": "user", "content": [
                {"type": "input_text", "text": PROMPT}, {"type": "input_image", "image_url": frame_ref(sha)}]}],
            "reasoning": {"effort": "minimal"},
            "max_output_tokens": META_MAX_OUTPUT,
            "store": False,
            "stream": True,
        }
        return clean_name(self._send(request, {sha: jpeg}))

    def _name_batch(self, jpegs: list[bytes]) -> list:
        if len(jpegs) == 1 and self.task is None:
            return [self._name(jpegs[0])]
        from .contracts.calls import frame_ref
        from .models.parse import json_reply

        shas = self._frames(jpegs)
        if self.task is not None:
            return self._task_batch(shas, jpegs)
        content = [{"type": "input_text", "text": BATCH_PROMPT}]
        for k, sha in enumerate(shas, 1):
            content += [{"type": "input_text", "text": f"Image {k}:"}, {"type": "input_image", "image_url": frame_ref(sha)}]
        request = {
            "model": self.model,
            "input": [{"type": "message", "role": "user", "content": content}],
            "reasoning": {"effort": "minimal"},
            "max_output_tokens": META_MAX_OUTPUT + 40 * len(jpegs),
            "store": False,
            "stream": True,
            "text": {"format": {"type": "json_schema", "name": "object_names", "schema": BATCH_SCHEMA, "strict": True}},
        }
        reply = json_reply(self._send(request, dict(zip(shas, jpegs, strict=True))))
        names = reply.get("names") if isinstance(reply, dict) else None
        if not isinstance(names, list):
            raise TypeError("the reply has no names list")
        return [clean_name(n) if isinstance(n, str) else None for n in names[:len(jpegs)]]

    def _task_batch(self, shas: list[str], jpegs: list[bytes]) -> list[tuple[str | None, int]]:
        """Names plus the task item each object is: [(name, 1-based item or 0)]."""
        from .contracts.calls import frame_ref
        from .models.parse import json_reply

        items = "\n".join(f"{k}. {name}" for k, name in enumerate(self.task.items, 1))
        content = [{"type": "input_text", "text": TASK_PROMPT.format(title=self.task.title, items=items)}]
        for k, sha in enumerate(shas, 1):
            content += [{"type": "input_text", "text": f"Image {k}:"}, {"type": "input_image", "image_url": frame_ref(sha)}]
        request = {
            "model": self.model,
            "input": [{"type": "message", "role": "user", "content": content}],
            "reasoning": {"effort": "minimal"},
            "max_output_tokens": META_MAX_OUTPUT + 60 * len(jpegs),
            "store": False,
            "stream": True,
            "text": {"format": {"type": "json_schema", "name": "task_objects", "schema": TASK_SCHEMA, "strict": True}},
        }
        reply = json_reply(self._send(request, dict(zip(shas, jpegs, strict=True))))
        objs = reply.get("objects") if isinstance(reply, dict) else None
        if not isinstance(objs, list):
            raise TypeError("the reply has no objects list")
        out = []
        for o in objs[:len(jpegs)]:
            name = clean_name(o.get("name")) if isinstance(o, dict) and isinstance(o.get("name"), str) else None
            item = o.get("item") if isinstance(o, dict) else 0
            item = item if isinstance(item, int) and 0 <= item <= len(self.task.items) else 0
            out.append((name, item))
        return out

    def close(self) -> None:
        super().close()
        for c in self._clients:
            c.close()
