"""The table's thinking: plans, checks, names and answers from Meta models, hosted or open-weight.

Two backends behind one small interface (:meth:`Brain.ask`: a prompt plus images in, text or JSON out):

``MetaBrain``
    Meta Model API (Muse Spark, Muse Wonder) through :class:`lightwork.models.client.ModelClient`: recorded under the
    session's call directory and held to the ``LW_BUDGET_USD`` cap. JSON answers use strict structured output.
``OpenBrain``
    Meta's open-weight models served by the hackathon's inference partners or on our own hardware, through their
    OpenAI-compatible chat API, tried in order (``LW_OPEN_ORDER``) and skipped for a minute after a failure:

    ===========  =====================================================  ==========================================
    provider     model (default)                                        set
    ===========  =====================================================  ==========================================
    baseten      ``meta-models/muse-spark-open`` (Muse Spark Open)      ``BASETEN_API_KEY`` (``BASETEN_MODEL``:
                                                                        ``meta-models/muse-spark-open-1026`` with
                                                                        the hackathon key)
    fireworks    ``accounts/fireworks/models/muse-spark-open``          ``FIREWORKS_API_KEY`` (``FIREWORKS_MODEL``)
    modal        ``muse-glimmer`` (Muse Glimmer on a Modal GPU,         ``MODAL_LLM_URL`` (+ ``MODAL_LLM_KEY``,
                 ``scripts/modal_glimmer.py``)                          ``MODAL_LLM_MODEL``)
    local        ``muse-glimmer`` (llama.cpp on our own Mac)            ``LW_CHAT_BASE_URL`` (+ ``LW_CHAT_MODEL``)
    ===========  =====================================================  ==========================================

    Only Meta open-weight model ids (Muse Spark Open, Muse Glimmer) are accepted: the hackathon allows no other model
    at runtime. Every call is recorded like the Meta ones: ``calls/<n>-<role>-<provider>/`` with the request (images
    by SHA-256, the JPEGs under ``frames/``), the raw response and ``meta.json`` (provider, host, model, latency,
    usage, status) - the provenance the open-weight prize's partner check asks for.

:class:`Jobs` runs brain calls on a worker thread so the camera loop never waits.
"""

from __future__ import annotations

import base64
import json
import os
import queue
import re
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urlsplit

import httpx

META_DEFAULT = "muse-spark-1.3"
OPEN_ORDER = ("baseten", "fireworks", "modal", "local")
COOLDOWN_S = 60.0  # a provider that failed is skipped this long
OPEN_MODEL_RE = re.compile(r"(?i)(muse[-_ ]?spark[-_ ]?open|glimmer)")


class BrainError(RuntimeError):
    pass


@dataclass
class Reply:
    text: str
    data: dict | list | None  # parsed JSON when asked for
    provider: str  # "meta" | "baseten" | ...
    model: str
    ms: float
    cost_usd: float = 0.0


def parse_json(text: str):
    """JSON from a model answer: the whole text, a fenced block, or the first {...} / [...] in it."""
    t = text.strip()
    for cand in (t, *re.findall(r"```(?:json)?\s*(.*?)```", t, flags=re.DOTALL)):
        try:
            return json.loads(cand)
        except (ValueError, TypeError):
            pass
    for open_, close in (("{", "}"), ("[", "]")):
        a, b = t.find(open_), t.rfind(close)
        if 0 <= a < b:
            try:
                return json.loads(t[a:b + 1])
            except ValueError:
                continue
    raise BrainError(f"no JSON in the answer: {t[:120]!r}")


class Brain:
    """Base: counters and the summary line."""

    label = "brain"

    def __init__(self) -> None:
        self.calls = 0
        self.cost_usd = 0.0
        self.ms: deque[float] = deque(maxlen=30)
        self.last: Reply | None = None
        self.trace: deque[str] = deque(maxlen=8)  # recent calls, for the screen
        self._lock = threading.Lock()

    def check(self) -> str | None:
        return None

    def ask(self, role: str, prompt: str, jpegs: list[bytes] = (), *, schema: dict | None = None,
            effort: str = "low", max_out: int = 4000, timeout_s: float = 60.0, model: str | None = None) -> Reply:
        raise NotImplementedError

    def _note(self, r: Reply, role: str) -> Reply:
        with self._lock:
            self.calls += 1
            self.cost_usd += r.cost_usd
            self.ms.append(r.ms)
            self.last = r
            self.trace.append(f"{role}: {r.model} @ {r.provider} {r.ms / 1000:.1f} s")
        return r

    def summary(self) -> str:
        ms = f", {sum(self.ms) / len(self.ms) / 1000:.1f} s per call" if self.ms else ""
        cost = f", ${self.cost_usd:.4f}" if self.cost_usd else ""
        return f"{self.label}: {self.calls} call{'' if self.calls == 1 else 's'}{ms}{cost}"

    def close(self) -> None:
        pass


# ---------------------------------------------------------------------------------------------------- Meta API


class MetaBrain(Brain):
    def __init__(self, settings, model: str = META_DEFAULT, *, run_dir: Path | None = None, client=None):
        super().__init__()
        from .models.budget import Budget
        from .models.client import ModelClient

        self.model = model
        self.label = f"{model} (Meta Model API)"
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        self.run_dir = Path(run_dir) if run_dir is not None else settings.state_dir / "play-calls" / f"{stamp}-brain"
        self._settings = settings
        self._client = client or ModelClient(settings, run_dir=self.run_dir, provenance="meta",
                                             budget=Budget.from_settings(settings), cache=None)
        self._index = 0
        self._ilock = threading.Lock()

    def check(self) -> str | None:
        return None if self._settings.model_api_key else "MODEL_API_KEY is not set (.env)"

    def ask(self, role, prompt, jpegs=(), *, schema=None, effort="low", max_out=4000, timeout_s=60.0, model=None):
        from .contracts.calls import frame_ref
        from .contracts.canonical import sha256_hex
        from .models.pricing import cost_usd
        from .models.wire import RESPONSES, read_body

        model = model or self.model
        frames_dir = self.run_dir / "frames"
        frames_dir.mkdir(parents=True, exist_ok=True)
        frames: dict[str, bytes] = {}
        content: list[dict] = [{"type": "input_text", "text": prompt}]
        for jpeg in jpegs:
            sha = sha256_hex(jpeg)
            (frames_dir / f"{sha}.jpg").write_bytes(jpeg)
            frames[sha] = jpeg
            content.append({"type": "input_image", "image_url": frame_ref(sha)})
        request: dict = {
            "model": model,
            "input": [{"type": "message", "role": "user", "content": content}],
            "reasoning": {"effort": effort},
            "max_output_tokens": max_out,
            "store": False,
            "stream": True,
        }
        if schema is not None:
            request["text"] = {"format": {"type": "json_schema", "name": re.sub(r"\W", "_", role)[:40] or "answer",
                                          "schema": schema, "strict": True}}
        with self._ilock:
            self._index += 1
            index = self._index
        t0 = time.perf_counter()
        result = self._client.execute(intent_id=f"play-{role}-{index}", role=_role_for_client(role),
                                      endpoint=RESPONSES, request=request, frames=frames, index=index,
                                      timeout_s=timeout_s)
        body = next((self.run_dir / result["call_dir"]).glob("response.*"))
        reading = read_body(body.read_bytes(), body.suffix[1:], RESPONSES)
        if result["status"] != "ok" or reading.status != "ok":
            raise BrainError(f"{model}: {result['status']}: {reading.error or 'no answer'}")
        text = reading.text or ""
        data = parse_json(text) if schema is not None else None
        return self._note(Reply(text=text, data=data, provider="meta", model=model,
                                ms=(time.perf_counter() - t0) * 1000.0,
                                cost_usd=cost_usd(model, result.get("usage"))), role)

    def close(self) -> None:
        self._client.close()


def _role_for_client(role: str) -> str:
    """The recorder's call role (its timeouts table knows a few); the brain's own role goes in the intent id."""
    return {"plan": "author", "check": "verdict"}.get(role, "why")


# ---------------------------------------------------------------------------------------------------- partners


@dataclass
class Provider:
    name: str
    label: str
    base_url: str
    model: str
    api_key: str | None = field(default=None, repr=False)
    json_mode: bool = True  # accepts response_format {"type": "json_object"}
    effort: bool = True  # accepts reasoning_effort
    down_until: float = 0.0
    error: str | None = None

    @property
    def host(self) -> str:
        return urlsplit(self.base_url).netloc


def providers(env: dict[str, str] | None = None, settings=None) -> list[Provider]:
    """The configured open-weight providers, in ``LW_OPEN_ORDER``."""
    env = dict(os.environ if env is None else env)
    if settings is not None:  # .env values (load_settings already merged the environment over them)
        from .config import parse_dotenv

        dotenv = Path(settings.root) / ".env"
        if dotenv.is_file():
            env = {**parse_dotenv(dotenv.read_text(encoding="utf-8")), **env}

    def get(k: str) -> str | None:
        v = env.get(k)
        return v.strip() if v and v.strip() else None

    found: dict[str, Provider] = {}
    if get("BASETEN_API_KEY"):
        found["baseten"] = Provider("baseten", "Baseten", get("BASETEN_BASE_URL") or "https://inference.baseten.co/v1",
                                    get("BASETEN_MODEL") or "meta-models/muse-spark-open", get("BASETEN_API_KEY"))
    if get("FIREWORKS_API_KEY"):
        found["fireworks"] = Provider("fireworks", "Fireworks AI",
                                      get("FIREWORKS_BASE_URL") or "https://api.fireworks.ai/inference/v1",
                                      get("FIREWORKS_MODEL") or "accounts/fireworks/models/muse-spark-open",
                                      get("FIREWORKS_API_KEY"))
    if get("MODAL_LLM_URL"):
        found["modal"] = Provider("modal", "Modal", get("MODAL_LLM_URL").rstrip("/"),
                                  get("MODAL_LLM_MODEL") or "muse-glimmer", get("MODAL_LLM_KEY"), json_mode=True,
                                  effort=False)
    if get("LW_CHAT_BASE_URL"):
        found["local"] = Provider("local", "own hardware (llama.cpp)", get("LW_CHAT_BASE_URL").rstrip("/"),
                                  get("LW_CHAT_MODEL") or "muse-glimmer", get("LW_CHAT_API_KEY"), json_mode=False,
                                  effort=False)
    order = [s.strip() for s in (get("LW_OPEN_ORDER") or ",".join(OPEN_ORDER)).split(",") if s.strip()]
    return [found[n] for n in order if n in found]


def is_open_meta_model(model: str) -> bool:
    return bool(OPEN_MODEL_RE.search(model))


class OpenBrain(Brain):
    """Muse Spark Open / Muse Glimmer through the partners (see the module doc), recorded under ``run_dir``."""

    def __init__(self, provs: list[Provider], *, run_dir: Path, http: httpx.Client | None = None,
                 clock=time.monotonic):
        super().__init__()
        bad = [p for p in provs if not is_open_meta_model(p.model)]
        if bad:
            raise BrainError(f"not a Meta open-weight model: {', '.join(f'{p.name}={p.model}' for p in bad)} "
                             "(only Muse Spark Open and Muse Glimmer may run here)")
        self.providers = provs
        self.run_dir = Path(run_dir)
        self._http = http or httpx.Client(trust_env=False)
        self._clock = clock
        self._index = 0
        self._ilock = threading.Lock()
        names = ", ".join(f"{p.label} ({p.model})" for p in provs)
        self.label = f"Meta open weights via {names}" if provs else "Meta open weights (no provider configured)"

    def check(self) -> str | None:
        if not self.providers:
            return ("no open-weight provider configured: set BASETEN_API_KEY, FIREWORKS_API_KEY, MODAL_LLM_URL or "
                    "LW_CHAT_BASE_URL (.env)")
        return None

    def ask(self, role, prompt, jpegs=(), *, schema=None, effort="low", max_out=4000, timeout_s=60.0, model=None):
        errors = []
        now = self._clock()
        live = [p for p in self.providers if p.down_until <= now] or list(self.providers)
        for p in live:
            try:
                return self._note(self._ask_one(p, role, prompt, list(jpegs), schema, effort, max_out, timeout_s),
                                  role)
            except BrainError as exc:
                p.error, p.down_until = str(exc), self._clock() + COOLDOWN_S
                errors.append(f"{p.name}: {exc}")
        raise BrainError("; ".join(errors) or "no provider")

    def _ask_one(self, p: Provider, role, prompt, jpegs, schema, effort, max_out, timeout_s) -> Reply:
        from .contracts.canonical import sha256_hex

        frames_dir = self.run_dir / "frames"
        frames_dir.mkdir(parents=True, exist_ok=True)
        text = prompt
        if schema is not None:
            text += "\n\nAnswer with one JSON object that follows this JSON schema exactly:\n" + json.dumps(schema)
        parts: list[dict] = [{"type": "text", "text": text}]
        shas = []
        for jpeg in jpegs:
            sha = sha256_hex(jpeg)
            (frames_dir / f"{sha}.jpg").write_bytes(jpeg)
            shas.append(sha)
            parts.append({"type": "image_url",
                          "image_url": {"url": "data:image/jpeg;base64," + base64.b64encode(jpeg).decode()}})
        body: dict = {"model": p.model, "messages": [{"role": "user", "content": parts}], "max_tokens": max_out,
                      "temperature": 0.3, "stream": False}
        if schema is not None and p.json_mode:
            body["response_format"] = {"type": "json_object"}
        if p.effort and effort:
            body["reasoning_effort"] = effort if effort in ("low", "medium", "high") else "low"
        with self._ilock:
            self._index += 1
            index = self._index
        call = self.run_dir / "calls" / f"{index:04d}-{role}-{p.name}"
        call.mkdir(parents=True, exist_ok=True)
        recorded = json.loads(json.dumps(body))  # the request as sent, images by hash
        k = 0
        for part in recorded["messages"][0]["content"]:
            if part["type"] == "image_url":
                part["image_url"] = {"$frame": shas[k]}
                k += 1
        (call / "request.json").write_text(json.dumps(recorded, indent=1) + "\n", encoding="utf-8")
        headers = {"Content-Type": "application/json"}
        if p.api_key:
            headers["Authorization"] = f"Bearer {p.api_key}"
        t0 = time.perf_counter()
        status, raw, err = None, b"", None
        for attempt in range(2):
            try:
                r = self._http.post(f"{p.base_url}/chat/completions", json=body, headers=headers, timeout=timeout_s)
                status, raw = r.status_code, r.content
            except httpx.HTTPError as exc:
                err = f"{type(exc).__name__}: {exc}"
                break
            if status == 400 and attempt == 0 and ("response_format" in body or "reasoning_effort" in body):
                body.pop("response_format", None)  # an older deployment: ask again without the extras
                body.pop("reasoning_effort", None)
                continue
            break
        ms = (time.perf_counter() - t0) * 1000.0
        (call / "response.json").write_bytes(raw or b"")
        meta = {"provider": p.name, "provider_label": p.label, "base_host": p.host, "model": p.model, "role": role,
                "http_status": status, "latency_ms": round(ms), "error": err, "frames": shas,
                "t_utc": datetime.now(UTC).isoformat(timespec="seconds")}
        out_text, usage = "", None
        if err is None and status == 200:
            try:
                doc = json.loads(raw)
                msg = doc["choices"][0]["message"]
                content = msg.get("content")
                if isinstance(content, list):
                    content = "".join(c.get("text", "") for c in content if isinstance(c, dict))
                out_text = content or ""
                usage = doc.get("usage")
                meta["served_model"] = doc.get("model")
                meta["response_id"] = doc.get("id")
                meta["finish_reason"] = doc["choices"][0].get("finish_reason")
            except (ValueError, KeyError, IndexError, TypeError) as exc:
                err = f"unreadable answer: {exc}"
        elif err is None:
            err = f"HTTP {status}: {raw[:300].decode('utf-8', 'replace')}"
        meta["usage"] = usage
        meta["status"] = "ok" if err is None and out_text else "error"
        (call / "meta.json").write_text(json.dumps(meta, indent=1) + "\n", encoding="utf-8")
        if err is not None:
            raise BrainError(err)
        if not out_text.strip():
            raise BrainError("empty answer")
        data = parse_json(out_text) if schema is not None else None
        return Reply(text=out_text, data=data, provider=p.name, model=p.model, ms=ms)

    def close(self) -> None:
        self._http.close()


def make_brain(kind: str, settings, *, run_dir: Path | None = None, model: str | None = None) -> Brain:
    """``meta``, ``open`` or ``auto`` (open weights when a partner is configured, else Meta Model API)."""
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    run_dir = Path(run_dir) if run_dir is not None else settings.state_dir / "play-calls" / f"{stamp}-brain"
    if kind in ("open", "auto"):
        provs = providers(settings=settings)
        if provs or kind == "open":
            return OpenBrain(provs, run_dir=run_dir)
    return MetaBrain(settings, model=model or META_DEFAULT, run_dir=run_dir)


# ---------------------------------------------------------------------------------------------------- jobs


class Jobs:
    """One worker thread: ``submit(key, fn)`` runs ``fn()`` later; ``result(key)`` -> (done, value, error)."""

    def __init__(self, name: str = "brain-jobs", size: int = 8):
        self._q: queue.Queue = queue.Queue(maxsize=size)
        self._done: dict = {}
        self._lock = threading.Lock()
        self.busy = False
        self._t = threading.Thread(target=self._loop, name=name, daemon=True)
        self._t.start()

    def submit(self, key, fn) -> bool:
        try:
            self._q.put_nowait((key, fn))
            return True
        except queue.Full:
            return False

    def result(self, key):
        with self._lock:
            if key in self._done:
                return (True, *self._done.pop(key))
        return False, None, None

    def _loop(self) -> None:
        while True:
            item = self._q.get()
            if item is None:
                return
            key, fn = item
            self.busy = True
            try:
                value, err = fn(), None
            except Exception as exc:  # noqa: BLE001 - a failed job is reported, never fatal
                value, err = None, f"{type(exc).__name__}: {exc}"
            self.busy = False
            with self._lock:
                self._done[key] = (value, err)

    def close(self) -> None:
        try:
            self._q.put_nowait(None)
        except queue.Full:
            pass
        self._t.join(timeout=2.0)
