"""Live transport (D2, D3, D21): send a derived request, record the call as C5, return the C1 ``model_result``.

The request is sent exactly as ``requests.build_request`` derived it, with each ``{"$frame": sha}`` replaced by a
data URL of those bytes. Retries (D3): 429/500/503, transport errors and retryable mid-stream errors back off from
0.5 s, doubling, with jitter, honouring ``Retry-After``, at most 4 attempts, all inside the call's deadline; 400,
401, 402, 403, 404, 413 and 504 are never retried. 401/402/403/404 are recorded and then raised as
``ModelAccessError``, because every further call would fail the same way.
"""

from __future__ import annotations

import base64
import email.utils
import asyncio
import json
import random
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import httpx

from lightwork.contracts.calls import FRAME_REF_KEY, Attempt, CallMeta, ServerProps, request_fingerprint
from lightwork.contracts.canonical import q_s, sha256_hex
from lightwork.contracts.events import ModelResult

from .budget import Budget
from .cache import DevCache
from .errors import BudgetExceeded, ModelAccessError, ModelConfigError, ProvenanceError
from .pricing import aggregate_usage, cost_usd
from .provenance import Provenance, classify
from .record import write_call
from .wire import CHAT, RESPONSES, RETRYABLE_CODES, STREAM_CUT, Reading, read_body

MAX_ATTEMPTS = 4
BACKOFF_S = 0.5
RETRY_STATUS = frozenset({429, 500, 503})
ACCESS_STATUS = frozenset({401, 402, 403, 404})
# Per-call deadlines, retries included (D3). Wonder authoring has a separate longer text-generation deadline.
DEFAULT_TIMEOUTS = {
    "meta": {"segment": 5.0, "verdict": 20.0, "ground": 20.0, "why": 20.0, "author": 300.0},
    "chat": {"verdict": 180.0, "ground": 180.0, "why": 180.0, "author": 180.0},
}
_KEEP_HEADERS = ("x-ratelimit-", "retry-after", "x-omniroute-")
_MIME = ((b"\xff\xd8\xff", "image/jpeg"), (b"\x89PNG\r\n\x1a\n", "image/png"), (b"%PDF-", "application/pdf"))


def timeout_for(role: str, backend: str) -> float:
    return DEFAULT_TIMEOUTS[backend][role]


def endpoint_url(base: str, endpoint: str) -> str:
    """``https://api.meta.ai/v1`` + ``/v1/responses`` -> ``https://api.meta.ai/v1/responses``."""
    base = base.rstrip("/")
    if base.endswith("/v1") and endpoint.startswith("/v1/"):
        return base + endpoint[3:]
    return base + endpoint


def wire_body(request: dict[str, Any], frames: dict[str, bytes]) -> bytes:
    """JSON bytes to send: every ``{"$frame": sha}`` becomes a data URL of ``frames[sha]`` (sha-checked)."""

    def sub(node: Any) -> Any:
        if isinstance(node, dict):
            if set(node) == {FRAME_REF_KEY}:
                sha = node[FRAME_REF_KEY]
                data = frames.get(sha)
                if data is None:
                    raise ValueError(f"request references frame {sha} but no bytes were given")
                if sha256_hex(data) != sha:
                    raise ValueError(f"bytes given for frame {sha} hash differently")
                return f"data:{_mime(data)};base64,{base64.b64encode(data).decode('ascii')}"
            return {k: sub(v) for k, v in node.items()}
        if isinstance(node, list):
            return [sub(v) for v in node]
        return node

    return json.dumps(sub(request), separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def _mime(data: bytes) -> str:
    for magic, mime in _MIME:
        if data.startswith(magic):
            return mime
    raise ValueError("media must be JPEG, PNG or PDF")


def fetch_server_props(http: httpx.Client, base_url: str, api_key: str | None) -> ServerProps | None:
    """llama.cpp ``GET /props`` subset (C5 ``server_props``); ``None`` when the server is not llama.cpp or is down."""
    parts = urlsplit(base_url)
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    try:
        resp = http.get(f"{parts.scheme}://{parts.netloc}/props", headers=headers, timeout=5.0)
        data = resp.json() if resp.status_code == 200 else None
    except (httpx.HTTPError, ValueError):
        return None
    if not isinstance(data, dict):
        return None
    mods = data.get("modalities")
    return ServerProps(
        build_info=data["build_info"] if isinstance(data.get("build_info"), str) else None,
        model_path=data["model_path"] if isinstance(data.get("model_path"), str) else None,
        modalities={k: v for k, v in mods.items() if isinstance(v, bool)} if isinstance(mods, dict) else None,
    )


def run_provenance(settings, backend: str, models: dict[str, str], *, transport: httpx.BaseTransport | None = None
                   ) -> tuple[Provenance, ServerProps | None]:
    """Provenance of a run before it starts: ``meta`` only if every model it will call is a Meta model."""
    if backend == "meta":
        host = urlsplit(settings.meta_api_base).netloc
        provs = {classify("meta", host, m, None) for m in models.values()}
        return ("meta" if provs == {"meta"} else "dev-standin"), None
    if not settings.chat_base_url:
        raise ModelConfigError("LW_CHAT_BASE_URL is not set")
    with httpx.Client(transport=transport, trust_env=False) as http:
        props = fetch_server_props(http, settings.chat_base_url, settings.chat_api_key)
    host = urlsplit(settings.chat_base_url).netloc
    provs = {classify("chat", host, m, props) for m in models.values()}
    return ("meta" if provs == {"meta"} else "dev-standin"), props


@dataclass(frozen=True)
class _Try:
    status: int | None  # HTTP status; None when no response arrived
    raw: bytes
    kind: str  # "sse" | "json"
    headers: httpx.Headers | None
    ttfb_ms: int | None
    ms: int
    error: str | None
    timed_out: bool = False
    reading: Reading | None = None  # body reading of an HTTP 200


class ModelClient:
    """Thread-safe; one per run. ``provenance`` is the run's: a ``meta`` run refuses any call that is not Meta."""

    def __init__(
        self,
        settings,
        *,
        run_dir: Path,
        provenance: Provenance,
        budget: Budget,
        cache: DevCache | None,
        recording: bool = True,
        transport: httpx.BaseTransport | None = None,
        sleep: Callable[[float], None] = time.sleep,
        rng: random.Random | None = None,
        clock: Callable[[], float] = time.monotonic,
    ):
        if provenance not in ("meta", "dev-standin"):
            raise ValueError(f"unknown provenance {provenance!r}")
        if recording and cache is not None:
            raise ModelConfigError("a recording must not use DevCache; pass recording=False for fixture development")
        self.settings = settings
        self.run_dir = Path(run_dir)
        self.provenance = provenance
        self.budget = budget
        self.cache = cache
        self._sleep = sleep
        self._rng = rng or random.Random()
        self._clock = clock
        self._transport = transport
        self._chat_http = httpx.Client(transport=transport, trust_env=False)  # local server or SSH tunnel
        self._props_lock = threading.Lock()
        self._props: ServerProps | None = None

    def close(self) -> None:
        self._chat_http.close()

    def __enter__(self) -> ModelClient:
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # ------------------------------------------------------------ chat server

    def server_props(self) -> ServerProps | None:
        """The chat server's llama.cpp props, fetched once it answers (a failed fetch is retried next call)."""
        with self._props_lock:
            if self._props is None and self.settings.chat_base_url:
                self._props = fetch_server_props(self._chat_http, self.settings.chat_base_url,
                                                 self.settings.chat_api_key)
            return self._props

    def wait_chat_ready(self, timeout_s: float = 90.0) -> bool:
        """Poll llama.cpp ``/health`` while it answers 503 (model loading or waking from sleep)."""
        if not self.settings.chat_base_url:
            raise ModelConfigError("LW_CHAT_BASE_URL is not set")
        parts = urlsplit(self.settings.chat_base_url)
        key = self.settings.chat_api_key
        headers = {"Authorization": f"Bearer {key}"} if key else {}
        deadline = self._clock() + timeout_s
        delay = BACKOFF_S
        while True:
            try:
                resp = self._chat_http.get(f"{parts.scheme}://{parts.netloc}/health", headers=headers, timeout=5.0)
                if resp.status_code != 503:
                    return resp.status_code < 500
            except httpx.HTTPError:
                pass
            if self._clock() + delay > deadline:
                return False
            self._sleep(delay)
            delay = min(delay * 2, 8.0)

    # ------------------------------------------------------------ calls

    def execute(self, *, intent_id: str, role: str, endpoint: str, request: dict, frames: dict[str, bytes],
                index: int, timeout_s: float) -> dict:
        """Send ``request``, write ``calls/<NNNN>-<role>-<model>/`` and return the C1 ``model_result`` payload."""
        t0 = self._clock()
        backend, base, key = self._backend(endpoint)
        host = urlsplit(base).netloc
        model = request["model"]
        fingerprint = request_fingerprint(request)
        props = self.server_props() if backend == "chat" else None
        call_prov = classify(backend, host, model, props)
        self._check_provenance(call_prov, role, model, host)

        if self.cache is not None:
            hit = self.cache.get(host, fingerprint)
            if hit is not None:
                meta = CallMeta(backend=backend, base_host=host, endpoint=endpoint, model=model, status="ok",
                                http_status=hit.meta.http_status, attempts=[], headers={},
                                response_id=hit.meta.response_id, ttfb_ms=None, total_ms=self._ms(t0),
                                usage=hit.meta.usage, cost_usd=0.0, fingerprint=fingerprint, server_props=props,
                                provenance=call_prov)
                call_dir = write_call(self.run_dir, index=index, role=role, request=request, raw=hit.raw,
                                      kind=hit.kind, meta=meta)
                return ModelResult(intent_id=intent_id, call_dir=call_dir, status="ok", http_status=hit.meta.http_status,
                                   latency_ms=meta.total_ms, usage=hit.meta.usage).model_dump()

        body = wire_body(request, frames)
        headers = {"Content-Type": "application/json",
                   "Accept": "text/event-stream" if request.get("stream") else "application/json"}
        if key:
            headers["Authorization"] = f"Bearer {key}"
        attempts, responses = self._send(endpoint_url(base, endpoint), body, headers, endpoint, t0 + timeout_s,
                                        model, request, backend == "meta")
        last = responses[-1]
        usage = aggregate_usage([r.reading.usage for r in responses if r.reading is not None])
        cost = cost_usd(model, usage) if backend == "meta" else 0.0

        if last.timed_out:
            status = "timeout"
        elif last.status == 200 and last.error is None:
            status = "ok"
        else:
            status = "error"
        meta = CallMeta(
            backend=backend, base_host=host, endpoint=endpoint, model=model, status=status, http_status=last.status,
            attempts=attempts, headers=_kept_headers(last.headers),
            response_id=last.reading.response_id if last.reading is not None else None,
            ttfb_ms=last.ttfb_ms, total_ms=self._ms(t0), usage=usage, cost_usd=cost, fingerprint=fingerprint,
            server_props=props, provenance=call_prov,
        )
        call_dir = write_call(self.run_dir, index=index, role=role, request=request, raw=last.raw, kind=last.kind,
                              meta=meta, attempt_responses=[(r.raw, r.kind) for r in responses[:-1]])
        if status == "ok" and self.cache is not None:
            self.cache.put(host, fingerprint, last.raw, last.kind, meta)
        result = ModelResult(intent_id=intent_id, call_dir=call_dir, status=status, http_status=last.status,
                             latency_ms=meta.total_ms, usage=usage).model_dump()
        if last.status in ACCESS_STATUS:
            message = _access_message(backend, model, last.status, last.error)
            message += "\nResponse body: " + last.raw.decode("utf-8", errors="replace")
            raise ModelAccessError(message, result)
        return result

    def _backend(self, endpoint: str) -> tuple[str, str, str | None]:
        if endpoint == RESPONSES:
            if not self.settings.model_api_key:
                raise ModelConfigError("MODEL_API_KEY is not set (Meta Model API)")
            return "meta", self.settings.meta_api_base, self.settings.model_api_key
        if endpoint == CHAT:
            if not self.settings.chat_base_url:
                raise ModelConfigError("LW_CHAT_BASE_URL is not set (chat backend)")
            return "chat", self.settings.chat_base_url, self.settings.chat_api_key
        raise ValueError(f"unknown endpoint {endpoint!r}")

    def _check_provenance(self, call_prov: Provenance, role: str, model: str, host: str) -> None:
        if call_prov == "meta":
            return
        if role == "segment":
            raise ProvenanceError(f"SAM must be Meta's: {model} at {host} is not (PLAN.md D15c)")
        if not self.settings.dev_allow_non_meta:
            raise ProvenanceError(f"{model} at {host} is not a Meta model; dev stand-ins need "
                                  "LW_DEV_ALLOW_NON_META=1 (PLAN.md D15)")
        if self.provenance == "meta":
            raise ProvenanceError(f"{model} at {host} is a dev stand-in but this run is recorded as meta")

    def _send(self, url: str, body: bytes, headers: dict[str, str], endpoint: str, deadline: float,
              model: str, request: dict, billed: bool) -> tuple[list[Attempt], list[_Try]]:
        attempts: list[Attempt] = []
        responses: list[_Try] = []
        wait = 0.0
        while True:
            try:
                reservation = self.budget.reserve(model, request) if billed else None
            except BudgetExceeded as e:
                if not responses:
                    raise
                attempts[-1] = attempts[-1].model_copy(update={"error": f"{attempts[-1].error}; retry blocked: {e}"})
                return attempts, responses
            actual = None
            try:
                t = asyncio.run(self._try(url, body, headers, endpoint, deadline))
                if t.reading is not None and t.reading.usage is not None:
                    actual = cost_usd(model, t.reading.usage)
                elif t.status in RETRY_STATUS | ACCESS_STATUS | {400, 413, 504} or (
                    t.status is None and t.error and t.error.startswith(("ConnectError:", "ConnectTimeout:"))
                ):
                    actual = 0.0  # rejected before inference
            finally:
                self.budget.settle(reservation, actual)
            responses.append(t)
            n = len(responses)
            attempts.append(Attempt(n=n, status=t.status, error=t.error, wait_s=q_s(wait), ms=t.ms))
            if t.timed_out or n == MAX_ATTEMPTS or not _retryable(t):
                return attempts, responses
            wait = self._backoff(n, t.headers)
            if self._clock() + wait >= deadline:
                return attempts, responses
            self._sleep(wait)

    async def _try(self, url: str, body: bytes, headers: dict[str, str], endpoint: str, deadline: float) -> _Try:
        start = self._clock()
        remaining = deadline - start
        if remaining <= 0:
            return _Try(None, b"", "json", None, None, 0, "deadline passed before sending", timed_out=True)
        chunks: list[bytes] = []
        status: int | None = None
        resp_headers: httpx.Headers | None = None
        kind = "json"
        ttfb: int | None = None
        try:
            timeout = httpx.Timeout(remaining, connect=min(remaining, 10.0))
            async with asyncio.timeout(remaining):
                async with httpx.AsyncClient(transport=self._transport, trust_env=endpoint == RESPONSES) as http:
                    async with http.stream("POST", url, content=body, headers=headers, timeout=timeout) as resp:
                        status, resp_headers = resp.status_code, resp.headers
                        kind = "sse" if "event-stream" in resp.headers.get("content-type", "") else "json"
                        async for chunk in resp.aiter_bytes():
                            if ttfb is None:
                                ttfb = self._ms(start)
                            chunks.append(chunk)
                            if self._clock() > deadline:
                                raise TimeoutError("call deadline")
        except (httpx.TimeoutException, TimeoutError) as e:
            return _Try(status, b"".join(chunks), kind, resp_headers, ttfb, self._ms(start),
                        f"{type(e).__name__}: timeout", timed_out=True)
        except httpx.TransportError as e:
            return _Try(status, b"".join(chunks), kind, resp_headers, ttfb, self._ms(start),
                        f"{type(e).__name__}: {e}"[:300])
        raw = b"".join(chunks)
        ms = self._ms(start)
        if status != 200:
            reading = read_body(raw, "json", endpoint)
            return _Try(status, raw, kind, resp_headers, ttfb, ms, reading.error or f"HTTP {status}", reading=reading)
        guard = _guard(resp_headers)
        reading = read_body(raw, kind, endpoint)
        if guard is not None:
            return _Try(status, raw, kind, resp_headers, ttfb, ms, guard, reading=reading)
        return _Try(status, raw, kind, resp_headers, ttfb, ms, reading.error if reading.status == "error" else None,
                    reading=reading)

    def _backoff(self, n: int, headers: httpx.Headers | None) -> float:
        delay = BACKOFF_S * 2 ** (n - 1)
        delay += self._rng.uniform(0, delay / 2)
        retry_after = _retry_after(headers.get("retry-after") if headers is not None else None)
        return max(delay, retry_after or 0.0)

    def _ms(self, since: float) -> int:
        return int(round((self._clock() - since) * 1000))


def _retryable(t: _Try) -> bool:
    if t.error and t.error.startswith("omniroute"):
        return False
    if t.status is None:
        return t.error is not None and not t.timed_out  # connection refused/reset before any response
    if t.status in RETRY_STATUS:
        return True
    if t.status == 200 and t.reading is not None and t.reading.status == "error":
        return t.reading.error_code in RETRYABLE_CODES or t.reading.error_code == STREAM_CUT
    if t.status == 200 and t.reading is None and t.error is not None and not t.error.startswith("omniroute"):
        return True  # body cut by a transport error after the headers
    return False


def _guard(headers: httpx.Headers | None) -> str | None:
    """omniroute stand-in pitfalls (briefs/standins.md): never accept a bridged image or a cached reply."""
    if headers is None:
        return None
    if "x-omniroute-modality-bridge" in headers:
        bridge = headers["x-omniroute-modality-bridge"]
        return f"omniroute replaced the image with a text description ({bridge}); use a vision model id"
    if headers.get("x-omniroute-cache", "").strip().upper() == "HIT":
        return "omniroute served a cached reply; only the Lightwork dev cache is allowed (D21)"
    return None


def _kept_headers(headers: httpx.Headers | None) -> dict[str, str]:
    if headers is None:
        return {}
    return {k.lower(): v for k, v in headers.items() if k.lower().startswith(_KEEP_HEADERS)}


def _retry_after(value: str | None) -> float | None:
    if not value:
        return None
    try:
        return max(0.0, float(value))
    except ValueError:
        pass
    try:
        when = email.utils.parsedate_to_datetime(value)
    except (TypeError, ValueError):
        return None
    if when.tzinfo is None:
        when = when.replace(tzinfo=timezone.utc)
    return max(0.0, (when - datetime.now(timezone.utc)).total_seconds())


def _access_message(backend: str, model: str, status: int | None, error: str | None) -> str:
    msg = f"HTTP {status} for {model}: {error}"
    key = "MODEL_API_KEY" if backend == "meta" else "LW_CHAT_API_KEY"
    if status == 404 and "model_not_found" in (error or ""):
        msg += (". Check on dev.meta.ai that this key's team has access to the model; intermittent model_not_found "
                "from cloud hosts is a Meta-side entitlement bug, so run from the laptop or the Mac Studio")
    elif status == 401:
        msg += f". Check {key}"
    elif status == 402:
        msg += ". Billing or credits: check the Model API dashboard"
    elif status == 403:
        msg += ". The key lacks access to this model or feature"
    return msg
