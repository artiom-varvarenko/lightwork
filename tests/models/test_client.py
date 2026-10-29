"""Offline httpx transport, retry, budget, cache, provenance and C5 evidence tests."""

import asyncio
import base64
import json
import random
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx
import pytest

from lightwork.config import Settings
from lightwork.contracts.calls import CallMeta, ServerProps, request_fingerprint
from lightwork.contracts.canonical import canonical_bytes, sha256_hex
from lightwork.models.budget import Budget
from lightwork.models.cache import DevCache
from lightwork.models.client import ModelClient, _retry_after, endpoint_url, run_provenance, wire_body
from lightwork.models.errors import BudgetExceeded, ModelAccessError, ModelConfigError, ProvenanceError
from lightwork.models.pricing import aggregate_usage, cost_usd, estimate_usd
from lightwork.models.provenance import classify
from lightwork.models.record import write_call
from lightwork.models.requests import META_MODELS, build_request
from lightwork.models.synthetic import chat_sse, responses_error_sse, responses_sse
from lightwork.models.wire import CHAT, RESPONSES, read_body

pytestmark = pytest.mark.usefixtures("no_network")
ANSWER = '{"done":"yes","contradiction":null,"evidence":"Blue rectangle in right slot."}'
MODEL = META_MODELS["verdict"]
IMAGE = b"\xff\xd8\xff" + b"synthetic fixture bytes, never sent to a real model"
SHA = sha256_hex(IMAGE)
USAGE = {"input_tokens": 2000, "input_tokens_details": {"cached_tokens": 500}, "output_tokens": 100}


def make_request(backend="meta", model=MODEL):
    return build_request("verdict", model=model, backend=backend, frame_sha=SHA,
                         ctx={"step_id": "s", "instruction": "Pack it.", "question": "Is it in its slot?"})


def successful(endpoint=RESPONSES, usage=USAGE):
    raw = responses_sse(ANSWER, model=MODEL, usage=usage) if endpoint == RESPONSES else chat_sse(ANSWER, model="muse-glimmer")
    return httpx.Response(200, content=raw, headers={"content-type": "text/event-stream", "x-ratelimit-remaining-requests": "149"})


def invoke(tmp_path, handler, *, backend="meta", model=MODEL, settings=None, cache=None, recording=True,
           provenance="meta", cap=120., index=1, clock=None, sleep=None, timeout=20):
    settings = settings or Settings(model_api_key="secret-never-record", root=tmp_path)
    kwargs = {"sleep": sleep or (lambda _: None)}
    if clock:
        kwargs["clock"] = clock
    budget = Budget(tmp_path / "spend.json", cap)
    with ModelClient(settings, run_dir=tmp_path, provenance=provenance, budget=budget, cache=cache, recording=recording,
                     transport=httpx.MockTransport(handler), rng=random.Random(0), **kwargs) as client:
        endpoint, request = make_request(backend, model)
        result = client.execute(intent_id="s/judge/0", role="verdict", endpoint=endpoint, request=request,
                                frames={SHA: IMAGE}, index=index, timeout_s=timeout)
    d = tmp_path / result["call_dir"]
    return result, CallMeta.model_validate_json((d / "meta.json").read_text()), d, budget


def test_live_transport_writes_exact_canonical_and_raw(tmp_path):
    def handler(req):
        assert str(req.url) == "https://api.meta.ai/v1/responses"
        body = json.loads(req.content)
        url = body["input"][0]["content"][0]["image_url"]
        assert base64.b64decode(url.split(",", 1)[1]) == IMAGE
        assert req.headers["authorization"] == "Bearer secret-never-record"
        return successful()
    result, meta, d, budget = invoke(tmp_path, handler)
    assert result["status"] == "ok" and result["usage"] == USAGE
    req = make_request()[1]
    assert (d / "request.json").read_bytes() == canonical_bytes(req) + b"\n"
    assert (d / "response.sse").read_bytes() == responses_sse(ANSWER, model=MODEL, usage=USAGE)
    assert meta.fingerprint == request_fingerprint(req) and len(meta.attempts) == 1
    assert meta.cost_usd == cost_usd(MODEL, USAGE) == budget.spent_usd
    assert meta.headers == {"x-ratelimit-remaining-requests": "149"}
    assert b"secret-never-record" not in (d / "meta.json").read_bytes() + (d / "request.json").read_bytes()


@pytest.mark.parametrize("status", [429, 500, 503])
def test_retryable_status_matrix(tmp_path, status):
    seen = []
    sleeps = []
    def handler(req):
        seen.append(req.content)
        if len(seen) == 1:
            return httpx.Response(status, json={"error": {"code": "server_error", "message": "try again"}},
                                  headers={"retry-after": "1.25"})
        return successful()
    result, meta, d, _ = invoke(tmp_path, handler, sleep=sleeps.append)
    assert result["status"] == "ok" and len(seen) == 2 and seen[0] == seen[1]
    assert sleeps == [1.25] and [a.wait_s for a in meta.attempts] == [0, 1.25]
    assert (d / "attempts" / "0001.json").is_file()


@pytest.mark.parametrize("status", [400, 401, 402, 403, 404, 413, 504, 502])
def test_permanent_status_matrix(tmp_path, status):
    seen = []
    def handler(req):
        seen.append(req)
        return httpx.Response(status, json={"error": {"code": "model_not_found", "message": "not accessible"}})
    if status in (401, 402, 403, 404):
        with pytest.raises(ModelAccessError) as err:
            invoke(tmp_path, handler)
        assert err.value.result["status"] == "error"
        if status == 404:
            assert "entitlement" in str(err.value) and "not accessible" in str(err.value)
    else:
        result, meta, _, _ = invoke(tmp_path, handler)
        assert result["status"] == "error" and meta.http_status == status
    assert len(seen) == 1
    assert len(list((tmp_path / "calls").glob("*/meta.json"))) == 1


def test_retry_limit_and_backoff(tmp_path):
    waits = []
    result, meta, d, _ = invoke(tmp_path, lambda _: httpx.Response(503, json={"error": {"message": "model lookup busy"}}),
                               sleep=waits.append)
    assert result["status"] == "error" and len(meta.attempts) == 4
    assert len(waits) == 3 and .5 <= waits[0] <= .75 and 1 <= waits[1] <= 1.5 and 2 <= waits[2] <= 3
    assert len(list((d / "attempts").iterdir())) == 3


@pytest.mark.parametrize("mode", ["error_event", "stream_cut", "failed_with_usage", "connection"])
def test_retryable_stream_and_transport_errors(tmp_path, mode):
    count = 0
    def handler(req):
        nonlocal count
        count += 1
        if count == 1:
            if mode == "connection":
                raise httpx.ConnectError("refused", request=req)
            if mode == "error_event":
                raw = responses_error_sse("server_shutting_down", "going down", model=MODEL)
            elif mode == "failed_with_usage":
                raw = responses_sse("", model=MODEL, terminal="failed", usage=USAGE)
            else:
                raw = b'data: {"type":"response.created","response":{"id":"r"}}\n\n'
            return httpx.Response(200, content=raw, headers={"content-type": "text/event-stream"})
        return successful()
    result, meta, d, _ = invoke(tmp_path, handler)
    assert result["status"] == "ok" and len(meta.attempts) == 2
    usages = []
    for p in sorted((d / "attempts").iterdir()) + [d / "response.sse"]:
        usages.append(read_body(p.read_bytes(), p.suffix[1:], RESPONSES).usage)
    assert result["usage"] == meta.usage == aggregate_usage(usages)
    if mode == "failed_with_usage":
        assert meta.usage["input_tokens"] == 4000
        assert meta.cost_usd == cost_usd(MODEL, aggregate_usage([USAGE, USAGE]))


def test_timeout_is_recorded_with_no_retry(tmp_path):
    def handler(req):
        raise httpx.ReadTimeout("late", request=req)
    result, meta, d, budget = invoke(tmp_path, handler)
    assert result["status"] == "timeout" and len(meta.attempts) == 1
    assert (d / "response.json").read_bytes() == b""
    assert budget.spent_usd == 0
    assert json.loads(budget.path.read_text())["unresolved_usd"] > 0


def test_whole_deadline_cancels_dribbling_stream(tmp_path):
    class Slow(httpx.AsyncByteStream):
        async def __aiter__(self):
            yield b": heartbeat\n\n"
            await asyncio.sleep(5)
            yield b"data: [DONE]\n\n"
    def handler(req):
        return httpx.Response(200, stream=Slow(), headers={"content-type": "text/event-stream"})
    result, meta, d, _ = invoke(tmp_path, handler, timeout=.05)
    assert result["status"] == "timeout" and meta.total_ms < 1000
    assert (d / "response.sse").read_bytes() == b": heartbeat\n\n"


def test_retry_after_beyond_deadline_does_not_sleep(tmp_path):
    sleeps = []
    result, meta, _, _ = invoke(tmp_path, lambda _: httpx.Response(429, headers={"retry-after": "60"},
                               json={"error": {"message": "slow down"}}), sleep=sleeps.append)
    assert result["status"] == "error" and len(meta.attempts) == 1 and sleeps == []
    assert _retry_after("not a date") is None and _retry_after("-3") == 0
    assert _retry_after("Thu, 01 Jan 1970 00:00:00 GMT") == 0


@pytest.mark.parametrize("header", ["x-omniroute-modality-bridge", "x-omniroute-cache"])
def test_omniroute_guards_preserve_raw_but_reject_content(tmp_path, header):
    posted = []
    def handler(req):
        if req.url.path == "/props":
            return httpx.Response(404)
        posted.append(req)
        resp = successful(CHAT)
        resp.headers[header] = "image->text;model=other" if "bridge" in header else "HIT"
        return resp
    settings = Settings(root=tmp_path, chat_base_url="http://127.0.0.1:20128/v1", dev_allow_non_meta=True)
    result, meta, _, _ = invoke(tmp_path, handler, backend="chat", model="cx/gpt-5.6-luna-low", settings=settings,
                                provenance="dev-standin")
    assert result["status"] == "error" and len(posted) == 1
    assert header in meta.headers and "omniroute" in meta.attempts[0].error
    assert meta.provenance == "dev-standin"


def test_provenance_allowlist_and_loaded_weights():
    glimmer = ServerProps(model_path="/models/Muse-Glimmer-30B-Q4_K_M.gguf", build_info="b11146", modalities={"vision": True})
    other = ServerProps(model_path="/models/qwen.gguf")
    assert classify("meta", "api.meta.ai", MODEL, None) == "meta"
    assert classify("meta", "api.meta.ai.evil.test", MODEL, None) == "dev-standin"
    assert classify("meta", "api.meta.ai", "cx/gpt", None) == "dev-standin"
    assert classify("chat", "localhost:8080", "muse-glimmer", glimmer) == "meta"
    assert classify("chat", "localhost:8080", "muse-glimmer", other) == "dev-standin"
    assert classify("chat", "localhost:11434", "muse-glimmer:30b-mlx", None) == "meta"
    assert classify("chat", "localhost:20128", "cx/gpt", None) == "dev-standin"


@pytest.mark.parametrize("allowed,provenance", [(False, "dev-standin"), (True, "meta")])
def test_standin_requires_flag_and_separate_provenance(tmp_path, allowed, provenance):
    posted = []
    def handler(req):
        if req.method == "POST":
            posted.append(req)
        return httpx.Response(404)
    settings = Settings(root=tmp_path, chat_base_url="http://127.0.0.1:20128/v1", dev_allow_non_meta=allowed)
    with pytest.raises(ProvenanceError):
        invoke(tmp_path, handler, backend="chat", model="cx/gpt", settings=settings, provenance=provenance)
    assert posted == []


def test_glimmer_props_recorded(tmp_path):
    props = {"model_path": "/models/Muse-Glimmer-Q4.gguf", "build_info": "b11146", "modalities": {"vision": True},
             "secret_server_extra": "must not be recorded"}
    def handler(req):
        if req.url.path == "/props":
            return httpx.Response(200, json=props)
        return successful(CHAT)
    settings = Settings(root=tmp_path, chat_base_url="http://127.0.0.1:8080/v1")
    result, meta, d, _ = invoke(tmp_path, handler, backend="chat", model="muse-glimmer", settings=settings)
    assert result["status"] == "ok" and meta.provenance == "meta"
    assert meta.server_props.model_path == props["model_path"] and meta.cost_usd == 0
    assert "secret_server_extra" not in (d / "meta.json").read_text()
    assert run_provenance(settings, "chat", {"verdict": "muse-glimmer"}, transport=httpx.MockTransport(handler))[0] == "meta"


def test_budget_blocks_before_http_and_persists(tmp_path):
    def forbidden(_):
        pytest.fail("budget-blocked call reached network")
    with pytest.raises(BudgetExceeded):
        invoke(tmp_path, forbidden, cap=0)
    budget = Budget(tmp_path / "budget.json", 10.)
    req = make_request()[1]
    reservation = budget.reserve(MODEL, req)
    budget.settle(reservation, .25)
    assert Budget(budget.path, 10.).spent_usd == .25
    assert json.loads(budget.path.read_text())["reservations"] == {}


def test_budget_is_shared_between_clients_and_threads(tmp_path):
    req = make_request()[1]
    bound = estimate_usd(MODEL, req)
    path = tmp_path / "budget.json"
    def reserve(_):
        b = Budget(path, bound)
        try:
            return b.reserve(MODEL, req)
        except BudgetExceeded:
            return None
    with ThreadPoolExecutor(max_workers=4) as pool:
        tokens = list(pool.map(reserve, range(8)))
    assert sum(t is not None for t in tokens) == 1
    Budget(path, bound).settle(next(t for t in tokens if t), 0.)
    assert Budget(path, bound).reserve(MODEL, req)


@pytest.mark.parametrize("contents", ['{"spent_usd":-1,"calls":0}', '{"spent_usd":NaN,"calls":0}', 'bad'])
def test_budget_corruption_fails_closed(tmp_path, contents):
    path = tmp_path / "spend.json"
    path.write_text(contents)
    with pytest.raises(ValueError, match="corrupt"):
        Budget(path, 120).reserve(MODEL, make_request()[1])


def test_cache_only_in_development(tmp_path):
    cache = DevCache(tmp_path / "cache")
    posted = []
    def handler(req):
        posted.append(req)
        return successful()
    with pytest.raises(ModelConfigError, match="recording"):
        invoke(tmp_path, handler, cache=cache)
    first = invoke(tmp_path, handler, cache=cache, recording=False)
    second = invoke(tmp_path, handler, cache=cache, recording=False, index=2)
    assert len(posted) == 1
    assert second[1].attempts == [] and second[1].cost_usd == 0
    assert (first[2] / "response.sse").read_bytes() == (second[2] / "response.sse").read_bytes()
    assert first[3].spent_usd == second[3].spent_usd


def test_write_call_refuses_overwrite_wrong_fingerprint_and_inline_media(tmp_path):
    _, meta, d, _ = invoke(tmp_path, lambda _: successful())
    request = make_request()[1]
    kwargs = dict(index=1, role="verdict", request=request, raw=b"", kind="json", meta=meta)
    with pytest.raises(FileExistsError):
        write_call(tmp_path, **kwargs)
    kwargs["index"] = 2
    kwargs["meta"] = meta.model_copy(update={"fingerprint": "0" * 64})
    with pytest.raises(ValueError, match="fingerprint"):
        write_call(tmp_path, **kwargs)
    request = {**request, "inline": "data:image/jpeg;base64,x"}
    kwargs.update(request=request, meta=meta.model_copy(update={"fingerprint": request_fingerprint(request)}))
    with pytest.raises(ValueError, match="inline data"):
        write_call(tmp_path, **kwargs)


def test_media_missing_or_wrong_hash_and_pdf():
    with pytest.raises(ValueError, match="no bytes"):
        wire_body(make_request()[1], {})
    with pytest.raises(ValueError, match="hash differently"):
        wire_body(make_request()[1], {SHA: b"other"})
    pdf = b"%PDF-1.7\nexample"
    sha = sha256_hex(pdf)
    body = json.loads(wire_body({"file_data": {"$frame": sha}}, {sha: pdf}))
    assert body["file_data"].startswith("data:application/pdf;base64,")
    assert endpoint_url("https://api.meta.ai/v1/", RESPONSES) == "https://api.meta.ai/v1/responses"


def test_budget_blocked_retry_still_records_first_response(tmp_path):
    bound = estimate_usd(MODEL, make_request()[1])
    raw = responses_sse("", model=MODEL, terminal="failed", usage=USAGE)
    result, meta, d, budget = invoke(tmp_path, lambda _: httpx.Response(
        200, content=raw, headers={"content-type": "text/event-stream"}), cap=bound)
    assert result["status"] == "error" and result["http_status"] == 200
    assert len(meta.attempts) == 1 and "retry blocked" in meta.attempts[0].error
    assert (d / "response.sse").read_bytes() == raw and budget.spent_usd == cost_usd(MODEL, USAGE)
