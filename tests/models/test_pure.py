"""Offline wire, schema, prompt and deterministic C9 tests (workstream F)."""

import copy
import json
from pathlib import Path

import numpy as np
import pytest
from meta_sam_parser import CompletedOutcome, decode_mask_to_rle, image_segmentation_format

from lightwork.contracts.backend import Mask
from lightwork.contracts.calls import request_fingerprint
from lightwork.contracts.canonical import canonical
from lightwork.contracts.stepgraph import StepGraph, validate_semantics
from lightwork.models.parse import parse_response
from lightwork.models.pricing import aggregate_usage, cost_usd
from lightwork.models.requests import META_MODELS, PROMPT_VERSIONS, build_request
from lightwork.models.synthetic import SamObject, chat_sse, responses_error_sse, responses_sse, sam_sse
from lightwork.models.wire import CHAT, RESPONSES, iter_sse, read_body

pytestmark = pytest.mark.usefixtures("no_network")
VERDICT = {"done": "yes", "contradiction": None, "evidence": "Blue rectangle is inside the right slot."}
CTX = {"step_id": "s1", "instruction": "Put the blue rectangle in its slot.", "question": "Is it in the right slot?"}
SHA = "a" * 64


def response(text, *, endpoint=RESPONSES, **kwargs):
    maker = responses_sse if endpoint == RESPONSES else chat_sse
    return maker(text, model="muse-glimmer" if endpoint == CHAT else META_MODELS["verdict"], **kwargs)


@pytest.fixture
def draft():
    check = {"id": "judge", "kind": "judge", "object": None, "region": None, "cells": [], "n": None,
             "direction": None, "min_delta": None, "ratio": None, "tol": None,
             "question": "Is the blue rectangle in its slot?", "required": True}
    return {"title": "Pack the kit", "objects": [{"id": "part", "noun": "blue rectangle", "kind": "item",
             "rows": None, "cols": None, "naming": None, "channel": None}], "steps": [{"id": "pack",
             "say": "Put the blue rectangle in its slot.", "detail": "Keep the kit complete.", "targets": ["part"],
             "arrow_from": None, "arrow_to": None, "checks": [check], "recovery": []}],
            "inserted_steps": [], "vocabulary": ["rectangle"]}


def test_sse_skips_nondata_and_joins_multiline():
    raw = b': keepalive\r\nevent: ignored\r\nid: 2\r\nretry: 50\r\ndata: {"a":\r\ndata: 1}\r\n\r\n'
    assert [e.data for e in iter_sse(raw)] == ['{"a":\n1}']
    assert len(iter_sse(b'data: {"a":1}\n\n\ndata: [DONE]\n\n')) == 2


@pytest.mark.parametrize("endpoint", [RESPONSES, CHAT])
@pytest.mark.parametrize("fenced", [False, True])
def test_verdict_roundtrip(endpoint, fenced):
    text = json.dumps(VERDICT, ensure_ascii=False)
    if fenced:
        text = "```json\n" + text + "\n```"
    raw = response(text, endpoint=endpoint)
    a = parse_response("verdict", raw, "sse", endpoint)
    assert a["status"] == "ok" and a["result"] == VERDICT
    assert canonical(a) == canonical(parse_response("verdict", raw, "sse", endpoint))
    assert a["usage"] is not None


@pytest.mark.parametrize("contradiction", ["", "none", "NULL", " None "])
def test_verdict_null_normalization(contradiction):
    result = parse_response("verdict", response(json.dumps({**VERDICT, "contradiction": contradiction})), "sse", RESPONSES)
    assert result["result"]["contradiction"] is None


def test_verdict_length_violation_requests_repair():
    raw = response(json.dumps({**VERDICT, "contradiction": "x" * 160, "evidence": "y" * 200}))
    out = parse_response("verdict", raw, "sse", RESPONSES)
    assert out["status"] == "invalid"
    assert "contradiction" in out["error"] and "evidence" in out["error"]
    assert len(json.loads(out["result"]["text"])["evidence"]) == 200


@pytest.mark.parametrize("obj", [{"done": "yes"}, {**VERDICT, "done": "maybe"}, {**VERDICT, "contradiction": False},
                                  {**VERDICT, "evidence": []}, {**VERDICT, "extra": 1}, []])
def test_verdict_rejects_wrong_schema(obj):
    out = parse_response("verdict", response(json.dumps(obj)), "sse", RESPONSES)
    assert out["status"] == "invalid" and out["error"] and out["result"]["text"]


@pytest.mark.parametrize("text", ["", "not json", "{} trailing", '{"x":NaN}', '```json\n{}\n``` prose'])
def test_invalid_json_reply(text):
    assert parse_response("verdict", response(text), "sse", RESPONSES)["status"] == "invalid"


@pytest.mark.parametrize("terminal,expected", [("completed", "ok"), ("incomplete", "incomplete"), ("failed", "error")])
def test_responses_terminal_events(terminal, expected):
    raw = response(json.dumps(VERDICT), terminal=terminal)
    reading = read_body(raw, "sse", RESPONSES)
    assert reading.status == expected and reading.usage
    if terminal == "incomplete":
        assert reading.error == "max_output_tokens"
    if terminal == "failed":
        assert reading.error_code == "server_error"


@pytest.mark.parametrize("code", ["server_shutting_down", "service_overloaded", "backend_unavailable",
                                  "rate_limit_exceeded", "server_error", "invalid_request_error"])
def test_responses_error_events(code):
    raw = responses_error_sse(code, "test error", model="muse-spark-1.3")
    out = read_body(raw, "sse", RESPONSES)
    assert out.status == "error" and out.error_code == code and "test error" in out.error


def test_responses_done_without_deltas_and_refusal():
    final = {"status": "completed", "id": "r", "output": [{"type": "reasoning"},
             {"type": "message", "content": [{"type": "output_text", "text": json.dumps(VERDICT)}]}]}
    raw = ("data: " + json.dumps({"type": "response.completed", "response": final}) + "\n\n").encode()
    assert parse_response("verdict", raw, "sse", RESPONSES)["result"] == VERDICT
    final["output"][1]["content"] = [{"type": "refusal", "refusal": "Cannot inspect that."}]
    assert read_body(json.dumps(final).encode(), "json", RESPONSES).error_code == "refusal"
    refusal = b'data: {"type":"response.refusal.delta","delta":"No"}\n\n'
    refusal += b'data: {"type":"response.completed","response":{"status":"completed"}}\n\n'
    assert read_body(refusal, "sse", RESPONSES).error_code == "refusal"


@pytest.mark.parametrize("endpoint", [RESPONSES, CHAT])
def test_empty_malformed_and_cut_stream(endpoint):
    for raw, code in [(b"", "empty"), (b"data: not-json\n\n", "malformed"), (b"data: [DONE]\n\n", "stream_cut")]:
        assert read_body(raw, "sse", endpoint).error_code == code
    assert read_body(b"[]", "json", endpoint).error_code == "malformed"


@pytest.mark.parametrize("finish,expected", [("stop", "ok"), ("length", "incomplete"), ("content_filter", "error"),
                                           ("tool_calls", "error")])
def test_chat_reasoning_finish_and_usage(finish, expected):
    text = "" if finish == "length" else json.dumps(VERDICT)
    raw = chat_sse(text, model="muse-glimmer", reasoning="Private reasoning", finish=finish)
    out = read_body(raw, "sse", CHAT)
    assert out.status == expected and out.reasoning == "Private reasoning"
    assert out.text == text and out.usage["completion_tokens"] > 0
    if finish == "length":
        assert out.error == "length"


def test_chat_reasoning_alias_and_llama_error_field():
    raw = b'data: {"choices":[{"delta":{"reasoning":"think"}}]}\n\n'
    raw += b'data: {"choices":[{"delta":{"content":"answer"},"finish_reason":"stop"}]}\n\n'
    assert read_body(raw, "sse", CHAT).reasoning == "think"
    error = b'error: {"code":503,"type":"server_error","message":"loading"}\n\n'
    assert read_body(error, "sse", CHAT).error_code == "server_error"


@pytest.mark.parametrize("endpoint", [RESPONSES, CHAT])
def test_nonstream_json_and_http_error(endpoint):
    if endpoint == RESPONSES:
        obj = {"id": "r", "status": "completed", "output": [{"type": "message", "content": [
            {"type": "output_text", "text": json.dumps(VERDICT)}]}]}
    else:
        obj = {"id": "r", "choices": [{"message": {"content": json.dumps(VERDICT)}, "finish_reason": "stop"}]}
    assert parse_response("verdict", json.dumps(obj).encode(), "json", endpoint)["result"] == VERDICT
    err = {"error": {"code": "model_not_found", "message": "Missing"}}
    assert read_body(json.dumps(err).encode(), "json", endpoint).error_code == "model_not_found"


def test_ground_shapes_and_clamp():
    obj = {"items": [{"object_name": "rectangle", "x": -1, "y": 1001},
                     {"object_name": "slot", "bbox": [{"x_min": 800, "x_max": 300, "y_min": 5, "y_max": 30}]},
                     {"object_name": "well", "points": [{"x": 5.8, "y": 7}], "count": 2}]}
    result = parse_response("ground", response(json.dumps(obj)), "sse", RESPONSES)["result"]
    assert result == {"items": [{"name": "rectangle", "x": 0, "y": 1000}, {"name": "slot", "bbox": [300, 5, 800, 30]},
                                {"name": "well", "x": 6, "y": 7}], "counts": {"well": 2}}
    assert parse_response("ground", response('{"items":[]}'), "sse", RESPONSES)["result"] == {"items": []}
    invalid = '{"items":[{"object_name":"part","x":1e999,"y":0}]}'
    assert parse_response("ground", response(invalid), "sse", RESPONSES)["status"] == "invalid"


def test_sam_one_bit_masks_and_reference_rle():
    mask = np.zeros((9, 13), bool)
    mask[2:7, 3:11] = True
    raw = sam_sse([SamObject(7, (20, 30, 33, 39), mask, 0.95), SamObject(1, (5, 5, 9, 8))], (160, 90))
    out = parse_response("segment", raw, "sse", RESPONSES)
    assert out["status"] == "ok" and out["result"]["size"] == [160, 90]
    masks = out["result"]["masks"]
    assert [m["object_id"] for m in masks] == [1, 7]
    assert masks[1]["area_px"] == 40 and masks[1]["centroid"] == [26.5, 34.0]
    assert masks[1]["box"] == [20, 30, 33, 39] and masks[1]["confidence"] == .95
    parser = image_segmentation_format().create_parser()
    parser.push(read_body(raw, "sse", RESPONSES).text)
    records = parser.finish(CompletedOutcome()).result.records
    for rec in records:
        if rec.kind == "mask":
            expected = decode_mask_to_rle(rec.mask)
            mine = next(m for m in masks if m["object_id"] == int(rec.object_id))
            Mask.model_validate(mine)
            assert mine["rle"] == {"size": list(expected.size), "counts": expected.counts}


def test_sam_lossless_documented_fixture():
    text = ("<0f>0<|box;x1=211;y1=228;x2=270;y2=254;w=320;h=334|>"
            "<|mask;x=0;y=0;data=27,60,~!!!!M!0c[0o91w?q1!pIH4pPRVp2B3'7`e.ioeAf6-k/#Xd8%dX9x(|>")
    out = parse_response("segment", response(text), "sse", RESPONSES)
    assert out["status"] == "ok", out
    mask = out["result"]["masks"][0]
    assert mask["box"] == [211, 228, 271, 255] and mask["area_px"] == 1539


def test_sam_zero_and_malformed_are_different():
    assert parse_response("segment", sam_sse([], (100, 100)), "sse", RESPONSES)["result"] == {"masks": [], "size": None}
    for text in ("not segmentation", "<0f>1<|box;x1=4|>"):
        assert parse_response("segment", response(text), "sse", RESPONSES)["status"] == "invalid"


def test_author_conversion_references_grid_and_recovery(draft):
    draft["objects"].append({"id": "plate", "noun": "well plate", "kind": "grid", "rows": 1, "cols": 3,
                              "naming": "A1", "channel": "b"})
    inserted = {k: v for k, v in draft["steps"][0].items() if k != "recovery"}
    inserted["id"] = "fix"
    draft["inserted_steps"] = [inserted]
    draft["steps"][0]["recovery"] = [{"on": "judge", "action": "insert_step", "cells": [], "steps": [],
                                      "insert_step_id": "fix", "say": "Try again."}]
    out = parse_response("author", response(json.dumps(draft)), "sse", RESPONSES)
    assert out["status"] == "ok", out
    graph = out["result"]["stepgraph"]
    assert graph["objects"][1]["grid"]["channel"] == "b"
    graph = StepGraph.model_validate({**graph, "id": "test", "source": {"kind": "text", "ref": "test",
                                    "licence": "Apache-2.0", "attribution": "Own text"}})
    assert validate_semantics(graph) == []
    assert graph.steps[0].recovery[0].step.id == "fix"


def test_author_validation_errors(draft):
    draft["steps"][0]["targets"] = ["missing"]
    out = parse_response("author", response(json.dumps(draft)), "sse", RESPONSES)
    assert out["status"] == "invalid" and "does not resolve" in out["error"]


@pytest.mark.parametrize("backend,model", [("meta", META_MODELS["verdict"]), ("chat", "muse-glimmer"),
                                          ("chat", "cx/gpt-5.6-luna-low")])
def test_request_fingerprint_stability_and_media_refs(backend, model):
    args = dict(model=model, backend=backend, frame_sha=SHA, ctx=CTX)
    endpoint, req = build_request("verdict", **args)
    assert request_fingerprint(req) == request_fingerprint(build_request("verdict", **args)[1])
    changed = {**req, "stream": not req["stream"], "stream_options": {"include_usage": False}}
    assert request_fingerprint(req) == request_fingerprint(changed)
    assert SHA in canonical(req) and "data:image" not in canonical(req)
    assert set(PROMPT_VERSIONS) == {"verdict", "ground", "why", "author", "segment"}
    if backend == "meta":
        assert endpoint == RESPONSES and req["store"] is False
        assert req["reasoning"] == {"effort": "minimal"} and req["max_output_tokens"] >= 4000
        assert req["text"]["format"]["strict"] is True
    else:
        assert endpoint == CHAT and "response_format" not in req
        if model == "muse-glimmer":
            assert [req[k] for k in ("temperature", "top_p", "top_k", "min_p")] == [1., .95, 64, 0]
            assert req["chat_template_kwargs"] == {"reasoning_strength": "low"}
        else:
            assert not (set(req) & {"temperature", "top_p", "top_k", "min_p", "chat_template_kwargs"})


@pytest.mark.parametrize("backend,token_key", [("meta", "max_output_tokens"), ("chat", "max_tokens")])
def test_repair_and_length_are_new_requests(backend, token_key):
    args = dict(model="muse-glimmer" if backend == "chat" else META_MODELS["verdict"], backend=backend, frame_sha=SHA)
    _, base = build_request("verdict", **args, ctx=CTX)
    _, length = build_request("verdict", **args, ctx={**CTX, "retry": {"reason": "length"}})
    _, repair = build_request("verdict", **args, ctx={**CTX, "retry": {"reason": "invalid", "previous": "bad", "error": "not JSON"}})
    assert length[token_key] == base[token_key] * 2
    assert len({request_fingerprint(r) for r in (base, length, repair)}) == 3
    assert '"assistant"' in canonical(repair) and "not JSON" in canonical(repair)


def test_sam_noun_box_point_requests():
    args = dict(model="sam-3.1", backend="meta", frame_sha=SHA)
    _, noun = build_request("segment", **args, ctx={"concept": "blue rectangle"})
    assert noun["metadata"] == {"include_confidence": "true", "mask_encoding": "one_bit"}
    ctx = {"w": 100, "h": 80, "object_prompts": [{"id": 3, "box": [5, 7, 20, 30], "points": [[8, 9]], "negative": [[10, 11]]}]}
    _, box = build_request("segment", **args, ctx=ctx)
    text = box["input"][0]["content"][0]["text"]
    assert "x2=19;y2=29" in text and "-<|point;x=10;y=11;w=100;h=80|>" in text
    ctx["object_prompts"] *= 17
    with pytest.raises(ValueError, match="at most 16"):
        build_request("segment", **args, ctx=ctx)
    with pytest.raises(ValueError, match="meta backend"):
        build_request("segment", model="qwen", backend="chat", frame_sha=SHA, ctx={"concept": "remote"})


def test_pdf_author_and_text_context(draft):
    _, req = build_request("author", model="muse-wonder-1.0", backend="meta", frame_sha=SHA, ctx={"filename": "procedure.pdf"})
    assert req["input"][0]["content"][0] == {"type": "input_file", "filename": "procedure.pdf", "file_data": {"$frame": SHA}}
    with pytest.raises(ValueError, match="PDF sources"):
        build_request("author", model="muse-glimmer", backend="chat", frame_sha=SHA, ctx={"filename": "x.pdf"})
    from jsonschema import Draft202012Validator
    Draft202012Validator(req["text"]["format"]["schema"]).validate(draft)


def test_prices_and_aggregate_usage():
    u = {"input_tokens": 2000, "input_tokens_details": {"cached_tokens": 1000}, "output_tokens": 1000,
         "output_tokens_details": {"reasoning_tokens": 800}}
    assert cost_usd("muse-spark-1.3", u) == .00565
    assert cost_usd("muse-spark-1.3-contributor", u) == .000302
    assert cost_usd("muse-wonder-1.0", u) == .0242
    assert cost_usd("sam-3.1", {"images_processed": 2}) == .005
    assert cost_usd("sam-3.1", {"video_frames_processed": 50}) == .01
    assert cost_usd("muse-glimmer", u) == 0 and cost_usd("muse-spark-1.3", None) == 0
    combined = aggregate_usage([u, None, u])
    assert combined["input_tokens"] == 4000 and combined["input_tokens_details"]["cached_tokens"] == 2000
    assert u["input_tokens"] == 2000 and aggregate_usage([None, None]) is None


def test_recorded_live_fixture_parsing():
    root = Path(__file__).parents[1] / "fixtures" / "calls" / "chat"
    calls = sorted(root.glob("calls/*/meta.json"))
    assert len(calls) >= 2, "record omniroute and Glimmer live smoke fixtures"
    models = set()
    for meta_path in calls:
        meta = json.loads(meta_path.read_text())
        request = json.loads((meta_path.parent / "request.json").read_text())
        models.add(meta["model"])
        assert meta["fingerprint"] == request_fingerprint(request)
        assert meta["attempts"] and meta["http_status"] == 200
        raw_path = next(meta_path.parent.glob("response.*"))
        parsed = parse_response("verdict", raw_path.read_bytes(), raw_path.suffix[1:], meta["endpoint"])
        assert parsed["status"] == "ok", parsed
        assert "data:image" not in canonical(request) and "Authorization" not in canonical(request)
    assert {"muse-glimmer", "cx/gpt-5.6-luna-low"} <= models


def test_a_sam_mask_capped_at_1024_is_scaled_back_to_its_box():
    """Recorded 4 Nov: SAM returned 1024x616 for a box of 1028x618, and that one object lost the whole call its outlines."""
    from pathlib import Path

    from lightwork.core.masks import rle_decode
    from lightwork.models.parse import parse_response
    from lightwork.models.wire import RESPONSES

    body = (Path(__file__).parents[1] / "fixtures/calls/sam-capped-mask/response.sse").read_bytes()
    out = parse_response("segment", body, "sse", RESPONSES)
    assert out["status"] == "ok", out["error"]
    assert {m["object_id"] for m in out["result"]["masks"]} == {100001, 100002}
    big = next(m for m in out["result"]["masks"] if m["box"][2] - m["box"][0] == 1028)
    bits = rle_decode(big["rle"])
    assert bits.shape == (618, 1028) and int(bits.sum()) == big["area_px"]
