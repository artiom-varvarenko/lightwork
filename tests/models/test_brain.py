"""lightwork.brain: open-weight providers (partners), failover, recording, Meta-only model ids."""

from __future__ import annotations

import json

import httpx
import pytest

from lightwork.brain import BrainError, OpenBrain, Provider, parse_json, providers

JPEG = b"\xff\xd8\xff\xe0fake-jpeg"


def ok_body(text: str, model: str = "meta-models/muse-spark-open") -> dict:
    return {"id": "cmpl-1", "model": model, "choices": [{"message": {"role": "assistant", "content": text},
                                                         "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5}}


def test_providers_from_env_in_order():
    env = {"BASETEN_API_KEY": "b", "FIREWORKS_API_KEY": "f", "MODAL_LLM_URL": "https://x.modal.run/v1/",
           "LW_CHAT_BASE_URL": "http://127.0.0.1:8080/v1"}
    names = [p.name for p in providers(env)]
    assert names == ["baseten", "fireworks", "modal", "local"]
    assert [p.name for p in providers({**env, "LW_OPEN_ORDER": "fireworks,local"})] == ["fireworks", "local"]
    b = providers({"BASETEN_API_KEY": "b", "BASETEN_MODEL": "meta-models/muse-spark-open-1026"})[0]
    assert b.base_url == "https://inference.baseten.co/v1" and b.model.endswith("-1026")
    assert providers({}) == []


def test_only_meta_open_weight_models_are_accepted(tmp_path):
    with pytest.raises(BrainError, match="not a Meta open-weight model"):
        OpenBrain([Provider("x", "X", "https://x/v1", "gpt-4o", "k")], run_dir=tmp_path)
    OpenBrain([Provider("m", "M", "https://x/v1", "muse-glimmer", None)], run_dir=tmp_path)


def test_a_call_is_recorded_with_images_by_hash(tmp_path):
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["body"] = json.loads(request.content)
        seen["auth"] = request.headers.get("authorization")
        return httpx.Response(200, json=ok_body('{"names": ["whisk"]}'))

    p = Provider("baseten", "Baseten", "https://inference.baseten.co/v1", "meta-models/muse-spark-open", "key")
    brain = OpenBrain([p], run_dir=tmp_path, http=httpx.Client(transport=httpx.MockTransport(handler)))
    r = brain.ask("name", "Name it.", [JPEG], schema={"type": "object"})
    assert r.data == {"names": ["whisk"]} and r.provider == "baseten"
    assert seen["auth"] == "Bearer key"
    assert seen["body"]["response_format"] == {"type": "json_object"}
    assert seen["body"]["messages"][0]["content"][1]["image_url"]["url"].startswith("data:image/jpeg;base64,")
    call = next((tmp_path / "calls").iterdir())
    rec = json.loads((call / "request.json").read_text())
    assert "$frame" in rec["messages"][0]["content"][1]["image_url"]
    meta = json.loads((call / "meta.json").read_text())
    assert meta["provider"] == "baseten" and meta["status"] == "ok" and meta["base_host"] == "inference.baseten.co"
    assert len(list((tmp_path / "frames").iterdir())) == 1
    assert "key" not in (call / "request.json").read_text()


def test_failover_to_the_next_partner_and_cooldown(tmp_path):
    hits = []

    def handler(request: httpx.Request) -> httpx.Response:
        hits.append(request.url.host)
        if request.url.host == "inference.baseten.co":
            return httpx.Response(503, text="busy")
        return httpx.Response(200, json=ok_body("a whisk", "accounts/fireworks/models/muse-spark-open"))

    provs = [Provider("baseten", "Baseten", "https://inference.baseten.co/v1", "meta-models/muse-spark-open", "b"),
             Provider("fireworks", "Fireworks", "https://api.fireworks.ai/inference/v1",
                      "accounts/fireworks/models/muse-spark-open", "f")]
    t = [100.0]
    brain = OpenBrain(provs, run_dir=tmp_path, http=httpx.Client(transport=httpx.MockTransport(handler)),
                      clock=lambda: t[0])
    assert brain.ask("ask", "What is this?").provider == "fireworks"
    assert hits == ["inference.baseten.co", "api.fireworks.ai"]
    brain.ask("ask", "And this?")  # baseten is cooling down: not tried
    assert hits[-1] == "api.fireworks.ai" and hits.count("inference.baseten.co") == 1
    t[0] += 61.0
    brain.ask("ask", "Again?")
    assert hits.count("inference.baseten.co") == 2


def test_a_400_is_retried_without_the_optional_fields(tmp_path):
    bodies = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        bodies.append(body)
        if "response_format" in body:
            return httpx.Response(400, json={"error": "response_format not supported"})
        return httpx.Response(200, json=ok_body('{"ok": true, "say": "fine"}'))

    p = Provider("modal", "Modal", "https://x.modal.run/v1", "muse-glimmer", None, effort=False)
    brain = OpenBrain([p], run_dir=tmp_path, http=httpx.Client(transport=httpx.MockTransport(handler)))
    r = brain.ask("check", "Is it smooth?", schema={"type": "object"})
    assert r.data == {"ok": True, "say": "fine"}
    assert "response_format" in bodies[0] and "response_format" not in bodies[1]


def test_all_failing_raises_with_every_reason(tmp_path):
    p = Provider("local", "own", "http://127.0.0.1:9/v1", "muse-glimmer", None)
    brain = OpenBrain([p], run_dir=tmp_path,
                      http=httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(500, text="boom"))))
    with pytest.raises(BrainError, match="local: HTTP 500"):
        brain.ask("ask", "hi")


@pytest.mark.parametrize("text", ['{"a": 1}', 'Sure:\n```json\n{"a": 1}\n```', 'thinking... {"a": 1} done'])
def test_parse_json(text):
    assert parse_json(text) == {"a": 1}
