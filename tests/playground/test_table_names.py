"""lightwork play --table --names: object names from a namer, announced once named; Meta calls recorded and costed."""

from __future__ import annotations

import json
import random

import httpx
import pytest

from lightwork.config import Settings
from lightwork.models.budget import Budget
from lightwork.models.client import ModelClient
from lightwork.models.synthetic import responses_sse
from lightwork.namer import PROMPT, MetaNamer, clean_name
from lightwork.tabletop import TableSession
from lightwork.vision.source import Frame

from .test_tabletop import FH, FW, GEO, Feed, page_rect, put, table


class FakeNamer:
    """Answers each request after ``delay`` s of session time."""

    label = "fake"
    error = None

    def __init__(self, names, delay: float = 0.5):
        self.names, self.delay = list(names), delay
        self.asked: dict[object, float] = {}
        self.crops: list[bytes] = []
        self.now = 0.0

    def request(self, key, jpeg):
        self.asked[key] = self.now
        self.crops.append(jpeg)
        return True

    def result(self, key):
        if key in self.asked and self.now - self.asked[key] >= self.delay:
            del self.asked[key]
            return True, self.names.pop(0) if self.names else None
        return False, None

    def summary(self):
        return "fake"


def session_with(namer, seconds: float, scene_at):
    feed = Feed()
    ts = TableSession(GEO, (FW, FH), namer=namer)
    ts.zone.learn(feed(table()))
    said, t, n = [], 0.0, 0
    while t < seconds:
        t += 1 / 30
        n += 1
        namer.now = t
        said += ts.step(Frame(feed(scene_at(t)), t, n), [], t, 1 / 30)[1]
        ts.scene(t)
    return ts, said


def test_a_new_object_is_named_and_announced_with_its_name():
    namer = FakeNamer(["hammer"])
    obj = put(table(), page_rect(300, 200, 180, 50, 10), (40, 60, 200))
    ts, said = session_with(namer, 3.0, lambda t: obj)
    (tr,) = ts.tracks.confirmed
    assert tr.name == "hammer" and len(namer.crops) == 1  # asked once
    assert said == [said[0]] and said[0].startswith("New hammer, about") or said[0].startswith("New hammer")
    labels = [p.text for p in ts.scene(4.0) if p.kind == "label" and p.id.startswith("obj-")]
    assert labels[0] == "HAMMER"
    assert namer.crops[0][:3] == b"\xff\xd8\xff"  # a JPEG


def test_an_unnamed_object_is_still_announced_and_asked_again_later():
    namer = FakeNamer([], delay=0.2)  # every request comes back without a name
    obj = put(table(), page_rect(300, 200, 120, 70, 0), (40, 60, 200))
    ts, said = session_with(namer, 8.0, lambda t: obj)
    assert said and said[0].startswith("New red object")  # announced after NAME_WAIT_S without a name
    assert len(namer.crops) == 3  # asked again every NAME_RETRY_S
    assert ts.tracks.confirmed[0].name is None


def test_clean_name():
    assert clean_name("A claw hammer.") == "claw hammer"
    assert clean_name("  Water bottle\n") == "water bottle"
    assert clean_name("phone (face down)") == "phone face down"
    assert clean_name("smartphone face down") == "smartphone face"  # labels stay within MAX_CHARS
    assert clean_name("...") is None


def test_meta_namer_records_each_call_and_counts_its_cost(tmp_path, no_network):
    seen = []

    def handler(req):
        body = json.loads(req.content)
        seen.append(body)
        usage = {"input_tokens": 900, "input_tokens_details": {"cached_tokens": 0}, "output_tokens": 20}
        raw = responses_sse("Claw hammer", model=body["model"], usage=usage)
        return httpx.Response(200, content=raw, headers={"content-type": "text/event-stream"})

    settings = Settings(model_api_key="secret-never-record", root=tmp_path)
    client = ModelClient(settings, run_dir=tmp_path / "calls-run", provenance="meta",
                         budget=Budget(tmp_path / "spend.json", 5.0), cache=None,
                         transport=httpx.MockTransport(handler), rng=random.Random(0))
    namer = MetaNamer(settings, run_dir=tmp_path / "calls-run", client=client)
    try:
        assert namer.check() is None
        jpeg = b"\xff\xd8\xff" + bytes(range(200))
        assert namer.request("a", jpeg)
        for _ in range(200):
            done, name = namer.result("a")
            if done:
                break
            import time

            time.sleep(0.01)
        assert done and name == "claw hammer"
        (body,) = seen
        assert body["model"] == "muse-spark-1.3-contributor"
        content = body["input"][0]["content"]
        assert content[0] == {"type": "input_text", "text": PROMPT}
        assert content[1]["image_url"].startswith("data:image/jpeg;base64,")
        calls = list((tmp_path / "calls-run" / "calls").iterdir())
        assert len(calls) == 1 and (calls[0] / "meta.json").is_file()
        assert namer.cost_usd == pytest.approx((900 * 0.10 + 20 * 0.20) / 1e6)
        assert json.loads((tmp_path / "spend.json").read_text())["reservations"] == {}
        assert "1 call" in namer.summary() and "$0.0001" in namer.summary()
        assert list((tmp_path / "calls-run" / "frames").glob("*.jpg"))
    finally:
        namer.close()


def test_meta_namer_without_a_key_says_so(tmp_path):
    namer = MetaNamer(Settings(model_api_key=None, root=tmp_path), run_dir=tmp_path / "r", client=object.__new__(ModelClient))
    try:
        assert "MODEL_API_KEY" in namer.check()
    finally:
        namer._q.put_nowait(None)


def test_several_crops_go_in_one_call_with_a_json_answer(tmp_path, no_network):
    from lightwork.namer import BATCH_PROMPT

    seen = []

    def handler(req):
        body = json.loads(req.content)
        seen.append(body)
        raw = responses_sse('{"names": ["Claw hammer", "TV remote!"]}', model=body["model"],
                            usage={"input_tokens": 1800, "input_tokens_details": {"cached_tokens": 0}, "output_tokens": 30})
        return httpx.Response(200, content=raw, headers={"content-type": "text/event-stream"})

    settings = Settings(model_api_key="secret-never-record", root=tmp_path)
    client = ModelClient(settings, run_dir=tmp_path / "run", provenance="meta",
                         budget=Budget(tmp_path / "spend.json", 5.0), cache=None,
                         transport=httpx.MockTransport(handler), rng=random.Random(0))
    namer = MetaNamer(settings, run_dir=tmp_path / "run", client=client)
    try:
        names = namer._name_batch([b"\xff\xd8\xff" + bytes([1] * 50), b"\xff\xd8\xff" + bytes([2] * 50)])
        assert names == ["claw hammer", "tv remote"]
        (body,) = seen
        content = body["input"][0]["content"]
        assert content[0]["text"] == BATCH_PROMPT and [c["type"] for c in content[1:]] == ["input_text", "input_image"] * 2
        assert body["text"]["format"]["type"] == "json_schema"
    finally:
        namer.close()


def test_crops_queued_while_a_call_is_out_go_together_in_the_next():
    import threading

    from lightwork.namer import _Namer

    gate = threading.Event()

    class Slow(_Namer):
        batch_max = 8

        def __init__(self):
            self.batches = []
            super().__init__(workers=1)

        def _name_batch(self, jpegs):
            self.batches.append(len(jpegs))
            gate.wait(2.0)
            return [f"thing {len(self.batches)}"] * len(jpegs)

    n = Slow()
    try:
        assert n.request("a", b"1")
        import time

        time.sleep(0.1)  # the worker took "a" and is waiting on the model
        for k in "bcd":
            assert n.request(k, k.encode())
        gate.set()
        for _ in range(200):
            if len(n._done) == 4:
                break
            time.sleep(0.01)
        assert n.batches == [1, 3]
        assert n.result("a") == (True, "thing 1") and n.result("c") == (True, "thing 2")
    finally:
        n.close()
