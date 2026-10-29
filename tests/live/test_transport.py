"""The live dispatcher uses F unchanged, including raw call records and crop-only media."""

import base64
import json
import threading

import cv2
import httpx
import numpy as np
import pytest

from lightwork.config import Settings
from lightwork.contracts.events import Event
from lightwork.live import LiveSession
from lightwork.live.diagnostic import FixtureCamera, FixtureClock
from lightwork.models.budget import Budget
from lightwork.models.client import ModelClient
from lightwork.models.synthetic import SamObject, responses_sse, sam_sse
from lightwork.runs import Recorder, verify_run
from lightwork.vision.measure_frames import encode_jpeg

from .test_session import artefacts

pytestmark = pytest.mark.usefixtures("no_network")


def test_real_model_client_records_concurrent_crop_only_calls(tmp_path):
    graph, layout, calib = artefacts()
    settings = Settings(root=tmp_path, model_api_key="synthetic-test-only")
    clock = FixtureClock()
    recorder = Recorder(tmp_path, graph.id, {"backend": "meta", "models": {
        "verdict": "muse-spark-1.3-contributor", "segment": "sam-3.1"}, "code_commit": "0000000",
        "dirty": False, "platform": "synthetic", "version": "0.1.0"}, provenance="synthetic", clock=clock)
    wire = []
    lock = threading.Lock()

    def respond(request):
        body = json.loads(request.content)
        with lock:
            wire.append(body)
        raw = (sam_sse([SamObject(1, (15, 10, 30, 25))], size=(80, 50)) if body["model"] == "sam-3.1"
               else responses_sse('{"done":"yes","contradiction":null,"evidence":"Synthetic test."}', model=body["model"]))
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=raw)

    client = ModelClient(settings, run_dir=recorder.run_dir, provenance="meta",
                         budget=Budget.from_settings(settings), cache=None, recording=True,
                         transport=httpx.MockTransport(respond))
    camera = FixtureCamera(layout.work_area_xywh, calib.camera.model_dump())
    camera.set_frame(encode_jpeg(np.full((50, 80, 3), 100, np.uint8)), .1)
    session = LiveSession(settings, graph, layout, calib, source=camera, recorder=recorder,
                          model_client=client, scripted=True, hands=False, voice=False, quiet=True, clock=clock)
    completed = threading.Event()

    def observe(events):
        if any(e.type == "step" and e.payload["step"] == "s1" and e.payload["state"] == "passed" for e in events):
            completed.set()

    session.subscribe(observe)
    try:
        session.start()
        clock.t = .2
        session.capture_now().result(3)
        clock.t = 1
        for state in ("moving", "settled"):
            session.feed_recorded_input(Event(seq=1, t=1, type="motion", src="gate", derived=False,
                payload={"state": state, "active_regions": ["slot"]})).result(3)
        camera.set_frame(encode_jpeg(np.full((50, 80, 3), 130, np.uint8)), .95)
        session.capture_now().result(3)
        assert completed.wait(5)
        session.stop()
        path = session.join(5)
        assert len(wire) == 4

        def images(value):
            if isinstance(value, str) and value.startswith("data:image/"):
                yield base64.b64decode(value.split(",", 1)[1])
            elif isinstance(value, dict):
                for child in value.values():
                    yield from images(child)
            elif isinstance(value, list):
                for child in value:
                    yield from images(child)

        for request in wire:
            media = list(images(request))
            assert media
            assert all(cv2.imdecode(np.frombuffer(img, np.uint8), cv2.IMREAD_COLOR).shape[:2] == (50, 80) for img in media)
        assert verify_run(path).ok
        assert json.loads((path / "manifest.json").read_bytes())["provenance"] == "synthetic"
    finally:
        session.stop()
        session.join(5)
        session.render.close()
        session.control.close()
