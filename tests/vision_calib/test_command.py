from __future__ import annotations

import argparse
import json
import socket
import threading
import time

import pytest
from websockets.exceptions import ConnectionClosed
from websockets.sync.client import connect

from lightwork.commands import calibrate as command
from lightwork.contracts.render import Ack, SceneMsg
from lightwork.vision.calib.flow import load_calibration
from lightwork.vision.calib.sim import SimCamera


def _args(*args):
    p = argparse.ArgumentParser()
    command.configure(p)
    return p.parse_args(args)


def test_card_command_is_keyless(tmp_path, capsys):
    out = tmp_path / "card.svg"
    assert command.run(_args("card", "--output", str(out))) == 0
    assert out.read_text().startswith("<svg")
    assert "100 mm" in capsys.readouterr().out


def test_flat_command_requires_privacy_selection(monkeypatch, capsys):
    def forbidden(spec):
        raise AssertionError("camera must not open without the required crop")

    monkeypatch.setattr(command, "_open_camera", forbidden)
    assert command.run(_args("flat")) == 2
    assert "--work-area" in capsys.readouterr().out


def test_missing_page_is_environment_error():
    class EmptyHub:
        def hello(self, role):
            return None

    with pytest.raises(RuntimeError, match="projector"):
        command._wait_projector(EmptyHub(), 0.06)


def test_saved_page_geometry_must_match(make_rig, truth_calib, tmp_path):
    from lightwork.vision.calib.flow import save_calibration

    rig, _, page, _ = make_rig()
    path = tmp_path / "calib.json"
    save_calibration(truth_calib(rig, page), path)
    assert command._existing(path, page.hello()).page.w == 854
    with pytest.raises(ValueError, match="changed"):
        command._existing(path, page.hello().model_copy(update={"innerWidth": 900}))


def test_command_real_server_fake_camera_and_page(make_rig, monkeypatch, tmp_path, truth_error, capsys):
    rig, _, page, _ = make_rig()
    camera = SimCamera(rig)
    monkeypatch.setattr(command, "_open_camera", lambda spec: camera)
    with socket.socket() as reservation:
        reservation.bind(("127.0.0.1", 0))
        port = reservation.getsockname()[1]
    stopped = threading.Event()
    errors = []
    patterns = []

    def peer():
        deadline = time.monotonic() + 8
        while not stopped.is_set() and time.monotonic() < deadline:
            try:
                ws = connect(f"ws://127.0.0.1:{port}/ws/render", open_timeout=0.5, proxy=None)
                break
            except OSError:
                time.sleep(0.02)
        else:
            errors.append("fake projector could not connect")
            return
        try:
            with ws:
                ws.send(page.hello().model_dump_json())
                while not stopped.is_set():
                    try:
                        raw = json.loads(ws.recv(timeout=0.2))
                    except TimeoutError:
                        continue
                    if raw["type"] == "calib":
                        page.set_calib(raw["H_cam_to_page"])
                    elif raw["type"] == "scene":
                        scene = SceneMsg.model_validate(raw)
                        drawn = page.draw(scene)
                        if drawn:
                            patterns.append(drawn)
                        ws.send(Ack(seq=scene.seq, drawn=drawn).model_dump_json(exclude_none=True))
        except ConnectionClosed:
            pass
        except Exception as exc:
            errors.append(repr(exc))

    thread = threading.Thread(target=peer, daemon=True)
    thread.start()
    try:
        path = tmp_path / "calib.json"
        code = command.run(_args("--output", str(path), "--port", str(port), "--page-wait", "5", "--mm", "nominal", "--repeat", "2"))
    finally:
        stopped.set()
        thread.join(3)
    assert not thread.is_alive() and not errors
    assert code == 0
    assert camera.closed
    assert len(patterns) >= 2
    assert truth_error(load_calibration(path).H_cam_to_page, rig).max() < 0.4
    assert "run 2/2" in capsys.readouterr().out
