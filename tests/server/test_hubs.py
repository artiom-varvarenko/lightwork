from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError
from starlette.websockets import WebSocketDisconnect

from lightwork.contracts import render
from lightwork.contracts.server import pack_camera_frame, unpack_camera_frame
from lightwork.server import hub as hub_mod
from lightwork.server.app import create_app
from lightwork.server.hub import make_hubs


def hello(role: str) -> dict:
    return {
        "type": "hello", "role": role, "innerWidth": 915, "innerHeight": 412, "dpr": 2.625,
        "screen": {"width": 915, "height": 412}, "orientation": "landscape-primary", "fullscreen": True,
        "secure": True, "wakelock": True, "ua": "test",
    }


def ring(rid: str) -> render.SceneMsg:
    return render.SceneMsg(seq=0, mode="normal", prims=[render.Ring(id=rid, c=[100.0, 200.0], r=30.0, role="target")])


@pytest.fixture
def hubs(no_network):
    h = make_hubs()
    client = TestClient(create_app(**h))
    with client:
        yield client, h["render_hub"], h["preview_hub"], h["control_hub"]


def ack(ws, seq: int, drawn=None) -> None:
    msg = {"type": "ack", "seq": seq}
    if drawn is not None:
        msg["drawn"] = drawn
    ws.send_json(msg)


def test_connect_gets_calib_then_full_scene(hubs):
    client, rh, _, _ = hubs
    rh.set_calib(render.Calib(H_cam_to_page=[[1, 0, 0], [0, 1, 0], [0, 0, 1]], page_size=[915, 412]))
    rh.publish_scene(ring("a"))
    with client.websocket_connect("/ws/render") as ws:
        ws.send_json(hello("projector"))
        assert ws.receive_json()["type"] == "calib"
        scene = ws.receive_json()
        assert scene["type"] == "scene" and scene["seq"] == 1 and scene["prims"][0]["id"] == "a"
        assert rh.hello("projector").innerWidth == 915 and rh.hello("screen") is None


def test_view_reaches_a_page_before_anything_it_turns(hubs):
    """The pattern must be drawn turned too, or the mapping solved from it does not hold for turned scenes."""
    client, rh, _, _ = hubs
    rh.set_view(render.View(rotate=180))
    rh.set_calib(render.Calib(H_cam_to_page=[[1, 0, 0], [0, 1, 0], [0, 0, 1]], page_size=[915, 412]))
    rh.publish_scene(ring("a"))
    with client.websocket_connect("/ws/render") as ws:
        ws.send_json(hello("projector"))
        assert ws.receive_json() == {"type": "view", "rotate": 180}
        assert ws.receive_json()["type"] == "calib"
        assert ws.receive_json()["type"] == "scene"
        ack(ws, 1)
        rh.set_view(render.View(rotate=0))  # a change reaches connected pages
        assert ws.receive_json() == {"type": "view", "rotate": 0}


def test_no_view_message_unless_one_is_set(hubs):
    client, rh, _, _ = hubs
    rh.publish_scene(ring("a"))
    with client.websocket_connect("/ws/render") as ws:
        ws.send_json(hello("projector"))
        assert ws.receive_json()["type"] == "scene"


def test_latest_wins_and_at_most_two_unacked(hubs):
    client, rh, _, _ = hubs
    with client.websocket_connect("/ws/render") as ws:
        ws.send_json(hello("projector"))
        assert ws.receive_json()["seq"] == 1
        rh.publish_scene(ring("a"))
        s2 = ws.receive_json()
        assert (s2["seq"], s2["prims"][0]["id"]) == (2, "a")
        for rid in ("b", "c", "d"):  # 2 unacked: none of these may be sent yet
            rh.publish_scene(ring(rid))
        ack(ws, 1)
        s3 = ws.receive_json()
        assert (s3["seq"], s3["prims"][0]["id"]) == (3, "d")  # b, c skipped
        ack(ws, 3)  # covers 2 as well
        rh.publish_scene(ring("e"))
        assert ws.receive_json()["prims"][0]["id"] == "e"
        st = rh.stats()[0]
        assert st["max_unacked"] == 2 and st["acks"] == 2 and st["ack_p50_ms"] is not None


def test_heartbeat_resends_current_scene(hubs, monkeypatch):
    monkeypatch.setattr(hub_mod, "HEARTBEAT_S", 0.05)
    client, rh, _, _ = hubs
    rh.publish_scene(ring("a"))
    with client.websocket_connect("/ws/render") as ws:
        ws.send_json(hello("screen"))
        first = ws.receive_json()
        ack(ws, first["seq"])
        again = ws.receive_json()
        assert again["seq"] == first["seq"] + 1 and again["prims"] == first["prims"]


def test_show_mode_waits_for_ack_and_normal_restores(hubs):
    client, rh, _, _ = hubs
    rh.publish_scene(ring("a"))
    with client.websocket_connect("/ws/render") as ws, ThreadPoolExecutor(1) as pool:
        ws.send_json(hello("projector"))
        ack(ws, ws.receive_json()["seq"])
        fut = pool.submit(rh.show_mode, "grey", None, 2.0)
        grey = ws.receive_json()
        assert grey["mode"] == "grey" and grey["prims"] == []
        rh.publish_scene(ring("b"))  # kept while the measurement mode is up, not sent
        ack(ws, grey["seq"])
        t_ack = fut.result(3)
        assert isinstance(t_ack, float)
        fut = pool.submit(rh.show_mode, "normal", None, 2.0)
        normal = ws.receive_json()
        assert normal["mode"] == "normal" and normal["prims"][0]["id"] == "b"
        ack(ws, normal["seq"])
        assert fut.result(3) >= t_ack
        # no ACK -> None after the timeout
        fut = pool.submit(rh.show_mode, "dark", None, 0.2)
        assert ws.receive_json()["mode"] == "dark"
        assert fut.result(3) is None


def test_show_mode_pattern_ack_carries_drawn_rects(hubs):
    client, rh, _, _ = hubs
    pattern = render.Pattern(markers=[render.MarkerRect(id=i, x=40.0 + 100 * i, y=40.0, size=60.0) for i in range(3)])
    with client.websocket_connect("/ws/render") as ws, ThreadPoolExecutor(1) as pool:
        ws.send_json(hello("projector"))
        ws.receive_json()
        fut = pool.submit(rh.show_mode, "pattern", pattern, 2.0)
        msg = ws.receive_json()
        while msg["mode"] != "pattern":
            msg = ws.receive_json()
        assert msg["pattern"]["aruco_dict"] == "DICT_4X4_50" and len(msg["pattern"]["markers"]) == 3
        drawn = [{"id": m["id"], "x": m["x"], "y": m["y"], "size": 60.952} for m in msg["pattern"]["markers"]]
        ack(ws, msg["seq"], drawn)
        assert fut.result(3) is not None
        assert [m.size for m in rh.last_ack().drawn] == [60.952] * 3


def test_show_mode_argument_checks():
    rh = hub_mod.RenderHub()
    with pytest.raises(ValueError):
        rh.show_mode("blue")
    with pytest.raises(ValueError):
        rh.show_mode("pattern")
    with pytest.raises(ValueError):
        rh.show_mode("grey", render.Pattern(markers=[]))
    assert rh.show_mode("grey", timeout=0.05) is None  # no page at all


def test_projector_is_the_link_page_and_screen_the_fallback(hubs):
    client, rh, _, _ = hubs
    with client.websocket_connect("/ws/render") as scr, ThreadPoolExecutor(1) as pool:
        scr.send_json(hello("screen"))
        scr.receive_json()
        fut = pool.submit(rh.show_mode, "grey", None, 2.0)
        ack(scr, scr.receive_json()["seq"])
        assert fut.result(3) is not None  # screen-only: the screen's ACK counts
        with client.websocket_connect("/ws/render") as proj:
            proj.send_json(hello("projector"))
            grey = proj.receive_json()
            assert grey["mode"] == "grey"
            ack(proj, grey["seq"])
            fut = pool.submit(rh.show_mode, "dark", None, 0.3)
            assert proj.receive_json()["mode"] == "dark"  # drawn but deliberately not acked
            ack(scr, scr.receive_json()["seq"])
            assert fut.result(3) is None  # projector connected: only its ACK counts
            fut = pool.submit(rh.show_mode, "normal", None, 2.0)
            msg = proj.receive_json()
            assert msg["mode"] == "normal"  # dark was never acked here: 1 unacked, so normal still fits
            ack(proj, msg["seq"])
            assert fut.result(3) is not None


def test_say_goes_to_projector_else_screen_and_tts_reports_back(hubs):
    client, rh, _, _ = hubs
    got: list[render.Tts] = []
    done = threading.Event()
    rh.on_tts(lambda t: (got.append(t), done.set()))
    with client.websocket_connect("/ws/render") as scr:
        scr.send_json(hello("screen"))
        scr.receive_json()
        sid = rh.say("Pipette into A2", "interrupt")
        msg = scr.receive_json()
        assert msg == {"type": "say", "id": sid, "text": "Pipette into A2", "priority": "interrupt"}
        with client.websocket_connect("/ws/render") as proj:
            proj.send_json(hello("projector"))
            proj.receive_json()
            sid2 = rh.say("Next")
            assert proj.receive_json()["id"] == sid2
            proj.send_json({"type": "tts", "id": sid2, "state": "start"})
            assert done.wait(2)
    assert got == [render.Tts(id=sid2, state="start")]


def test_sfx_goes_to_projector_else_screen_before_the_scene(hubs):
    client, rh, _, _ = hubs
    rh.sfx("scan")  # no page: dropped, never delivered later
    with client.websocket_connect("/ws/render") as scr:
        scr.send_json(hello("screen"))
        assert scr.receive_json()["type"] == "scene"
        rh.sfx("found", 0.5)
        assert scr.receive_json() == {"type": "sfx", "name": "found", "gain": 0.5}
        with client.websocket_connect("/ws/render") as proj:
            proj.send_json(hello("projector"))
            assert proj.receive_json()["type"] == "scene"
            sid = rh.say("Found it")
            rh.sfx("answer", 2.0)  # clamped to the contract's 0..1
            rh.publish_scene(ring("a"))
            assert proj.receive_json()["id"] == sid
            assert proj.receive_json() == {"type": "sfx", "name": "answer", "gain": 1.0}
            assert proj.receive_json()["prims"][0]["id"] == "a"
            assert scr.receive_json()["prims"][0]["id"] == "a"  # projector connected: no sound on the screen
    with pytest.raises(ValidationError):
        rh.sfx("boom")


def test_render_requires_hello_first(hubs):
    client, _, _, _ = hubs
    with client.websocket_connect("/ws/render") as ws:
        ws.send_json({"type": "ack", "seq": 1})
        with pytest.raises(WebSocketDisconnect) as e:
            ws.receive_json()
        assert e.value.code == 1008


def test_missing_hub_closes_1011(no_network):
    client = TestClient(create_app())
    with client.websocket_connect("/ws/camera") as ws:
        with pytest.raises(WebSocketDisconnect) as e:
            ws.receive_bytes()
        assert e.value.code == 1011


def test_camera_header_round_trip():
    data = pack_camera_frame(2**32 + 7, 12345.678901, b"\xff\xd8jpeg")
    assert len(data) == 12 + 6
    assert unpack_camera_frame(data) == (7, 12345.678901, b"\xff\xd8jpeg")


def test_preview_latest_wins_with_two_in_flight(hubs):
    client, _, ph, _ = hubs
    assert not ph.has_viewers
    ph.push(1, 10.0, b"f1")
    with client.websocket_connect("/ws/camera") as ws:
        assert unpack_camera_frame(ws.receive_bytes()) == (1, 10.0, b"f1")
        assert ph.has_viewers
        ph.push(2, 10.033, b"f2")
        assert unpack_camera_frame(ws.receive_bytes())[0] == 2
        ph.push(3, 10.066, b"f3")  # 2 in flight: held back, then superseded
        ph.push(4, 10.1, b"f4")
        ws.send_text('{"ack": 2}')  # covers frame 1 too
        assert unpack_camera_frame(ws.receive_bytes()) == (4, 10.1, b"f4")
        ws.send_text("not json")  # ignored
        ws.send_text('{"ack": 4}')
        ph.push(5, 10.133, b"f5")
        assert unpack_camera_frame(ws.receive_bytes())[0] == 5
        st = ph.stats()[0]
        assert (st["pushed"], st["sent"]) == (5, 4) and st["skipped"] == pytest.approx(0.2)


def test_control_state_merge_and_actions(hubs):
    client, _, _, ch = hubs
    actions: list[tuple[str, dict]] = []
    got = threading.Event()
    ch.on_action(lambda a, args: (actions.append((a, args)), got.set()))
    ch.push_state({"run": {"mode": "live"}, "steps": []})
    with client.websocket_connect("/ws/control") as ws:
        assert ws.receive_json() == {"type": "state", "state": {"run": {"mode": "live"}, "steps": []}}
        ch.push_state({"hud": {"voice": "live"}})
        assert ws.receive_json()["state"] == {"run": {"mode": "live"}, "steps": [], "hud": {"voice": "live"}}
        ws.send_text("garbage")
        ws.send_json({"type": "nope"})
        ws.send_json({"type": "ui", "action": "layout_rect", "args": {"object": "tray", "xywh": [1, 2, 3, 4]}})
        assert got.wait(2)
    assert actions == [("layout_rect", {"object": "tray", "xywh": [1, 2, 3, 4]})]
    assert ch.state()["hud"] == {"voice": "live"}


def test_encode_preview_size():
    import cv2
    import numpy as np

    jpeg = hub_mod.encode_preview(np.full((1080, 1920, 3), 128, np.uint8))
    img = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
    assert img.shape == (540, 960, 3)
