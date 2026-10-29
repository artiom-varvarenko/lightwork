from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from lightwork.server.app import WEB_DIR, create_app
from lightwork.server.hub import make_hubs

SHA = "ab" * 32


@pytest.fixture
def runs(tmp_path):
    root = tmp_path / "runs"
    run = root / "20261030T120000Z_kit"
    (run / "frames").mkdir(parents=True)
    (run / "events.jsonl").write_text("{}\n")
    (run / "manifest.json").write_text(json.dumps({"created_utc": "2026-10-30T12:00:00Z", "provenance": "meta",
                                                   "totals": {"passed": 3}, "files": {"x": "y"}}))
    (run / "frames" / f"{SHA}.jpg").write_bytes(b"\xff\xd8run")
    (root / "20261030T130000Z_partial").mkdir()
    (root / "20261030T130000Z_partial" / "events.jsonl").write_text("")
    (root / "not-a-run").mkdir()
    (tmp_path / "secret.txt").write_text("secret")
    (run / "escape.txt").symlink_to(tmp_path / "secret.txt")
    return root


@pytest.fixture
def client(runs, no_network):
    return TestClient(create_app(runs_root=runs, **make_hubs()))


def test_pages_served(client):
    for path in ("/projector", "/screen"):
        r = client.get(path)
        assert r.status_code == 200 and r.headers["content-type"].startswith("text/html")
        assert r.headers["cache-control"] == "no-cache"
    assert client.get("/viewer").status_code == (200 if (WEB_DIR / "viewer.html").is_file() else 404)
    for js in ("scene", "aruco", "link", "tts", "projector", "screen"):
        assert client.get(f"/static/js/{js}.js").status_code == 200


def test_list_runs(client):
    rows = client.get("/api/runs").json()
    assert rows == [
        {"id": "20261030T120000Z_kit", "complete": True, "created_utc": "2026-10-30T12:00:00Z",
         "provenance": "meta", "totals": {"passed": 3}},
        {"id": "20261030T130000Z_partial", "complete": False},
    ]


def test_list_runs_without_root(tmp_path, no_network):
    assert TestClient(create_app(runs_root=tmp_path / "none")).get("/api/runs").json() == []


def test_run_file_and_path_safety(client):
    assert client.get("/api/runs/20261030T120000Z_kit/events.jsonl").text == "{}\n"
    assert client.get(f"/api/runs/20261030T120000Z_kit/frames/{SHA}.jpg").content == b"\xff\xd8run"
    for bad in (
        "/api/runs/20261030T120000Z_kit/escape.txt",  # symlink out of the run
        "/api/runs/20261030T120000Z_kit/..%2F..%2Fsecret.txt",
        "/api/runs/20261030T120000Z_kit/%2Fetc%2Fpasswd",
        "/api/runs/..%2F/secret.txt",
        "/api/runs/../secret.txt",
        "/api/runs/20261030T120000Z_kit/frames",  # a directory
        "/api/runs/nope/events.jsonl",
    ):
        r = client.get(bad)
        assert r.status_code == 404, bad
        assert b"secret" not in r.content


def test_frames(client, tmp_path):
    assert client.get(f"/frames/{SHA}").content == b"\xff\xd8run"
    assert client.get(f"/frames/{SHA.upper()}").status_code == 404
    assert client.get("/frames/..%2Fsecret").status_code == 404
    assert client.get(f"/frames/{'cd' * 32}").status_code == 404
    live = tmp_path / "live-frames"
    live.mkdir()
    (live / f"{'cd' * 32}.jpg").write_bytes(b"live")
    client.app.state.frames_dirs = [live]
    assert client.get(f"/frames/{'cd' * 32}").content == b"live"


def test_api_pages_empty(client):
    assert client.get("/api/pages").json() == {"render": [], "camera": []}
