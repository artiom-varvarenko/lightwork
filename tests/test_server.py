from __future__ import annotations

from fastapi.testclient import TestClient

from lightwork.server.app import create_app, discover_routers


def test_routes_discovered_and_health(no_network):
    assert "health" in {name for name, _ in discover_routers()}
    client = TestClient(create_app(mode="replay"))
    r = client.get("/api/health")
    assert r.status_code == 200 and r.json()["mode"] == "replay"
    assert client.get("/static/README.txt").status_code == 200
