"""lightwork play --table end to end without hardware: the real frame loop, hub and session, with a fake camera, hand
tracker and projector page (it ACKs every scene at once)."""

from __future__ import annotations

import argparse
import time

import numpy as np
import pytest

from lightwork.commands import play as play_cmd
from lightwork.contracts.render import Hello
from lightwork.server.hub import make_hubs
from lightwork.vision.source import Frame

from .test_tabletop import FH, FW, HI, PAGE_H, PAGE_W, Feed, page_rect, put, table


class Cam:
    """Serves the synthetic table in real time; an object lands after ``land_s``; stops the loop after ``stop_s``."""

    def __init__(self, land_s: float, stop_s: float):
        self.t0 = time.monotonic()
        self.land_s, self.stop_s = land_s, stop_s
        self.feed = Feed()
        self.base = table()
        self.obj = put(self.base, page_rect(300, 200, 120, 70, 15), (40, 60, 200))
        self.n = 0

    def wait_frame(self, after_t, timeout):
        time.sleep(1 / 60)
        now = time.monotonic()
        if now - self.t0 > self.stop_s:
            raise KeyboardInterrupt
        self.n += 1
        scene = self.obj if now - self.t0 > self.land_s else self.base
        return Frame(self.feed(scene), now, self.n)

    def latest(self):
        return None


class Tracker:
    def poses(self, frame):
        return []

    def set_region(self, region):
        return [region]


def hello() -> Hello:
    return Hello(role="projector", innerWidth=int(PAGE_W), innerHeight=int(PAGE_H), dpr=1.0, screen={}, orientation="",
                 fullscreen=True, secure=True, wakelock=True, ua="test")


def test_table_mode_runs_scans_and_finds_an_object(monkeypatch, capsys):
    p = argparse.ArgumentParser()
    play_cmd.configure(p)
    args = p.parse_args(["--table", "--no-hand-check", "--quiet", "--no-frame"])
    args.recal_every = 0.0
    hubs = make_hubs()
    render = hubs["render_hub"]
    monkeypatch.setattr(render, "acked_at", lambda version: time.monotonic() - 1.0)  # the page drew it long ago
    sounds = []
    monkeypatch.setattr(render, "sfx", lambda name, gain=1.0: sounds.append(name))
    cam = Cam(land_s=5.0, stop_s=7.5)  # learning takes the 3 s "clear the table" grace first
    play = play_cmd.Play(args, cam, hubs, Tracker(), (FW, FH), gain=0)
    page = hello()

    def calibrate(page_, *, gain):
        play._adopt(np.linalg.inv(HI).tolist(), page_)
        play.calib_t = time.monotonic()

    monkeypatch.setattr(play, "calibrate", calibrate)
    with pytest.raises(KeyboardInterrupt):
        play.run(page)
    t = play.table
    out = capsys.readouterr().out
    assert t.zone.status == "ready" and t.scans >= 1
    assert [tr.colour for tr in t.tracks.confirmed] == ["red"], out
    assert "found" in sounds and "scan" in sounds
    assert any(p.kind == "poly" and p.id.startswith("obj-") for p in render._scene.prims)
