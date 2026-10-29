from __future__ import annotations

import importlib.util
import json
import resource
import subprocess
import sys
from pathlib import Path

import httpx
import pytest

from lightwork.contracts.events import Hands
from lightwork.contracts.layout import Layout
from lightwork.contracts.stepgraph import StepGraph
from lightwork.geometry import Regions
from lightwork.vision import hands
from lightwork.vision.hands import HandEvents, HandObs, HandsUnavailable, ensure_model

REPO = Path(__file__).resolve().parents[2]


def _regions() -> Regions:
    # 2x3 plate, pitch 60: cell A1 spans x 70..130, y 70..130; A2 spans x 130..190. Zone "bin" is 500..600 square.
    sg = StepGraph.model_validate({
        "id": "p", "title": "p", "source": {"kind": "text", "ref": "x", "licence": "x", "attribution": "x"},
        "objects": [
            {"id": "plate", "noun": "well plate", "kind": "grid", "grid": {"rows": 2, "cols": 3, "naming": "A1"}},
            {"id": "bin", "noun": "bin", "kind": "zone"},
        ],
        "steps": [{"id": "s", "say": "x", "checks": [{"id": "c", "kind": "confirm"}]}],
    })
    lay = Layout.model_validate({"pack": "p", "created": "x", "work_area_xywh": [0, 0, 1280, 720], "objects": {
        "plate": {"kind": "grid", "corners_cam_px": [[100, 100], [220, 100], [220, 160], [100, 160]], "sam": {"mode": "noun", "noun": "well plate"}},
        "bin": {"kind": "zone", "polygon_cam_px": [[500, 500], [600, 500], [600, 600], [500, 600]], "sam": {"mode": "noun", "noun": "bin"}},
    }})
    return Regions(lay, sg)


def at(x: float, y: float, *more: tuple[float, float]) -> HandObs:
    return HandObs(present=True, tip_xy=(x, y), tips=((x, y), *more))


NONE = HandObs(present=False, tip_xy=None)


def test_no_hand_emits_nothing():
    ev = HandEvents(_regions())
    assert ev.update(NONE, 0.0) == [] and ev.update(NONE, 1.0) == []


def test_appearance_payload_is_c1_and_quantized():
    ev = HandEvents(_regions())
    out = ev.update(at(100.04, 99.96), 0.0)
    assert out == [("hands", {"over_regions": ["plate", "plate:A1"], "tip_xy": [100.0, 100.0]})]
    Hands.model_validate(out[0][1])


def test_only_changes_are_emitted():
    ev = HandEvents(_regions())
    ev.update(at(100, 100), 0.0)
    for k in range(1, 30):  # 1 px jitter for a second: below min_move, same regions
        assert ev.update(at(100 + (k % 2), 100), k / 30) == []
    assert ev.update(at(110, 100), 1.0) == [("hands", {"over_regions": ["plate", "plate:A1"], "tip_xy": [110.0, 100.0]})]


def test_tip_updates_are_capped_at_5hz():
    ev = HandEvents(_regions())
    emitted = []
    for k in range(90):  # 3 s at 30 fps, tip sweeping 4 px per frame through the bin
        t = k / 30
        for _ in ev.update(at(510 + 4 * (k % 20), 550), t):
            emitted.append(t)
    gaps = [b - a for a, b in zip(emitted, emitted[1:])]
    assert len(emitted) <= 3 * 5 + 1 and min(gaps) >= 0.2 - 1e-9
    assert len(emitted) >= 12  # still tracks the moving tip near the cap (float frame times give 6- or 7-frame gaps)


def test_change_inside_window_is_delivered_after_it():
    ev = HandEvents(_regions())
    ev.update(at(100, 100), 0.0)
    assert ev.update(at(550, 550), 0.05) == []  # moved to the bin inside the 0.2 s window
    assert ev.update(at(550, 550), 0.1) == []
    assert ev.update(at(550, 550), 0.2) == [("hands", {"over_regions": ["bin"], "tip_xy": [550.0, 550.0]})]


def test_short_visit_reverted_inside_window_is_not_emitted():
    ev = HandEvents(_regions())
    ev.update(at(100, 100), 0.0)
    ev.update(at(550, 550), 0.05)
    assert ev.update(at(100, 100), 0.21) == []  # back where the last event left it


def test_region_membership_grid_cells_and_zone():
    ev = HandEvents(_regions())
    assert ev.update(at(160, 150), 0.0)[0][1]["over_regions"] == ["plate", "plate:B2"]
    assert ev.update(at(400, 400), 1.0)[0][1]["over_regions"] == []  # hand present, over nothing
    assert ev.update(at(599, 501), 2.0)[0][1]["over_regions"] == ["bin"]


def test_boundary_hysteresis():
    ev = HandEvents(_regions())
    assert ev.update(at(125, 100), 0.0)[0][1]["over_regions"] == ["plate", "plate:A1"]
    # 3 px across the A1|A2 boundary (x=130): A2 entered, A1 held within the 6 px exit margin
    assert ev.update(at(133, 100), 1.0)[0][1]["over_regions"] == ["plate", "plate:A1", "plate:A2"]
    assert ev.update(at(127, 100), 2.0)[0][1]["over_regions"] == ["plate", "plate:A1", "plate:A2"]
    assert ev.update(at(140, 100), 3.0)[0][1]["over_regions"] == ["plate", "plate:A2"]


def test_short_dropout_holds_long_dropout_clears():
    ev = HandEvents(_regions())
    ev.update(at(100, 100), 0.0)
    assert ev.update(at(100, 100), 1.0) == []
    assert ev.update(NONE, 1.1) == [] and ev.update(NONE, 1.15) == []  # < absent_s since last seen: held
    assert ev.update(at(100, 100), 1.18) == []
    assert ev.update(NONE, 1.3) == []
    assert ev.update(NONE, 1.4) == [("hands", {"over_regions": [], "tip_xy": None})]
    assert ev.update(NONE, 5.0) == []


def test_two_hands_union_of_regions_primary_tip():
    ev = HandEvents(_regions())
    out = ev.update(at(550, 550, (100, 100)), 0.0)
    assert out == [("hands", {"over_regions": ["bin", "plate", "plate:A1"], "tip_xy": [550.0, 550.0]})]


def test_ensure_model_refuses_mismatching_file(tmp_path, no_network):
    p = tmp_path / "hand_landmarker.task"
    p.write_bytes(b"not the model")
    with pytest.raises(HandsUnavailable, match="SHA-256"):
        ensure_model(p)
    assert p.read_bytes() == b"not the model"


def test_ensure_model_rejects_unpinned_download(tmp_path, monkeypatch):
    def fake_stream(method, url, **kw):
        return httpx.Client(transport=httpx.MockTransport(lambda req: httpx.Response(200, content=b"x" * 1000))).stream(method, url)

    monkeypatch.setattr(hands.httpx, "stream", fake_stream)
    p = tmp_path / "models" / "hand_landmarker.task"
    with pytest.raises(HandsUnavailable, match="pinned"):
        ensure_model(p)
    assert list(p.parent.iterdir()) == []


def test_missing_mediapipe_is_a_clear_error(monkeypatch):
    monkeypatch.setitem(sys.modules, "mediapipe", None)
    with pytest.raises(HandsUnavailable, match="uv sync --extra live"):
        hands.HandTracker()


_SMOKE = """
import json, sys
from lightwork.testing.synth import add_hand, kit_scene
from lightwork.vision.hands import HandTracker
from lightwork.vision.source import Frame
bg = kit_scene([], size=(1920, 1080))
with HandTracker(sys.argv[1]) as tr:
    obs = [tr.process(Frame(image=img, t_mono=i / 30, idx=i)) for i, img in enumerate([bg, add_hand(bg, (900, 500)), bg] * 10)]
print(json.dumps({"n": len(obs), "empty_present": any(o.present for o in obs[::3])}))
"""


def test_real_tracker_smoke_on_synthetic_frames():
    # find_spec, not importorskip: importing mediapipe here would open PortAudio in pytest and the child would inherit
    # RLIMIT_RTTIME=0 (see hands._import_mediapipe).
    if importlib.util.find_spec("mediapipe") is None:
        pytest.skip("mediapipe not installed (live extra)")
    model = REPO / ".lightwork" / "models" / hands.MODEL_FILE
    if not model.is_file():
        pytest.skip(f"{model} not downloaded (created by the first HandTracker() on a networked machine)")
    if hasattr(resource, "RLIMIT_RTTIME") and resource.getrlimit(resource.RLIMIT_RTTIME)[1] == 0:
        # rlimits are inherited and the hard limit cannot be raised again, so a child would refuse to start MediaPipe.
        pytest.skip("an earlier test imported mediapipe/sounddevice in-process (PortAudio set RLIMIT_RTTIME=0)")
    # Subprocess: a regression of the PortAudio/RLIMIT_RTTIME SIGKILL must fail this test, not kill pytest.
    r = subprocess.run([sys.executable, "-c", _SMOKE, str(model)], capture_output=True, text=True, timeout=120, cwd=REPO)
    assert r.returncode == 0, f"exit {r.returncode}\n{r.stderr[-2000:]}"
    res = json.loads(r.stdout.strip().splitlines()[-1])
    assert res == {"n": 30, "empty_present": False}
