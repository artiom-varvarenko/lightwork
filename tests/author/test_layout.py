import json
from types import SimpleNamespace

import numpy as np
import pytest

from lightwork.author.layout import LayoutSession
from lightwork.author.workflow import validated_graph
from lightwork.config import Settings
from lightwork.contracts.layout import Layout
from lightwork.server.hub import make_hubs
from lightwork.vision.measure_frames import encode_jpeg


def graph():
    return validated_graph({'id': 'layout-test', 'title': 'Synthetic layout', 'source': {'kind': 'text', 'ref': 'tests', 'licence': 'original', 'attribution': 'Test'}, 'objects': [{'id': 'plate', 'noun': 'well plate', 'kind': 'grid', 'grid': {'rows': 2, 'cols': 3, 'naming': 'A1'}}, {'id': 'zone', 'noun': 'bench zone', 'kind': 'zone'}], 'steps': [{'id': 'inspect', 'say': 'Inspect plate.', 'detail': '', 'targets': ['plate:A1', 'zone'], 'checks': [{'id': 'judge', 'kind': 'judge', 'question': 'Is plate present?', 'required': True}], 'recovery': []}], 'vocabulary': []})


@pytest.fixture
def session(tmp_path):
    frame = SimpleNamespace(image=np.full((300, 500, 3), 150, np.uint8), idx=1, t_mono=10)
    camera = SimpleNamespace(latest=lambda: frame)
    return LayoutSession(graph(), camera, make_hubs(), output=tmp_path/'layout.json', settings=Settings(root=tmp_path), manual=True, mode='normal')


def capture(session, monkeypatch):
    def measured(cam, hub, mode, crop):
        assert crop == [100, 50, 300, 200]
        return encode_jpeg(np.full((200, 300, 3), 150, np.uint8)), {'crop_xywh': crop, 'mode': mode, 'verified': True, 't_capture': 10}
    monkeypatch.setattr('lightwork.author.layout.capture_keyframe', measured)
    session.action('layout_capture', {})


def test_crop_required_before_capture_models_geometry(session):
    for action, args in [('layout_capture', {}), ('layout_corners', {'object': 'plate', 'corners_cam_px': [[0,0],[1,0],[1,1],[0,1]]}), ('layout_suggest', {'object': 'plate'})]:
        with pytest.raises(ValueError):
            session.action(action, args)
    assert not session.session_dir.exists()
    assert session.calls == 0


def test_manual_centre_coordinates_round_trip(session, monkeypatch):
    session.action('layout_work_area', {'xywh': [100, 50, 300, 200]})
    capture(session, monkeypatch)
    points = [[140,90], [360,90], [360,210], [140,210]]
    session.action('layout_corners', {'object': 'plate', 'corners_cam_px': points})
    session.action('layout_rect', {'object': 'zone', 'xywh': [110, 60, 60, 30]})
    session.action('layout_accept', {})
    saved = Layout.model_validate_json(session.output.read_bytes())
    assert saved.objects['plate'].corners_cam_px == points
    assert saved.work_area_xywh == [100, 50, 300, 200]
    assert not saved.objects['plate'].sam.validated
    assert saved.objects['zone'].polygon_cam_px == [[110,60],[170,60],[170,90],[110,90]]
    assert session.calls == 0
    assert session.done.is_set()


def test_no_edges_for_grid_and_no_outside_crop(session):
    session.action('layout_work_area', {'xywh': [100, 50, 300, 200]})
    with pytest.raises(ValueError, match='CENTRE'):
        session.action('layout_rect', {'object': 'plate', 'xywh': [100,50,200,100]})
    with pytest.raises(ValueError, match='inside'):
        session.action('layout_rect', {'object': 'zone', 'xywh': [90,50,200,100]})
    with pytest.raises(ValueError, match='locked'):
        session.action('layout_work_area', {'xywh': [110, 50, 300, 200]})


def test_unverified_capture_not_saved(session, monkeypatch):
    session.action('layout_work_area', {'xywh': [100,50,300,200]})
    monkeypatch.setattr('lightwork.author.layout.capture_keyframe', lambda *a: (b'not-stored', {'verified': False}))
    with pytest.raises(ValueError, match='capture failed'):
        session.action('layout_capture', {})
    assert not session.session_dir.exists()


def test_manual_never_calls_models(session, monkeypatch):
    session.action('layout_work_area', {'xywh': [100,50,300,200]})
    capture(session, monkeypatch)
    for action in ('layout_suggest', 'layout_noun'):
        with pytest.raises(ValueError, match='never calls'):
            session.action(action, {'object': 'plate'})
    assert session.calls == 0


def test_layout_requires_all_objects_and_measured_frame(session, monkeypatch):
    session.action('layout_work_area', {'xywh': [100,50,300,200]})
    with pytest.raises(ValueError, match='measured'):
        session.action('layout_accept', {})
    capture(session, monkeypatch)
    with pytest.raises(ValueError, match='Missing'):
        session.action('layout_accept', {})
    assert not session.output.exists()
