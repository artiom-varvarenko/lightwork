"""Collection review and persistence boundaries without a physical camera."""
from concurrent.futures import Future
import json
from types import SimpleNamespace

from lightwork.contracts.evaluation import TrialCard
from lightwork.eval.collection import Collection


class Session:
    def __init__(self, path):
        self.recorder = SimpleNamespace(run_dir=path, run_id='test-run')
        self.state = {}
        self.hubs = {'control_hub':SimpleNamespace(push_state=self.state.update)}
        self.callbacks = []
        self.closers = []
        self.future = None

    def subscribe(self, callback):
        self.callbacks.append(callback)

    def before_close(self, callback):
        self.closers.append(callback)

    def set_scene_transform(self, callback):
        self.transform = callback

    def capture_now(self):
        self.future = Future()
        return self.future


def card(index):
    return TrialCard(id=str(index),instruction='Leave remote on bench',step_id='remote',question='Is remote packed?',
                     expected={'done':'no','contradiction':None,'evidence':'scripted omission'})


def test_capture_review_reject_accept_then_close(tmp_path):
    session = Session(tmp_path)
    collection = Collection(session,[card(1)],session_id='physical',seed=17)
    collection.start()
    collection.action('trial_accept')
    assert not collection.labels
    collection.action('trial_capture')
    assert collection.phase == 'capturing'
    session.future.set_result(SimpleNamespace(type='keyframe',payload={'verified':True,'frame_sha':'a'*64}))
    assert collection.phase == 'review'
    assert not collection.labels[0].reviewed
    collection.action('trial_next')
    assert collection.index == 0
    collection.action('trial_reject')
    assert collection.phase == 'perform'
    collection.action('trial_capture')
    session.future.set_result(SimpleNamespace(type='keyframe',payload={'verified':True,'frame_sha':'a'*64}))
    assert len(collection.labels) == 1
    collection.action('trial_accept')
    assert collection.labels[0].reviewed
    assert collection.labels[0].expected.done == 'no'
    collection.action('trial_next')
    assert collection.phase == 'complete'
    session.closers[0](session)
    saved = json.loads((tmp_path/'labels.jsonl').read_text())
    assert saved['reviewed'] and saved['session_id'] == 'physical'
    assert json.loads((tmp_path/'collection.json').read_text())['seed'] == 17


def test_randomization_is_seeded_and_failed_capture_never_labels(tmp_path):
    a = Collection(Session(tmp_path),[card(i) for i in range(12)],session_id='one',seed=9)
    b = Collection(Session(tmp_path),[card(i) for i in range(12)],session_id='one',seed=9)
    assert [c.id for c in a.cards] == [c.id for c in b.cards]
    assert [c.id for c in a.cards] != [str(i) for i in range(12)]
    a.start()
    a.action('trial_capture')
    a.session.future.set_exception(RuntimeError('no projector ACK'))
    assert a.phase == 'error' and not a.labels
    a.action('trial_capture')
    a.session.future.set_result(SimpleNamespace(type='keyframe',payload={'verified':False,'frame_sha':'a'*64}))
    assert a.phase == 'error' and not a.labels
