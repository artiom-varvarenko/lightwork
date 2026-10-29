"""Operator-reviewed staged cards atop LiveSession; no camera or inference implementation here."""
from __future__ import annotations

import json
import random
import threading
from pathlib import Path

from lightwork.contracts.canonical import canonical
from lightwork.contracts.evaluation import TrialCard, TrialLabel
from lightwork.contracts.render import Label
from lightwork.runs.store import write_atomic


def load_cards(path, graph):
    cards = [TrialCard.model_validate(item) for item in json.loads(Path(path).read_text())]
    if not cards or len({card.id for card in cards}) != len(cards):
        raise ValueError('trial cards must be nonempty with unique ids')
    steps = {step.id for step in graph.steps}
    if any(card.step_id not in steps for card in cards):
        raise ValueError('trial card references a step absent from the graph')
    return cards


class Collection:
    """Bind before start; state changes serialized by a lock, captures by LiveSession.

    Labels are written on capture/review and again by before_close. Unreviewed or
    rejected captures remain unreviewed evidence, never silently become training rows.
    """
    def __init__(self, session, cards, *, session_id, seed):
        if not session_id.strip():
            raise ValueError('--session-id must identify the physical collection session')
        self.session = session
        self.session_id = session_id
        self.seed = seed
        self.cards = list(cards)
        random.Random(seed).shuffle(self.cards)
        self.index = 0
        self.phase = 'perform'
        self.labels = []
        self.pending = None
        self.error = None
        self.started = False
        self.lock = threading.RLock()
        session.subscribe(self.on_events)
        session.before_close(self.close)

    def start(self):
        with self.lock:
            if self.started:
                return
            self.started = True
            write_atomic(self.session.recorder.run_dir / 'collection.json', (canonical({
                'session_id': self.session_id, 'seed': self.seed,
                'cards': [card.model_dump(mode='json') for card in self.cards]}) + '\n').encode())
            self.persist()
            self.publish()

    def scene(self, scene):
        with self.lock:
            if self.index >= len(self.cards):
                return scene
            x, y, w, h = self.session.layout.work_area_xywh
            # Just below the work area: no card pixels contaminate crop measurements.
            label = Label(id='collection-card', at=[x, y + h + 36], text=self.cards[self.index].instruction, role='info')
            return scene.model_copy(update={'prims': [*scene.prims, label]})

    def publish(self):
        card = self.cards[self.index] if self.index < len(self.cards) else None
        state = {'session_id': self.session_id, 'seed': self.seed, 'index': self.index,
                 'total': len(self.cards), 'card': card.model_dump(mode='json') if card else None,
                 'phase': self.phase, 'error': self.error,
                 'frame_sha': self.labels[self.pending].frame_sha if self.pending is not None else None}
        self.session.hubs['control_hub'].push_state({'collection': state})
        self.session.set_scene_transform(self.scene)

    def persist(self):
        write_atomic(self.session.recorder.run_dir / 'labels.jsonl',
                     ''.join(canonical(label) + '\n' for label in self.labels).encode())

    def on_events(self, events):
        if not self.started:
            self.start()
        for event in events:
            if event.type == 'ui' and event.payload['action'].startswith('trial_'):
                self.action(event.payload['action'])

    def action(self, action):
        with self.lock:
            self.error = None
            if action == 'trial_capture' and self.phase in ('perform', 'error'):
                self.phase = 'capturing'
                self.publish()
                self.session.capture_now().add_done_callback(self.captured)
                return
            if action == 'trial_accept' and self.phase == 'review':
                self.labels[self.pending] = self.labels[self.pending].model_copy(update={'reviewed': True})
                self.phase = 'accepted'
                self.persist()
            elif action == 'trial_reject' and self.phase == 'review':
                self.pending = None
                self.phase = 'perform'
            elif action == 'trial_next' and self.phase == 'accepted':
                self.index += 1
                self.pending = None
                self.phase = 'complete' if self.index == len(self.cards) else 'perform'
            self.publish()

    def captured(self, future):
        with self.lock:
            try:
                event = future.result()
                if event.type != 'keyframe' or not event.payload['verified']:
                    raise ValueError('capture did not return a verified keyframe')
                card = self.cards[self.index]
                label = TrialLabel(session_id=self.session_id, run_id=self.session.recorder.run_id,
                                   frame_sha=event.payload['frame_sha'], trial_id=card.id, step_id=card.step_id,
                                   question=card.question, expected=card.expected, source='trial-script', reviewed=False)
                # An unchanged retake reuses its unreviewed row; no duplicate observation.
                existing = next((i for i, row in enumerate(self.labels)
                                 if (row.trial_id, row.frame_sha) == (label.trial_id, label.frame_sha)), None)
                if existing is None:
                    self.labels.append(label)
                    self.pending = len(self.labels) - 1
                else:
                    self.labels[existing] = label
                    self.pending = existing
                self.phase = 'review'
                self.persist()
            except Exception as error:
                self.error = str(error)
                self.phase = 'error'
            self.publish()

    def close(self, session):
        with self.lock:
            self.persist()
