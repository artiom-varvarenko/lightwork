"""Private crop-first layout; measured frames and explicit human acceptance, with offline click fallback."""
from __future__ import annotations

import math
import queue
import threading
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path

import cv2
import numpy as np

from lightwork.contracts.canonical import canonical_bytes
from lightwork.contracts.layout import Layout
from lightwork.core.graph import validate
from lightwork.core.masks import to_camera
from lightwork.models.budget import Budget
from lightwork.models.client import ModelClient
from lightwork.models.requests import META_MODELS, build_request
from lightwork.server.hub import encode_preview
from lightwork.vision.measure_frames import capture_keyframe
from .workflow import read_call


class LayoutSession:
    def __init__(self, graph, camera, hubs, *, output: Path, settings, manual=True, mode='dark'):
        self.graph, self.camera, self.hubs = graph, camera, hubs
        self.output, self.settings, self.manual, self.mode = Path(output), settings, manual, mode
        self.objects = {}
        self.area = None
        self.capture = None
        self.done = threading.Event()
        self.actions = queue.Queue()
        self.calls = 0
        self.session_dir = settings.state_dir / 'layout-sessions' / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
        self.result = None
        self.message = 'Select a private work area before preview or inference. Grids: click extreme CELL CENTRES, not plate edges.'
        hubs['control_hub'].on_action(lambda action, args: self.actions.put((action, args)))
        self.publish()

    def publish(self):
        frame = self.camera.latest()
        h, w = frame.image.shape[:2]
        objects = [{'id': obj.id, 'kind': obj.kind, **self.objects.get(obj.id, {})} for obj in self.graph.objects]
        self.hubs['control_hub'].push_state({'camera': {'w': w, 'h': h}, 'preview_xywh': self.area or [0, 0, w, h],
            'run': {'pack': self.graph.id, 'mode': 'layout'}, 'banner': self.message,
            'layout': {'enabled': True, 'objects': objects, 'work_area_xywh': self.area, 'message': self.message}})

    def _area(self, values):
        if len(values) != 4 or any(isinstance(v, bool) or not isinstance(v, int) for v in values):
            raise ValueError('work area requires four integer camera pixels')
        x, y, w, h = values
        height, width = self.camera.latest().image.shape[:2]
        if x < 0 or y < 0 or w <= 1 or h <= 1 or x + w > width or y + h > height:
            raise ValueError('work area must fit entirely inside the camera frame')
        if self.area is not None and values != self.area:
            raise ValueError('Work area is locked for privacy; restart layout to choose a different crop')
        self.area = list(values)
        self.message = 'Private crop selected. Capture measured frame, then place every object and accept.'

    def _points(self, points, *, grid=False, rows=2, cols=2):
        if self.area is None:
            raise ValueError('Select the private work area first')
        x, y, w, h = self.area
        if len(points) < 3 or any(len(p) != 2 or any(not isinstance(v, (float, int)) or not math.isfinite(v) for v in p) for p in points):
            raise ValueError('Invalid geometry')
        if any(not (x <= px <= x+w and y <= py <= y+h) for px, py in points):
            raise ValueError('Geometry must stay inside the selected crop')
        arr = np.asarray(points, np.float32)
        if grid and (rows == 1 or cols == 1):
            # Degenerate one-row/column grids intentionally repeat the extreme centres.
            if np.linalg.norm(arr[1 if cols > 1 else 3] - arr[0]) < 1:
                raise ValueError('Grid extreme centres must be distinct along its non-singleton axis')
        elif abs(cv2.contourArea(arr)) < 1 or not cv2.isContourConvex(arr):
            raise ValueError('Use a non-crossing convex polygon in TL, TR, BR, BL order')
        return [[round(float(v), 1) for v in p] for p in points]

    def _capture(self):
        if self.area is None:
            raise ValueError('Select the private work area first')
        result = capture_keyframe(self.camera, self.hubs['render_hub'], self.mode, self.area)
        if result is None or not result[1]['verified']:
            raise ValueError('Measured capture failed: keep hands out, steady lighting, and connect screen/projector for ACK')
        jpeg, payload = result
        digest = sha256(jpeg).hexdigest()
        self.session_dir.mkdir(parents=True, exist_ok=True)
        (self.session_dir / f'{digest}.jpg').write_bytes(jpeg)
        (self.session_dir / 'capture.json').write_bytes(canonical_bytes({**payload, 'frame_sha': digest}))
        self.capture = (digest, jpeg)
        self.message = 'Measured crop captured. Choose geometry; grid corners mean extreme cell centres.'

    def _call(self, client, role, ctx):
        digest, jpeg = self.capture
        endpoint, request = build_request(role, model=META_MODELS[role], backend='meta', frame_sha=digest, ctx=ctx)
        self.calls += 1
        result = client.execute(intent_id=f'layout-{self.calls}', role=role, endpoint=endpoint,
                                request=request, frames={digest: jpeg}, index=self.calls,
                                timeout_s=5 if role == 'segment' else 20)
        parsed = read_call(self.session_dir, result, role, endpoint)
        if result['status'] != 'ok' or parsed['status'] != 'ok':
            raise ValueError(f'{role} failed; raw call retained: {parsed.get("error")}')
        return parsed['result']

    def _suggest(self, oid, noun_only=False):
        if self.manual:
            raise ValueError('Manual layout never calls models; restart without --manual to request SAM/Spark')
        if self.capture is None:
            raise ValueError('Capture a verified private crop before calling models')
        obj = next(o for o in self.graph.objects if o.id == oid)
        x, y, w, h = self.area
        with ModelClient(self.settings, run_dir=self.session_dir, provenance='meta',
                         budget=Budget.from_settings(self.settings), cache=None) as client:
            if noun_only:
                ctx, mode = {'concept': obj.noun}, 'noun'
            else:
                ground = self._call(client, 'ground', {'nouns': [obj.noun], 'shape': 'box'})
                items = [item for item in ground['items'] if item['name'] == obj.noun and 'bbox' in item]
                if len(items) != 1:
                    raise ValueError('Grounding was ambiguous; use SAM noun or manual geometry')
                a, b, c, d = items[0]['bbox']
                box = [int(a*w/1000), int(b*h/1000), min(w, math.ceil(c*w/1000)), min(h, math.ceil(d*h/1000))]
                ctx, mode = {'object_prompts': [{'id': 1, 'box': box}], 'w': w, 'h': h}, 'grounded'
            masks = self._call(client, 'segment', ctx)['masks']
        if len(masks) != 1:
            raise ValueError('SAM found zero or multiple objects; select manual geometry instead')
        mask = to_camera(masks[0], self.area)
        contours, _ = cv2.findContours(mask.bits.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contour = max(contours, key=cv2.contourArea)
        polygon = cv2.approxPolyDP(cv2.convexHull(contour), 2, True).reshape(-1, 2)
        points = [[float(px+mask.box[0]), float(py+mask.box[1])] for px, py in polygon]
        if obj.kind == 'grid':
            # Plate margins are not known from a SAM silhouette. Never equate outer edges with cell centres.
            self.objects[oid] = {'kind': obj.kind, 'sam': {'mode': mode, 'noun': obj.noun, 'validated': True}}
            self.message = f'{oid}: SAM boundary found; explicitly click four extreme CELL CENTRES to define the grid.'
        else:
            self.objects[oid] = {'kind': obj.kind, 'polygon_cam_px': self._points(points),
                                 'sam': {'mode': mode, 'noun': obj.noun, 'validated': True}}
            self.message = f'{oid}: model suggestion only; inspect/correct before Accept layout.'

    def action(self, action, args):
        if action == 'layout_work_area':
            self._area(args['xywh'])
        elif action == 'layout_capture':
            self._capture()
        elif action in ('layout_suggest', 'layout_noun'):
            self._suggest(args['object'], action == 'layout_noun')
        elif action in ('layout_rect', 'layout_corners'):
            obj = next((o for o in self.graph.objects if o.id == args['object']), None)
            if obj is None:
                raise ValueError('Unknown layout object')
            if action == 'layout_rect':
                if obj.kind == 'grid':
                    raise ValueError('Grids require four extreme CELL CENTRE clicks, not a drag rectangle')
                x, y, w, h = args['xywh']
                points = [[x,y], [x+w,y], [x+w,y+h], [x,y+h]]
            else:
                points = args['corners_cam_px']
                if len(points) != 4:
                    raise ValueError('Four corners are required')
            points = self._points(points, grid=obj.kind == 'grid', rows=obj.grid.rows if obj.grid else 2, cols=obj.grid.cols if obj.grid else 2)
            sam = self.objects.get(obj.id, {}).get('sam', {'mode': 'noun', 'noun': obj.noun, 'validated': False})
            self.objects[obj.id] = {'kind': obj.kind, 'corners_cam_px' if obj.kind == 'grid' else 'polygon_cam_px': points, 'sam': sam}
            self.message = f'{obj.id}: geometry set; inspect every object before accepting.'
        elif action == 'layout_accept':
            if self.capture is None:
                raise ValueError('Capture a measured crop before accepting layout')
            missing = [o.id for o in self.graph.objects if o.id not in self.objects]
            if missing:
                raise ValueError('Missing objects: ' + ', '.join(missing))
            layout = Layout(pack=self.graph.id, created=datetime.now(timezone.utc).isoformat(), work_area_xywh=self.area, objects=self.objects)
            errors = validate(self.graph, layout)
            if errors:
                raise ValueError('; '.join(errors))
            if self.output.exists():
                raise ValueError('Layout output already exists; choose a new session output')
            self.output.parent.mkdir(parents=True, exist_ok=True)
            self.output.write_bytes(canonical_bytes(layout.model_dump(mode='json')))
            (self.session_dir / 'accepted.json').write_bytes(canonical_bytes({'layout': str(self.output), 'manual': self.manual, 'calls': self.calls}))
            self.result = layout
            self.message = 'Layout accepted and saved.'
            self.done.set()
        self.publish()

    def run(self):
        last_idx = -1
        while not self.done.is_set():
            try:
                action, args = self.actions.get(timeout=.05)
                try:
                    self.action(action, args)
                except (ValueError, RuntimeError, OSError, KeyError, StopIteration) as exc:
                    self.message = str(exc)
                    self.publish()
            except queue.Empty:
                pass
            frame = self.camera.latest()
            if self.area and frame.idx != last_idx and self.hubs['preview_hub'].has_viewers:
                x, y, w, h = self.area
                self.hubs['preview_hub'].push(frame.idx, frame.t_mono, encode_preview(frame.image[y:y+h, x:x+w]))
                last_idx = frame.idx
        return self.result
