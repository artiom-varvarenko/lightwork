"""lightwork play --table: objects are found only in scans, on frames taken with our own light switched off.

A fake projector page draws the newest scene onto the synthetic camera image (white strokes and label boxes) after a
draw delay plus a camera delay, and ACKs it after the draw delay, like the Tank page over adb.
"""

from __future__ import annotations

import cv2
import pytest
import numpy as np

from lightwork.contracts import render
from lightwork.geometry import apply_h
from lightwork.tabletop import SCAN_AFTER_ACK_S, TableSession
from lightwork.vision.source import Frame

from .test_tabletop import FH, FW, GEO, HI, Feed, page_rect, pose, put, table

DT = 1 / 30


class FakePage:
    """Publishes like the render hub (a version per changed scene), draws ``draw_s`` later and ACKs then; the camera
    shows a drawn scene ``cam_s`` after that."""

    def __init__(self, draw_s: float = 0.04, cam_s: float = 0.07):
        self.draw_s, self.cam_s = draw_s, cam_s
        self.v = 0
        self.now = 0.0
        self.history: list[tuple[int, float, list]] = [(0, -1.0, [])]

    def publish(self, prims: list, t: float) -> None:
        if prims != self.history[-1][2]:
            self.v += 1
            self.history.append((self.v, t, prims))

    def version(self) -> int:
        return self.v

    def acked_at(self, version: int) -> float | None:
        for v, t, _ in self.history:
            if v >= version and t + self.draw_s <= self.now:
                return t + self.draw_s
        return None

    def seen(self, t: float) -> list:
        """The scene the camera sees at time ``t``."""
        shown = [p for v, tp, p in self.history if tp + self.draw_s + self.cam_s <= t]
        return shown[-1] if shown else []


def lit(img: np.ndarray, prims: list) -> np.ndarray:
    """Our projected light on the table: white strokes, labels as white text-sized bars (the camera blurs text)."""
    out = img.copy()
    for p in prims:
        if p.kind == "poly":
            cv2.polylines(out, [np.round(np.asarray(p.pts)).astype(np.int32)], p.closed, (250, 250, 250), 5)
        elif p.kind == "label":
            x, y = (round(v) for v in p.at)
            cv2.rectangle(out, (x, y - 8), (x + 11 * len(p.text), y + 8), (250, 250, 250), -1)
    return out


class Rig:
    def __init__(self, **kw):
        self.page = FakePage()
        self.feed = Feed()
        self.ts = TableSession(GEO, (FW, FH), link=self.page, **kw)
        self.ts.zone.learn(self.feed(table()))
        self.t, self.n = 0.0, 0
        self.said: list[str] = []
        self.scenes = 0

    def run(self, seconds: float, scene, poses=lambda t: [], frame_hook=None) -> None:
        end = self.t + seconds
        while self.t < end:
            self.t += DT
            self.n += 1
            self.page.now = self.t
            img = lit(scene(self.t) if callable(scene) else scene, self.page.seen(self.t))
            img = self.feed(img)
            if frame_hook is not None:
                img = frame_hook(self.t, img)
            self.said += self.ts.step(Frame(img, self.t, self.n), poses(self.t), self.t, DT)[1]
            self.page.publish(self.ts.scene(self.t), self.t)


def test_our_own_light_never_becomes_an_object():
    """2 Nov: 3 objects on the table, 10 reported, then 6 -> 20 in five minutes with no hand in view."""
    rig = Rig()
    obj = put(table(), page_rect(300, 200, 120, 70, 20), (40, 60, 200))
    rig.run(12.0, obj)
    assert [tr.id for tr in rig.ts.tracks.confirmed] == [1]
    assert any(p.kind == "poly" and p.id.startswith("obj-") for p in rig.page.history[-1][2])  # outlined all along
    assert rig.ts.scans <= 3  # our outlines appearing are not a change of the table: no rescans for them
    assert rig.said and rig.said[0].startswith("New red object")


def test_a_scan_takes_frames_only_after_the_page_showed_the_table_dark():
    rig = Rig()
    obj = put(table(), page_rect(300, 200, 120, 70, 0), (40, 60, 200))
    rig.run(2.0, obj)
    (tr,) = rig.ts.tracks.confirmed
    outline = np.round(tr.cam_outline(GEO)).astype(np.int32)
    band = np.zeros((FH, FW), np.uint8)
    cv2.polylines(band, [outline], True, 255, 3)
    assert float(rig.ts.scan_image.astype(np.float32)[band > 0].mean()) < 235  # the first, whole scan: no outline yet
    assert rig.ts.scan_aborts == 0


def test_moving_one_object_rescans_only_there_and_leaves_the_others_lit():
    """3 Nov: "if I move one object, the other two that are already scanned should stay scanned" - no whole-table
    blink: only what lies where something changed goes dark for the scan."""
    rig = Rig()
    a = page_rect(220, 150, 110, 60, 0)
    obj = put(table(), a, (40, 60, 200))
    rig.run(2.0, obj)
    (first,) = rig.ts.tracks.confirmed
    seen_local = []
    orig = rig.ts.scene

    def scene(t, also_lit=()):
        prims = orig(t, also_lit)
        if rig.ts.scan is not None and rig.ts.scan.region is not None:
            seen_local.append({p.id for p in prims})
        return prims

    rig.ts.scene = scene
    second = put(obj, page_rect(560, 300, 60, 60, 0), (180, 90, 30))  # put down far from the first
    rig.run(2.0, second)
    assert len(rig.ts.tracks.confirmed) == 2
    assert seen_local and all(f"obj-{first.uid}" in ids or f"obj-{first.uid}-s" in ids for ids in seen_local)
    assert first.id == 1 and np.allclose(first.rect[0], (220, 150), atol=3)  # untouched


def test_without_an_ack_nothing_is_scanned_and_the_scan_is_retried():
    rig = Rig()
    rig.page.draw_s = 99.0  # the page never draws (a dead link)
    obj = put(table(), page_rect(300, 200, 120, 70, 0), (40, 60, 200))
    rig.run(3.0, obj)
    assert rig.ts.tracks.confirmed == [] and rig.ts.scan_aborts >= 1
    rig.page.draw_s = 0.04  # the page is back
    rig.run(3.0, obj)
    assert len(rig.ts.tracks.confirmed) == 1


def test_frames_that_disagree_are_no_scan():
    rig = Rig()
    obj = put(table(), page_rect(300, 200, 120, 70, 0), (40, 60, 200))
    broken = []

    def grey_band(t, img):  # a truncated MJPEG frame decodes grey from the cut on: the first frame of the first scan
        s = rig.ts.scan
        if not broken and s is not None and s.clean_after is not None and t >= s.clean_after and not s.frames:
            broken.append(t)
            img = img.copy()
            img[FH // 2:] = 128
        return img

    rig.run(4.0, obj, frame_hook=grey_band)
    assert broken and rig.ts.scan_aborts == 1
    assert [tr.id for tr in rig.ts.tracks.confirmed] == [1]  # found anyway, in the next, clean scan
    assert rig.ts.tracks.confirmed[0].shown[0] < 140  # the object, not the grey band


def test_an_object_moved_by_hand_keeps_its_number_and_name():
    class Namer:
        label, error = "fake", None

        def __init__(self):
            self.asked = []

        def request(self, key, jpeg):
            self.asked.append(key)
            return True

        def result(self, key):
            return (True, "stapler") if key in self.asked else (False, None)

        def summary(self):
            return "fake"

    namer = Namer()
    rig = Rig(namer=namer)
    a = page_rect(250, 180, 110, 50, 0)
    rig.run(2.0, put(table(), a, (40, 60, 200)))
    (tr,) = rig.ts.tracks.confirmed
    assert tr.name == "stapler" and tr.id == 1
    def hand_at(x, y):
        return put(table(), page_rect(x, y, 90, 110, 0), (120, 150, 210))

    w, k = apply_h(HI, [(250, 230), (250, 160)])
    rig.run(0.6, hand_at(250, 190), poses=lambda t: [pose(w, k)])  # a hand closes over it ...
    w2, k2 = apply_h(HI, [(520, 280), (520, 210)])
    rig.run(0.6, hand_at(520, 240), poses=lambda t: [pose(w2, k2)])  # ... carries it off ...
    b = page_rect(520, 250, 110, 50, 90)  # ... and puts it down, turned
    rig.run(2.0, put(table(), b, (40, 60, 200)))
    (tr,) = rig.ts.tracks.confirmed
    assert tr.id == 1 and tr.name == "stapler" and np.allclose(tr.rect[0], (520, 250), atol=4)
    assert len(namer.asked) == 1  # not named again


def test_nothing_appears_where_nothing_moved():
    rig = Rig()
    rig.run(1.0, table())
    rig.ts.anywhere = False
    patch = page_rect(450, 300, 80, 60, 0)
    for k in range(1, 61):  # a shadow creeping in over 2 s: never a frame-to-frame change
        rig.run(DT, put(table(), patch, np.array([120, 165, 215]) * (1 - 0.006 * k)))
    obj = page_rect(200, 150, 70, 70, 0)  # meanwhile an object is put down elsewhere
    rig.run(2.0, put(put(table(), patch, np.array([120, 165, 215]) * 0.64), obj, (180, 90, 30)))
    assert [tr.colour for tr in rig.ts.tracks.confirmed] == ["blue"]


def test_the_empty_table_is_learned_only_once_cleared():
    """3 Nov: after two reminders it learned anyway, with five objects on the table: never, unless LEARN is held."""
    page = FakePage()
    feed = Feed()
    ts = TableSession(GEO, (FW, FH), link=page)
    ts.learn_at = 0.0
    hammer = put(table(), page_rect(350, 220, 300, 50, 10), (30, 30, 40))
    said, t, n = [], 0.0, 0
    while t < 20.0:
        t += DT
        n += 1
        page.now = t
        scene = hammer if t < 16.0 else table()  # taken away only after several reminders
        said += ts.step(Frame(feed(lit(scene, page.seen(t))), t, n), [], t, DT)[1]
        page.publish(ts.scene(t), t)
        if t < 16.0:
            assert ts.zone.ref is None  # never learned with the hammer on it
    assert said[0].startswith("Clear the table")
    assert said.count("Something is still on the table. Clear it, or hold LEARN to learn it as it is.") >= 2
    assert said[-1] == "Table learned. Put objects on it." and ts.zone.status == "ready"
    lum = ts.zone.ref.mean(axis=2)
    assert float(lum.min()) > 80  # the hammer is not in the learned table


def test_holding_learn_while_it_refuses_learns_the_table_as_it_is():
    page = FakePage()
    feed = Feed()
    ts = TableSession(GEO, (FW, FH), link=page)
    ts.learn_at = 0.0
    stain = put(table(), page_rect(350, 220, 60, 40, 0), (30, 30, 40))  # something that will not go away
    t, n = 0.0, 0
    while t < 6.0:
        t += DT
        n += 1
        page.now = t
        ts.step(Frame(feed(lit(stain, page.seen(t))), t, n), [], t, DT)
        page.publish(ts.scene(t), t)
    assert ts.zone.ref is None and ts._learn_tries >= 1
    ts._force_learn, ts.learn_at = True, t  # as when LEARN is held again
    while t < 8.0:
        t += DT
        n += 1
        page.now = t
        ts.step(Frame(feed(lit(stain, page.seen(t))), t, n), [], t, DT)
        page.publish(ts.scene(t), t)
    assert ts.zone.ref is not None and ts.zone.status == "ready"


def test_sounds_and_effects():
    rig = Rig()
    obj = put(table(), page_rect(300, 200, 120, 70, 0), (40, 60, 200))
    rig.run(2.0, obj)
    assert "found" in rig.ts.sfx
    prims = rig.ts.scene(rig.t)
    outline = next(p for p in prims if p.kind == "poly" and p.id.startswith("obj-"))
    assert outline.glow and outline.draw_ms > 0
    labels = [p for p in prims if p.kind == "label" and p.id.startswith("obj-")]
    assert labels and all(p.type_ms > 0 for p in labels)
    rig.ts.sfx.clear()
    rig.run(2.0, table())  # taken away (no hand: the change itself is the motion)
    assert rig.ts.tracks.confirmed == [] and "gone" in rig.ts.sfx


def test_scan_waits_for_the_after_ack_delay():
    assert 0.2 <= SCAN_AFTER_ACK_S <= 0.35  # compositor, DLP, exposure, C920 delay; 0.12 showed fading labels on Wi-Fi
    assert render.Sfx(name="scan").type == "sfx"


def test_an_object_taken_away_during_a_light_change_is_dropped():
    """3 Nov: a light-change scan used up the motion seen meanwhile; the card wallet taken away then stayed outlined."""
    rig = Rig()
    wallet = put(table(), page_rect(300, 200, 110, 80, 25), (30, 30, 35))
    rig.run(1.5, wallet)
    assert len(rig.ts.tracks.confirmed) == 1
    rig.run(8.0, table() * 1.18)  # taken away while the daylight brightens
    assert rig.ts.tracks.confirmed == []


def test_a_big_object_is_an_object_not_a_light_change():
    """3 Nov: a bag of chips (a third of the table) was taken for a light change and never outlined."""
    rig = Rig()
    rig.run(1.0, table())
    bag = put(table(), page_rect(400, 230, 380, 260, 5), (40, 140, 210))
    rig.run(2.0, bag)
    assert rig.ts.zone.status == "ready"
    (tr,) = rig.ts.tracks.confirmed
    assert tr.shown[0] == pytest.approx(380, rel=0.06)


def test_a_soft_shadow_is_not_part_of_the_object():
    """3 Nov, daylight: a power bank was outlined with its shadow."""
    rig = Rig()
    rig.run(1.0, table())
    base = table()
    shadow = page_rect(300, 238, 150, 40, 0)  # the shadow falls along one long side
    yy, xx = np.mgrid[0:FH, 0:FW]
    m = np.zeros((FH, FW), np.uint8)
    cv2.fillPoly(m, [np.round(apply_h(HI, shadow)).astype(np.int32)], 255)
    m = cv2.GaussianBlur(m, (0, 0), 3)[..., None] / 255.0
    shaded = base * (1 - 0.3 * m)  # 70 % of the light: even, soft
    scene = put(shaded, page_rect(300, 200, 150, 50, 0), (35, 35, 40))
    rig.run(2.0, scene)
    (tr,) = rig.ts.tracks.confirmed
    assert tr.shown[0] == pytest.approx(150, rel=0.06) and tr.shown[1] == pytest.approx(50, rel=0.12)


def test_a_grey_neck_between_two_dark_parts_keeps_one_object():
    """3 Nov: the shadow filter took a hammer's grey metal neck for shadow; the head became a second object."""
    rig = Rig()
    rig.run(1.0, table())
    base = table()
    neck = page_rect(395, 200, 30, 20, 0)
    yy, xx = np.mgrid[0:FH, 0:FW]
    m = np.zeros((FH, FW), np.uint8)
    cv2.fillPoly(m, [np.round(apply_h(HI, neck)).astype(np.int32)], 255)
    scene = base * (1 - 0.3 * (m[..., None] / 255.0))  # the neck darkens the table evenly, like a shadow
    scene = put(scene, page_rect(290, 200, 180, 28, 0), (30, 30, 35))  # handle
    scene = put(scene, page_rect(425, 200, 30, 70, 0), (55, 55, 60))  # head
    rig.run(2.0, scene)
    (tr,) = rig.ts.tracks.confirmed
    assert tr.shown[0] == pytest.approx(240, rel=0.06)


def test_an_object_with_a_shiny_gap_stays_one_object():
    """3 Nov: the hammer's polished face reflected the white table; the top of the head became a separate object."""
    rig = Rig()
    rig.run(1.0, table())
    scene = put(table(), page_rect(300, 200, 180, 28, 0), (30, 30, 35))  # handle and neck
    scene = put(scene, page_rect(404, 200, 14, 60, 0), (55, 55, 60))  # head, 9 page px (10 mm) beyond: a bright gap
    rig.run(2.0, scene)
    (tr,) = rig.ts.tracks.confirmed
    assert tr.shown[0] == pytest.approx(204, rel=0.06)


def test_an_object_by_the_button_column_is_found_when_no_hand_is_around():
    """3 Nov: a plush penguin touching the edge of the detection area (next to the buttons) was taken for an arm."""
    from lightwork.tabletop import COL_X1

    rig = Rig()
    rig.run(1.0, table())
    x = GEO.p2d((0, 0))  # noqa: F841 - design space and page px coincide closely here
    left = COL_X1 * GEO.s + GEO.ox  # page px where the button column ends
    rig.run(2.0, put(table(), page_rect(left + 45, 220, 90, 60, 0), (40, 60, 200)))
    assert len(rig.ts.tracks.confirmed) == 1


def test_a_sam_outline_stays_while_the_blob_only_wobbles():
    """3 Nov: a plush toy's fuzzy edge changed its blob a little every scan; the SAM outline was dropped and SAM asked
    again each time, and the new outline started the next scan."""
    from lightwork.tabletop import ObjectTracks, Track, rect_moved

    a = ((300.0, 200.0), (120.0, 70.0), 10.0)
    assert not rect_moved(a, ((302.0, 201.0), (128.0, 74.0), 13.0))  # 7 % larger, 3 degrees: the same object
    assert rect_moved(a, ((320.0, 200.0), (120.0, 70.0), 10.0))  # 20 px away
    assert rect_moved(a, ((300.0, 200.0), (120.0, 70.0), 40.0))  # turned
    assert ObjectTracks and Track


def test_light_brighter_than_the_table_is_no_object():
    """3 Nov: a phone notification drawn over the projector page lit the table and was scanned as "blurry text"."""
    rig = Rig()
    rig.run(1.0, table())
    banner = put(table(), page_rect(400, 150, 260, 90, 0), (235, 238, 240))  # a light grey panel, brighter than the table
    rig.run(2.0, banner)
    assert rig.ts.tracks.confirmed == []


def test_a_hand_hovering_over_objects_starts_no_scan():
    """3 Nov: "when I was hovering my hands over the table, it kept rescanning every time"."""
    rig = Rig()
    obj = put(table(), page_rect(300, 200, 110, 60, 0), (40, 60, 200))
    rig.run(2.0, obj)
    scans = rig.ts.scans
    for x in (260, 330, 400, 470):  # a hand passes over the object and the table, puts nothing down
        w, k = apply_h(HI, [(x, 260), (x, 190)])
        hand = put(obj, page_rect(x, 220, 90, 110, 0), (120, 150, 210))
        rig.run(0.3, hand, poses=lambda t, w=w, k=k: [pose(w, k)])
    rig.run(2.0, obj)
    assert rig.ts.scans == scans and rig.ts.skipped >= 1
    assert [tr.id for tr in rig.ts.tracks.confirmed] == [1]


def test_an_object_taken_away_by_hand_goes_once_the_hand_has_left():
    """3 Nov: a scan with the hand still on the ring kept it ("held"); the ring stayed outlined after it was gone."""
    rig = Rig()
    ring = page_rect(400, 220, 90, 90, 0)
    rig.run(2.0, put(table(), ring, (160, 140, 30)))
    assert len(rig.ts.tracks.confirmed) == 1
    w, k = apply_h(HI, [(400, 290), (400, 230)])
    hand = put(table(), page_rect(400, 240, 90, 110, 0), (120, 150, 210))  # the hand closes over it and holds still
    rig.run(1.5, hand, poses=lambda t: [pose(w, k)])
    rig.run(2.0, table())  # ... and takes it away
    assert rig.ts.tracks.confirmed == []


def test_a_local_scan_takes_an_object_half_inside_it_whole():
    """3 Nov: objects were lifted next to a power bank; the local scan's area cut through it, the cut piece did not match
    the power bank, and it became a new "black box" while the power bank was dropped."""
    rig = Rig()
    bank = page_rect(260, 150, 180, 60, 0)
    toy = page_rect(400, 250, 70, 70, 0)  # 30 px (3 cm) from the bank's corner
    scene = put(put(table(), bank, (30, 30, 35)), toy, (40, 60, 200))
    rig.run(2.0, scene)
    assert len(rig.ts.tracks.confirmed) == 2
    bank_id = next(tr.id for tr in rig.ts.tracks.confirmed if tr.shown[0] > 150)
    w, k = apply_h(HI, [(400, 320), (400, 260)])
    rig.run(0.6, put(put(table(), bank, (30, 30, 35)), page_rect(400, 260, 90, 110, 0), (120, 150, 210)),
            poses=lambda t: [pose(w, k)])  # a hand takes the toy, next to the power bank's end
    rig.run(2.0, put(table(), bank, (30, 30, 35)))
    (tr,) = rig.ts.tracks.confirmed
    assert tr.id == bank_id and tr.shown[0] == pytest.approx(180, rel=0.06)


def test_two_objects_too_close_to_tell_apart_are_never_lost():
    """Closer than the merge distance they become one outline, but the lower-contrast one is not dropped."""
    rig = Rig()
    scene = put(put(table(), page_rect(260, 150, 180, 60, 0), (30, 30, 35)), page_rect(390, 230, 70, 70, 0), (40, 60, 200))
    rig.run(2.0, scene)
    (tr,) = rig.ts.tracks.confirmed
    outline = tr.outline.astype(np.float32)
    assert cv2.pointPolygonTest(outline, (390.0, 230.0), False) > 0  # the red toy is inside the outline
