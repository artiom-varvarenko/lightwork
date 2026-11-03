"""lightwork play --table: objects found, measured and followed on a synthetic table; hands kept out; two hand slots."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from lightwork.geometry import apply_h
from lightwork.playground import Geo
from lightwork.tabletop import (
    CARD_SEARCH_S,
    PAD_PAGE,
    CONFIRM_S,
    HUD_RECTS,
    ID1_MM,
    LOST_S,
    HandScale,
    HandSlots,
    ObjectTracks,
    Scale,
    TableSession,
    TableZone,
    find_card,
    hand_mm_per_px,
    load_card_scale,
    offset_outline,
    rel_colour_name,
    save_card_scale,
)
from lightwork.vision.hands import HandPose

FW, FH = 960, 540
PAGE_W, PAGE_H = 749.0, 421.0
HI = np.array([[1.1, 0.02, 50.0], [0.01, 1.1, 40.0], [0.0, 0.00002, 1.0]])  # page -> camera px, slightly keystoned
GEO = Geo(np.linalg.inv(HI), PAGE_W, PAGE_H)
DT = 1 / 15  # zone updates per second (every second frame at 30 fps)


def table(seed: int = 0) -> np.ndarray:
    """A white table in warm light: orange cast, a soft light falloff and sensor-like texture."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:FH, 0:FW].astype(np.float32)
    fall = 1.0 - 0.5 * (((xx - FW * 0.4) / FW) ** 2 + ((yy - FH * 0.3) / FH) ** 2)
    img = np.stack([120 * fall, 165 * fall, 215 * fall], axis=2)
    img += cv2.GaussianBlur(rng.normal(0, 6, img.shape).astype(np.float32), (0, 0), 2)
    return img


def page_rect(cx, cy, w, h, angle_deg) -> np.ndarray:
    """Corners of a w x h rectangle (page px) centred at (cx, cy), turned by angle."""
    a = np.deg2rad(angle_deg)
    u, v = np.array([np.cos(a), np.sin(a)]), np.array([-np.sin(a), np.cos(a)])
    c = np.array([cx, cy])
    return np.array([c - u * w / 2 - v * h / 2, c + u * w / 2 - v * h / 2, c + u * w / 2 + v * h / 2, c - u * w / 2 + v * h / 2])


def put(img: np.ndarray, corners_page: np.ndarray, bgr) -> np.ndarray:
    out = img.copy()
    cam = apply_h(HI, corners_page)
    mask = np.zeros((FH, FW), np.uint8)
    cv2.fillPoly(mask, [np.round(cam * 8).astype(np.int32)], 255, cv2.LINE_AA, 3)
    a = (mask.astype(np.float32) / 255.0)[..., None]
    return out * (1 - a) + np.asarray(bgr, np.float32) * a


class Feed:
    def __init__(self, noise: float = 3.0, seed: int = 1):
        self.rng = np.random.default_rng(seed)
        self.noise = noise

    def __call__(self, scene: np.ndarray) -> np.ndarray:
        return np.clip(scene + self.rng.normal(0, self.noise, scene.shape), 0, 255).astype(np.uint8)


def zone() -> TableZone:
    return TableZone(GEO, (FW, FH), exclude_design=HUD_RECTS)


def run(z: TableZone, tracks: ObjectTracks, feed: Feed, scene, t: float, seconds: float, hands=None):
    events = []
    for _ in range(int(seconds / DT)):
        t += DT
        blobs = z.update(feed(scene), t, hands=hands, hand_visible=hands is not None)
        if blobs is not None:
            events += tracks.update(blobs, t, lambda poly: z.touches(hands, poly))
    return t, events


def learned(scene=None) -> tuple[TableZone, ObjectTracks, Feed, np.ndarray]:
    base = table() if scene is None else scene
    feed = Feed()
    z = zone()
    z.learn(feed(base))
    return z, ObjectTracks(), feed, base


def test_object_is_measured_in_page_px_whatever_its_turn():
    z, tracks, feed, base = learned()
    scene = put(base, page_rect(300, 160, 120, 70, 27), (40, 60, 200))  # red, BGR
    _, events = run(z, tracks, feed, scene, 0.0, 1.5)
    assert [k for k, _ in events] == ["new"]
    (tr,) = tracks.confirmed
    long, short = tr.shown
    assert long == pytest.approx(120, rel=0.04) and short == pytest.approx(70, rel=0.05)
    assert tr.id == 1 and tr.colour == "red"
    assert np.allclose(tr.rect[0], (300, 160), atol=2.0)


def test_objects_appear_only_after_confirming_and_go_after_being_lost():
    z, tracks, feed, base = learned()
    scene = put(base, page_rect(500, 200, 60, 60, 0), (180, 90, 30))  # blue
    t, _ = run(z, tracks, feed, scene, 0.0, CONFIRM_S / 2)
    assert tracks.confirmed == []
    t, events = run(z, tracks, feed, scene, t, 1.0)
    assert [k for k, _ in events] == ["new"]
    t, events = run(z, tracks, feed, base, t, LOST_S + 0.5)
    assert [k for k, _ in events] == ["gone"] and tracks.tracks == []


def test_numbers_stay_with_their_objects_and_freed_numbers_are_reused():
    z, tracks, feed, base = learned()
    a = put(base, page_rect(200, 150, 80, 50, 0), (40, 60, 200))
    t, _ = run(z, tracks, feed, a, 0.0, 1.0)
    ab = put(a, page_rect(520, 250, 70, 70, 10), (60, 170, 60))
    t, _ = run(z, tracks, feed, ab, t, 1.0)
    ids = {tr.colour: tr.id for tr in tracks.confirmed}
    assert ids == {"red": 1, "green": 2}
    b = put(base, page_rect(520, 250, 70, 70, 10), (60, 170, 60))  # red taken away (no hand: gone, not lost)
    t, _ = run(z, tracks, feed, b, t, LOST_S + 0.5)
    assert {tr.colour: tr.id for tr in tracks.confirmed} == {"green": 2}
    bc = put(b, page_rect(300, 260, 60, 40, 0), (180, 90, 30))
    run(z, tracks, feed, bc, t, 1.0)
    assert {tr.colour: tr.id for tr in tracks.confirmed} == {"green": 2, "blue": 1}


def test_an_object_slid_without_a_hand_is_followed():
    z, tracks, feed, base = learned()
    t = 0.0
    for x in np.arange(200, 400, 4.0):
        t, _ = run(z, tracks, feed, put(base, page_rect(x, 180, 90, 60, 0), (40, 60, 200)), t, DT * 1.01)
    t, _ = run(z, tracks, feed, put(base, page_rect(400, 180, 90, 60, 0), (40, 60, 200)), t, 0.5)
    (tr,) = tracks.confirmed
    assert tr.id == 1 and np.allclose(tr.rect[0], (400, 180), atol=3.0)


def hand_box(cx_page, cy_page, w=90, h=110) -> np.ndarray:
    """A skin-coloured hand stand-in (page px) and a small-px mask that covers it, as `hand_mask` would."""
    return page_rect(cx_page, cy_page, w, h, 0)


def hand_mask_for(z: TableZone, corners_page) -> np.ndarray:
    m = z.poly_mask(apply_h(HI, corners_page))
    return cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))


def test_a_hand_never_becomes_an_object_and_freezes_what_it_touches():
    z, tracks, feed, base = learned()
    obj = page_rect(300, 160, 100, 60, 0)
    scene = put(base, obj, (40, 60, 200))
    t, _ = run(z, tracks, feed, scene, 0.0, 1.0)
    (tr,) = tracks.confirmed
    before = tr.rect
    hand = hand_box(370, 160)  # overlaps the object's right end
    held = put(scene, hand, (120, 150, 210))
    t, events = run(z, tracks, feed, held, t, 1.0, hands=hand_mask_for(z, hand))
    assert events == [] and len(tracks.confirmed) == 1
    assert tracks.confirmed[0].held and tracks.confirmed[0].rect == before
    alone = hand_box(550, 150)  # a hand on the empty table
    t, events = run(z, tracks, feed, put(scene, alone, (120, 150, 210)), t, 1.0, hands=hand_mask_for(z, alone))
    assert events == [] and len(tracks.confirmed) == 1 and not tracks.confirmed[0].held


def test_an_object_moved_by_hand_keeps_its_number():
    z, tracks, feed, base = learned()
    a = page_rect(220, 140, 100, 60, 0)
    t, _ = run(z, tracks, feed, put(base, a, (60, 170, 60)), 0.0, 1.0)
    other = page_rect(600, 160, 50, 50, 0)
    t, _ = run(z, tracks, feed, put(put(base, a, (60, 170, 60)), other, (180, 90, 30)), t, 1.0)
    assert {tr.colour: tr.id for tr in tracks.confirmed} == {"green": 1, "blue": 2}
    hand = hand_box(220, 140, 130, 100)  # the hand closes over it ...
    covered = put(put(base, other, (180, 90, 30)), hand, (120, 150, 210))
    t, _ = run(z, tracks, feed, covered, t, 0.6, hands=hand_mask_for(z, hand))
    carried = hand_box(400, 300, 130, 100)  # ... carries it off (the object goes with the hand) ...
    t, _ = run(z, tracks, feed, put(put(base, other, (180, 90, 30)), carried, (120, 150, 210)), t, 1.5,
               hands=hand_mask_for(z, carried))
    b = page_rect(420, 260, 100, 60, 90)  # ... and puts it down turned
    t, events = run(z, tracks, feed, put(put(base, other, (180, 90, 30)), b, (60, 170, 60)), t, 1.0)
    assert ("moved", tracks.confirmed[0]) in events or any(k == "moved" for k, _ in events)
    assert {tr.colour: tr.id for tr in tracks.confirmed} == {"green": 1, "blue": 2}


def test_an_object_slid_far_by_a_hand_keeps_its_number_at_once():
    z, tracks, feed, base = learned()
    a = page_rect(220, 160, 100, 60, 0)
    t, _ = run(z, tracks, feed, put(base, a, (60, 170, 60)), 0.0, 1.0)
    hand = hand_box(230, 160, 130, 100)
    t, _ = run(z, tracks, feed, put(put(base, a, (60, 170, 60)), hand, (120, 150, 210)), t, 0.3,
               hands=hand_mask_for(z, hand))
    b = page_rect(520, 200, 100, 60, 0)  # slid 300 px and let go within LOST_S
    t, events = run(z, tracks, feed, put(base, b, (60, 170, 60)), t, 0.6)
    assert [k for k, _ in events] == ["moved"]
    (tr,) = tracks.confirmed
    assert tr.id == 1 and np.allclose(tr.rect[0], (520, 200), atol=3.0)
    t, events = run(z, tracks, feed, put(base, b, (60, 170, 60)), t, LOST_S + 0.5)
    assert events == [] and [tr.id for tr in tracks.confirmed] == [1]


def test_an_object_that_was_there_when_the_table_was_learned_heals_when_taken_away():
    base = table()
    present = put(base, page_rect(300, 200, 110, 80, 15), (40, 40, 60))  # dark object, already on the table
    z, tracks, feed, _ = learned(present)
    t, events = run(z, tracks, feed, base, 0.0, 2.0)
    assert tracks.confirmed == [] and events == []
    assert z.healed >= 1


def test_our_own_projected_outlines_are_not_objects():
    z, tracks, feed, base = learned()
    obj = page_rect(300, 160, 100, 60, 0)
    scene = put(base, obj, (40, 60, 200))
    t, _ = run(z, tracks, feed, scene, 0.0, 1.0)
    (tr,) = tracks.confirmed
    session_like_outline = tr.cam_outline(GEO)  # what the page draws around it
    lit = scene.copy()
    cv2.polylines(lit, [np.round(session_like_outline).astype(np.int32)], True, (255, 255, 255), 5)
    from lightwork.contracts.render import Poly
    from lightwork.tabletop import OverlayHistory

    hist = OverlayHistory(GEO, z)
    hist.set([Poly(id="o", pts=session_like_outline.tolist(), closed=True, role="target")], t)
    events = []
    for _ in range(20):
        t += DT
        blobs = z.update(feed(lit), t, overlay=hist.at(t))
        events += tracks.update(blobs, t, lambda poly: False)
    assert events == [] and len(tracks.tracks) == 1
    assert tracks.confirmed[0].shown[0] == pytest.approx(100, rel=0.05)


def test_card_sets_the_scale():
    z, tracks, feed, base = learned()
    mm_per_px = 0.8
    card = page_rect(350, 180, ID1_MM[0] / mm_per_px, ID1_MM[1] / mm_per_px, 33)
    other = page_rect(620, 200, 80, 80, 0)
    scene = put(put(base, card, (150, 60, 30)), other, (40, 60, 200))
    run(z, tracks, feed, scene, 0.0, 1.5)
    found = find_card(tracks.tracks)
    assert found is not None
    scale, tr = found
    assert tr.colour == "blue"
    assert scale.mm_per_px == pytest.approx(mm_per_px, rel=0.03)
    assert scale.text(tr.shown) in ("86x54mm", "85x54mm", "86x53mm", "85x53mm", "87x54mm", "86x55mm")


def test_saved_card_scale_needs_the_same_page(tmp_path):
    save_card_scale(tmp_path, Scale(0.812, "card"), PAGE_W, PAGE_H)
    assert load_card_scale(tmp_path, PAGE_W, PAGE_H) == Scale(0.812, "card")
    assert load_card_scale(tmp_path, 915.0, 412.0) is None
    assert load_card_scale(tmp_path / "missing", PAGE_W, PAGE_H) is None


def test_colour_names_divide_out_the_table_light():
    warm_table = (120.0, 165.0, 215.0)
    assert rel_colour_name((120, 165, 215), warm_table) == "white"
    assert rel_colour_name((40, 60, 200), warm_table) == "red"
    assert rel_colour_name((150, 60, 30), warm_table) == "blue"
    assert rel_colour_name((60, 160, 60), warm_table) == "green"
    assert rel_colour_name((20, 25, 30), warm_table) == "black"


def pose(wrist, knuckle, side="Left", score=0.9, palm_mm=90.0) -> HandPose:
    """A flat hand from wrist and middle knuckle (camera px), fingers straight on, with a metric palm."""
    w, k = np.asarray(wrist, float), np.asarray(knuckle, float)
    d = k - w
    n = np.array([-d[1], d[0]])
    img = np.zeros((21, 2))
    img[0], img[9] = w, k
    for mcp, off in ((5, 0.35), (9, 0.0), (13, -0.3), (17, -0.55)):
        base = k + n * off - d * abs(off) * 0.2
        img[mcp] = base
        for j in range(1, 4):
            img[mcp + j] = base + d * 0.35 * j
    for j in range(1, 5):
        img[j] = w + n * 0.5 * j / 2 + d * 0.25 * j
    s = palm_mm / 1000.0 / np.linalg.norm(d)
    world = np.column_stack([(img - img.mean(axis=0)) * s, np.zeros(21)])
    return HandPose(img, world, side, score)


def test_hand_slots_keep_each_hand_whatever_order_mediapipe_lists_them():
    slots = HandSlots(GEO)
    left = pose((200, 400), (230, 320), side="Right")  # MediaPipe's "Right" is the left hand of a looked-down-at table
    right = pose((700, 400), (670, 320), side="Left")
    slots.update([left, right], 0.0)
    k_left = next(s.k for s in slots.active if s.pose is left)
    for i in range(1, 6):
        shift = (i * 4.0, 0.0)
        a, b = left.shifted(*shift), right.shifted(*shift)
        slots.update([b, a] if i % 2 else [a, b], i / 30)
    by_k = {s.k: s for s in slots.active}
    assert np.allclose(by_k[k_left].pose.img[0], (220, 400))
    assert by_k[k_left].side == "L" and by_k[1 - k_left].side == "R"
    assert len(slots.tips()) == 2


def test_hand_slots_drop_a_hand_after_a_short_dropout():
    slots = HandSlots(GEO)
    slots.update([pose((200, 400), (230, 320))], 0.0)
    slots.update([], 0.1)
    assert len(slots.active) == 1  # held through a short dropout
    slots.update([], 0.5)
    assert slots.active == []


def test_hand_scale_from_a_flat_hand():
    page_mm = 0.6  # mm per page px: a 90 mm palm spans 150 page px
    s = 1.0 / page_mm
    wrist_page, knuckle_page = np.array([300.0, 300.0]), np.array([300.0, 300.0 - 90.0 * s])
    w, k = apply_h(HI, [wrist_page, knuckle_page])
    p = pose(w, k, palm_mm=90.0)
    r = hand_mm_per_px(p, GEO.H)
    assert r == pytest.approx(page_mm, rel=0.1)
    hs = HandScale()
    for _ in range(20):
        hs.add([p], GEO.H)
    assert hs.value().approx and hs.value().mm_per_px == pytest.approx(page_mm, rel=0.1)


def test_session_learns_the_table_after_saying_so_then_tracks():
    from lightwork.vision.source import Frame

    feed = Feed()
    base = table()
    ts = TableSession(GEO, (FW, FH))
    ts.learn_at = 0.0
    said = []
    t, n = 0.0, 0
    scene = base
    while t < 8.0:
        t += 1 / 30
        n += 1
        if t > 4.5:
            scene = put(base, page_rect(300, 160, 120, 70, 0), (40, 60, 200))
        _, say = ts.step(Frame(feed(scene), t, n), [], t, 1 / 30)
        said += say
        ts.scene(t)
    assert said[0].startswith("Clear the table") and "Table learned" in said[1]
    assert said[2].startswith("New red object")
    assert len(ts.tracks.confirmed) == 1


def test_outline_follows_a_concave_object_not_its_box():
    """2 Nov: "if I put a hammer down, its hitbox is just a box" - the outline must follow the shape."""
    z, tracks, feed, base = learned()
    bar = page_rect(290, 260, 180, 36, 0)  # an L: a long bar ...
    post = page_rect(218, 190, 36, 140, 0)  # ... and a post standing on its left end
    t, _ = run(z, tracks, feed, put(put(base, bar, (40, 60, 200)), post, (40, 60, 200)), 0.0, 1.0)
    (tr,) = tracks.confirmed
    outline = tr.outline.astype(np.float32)
    notch = (320.0, 180.0)  # inside the L's bounding box, 60+ px from both arms
    assert cv2.pointPolygonTest(outline, notch, False) < 0
    for corner in np.concatenate([bar, post]):  # the whole object is inside, with room to spare
        assert cv2.pointPolygonTest(outline, (float(corner[0]), float(corner[1])), True) > PAD_PAGE - 2.5  # rounded, simplified corners
    assert tr.shown[0] == pytest.approx(180, rel=0.05)  # the size is still the minimum-area rectangle


def test_offset_outline_keeps_the_pad_all_round():
    square = page_rect(100, 100, 40, 40, 0)
    out = offset_outline(square, 14.0)
    d = [cv2.pointPolygonTest(out.astype(np.float32), (float(x), float(y)), True) for x, y in square]
    assert all(13.0 <= v <= 15.5 for v in d)  # corners are rounded off at the pad
    assert len(out) < 40  # simplified, not one point per pixel


def test_scale_press_waits_for_a_card_that_is_still_being_put_down():
    from lightwork.vision.source import Frame

    feed = Feed()
    base = table()
    ts = TableSession(GEO, (FW, FH))
    ts.zone.learn(feed(base))
    t, n = 0.0, 0
    ts.card_until = t + CARD_SEARCH_S  # as when SCALE is held
    said = []
    card = page_rect(350, 200, ID1_MM[0] / 0.8, ID1_MM[1] / 0.8, 20)
    while t < 3.0:
        t += 1 / 30
        n += 1
        scene = put(base, card, (150, 60, 30)) if t > 1.0 else base  # the card lands a second after the press
        said += ts.step(Frame(feed(scene), t, n), [], t, 1 / 30)[1]
    assert "Scale set from the card." in said
    assert ts.scale.source == "card" and ts.scale.mm_per_px == pytest.approx(0.8, rel=0.03)


def test_a_wallet_with_a_card_in_it_stays_one_object_under_its_own_outline():
    """2 Nov: a black wallet with a red card showing became two objects; each outline then cut the other."""
    from lightwork.contracts.render import Poly
    from lightwork.tabletop import OverlayHistory

    z, tracks, feed, base = learned()
    wallet = page_rect(400, 220, 70, 110, 0)
    card = page_rect(385, 220, 30, 96, 0)  # red card along its left half: much less contrast than the leather
    scene = put(put(base, wallet, (20, 20, 25)), card, (60, 80, 150))
    t, _ = run(z, tracks, feed, scene, 0.0, 1.0)
    (tr,) = tracks.confirmed
    assert tr.shown[0] == pytest.approx(110, rel=0.05) and tr.shown[1] == pytest.approx(70, rel=0.06)
    # a projected line across the wallet (another object's outline) must not cut it
    line = [[400.0, 150.0], [400.0, 290.0]]
    lit = scene.copy()
    cam = apply_h(HI, line)
    cv2.line(lit, tuple(np.round(cam[0]).astype(int)), tuple(np.round(cam[1]).astype(int)), (60, 255, 60), 6)
    hist = OverlayHistory(GEO, z)
    hist.set([Poly(id="other", pts=cam.tolist(), closed=False, role="target")], t)
    for _ in range(20):
        t += DT
        tracks.update(z.update(feed(lit), t, overlay=hist.at(t)), t, lambda poly: False)
    assert len(tracks.tracks) == 1 and tracks.confirmed[0].shown[1] == pytest.approx(70, rel=0.06)


def test_colour_smear_of_our_own_outline_is_no_object_and_lets_a_removed_one_go():
    """2 Nov: beside a projected red outline the camera read R +57, G -36, B -21 (compressed-video colour smear). With
    "darker in any channel" that made phantom objects, and kept a removed object's track alive at its outline."""
    from lightwork.contracts.render import Poly
    from lightwork.tabletop import OverlayHistory

    z, tracks, feed, base = learned()
    obj = page_rect(300, 200, 100, 60, 0)
    t, _ = run(z, tracks, feed, put(base, obj, (40, 60, 200)), 0.0, 1.0)
    (tr,) = tracks.confirmed
    outline = tr.cam_outline(GEO)
    hist = OverlayHistory(GEO, z)
    hist.set([Poly(id="o", pts=outline.tolist(), closed=True, role="error")], t)
    smeared = base.copy()  # the object is gone; its red outline stays lit, smearing colour into its neighbours
    band = np.zeros((FH, FW), np.uint8)
    cv2.polylines(band, [np.round(outline).astype(np.int32)], True, 255, 14)
    smeared[band > 0] += np.array([-21.0, -36.0, 57.0])
    events = []
    for _ in range(int((LOST_S + 1.0) / DT)):
        t += DT
        events += tracks.update(z.update(feed(smeared), t, overlay=hist.at(t)), t, lambda poly: False)
    assert [k for k, _ in events] == ["gone"] and tracks.tracks == []


def test_a_track_never_jumps_to_an_object_of_another_size():
    """2 Nov: the hammer was taken away and a wallet put down nearby; the hammer track (wide gate) took the wallet."""
    z, tracks, feed, base = learned()
    hammer = page_rect(400, 220, 300, 60, 0)
    t, _ = run(z, tracks, feed, put(base, hammer, (40, 60, 200)), 0.0, 1.0)
    assert [tr.id for tr in tracks.confirmed] == [1]
    wallet = page_rect(440, 230, 110, 90, 0)  # within the hammer gate, a third of its length
    t, events = run(z, tracks, feed, put(base, wallet, (25, 25, 30)), t, LOST_S + 1.0)
    assert ("gone", 1) in [(k, tr.id) for k, tr in events]
    (tr,) = tracks.confirmed
    assert tr.shown[0] == pytest.approx(110, rel=0.06) and tr.colour == "black"


def test_a_still_hand_is_drawn_still_on_the_table():
    """2 Nov: "the projection keeps shaking while tracking my hands, even when they're still". MediaPipe moves every
    landmark by a few px each frame; the skeleton and fingertip cursor are drawn from steadied landmarks."""
    slots = HandSlots(GEO)
    still = pose((400, 380), (420, 300))  # palm about 80 camera px
    rng = np.random.default_rng(3)
    tips, drawn, raw = [], [], []
    for i in range(75):  # 3 s at 25 fps
        noisy = HandPose(still.img + rng.normal(0, 2.0, still.img.shape), still.world, still.side, still.score)
        slots.update([noisy], i / 25)
        s = slots.active[0]
        tips.append(s.tip)
        drawn.append(s.shown)
        raw.append(noisy.img)
    steps = lambda a: np.linalg.norm(np.diff(np.asarray(a)[25:], axis=0), axis=-1)  # noqa: E731
    assert float(np.sqrt(np.mean(steps(drawn) ** 2))) < float(np.sqrt(np.mean(steps(raw) ** 2))) / 20
    assert float(np.max(steps(tips))) < 0.5  # design px: the cursor stands still
    assert slots.active[0].pose.img is not None and not np.array_equal(slots.active[0].pose.img, slots.active[0].shown)


def test_a_relearn_never_learns_our_own_light():
    """2 Nov: a relearn took in the labels lit at that moment; once they moved, the bare table where they had been
    became phantom objects ("charging cable" where nothing lay)."""
    from lightwork.contracts.render import Label
    from lightwork.playground import RELEARN_AFTER_S
    from lightwork.server import projector_style as style
    from lightwork.tabletop import LABEL_EM, OverlayHistory

    z, tracks, feed, base = learned()
    font = style.TEXT_PROJ_PX * style.unit_css_px(PAGE_W, PAGE_H)  # page px
    text, at = "CHARGING CABLE", (300.0, 250.0)  # page px
    hist = OverlayHistory(GEO, z)
    hist.set([Label(id="label", at=apply_h(HI, [at])[0].tolist(), text=text, role="info")], 0.0)
    w = LABEL_EM * font * len(text)
    glyphs = page_rect(at[0] + w / 2, at[1], w, 0.6 * font, 0)  # where the text lights the table
    brighter = base * 1.2  # the room light changes while the label is lit
    lit = put(brighter, glyphs, (255, 255, 255))
    t = 0.0
    for _ in range(int((RELEARN_AFTER_S + 2.0) / DT)):
        t += DT
        z.update(feed(lit), t, overlay=hist.at(t))
    assert z.relearns == 1
    under = z.poly_mask(apply_h(HI, page_rect(at[0] + w / 2, at[1], w - 10, 0.3 * font, 0))) > 0
    learned_there = z.ref[under].mean(axis=0) * z.gain
    bare = z._prep(feed(brighter))[under].mean(axis=0)
    assert np.all(np.abs(learned_there - bare) < 12)  # bare table in the new light, not the lit label
    hist.set([], t)  # the label goes dark
    _, events = run(z, tracks, feed, brighter, t, 2.0)
    assert events == [] and tracks.tracks == []


def test_a_phantom_already_tracked_still_heals():
    """An object in the reference was taken away, and its ghost was confirmed as a track before it could heal (labels
    all around it): it heals as soon as it can, tracked or not."""
    obj = page_rect(300, 200, 90, 50, 10)
    with_obj = put(table(), obj, (30, 35, 40))
    z, tracks, feed, _ = learned(with_obj)  # learned with the object on the table
    z.set_objects([apply_h(HI, obj)])  # a confirmed track sits on its ghost
    blobs = z.update(feed(table()), 1.0)  # the object is gone
    assert blobs == [] and z.healed == 1
    assert z.update(feed(table()), 1.1) == []


def test_a_relearn_after_a_light_change_never_takes_in_an_untracked_object():
    """3 Nov: a cloud changed the light just after a hammer was moved; the relearn made the hammer part of the empty
    table (it was not tracked at its new place yet), and it was never outlined there again."""
    from lightwork.playground import RELEARN_AFTER_S

    z, tracks, feed, base = learned()
    hammer = page_rect(400, 220, 300, 50, 20)
    brighter = put(base * 1.2, hammer, (30, 60, 220))  # the light changes while the hammer lies there untracked
    t = 0.0
    for _ in range(int((RELEARN_AFTER_S + 2.0) / DT)):
        t += DT
        z.update(feed(brighter), t, quiet=True)
    assert z.relearns == 1
    blobs = z.update(feed(brighter), t + DT, quiet=True)
    assert len(blobs) == 1 and blobs[0].sides[0] == pytest.approx(300, rel=0.06)  # still an object, whole
