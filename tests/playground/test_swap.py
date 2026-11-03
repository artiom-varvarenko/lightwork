"""3 Nov: a black card put where a teal ring had been took over the ring's track and its name ("blue rubber ring")."""

from __future__ import annotations

import numpy as np

from lightwork.tabletop import Blob, ObjectTracks, Track, model_tag, other_object


def blob(cx, cy, w, h, bgr, fill=0.9) -> Blob:
    return Blob(rect=((cx, cy), (w, h), 0.0), cam_poly=np.zeros((4, 2)), area=w * h * fill, fill=fill,
                bgr=np.array(bgr, float), colour="?", near_hand=False)


def tracks_with(b: Blob) -> tuple[ObjectTracks, Track]:
    tracks = ObjectTracks()
    tr = Track(blob=b, rect=b.rect, born=0.0, seen=0.0, id=1)
    tr.observe(b, force=True)
    tr.name = "blue rubber ring"
    tracks.tracks = [tr]
    return tracks, tr


def test_a_different_object_in_the_same_place_is_not_the_old_track():
    ring = blob(200, 150, 99, 90, (150, 140, 40), fill=0.6)  # teal ring: round, a hole
    card = blob(203, 152, 106, 75, (40, 40, 40), fill=0.95)  # black card: oblong, solid
    assert other_object(ring, (99, 90), card)
    tracks, tr = tracks_with(ring)
    assert tracks._match([card]) == {}


def test_a_cup_filling_with_dye_stays_the_same_object():
    cup = blob(200, 150, 80, 78, (200, 200, 200), fill=0.8)
    blue = blob(200, 150, 81, 78, (160, 90, 40), fill=0.8)  # same shape, new colour
    assert not other_object(cup, (80, 78), blue)
    tracks, tr = tracks_with(cup)
    assert tracks._match([blue]) == {0: 0}


def test_model_tag():
    assert model_tag("muse-spark-1.3", "meta") == "MUSE-SPARK-1.3"
    assert model_tag("meta-models/muse-spark-open", "baseten") == "MUSE-SPARK-OPEN @ BASETEN"
