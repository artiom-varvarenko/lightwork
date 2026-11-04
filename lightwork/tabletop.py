"""Table mode of ``lightwork play`` (``--table``): objects on the table outlined with their size, two hands tracked.

Local only (OpenCV + MediaPipe hand landmarks), not a procedure run and not evidence: no model calls, nothing recorded.

Objects
    Change detection over the projected page, minus the button column and the status line, against a learned empty
    table, at 1/``DOWN`` of the camera resolution. Light is handled as in :class:`lightwork.playground.DropZone`: a
    per-channel gain, slow background adaptation and a relearn after a lasting light change. Hands and forearms are
    masked with their landmarks, our own outlines and labels with the overlay. A blob that now looks like the table
    around it, where the reference held something else, is an object that lay there when the table was learned and has
    been taken away: the reference heals there at once.
Tracks
    Blobs are matched to tracks by centre distance. A track is shown once it held for ``CONFIRM_S``, is frozen while a
    hand touches it (its blob then shows the hand as much as the object) and is dropped ``LOST_S`` after it was last
    seen. A track that left with a hand gets its number back when a blob of its size and colour turns up within
    ``REID_S``: the object was moved.
Outlines and sizes
    The outline follows the object's own contour, pushed out by ``PAD_PAGE`` (a hammer gets a hammer-shaped outline).
    The size is the minimum-area rectangle of the contour mapped to page px through ``H_cam_to_page`` (the table plane).
    mm per page px, best first: ``--page-mm`` (the projected page's measured width), an ID-1 card (a bank card,
    85.60 x 53.98 mm) measured with the SCALE button and saved for later sessions, or the hands: MediaPipe's metric
    hand landmarks against their size on the table. Those assume an average hand and are shown with "~".
Names
    Optional (``play --names ollama``, :mod:`lightwork.namer`): each shown object's camera crop goes to a local
    vision-language model once; the answer ("hammer") labels it.
Hands
    Up to two, each in a stable slot across frames, with its skeleton projected onto it and its index fingertip as a
    cursor for the buttons. Both are drawn from steadied landmarks (:class:`lightwork.vision.hands.PoseFilter`): raw
    MediaPipe landmarks shake on a still hand. The buttons sit in a column at the left end, halfway across the table: by an edge of the
    page the camera saw only the fingertips of a hand pressing them, and MediaPipe needs the palm (2 Nov). Halfway
    across, the whole hand is in view whichever side it comes from. "L"/"R" is MediaPipe's handedness swapped: it assumes a mirrored selfie image, and a camera
    looking down at a table is not mirrored.

Geometry is laid out in the playground's design space (750 x 420, the projector page scaled): with ``play --rotate
180`` the page turns as a whole, so design space reads upright from the far side of the table.
"""

from __future__ import annotations

import itertools
import json
import math
import re
import time
from collections import deque
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from pathlib import Path

import cv2
import numpy as np

from .benchtask import BenchTask, match_item, spoken_list
from .games import GUESS_PROMPT, GUESS_SCHEMA, DrawGuess, GameHost, Pen, RockPaperScissors
from .helper import Helper, _with_article, asked_for_game, asked_to_go, asked_to_show, parse_place
from .helper import _words as _words_of
from .wireframe import build as build_model
from .guide import CHECK_PROMPT, CHECK_SCHEMA, PLAN_PROMPT, PLAN_SCHEMA, Guide, Seen, find_builtin, parse_plan, \
    plan_request_context, scale_plan
from .geometry import apply_h
from .playground import (
    ADAPT_RATE,
    DESIGN_H,
    DESIGN_W,
    EDGE_PX,
    HAND_CHAINS,
    INSET_PX,
    LIGHT_FG,
    LIGHT_GAIN,
    OVERLAY_HOLD_S,
    RELEARN_AFTER_S,
    STILL_DIFF,
    STILL_S,
    STUCK_EDGE_S,
    TIP_HOLD_S,
    Button,
    Canvas,
    Geo,
)
from .server import projector_style as style
from .vision.hands import INDEX_TIP, HandPose, JitterMeter, Pinch, PoseFilter, ShapeVote, finger_bends, pinch_ratio_3d
from .vision.overlay_mask import SUPERSAMPLE
from .vision.overlay_mask import _draw as draw_prim

DOWN = 2  # detection runs at 1/DOWN of the camera resolution
UP = 4  # an object's outline is fitted at 1/UP of a detection px
BLUR = 5  # Gaussian kernel, small px
OPEN_R, CLOSE_R = 1, 6  # morphology radii, small px (close 6: parts up to ~14 mm apart are one object; 3 Nov,
# a hammer's polished face reflected the table and its head became an object of its own)
EDGE_TOUCH = 5  # small px of a blob in the edge band that make it reach across the edge (an arm, a cut-off object)
NEAR_HAND = 4  # small px: a blob this close to the hand mask is touched by the hand
HAND_GROW = 0.3  # the hand mask grows by this share of the palm length (wrist to middle knuckle)
ARM_LEN = 3.0  # forearm masked from the wrist, in palm lengths
GHOST_RING = (3, 9)  # small px: the ring around a blob that shows the table next to it
GHOST_MIN_PX = 20
COL_X1 = 64.0  # design px: the button column runs along the left end up to here (3 Nov: 104 took too much table)
STATUS_Y1 = 62.0  # design px: the two status lines run along the far edge down to here
STATUS_Y = 17.0  # first line (what to do), vertical middle
STATUS_Y2 = 45.0  # second line (progress, a hint)
BTN_R = 15.0  # button radius: an icon, no word under it (the status bar says what a hovered button does)
ON_PAGE_PAD = 30.0  # design px: a hand whose fingertip is further off the page plays no part in a game (3 Nov: a plush
# toy beside the table was "a hand" at 0.5-0.99 confidence, and draw and guess, seeing a hand, never guessed)
HUD_RECTS = ((0.0, 0.0, COL_X1 + 4.0, DESIGN_H), (0.0, 0.0, DESIGN_W, STATUS_Y1 + 4.0))  # cut out of detection
PAD_PAGE = 7.0  # page px between an object and its projected outline (3 Nov: 14 read as "the outline is too big")
OUTLINE_EPS = 0.75  # page px: the outline polygon keeps the contour to this tolerance
OVERLAY_GROW_PX = 10  # camera px our projected strokes are grown by: camera blur plus calibration error
LABEL_EM = 0.75  # label width per character, font px (bold sans, digits and capitals), for laying labels out
MASK_EM = 0.95  # ... and for masking them: wide letters (W, M) must not stick out of the mask
CONFIRM_S = 0.4
CONFIRM_HITS = 3
LOST_S = 0.8
REID_S = 8.0
REID_SIZE = 0.2  # a re-found object's sides agree within this share ...
REID_BGR = 45.0  # ... and its median colour within this, per channel (in the learned light)
MATCH_SIZE = 0.35  # a blob matches a track only with sides within this share of the track sides
SWAP_ASPECT = 0.15  # with a different colour, an aspect ratio this different (share) ...
SWAP_FILL = 0.15  # ... or a fill this different, or sides this different (share): another object
SWAP_SIZE = 0.12
MOVE_PAGE = 2.0  # page px: smaller changes leave the drawn outline alone (no flicker, no overlay churn)
SIZE_TOL = 0.04
ANGLE_TOL = 4.0
GATE_PAGE = 40.0  # page px: a blob this far from a track (or 0.6 of its long side) can still be it
SIZE_HISTORY = 9
SHOW_PAGE = 2.0  # displayed sides change by at least this (page px) and SHOW_FRAC
SHOW_FRAC = 0.03
LEARN_QUIET_S = 1.0  # the table is learned once no hand was seen this long ...
LEARN_GRACE_S = 3.0  # ... and no sooner than this after "clear the table" was said
DWELL = {"learn": 1.0, "scale": 0.6, "home": 0.8, "recal": 1.2}  # LEARN forgets every object: it takes a longer hold (3 Nov: 1.5/1.0 felt slow)
ID1_MM = (85.60, 53.98)
CARD_ASPECT_TOL = 0.08
CARD_FILL = 0.85
SCALE_FILE = "play-scale.json"
PALM_SEGMENTS = ((0, 5), (0, 17), (5, 17), (0, 9))
HAND_SCALE_LEN = 300
HAND_SCALE_MIN = 15
HAND_ROLES = ("info", "info")  # white: coloured light smears colour into the camera image (see TableZone.update)
NOTE_S = 4.0
CARD_SEARCH_S = 6.0  # after SCALE, a card put down (and let go) within this long still counts
NAME_RETRY_S = 3.0  # a failed naming request is repeated after this
NAME_WAIT_S = 6.0  # a new object is announced once named, or after this (Spark: about 3 s per name)
CROP_MAX = 320  # px: long side of the crop sent for naming
NAME_MIN_PAGE = 16.0  # page px: objects narrower than this (crumbs, slivers) are not sent for a name
SCAN_SETTLE_S = 0.35  # the view has been still this long since it changed: scan it
SCAN_GAP_S = 0.8  # at least this long between scans
SCAN_ACK_TIMEOUT_S = 1.0  # the projector page draws the blanked table within this, or the scan is dropped and retried
SCAN_AFTER_ACK_S = 0.25  # frames dequeued this long after the ACK show the dark page (D7 said 0.12 over USB; 3 Nov,
# phone on Wi-Fi: frames 0.12 s after the ACK still showed the labels fading, and a relearn took one in)
SCAN_NO_LINK_S = 0.3  # without a page to ACK (tests, no projector): wait this long instead
SCAN_FRAMES = 2  # consecutive clean frames per scan, averaged
SCAN_AGREE = 5.0  # mean grey-level difference between them above which they disagree: something moved, a broken frame
MOTION_DOWN = 4  # motion is measured at 1/MOTION_DOWN of the detection resolution (1/8 of the camera's)
MOTION_DIFF = 16  # grey levels of frame-to-frame change that count as moving
MOTION_FRAC = 0.002  # share of the table that must move for a change (a hand moving across it: 1-3 %)
MOTION_GROW = 3  # motion px: where something moved, grown (a hand puts an object down beside where it moved)
TABLE_LIGHT_FG = 0.6  # share of the table changed that means the light changed (DropZone: 0.25; 3 Nov, a bag of chips
# put down with an arm over the table passed 0.25 and was taken for a light change)
SHADOW_RATIO = (0.55, 0.92)  # a shadow darkens the table to this share of its light ...
SHADOW_SPREAD = 0.07  # ... evenly: its three channel ratios lie this close together (3 Nov: daylight shadows outlined)
SHADOW_BRIDGE = 8  # small px (about 9 mm): a shadow-grey gap this short between two parts is part of the object
# (3 Nov: a hammer's grey metal neck passed for shadow and the head became an object of its own)
MISS_SCANS = 2  # a tracked object absent from this many clean scans in a row is gone, moved over or not
LOCAL_MAX = 0.45  # a change over more of the table than this is scanned whole; less, only where it happened (3 Nov:
LOCAL_GROW = 10  # small px around it; "if I move one object, the others should stay", no whole-table blink)
HAND_MOVE = 0.08  # palm lengths a hand moves between frames that count as a change (MediaPipe jitter is ~0.02)
LEARN_RETRY_S = 4.0  # with something dark still on the table, learning is tried again this often; never learned
# anyway (3 Nov: after two reminders it learned five objects into the "empty" table), unless LEARN is held again
OUTLINE_DRAW_MS = 320  # outlines draw themselves again after every scan ...
NEW_DRAW_MS = 650  # ... a new object's more slowly
TYPE_MS_PER_CHAR = 28  # names type themselves in
POINT_RECENT_S = 3.0  # a question is about what a fingertip pointed at within this long before it was heard ...
POINT_NEAR = 60.0  # ... within this many design px of the object's outline
ASKED_GLOW_S = 6.0  # the object asked about stays lit (green) this long after the answer
LOOK_AFTER_ACK_S = 0.35  # the helper's look: frames this long after the page showed dark (games fade things out, 0.18 s)
LOOK_FRAMES = 3  # averaged: games run the camera at a short exposure and a high gain
LOOK_TRIES = 3  # frames that disagree (a truncated camera frame, a moving hand) are taken again this often
LOOK_REUSE_S = 3.0  # a look this recent, with no hand seen since, is still what is on the table: no second blink
HEARD_AFTER_S = 8.0  # a transcript this long after the helper's mic shut is still of what was said into it; later, it isn't
PARTIAL_AFTER_S = 2.0  # let go this long ago, no final transcript: the last partial will do
PARTIAL_LATE_S = 15.0  # a final this soon after its partial was acted on is the same sentence
LOOK_HANDS_WAIT_S = 1.5  # a look waits this long for hands to leave the page (the one that held the mic)
LOOK_SCALE = 2  # px per design px of the picture the model gets: about what the camera has there
LOOK_MOVED = 10.0  # mean grey levels of change over a thing's box between two looks: it moved, or something did on it
SNAP_REACH = 30.0  # design px around a model's box where the object it means may lie (its boxes are off by up to 20)
SNAP_DIFF = 28.0  # colour levels from the table around it that count as the object
OUTLINE_GROW = 3.0  # design px SAM's box is grown by at most: the boxes are tight (Muse Wonder, then snap_box)
OUTLINE_MIN_ROUND = 0.18  # 4 pi area / perimeter^2 of a mask's contour: below this it is the frayed table around the thing
LOOK_LABELS = {"see": "SEEN", "plan": "PLAN", "locate": "FOUND", "check": "CHECKED", "answer": "ANSWER", "model": "MODEL", "place": "FOUND"}
STATUS_CHARS = 44  # a status line holds about this many characters
SWEEP_GAP_S = 12.0  # Muse Spark looks over the whole table at most this often ...
SWEEP_IDLE_S = 3.0  # ... and, while a job still misses things, once the table has been still this long
SWEEP_MAX_SHARE = 0.4  # a box over more of the table than this is not one object
SWEEP_PROMPT = ("You are an object grounding expert. This is a top-down photo of a white work table. List every "
                "separate physical object lying on it, including pale or white ones that are hard to see against the "
                "table. Ignore the table itself, shadows, reflections, stains and any light or lines projected onto it. "
                "Do not miss any object, and do not list one object twice. For each object give a short plain name "
                "(1 to 3 words) and its bounding box. The coordinates should be in the 0-1000 range of this image, "
                "x to the right and y down. Return a JSON object {\"objects\": [...]}.")
SWEEP_SCHEMA = {
    "type": "object",
    "properties": {"objects": {"type": "array", "items": {
        "type": "object",
        "properties": {"name": {"type": "string"}, "x_min": {"type": "integer"}, "y_min": {"type": "integer"},
                       "x_max": {"type": "integer"}, "y_max": {"type": "integer"}},
        "required": ["name", "x_min", "y_min", "x_max", "y_max"],
        "additionalProperties": False,
    }}},
    "required": ["objects"],
    "additionalProperties": False,
}
LOOK_WORDS = ("look again", "look at the table", "what's on the table", "whats on the table", "what is on the table",
              "scan the table", "find everything", "look closer", "look closely", "you missed")
BUTTON_TIPS = {"learn": ("HOLD TO LEARN THE EMPTY TABLE AGAIN", "TAKE EVERYTHING OFF IT FIRST"),
               "scale": ("HOLD, THEN PUT A BANK CARD DOWN", "THE CARD SETS THE MM SCALE"),
               "home": ("HOLD FOR THE MENU", "GAMES AND THE OBJECT BOARD"),
               "recal": ("HOLD TO RECALIBRATE: CLEAR THE TABLE", "THEN TOUCH THE THREE CIRCLES")}
_uids = itertools.count(1)


def _disk(r: int) -> np.ndarray:
    r = max(1, int(r))
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))


def local_scale(H: np.ndarray, xy) -> float:
    """Page px per camera px around camera point ``xy`` (square root of the homography's area scale there)."""
    p = np.asarray(xy, dtype=np.float64)
    a, b, c = apply_h(H, [p, p + (1.0, 0.0), p + (0.0, 1.0)])
    u, v = b - a, c - a
    return math.sqrt(abs(u[0] * v[1] - u[1] * v[0]))


def rel_colour_name(bgr, table_bgr) -> str:
    """Colour word for ``bgr`` seen in the same light as the (white-ish) table: the table's colour is divided out, which
    undoes a warm room light or a locked white balance."""
    wb = np.clip(np.asarray(bgr, np.float64) / np.maximum(np.asarray(table_bgr, np.float64), 1.0) * 200.0, 0, 255)
    h, s, v = (int(x) for x in cv2.cvtColor(wb.astype(np.uint8).reshape(1, 1, 3), cv2.COLOR_BGR2HSV)[0, 0])
    if v < 60:
        return "black"
    if s < 50:
        return "white" if v > 165 else "grey"
    for limit, name in ((8, "red"), (22, "orange"), (35, "yellow"), (85, "green"), (100, "cyan"), (130, "blue"), (165, "purple")):
        if h < limit:
            return name
    return "red"


def luma(bgr: np.ndarray) -> np.ndarray:
    """Brightness of a float BGR image (BT.601 weights)."""
    return bgr[..., 0] * 0.114 + bgr[..., 1] * 0.587 + bgr[..., 2] * 0.299


def canonical(rect) -> tuple[float, float, float, float, float]:
    """(cx, cy, long, short, angle of the long side in [0, 180)) of an OpenCV rotated rect."""
    (cx, cy), (w, h), a = rect
    return (cx, cy, max(w, h), min(w, h), (a if w >= h else a + 90.0) % 180.0)


def rect_changed(a, b) -> bool:
    ax, ay, al, as_, aa = canonical(a)
    bx, by, bl, bs, ba = canonical(b)
    if math.hypot(ax - bx, ay - by) > MOVE_PAGE:
        return True
    if abs(al - bl) > max(MOVE_PAGE, SIZE_TOL * al) or abs(as_ - bs) > max(MOVE_PAGE, SIZE_TOL * as_):
        return True
    period = 90.0 if min(al / max(as_, 1e-6), bl / max(bs, 1e-6)) < 1.15 else 180.0  # a square's angle is mod 90
    d = abs(aa - ba) % period
    return min(d, period - d) > ANGLE_TOL


def rect_moved(a, b, dist: float = 6.0, angle: float = 12.0) -> bool:
    """The object itself moved (page px, degrees), not just its blob's edge: fuzzy outlines and shadows change a blob's
    size from scan to scan (3 Nov: a plush toy's SAM outline was dropped and asked for again every scan)."""
    ax, ay, al, as_, aa = canonical(a)
    bx, by, bl, bs, ba = canonical(b)
    if math.hypot(ax - bx, ay - by) > dist:
        return True
    period = 90.0 if min(al / max(as_, 1e-6), bl / max(bs, 1e-6)) < 1.15 else 180.0
    d = abs(aa - ba) % period
    return min(d, period - d) > angle


def same_size(a: tuple[float, float], b: tuple[float, float], tol: float = MATCH_SIZE) -> bool:
    """Both sides within ``tol`` of each other (long with long, short with short); a few px always pass."""
    return all(abs(x - y) <= max(tol * max(x, y), 6.0) for x, y in zip(a, b, strict=True))


def other_object(tr_blob: Blob, sides: tuple[float, float], b: Blob) -> bool:
    """``b`` is plainly another object than the track's, although it lies where the track was: a different colour and
    a different shape (3 Nov: a black card put where a teal ring had been took over the ring's track and its name,
    "blue rubber ring"). A colour change alone is the same object (a cup filling with dye), and so is a shape change
    alone (an object turned over, a blob that took in a shadow)."""
    if float(np.max(np.abs(np.asarray(tr_blob.bgr, float) - np.asarray(b.bgr, float)))) <= REID_BGR:
        return False
    (la, sa), (lb, sb) = sides, b.sides
    aspect_a, aspect_b = la / max(sa, 1e-6), lb / max(sb, 1e-6)
    shape = abs(aspect_a - aspect_b) / max(aspect_a, aspect_b) > SWAP_ASPECT or abs(tr_blob.fill - b.fill) > SWAP_FILL
    size = any(abs(x - y) > SWAP_SIZE * max(x, y) for x, y in zip(sides, b.sides, strict=True))
    return shape or size


def grown(rect, pad: float):
    (cx, cy), (w, h), a = rect
    return ((cx, cy), (w + 2 * pad, h + 2 * pad), a)


def offset_outline(contour, pad: float, eps: float = OUTLINE_EPS) -> np.ndarray:
    """The contour (page px) pushed out by ``pad`` everywhere, concave parts included, as a simplified polygon: drawn
    at 2 px per page px, dilated by a disk and traced again."""
    c = np.asarray(contour, dtype=np.float64).reshape(-1, 2)
    k = 2.0
    o = c.min(axis=0) - pad - 2
    size = np.ceil((c.max(axis=0) + pad + 2 - o) * k).astype(int) + 1
    canvas = np.zeros((size[1], size[0]), np.uint8)
    cv2.fillPoly(canvas, [np.round((c - o) * k).astype(np.int32)], 255)
    canvas = cv2.dilate(canvas, _disk(round(pad * k)))
    cs, _ = cv2.findContours(canvas, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    big = max(cs, key=cv2.contourArea)
    poly = cv2.approxPolyDP(big, eps * k, True).reshape(-1, 2).astype(np.float64)
    return poly / k + o


# ---------------------------------------------------------------------------------------------------- detection


@dataclass(frozen=True, eq=False)
class Blob:
    rect: tuple  # ((cx, cy), (w, h), angle): minimum-area rectangle, page px
    cam_poly: np.ndarray  # its corners, camera px
    area: float  # camera px
    fill: float  # blob area / rectangle area
    bgr: np.ndarray  # median colour in the learned light
    colour: str
    near_hand: bool
    contour: np.ndarray | None = None  # the object's outer contour, page px

    @property
    def centre(self) -> np.ndarray:
        return np.asarray(self.rect[0], dtype=np.float64)

    @property
    def sides(self) -> tuple[float, float]:
        """(long, short), page px."""
        w, h = self.rect[1]
        return (max(w, h), min(w, h))


class TableZone:
    """Change detection over the projected page against a learned empty table (see the module doc)."""

    def __init__(self, geo: Geo, frame_size: tuple[int, int], *, exclude_design=(), threshold: int = 28,
                 min_area: int = 400, down: int = DOWN):
        self.geo = geo
        self.down = down
        fw, fh = frame_size
        page = apply_h(geo.Hi, [(0, 0), (geo.page_w, 0), (geo.page_w, geo.page_h), (0, geo.page_h)])
        x0, y0 = (max(0, math.floor(v)) for v in page.min(axis=0))
        x1, y1 = min(fw, math.ceil(page[:, 0].max())), min(fh, math.ceil(page[:, 1].max()))
        w, h = (x1 - x0) // down * down, (y1 - y0) // down * down
        if w < 16 * down or h < 16 * down:
            raise ValueError("the projected page is outside the camera view")
        self.roi = (x0, y0, w, h)  # camera px
        self.small = (w // down, h // down)
        full = np.zeros(self.small[::-1], np.uint8)
        cv2.fillPoly(full, [self._pts(page)], 255)
        for x0d, y0d, x1d, y1d in exclude_design:
            cv2.fillPoly(full, [self._pts(geo.d2c([(x0d, y0d), (x1d, y0d), (x1d, y1d), (x0d, y1d)]))], 0)
        self.inner = cv2.erode(full, _disk(round(INSET_PX / down)))
        self.edge = cv2.subtract(self.inner, cv2.erode(self.inner, _disk(round(EDGE_PX / down))))
        self.inner_px = int(np.count_nonzero(self.inner))
        if self.inner_px < 400:
            raise ValueError("the table area is too small in the camera view")
        self.threshold = threshold
        self.min_area = max(4, min_area / down**2)  # small px
        self.ref: np.ndarray | None = None  # float32 blurred small crop, in the light of the last (re)learn
        self.table_bgr = np.array([200.0, 200.0, 200.0])
        self.status = "learning"  # learning | ready | light
        self.gain = np.ones(3, np.float32)
        self.fg_fraction = 0.0
        self.relearns = 0
        self.healed = 0  # ghosts of objects taken away since the table was learned
        self._objmask = np.zeros(self.small[::-1], np.uint8)
        self._prev: np.ndarray | None = None
        self._still_since = 0.0
        self._hand_seen = -1e9
        self._light_since: float | None = None
        self._edge_since: float | None = None
        self.last_edge = False  # the last update saw something across the zone edge (an arm, a cut-off object)
        self.light_fg = TABLE_LIGHT_FG
        self.shadows = True  # drop soft shadows (even darkening) from the foreground

    # ---- coordinates

    def _pts(self, cam) -> np.ndarray:
        x0, y0 = self.roi[:2]
        p = (np.asarray(cam, dtype=np.float64).reshape(-1, 2) - (x0, y0)) / self.down
        return np.round(p).astype(np.int32)

    def to_cam(self, small_pts) -> np.ndarray:
        """Small px (pixel centres) -> camera px."""
        x0, y0 = self.roi[:2]
        return (np.asarray(small_pts, dtype=np.float64).reshape(-1, 2) + 0.5) * self.down - 0.5 + (x0, y0)

    def shrink(self, mask: np.ndarray | None) -> np.ndarray | None:
        """A full-frame camera mask cut to the zone, in small px; a pixel touched at all stays set."""
        if mask is None:
            return None
        x, y, w, h = self.roi
        crop = mask[y:y + h, x:x + w]
        small = cv2.resize(crop, self.small, interpolation=cv2.INTER_AREA)
        return np.where(small > 0, 255, 0).astype(np.uint8)

    def poly_mask(self, cam_poly) -> np.ndarray:
        m = np.zeros(self.small[::-1], np.uint8)
        cv2.fillPoly(m, [self._pts(cam_poly)], 255)
        return m

    # ---- reference

    def _prep(self, img: np.ndarray) -> np.ndarray:
        x, y, w, h = self.roi
        small = cv2.resize(img[y:y + h, x:x + w], self.small, interpolation=cv2.INTER_AREA)
        return cv2.GaussianBlur(small, (BLUR, BLUR), 0).astype(np.float32)

    def _set_ref(self, ref: np.ndarray) -> None:
        self.ref = ref
        self.table_bgr = np.median(ref[self.inner > 0][::5], axis=0).astype(np.float64)
        self.gain = np.ones(3, np.float32)
        self._light_since = self._edge_since = None

    def learn(self, img: np.ndarray) -> None:
        """(Re)learn the empty table: whatever lies on it now becomes table."""
        self._set_ref(self._prep(img))
        self._objmask[:] = 0
        self.status = "ready"

    def adopt(self, old: TableZone, img: np.ndarray) -> None:
        """Take over ``old``'s reference after a new mapping (same camera px): where the zones overlap the learned table
        is kept, the rest of this zone is learned from ``img``."""
        cur = self._prep(img)
        if old.ref is not None and old.down == self.down:
            ox, oy = (old.roi[0] - self.roi[0]) // self.down, (old.roi[1] - self.roi[1]) // self.down
            ow, oh = old.small
            sx0, sy0 = max(0, ox), max(0, oy)
            sx1, sy1 = min(self.small[0], ox + ow), min(self.small[1], oy + oh)
            if sx1 > sx0 and sy1 > sy0:
                cur[sy0:sy1, sx0:sx1] = old.ref[sy0 - oy:sy1 - oy, sx0 - ox:sx1 - ox] * old.gain
        self._set_ref(cur)
        self.status = "ready" if old.ref is not None else "learning"
        if old.ref is None:
            self.ref = None

    def _relearn(self, cur: np.ndarray, overlay: np.ndarray | None = None) -> None:
        """After a light change: the reference becomes the current view, except under objects, under our own
        projected light (``overlay``) and wherever the view does not look like bare table. 2 Nov: relearns took in the
        labels and outlines lit at that moment; once those moved, the bare table where they had been looked like new
        objects (a "charging cable" where nothing lay). 3 Nov: a relearn after a cloud took in a hammer just moved and
        not yet tracked there; it then split in two where it crossed its own ghost."""
        ref = cur.copy()
        keep = (self._objmask > 0) | ~self.table_like(cur)
        if overlay is not None:
            keep |= overlay > 0
        ref[keep] = self.ref[keep] * self.gain
        self._set_ref(ref)
        self.relearns += 1

    def table_like(self, cur: np.ndarray) -> np.ndarray:
        """Small px that look like bare (white) table in ``cur`` (a ``_prep`` image): not much darker than the table
        around them (a median over about 12 cm) and of its colour. Grown a little: object edges blur."""
        lum = luma(cur)
        k = max(31, (max(self.small) // 6) | 1)
        local = cv2.medianBlur(np.clip(lum, 0, 255).astype(np.uint8), k).astype(np.float32)
        dark = (local - lum) > self.threshold
        table = np.median(cur[self.inner > 0][::7], axis=0)
        chroma = cur / np.maximum(lum, 1.0)[..., None] - table / max(float(luma(table[None])[0]), 1.0)
        coloured = np.abs(chroma).max(axis=2) > 0.18
        thing = cv2.dilate((dark | coloured).astype(np.uint8) * 255, _disk(3))
        return thing == 0

    def set_objects(self, cam_polys) -> None:
        """Camera-px outlines of the tracked objects: kept out of the light estimate, the adaptation and a relearn."""
        m = np.zeros(self.small[::-1], np.uint8)
        for poly in cam_polys:
            cv2.fillPoly(m, [self._pts(poly)], 255)
        self._objmask = cv2.dilate(m, _disk(3)) if cam_polys else m

    def _background(self, overlay, hands) -> np.ndarray:
        bg = self.inner.copy()
        bg[self._objmask > 0] = 0
        if overlay is not None:
            bg[overlay > 0] = 0
        if hands is not None:
            bg[hands > 0] = 0
        return bg

    def _estimate_gain(self, cur: np.ndarray, bg: np.ndarray) -> np.ndarray:
        """Per channel: the median of per-pixel current / reference ratios, then again over the ratios near it (the
        bare table). A ratio of medians mixed objects with the light falloff: 3 Nov, a bag of chips over 40 % of the
        table moved it by more than ``LIGHT_GAIN`` and was taken for a light change."""
        sel = bg[::3, ::3] > 0
        if np.count_nonzero(sel) < 200:
            return self.gain
        c, r = cur[::3, ::3][sel], self.ref[::3, ::3][sel]
        ratio = c / np.maximum(r, 1.0)
        m = np.median(ratio, axis=0)
        near = np.all(np.abs(ratio - m) < 0.15, axis=1)
        if np.count_nonzero(near) >= 100:
            m = np.median(ratio[near], axis=0)
        return m.astype(np.float32)

    def _adapt(self, cur: np.ndarray, bg: np.ndarray, fg: np.ndarray) -> None:
        """Quiet background drifts into the reference, keeping its overall level (see ``DropZone._adapt``)."""
        sel = bg[::3, ::3] > 0
        if np.count_nonzero(sel) < 200:
            return
        before = np.median(self.ref[::3, ::3][sel], axis=0)
        adapt = cv2.bitwise_and(bg, cv2.bitwise_not(fg))
        cv2.accumulateWeighted(cur / self.gain, self.ref, ADAPT_RATE, mask=adapt)  # ref += rate * (cur - ref) there
        after = np.median(self.ref[::3, ::3][sel], axis=0)
        self.ref *= (before / np.maximum(after, 1.0)).astype(np.float32)

    # ---- detection

    def hand_mask(self, poses: list[HandPose]) -> np.ndarray | None:
        """Small-px mask of the hands (landmark hull) and forearms (from the wrist, away from the fingers), grown."""
        if not poses:
            return None
        m = np.zeros(self.small[::-1], np.uint8)
        grow = 1.0
        for p in poses:
            img = p.img
            palm = float(np.linalg.norm(img[9] - img[0]))
            if palm < 2.0:
                continue
            cv2.fillPoly(m, [self._pts(cv2.convexHull(img.astype(np.float32)).reshape(-1, 2))], 255)
            d = (img[0] - img[9]) / palm
            n = np.array([-d[1], d[0]])
            half = 0.6 * max(float(np.linalg.norm(img[5] - img[17])), 0.6 * palm)
            a, b = img[0] - 0.3 * palm * d, img[0] + ARM_LEN * palm * d
            cv2.fillPoly(m, [self._pts([a + n * half, b + n * half * 1.2, b - n * half * 1.2, a - n * half])], 255)
            grow = max(grow, HAND_GROW * palm / self.down)
        return cv2.dilate(m, _disk(round(grow)))

    def touches(self, hands: np.ndarray | None, cam_poly) -> bool:
        """A hand (``hand_mask``) lies on or within ``NEAR_HAND`` small px of the camera-px polygon."""
        if hands is None:
            return False
        pts = self._pts(cam_poly)
        x0, y0 = np.maximum(pts.min(axis=0) - NEAR_HAND, 0)
        x1, y1 = np.minimum(pts.max(axis=0) + NEAR_HAND + 1, self.small)
        if x1 <= x0 or y1 <= y0:
            return False
        win = np.zeros((y1 - y0, x1 - x0), np.uint8)
        cv2.fillPoly(win, [pts - (x0, y0)], 255)
        win = cv2.dilate(win, _disk(NEAR_HAND))
        return bool(np.any(hands[y0:y1, x0:x1][win > 0]))

    def stray_objects(self, img: np.ndarray) -> int:
        """Dark things on the white table in ``img``, found without a reference: blobs of at least ``min_area`` well
        below the local table brightness (a median over about 12 cm). Checked before the empty table is learned: 2 Nov,
        a hammer left lying there was learned as table and never outlined again."""
        lum = np.clip(luma(self._prep(img)), 0, 255).astype(np.uint8)
        k = max(31, (max(self.small) // 6) | 1)
        dark = ((cv2.medianBlur(lum, k).astype(np.int16) - lum) > 2 * self.threshold) & (self.inner > 0)
        n, _, stats, _ = cv2.connectedComponentsWithStats(dark.astype(np.uint8), connectivity=8)
        return sum(1 for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] >= self.min_area)

    def update(self, img: np.ndarray, t: float, *, overlay: np.ndarray | None = None, hands: np.ndarray | None = None,
               hand_visible: bool = False, quiet: bool | None = None, keep_edge: bool = False,
               region: np.ndarray | None = None) -> list[Blob] | None:
        """Blobs on the table now, or None (no reference, or the light changed). ``overlay`` and ``hands`` are small-px
        masks (``shrink``, ``hand_mask``). ``quiet``: the caller knows whether the view is still with no hand around
        (scans, far apart in time); None works it out from successive updates."""
        if self.ref is None:
            return None
        if hand_visible:
            self._hand_seen = t
        cur = self._prep(img)
        if self._prev is None or float(np.mean(np.abs(cur[::3, ::3] - self._prev[::3, ::3]))) > STILL_DIFF:
            self._still_since = t
        self._prev = cur
        bg = self._background(overlay, hands)
        self.gain = self._estimate_gain(cur, bg)
        lit = self.ref * self.gain
        b, g, r = cv2.split(cv2.absdiff(cur, lit))
        d = cv2.max(cv2.max(b, g), r)  # largest channel difference
        fg = (d > self.threshold).astype(np.uint8) * 255
        fg &= self.inner
        if self.shadows:
            # a soft shadow keeps the table's colour and darkens all three channels alike, moderately; a black object
            # darkens far more, a coloured one unevenly. Mid-grey objects lose pixels here; SAM restores their shape.
            ratio = cur / np.maximum(lit, 1.0)
            lo, hi = ratio.min(axis=2), ratio.max(axis=2)
            shadow = (lo >= SHADOW_RATIO[0]) & (hi <= SHADOW_RATIO[1]) & (hi - lo <= SHADOW_SPREAD)
            core = fg.copy()
            core[shadow] = 0
            bridged = cv2.morphologyEx(core, cv2.MORPH_CLOSE, _disk(SHADOW_BRIDGE))
            fg = core | (fg & bridged)  # shadow-grey only where it joins parts of one object
        if overlay is not None:
            # Our projected light only adds light. Under it, a pixel is an object only if it is darker than the table:
            # the outline of one object crossing another must not cut that one in two (2 Nov: a wallet with a card in
            # it became two objects, each outline keeping the other cut). Darker in brightness (luma), not in a
            # channel: the camera's compressed video smears the colour of a saturated line into its neighbours (a red
            # outline read R +57, G -36, B -21 beside it, 2 Nov), and "darker in G" made phantom objects of our own
            # outlines, which then got outlines of their own.
            darker = (luma(lit) - luma(cur) > self.threshold).astype(np.uint8) * 255
            fg = np.where(overlay > 0, fg & darker, fg)
        fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, _disk(OPEN_R))
        fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, _disk(CLOSE_R))
        free = fg if hands is None else cv2.bitwise_and(fg, cv2.bitwise_not(hands))
        self.fg_fraction = float(np.count_nonzero(free)) / self.inner_px
        self.last = {"cur": cur, "d": d, "fg": fg, "free": free, "overlay": overlay, "hands": hands}  # for dump()

        if quiet is None:
            quiet = t - self._still_since >= STILL_S and t - self._hand_seen >= STILL_S
        self.last_edge = False
        if self.fg_fraction > self.light_fg or float(np.max(np.abs(self.gain - 1.0))) > LIGHT_GAIN:
            self._light_since = t if self._light_since is None else self._light_since
            self.status = "light"
            if t - self._light_since >= RELEARN_AFTER_S and quiet:
                self._relearn(cur, overlay)
            return None
        self._light_since = None
        self.status = "ready"
        blobs, edge = self._blobs(cur, d, free, bg, hands, overlay, keep_edge=keep_edge, region=region)
        self.last_edge = edge
        if edge:
            self._edge_since = t if self._edge_since is None else self._edge_since
            if t - self._edge_since >= STUCK_EDGE_S and quiet:  # something lies across the edge: it is table now
                self._relearn(cur, overlay)
                return None
        else:
            self._edge_since = None
        self._adapt(cur, bg, fg)
        return blobs

    def dump(self, path: Path) -> None:
        """What the zone saw at its last update, for debugging: crops of the frame, reference and masks (small px)."""
        path.mkdir(parents=True, exist_ok=True)
        last = getattr(self, "last", None)
        if self.ref is not None:
            cv2.imwrite(str(path / "ref.png"), np.clip(self.ref, 0, 255).astype(np.uint8))
        cv2.imwrite(str(path / "inner.png"), self.inner)
        if last is None:
            return
        cv2.imwrite(str(path / "cur.png"), np.clip(last["cur"], 0, 255).astype(np.uint8))
        cv2.imwrite(str(path / "diff.png"), np.clip(last["d"] * 2, 0, 255).astype(np.uint8))
        for k in ("fg", "free", "overlay", "hands"):
            if last[k] is not None:
                cv2.imwrite(str(path / f"{k}.png"), last[k])

    def _blobs(self, cur, d, free, bg, hands, overlay=None, keep_edge: bool = False,
               region: np.ndarray | None = None) -> tuple[list[Blob], bool]:
        """``region``: a local scan's area; blobs that reach into it are taken whole, the rest left alone (3 Nov: the area
        was cut out of the view, and an untouched power bank half inside it became a new, smaller "black box")."""
        n, labels, stats, cents = cv2.connectedComponentsWithStats(free, connectivity=8)
        near = None if hands is None else cv2.dilate(hands, _disk(NEAR_HAND))
        out, edge = [], False
        for i in range(1, n):
            area = int(stats[i, cv2.CC_STAT_AREA])
            if area < self.min_area:
                continue
            x, y, w, h = (int(v) for v in stats[i, :4])
            sub = labels[y:y + h, x:x + w] == i
            if region is not None and not np.any(region[y:y + h, x:x + w][sub]):
                continue
            if np.count_nonzero(sub & (self.edge[y:y + h, x:x + w] > 0)) > EDGE_TOUCH and not keep_edge:
                edge = True  # an arm reaching in, or an object cut by the edge
                continue
            touched = near is not None and bool(np.any(near[y:y + h, x:x + w][sub]))
            # tracked blobs too: a phantom that outlived its first frames was confirmed, and then never healed (2 Nov)
            if not touched and self._heal_ghost(labels, i, (x, y, w, h), cur, bg, overlay):
                continue
            if self._stray_light(cur, sub, x, y):  # after healing: a removed object's ghost is brighter too
                continue
            out.append(self._measure(self._shape(sub, x, y, d), sub, x, y, cur, touched))
        return out, edge

    def _stray_light(self, cur: np.ndarray, sub: np.ndarray, x: int, y: int) -> bool:
        """The blob is mostly brighter than the bare table there: light from elsewhere, not a thing on a white table
        (3 Nov: a phone notification drawn over the projector page was scanned as "blurry text"; sun patches)."""
        h, w = sub.shape
        win = cur[y:y + h, x:x + w][sub]
        table = (self.ref[y:y + h, x:x + w] * self.gain)[sub]
        brighter = (luma(win) - luma(table)) > 0.5 * self.threshold
        return bool(np.mean(brighter) > 0.5)

    def _heal_ghost(self, labels: np.ndarray, i: int, box, cur: np.ndarray, bg: np.ndarray,
                    overlay: np.ndarray | None = None) -> bool:
        """The blob looks like the table around it now, and the reference held something else there: an object that
        lay there when the table was learned was taken away. Learn the table there; True when healed."""
        x, y, w, h = box
        r0, r1 = GHOST_RING
        X0, Y0 = max(0, x - r1), max(0, y - r1)
        X1, Y1 = min(self.small[0], x + w + r1), min(self.small[1], y + h + r1)
        c = (labels[Y0:Y1, X0:X1] == i).astype(np.uint8) * 255
        ring = cv2.subtract(cv2.dilate(c, _disk(r1)), cv2.dilate(c, _disk(r0)))
        ring &= bg[Y0:Y1, X0:X1]
        if np.count_nonzero(ring) < GHOST_MIN_PX:
            return False
        win_cur, win_ref = cur[Y0:Y1, X0:X1], self.ref[Y0:Y1, X0:X1]
        inside = c > 0
        table = np.median(win_cur[ring > 0], axis=0)
        now = np.median(win_cur[inside], axis=0)
        before = np.median(win_ref[inside], axis=0) * self.gain
        if np.max(np.abs(now - table)) < 0.5 * self.threshold and np.max(np.abs(before - table)) > self.threshold:
            heal = cv2.dilate(c, _disk(2)) > 0
            if overlay is not None:
                heal &= overlay[Y0:Y1, X0:X1] == 0  # never learn our own light as table
            win_ref[heal] = win_cur[heal] / self.gain
            self.healed += 1
            return True
        return False

    def _shape(self, sub: np.ndarray, x: int, y: int, d: np.ndarray) -> tuple[np.ndarray, int, int, int]:
        """The blob's outline at sub-pixel precision: its window of the difference map, upsampled ``UP`` times and cut
        where the blob's own contrast has fallen to half (never below the threshold). The threshold alone cuts a blurred
        high-contrast edge well outside the object (+4-5 page px on a 120 x 70 px block), and a soft shadow along it
        with it. The blob's own interior always stays in: a part of lower contrast than the rest, like a
        card in a black wallet, is still the same object. Returns (mask, x0, y0, up): mask pixel j is centred on small
        px x0 + (j + 0.5) / up - 0.5."""
        h, w = sub.shape
        X0, Y0 = max(0, x - 2), max(0, y - 2)
        X1, Y1 = min(self.small[0], x + w + 2), min(self.small[1], y + h + 2)
        blob = np.zeros((Y1 - Y0, X1 - X0), np.uint8)
        blob[y - Y0:y - Y0 + h, x - X0:x - X0 + w][sub] = 255
        win = d[Y0:Y1, X0:X1]
        core = cv2.erode(blob, _disk(1), borderType=cv2.BORDER_CONSTANT, borderValue=0)
        cut = float(self.threshold)
        if np.count_nonzero(core) >= 8:
            cut = max(cut, 0.5 * float(np.median(win[core > 0])))
        big = cv2.resize(win, None, fx=UP, fy=UP, interpolation=cv2.INTER_LINEAR)
        near = cv2.resize(cv2.dilate(blob, _disk(1)), big.shape[::-1], interpolation=cv2.INTER_NEAREST)
        # never cut a part of it away; eroded by 2 px, as the threshold put a high-contrast edge up to 1.5 px outside
        inside = cv2.erode(blob, _disk(2), borderType=cv2.BORDER_CONSTANT, borderValue=0)
        inside = cv2.resize(inside, big.shape[::-1], interpolation=cv2.INTER_NEAREST)
        m = (((big > cut) & (near > 0)) | (inside > 0)).astype(np.uint8)
        n, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
        if n >= 2:
            whole = np.count_nonzero(sub) * UP * UP
            big = [k for k in range(1, n) if stats[k, cv2.CC_STAT_AREA] >= 0.15 * whole]
            if len(big) == 1 and stats[big[0], cv2.CC_STAT_AREA] >= 0.3 * whole:
                return labels == big[0], X0, Y0, UP
            # several big parts: two objects merged, one of lower contrast. Never drop one (3 Nov: a red toy 5 mm from a
            # black power bank vanished); the blob stays whole
        return sub, x, y, 1

    def _measure(self, shape, sub: np.ndarray, x: int, y: int, cur: np.ndarray, touched: bool) -> Blob:
        mask, X0, Y0, up = shape
        cs, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        pts = max(cs, key=len).reshape(-1, 2).astype(np.float64)
        cam = self.to_cam((pts + 0.5) / up - 0.5 + (X0, Y0))
        s = local_scale(self.geo.H, cam.mean(axis=0))
        page = apply_h(self.geo.H, cam)
        (cx, cy), (w, h), a = cv2.minAreaRect(page.astype(np.float32))
        k = self.down / up * s  # contour points are mask pixel centres: the edge lies half a mask px further out
        rect = ((float(cx), float(cy)), (float(w) + k, float(h) + k), float(a))
        area_page = np.count_nonzero(mask) / up**2 * (self.down * s) ** 2
        bgr = np.median(cur[y:y + sub.shape[0], x:x + sub.shape[1]][sub], axis=0) / np.maximum(self.gain, 0.05)
        return Blob(rect=rect, cam_poly=apply_h(self.geo.Hi, cv2.boxPoints(rect)),
                    area=float(np.count_nonzero(mask)) / up**2 * self.down**2,
                    fill=float(min(1.0, area_page / max(rect[1][0] * rect[1][1], 1e-6))), bgr=bgr,
                    colour=rel_colour_name(bgr, self.table_bgr), near_hand=touched, contour=page)


# ---------------------------------------------------------------------------------------------------- tracks


@dataclass(eq=False)
class Track:
    blob: Blob  # latest observation
    rect: tuple  # drawn rectangle, page px (changes only past MOVE_PAGE / SIZE_TOL / ANGLE_TOL)
    born: float
    seen: float
    id: int = 0  # 0 until confirmed
    hits: int = 1
    held: bool = False
    held_t: float = -1e9
    misses: int = 0  # clean scans in a row that did not see it
    lost_t: float | None = None
    sizes: deque = field(default_factory=lambda: deque(maxlen=SIZE_HISTORY))
    shown: tuple[float, float] | None = None  # displayed (long, short), page px
    outline: np.ndarray | None = None  # drawn outline, page px: the contour pushed out by PAD_PAGE
    uid: int = field(default_factory=lambda: next(_uids))  # unique in the process (the shown id is reused)
    name: str | None = None
    name_state: str = "none"  # none | pending | done
    name_t: float = -1e9  # when it was last asked for
    born_scan: int = 0  # the scan that first saw it (its outline draws itself more slowly then)
    refined: bool = False  # the outline follows a SAM mask
    sam_state: str = "none"  # none | pending | done
    sam_rect: tuple | None = None  # the rect when SAM was asked: a reply for an object that moved since is dropped
    item: int = -1  # task item it is (1-based), 0 = not on the task's list, -1 = not known yet
    asked_until: float = -1e9  # lit while a spoken question about it is answered
    sticky: bool = False  # found by a Muse Spark sweep, not by the scans (too pale to tell from the table): kept
    # until something moves over it

    def __post_init__(self) -> None:
        if self.outline is None:
            self.outline = self._outline(self.blob)

    @staticmethod
    def _outline(b: Blob) -> np.ndarray:
        if b.contour is None or len(b.contour) < 3:
            return cv2.boxPoints(grown(b.rect, PAD_PAGE)).astype(np.float64)
        return offset_outline(b.contour, PAD_PAGE)

    @property
    def confirmed(self) -> bool:
        return self.id > 0

    @property
    def colour(self) -> str:
        return self.blob.colour

    def observe(self, b: Blob, *, force: bool = False) -> None:
        self.blob = b
        self.sizes.append(b.sides)
        if self.refined and not force and not rect_moved(self.rect, b.rect):
            return  # the SAM outline and size stand while the object stays put
        if force or rect_changed(self.rect, b.rect):
            self.rect = b.rect
            self.outline = self._outline(b)
            self.refined, self.sam_state = False, "none"  # moved: the SAM outline no longer fits
        if self.refined and not force:  # the SAM mask's size stands while the object stays put
            return
        med = tuple(float(v) for v in np.median(np.array(self.sizes), axis=0))
        if force or self.shown is None or any(
                abs(m - s) > max(SHOW_PAGE, SHOW_FRAC * s) for m, s in zip(med, self.shown, strict=True)):
            self.shown = med

    def cam_outline(self, geo: Geo) -> np.ndarray:
        return apply_h(geo.Hi, self.outline)

    def refine(self, contour_page: np.ndarray) -> None:
        """Outline and size from a segmentation mask's contour (page px) instead of the change blob's (which also holds
        soft shadows: 3 Nov, a card holder measured 120 x 84 mm)."""
        self.outline = offset_outline(contour_page, PAD_PAGE)
        (_, _), (w, h), _ = cv2.minAreaRect(contour_page.astype(np.float32))
        self.shown = (float(max(w, h)), float(min(w, h)))
        self.refined, self.sam_state = True, "done"


class ObjectTracks:
    """Blobs -> tracks with stable numbers (see the module doc)."""

    def __init__(self) -> None:
        self.tracks: list[Track] = []
        self.lost: list[Track] = []  # confirmed tracks that left with a hand, kept for REID_S

    @property
    def confirmed(self) -> list[Track]:
        return [tr for tr in self.tracks if tr.confirmed]

    def clear(self) -> None:
        self.tracks, self.lost = [], []

    def _free_id(self) -> int:
        used = {tr.id for tr in self.tracks + self.lost}
        return next(i for i in range(1, len(used) + 2) if i not in used)

    def _reid(self, b: Blob, candidates: list[Track]) -> Track | None:
        """The candidate a new blob most likely is: sides within ``REID_SIZE`` and a similar colour. The colour word of
        a many-coloured object flips (a hammer: red handle, grey head) - compare the median colours instead."""
        best, best_err = None, REID_SIZE
        for tr in candidates:
            if float(np.max(np.abs(np.asarray(tr.blob.bgr) - np.asarray(b.bgr)))) > REID_BGR:
                continue
            err = max(abs(x - y) / max(y, 1e-6) for x, y in zip(b.sides, tr.shown or tr.blob.sides, strict=True))
            if err <= best_err:
                best, best_err = tr, err
        return best

    def update(self, blobs: list[Blob], t: float, touched) -> list[tuple[str, Track]]:
        """``touched(cam_poly) -> bool``: a hand lies on or next to that polygon. Returns events (kind, track) with kind
        ``new`` (confirmed), ``moved`` (re-found elsewhere after a hand took it) or ``gone``."""
        events: list[tuple[str, Track]] = []
        pairs = sorted(
            (float(np.hypot(*(tr.blob.centre - b.centre))), i, j)
            for i, tr in enumerate(self.tracks) for j, b in enumerate(blobs)
        )
        match: dict[int, int] = {}
        taken: set[int] = set()
        for d, i, j in pairs:
            if i in match or j in taken or d > max(GATE_PAGE, 0.6 * self.tracks[i].blob.sides[0]):
                continue
            tr, b = self.tracks[i], blobs[j]
            if not b.near_hand and not same_size(tr.shown or tr.blob.sides, b.sides):
                continue  # 2 Nov: a hammer track took over a wallet put down near where the hammer had been
            if not b.near_hand and other_object(tr.blob, tr.shown or tr.blob.sides, b):
                continue
            match[i] = j
            taken.add(j)
        keep: list[Track] = []
        for i, tr in enumerate(self.tracks):
            j = match.get(i)
            if j is not None:
                b = blobs[j]
                tr.seen = t
                if b.near_hand:  # the blob shows the hand too: keep the object as it was
                    tr.held, tr.held_t = True, t
                else:
                    tr.held = False
                    tr.observe(b)
                    tr.hits += 1
                    if not tr.confirmed and tr.hits >= CONFIRM_HITS and t - tr.born >= CONFIRM_S:
                        tr.id = self._free_id()
                        events.append(("new", tr))
                keep.append(tr)
            elif touched(tr.blob.cam_poly):  # under a hand
                tr.held, tr.held_t, tr.seen = True, t, t
                keep.append(tr)
            elif t - tr.seen <= LOST_S:
                tr.held = False
                keep.append(tr)
            elif tr.confirmed:
                if t - tr.held_t <= LOST_S + 0.5:  # it left with a hand: it may turn up elsewhere
                    tr.lost_t = t
                    self.lost.append(tr)
                events.append(("gone", tr))
        self.lost = [tr for tr in self.lost if t - tr.lost_t <= REID_S]
        # confirmed tracks a hand held a moment ago and that are not where they were: a hand slid them further than
        # the match gate and let go before they were dropped
        orphans = [tr for i, tr in enumerate(self.tracks)
                   if i not in match and tr in keep and tr.confirmed and not tr.held and t - tr.held_t <= REID_S]
        for j, b in enumerate(blobs):
            if j in taken or b.near_hand:  # a blob a hand touches may be its shadow: never a new object
                continue
            old = self._reid(b, orphans + self.lost)
            if old is not None:
                if old in orphans:
                    orphans.remove(old)
                else:
                    self.lost.remove(old)
                    keep.append(old)
                old.observe(b, force=True)
                old.seen, old.held, old.lost_t = t, False, None
                events.append(("moved", old))
                continue
            tr = Track(blob=b, rect=b.rect, born=t, seen=t)
            tr.observe(b, force=True)
            keep.append(tr)
        self.tracks = keep
        return events

    def _match(self, blobs: list[Blob]) -> dict[int, int]:
        """Track index -> blob index: nearest first, within the gate, of the same size (a blob a hand touches shows the
        hand too, so its size is not compared)."""
        pairs = sorted(
            (float(np.hypot(*(tr.blob.centre - b.centre))), i, j)
            for i, tr in enumerate(self.tracks) for j, b in enumerate(blobs)
        )
        match: dict[int, int] = {}
        taken: set[int] = set()
        for d, i, j in pairs:
            if i in match or j in taken or d > max(GATE_PAGE, 0.6 * self.tracks[i].blob.sides[0]):
                continue
            tr, b = self.tracks[i], blobs[j]
            if not b.near_hand and not same_size(tr.shown or tr.blob.sides, b.sides):
                continue
            if not b.near_hand and other_object(tr.blob, tr.shown or tr.blob.sides, b):
                continue  # a different thing put where it was: a new track (and a new name)
            match[i] = j
            taken.add(j)
        return match

    def update_scan(self, blobs: list[Blob], t: float, touched, allowed=None, inside=None) -> list[tuple[str, Track]]:
        """One scan: a clean, still view in which every object on the table is a blob, except under a hand.

        ``touched(cam_poly)``: a hand lies on or next to that polygon. ``allowed(cam_poly)``: something moved over it
        since the last scan (None: anywhere). Objects neither appear nor vanish where nothing moved: what does is a
        light change or a miss, not an object. A blob matching a track follows it (frozen while a hand touches it); a
        track with no blob is gone, unless a hand covers it or nothing moved there; a blob with no track is an object
        that moved here (same size and colour as one gone now or within ``REID_S``: it keeps its number and name), or a
        new one, confirmed at once. Returns events (kind, track): ``new``, ``moved``, ``gone``."""
        events: list[tuple[str, Track]] = []
        fixed: list[Track] = []
        if inside is not None:  # a local scan: what lies outside where it looked stays as it was
            fixed = [tr for tr in self.tracks if not inside(tr.blob.cam_poly)]
            self.tracks = [tr for tr in self.tracks if tr not in fixed]
            blobs = [b for b in blobs if inside(b.cam_poly)]
        self._fixed = fixed
        match = self._match(blobs)
        taken = set(match.values())
        keep: list[Track] = []
        vanished: list[Track] = []
        for i, tr in enumerate(self.tracks):
            j = match.get(i)
            if j is not None:
                b = blobs[j]
                tr.seen, tr.misses = t, 0
                if b.near_hand:
                    tr.held, tr.held_t = True, t
                else:
                    tr.held = False
                    tr.observe(b)
                    tr.hits += 1
                    if not tr.confirmed:
                        tr.id = self._free_id_with(keep)
                        events.append(("new", tr))
                keep.append(tr)
            elif touched(tr.blob.cam_poly):
                tr.held, tr.held_t, tr.seen = True, t, t
                keep.append(tr)
            elif tr.sticky and not (allowed is not None and allowed(tr.blob.cam_poly)):
                keep.append(tr)  # a model found it where the scans see nothing; nothing moved there since
            elif allowed is not None and not allowed(tr.blob.cam_poly) and tr.misses + 1 < MISS_SCANS:
                tr.misses += 1
                keep.append(tr)  # nothing moved there: still there, unless the next scan misses it too
            else:
                vanished.append(tr)
        self.lost = [tr for tr in self.lost if t - tr.lost_t <= REID_S]
        for j, b in enumerate(blobs):
            if j in taken or b.near_hand:  # a blob a hand touches may be its shadow: never a new object
                continue
            old = self._reid(b, [tr for tr in vanished if tr.confirmed] + self.lost)
            if old is not None:
                if old in vanished:
                    vanished.remove(old)
                else:
                    self.lost.remove(old)
                old.observe(b, force=True)
                old.seen, old.held, old.lost_t = t, False, None
                keep.append(old)
                events.append(("moved", old))
                continue
            if allowed is not None and not allowed(b.cam_poly):
                continue
            tr = Track(blob=b, rect=b.rect, born=t, seen=t)
            tr.observe(b, force=True)
            tr.id = self._free_id_with(keep)
            keep.append(tr)
            events.append(("new", tr))
        for tr in vanished:
            if tr.confirmed:
                tr.lost_t, tr.held = t, False
                self.lost.append(tr)
                events.append(("gone", tr))
        self.tracks = fixed + keep
        self._fixed = []
        return events

    def _free_id_with(self, extra: list[Track]) -> int:
        used = {tr.id for tr in self.tracks + self.lost + extra + getattr(self, "_fixed", [])}
        return next(i for i in range(1, len(used) + 2) if i not in used)


# ---------------------------------------------------------------------------------------------------- scale


@dataclass(frozen=True)
class Scale:
    mm_per_px: float  # mm on the table per page CSS px
    source: str  # page | card | hands

    @property
    def approx(self) -> bool:
        return self.source == "hands"

    def mm(self, sides_page: tuple[float, float]) -> tuple[float, float]:
        return (sides_page[0] * self.mm_per_px, sides_page[1] * self.mm_per_px)

    def text(self, sides_page: tuple[float, float]) -> str:
        a, b = self.mm(sides_page)
        return f"{'~' if self.approx else ''}{a:.0f}x{b:.0f}mm"


def scale_path(state_dir: Path) -> Path:
    return state_dir / SCALE_FILE


def load_card_scale(state_dir: Path, page_w: float, page_h: float) -> Scale | None:
    """A card scale saved by an earlier session on a page of this size (the projector has not been zoomed or moved:
    page px on the table keep their size)."""
    try:
        data = json.loads(scale_path(state_dir).read_text(encoding="utf-8"))
        if [round(page_w), round(page_h)] != [round(v) for v in data["page"]]:
            return None
        return Scale(float(data["mm_per_page_px"]), "card")
    except (OSError, ValueError, KeyError, TypeError):
        return None


def save_card_scale(state_dir: Path, scale: Scale, page_w: float, page_h: float) -> Path:
    path = scale_path(state_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {"mm_per_page_px": round(scale.mm_per_px, 5), "page": [page_w, page_h], "source": scale.source,
            "created": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")}
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    return path


def find_card(tracks: list[Track]) -> tuple[Scale, Track] | None:
    """An ID-1 card (bank card) among the shown objects: long/short within ``CARD_ASPECT_TOL`` of 1.586 and nearly
    rectangular. The best-shaped one sets the scale."""
    ratio = ID1_MM[0] / ID1_MM[1]
    best = None
    for tr in tracks:
        if not tr.confirmed or tr.held or tr.shown is None:
            continue
        long, short = tr.shown
        err = abs(long / max(short, 1e-6) / ratio - 1)
        if err <= CARD_ASPECT_TOL and tr.blob.fill >= CARD_FILL and (best is None or err < best[0]):
            best = (err, tr)
    if best is None:
        return None
    tr = best[1]
    long, short = tr.shown
    return Scale((ID1_MM[0] / long + ID1_MM[1] / short) / 2, "card"), tr


def hand_mm_per_px(pose: HandPose, H: np.ndarray) -> float | None:
    """mm per page px from one hand: MediaPipe's metric palm against the palm's size mapped onto the table. Every
    palm segment is foreshortened by a tilt, which only lengthens its ratio, so the shortest ratio is the best."""
    page = apply_h(H, pose.img)
    best = None
    for a, b in PALM_SEGMENTS:
        d_page = float(np.linalg.norm(page[a] - page[b]))
        d_mm = 1000.0 * float(np.linalg.norm(pose.world[a] - pose.world[b]))
        if d_page < 5.0 or d_mm < 20.0:
            continue
        r = d_mm / d_page
        best = r if best is None else min(best, r)
    return best


class HandScale:
    def __init__(self) -> None:
        self.samples: deque[float] = deque(maxlen=HAND_SCALE_LEN)

    def add(self, poses: list[HandPose], H: np.ndarray) -> None:
        for p in poses:
            r = hand_mm_per_px(p, H)
            if r is not None:
                self.samples.append(r)

    def value(self) -> Scale | None:
        if len(self.samples) < HAND_SCALE_MIN:
            return None
        return Scale(float(np.median(self.samples)), "hands")


# ---------------------------------------------------------------------------------------------------- hands


@dataclass(eq=False)
class Slot:
    k: int
    pose: HandPose | None = None  # latest raw landmarks
    seen: float = -1e9
    tip: tuple[float, float] | None = None  # index fingertip, design px, steadied and cursor-corrected
    votes: deque = field(default_factory=lambda: deque(maxlen=15))
    side: str = ""  # L | R
    shown: np.ndarray | None = None  # landmarks to draw, camera px: steadied (PoseFilter)
    steady: PoseFilter = field(default_factory=PoseFilter)
    jitter: JitterMeter = field(default_factory=JitterMeter)
    pinch: Pinch = field(default_factory=Pinch)  # thumb and index finger together: the pen is down (draw and guess)
    shape: ShapeVote = field(default_factory=ShapeVote)  # rock, paper, scissors or "" (rock paper scissors)

    def clear(self) -> None:
        self.pose, self.tip, self.side, self.shown = None, None, "", None
        self.votes.clear()
        self.steady.reset()
        self.jitter.clear()
        self.pinch.reset()
        self.shape.reset()


class HandSlots:
    """Up to two hands in stable slots across frames (MediaPipe lists hands in any order)."""

    def __init__(self, geo: Geo, n: int = 2):
        self.geo = geo
        self.slots = [Slot(k) for k in range(n)]
        self.offset = (0.0, 0.0)  # fingertip landmark - contact point, from the hand check (design px)

    @property
    def active(self) -> list[Slot]:
        return [s for s in self.slots if s.pose is not None]

    @property
    def poses(self) -> list[HandPose]:
        return [s.pose for s in self.active]

    def tips(self) -> list[tuple[float, float]]:
        return [s.tip for s in self.active if s.tip is not None]

    def update(self, poses: list[HandPose], now: float, t: float | None = None) -> None:
        """``t``: the frame's capture time (the landmark filter's clock; default ``now``)."""
        t = now if t is None else t
        pairs = []
        for s in self.slots:
            if s.pose is None:
                continue
            palm = max(float(np.linalg.norm(s.pose.img[9] - s.pose.img[0])), 10.0)
            for j, p in enumerate(poses):
                d = float(np.linalg.norm(p.img[0] - s.pose.img[0]))
                if d <= 2.5 * palm:
                    pairs.append((d, s.k, j))
        done_s: set[int] = set()
        done_p: set[int] = set()
        for _, k, j in sorted(pairs):
            if k not in done_s and j not in done_p:
                self._set(self.slots[k], poses[j], now, t)
                done_s.add(k)
                done_p.add(j)
        for j, p in enumerate(poses):
            if j in done_p:
                continue
            free = [s for s in self.slots if s.k not in done_s and (s.pose is None or now - s.seen > TIP_HOLD_S)]
            if not free:
                free = [s for s in self.slots if s.k not in done_s]
            if not free:
                break
            s = min(free, key=lambda s: s.seen)
            s.clear()
            self._set(s, p, now, t)
            done_s.add(s.k)
        for s in self.slots:
            if s.k not in done_s and s.pose is not None and now - s.seen > TIP_HOLD_S:
                s.clear()
        self._sides()

    def _set(self, s: Slot, p: HandPose, now: float, t: float) -> None:
        s.pose, s.seen = p, now
        s.votes.append("R" if p.side == "Left" else "L" if p.side == "Right" else "")
        s.shown = s.steady.update(p.img, t)
        s.jitter.add(p.img, s.shown)
        s.pinch.update(p)
        s.shape.update(p)
        tip = self.geo.c2d(s.shown[INDEX_TIP])
        s.tip = (tip[0] - self.offset[0], tip[1] - self.offset[1])

    def jitter(self) -> tuple[int, float, float, float]:
        """Still-hand jitter since the last call, over both slots: (frames, raw RMS, drawn RMS, share redrawn), the
        RMS values in page px (see ``JitterMeter``)."""
        n = raw2 = shown2 = moved = 0.0
        for s in self.slots:
            k, r, d, m = s.jitter.take()
            n, raw2, shown2, moved = n + k, raw2 + k * r * r, shown2 + k * d * d, moved + k * m
        if not n:
            return 0, 0.0, 0.0, 0.0
        g = self.geo
        s = local_scale(g.H, apply_h(g.Hi, [(g.page_w / 2, g.page_h / 2)])[0])  # page px per camera px
        return int(n), s * math.sqrt(raw2 / n), s * math.sqrt(shown2 / n), moved / n

    def _sides(self) -> None:
        act = self.active
        for s in act:
            votes = [v for v in s.votes if v]
            s.side = max(("L", "R"), key=votes.count) if votes else ""
        if len(act) == 2 and act[0].side == act[1].side:  # both voted the same: the user's left hand is on the left
            a, b = sorted(act, key=lambda s: self.geo.c2d(s.pose.img[0])[0])
            a.side, b.side = "L", "R"


# ---------------------------------------------------------------------------------------------------- overlay


class OverlayHistory:
    """Small-px mask of everything we projected into the zone during the last ``OVERLAY_HOLD_S``: the camera sees a
    change that late, and an outline that follows a moving object changes every frame."""

    def __init__(self, geo: Geo, zone: TableZone):
        self.geo, self.zone = geo, zone
        self._key: object = None
        self._masks: deque[tuple[float, np.ndarray | None]] = deque()
        self._union: np.ndarray | None = None
        self._dirty = False
        self._grow = max(2, round(OVERLAY_GROW_PX / zone.down))

    def _render(self, prims: list) -> np.ndarray | None:
        if not prims:
            return None
        g, z = self.geo, self.zone
        canvas = np.zeros((math.ceil(g.page_h * SUPERSAMPLE), math.ceil(g.page_w * SUPERSAMPLE)), np.uint8)
        unit = style.unit_css_px(g.page_w, g.page_h)
        font = style.TEXT_PROJ_PX * unit
        for p in prims:
            if p.kind == "label":  # under it only what is darker than the table counts, so a generous box is cheap
                x, y = apply_h(g.H, [p.at])[0]
                box = np.array([[x - 3, y - 0.65 * font], [x + MASK_EM * font * len(p.text) + 3, y - 0.65 * font],
                                [x + MASK_EM * font * len(p.text) + 3, y + 0.65 * font], [x - 3, y + 0.65 * font]])
                cv2.fillPoly(canvas, [np.round(box * SUPERSAMPLE).astype(np.int32)], 255)
            else:
                draw_prim(canvas, p, g.H, unit)
        x0, y0 = z.roi[:2]
        to_small = np.array([[1 / z.down, 0, -x0 / z.down], [0, 1 / z.down, -y0 / z.down], [0, 0, 1.0]])
        m = to_small @ g.Hi @ np.diag([1 / SUPERSAMPLE, 1 / SUPERSAMPLE, 1.0])
        warped = cv2.warpPerspective(canvas, m, z.small, flags=cv2.INTER_LINEAR, borderValue=0)
        return cv2.dilate(np.where(warped > 0, 255, 0).astype(np.uint8), _disk(self._grow))

    def set(self, prims: list, t: float) -> None:
        key = tuple(p.model_dump_json() for p in prims)
        if key == self._key:
            return
        self._key = key
        self._masks.append((t, self._render(prims)))
        self._dirty = True

    def latest(self) -> np.ndarray | None:
        """Mask of what is projected now (no history): a settled camera view shows just that."""
        return self._masks[-1][1] if self._masks else None

    def at(self, t: float) -> np.ndarray | None:
        while len(self._masks) > 1 and t - self._masks[1][0] >= OVERLAY_HOLD_S:  # [0] is shown until [1] took over
            self._masks.popleft()
            self._dirty = True
        if self._dirty:
            masks = [m for _, m in self._masks if m is not None]
            self._union = None if not masks else masks[0] if len(masks) == 1 else np.bitwise_or.reduce(masks)
            self._dirty = False
        return self._union


# ---------------------------------------------------------------------------------------------------- scans


class MotionWatch:
    """Frame-to-frame change over the table at 1/8 of the camera resolution, outside our own recent light: whether the
    view moves now, and where it changed since :meth:`take`. Scans wait for the view to be still; a new object can
    only appear, and an object only vanish, where something moved."""

    def __init__(self, zone: TableZone):
        self.zone = zone
        w, h = zone.small
        self.size = (max(8, w // MOTION_DOWN), max(8, h // MOTION_DOWN))
        self.inner = cv2.resize(zone.inner, self.size, interpolation=cv2.INTER_AREA) > 127
        self.n_inner = max(1, int(np.count_nonzero(self.inner)))
        self.changed = np.zeros(self.size[::-1], bool)
        self.frac = 0.0
        self._prev: np.ndarray | None = None

    def _small(self, img: np.ndarray) -> np.ndarray:
        x, y, w, h = self.zone.roi
        half = img[y:y + h:2, x:x + w:2]  # every other pixel first: area-averaging the full crop cost 3.6 ms a frame
        return cv2.resize(half, self.size, interpolation=cv2.INTER_AREA).astype(np.int16)

    def update(self, img: np.ndarray, overlay: np.ndarray | None = None) -> bool:
        """``overlay``: zone small-px mask of what we projected lately (its changes are ours, not the table's)."""
        cur = self._small(img)
        prev, self._prev = self._prev, cur
        if prev is None or prev.shape != cur.shape:
            return False
        # largest channel change: skin in warm light can be as bright as the table and differ only in colour
        moving = (np.abs(cur - prev).max(axis=2) > MOTION_DIFF) & self.inner
        if overlay is not None:
            moving &= cv2.resize(overlay, self.size, interpolation=cv2.INTER_AREA) == 0
        self.frac = float(np.count_nonzero(moving)) / self.n_inner
        if self.frac <= MOTION_FRAC:
            return False
        self.changed |= moving
        return True

    def mark(self, mask: np.ndarray | None) -> None:
        """Count a zone small-px mask (where a hand is) as changed."""
        if mask is not None:
            self.changed |= cv2.resize(mask, self.size, interpolation=cv2.INTER_AREA) > 0

    def peek(self) -> np.ndarray:
        """Zone small-px mask (uint8) of where the view changed since the last ``take``, grown."""
        m = cv2.dilate(self.changed.astype(np.uint8) * 255, _disk(MOTION_GROW))
        return cv2.resize(m, self.zone.small, interpolation=cv2.INTER_NEAREST)

    def take(self) -> np.ndarray:
        """``peek``, then start over."""
        m = self.peek()
        self.changed[:] = False
        return m

    def reset(self) -> None:
        self.changed[:] = False
        self._prev = None


@dataclass
class Scan:
    """One clean look at the table: our light in it is switched off, the page's ACK waited for, two frames taken."""

    reason: str  # learn | change
    t0: float
    region: np.ndarray | None = None  # zone small px looked at (what changed); None: the whole table
    target: int | None = None  # scene version with the table (or the region) dark
    clean_after: float | None = None  # frames dequeued from then on show it
    frames: list = field(default_factory=list)


def snap_box(view: np.ndarray, box, scale: float) -> tuple[float, float, float, float] | None:
    """The object a model's box means, found in the picture: the blob differing from the table around it that covers
    most of the box. Design px, or None when the picture does not say (white on white, things touching, something
    reaching out of the search area). 4 Nov: boxes off by 20 px put a ring beside an egg and cost SAM its mask."""
    h, w = view.shape[:2]
    x0, y0, x1, y1 = box
    ax0, ay0 = max(0, int((x0 - SNAP_REACH) * scale)), max(0, int((y0 - SNAP_REACH) * scale))
    ax1, ay1 = min(w, int((x1 + SNAP_REACH) * scale)), min(h, int((y1 + SNAP_REACH) * scale))
    if ax1 - ax0 < 12 or ay1 - ay0 < 12:
        return None
    crop = view[ay0:ay1, ax0:ax1].astype(np.float32)
    ring = np.concatenate([crop[:4].reshape(-1, 3), crop[-4:].reshape(-1, 3), crop[:, :4].reshape(-1, 3),
                           crop[:, -4:].reshape(-1, 3)])
    mask = (np.abs(crop - np.median(ring, axis=0)).max(axis=2) > SNAP_DIFF).astype(np.uint8) * 255
    mask = cv2.morphologyEx(cv2.morphologyEx(mask, cv2.MORPH_CLOSE, _disk(3)), cv2.MORPH_OPEN, _disk(1))
    n, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    bx0, by0, bx1, by1 = x0 * scale - ax0, y0 * scale - ay0, x1 * scale - ax0, y1 * scale - ay0
    best, most = None, 0.0
    for i in range(1, n):
        sx, sy, sw, sh = (int(v) for v in stats[i][:4])
        cover = max(0.0, min(sx + sw, bx1) - max(sx, bx0)) * max(0.0, min(sy + sh, by1) - max(sy, by0))
        if cover > most:
            best, most = (sx, sy, sw, sh), cover
    if best is None:
        return None
    sx, sy, sw, sh = best
    if sx <= 1 or sy <= 1 or sx + sw >= crop.shape[1] - 1 or sy + sh >= crop.shape[0] - 1:
        return None  # it goes on beyond where the object can be: not one thing lying here
    ratio = (sw * sh) / max((bx1 - bx0) * (by1 - by0), 1.0)
    if not 0.3 <= ratio <= 3.5:
        return None
    return (ax0 + sx) / scale, (ay0 + sy) / scale, (ax0 + sx + sw) / scale, (ay0 + sy + sh) / scale


def frames_agree(a: np.ndarray, b: np.ndarray) -> float:
    """Mean grey-level difference of two frames at 1/8 resolution."""
    size = (max(1, a.shape[1] // 8), max(1, a.shape[0] // 8))
    ga = cv2.resize(cv2.cvtColor(a, cv2.COLOR_BGR2GRAY), size, interpolation=cv2.INTER_AREA).astype(np.float32)
    gb = cv2.resize(cv2.cvtColor(b, cv2.COLOR_BGR2GRAY), size, interpolation=cv2.INTER_AREA).astype(np.float32)
    return float(np.mean(np.abs(ga - gb)))


# ---------------------------------------------------------------------------------------------------- dashboard


@dataclass
class TableDashboard:
    """The table layout in design space: a button column at the left end, a status line along the far edge, objects
    anywhere else."""

    geo: Geo
    buttons: list[Button] = field(default_factory=lambda: [
        Button("home", "HOME", (COL_X1 / 2, 120.0), r=BTN_R), Button("learn", "LEARN", (COL_X1 / 2, 192.0), r=BTN_R),
        Button("scale", "SCALE", (COL_X1 / 2, 264.0), r=BTN_R), Button("recal", "RECAL", (COL_X1 / 2, 336.0), r=BTN_R),
    ])
    note: str = ""
    note_role: str = "info"
    note_until: float = 0.0
    task_line: str | None = None  # the bench check's progress, in place of the object count
    task_role: str = "info"
    lines: tuple[str, str, str] | None = None  # (first line, second line, role): a guided task's, or "planning"
    voice: bool = False  # spoken questions are on: the idle hint says what to say
    views: dict = field(default_factory=dict)  # uid -> (role, label lines) from a guided task
    extra: object = None  # callable(canvas, now): a guided task's counter and timer, drawn over the table
    table_buttons: list = field(default_factory=list)  # LEARN/SCALE, put away while a game is on

    def game_mode(self, on: bool) -> None:
        """While a game is on, the column holds one HOME button (the games menu) instead of LEARN and SCALE."""
        if on and not self.table_buttons:
            self.table_buttons = self.buttons
            self.buttons = [Button("home", "HOME", (COL_X1 / 2, 190.0), r=BTN_R),
                            Button("recal", "RECAL", (COL_X1 / 2, 300.0), r=BTN_R)]
        elif not on and self.table_buttons:
            self.buttons, self.table_buttons = self.table_buttons, []

    def game_scene(self, slots: HandSlots, now: float, draw, cursors: bool = True) -> list:
        """The status bar, the HOME button, the game (``draw(canvas, now)``) and, unless the game draws its own
        (``cursors`` False), a ring under each fingertip."""
        hud, game = Canvas(self.geo), Canvas(self.geo)
        self._hud(hud, True, [], slots, now)
        draw(game, now)
        for s in slots.active:
            if cursors and s.tip is not None:
                game.circle(f"h{s.k}-tip", s.tip, 9, "info", n=16)
        return hud.prims + game.prims

    def notify(self, text: str, now: float, role: str = "info") -> None:
        self.note, self.note_role, self.note_until = text, role, now + NOTE_S

    def step(self, tips: list[tuple[float, float]], now: float, dt: float) -> list[str]:
        """Dwell any fingertip on a button to press it; returns the keys pressed."""
        dt = min(max(dt, 0.0), 0.1)
        fired = []
        for b in self.buttons:
            inside = any(math.dist(tip, b.c) <= b.r + 10 for tip in tips)
            if not inside:
                b.dwell, b.armed = 0.0, True
            elif b.armed:
                b.dwell += dt
                if b.dwell >= DWELL.get(b.key, 1.0):
                    b.dwell, b.armed, b.flash_until = 0.0, False, now + 0.5
                    fired.append(b.key)
        return fired

    @property
    def font(self) -> float:
        """Label font size, design px."""
        return style.TEXT_PROJ_PX * style.unit_css_px(self.geo.page_w, self.geo.page_h) / self.geo.s

    def scene(self, zone: TableZone | None, tracks: list[Track], slots: HandSlots, scale: Scale | None,
              now: float, *, blank: bool = False, scan_n: int = 0, keep=None, label_ok=None) -> tuple[list, list]:
        """(all prims, the prims that light the detection area: they are masked out of it). Skeletons count: drawn a
        frame late, a moving hand's skeleton lands on the bare table beside it. ``blank``: a scan is looking at the
        table, so only the HUD (outside the detection area) is drawn, plus, in a local scan, the objects ``keep(tr)``
        allows and their labels where ``label_ok(rect_design)``: away from where the scan looks."""
        hud, objs, hands = Canvas(self.geo), Canvas(self.geo), Canvas(self.geo)
        self._hud(hud, zone, tracks, slots, now, scanning=blank)
        if blank:
            if keep is not None:
                self._objects(objs, tracks, scale, scan_n, now, keep=keep, label_ok=label_ok)
            return hud.prims + objs.prims, objs.prims
        self._objects(objs, tracks, scale, scan_n, now)
        self._hands(hands, slots)
        return hud.prims + objs.prims + hands.prims, objs.prims + hands.prims

    def _hud(self, c: Canvas, zone, tracks, slots, now, scanning: bool = False) -> None:
        c.poly("hud-line", [(COL_X1, STATUS_Y1 + 6), (COL_X1, DESIGN_H - 10)], "info")
        tips = slots.tips()
        hover = None
        for b in self.buttons:
            hot = any(math.dist(tip, b.c) <= b.r + 10 for tip in tips)
            hover = hover or (b.key if hot else None)
            role = "target" if now < b.flash_until else ("warn" if hot and b.armed else "info")
            c.circle(f"btn-{b.key}", b.c, b.r, role, n=24)
            c.arc(f"btn-{b.key}-arc", b.c, b.r + 7, b.dwell / DWELL.get(b.key, 1.0), "target")
            self._icon(c, b, role)
        line1, line2, role = self._status(zone, tracks, slots, now, hover)
        x = COL_X1 + 14
        c.label("status", (x, STATUS_Y), line1[:STATUS_CHARS], role)
        if line2:
            c.label("status-2", (x, STATUS_Y2), line2[:STATUS_CHARS], role if role in ("error", "warn") else "info")

    @staticmethod
    def _icon(c: Canvas, b: Button, role: str) -> None:
        """LEARN: a circular arrow (start again). SCALE: a card (sizes come from a bank card)."""
        x, y, r = b.c[0], b.c[1], b.r
        if b.key == "recal":  # cross hairs: line the light up with the hand again
            c.circle(f"btn-{b.key}-ico", (x, y), 0.42 * r, role, n=16)
            for k, (dx, dy) in enumerate(((1, 0), (-1, 0), (0, 1), (0, -1))):
                c.poly(f"btn-{b.key}-t{k}", [(x + dx * 0.2 * r, y + dy * 0.2 * r), (x + dx * 0.68 * r, y + dy * 0.68 * r)],
                       role)
            return
        if b.key == "home":  # a little house
            w = 0.5 * r
            c.poly(f"btn-{b.key}-ico", [(x - w, y + 0.1 * r), (x, y - 0.55 * r), (x + w, y + 0.1 * r)], role)
            c.poly(f"btn-{b.key}-ico2", [(x - 0.7 * w, y), (x - 0.7 * w, y + 0.5 * r), (x + 0.7 * w, y + 0.5 * r),
                                         (x + 0.7 * w, y)], role)
            return
        if b.key == "learn":
            rr = 0.52 * r
            pts = [(x + rr * math.cos(t), y + rr * math.sin(t)) for t in np.linspace(math.radians(60), math.radians(330), 12)]
            c.poly(f"btn-{b.key}-ico", pts, role)
            ex, ey = pts[-1]
            c.poly(f"btn-{b.key}-ico2", [(ex - 0.34 * r, ey - 0.02 * r), (ex, ey), (ex + 0.02 * r, ey - 0.36 * r)], role)
        else:
            w, h = 0.6 * r, 0.38 * r
            c.rect(f"btn-{b.key}-ico", x - w, y - h, x + w, y + h, role)
            c.poly(f"btn-{b.key}-ico2", [(x - w, y - 0.1 * r), (x + w, y - 0.1 * r)], role)

    def _status(self, zone, tracks, slots, now: float, hover: str | None) -> tuple[str, str, str]:
        """The two lines along the far edge: what to do now, then progress or a hint."""
        if now < self.note_until:
            return self.note, "", self.note_role
        if hover is not None:
            a, b = BUTTON_TIPS[hover]
            return a, b, "warn"
        if zone is True:  # a game: its own lines
            return self.lines or ("", "", "info")
        if zone is None or zone.status in ("learning", "light"):  # never projected onto the table being learned
            return {"learning": ("CLEAR THE TABLE: LEARNING IT EMPTY", "KEEP YOUR HANDS AWAY FOR A MOMENT", "warn"),
                    "light": ("THE LIGHT CHANGED: ADJUSTING", "", "warn")}.get(
                getattr(zone, "status", ""), ("TABLE NOT IN VIEW", "", "warn"))
        if self.lines is not None:
            return self.lines
        if self.task_line:
            return self.task_line, "", self.task_role
        n = sum(tr.confirmed for tr in tracks)
        if n == 0:
            return ("PUT THINGS ON THE TABLE", "THEN SAY WHAT YOU WANT TO DO" if self.voice else "", "info")
        what = f"{n} THING{'' if n == 1 else 'S'} ON THE TABLE"
        if self.voice:
            return what, "POINT AND ASK, OR SAY WHAT TO DO", "info"
        h = len(slots.active)
        return what, f"{h} HAND{'' if h == 1 else 'S'} IN VIEW" if h else "", "info"

    def _role(self, tr: Track, now: float) -> str:
        """White by default; yellow while held; green while a question about it is answered; with a task, green for
        a listed item and red for one that is not needed."""
        v = self.views.get(tr.uid)
        if v is not None:
            return v[0]
        if tr.held:
            return "warn"
        if tr.asked_until > now:
            return "target"
        if self.task_line is not None and tr.item >= 0:
            return "target" if tr.item > 0 else "error"
        return "info"

    def _objects(self, c: Canvas, tracks: list[Track], scale: Scale | None, scan_n: int = 0,
                 now: float = 0.0, keep=None, label_ok=None) -> None:
        """Outlines (white: a saturated colour smears into its neighbours in the camera's compressed video) that draw
        themselves after each scan, and labels laid out where they cover no object and no other label."""
        g = self.geo
        font = self.font
        shown = [tr for tr in tracks if tr.confirmed]
        boxes = {}
        for tr in shown:
            d = np.array([g.p2d(p) for p in tr.outline])
            boxes[tr.uid] = (float(d[:, 0].min()), float(d[:, 1].min()), float(d[:, 0].max()), float(d[:, 1].max()))
        taken = list(boxes.values())
        step = 1.15 * font
        for tr in sorted(shown, key=lambda tr: tr.id):
            fresh = tr.born_scan >= scan_n
            sam = "-s" if tr.refined else ""  # a SAM outline is a new prim: it draws itself in place of the first one
            size = None if scale is None or tr.shown is None else scale.text(tr.shown)
            lines = [tr.name.upper(), size or str(tr.id)] if tr.name else [f"{tr.id}: {size}" if size else str(tr.id)]
            v = self.views.get(tr.uid)
            if v is not None and v[1]:
                lines = [t for t in v[1] if t] or lines
            w = LABEL_EM * font * max(len(t) for t in lines)
            x0, y0, _, _ = self._place(boxes[tr.uid], w, step * len(lines), taken)  # laid out as always: no jumps
            rect = (x0, y0, x0 + w, y0 + step * len(lines))
            taken.append(rect)
            if keep is not None and not keep(tr):
                continue
            c.cam_poly(f"obj-{tr.uid}{sam}", tr.cam_outline(g), self._role(tr, now), glow=True,
                       draw_ms=NEW_DRAW_MS if fresh else OUTLINE_DRAW_MS)
            if label_ok is not None and not label_ok(rect):
                continue
            for k, text in enumerate(lines):
                c.label(f"obj-{tr.uid}-txt{k}", (x0, y0 + (k + 0.5) * step), text, "info",
                        type_ms=TYPE_MS_PER_CHAR * len(text))
        if keep is None and callable(self.extra):
            self.extra(c, now)

    @staticmethod
    def _place(box, w: float, h: float, taken: list) -> tuple[float, float, float, float]:
        """A w x h label block beside ``box`` (design px): above, below, right or left, the first that overlaps no
        object or label already placed, else the one that overlaps least; always on the table part of the page."""
        x0, y0, x1, y1 = box
        gap = 6.0
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        best, best_cost = None, math.inf
        for lx, ly in ((cx - w / 2, y0 - gap - h), (cx - w / 2, y1 + gap), (x1 + gap, cy - h / 2), (x0 - gap - w, cy - h / 2)):
            lx = min(max(lx, COL_X1 + 8.0), DESIGN_W - 10.0 - w)
            ly = min(max(ly, STATUS_Y1 + 6.0), DESIGN_H - 6.0 - h)
            rect = (lx, ly, lx + w, ly + h)
            cost = sum(max(0.0, min(rect[2], r[2]) - max(rect[0], r[0])) * max(0.0, min(rect[3], r[3]) - max(rect[1], r[1]))
                       for r in taken)
            if cost < best_cost:
                best, best_cost = rect, cost
            if cost == 0:
                break
        return best

    def _hands(self, c: Canvas, slots: HandSlots) -> None:
        for s in slots.active:
            role = HAND_ROLES[s.k % len(HAND_ROLES)]
            img = s.shown if s.shown is not None else s.pose.img  # steadied: raw landmarks shake on a still hand
            for i, chain in enumerate(HAND_CHAINS):
                c.cam_poly(f"h{s.k}-bone-{i}", img[list(chain)], role, closed=False)
            if s.tip is not None:
                c.circle(f"h{s.k}-tip", s.tip, 12, role, n=20)
            wrist = np.array(self.geo.c2d(img[0]))
            knuckle = np.array(self.geo.c2d(img[9]))
            away = wrist - knuckle
            n = float(np.linalg.norm(away))
            at = wrist + (away / n * 30 if n > 1e-6 else 0)
            c.label(f"h{s.k}-side", (float(at[0]) - 8, float(at[1])), s.side or str(s.k + 1), role)


def table_text(zone: TableZone | None, tracks: list[Track], scale: Scale | None) -> str:
    """One status line for the terminal and /screen."""
    if zone is None or zone.status == "learning":
        return "learning the empty table"
    if zone.status == "light":
        return "light changed"
    shown = [tr for tr in tracks if tr.confirmed]
    if not shown:
        return "no objects"
    parts = []
    for tr in sorted(shown, key=lambda tr: tr.id):
        size = "" if scale is None or tr.shown is None else " " + scale.text(tr.shown)
        what = (tr.name if tr.colour in tr.name.split() else f"{tr.colour} {tr.name}") if tr.name else tr.colour
        parts.append(f"{tr.id} {what}{size}{' (held)' if tr.held else ''}")
    return "; ".join(parts)


def spoken_new(tr: Track, scale: Scale | None) -> str:
    what = tr.name or f"{tr.colour} object"
    if scale is None or tr.shown is None:
        return f"New {what}."
    a, b = scale.mm(tr.shown)
    if max(a, b) >= 100:
        size = f"{a / 10:.0f} by {b / 10:.0f} centimetres"
    else:
        size = f"{a:.0f} by {b:.0f} millimetres"
    return f"New {what}, {'about ' if scale.approx else ''}{size}."


# ---------------------------------------------------------------------------------------------------- session


class TableSession:
    """The table mode's per-frame work: hands, buttons, scans of the table, tracks, names, the mm scale, the scene.

    Objects are looked for only in scans (3 Nov): the camera saw our own outlines and labels and took them for objects,
    which then got outlines too (2 Nov: 3 objects on the table, 10 reported; 6 -> 20 in five minutes with no hand
    around). A scan starts once the view has been still for ``SCAN_SETTLE_S`` after it changed (``MotionWatch``):
    everything we draw over the table goes dark, the page's ACK of that is waited for (``link``: the render hub), and
    two frames dequeued ``SCAN_AFTER_ACK_S`` later are averaged and compared with the learned empty table. They must
    agree (a moving hand, a broken camera frame). Between scans the outlines stay where the last scan saw the objects;
    a hand on one freezes it (drawn yellow) until the next scan. The empty table is learned the same way, once the
    page shows it dark and nothing dark lies on it (``TableZone.stray_objects``).
    """

    def __init__(self, geo: Geo, frame_size: tuple[int, int], *, threshold: int = 28, min_area: int = 400,
                 page_mm: float | None = None, state_dir: Path | None = None, namer=None, outliner=None, link=None,
                 asker=None, task: BenchTask | None = None, plan=None, brain=None, voice: bool = False,
                 planner_model: str | None = None, game: str | None = None, mic_gate=None, painter=None):
        self.namer = namer  # lightwork.namer: names each shown object once; None = no names
        self.asker = asker  # lightwork.asker: answers spoken questions about the table; None = none
        self.task = task  # lightwork.benchtask: what the job needs on the table; None = just name things
        self._asking: tuple[int, Track | None, str] | None = None  # (key, object asked about, question)
        self._ask_n = 0
        self._tips_seen: tuple[float, list] = (-1e9, [])  # fingertips last seen (design px), for "what is this?"
        self._ready_said = False
        self.outliner = outliner  # lightwork.outliner: SAM outlines; None = the change blobs' own
        self.link = link  # .version() and .acked_at(version) of the page that draws the scene (RenderHub); None = none
        self._announce: list[tuple[Track, float]] = []  # new objects waiting for their name before they are spoken
        self.card_until: float | None = None  # SCALE pressed: a card found before this sets the scale
        self.log: list[str] = []  # lines for the terminal (names as they arrive); the caller prints and clears
        self.sfx: list[str] = []  # sounds for the projector page (render.Sfx names); the caller sends and clears
        self.frame_size = frame_size
        self.threshold, self.min_area = threshold, min_area
        self.page_mm = page_mm
        self.state_dir = state_dir
        self.tracks = ObjectTracks()
        self.hand_scale = HandScale()
        self.card: Scale | None = None
        self.zone: TableZone | None = None
        self.motion: MotionWatch | None = None
        self.geo = geo
        self.dash = TableDashboard(geo)
        if task is not None:
            self.dash.task_line = self._task_line()
            if namer is not None and hasattr(namer, "task"):
                namer.task = task
        self.slots = HandSlots(geo)
        self.overlay: OverlayHistory | None = None
        self.learn_at: float | None = None  # learn the empty table once hands are away, from this time on
        self._learn_said = False
        self._learn_tries = 0
        self._force_learn = False
        self.hands_mask: np.ndarray | None = None
        self.zone_error: str | None = None
        self.last_hand_t = -1e9
        self._wrists: list = []  # (wrist px, palm px) of the hands in the last frame
        self.scan: Scan | None = None
        self._done_region: np.ndarray | None = None
        self.clean: np.ndarray | None = None  # zone small px: the table as the scans last saw it, our light off
        self.skipped = 0  # scans not needed: what moved was a hand passing over
        self.scans = 0  # completed
        self.scan_aborts = 0
        self.last_scan_t = -1e9
        self.scan_image: np.ndarray | None = None  # the last clean view of the table
        self.dirty = True  # the view changed since the last scan (at start: never looked)
        self.anywhere = True  # the next scan may find objects where nothing moved (none looked at this table yet)
        self.last_change_t = -1e9
        self.rescan_at: float | None = None  # look again then, changed or not (a light change, an arm across an edge)
        self._n = 0
        self.guide: Guide | None = None  # lightwork.guide: a job the table walks through (--do, or asked for aloud)
        self.brain = brain  # lightwork.brain: plans, checks; None = built-in plans only, no checks
        self.planner_model = planner_model
        self.jobs = None
        if brain is not None:
            from .brain import Jobs

            self.jobs = Jobs("table-brain")
        self._plan_job: tuple | None = None  # (key, request, since)
        self._sweep_job: tuple | None = None  # (key, view box in camera px, since)
        self.last_sweep_t = -1e9
        self.sweep_gap = SWEEP_GAP_S  # automatic sweeps: doubles after one that found nothing new
        self.sweeps = 0
        self._check_job: tuple | None = None  # (key, step number)
        self._job_n = 0
        self._pending_say: list[str] = []
        self.painter = painter  # lightwork.painter.Painter: pictures for the helper's answers; None: answers go without
        self._paint_job: tuple | None = None  # (key, the tile that asked, its epoch)
        self.mic_gate = mic_gate  # lightwork.voice.capture.MicGate: only the helper tile's mic bubble opens it
        self._partial_used = ("", -1e9)  # a sentence acted on from its last partial, and when: its final is no new sentence
        self.dash.voice = voice and mic_gate is None
        self.games = GameHost(state_dir=state_dir, ask_guess=self._ask_guess if brain is not None else None,
                              voice=voice and mic_gate is None, listens=voice, thinks=brain is not None)
        self._guess_job: tuple | None = None
        self._look: Scan | None = None  # the helper tile's clean look under way
        self._look_ask = None  # the model call it is for
        self._look_tries = 0
        self._look_img: np.ndarray | None = None  # the last one: the camera frame, with our light off ...
        self._look_view: np.ndarray | None = None  # ... the page in it, rectified ...
        self._look_jpeg: bytes | None = None  # ... as sent to the model
        self._look_t = -1e9
        self._boxes_img: np.ndarray | None = None  # the look the model's boxes come from (a check's look is not one)
        self._helper_jobs: list[tuple] = []  # (key, ask, the tile that asked)
        self._outlining: set[int] = set()  # things SAM is outlining for the helper tile
        self.set_geo(geo, None)
        if plan is not None:
            self._pending_say += self.start_guide(plan, time.monotonic())
        self._first_game = game  # started on the first frame, on the frames' clock

    # ---- hand games (lightwork.games): the object scans pause while one is on

    def _smooth(self, on: bool) -> None:
        if self.link is not None and hasattr(self.link, "set_smooth"):
            self.link.set_smooth(on)

    def start_game(self, key: str, now: float) -> list[str]:
        say = self.games.start(key, now)
        self.dash.game_mode(True)
        self._smooth(True)
        self.scan = None
        self.log.append(f"game: {key}")
        return say

    def _leave_games(self, now: float) -> list[str]:
        """Back to the object board: the table is learned first if it never was (a session started in a game), else
        looked at again where anything may have been put down or taken."""
        self.dash.game_mode(False)
        self._smooth(False)
        if self.zone is not None and self.zone.ref is None:
            self.relearn(now)
            return ["Object board. Clear the table so I can learn it, then put things down."]
        self.dirty, self.anywhere, self.last_change_t = True, True, now  # look at the table again: things may
        if self.motion is not None:  # have been put down or taken while we played
            self.motion.reset()
        return ["Object board. Put things on the table."]

    def _ask_guess(self, jpeg: bytes, not_these=()) -> None:
        """Draw and guess: the drawing goes to the brain (Muse Spark) on the worker; ``not_these``: guesses the player
        already said no to."""
        if self.brain is None or self.jobs is None:
            return
        brain = self.brain
        prompt = GUESS_PROMPT + (f" The player already said it is NOT: {', '.join(not_these)}. Guess something else."
                                 if not_these else "")
        self._job_n += 1
        key = ("guess", self._job_n)
        if self.jobs.submit(key, lambda: brain.ask("guess", prompt, [jpeg], schema=GUESS_SCHEMA,
                                                   effort="minimal", max_out=1500, timeout_s=30.0)):
            self._guess_job = (key, time.monotonic())

    def _guess_done(self, now: float) -> None:
        if self._guess_job is None or self.jobs is None:
            return
        key, t0 = self._guess_job
        done, reply, err = self.jobs.result(key)
        if not done:
            return
        self._guess_job = None
        data = reply.data if reply is not None and isinstance(reply.data, dict) else {}
        game = self.games.game
        if isinstance(game, DrawGuess):
            game.answer(str(data.get("guess") or "").strip() or None, str(data.get("say") or "").strip() or None, now)
        if err:
            self.log.append(f"guess failed: {err}")
        else:
            self.log.append(f"guess: {data} ({reply.model} via {reply.provider}, {reply.ms / 1000:.1f} s)")
            self.dash.notify(f"GUESSED BY {model_tag(reply.model, reply.provider)} · {reply.ms / 1000:.0f} S", now,
                             "done")

    def _game_slots(self) -> list:
        """The hands that play: fingertip on the page or just off it."""
        pad = ON_PAGE_PAD
        return [s for s in self.slots.active if s.tip is not None
                and -pad <= s.tip[0] <= DESIGN_W + pad and -pad <= s.tip[1] <= DESIGN_H + pad]

    def _game_points(self, now: float, slots: list) -> tuple[list, list]:
        """Hand points for a game (raw landmarks: the steadied ones trail a fast hand) and, for hands seen in this and
        a recent frame, the segments they moved along (wrist, finger tips, palm centre; design px)."""
        idx = (0, 4, 8, 12, 16, 20, 9)
        points, segments = [], []
        prev = getattr(self, "_game_prev", {})
        fresh = {}
        ox, oy = self.slots.offset  # the hand check's fingertip correction, as for the cursor
        for s in slots:
            pts = [(x - ox, y - oy) for x, y in (self.geo.c2d(s.pose.img[i]) for i in idx)]
            points += pts
            if s.seen == now:
                before = prev.get(s.k)
                if before is not None and now - before[0] <= 0.35:
                    segments += list(zip(before[1], pts, strict=True))
                fresh[s.k] = (now, pts)
        self._game_prev = {**{k: v for k, v in prev.items() if now - v[0] <= 0.35}, **fresh}
        return points, segments

    def _game_step(self, now: float, dt: float, frame=None) -> tuple[list[str], list[str]]:
        slots = self._game_slots()
        tips = [s.tip for s in slots]
        fired = self.dash.step(tips, now, dt)
        say: list[str] = []
        if self._pending_say:
            say, self._pending_say = self._pending_say, []
        if "home" in fired:
            self.sfx.append("press")
            say += self.games.start("menu", now)
        points, segments = self._game_points(now, slots)
        pens = []
        held = getattr(self, "_pen_at", {})
        for s in slots:  # while fingers that were together read apart (letting go, or a blip) the pen holds still:
            if s.pinch.apart and s.k in held:  # no little hook at the end of every line
                pens.append(Pen(held[s.k], True, s.k, s.shape.name))
                continue
            held[s.k] = s.tip
            pens.append(Pen(s.tip, s.pinch.down, s.k, s.shape.name))
        self._pen_at = held
        if isinstance(self.games.game, DrawGuess):
            self._pinch_log(now, slots)
        lines, sounds = self.games.update(now, points, tips, segments, pens)
        self.sfx += sounds
        if isinstance(self.games.game, RockPaperScissors):
            self._shape_log(slots)
            self.log += self.games.game.log
            self.games.game.log = []
        if not self.games.active:  # OBJECTS picked on the menu: the object board
            lines = [line for line in lines if not line.startswith("Object board")] + self._leave_games(now)
        self._guess_done(now)
        if isinstance(self.games.game, Helper):
            lines += self._helper_step(frame, now)
        game = self.games.game
        if game is not None and game.say:
            lines += game.say
            game.say = []
        return fired, say + lines + self._answers(now)  # a question asked during a game is answered during it

    def _pinch_log(self, now: float, slots: list) -> None:
        """Draw and guess: each pen down and up in the log, with the thumb-to-finger distance (palm lengths), and
        the readings of the last second, so the pinch thresholds can be set from what this camera sees."""
        seen = getattr(self, "_pinch_seen", None)
        if seen is None:
            seen = self._pinch_seen = {"down": {}, "t": now, "r": [], "csv": None}
            if self.state_dir is not None:  # every reading, for setting the thresholds from this camera's pinches
                try:
                    path = self.state_dir / "play-debug" / f"pinch-{time.strftime('%Y%m%dT%H%M%S')}.csv"
                    path.parent.mkdir(parents=True, exist_ok=True)
                    seen["csv"] = path.open("a", encoding="utf-8", buffering=1)
                    seen["csv"].write("t,hand,side,ratio_3d,ratio_img,down,tip_x,tip_y,score\n")
                except OSError:
                    seen["csv"] = None
        for s in slots:
            if s.seen == now:
                seen["r"].append(s.pinch.ratio)
                if seen["csv"] is not None:
                    tip = s.tip or (math.nan, math.nan)
                    seen["csv"].write(f"{now:.3f},{s.k},{s.side},{pinch_ratio_3d(s.pose):.3f},{s.pinch.ratio:.3f},"
                                      f"{int(s.pinch.down)},{tip[0]:.1f},{tip[1]:.1f},{s.pose.score:.2f}\n")
            if seen["down"].get(s.k, False) != s.pinch.down:
                seen["down"][s.k] = s.pinch.down
                self.log.append(f"pen {'down' if s.pinch.down else 'up'}: hand {s.side or '?'}, thumb to finger "
                                f"{s.pinch.ratio:.2f} palm")
        if now - seen["t"] >= 1.0:
            if seen["r"]:
                r = sorted(seen["r"])
                self.log.append(f"pinch readings: {len(r)} frames, {r[0]:.2f} min, {r[len(r) // 2]:.2f} median, "
                                f"{r[-1]:.2f} max (down below {self.slots.slots[0].pinch.down_at:.2f}, "
                                f"up above {self.slots.slots[0].pinch.up_at:.2f})")
            seen["t"], seen["r"] = now, []

    def _shape_log(self, slots: list) -> None:
        """Rock paper scissors: each hand's shape in the log as it changes, with what its fingers read (bend at the
        middle joint in degrees / reach, index to little finger), so the thresholds can be set from this camera."""
        seen = getattr(self, "_shape_seen", None)
        if seen is None:
            seen = self._shape_seen = {}
        for s in slots:
            if seen.get(s.k) != s.shape.name:
                seen[s.k] = s.shape.name
                fingers = " ".join(f"{b:.0f}/{r:.2f}" for b, r in finger_bends(s.pose))
                self.log.append(f"hand {s.side or '?'}: {s.shape.name or 'no shape'} (fingers {fingers})")

    # ---- the general helper tile (lightwork.helper): its clean looks and model calls

    def _helper_hears(self, g: Helper, q: str, now: float) -> list[str]:
        """A sentence said at the helper tile. A game (or the menu) starts at once, a command acts at once; anything
        else is collected until the person has finished talking, and goes to the model."""
        tile = asked_for_game(q)
        if tile is not None:
            self.log.append(f"helper: {q!r} starts {tile}")
            self.sfx.append("press")
            self._look, self._look_ask = None, None
            say = self.games.start(tile, now)
            if not self.games.active:  # the object board
                say = [line for line in say if not line.startswith("Object board")] + self._leave_games(now)
            return say
        verb = command_of(q.lower())
        if verb in (None, "check") and g.model_command(q):
            g.dealt_with()
            self.log.append(f"helper: {q!r} acts on the model")
            say, g.say = g.say, []
            return say
        thing = asked_to_show(q) if verb is None else None
        if thing is not None and g.thinks:  # nothing on the table is needed for that: no look
            g.dealt_with()
            g.show_model(thing, now)
            self.log.append(f"helper: {q!r} asks for a model of {thing!r}")
            return []
        place = asked_to_go(q) if verb is None else None
        if place is not None and g.thinks:
            g.dealt_with()
            g.go_to(place, now)
            self.log.append(f"helper: {q!r} asks for the map of {place!r}")
            return []
        if verb is None:
            g.heard(q, now)
            return []
        g.dealt_with()  # a bare "next" is never a job, whatever the state
        if g.command(verb, now):
            self.sfx.append("press")
        say, g.say = g.say, []
        return say

    def _helper_step(self, frame, now: float) -> list[str]:
        """One frame of the tile: lines to say. Takes the clean look it wants, makes its model calls on the worker, and
        hands back what comes in."""
        g = self.games.game
        if g.start_game is not None:  # the model read a game out of what was said
            key, g.start_game = g.start_game, None
            self.sfx.append("press")
            self._look, self._look_ask = None, None
            self.log.append(f"helper: the model says it's {key}")
            return self.games.start(key, now)
        if g.hush:  # the person is about to talk, or moved on: stop the phone (the mic stays muted while it speaks)
            g.hush = False
            if self.link is not None and hasattr(self.link, "say"):
                self.link.say("", "interrupt")
        self._helper_outlines(g, now)
        self._helper_replies(g, now)
        self._helper_picture(g, now)
        if self._look is not None:
            img = self._look_step(frame, now)
            if img is not None:
                self._look_img, self._look_t = img, now
                self._look_jpeg = self._page_view(img)
                ask, self._look_ask = self._look_ask, None
                if ask.epoch == g.epoch:  # still wanted
                    self._helper_look_done(g, ask, now)
        elif g.want is not None:
            ask = g.want
            recent = now - self._look_t < LOOK_REUSE_S and self.last_hand_t < self._look_t
            if not ask.look or (recent and ask.role != "check"):  # a check judges what is there now, whatever it costs
                g.want = None
                self._helper_look_done(g, ask, now)
            elif not self._game_slots() or now - g.want_t >= LOOK_HANDS_WAIT_S:
                g.want = None
                self._look, self._look_ask, self._look_tries = Scan("helper", now), ask, 0
                self.sfx.append("scan")
        self.log += g.log
        g.log = []
        return []

    def _keep_model(self, ask, data) -> None:
        file = self._model_file(ask)
        if file is None:
            return
        try:
            file.parent.mkdir(parents=True, exist_ok=True)
            file.write_text(json.dumps(data))
        except OSError as exc:
            self.log.append(f"helper: could not keep the model: {exc}")

    def _helper_picture(self, g: Helper, now: float) -> None:
        """Muse Image draws what the answer asked for, on the worker: the frame loop never waits for it."""
        if g.want_picture is not None:
            prompt, g.want_picture = g.want_picture, None
            painter = self.painter
            self._job_n += 1
            key = ("picture", self._job_n)
            if painter is None or self.jobs is None or not self.jobs.submit(key, lambda: painter.paint(prompt)):
                g.pictured(None, now)
            else:
                self._paint_job = (key, g, g.epoch)
                self.log.append(f"helper: picture asked for: {prompt!r}")
        if self._paint_job is None:
            return
        key, owner, epoch = self._paint_job
        done, painted, err = self.jobs.result(key)
        if not done:
            return
        self._paint_job = None
        if err is not None:
            self.log.append(f"helper: no picture: {err}")
        else:
            self.log.append(f"helper: picture {painted.src} ({painted.model}, {painted.ms / 1000:.1f} s)")
            self.dash.notify(f"PICTURE BY {model_tag(painted.model, painted.provider)} · {painted.ms / 1000:.0f} S", now, "done")
        if owner is g and epoch == g.epoch:  # they may have asked something else by now
            g.pictured(painted, now)

    def _helper_look_done(self, g: Helper, ask, now: float) -> None:
        """The picture for ``ask`` is there: to the model, unless it only asks where things are and they did not move
        since the model last said so."""
        if ask.role == "locate" and g.parts_known and self._unmoved(self._boxes_img, self._look_img, g.lit):
            g.answered(ask, {"found": []}, None, now, by="(nothing moved: no model call)")
            return
        if ask.role in ("see", "locate"):
            self._boxes_img = self._look_img
        self._helper_submit(g, ask, now)

    def _look_step(self, frame, now: float) -> np.ndarray | None:
        """Like a scan: everything we draw has gone dark (see ``scene``), the page's ACK is waited for, then a few frames
        are averaged. Without an ACK it looks anyway: the model is told to ignore our light, and a helper that stops
        over a slow page is worse than a picture with a line of light in it."""
        s = self._look
        if s.clean_after is None:
            if self.link is None:
                s.clean_after = s.t0 + SCAN_NO_LINK_S
            else:
                if s.target is None:  # the dark scene went out after the previous step
                    s.target = self.link.version()
                acked = self.link.acked_at(s.target)
                if acked is not None:
                    s.clean_after = acked + LOOK_AFTER_ACK_S
                elif now - s.t0 > SCAN_ACK_TIMEOUT_S:
                    s.clean_after = now
                    self.log.append("helper: the page did not show the dark table in time; looking anyway")
                else:
                    return None
        if frame.t_mono < s.clean_after:
            return None
        s.frames.append(frame.image)
        if len(s.frames) < LOOK_FRAMES:
            return None
        if frames_agree(s.frames[0], s.frames[-1]) > SCAN_AGREE and self._look_tries < LOOK_TRIES:
            self._look_tries += 1
            s.frames = []
            return None
        self._look = None
        return np.mean(np.stack(s.frames), axis=0).astype(np.uint8)

    def _unmoved(self, before: np.ndarray | None, img: np.ndarray, things: list) -> bool:
        """Between two clean looks, nothing changed where these things lie: they stay where the model last saw them
        (its boxes vary by a few per cent of the view from one call to the next, 4 Nov: a bowl that never moved jumped by
        25 px between two calls)."""
        if before is None or before.shape != img.shape or not things:
            return False
        diff = cv2.absdiff(cv2.cvtColor(before, cv2.COLOR_BGR2GRAY), cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
        fw, fh = self.frame_size
        for t in things:
            x0, y0, x1, y1 = t.box
            cam = self.geo.d2c([(x0, y0), (x1, y0), (x1, y1), (x0, y1)])
            (cx0, cy0), (cx1, cy1) = np.floor(cam.min(axis=0)).astype(int), np.ceil(cam.max(axis=0)).astype(int)
            patch = diff[max(0, cy0):min(fh, cy1), max(0, cx0):min(fw, cx1)]
            if patch.size == 0 or float(patch.mean()) > LOOK_MOVED:
                return False
        return True

    def _page_view(self, img: np.ndarray) -> bytes | None:
        """The projected page in the picture, rectified: upright for the person and in design coordinates, so a box on
        the model's 0-1000 grid is a design-px box; only the table, never the room. The table made white again: a warm
        room light turned every object orange."""
        corners = np.float32(self.geo.d2c([(0, 0), (DESIGN_W, 0), (DESIGN_W, DESIGN_H), (0, DESIGN_H)]))
        w, h = int(DESIGN_W * LOOK_SCALE), int(DESIGN_H * LOOK_SCALE)
        m = cv2.getPerspectiveTransform(corners, np.float32([(0, 0), (w, 0), (w, h), (0, h)]))
        view = cv2.warpPerspective(img, m, (w, h), flags=cv2.INTER_LINEAR).astype(np.float32)
        white = np.median(view.reshape(-1, 3)[::97], axis=0)
        view = view * np.clip(225.0 / np.maximum(white, 1.0), 0.7, 2.5)
        self._look_view = np.clip(view, 0, 255).astype(np.uint8)
        ok, buf = cv2.imencode(".jpg", self._look_view, [cv2.IMWRITE_JPEG_QUALITY, 88])
        return buf.tobytes() if ok else None

    def _model_file(self, ask) -> Path | None:
        """Where a built model or a found place is kept: a retake costs neither the wait nor the money."""
        slug = re.sub(r"[^a-z0-9]+", "-", ask.subject.lower()).strip("-")
        name = f"{slug}.json" if ask.role == "place" else f"solid-{slug}.json"
        return None if self.state_dir is None or not slug else Path(self.state_dir) / "play-models" / name

    def _still_served(self, data: dict) -> bool:
        """A kept model's files are still where the page loads them from."""
        src = data.get("src")
        return isinstance(src, str) and (Path(self.state_dir) / "pictures" / src.rsplit("/", 1)[-1]).is_file()

    def _helper_submit(self, g: Helper, ask, now: float) -> None:
        g.want = None
        if ask.role in ("model", "place") and (file := self._model_file(ask)) is not None and file.is_file():
            try:
                data = json.loads(file.read_text())
                if ask.role != "model" or self._still_served(data):
                    g.answered(ask, data, None, now, by="(built before: no model call)")
                    return
            except (OSError, ValueError) as exc:
                self.log.append(f"helper: cached model unreadable: {exc}")
        if self.brain is None or self.jobs is None or (ask.role == "model" and self.state_dir is None):
            g.answered(ask, None, "no model", now)
            return
        brain = self.brain
        jpegs = [self._look_jpeg] if self._look_jpeg is not None and ask.role not in ("model", "place") else []
        self._job_n += 1
        key = ("helper", self._job_n)
        if ask.role == "model":
            folder = Path(self.state_dir) / "pictures"
            job = lambda: build_model(brain, _with_article(ask.subject), folder)  # noqa: E731 - Muse Wonder writes a program, which then runs
        else:
            job = lambda: brain.ask(f"helper-{ask.role}", ask.prompt, jpegs, schema=ask.schema,  # noqa: E731
                                    effort=ask.effort, max_out=ask.max_out, timeout_s=ask.timeout_s)
        if self.jobs.submit(key, job):
            self._helper_jobs.append((key, ask, g))
        else:
            g.answered(ask, None, "the worker is busy", now)

    def _helper_replies(self, g: Helper, now: float) -> None:
        for job in list(self._helper_jobs):
            key, ask, owner = job
            done, reply, err = self.jobs.result(key)
            if not done:
                continue
            self._helper_jobs.remove(job)
            if owner is not g:
                continue
            by = "" if reply is None else f"({reply.model} via {reply.provider}, {reply.ms / 1000:.1f} s)"
            g.answered(ask, None if reply is None else reply.data, err, now, by=by)
            # also one that came too late to show: it is paid for. But never a place that is not on the map: it would stay
            if err is None and (ask.role == "model" or (ask.role == "place" and parse_place(reply.data) is not None)):
                self._keep_model(ask, reply.data)
            if ask.role in ("see", "locate") and self._look_view is not None:  # before SAM gets the boxes
                for t in g.want_outlines:
                    snapped = snap_box(self._look_view, t.box, LOOK_SCALE)
                    if snapped is not None:
                        g.place(t, snapped)
            if err is None and reply is not None:
                self.dash.notify(f"{LOOK_LABELS[ask.role]} BY {model_tag(reply.model, reply.provider)} · "
                                 f"{reply.ms / 1000:.0f} S", now, "done")

    def _helper_outlines(self, g: Helper, now: float) -> None:
        """SAM 3.1 outlines of the things the model found, asked for with its boxes barely grown. With too much table
        in the box SAM outlines the table: 4 Nov, true boxes of a composed Lego table grown by 2 / 8 / 18 design px gave
        a 2x4 brick IoU 0.87 / 0.64 / 0.06, and the more the table the worse, whatever the object's size."""
        if self.outliner is None or self._look_img is None:
            g.want_outlines = []
            return
        if g.want_outlines:
            objects = []
            for t in g.want_outlines:
                x0, y0, x1, y1 = t.box
                pad = min(0.3 * max(x1 - x0, y1 - y0), OUTLINE_GROW)
                cam = self.geo.d2c([(x0 - pad, y0 - pad), (x1 + pad, y0 - pad), (x1 + pad, y1 + pad), (x0 - pad, y1 + pad)])
                (cx0, cy0), (cx1, cy1) = np.floor(cam.min(axis=0)), np.ceil(cam.max(axis=0))
                fw, fh = self.frame_size
                objects.append((t.uid, (int(max(0, cx0)), int(max(0, cy0)), int(min(fw, cx1 + 1)), int(min(fh, cy1 + 1)))))
            if self.outliner.request(self._look_img, objects):
                self._outlining |= {uid for uid, _ in objects}
                for t in g.want_outlines:
                    t.asked = True
                g.want_outlines = []
        for uid in list(self._outlining):
            done, contour = self.outliner.result(uid)
            if not done:
                continue
            self._outlining.discard(uid)
            if contour is None:
                continue
            c32 = np.asarray(contour, np.float32)
            perimeter = cv2.arcLength(c32, True)
            if perimeter <= 0 or 4 * math.pi * cv2.contourArea(c32) / perimeter ** 2 < OUTLINE_MIN_ROUND:
                self.log.append(f"helper: SAM's outline of {uid} is frayed: the table, not the thing; kept the box")
                continue
            page = offset_outline(apply_h(self.geo.H, contour), PAD_PAGE)
            g.outlined(uid, [self.geo.p2d(p) for p in page])

    @property
    def _bench(self) -> BenchTask | None:
        """What objects are matched against: the bench check's list, or the guided job's items."""
        if self.task is not None:
            return self.task
        return self.guide.plan.bench if self.guide is not None else None

    # ---- guided jobs (lightwork.guide)

    def start_guide(self, plan, now: float) -> list[str]:
        """Walk ``plan`` from now on; objects already on the table are matched against its items again."""
        self.guide = Guide(plan, now)
        self.task = None
        self.dash.task_line = None
        bench = plan.bench
        rename = self.namer is not None and hasattr(self.namer, "task")
        if rename:
            self.namer.task = bench
        for tr in self.tracks.tracks:  # by words at once; the namer, told the job, answers properly in a moment
            tr.item = match_item(tr.name, bench.items) if tr.name else -1
            if rename and tr.name_state == "done":
                tr.name_state, tr.name_t = "none", -1e9
        self.log.append(f"job: {plan.title} ({plan.source}): {len(plan.items)} items, {len(plan.steps)} steps")
        return self.guide.start(now)

    def request_plan(self, text: str, now: float) -> list[str]:
        """A spoken request ("help me make pancakes for four"): a plan from the brain, or a built-in one."""
        if self.brain is None or self.jobs is None:
            found = find_builtin(text)
            if found is None:
                from .guide import builtin_plans

                return ["Without a model I only know these jobs: " + spoken_list(
                    [n.replace("-", " ") for n in builtin_plans()]) + "."]
            return self.start_guide(scale_plan(*found), now)
        if self._plan_job is not None:
            return ["One moment, I'm still planning the last one."]
        seen = [(tr.name or f"{tr.colour} object", self.scale.mm(tr.shown) if self.scale and tr.shown else None)
                for tr in sorted(self.tracks.confirmed, key=lambda tr: tr.id)]
        prompt = PLAN_PROMPT.format(request=text.strip(), context=plan_request_context(seen))
        jpegs = []
        if self.scan_image is not None:
            view = self._table_jpeg(self.scan_image)
            jpegs = [view] if view is not None else []
        brain, model = self.brain, self.planner_model
        self._job_n += 1
        key = ("plan", self._job_n)
        if not self.jobs.submit(key, lambda: brain.ask("plan", prompt, jpegs, schema=PLAN_SCHEMA, effort="minimal",
                                                       max_out=6000, timeout_s=120.0, model=model)):
            return ["Sorry, I'm busy. Ask me again in a moment."]
        self._plan_job = (key, text.strip(), now)
        self.sfx.append("listen")
        return ["Okay. Let me plan that."]

    def _plan_done(self, now: float) -> list[str]:
        if self._plan_job is None or self.jobs is None:
            return []
        key, text, t0 = self._plan_job
        done, reply, err = self.jobs.result(key)
        if not done:
            return []
        self._plan_job = None
        plan = None
        if err is None and reply is not None and reply.data is not None:
            try:
                plan = parse_plan(reply.data, source=f"{reply.model} via {reply.provider}")
            except ValueError as exc:
                err = f"the plan did not hold together: {exc}"
        if plan is None:
            self.log.append(f"plan for {text!r} failed: {err}")
            found = find_builtin(text)
            if found is not None:
                return ["The planner failed; using my saved plan."] + self.start_guide(scale_plan(*found), now)
            self.sfx.append("error")
            return ["Sorry, I couldn't plan that. Try saying it another way."]
        self.log.append(f"planned {plan.title!r} in {now - t0:.1f} s: {reply.model} via {reply.provider}")
        self.dash.notify(f"PLAN BY {model_tag(reply.model, reply.provider)} · {now - t0:.0f} S", now, "done")
        return self.start_guide(plan, now)

    def _guide_step(self, now: float) -> list[str]:
        g = self.guide
        say = g.update(now, self.seen(), self.hand_points())
        self.sfx += g.sfx
        g.sfx.clear()
        self.log += g.log
        g.log.clear()
        if g.want_check is not None and self._check_job is None:
            k, target = g.want_check
            g.want_check = None
            if self.brain is None or self.jobs is None:
                return say + g.check_result(k, True, "No model to look with: I'll take your word for it.", now)
            step = g.plan.steps[k]
            by_uid = {tr.uid: tr for tr in self.tracks.confirmed}
            jpegs: list[bytes] = []
            if self.scan_image is not None:
                for o in g.resolve(target)[:2]:
                    tr = by_uid.get(o.uid)
                    crop = self.crop(self.scan_image, tr) if tr is not None else None
                    jpegs += [crop] if crop is not None else []
                view = self._table_jpeg(self.scan_image)
                jpegs += [view] if view is not None else []
            prompt = CHECK_PROMPT.format(title=g.plan.title, step=step.say, question=step.question or step.say)
            brain = self.brain
            self._job_n += 1
            key = ("check", self._job_n)
            if self.jobs.submit(key, lambda: brain.ask("check", prompt, jpegs, schema=CHECK_SCHEMA, effort="low",
                                                       max_out=3000, timeout_s=60.0)):
                self._check_job = (key, k)
        if self._check_job is not None:
            key, k = self._check_job
            done, reply, err = self.jobs.result(key)
            if done:
                self._check_job = None
                data = reply.data if reply is not None and isinstance(reply.data, dict) else {}
                if err:
                    self.log.append(f"check failed: {err}")
                    say += g.check_result(k, False, "I couldn't look just now.", now)
                else:
                    self.log.append(f"check: {data} ({reply.model} via {reply.provider}, {reply.ms / 1000:.1f} s)")
                    self.dash.notify(f"CHECKED BY {model_tag(reply.model, reply.provider)} · {reply.ms / 1000:.0f} S",
                                     now, "done" if data.get("ok") is True else "warn")
                    say += g.check_result(k, data.get("ok") is True, str(data.get("say") or ""), now)
        return say

    def seen(self) -> list[Seen]:
        """The shown objects as the guide sees them (design px)."""
        sc = self.scale
        out = []
        for tr in self.tracks.confirmed:
            poly = np.array([self.geo.p2d(pt) for pt in tr.outline], dtype=np.float64)
            out.append(Seen(uid=tr.uid, item=tr.item, name=tr.name, poly=poly, held=tr.held,
                            sides_mm=sc.mm(tr.shown) if sc is not None and tr.shown is not None else None))
        return out

    def hand_points(self) -> list[list[tuple[float, float]]]:
        """Per hand: wrist, thumb tip, index tip, middle tip and middle knuckle, design px."""
        out = []
        for s in self.slots.active:
            img = s.shown if s.shown is not None else s.pose.img
            out.append([self.geo.c2d(img[i]) for i in (0, 4, 8, 12, 9)])
        return out

    def _guide_news(self, due: list[Track]) -> list[str]:
        """While the job's things are being put out: each one ticked off, and what is still missing."""
        g = self.guide
        if g.phase != "gather":
            return []
        ticked = [g.plan.items[tr.item - 1].name for tr in due if tr.item > 0]
        if not ticked:
            return []
        missing = g.missing()
        line = f"{spoken_list(ticked).capitalize()}, check."
        if missing:
            line += f" Still {'need' if len(missing) > 1 else 'needed'}: {spoken_list(missing)}."
        return [line]

    # ---- Muse Spark looks over the whole table (objects too pale for the scans)

    def request_sweep(self, now: float, *, why: str = "asked") -> bool:
        """Send the clean view of the table to the brain for every object with a box; returns False when it cannot
        (no brain, no clean view yet, one running, or the last one was too recent)."""
        if self.brain is None or self.jobs is None or self.scan_image is None or self.zone is None:
            return False
        if self._sweep_job is not None or (why != "asked" and now - self.last_sweep_t < self.sweep_gap):
            return False
        x, y, w, h = self.zone.roi
        jpeg = self._table_jpeg(self.scan_image)
        if jpeg is None:
            return False
        brain = self.brain
        self._job_n += 1
        key = ("sweep", self._job_n)
        if not self.jobs.submit(key, lambda: brain.ask("sweep", SWEEP_PROMPT, [jpeg], schema=SWEEP_SCHEMA,
                                                       effort="minimal", max_out=3000, timeout_s=45.0)):
            return False
        self._sweep_job = (key, (x, y, w, h), now)
        self.last_sweep_t = now
        self.log.append(f"sweep ({why}): asking {getattr(brain, 'label', 'the brain')} for every object")
        return True

    def _sweep_done(self, now: float) -> list[str]:
        if self._sweep_job is None or self.jobs is None:
            return []
        key, (x, y, w, h), t0 = self._sweep_job
        done, reply, err = self.jobs.result(key)
        if not done:
            return []
        self._sweep_job = None
        self.sweeps += 1
        objs = reply.data.get("objects") if reply is not None and isinstance(reply.data, dict) else None
        if err or not isinstance(objs, list):
            self.log.append(f"sweep failed: {err or 'no objects list'}")
            return []
        added = []
        for o in objs[:24]:
            if not isinstance(o, dict):
                continue
            try:
                bx = [float(o[k]) for k in ("x_min", "y_min", "x_max", "y_max")]
            except (KeyError, TypeError, ValueError):
                continue
            x0, x1 = sorted((min(max(bx[0], 0), 1000), min(max(bx[2], 0), 1000)))
            y0, y1 = sorted((min(max(bx[1], 0), 1000), min(max(bx[3], 0), 1000)))
            cam = np.array([(x + x0 * w / 1000, y + y0 * h / 1000), (x + x1 * w / 1000, y + y0 * h / 1000),
                            (x + x1 * w / 1000, y + y1 * h / 1000), (x + x0 * w / 1000, y + y1 * h / 1000)])
            tr = self._sweep_track(cam, str(o.get("name") or ""), now)
            if tr is not None:
                added.append(tr)
        found = len(objs)
        self.log.append(f"sweep: {found} objects seen by {reply.model} via {reply.provider} in {reply.ms / 1000:.1f} s;"
                        f" {len(added)} new: {', '.join(tr.name or '?' for tr in added) or 'none'}")
        self.sweep_gap = SWEEP_GAP_S if added else min(self.sweep_gap * 2, 120.0)
        if not added:
            return []
        self.zone.set_objects([tr.cam_outline(self.geo) for tr in self.tracks.confirmed])
        self.sfx.append("found")
        self.dash.notify(f"{len(added)} MORE FOUND BY {model_tag(reply.model, reply.provider)}", now, "done")
        if self.scan_image is not None:
            self._ask(self.scan_image, now)  # SAM outlines (and names with the job's list) for them
        if self.guide is not None:
            return self._guide_news(added)
        names = [tr.name for tr in added if tr.name]
        return [f"I also see {spoken_list(['a ' + n for n in names])}."] if names else []

    def _sweep_track(self, cam: np.ndarray, name: str, now: float) -> Track | None:
        """A track for a box the model found, unless a shown object is already there or it is not one object."""
        z = self.zone
        page = apply_h(self.geo.H, cam)
        rect = cv2.minAreaRect(page.astype(np.float32))
        (cx, cy), (rw, rh), _ = rect
        if min(rw, rh) < NAME_MIN_PAGE * 0.75:
            return None
        poly = z.poly_mask(cam)
        area = int(np.count_nonzero(poly))
        if area == 0 or np.count_nonzero(poly & z.inner) < 0.6 * area or area > SWEEP_MAX_SHARE * z.inner_px:
            return None  # off the table, under the buttons or status bar, or the whole table
        for tr in self.tracks.confirmed:
            other = z.poly_mask(tr.cam_outline(self.geo))
            inter = int(np.count_nonzero(poly & other))
            if inter > 0.3 * min(area, max(1, int(np.count_nonzero(other)))):
                return None  # already outlined
        x0, y0 = np.floor(cam.min(axis=0)).astype(int)
        x1, y1 = np.ceil(cam.max(axis=0)).astype(int)
        mx, my = (x1 - x0) // 4, (y1 - y0) // 4
        patch = self.scan_image[max(0, y0 + my):max(0, y1 - my), max(0, x0 + mx):max(0, x1 - mx)]
        bgr = np.median(patch.reshape(-1, 3), axis=0) if patch.size else z.table_bgr
        b = Blob(rect=rect, cam_poly=cam.astype(np.float64), area=float((x1 - x0) * (y1 - y0)), fill=0.8,
                 bgr=np.asarray(bgr, np.float64), colour=rel_colour_name(bgr, z.table_bgr * z.gain), near_hand=False,
                 contour=page.astype(np.float64))
        tr = Track(blob=b, rect=rect, born=now, seen=now)
        tr.observe(b, force=True)
        tr.id = self.tracks._free_id_with([])
        tr.sticky, tr.born_scan = True, self.scans
        from .namer import clean_name

        tr.name = clean_name(name) if name else None
        bench = self._bench
        if bench is not None and self.namer is not None and hasattr(self.namer, "task"):
            tr.item = match_item(tr.name, bench.items) if tr.name else -1
            tr.name_state = "none"  # the namer, told the job, says which item it is
        else:
            tr.name_state = "done" if tr.name else "none"
            if bench is not None and tr.name:
                tr.item = match_item(tr.name, bench.items)
        self.tracks.tracks.append(tr)
        return tr

    @property
    def scale(self) -> Scale | None:
        if self.page_mm:
            return Scale(self.page_mm / self.geo.page_w, "page")
        return self.card or self.hand_scale.value()

    def set_geo(self, geo: Geo, img: np.ndarray | None) -> None:
        """A new mapping: rebuild the zone in the new camera px; the learned table carries over where it can."""
        self.geo = geo
        self.dash.geo = geo
        self.slots.geo = geo
        self._look, self._look_ask, self._look_img, self._look_t = None, None, None, -1e9  # taken with the old mapping
        self._boxes_img = None
        if self.state_dir is not None and not self.page_mm:
            self.card = load_card_scale(self.state_dir, geo.page_w, geo.page_h)
        old = self.zone
        self.scan = None
        try:
            self.zone = TableZone(geo, self.frame_size, exclude_design=HUD_RECTS,
                                  threshold=self.threshold, min_area=self.min_area)
        except ValueError as exc:  # shown on the page; the hands and buttons still work
            self.zone, self.overlay, self.motion, self.zone_error = None, None, None, str(exc)
            self.tracks.clear()
            return
        self.zone_error = None
        self.overlay = OverlayHistory(geo, self.zone)
        self.motion = MotionWatch(self.zone)
        if old is not None and old.ref is not None and img is not None:
            self.zone.adopt(old, img)
            self.zone.set_objects([tr.cam_outline(geo) for tr in self.tracks.confirmed])
            self.dirty, self.anywhere, self.last_change_t = True, True, time.monotonic()  # look again, new mapping
        else:
            self.tracks.clear()

    def relearn(self, now: float, delay: float = 1.0) -> None:
        """Forget the objects and learn the empty table again once hands are away."""
        self.tracks.clear()
        self.scan = None
        if self.zone is not None:
            self.zone.status = "learning"
            self.zone.ref = None
        self.learn_at = now + delay
        self._learn_said = False
        self._learn_tries = 0
        self._force_learn = False

    # ---- per frame

    def step(self, frame, poses: list[HandPose], now: float, dt: float, *, detect: bool = True
             ) -> tuple[list[str], list[str]]:
        """One camera frame. Returns (buttons pressed, lines to speak). ``detect=False`` starts no scan (the camera
        may still see a calibration pattern)."""
        say: list[str] = []
        self._n += 1
        if self.mic_gate is not None:
            self.mic_gate.set_open(isinstance(self.games.game, Helper) and self.games.game.mic_open)
        self.slots.update(poses, now, frame.t_mono)
        if poses:
            self.last_hand_t = now
            if self._n % 3 == 0:
                self.hand_scale.add(poses, self.geo.H)
        tips = self.slots.tips()
        if tips:
            self._tips_seen = (now, tips)
        timers = self.games.timers  # they run whatever is on the table
        self._pending_say += timers.update(now, tips, min(max(dt, 0.0), 0.1))
        self.sfx += timers.sfx
        timers.sfx = []
        if self._first_game is not None:
            self._pending_say += self.start_game(self._first_game, now)
            self._first_game = None
        if self.games.active:
            return self._game_step(now, dt, frame)
        fired = self.dash.step(tips, now, dt)
        if fired:
            self.sfx.append("press")
        if "home" in fired:  # the menu: games and the object board
            self._pending_say += self.start_game("menu", now)
            return fired, []
        if "learn" in fired:
            if self.learn_at is not None and self._learn_tries > 0:
                self._force_learn = True  # held again while it refuses: learn the table as it is
                self.learn_at = now
            else:
                self.relearn(now)
        if "scale" in fired:  # a card put down just before (still under the hand, not yet shown) still counts
            self.card_until = now + CARD_SEARCH_S
            self.dash.notify("LOOKING FOR A BANK CARD", now)
        z = self.zone
        if z is None:
            return fired, say
        before, self.hands_mask = self.hands_mask, z.hand_mask(poses)
        overlay = self.overlay.at(frame.t_mono) if self.overlay is not None else None
        if self.motion is not None and self.motion.update(frame.image, overlay):
            self.last_change_t, self.dirty = now, True
        if self._hands_moved(poses) and self.motion is not None:  # what the tracker sees move counts too: where the
            self.last_change_t, self.dirty = now, True  # hand was (it may have taken something) and where it is
            self.motion.mark(before)
            self.motion.mark(self.hands_mask)
        for tr in self.tracks.confirmed:  # a hand on an object freezes it until a scan sees it again
            if self.hands_mask is not None and z.touches(self.hands_mask, tr.blob.cam_poly):
                tr.held, tr.held_t = True, now
            elif tr.held and now - tr.held_t > TIP_HOLD_S:
                tr.held = False
        if self.learn_at is not None and not self._learn_said:  # time to take things off before it is learned
            self._learn_said = True
            self.learn_at = max(self.learn_at, now + LEARN_GRACE_S)
            say.append("Clear the table: learning it empty.")
        if self.scan is not None:
            done = self._scan_step(frame, now)
            if done is not None:
                say += self._scan_done(*done, frame, poses, now)
        elif detect:
            self._maybe_scan(now, frame.image, overlay)
        if self.card_until is not None and now >= self.card_until:
            msg = self._card(now)
            if msg:
                say.append(msg)
        if self.scan is None and not self.dirty and self.scan_image is not None and self.namer is not None:
            self._ask(self.scan_image, now, names_only=True)  # a failed name is asked again from the same view
        say += self._collect(now)
        if self._pending_say:
            say, self._pending_say = self._pending_say + say, []
        say += self._plan_done(now)
        say += self._sweep_done(now)
        if self.guide is not None:
            say += self._guide_step(now)
            if (self.guide.phase == "gather" and self.scan is None and not self.dirty
                    and now - self.last_change_t >= SWEEP_IDLE_S and now - self.last_hand_t >= SWEEP_IDLE_S
                    and self.guide.missing()):
                self.request_sweep(now, why="job still misses " + ", ".join(self.guide.missing()))
        return fired, say

    def _hands_moved(self, poses: list[HandPose]) -> bool:
        """A hand came, went, or moved more than ``HAND_MOVE`` of its palm since the last frame."""
        wrists = [(p.img[0].copy(), max(float(np.linalg.norm(p.img[9] - p.img[0])), 10.0)) for p in poses]
        prev, self._wrists = self._wrists, wrists
        if len(wrists) != len(prev):
            return True
        return any(min(float(np.linalg.norm(w - q)) for q, _ in prev) > HAND_MOVE * palm for w, palm in wrists)

    def _maybe_scan(self, now: float, img: np.ndarray | None = None, overlay: np.ndarray | None = None) -> None:
        z = self.zone
        if self.learn_at is not None:
            if now >= self.learn_at and now - self.last_hand_t >= LEARN_QUIET_S and now - self.last_change_t >= SCAN_SETTLE_S:
                self._start_scan("learn", now)
            return
        if z.ref is None:
            return
        # no scan while a hand is around (3 Nov: "when I hover my hands over the table it rescans every time", and a
        # scan with a hand on a ring kept the ring after the hand took it away)
        hands_gone = now - self.last_hand_t >= SCAN_SETTLE_S
        due = self.dirty and now - self.last_change_t >= SCAN_SETTLE_S and hands_gone
        rescan = self.rescan_at is not None and now >= self.rescan_at
        if not (due or rescan) or now - self.last_scan_t < SCAN_GAP_S:
            return
        now_lit = self.overlay.latest() if self.overlay is not None else None
        if not rescan and not self.anywhere and img is not None and self._unchanged(img, now_lit):
            self.motion.take()  # a hand passed over: the table there looks as it did at the last scan
            self.dirty = False
            self.skipped += 1
            return
        self._start_scan("change", now)

    def _unchanged(self, img: np.ndarray, overlay: np.ndarray | None) -> bool:
        """Where the view changed since the last scan, the camera sees the table as the scans left it: nothing was put
        down, taken or moved (our own light there is left out). Then no scan, and no blink."""
        z = self.zone
        if self.clean is None or self.motion is None:
            return False
        region = (self.motion.peek() > 0) & (z.inner > 0)
        lit = np.zeros_like(region) if overlay is None else cv2.dilate(overlay, _disk(3)) > 0
        valid = region & ~lit
        quiet = ~region & (z.inner > 0) & ~lit  # the light now against then, from where nothing changed
        if np.count_nonzero(valid) < 50 or np.count_nonzero(quiet) < 200:
            return False
        if np.count_nonzero(region & lit) > 0.35 * np.count_nonzero(region):
            return False  # too much of it lies under our light to tell: look properly
        cur = z._prep(img)
        ratio = np.median(cur[quiet][::5] / np.maximum(self.clean[quiet][::5], 1.0), axis=0)
        d = np.abs(cur - self.clean * ratio).max(axis=2)
        changed = ((d > z.threshold) & valid).astype(np.uint8) * 255
        changed = cv2.morphologyEx(changed, cv2.MORPH_OPEN, _disk(OPEN_R + 1))
        return np.count_nonzero(changed) < z.min_area

    def _start_scan(self, reason: str, now: float) -> None:
        """A local scan when what changed is a small part of the table, else a whole one (learning, a light change,
        the first look at a table, a big change)."""
        region = None
        z = self.zone
        if reason == "change" and not self.anywhere and self.rescan_at is None and self.motion is not None:
            m = self.motion.peek()
            if self.hands_mask is not None:
                m = cv2.bitwise_or(m, self.hands_mask)
            share = float(np.count_nonzero(m & z.inner)) / max(1, z.inner_px)
            if 0.0 < share <= LOCAL_MAX:
                region = cv2.dilate(m, _disk(LOCAL_GROW))
        self.scan = Scan(reason, now, region=region)

    def _in_region(self, cam_poly) -> bool:
        s = self.scan
        if s is None or s.region is None:
            return True
        return bool(np.any(s.region[self.zone.poly_mask(cam_poly) > 0]))

    def _abort_scan(self, why: str, now: float) -> None:
        self.scan = None
        self.scan_aborts += 1
        self.dirty, self.last_change_t = True, now  # try again once the view is still
        self.log.append(f"scan dropped: {why} ({self.scan_aborts} so far)")

    def _scan_step(self, frame, now: float) -> tuple[np.ndarray, str] | None:
        """Advance the scan; returns (the clean view, reason) once it is complete."""
        s = self.scan
        if s.clean_after is None:
            if self.link is None:
                s.clean_after = s.t0 + SCAN_NO_LINK_S
            else:
                if s.target is None:  # the dark scene went out after the previous step
                    s.target = self.link.version()
                acked = self.link.acked_at(s.target)
                if acked is None:
                    if now - s.t0 > SCAN_ACK_TIMEOUT_S:
                        self._abort_scan("the projector page did not show the dark table in time", now)
                    return None
                s.clean_after = acked + SCAN_AFTER_ACK_S
        if frame.t_mono < s.clean_after:
            return None
        s.frames.append(frame.image)
        if len(s.frames) < SCAN_FRAMES:
            return None
        a, b = s.frames[0], s.frames[-1]
        diff = frames_agree(a, b)
        if diff > SCAN_AGREE:
            self._abort_scan(f"the scan frames differ by {diff:.1f} grey levels", now)
            return None
        img = a if a is b else cv2.addWeighted(a, 0.5, b, 0.5, 0.0)
        self._done_region = s.region
        self.scan = None
        self.last_scan_t = now
        self.scans += 1
        return img, s.reason

    def _scan_done(self, img: np.ndarray, reason: str, frame, poses: list[HandPose], now: float) -> list[str]:
        z = self.zone
        self.scan_image = img
        cur = z._prep(img)  # the table as the scans see it: whole after a whole scan, patched after a local one
        if self._done_region is None or self.clean is None or reason == "learn":
            self.clean = cur
        else:
            self.clean[self._done_region > 0] = cur[self._done_region > 0]
        if reason == "learn":
            stray = z.stray_objects(img)
            if stray and not self._force_learn:
                self._learn_tries += 1
                self.learn_at = now + LEARN_RETRY_S
                self.log.append(f"not learning the table yet: {stray} dark thing{'s' if stray > 1 else ''} on it")
                self.dash.notify("CLEAR THE TABLE, OR HOLD LEARN", now, "warn")
                if self._learn_tries in (1, 3) or self._learn_tries % 6 == 0:  # remind, without nagging
                    self.sfx.append("error")
                    return ["Something is still on the table. Clear it, or hold LEARN to learn it as it is."]
                return []
            z.learn(img)
            self.learn_at, self._learn_tries, self._force_learn = None, 0, False
            self.motion.reset()
            self.dirty, self.anywhere, self.rescan_at = False, False, None
            self.sfx.append("scan")
            return ["Table learned. Put objects on it."]
        # with no hand around, what touches the edge of the detection area is no arm: an object near the buttons or
        # the page edge (3 Nov: a plush penguin next to the button column was never outlined)
        keep_edge = not poses and now - self.last_hand_t > 1.0
        region, self._done_region = self._done_region, None
        overlay = None if region is None else self.overlay.at(frame.t_mono)  # the objects left lit, elsewhere
        blobs = z.update(img, frame.t_mono, overlay=overlay, hands=self.hands_mask, hand_visible=bool(poses),
                         quiet=not poses, keep_edge=keep_edge, region=region)
        self.rescan_at = None
        if blobs is None:  # the motion seen so far still counts for the next scan that sees objects (3 Nov: it was
            if z.status == "light":  # used up here, and a wallet taken away meanwhile stayed outlined for good)
                self.rescan_at = now + RELEARN_AFTER_S + 0.2  # look again: a lasting light change is relearned
            self.dirty = False
            return []
        allowed_mask = self.motion.take()
        self.dirty = False
        if z.last_edge:  # something across the edge: absorbed if it stays (TableZone.update)
            self.rescan_at = now + STUCK_EDGE_S + 0.2

        def allowed(poly) -> bool:
            return bool(np.any(allowed_mask[z.poly_mask(poly) > 0]))

        inside = None
        if region is not None:
            def inside(poly) -> bool:
                return bool(np.any(region[z.poly_mask(poly) > 0]))
        events = self.tracks.update_scan(blobs, frame.t_mono, lambda poly: z.touches(self.hands_mask, poly),
                                         None if self.anywhere else allowed, inside=inside)
        self.anywhere = False
        z.set_objects([tr.cam_outline(self.geo) for tr in self.tracks.confirmed])
        new = [tr for kind, tr in events if kind == "new"]
        for tr in new:
            tr.born_scan = self.scans
        if new:
            self.sfx.append("found")
        elif any(kind == "gone" for kind, _ in events):
            self.sfx.append("gone")
        say: list[str] = []
        if self.card_until is not None:
            msg = self._card(now)
            if msg:
                say.append(msg)
        self._ask(img, now)
        if self.namer is None:
            if len(new) == 1:
                say.append(spoken_new(new[0], self.scale))
            elif new:
                say.append(f"{len(new)} new objects.")
        else:
            self._announce += [(tr, now + NAME_WAIT_S) for tr in new]
        return say

    # ---- names and outlines (Meta models, on worker threads)

    def _ask(self, img: np.ndarray, now: float, *, names_only: bool = False) -> None:
        """Ask for a name and a SAM outline for every shown object that has none, from the clean view ``img``."""
        todo = [tr for tr in self.tracks.confirmed if not tr.held and tr.shown is not None]
        if names_only:
            todo = [tr for tr in todo if tr.name_state == "none" and now - tr.name_t >= NAME_RETRY_S]
            if not todo:
                return
        if self.namer is not None:
            for tr in todo:
                if tr.name_state == "none" and now - tr.name_t >= NAME_RETRY_S and tr.shown[1] >= NAME_MIN_PAGE:
                    jpeg = self.crop(img, tr)
                    if jpeg is not None and self.namer.request(tr.uid, jpeg):
                        tr.name_state, tr.name_t = "pending", now
        if self.outliner is not None and not names_only:
            ask = [tr for tr in todo if tr.sam_state == "none"]
            if ask and self.outliner.request(img, [(tr.uid, self._cam_box(tr)) for tr in ask]):
                for tr in ask:
                    tr.sam_state, tr.sam_rect = "pending", tr.rect

    def _cam_box(self, tr: Track) -> tuple[int, int, int, int]:
        """Half-open camera-px box around the object, a little grown (a box prompt for SAM). A box from a sweep is
        grown by 30 %: Muse Spark's boxes are a few per cent of the view off, and SAM given a box that misses part of
        the object outlines shadows instead (3 Nov, table photo: tight boxes 0/5 clean outlines, grown 5/5)."""
        pad = 0.3 * max(tr.blob.sides) if tr.sticky and not tr.refined else 0.3 * PAD_PAGE
        cam = apply_h(self.geo.Hi, cv2.boxPoints(grown(tr.rect, pad)))
        (x0, y0), (x1, y1) = np.floor(cam.min(axis=0)), np.ceil(cam.max(axis=0))
        w, h = self.frame_size
        return int(max(0, x0)), int(max(0, y0)), int(min(w, x1 + 1)), int(min(h, y1 + 1))

    def _collect(self, now: float) -> list[str]:
        """Names and outlines that arrived; announcements that are due."""
        if self.namer is not None:
            for tr in self.tracks.tracks:
                if tr.name_state == "pending":
                    done, name = self.namer.result(tr.uid)
                    if done:
                        tr.name, tr.name_state = (name, "done") if name else (None, "none")
                        bench = self._bench
                        if bench is not None and name:
                            item = self.namer.match(tr.uid) if hasattr(self.namer, "match") else None
                            tr.item = item if item is not None else match_item(name, bench.items)
                        err = f" ({self.namer.error})" if self.namer.error else ""
                        self.log.append(f"object {tr.id}: {name or 'no name'}{err} [{self.namer.summary()}]")
        if self.outliner is not None:
            for tr in self.tracks.tracks:
                if tr.sam_state != "pending":
                    continue
                done, contour = self.outliner.result(tr.uid)
                if not done:
                    continue
                tr.sam_state = "done"
                if contour is None or tr.sam_rect is None or rect_changed(tr.sam_rect, tr.rect):
                    continue  # no mask, or the object moved since the scan it was asked for
                page = apply_h(self.geo.H, contour)
                if tr.sticky and not tr.refined:
                    if sweep_fits(page, tr):  # where SAM says the object really is, near the model's box
                        rect = cv2.minAreaRect(page.astype(np.float32))
                        tr.blob = replace(tr.blob, rect=rect, cam_poly=np.asarray(contour, np.float64),
                                          contour=page.astype(np.float64))
                        tr.rect = rect
                        tr.refine(page)
                        if self.zone is not None:
                            self.zone.set_objects([t.cam_outline(self.geo) for t in self.tracks.confirmed])
                        self.log.append(f"object {tr.id}: SAM outline (found by a sweep)")
                    else:
                        self.log.append(f"object {tr.id}: SAM outline does not fit the sweep's box; kept the box")
                elif sam_fits(page, tr):
                    tr.refine(page)
                    self.log.append(f"object {tr.id}: SAM outline")
        say: list[str] = []
        if self._announce:
            due = [tr for tr, deadline in self._announce if tr.name or now >= deadline]
            self._announce = [(tr, d) for tr, d in self._announce if tr not in due]
            due = [tr for tr in due if tr in self.tracks.tracks]
            if self.guide is not None:
                say += self._guide_news(due)
            elif self.task is None:
                say += [spoken_new(tr, self.scale) for tr in due][:3]
            else:
                say += self._task_news(due)
        if self.task is not None:
            ready = self._task_ready()
            self.dash.task_line, self.dash.task_role = self._task_line(), "target" if ready else "info"
            if ready and not self._ready_said:
                self._ready_said = True
                self.sfx.append("answer")
                say.append(f"Everything for {self.task.title} is on the table. Ready to start.")
            elif not ready:
                self._ready_said = False
        say += self._answers(now)
        return say

    # ---- bench check (--task)

    def _present(self) -> dict[int, Track]:
        """Listed item number -> the first shown object that is it."""
        out: dict[int, Track] = {}
        for tr in sorted(self.tracks.confirmed, key=lambda tr: tr.id):
            if tr.item > 0 and tr.item not in out:
                out[tr.item] = tr
        return out

    def _task_ready(self) -> bool:
        return self.task is not None and len(self._present()) == len(self.task.items)

    def _task_line(self) -> str:
        t = self.task
        have = self._present()
        line = f"{t.title.upper()} {len(have)}/{len(t.items)}"
        missing = t.missing(set(have))
        if not missing:
            return f"{line} READY"
        line += " NEED " + ", ".join(m.upper() for m in missing)
        return line if len(line) <= STATUS_CHARS else line[:STATUS_CHARS - 3].rstrip(", ") + "..."

    def _task_news(self, due: list[Track]) -> list[str]:
        """What to say about objects that were just named, with a task: a listed item ticked off, or one not needed."""
        t = self.task
        out: list[str] = []
        ticked = []
        for tr in due:
            if tr.item > 0 and self._present().get(tr.item) is tr:
                ticked.append(t.items[tr.item - 1])
            elif tr.item == 0:
                self.sfx.append("error")
                out.append(f"{(tr.name or 'that').capitalize()} is not on the list for {t.title}.")
        if ticked:
            missing = t.missing(set(self._present()))
            line = f"{spoken_list(ticked).capitalize()}, check."
            if missing:
                line += f" {len(missing)} to go: {spoken_list(missing)}."
            out.insert(0, line)
        return out

    # ---- spoken questions (--voice)

    def heard_partly(self, text: str, t: float, now: float) -> list[str]:
        """They let go and the transcriber showed words (``text``, its latest, at ``t``) but never finished the sentence:
        what it heard last will do, rather than "I didn't catch that"."""
        g = self.games.game
        if not isinstance(g, Helper) or g.mic_open or g.released_t is None or g.utter:
            return []
        if not text or t < g.opened_t or now - g.released_t < PARTIAL_AFTER_S:
            return []
        self.log.append(f"helper: no final transcript, going by the last partial {text!r}")
        said = self.hear(text, now)
        self._partial_used = (_words_of(text), now)
        return said

    def _final_of_used_partial(self, q: str, now: float) -> bool:
        """The final transcript of a sentence already acted on from its last partial."""
        words, t = self._partial_used
        if not words or now - t > PARTIAL_LATE_S:
            return False
        said = _words_of(q)
        if said.startswith(words) or words.startswith(said):
            self._partial_used = ("", -1e9)
            return True
        return False

    def hear(self, text: str, now: float) -> list[str]:
        """A sentence heard by Muse Voice Transcribe. "What is missing?" is answered from the bench check; anything
        else goes to the asker with a clean view of the object pointed at (or of the whole table)."""
        q = " ".join(text.split())
        if not q:
            return []
        if self.mic_gate is not None and not self.mic_gate.recently_open(HEARD_AFTER_S):
            # 4 Nov: the transcriber said 'by' with the mic shut, out of pure silence. On the helper tile that would be a job
            self.log.append(f"heard with the mic shut, ignored: {q!r}")
            return []
        if self._final_of_used_partial(q, now):
            self.log.append(f"helper: late final of what the partial already did: {q!r}")
            return []
        self.dash.notify(f"HEARD: {q.upper()}"[:STATUS_CHARS], now)
        low = q.lower()
        timers = self.games.timers
        said = timers.hear(q, now)
        if said is not None:
            self.sfx += timers.sfx
            timers.sfx = []
            if isinstance(self.games.game, Helper):
                self.games.game.dealt_with()
            return said
        if isinstance(self.games.game, Helper):
            return self._helper_hears(self.games.game, q, now)
        was_playing = self.games.active
        said = self.games.hear(q, now)
        if said is not None:
            if self.games.active and not was_playing:
                self.dash.game_mode(True)
                self._smooth(True)
                self.scan = None
            elif was_playing and not self.games.active:
                said = [line for line in said if not line.startswith("Object board")] + self._leave_games(now)
            self.sfx.append("press")
            return said
        g = self.guide
        verb = command_of(low)
        if g is not None and g.phase in ("gather", "steps") and verb is not None:
            self.sfx.append("press")
            return g.command(verb, now)
        if g is not None and g.phase == "gather" and any(w in low for w in ("missing", "what else", "still need")):
            missing = g.missing()
            return [f"Still missing {spoken_list(missing)}." if missing else "Nothing. Say start."]
        if any(w in low for w in LOOK_WORDS):
            if self.request_sweep(now, why="asked"):
                self.sfx.append("listen")
                self.dash.notify("LOOKING OVER THE WHOLE TABLE", now)
                return ["Looking."]
            return ["I can't look closer right now." if self.brain is None else "Give me a moment."]
        if is_plan_request(low):
            return self.request_plan(q, now)
        if self.task is not None and any(w in low for w in ("missing", "what else", "do i need", "still need")):
            missing = self.task.missing(set(self._present()))
            self.sfx.append("answer")
            return [f"Still missing: {spoken_list(missing)}." if missing else f"Nothing. Everything for {self.task.title} is here."]
        if self.asker is None or is_filler(low) or (self.games.active and not is_question(low)):
            return []
        if self._asking is not None:
            return ["One moment, I am still answering the last question."]
        target = None if any(w in low for w in ("table", "these", "everything", "all of")) else self._pointed(now)
        names = [f"{tr.id}: {tr.name or tr.colour + ' object'}" for tr in sorted(self.tracks.confirmed, key=lambda tr: tr.id)]
        context = "Objects the camera found on the table: " + ("; ".join(names) if names else "none") + "."
        if self.task is not None:
            context += f" The person is getting ready for: {self.task.title} (needs {spoken_list(list(self.task.items))})."
        if g is not None and g.phase == "steps":
            step = g.plan.steps[g.k]
            context += (f" They are doing: {g.plan.title} ({g.plan.note}), step {g.k + 1} of {len(g.plan.steps)}: "
                        f"{step.say}. The plan's items: "
                        + "; ".join(f"{it.name} {it.amount_text()}".strip() for it in g.plan.items) + ".")
        jpegs: list[bytes] = []
        if target is not None:
            context += f" They are pointing at object {target.id}" + (f" ({target.name})." if target.name else ".")
            target.asked_until = now + 60.0  # lit until answered (then ASKED_GLOW_S more)
            if self.scan_image is not None:
                crop = self.crop(self.scan_image, target)
                jpegs += [crop] if crop is not None else []
        if self.scan_image is not None:
            view = self._table_jpeg(self.scan_image)
            jpegs += [view] if view is not None else []
        self._ask_n += 1
        if not self.asker.ask(self._ask_n, q, jpegs, context):
            return ["Sorry, I am busy. Ask me again in a moment."]
        self._asking = (self._ask_n, target, q)
        self.sfx.append("listen")
        return []

    def _pointed(self, now: float) -> Track | None:
        t, tips = self._tips_seen
        if now - t > POINT_RECENT_S or not tips:
            return None
        best, best_d = None, POINT_NEAR
        for tr in self.tracks.confirmed:
            poly = np.array([self.geo.p2d(p) for p in tr.outline], dtype=np.float32)
            for tip in tips:
                d = max(0.0, -cv2.pointPolygonTest(poly, (float(tip[0]), float(tip[1])), True))
                if d < best_d:
                    best, best_d = tr, d
        return best

    def _table_jpeg(self, img: np.ndarray) -> bytes | None:
        """The clean view of the projected page area, at most 1024 px wide."""
        if self.zone is None:
            return None
        x, y, w, h = self.zone.roi
        view = img[y:y + h, x:x + w]
        f = 1024 / max(view.shape[:2])
        if f < 1:
            view = cv2.resize(view, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)
        ok, buf = cv2.imencode(".jpg", view, [cv2.IMWRITE_JPEG_QUALITY, 85])
        return buf.tobytes() if ok else None

    def _answers(self, now: float) -> list[str]:
        if self._asking is None or self.asker is None:
            return []
        key, target, q = self._asking
        done, answer = self.asker.result(key)
        if not done:
            return []
        self._asking = None
        if target is not None:
            target.asked_until = now + ASKED_GLOW_S
        err = f" ({self.asker.error})" if self.asker.error else ""
        self.log.append(f"asked {q!r}: {answer or 'no answer'}{err} [{self.asker.summary()}]")
        self.sfx.append("answer" if answer else "error")
        return [answer or "Sorry, I could not answer that."]

    def crop(self, img: np.ndarray, tr: Track) -> bytes | None:
        """JPEG of the object alone, white-balanced against the table: its contour's box plus a small margin."""
        contour = tr.blob.contour if tr.blob.contour is not None else cv2.boxPoints(tr.rect)
        box = grown(cv2.minAreaRect(np.asarray(contour, np.float32)), 0.4 * PAD_PAGE)
        cam = apply_h(self.geo.Hi, cv2.boxPoints(box))
        h, w = img.shape[:2]
        x0, y0 = np.floor(cam.min(axis=0)).astype(int)
        x1, y1 = np.ceil(cam.max(axis=0)).astype(int)
        x0, y0, x1, y1 = max(0, x0), max(0, y0), min(w, x1), min(h, y1)
        if x1 - x0 < 8 or y1 - y0 < 8:
            return None
        c = img[y0:y1, x0:x1].astype(np.float32)
        if self.zone is not None:  # the table white again: a warm room light turned every object orange
            table = np.maximum(self.zone.table_bgr * self.zone.gain, 1.0)
            c = c * (225.0 / table)
        c = np.clip(c, 0, 255).astype(np.uint8)
        f = CROP_MAX / max(c.shape[:2])
        if f < 1:
            c = cv2.resize(c, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)
        ok, buf = cv2.imencode(".jpg", c, [cv2.IMWRITE_JPEG_QUALITY, 88])
        return buf.tobytes() if ok else None

    def _card(self, now: float) -> str | None:
        """While a SCALE press is fresh: the scale once a card shows (and is let go), or a hint when none did."""
        found = find_card(self.tracks.tracks)
        if found is None:
            if now < self.card_until:
                return None
            self.card_until = None
            self.dash.notify("NO BANK CARD FOUND", now, "warn")
            return "No bank card found. Put one on the table, let go, then hold SCALE again."
        self.card_until = None
        scale, tr = found
        self.card = scale
        if self.state_dir is not None:
            save_card_scale(self.state_dir, scale, self.geo.page_w, self.geo.page_h)
        self.dash.notify(f"SCALE FROM CARD {tr.id}", now, "target")
        if self.page_mm:
            return "Card measured, but the page width you gave sets the scale."
        return "Scale set from the card."

    def dump(self, path: Path, img: np.ndarray) -> None:
        """Debug snapshot: the camera frame, the last clean scan, the zone's view and every track."""
        path.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(path / "frame.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 92])
        hands = [{"slot": s.k, "side": s.side, "seen": s.seen, "img": s.pose.img.round(1).tolist(),
                  "pinch": round(s.pinch.ratio, 3), "down": s.pinch.down} for s in self.slots.active]
        (path / "hands.json").write_text(json.dumps(hands) + "\n", encoding="utf-8")
        if self.scan_image is not None:
            cv2.imwrite(str(path / "scan.jpg"), self.scan_image, [cv2.IMWRITE_JPEG_QUALITY, 92])
        if self.zone is not None:
            self.zone.dump(path)
        rows = [{"id": tr.id, "uid": tr.uid, "colour": tr.colour, "name": tr.name, "held": tr.held,
                 "hits": tr.hits, "sides_page": tr.shown, "rect_page": [list(tr.rect[0]), list(tr.rect[1]), tr.rect[2]],
                 "refined": tr.refined, "cam_outline": np.round(tr.cam_outline(self.geo), 1).tolist()}
                for tr in self.tracks.tracks]
        meta = {"roi": list(self.zone.roi) if self.zone else None, "status": getattr(self.zone, "status", None),
                "gain": [float(g) for g in self.zone.gain] if self.zone else None, "scans": self.scans,
                "scan_aborts": self.scan_aborts, "tracks": rows}
        (path / "tracks.json").write_text(json.dumps(meta, indent=1, default=float) + "\n", encoding="utf-8")

    def scene(self, now: float, also_lit: list = ()) -> list:
        """Prims to publish now; what lights the table (plus ``also_lit``, e.g. the page border) joins the overlay mask.
        During a scan only the HUD is drawn."""
        keep = label_ok = None
        if self.scan is not None and self.scan.region is not None:
            g = self.geo

            def keep(tr: Track) -> bool:
                return not self._in_region(tr.cam_outline(g))

            def label_ok(rect) -> bool:
                x0, y0, x1, y1 = rect
                return not self._in_region(g.d2c([(x0, y0), (x1, y0), (x1, y1), (x0, y1)]))
        if self.games.active:
            if self._look is not None:  # the helper tile looks at the table: nothing of ours in the picture
                return []
            self.dash.lines = self.games.hud(now)
            return self.dash.game_scene(self.slots, now, self.games.draw,
                                        cursors=not getattr(self.games.game, "draws_hands", False))
        guide = self.guide
        if guide is not None:
            self.dash.views = {o.uid: v for o in self.seen() if (v := guide.view(o, now)) is not None}
            self.dash.extra = guide.draw
            self.dash.lines = guide.hud(now)
        else:
            self.dash.views, self.dash.extra = {}, None
            self.dash.lines = None
        if self.games.timers.active:  # on the object board too; a scan draws no extras, and what we draw is masked
            below = self.dash.extra

            def extra(c, t) -> None:
                if callable(below):
                    below(c, t)
                self.games.timers.draw(c, t)

            self.dash.extra = extra
        if self._plan_job is not None:
            self.dash.lines = (("PLANNING: " + self._plan_job[1].upper())[:STATUS_CHARS], "ONE MOMENT...", "info")
        prims, lit = self.dash.scene(self.zone, self.tracks.tracks, self.slots, self.scale, now,
                                     blank=self.scan is not None, scan_n=self.scans, keep=keep, label_ok=label_ok)
        if self.overlay is not None:
            self.overlay.set(lit + list(also_lit), now)
        return prims


COMMANDS = (  # (spoken phrases, guide verb): a short sentence that is one of these drives the guided job
    (("next step", "next", "done", "i'm done", "im done", "finished", "go on", "continue", "skip", "okay next",
      "ok next"), "next"),
    (("go back", "back", "previous", "previous step"), "back"),
    (("repeat", "say again", "say that again", "again", "what was that", "what now", "what's next"), "repeat"),
    (("let's start", "lets start", "start", "let's go", "lets go", "begin", "ready"), "start"),
    (("stop", "cancel", "quit", "stop the job"), "stop"),
    (("undo", "minus one", "that didn't count", "that one didn't count", "doesn't count"), "undo"),
    (("how many", "count"), "how many"),
    (("check", "check again", "check it", "have a look"), "check"),
)
PLAN_WORDS = ("help me", "i want to", "i'd like to", "id like to", "let's make", "lets make", "let's do", "lets do",
              "let's cook", "let's bake", "how do i", "how to", "show me how", "i need to", "can you help",
              "walk me through", "guide me", "teach me", "make me", "let's prepare", "i'm going to", "im going to")


def model_tag(model: str, provider: str) -> str:
    """"MUSE-SPARK-1.3" or "MUSE-SPARK-OPEN @ BASETEN": who did it, short enough for the status bar."""
    name = model.rsplit("/", 1)[-1].upper()
    return name if provider == "meta" else f"{name} @ {provider.upper()}"


def command_of(low: str) -> str | None:
    """The guide verb a short sentence is ("next", "go back", "how many?"), or None."""
    t = re.sub(r"[^a-z' ]+", " ", low).strip()
    t = re.sub(r"^(please|okay|ok|lightwork|hey lightwork)\s+", "", t).strip()
    t = re.sub(r"\s+(please|now|then)$", "", t).strip()
    if not t or len(t.split()) > 4:
        return None
    for phrases, verb in COMMANDS:
        if t in phrases:
            return verb
    return None


def is_plan_request(text: str) -> bool:
    low = text.lower()
    return any(w in low for w in PLAN_WORDS)


FILLERS = {"um", "umm", "uh", "uhm", "erm", "er", "hmm", "hm", "mm", "mhm", "ah", "oh", "eh", "huh"}
QUESTION_WORDS = ("what", "what's", "whats", "how", "where", "why", "who", "which", "when", "can", "could", "is",
                  "are", "do", "does", "should", "will", "would", "tell", "show", "explain")


def is_filler(text: str) -> bool:
    """Nothing but "um", "hmm"... (3 Nov: "um" went to Muse Spark as a question)."""
    words = re.sub(r"[^\w' ]+", " ", text.lower()).split()
    return all(w in FILLERS for w in words)


def is_question(text: str) -> bool:
    """A sentence that asks something: it ends with a question mark or starts with a question word. During a game only
    these go to the asker (3 Nov: talk at the table, in English or Russian, came back as answers mid-game)."""
    low = text.lower().strip()
    words = re.sub(r"[^\w' ]+", " ", low).split()
    return bool(words) and not is_filler(low) and (low.endswith("?") or words[0] in QUESTION_WORDS)


def sweep_fits(contour_page: np.ndarray, tr: Track) -> bool:
    """A SAM contour for a box from a sweep: centred inside the (30 % grown) box, between a twentieth and 1.6 times its
    area. Looser than ``sam_fits``: the box itself is only roughly where the object is."""
    if len(contour_page) < 3:
        return False
    area = abs(float(cv2.contourArea(contour_page.astype(np.float32))))
    long, short = tr.blob.sides
    box = cv2.boxPoints(grown(tr.rect, 0.3 * long)).astype(np.float32)
    centre = contour_page.mean(axis=0)
    return (0.05 <= area / max(long * short, 1.0) <= 1.6
            and cv2.pointPolygonTest(box.reshape(-1, 1, 2), (float(centre[0]), float(centre[1])), False) >= 0)


def sam_fits(contour_page: np.ndarray, tr: Track) -> bool:
    """A SAM contour (page px) plausibly outlines the tracked object: about its area, centred on it."""
    if len(contour_page) < 3:
        return False
    area = abs(float(cv2.contourArea(contour_page.astype(np.float32))))
    long, short = tr.shown or tr.blob.sides
    expected = max(long * short * max(tr.blob.fill, 0.3), 1.0)
    centre = contour_page.mean(axis=0)
    return 0.4 <= area / expected <= 1.8 and float(np.hypot(*(centre - tr.blob.centre))) <= 0.5 * long
