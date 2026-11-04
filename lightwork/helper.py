"""General helper tile (``lightwork play --table --voice --game helper``, or GENERAL HELPER on the menu): say what you want.

Hold a fingertip on the mic bubble and talk. The microphone is open only then (:class:`lightwork.voice.capture.MicGate`),
so talk at the table goes nowhere.

- "Start rock paper scissors", "let's start the catch light game": the game starts at once, no model involved.
- Anything else is a job ("help me build this Lego", "what can I cook with this?"). The table goes dark for a moment, the
  camera takes a clean picture and a Meta model works out what is there and what it still has to know. It asks only what
  the picture cannot answer: how many grams are in the bowl, but never how many eggs it can count. A manual page or a
  picture of the finished thing lying on the table is read through the camera.
- Then the job as steps: the text on the table, what to take outlined (SAM 3.1) and an arrow to where it goes. Each step
  looks at the table again, because things move. NEXT and BACK are held like any button; a step with a question for the
  camera is checked once the hands are away, and moves on by itself when the model sees it done. NEXT always overrides.

This module is the logic and the drawing: no camera, no network. It says what it wants in ``want`` (an :class:`Ask`: a
prompt, whether a fresh clean look is needed) and ``want_outlines``; :class:`lightwork.tabletop.TableSession` takes the
picture, makes the recorded model calls and hands the answers back through :meth:`Helper.answered` and
:meth:`Helper.outlined`. Everything is in design px (750 x 420, the page); the picture sent to the model is the page
rectified to that frame, so a box on the model's 0-1000 grid maps straight to design px.
"""

from __future__ import annotations

import math
import re
import textwrap
from dataclasses import dataclass

import numpy as np

from .games import Bubble, Game, _near, mic_icon
from .guide import plural
from .playground import DESIGN_H, DESIGN_W
from .timers import Timers, clock_text

STRIP_X = 628.0  # the controls keep to the right of this line, so a hand at work cannot press them
WORK = (72.0, 70.0, 620.0, 358.0)  # where the text goes: right of the buttons, under the status bar, over the timers
BUBBLE_R = 36.0
# 4 Nov, real footage: nearly every tracking miss was a hand on a mic beside the near edge, its palm and wrist out of the
# camera's view. The mic is held longest, so it goes where the whole hand is seen; BACK, the least used, nearest the person.
NEXT_C, MIC_C, BACK_C = (688.0, 96.0), (688.0, 204.0), (688.0, 338.0)
LINE_H = 26.0  # design px between label lines
CHARS = 30  # characters per line of the text block
TEXT_W = 16.5  # design px per character for laying text out: on the safe side of what bold capitals take (games.EM)
MIC_OPEN_S = 0.15  # a fingertip this long on the bubble opens the mic: a hand passing over it does not
MIC_TAIL_S = 0.6  # the tracker loses a still hand for a moment now and then, and the last word needs time
UTTER_END_S = 1.5  # after letting go, or the last transcript: what was said is complete
NO_WORDS_S = 6.0  # after letting go with nothing transcribed: say so
MAX_QUESTIONS = 3  # then it plans with what it has
MAP_RECT = (72.0, 70.0, 620.0, 358.0)  # design px: the map fills the work area: the mic, the status bar and the clocks stay out
FLY_S = 7.0  # the flight from the whole world down to the place
MAP_PITCH = 62.0  # degrees: tall buildings stand up
ORBIT_DEG_S = 2.0  # the view turns slowly while nobody holds the map
TILE_PX = 512.0  # the map's world is this many px wide at zoom 0
PLACE_ZOOM = (3.0, 18.0)
MODEL_C = (346.0, 214.0)  # design px: the middle of the work area
MODEL_R = 140.0  # design px per model unit at zoom 1: the model is within -1 to 1
TURN_PER_PX = 0.012  # radians of turn per design px dragged
TILT_PER_PX = 0.008
TILT_RANGE = (-0.4, 1.35)  # from a little below to nearly straight down: never upside down
ZOOM_RANGE = (0.45, 2.4)
IDLE_SPIN = 0.35  # radians per second while nobody holds it: a still wireframe does not read as 3D
SPIN_DECAY_S = 1.2  # what a drag gave it runs out in about this long, back to the idle spin
DRAG_JUMP = 90.0  # design px in one frame: the tracker jumped, not the hand
MAX_WIRE_POINTS = 700  # all polylines together: a scene goes out 30 times a second
MAX_WIRE_LINES = 90
PIC_W, PIC_H = 345.0, 230.0  # design px, 38 x 25 cm on the table: Muse Image's 3:2
NOTE_CHARS = 22  # characters per line of an answer's notes: beside a book and a picture there is no room for more
MAX_NOTE_LINES = 3
MAX_THINGS = 16
MAX_STEPS = 8
MAX_CHECKS = 6  # automatic camera checks per step
CHECK_AFTER_S = 1.5  # hands away this long after working: look
ADVANCE_S = 2.0  # a finished step stays up this long
TRIP_GAP_S = 0.8  # two counted trips are at least this far apart ...
OUTSIDE_S = 0.4  # ... and the hand was out of the target this long between them
OUTLINE_PAD = 6.0  # design px around a box drawn as an outline (no SAM outline yet)
OUTLINE_MOVE = 30.0  # design px a thing's box moves before it is outlined again (4 Nov: a model's boxes vary by 20)
UID_BASE = 100_000  # outliner keys: never a table track's

# ---------------------------------------------------------------------------------------------------- what was said

GAME_NAMES = {  # the name alone is enough ...
    "rps": ("rock paper scissors", "rock paper scissor", "rock paper and scissors", "rock and paper and scissors"),
    "catch": ("catch the light", "catch the lights", "catch light", "catch lights", "catching the light"),
    "follow": ("follow the lights", "follow the light", "follow lights", "follow light"),
    "draw": ("draw and guess", "drawn guess", "draw guess"),
}
GAME_WORDS = {  # ... these only with "play" or "game" ("help me follow the manual" is no game)
    "rps": ("rock paper", "rock"),
    "catch": ("catch",),
    "follow": ("follow", "memory", "simon"),
    "draw": ("draw", "drawing", "pictionary"),
}
BOARD_WORDS = ("object board", "objects board")
MENU_WORDS = ("menu", "go home", "go back home")
TAKE_ME = re.compile(r"^(?:(?:please|can you|could you|lightwork|hey|ok|okay|now) )*(?:take me to|fly me to|fly to|let's go to|lets go to) (?:the )?(.+)$")
SHOW_ME = re.compile(r"^(?:(?:please|can you|could you|lightwork|hey|ok|okay|now) )*show me (?:a |an |the |some )?(.+)$")
NOT_A_THING = ("how", "what", "where", "why", "when", "which", "who", "this", "that", "these", "those", "my", "it", "me",
               "more", "something", "around", "next", "again", "step", "steps", "manual", "book", "page", "table", "map")
NEW_JOB_WORDS = ("help me", "i want", "i'd like", "id like", "let's", "lets", "new job", "something else", "instead")


def _words(text: str) -> str:
    return " ".join(re.sub(r"[^a-z0-9' ]+", " ", text.lower()).split())


def _has(text: str, phrase: str) -> bool:
    return re.search(rf"\b{re.escape(phrase)}\b", text) is not None


def asked_for_game(text: str) -> str | None:
    """The tile a sentence asks for (a game's key, "menu" or "objects"), or None: it is about a job."""
    t = _words(text)
    for key, names in GAME_NAMES.items():
        if any(_has(t, n) for n in names):
            return key
    if any(_has(t, w) for w in BOARD_WORDS):
        return "objects"
    if any(_has(t, w) for w in MENU_WORDS):
        return "menu"
    if not any(_has(t, w) for w in ("play", "game", "games")):
        return None
    for key, words in GAME_WORDS.items():
        if any(_has(t, w) for w in words):
            return key
    return None if _has(t, "help me") else "menu"  # "let's play a game"; but "help me play chess" is a job


def asked_to_go(text: str) -> str | None:
    """The place a sentence asks to be taken to ("take me to Paris" -> "paris"), or None. Keeps it a name: at most 5 words."""
    m = TAKE_ME.match(_words(text))
    if m is None:
        return None
    place = re.sub(r" please$", "", m.group(1)).strip()
    return place if 1 <= len(place.split()) <= 5 else None


def asked_to_show(text: str) -> str | None:
    """The thing a sentence asks to see ("show me a bicycle" -> "bicycle"), or None. Nothing on the table is needed for
    that, so it goes to the model without a look."""
    m = SHOW_ME.match(_words(text))
    if m is None:
        return None
    thing = re.sub(r" (?:in 3d|in three d|in three dimensions|as a model|as a 3d model|please)$", "", m.group(1)).strip()
    words = thing.split()
    if not 1 <= len(words) <= 4 or words[0] in NOT_A_THING or "how" in words:
        return None
    return thing


def _clean(text, limit: int) -> str:
    return " ".join(str(text or "").split())[:limit].strip()


def _name_key(name: str) -> str:
    """A thing's name, for comparing: "the 3 eggs" -> "egg"."""
    words = (w for w in _words(name).split() if w not in ("a", "an", "the", "of", "some", "my") and not w.isdigit())
    return " ".join(w[:-1] if len(w) > 3 and w.endswith("s") and not w.endswith("ss") else w for w in words)


def name_match(a: str, b: str) -> float:
    """How much two names of a thing agree, 0 to 1: their shared words over the shorter name's."""
    x, y = set(_name_key(a).split()), set(_name_key(b).split())
    return len(x & y) / min(len(x), len(y)) if x and y else 0.0


# ---------------------------------------------------------------------------------------------------- model calls

TABLE_INTRO = ("You are Lightwork, a helper on a work table. A small projector draws on the table and a camera looks "
               "straight down at it. The image is the camera's view of the table, turned so the person at the table "
               "reads it upright. Ignore thin lines of light along its edges (the projector's border).")
BOX_RULE = ("Boxes are on a 0-1000 grid of this image, x to the right and y down: tight around the thing, whole of it "
            "in view.")
SEE_PROMPT = TABLE_INTRO + """
The person said: "{request}"{history}
Work out what they want and what is on the table for it. Reply with JSON:
- "intent": "job" if they want to do or make something at the table and need steps. "question" if they want to know or \
understand something, about what lies on the table or about anything else. "model" if they want to see a thing in 3D \
("show me a bicycle"). "place" if they want to see a place on a map ("take me to Paris"). "game" if they want to play \
one of the table's games.
- "game": if they want to play one of the table's games, its key: "catch" (catch the light), "follow" (follow the \
lights), "draw" (draw and guess), "rps" (rock paper scissors). Otherwise "none".
- "job": what they want to do in 1 to 3 words ("cottage cheese pancakes", "lego car"), "" if you cannot tell yet.
- "things": what is on the table that matters for this job, at most 16 entries. One entry per separate object, each \
with its own box, even when they are identical: three eggs are three entries named "egg". A short plain "name" a \
camera could check ("bowl of cottage cheese", "egg", "red 2x4 brick"). A manual page lying there is an entry too, \
named "manual". """ + BOX_RULE + """
- "manual": if a manual page, instructions or a picture of the finished thing lies on the table, what it shows, in at \
most 60 words; otherwise "".
- "question": for intent "job" only, otherwise "". One short question for what you need to know and the picture \
cannot tell you. You can count, so never \
ask how many eggs, bricks or tools there are. You cannot weigh, so when the job depends on how much soft food, powder \
or liquid is in a bowl or jar, ask for it ("How many grams of cottage cheese is it?") unless they already told you. \
If they want to build or make something and neither their words nor the table show what, ask ("What do you want to \
build?"). But when they leave the choice to you ("cook something with this"), choose what the table suggests yourself \
and put it in "job": do not ask what to make. Never ask about taste or preferences: pick a sensible default. Never ask \
again what they already answered. "" when you know enough to plan.
- "say": one short sentence to say out loud, at most 14 words, naming what you see ("I see a bowl of cottage cheese \
and three eggs."). Not the question.
- "answer": for intent "question", the answer to say out loud: plain, friendly, at most 45 words. If it is about what \
lies on the table, go by what you can really read there: this camera can read a cover, headings and big diagrams, not \
small print, so say so rather than guess at a page. Otherwise "".
- "notes": for intent "question", 1 or 2 key facts to show on the table, each at most 22 characters. Otherwise [].
- "picture": for intent "question", if a picture would make the answer clearer: one sentence for an illustrator, what \
to draw and which parts to label ("A labelled diagram of a plant cell: nucleus, chloroplast, vacuole, cell wall."). "" \
if no picture would help.
- "subject": for intent "model", the thing to show in 1 to 3 words ("bicycle"). For intent "place", the place as it is \
on a map ("Paris"). Otherwise "".
Answer in English."""
SEE_SCHEMA = {
    "type": "object",
    "properties": {
        "intent": {"type": "string", "enum": ["job", "question", "model", "place", "game"]},
        "game": {"type": "string", "enum": ["none", "catch", "follow", "draw", "rps"]},
        "job": {"type": "string"},
        "things": {"type": "array", "items": {
            "type": "object",
            "properties": {"name": {"type": "string"}, "x_min": {"type": "integer"}, "y_min": {"type": "integer"},
                           "x_max": {"type": "integer"}, "y_max": {"type": "integer"}},
            "required": ["name", "x_min", "y_min", "x_max", "y_max"],
            "additionalProperties": False,
        }},
        "manual": {"type": "string"},
        "question": {"type": "string"},
        "say": {"type": "string"},
        "answer": {"type": "string"},
        "notes": {"type": "array", "items": {"type": "string"}},
        "picture": {"type": "string"},
        "subject": {"type": "string"},
    },
    "required": ["intent", "game", "job", "things", "manual", "question", "say", "answer", "notes", "picture", "subject"],
    "additionalProperties": False,
}
PLAN_PROMPT = TABLE_INTRO + """
The person said: "{request}"{history}
The job: {job}.
On the table: {things}.{manual}
Write the job as steps the table walks them through, using what you can see and what they told you. Rules:
- At most 8 steps, in order, one action each. If the job needs something that is not on the table, the first step is \
to fetch it, with the amounts ("Get 3 tablespoons of flour, some sugar and a fork").
- If a manual or a picture lies on the table, follow it, in its order.
- "title": the job in 1 to 3 words, the same job as above.
- "say": a natural spoken instruction of at most 12 words, a full sentence as a person would say it, with the amount \
in it: "Crack the three eggs into the bowl", "Put the red brick on the green plate" (not "the one red brick").
- "parts": which of the things on the table this step uses, each name once, exactly as listed above and without the \
number: what gets picked \
up first, where it goes last. The table outlines them and draws an arrow from the first to the last. Nearly every step \
uses something on the table, so leave it [] only when none is touched.
- "kind": "do" for most steps. "add" only when something goes in spoon by spoon or scoop by scoop: "count" is how many, \
and the table counts the trips to the last part. "timer" for a short wait they stay for, like mixing: "seconds" is how \
long, and the job waits. "background" for what runs by itself while they get on with something else, like the oven, a \
pot simmering or dough resting: "seconds" is how long, "label" one word for it ("oven"). A clock starts when they finish \
that step and stays on the table through the next steps, so put what they can do meanwhile right after it.
- "check": for every "do" step whose result shows from above, a yes/no question for the camera to answer once it is \
done ("Are the eggs out of their shells and in the bowl?", "Is the red brick on the green plate?"); "" only if a \
camera looking down could not tell.
- Cooking on a stove or in an oven happens away from the table: say so and use a "background" step for it ("Put the dish \
in the oven for 30 minutes"). A dish with raw egg or meat is cooked through.
- "count" and "seconds" are 0 when they do not apply, "label" is "" unless the step is "background".
Answer with JSON, in English."""
PLAN_SCHEMA = {
    "type": "object",
    "properties": {
        "title": {"type": "string"},
        "steps": {"type": "array", "items": {
            "type": "object",
            "properties": {
                "kind": {"type": "string", "enum": ["do", "add", "timer", "background"]}, "say": {"type": "string"},
                "parts": {"type": "array", "items": {"type": "string"}}, "count": {"type": "integer"},
                "seconds": {"type": "integer"}, "check": {"type": "string"}, "label": {"type": "string"},
            },
            "required": ["kind", "say", "parts", "count", "seconds", "check", "label"],
            "additionalProperties": False,
        }},
    },
    "required": ["title", "steps"],
    "additionalProperties": False,
}
LOCATE_PROMPT = TABLE_INTRO + """
The job: {title}. The step now: "{say}".
Where are these now: {names}? One JSON entry per separate object you can see, even when they are identical: its \
"name" as written here, "visible": true, and its own box. A thing you cannot see gets one entry with "visible": false \
and a box of all 0. """ + BOX_RULE + """ Reply with {{"found": [...]}}."""
LOCATE_SCHEMA = {
    "type": "object",
    "properties": {"found": {"type": "array", "items": {
        "type": "object",
        "properties": {"name": {"type": "string"}, "visible": {"type": "boolean"}, "x_min": {"type": "integer"},
                       "y_min": {"type": "integer"}, "x_max": {"type": "integer"}, "y_max": {"type": "integer"}},
        "required": ["name", "visible", "x_min", "y_min", "x_max", "y_max"],
        "additionalProperties": False,
    }}},
    "required": ["found"],
    "additionalProperties": False,
}
CHECK_PROMPT = TABLE_INTRO + """
The job: {title}. The step they just did: "{say}".
Look at the image and answer: {question}
Reply with JSON: {{"ok": true or false, "say": "one short sentence, at most 14 words, to say out loud"}}. ok only if the \
image clearly shows it. If not, say what is still missing or what you would need to see."""
CHECK_SCHEMA = {
    "type": "object",
    "properties": {"ok": {"type": "boolean"}, "say": {"type": "string"}},
    "required": ["ok", "say"],
    "additionalProperties": False,
}
MODEL_PROMPT = ("Draw a {what} as a 3D wireframe a small projector can show on a table and a person can turn with their "
                "fingers. Reply with JSON: \"name\", and \"lines\": a list of polylines, each a list of [x, y, z] points, "
                "all within -1 to 1, z up. Use 20 to 60 polylines and at most 400 points in all, so the shape is clearly "
                "recognisable from any side: curves as many short segments, a full circle at least 24. No text.")
MODEL_SCHEMA = {
    "type": "object",
    "properties": {"name": {"type": "string"}, "lines": {"type": "array", "items": {
        "type": "array", "items": {"type": "array", "items": {"type": "number"}}}}},
    "required": ["name", "lines"],
    "additionalProperties": False,
}
PLACE_PROMPT = ("Where is {place}? Reply with JSON for a 3D map that flies there: \"name\": its name in 1 to 3 words; \"lat\" "
                "and \"lon\": in degrees, the most recognisable spot of it, where its famous or tall buildings stand; \"zoom\": "
                "15 for a city or a landmark, down to 11 for a big region; \"bearing\": the compass direction in degrees the view "
                "should face for the best look; \"say\": one sentence to say out loud on arrival, a fact worth knowing, at most "
                "14 words. Answer in English.")
PLACE_SCHEMA = {
    "type": "object",
    "properties": {"name": {"type": "string"}, "lat": {"type": "number"}, "lon": {"type": "number"},
                   "zoom": {"type": "number"}, "bearing": {"type": "number"}, "say": {"type": "string"}},
    "required": ["name", "lat", "lon", "zoom", "bearing", "say"],
    "additionalProperties": False,
}
ANSWER_PROMPT = TABLE_INTRO + """
The job: {title}. The step now: "{say}".
The person asked: "{question}"
Reply with JSON: {{"say": "the answer in one short sentence, at most 20 words, to say out loud"}}."""
ANSWER_SCHEMA = {
    "type": "object",
    "properties": {"say": {"type": "string"}},
    "required": ["say"],
    "additionalProperties": False,
}


@dataclass(frozen=True)
class Ask:
    """One model call the helper wants made."""

    role: str  # see | plan | locate | check | answer | model | place
    prompt: str
    schema: dict
    look: bool = True  # a fresh clean look at the table first; False: the last one will do
    effort: str = "low"
    max_out: int = 4000
    timeout_s: float = 60.0
    epoch: int = 0  # the job or step it belongs to: an answer from an earlier one is dropped
    subject: str = ""  # a model: what it is of, the key it is cached under


# ---------------------------------------------------------------------------------------------------- the job


@dataclass
class Thing:
    """One object on the table, as the model sees it (design px)."""

    uid: int
    name: str
    box: tuple[float, float, float, float]
    outline: list | None = None  # SAM's, once it answered
    asked: bool = False  # SAM was asked for this position (set by the session): not again unless it moves

    @property
    def centre(self) -> tuple[float, float]:
        return (self.box[0] + self.box[2]) / 2, (self.box[1] + self.box[3]) / 2


@dataclass(frozen=True)
class Step:
    kind: str  # do | add | timer | background
    say: str
    parts: tuple[str, ...] = ()
    count: int = 0
    seconds: float = 0.0
    check: str = ""
    label: str = ""  # background: what the clock is for


@dataclass(frozen=True)
class Job:
    title: str
    steps: tuple[Step, ...]


@dataclass(frozen=True)
class Wire:
    """A 3D wireframe from the model: polylines of x, y, z, centred and within -1 to 1, z up."""

    name: str
    lines: tuple[np.ndarray, ...]

    @property
    def points(self) -> int:
        return sum(len(line) for line in self.lines)


def parse_wire(data, subject: str = "") -> Wire | None:
    """A model's wireframe, centred and scaled to -1..1 and thinned to what a scene can carry; None when it is no
    drawing: too few usable lines, or everything in one spot."""
    if not isinstance(data, dict) or not isinstance(data.get("lines"), list):
        return None
    lines = []
    for raw in data["lines"][:MAX_WIRE_LINES]:
        if not isinstance(raw, list):
            continue
        pts = []
        for p in raw:
            if isinstance(p, (list, tuple)) and len(p) >= 3 and all(
                    isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) for v in p[:3]):
                pts.append([float(v) for v in p[:3]])
        if len(pts) >= 2:
            lines.append(np.asarray(pts, dtype=np.float64))
    if len(lines) < 3:
        return None
    every = np.concatenate(lines)
    low, high = every.min(axis=0), every.max(axis=0)
    size = float((high - low).max())
    if size < 1e-6:
        return None
    centre, scale = (low + high) / 2, 2.0 / size
    lines = [(line - centre) * scale for line in lines]
    total = sum(len(line) for line in lines)
    if total > MAX_WIRE_POINTS:
        keep = MAX_WIRE_POINTS / total
        lines = [line[np.unique(np.linspace(0, len(line) - 1, max(2, round(len(line) * keep))).round().astype(int))]
                 for line in lines]
    name = _clean(data.get("name"), 28) or subject or "model"
    return Wire(name=name, lines=tuple(lines))


def project(line: np.ndarray, yaw: float, tilt: float, zoom: float, centre=MODEL_C, perspective: float = 3.5
            ) -> tuple[np.ndarray, float]:
    """A wireframe line on the table, seen turned by ``yaw`` about its upright and from ``tilt`` above: design px, and how
    far back it lies on average (> 0: behind the middle)."""
    cy, sy = math.cos(yaw), math.sin(yaw)
    ct, st = math.cos(tilt), math.sin(tilt)
    x = line[:, 0] * cy - line[:, 1] * sy
    y = line[:, 0] * sy + line[:, 1] * cy
    up = y * st + line[:, 2] * ct
    depth = y * ct - line[:, 2] * st
    f = perspective / (perspective + depth)
    return np.column_stack([centre[0] + MODEL_R * zoom * f * x, centre[1] - MODEL_R * zoom * f * up]), float(depth.mean())


@dataclass(frozen=True)
class Place:
    name: str
    lon: float
    lat: float
    zoom: float
    bearing: float
    say: str


def parse_place(data, subject: str = "") -> Place | None:
    """A model's place; None when it is not somewhere on the map."""
    if not isinstance(data, dict):
        return None
    try:
        lat, lon = float(data["lat"]), float(data["lon"])
        zoom, bearing = float(data.get("zoom", 15)), float(data.get("bearing", 0))
    except (KeyError, TypeError, ValueError):
        return None
    if not (math.isfinite(lat) and math.isfinite(lon) and -85 <= lat <= 85 and -180 <= lon <= 180):
        return None
    zoom = min(max(zoom, PLACE_ZOOM[0]), PLACE_ZOOM[1]) if math.isfinite(zoom) else 15.0
    bearing = bearing % 360 if math.isfinite(bearing) else 0.0
    return Place(_clean(data.get("name"), 28) or subject, lon, lat, zoom, bearing, _clean(data.get("say"), 140))


def _to_world(lon: float, lat: float) -> tuple[float, float]:
    """Web Mercator, 0 to 1 across the world, y down."""
    s = math.sin(math.radians(lat))
    return (lon + 180.0) / 360.0, 0.5 - math.log((1 + s) / (1 - s)) / (4 * math.pi)


def _from_world(x: float, y: float) -> tuple[float, float]:
    lat = math.degrees(2 * math.atan(math.exp((0.5 - y) * 2 * math.pi)) - math.pi / 2)
    return ((x % 1.0) * 360.0 - 180.0, min(max(lat, -85.0), 85.0))


def box_of(o: dict) -> tuple[float, float, float, float] | None:
    """A model's box on the 0-1000 grid as design px; None when it is unusable (no area, or most of the table)."""
    try:
        xs = sorted(min(max(float(o[k]), 0.0), 1000.0) for k in ("x_min", "x_max"))
        ys = sorted(min(max(float(o[k]), 0.0), 1000.0) for k in ("y_min", "y_max"))
    except (KeyError, TypeError, ValueError):
        return None
    x0, x1 = xs[0] * DESIGN_W / 1000, xs[1] * DESIGN_W / 1000
    y0, y1 = ys[0] * DESIGN_H / 1000, ys[1] * DESIGN_H / 1000
    if x1 - x0 < 6 or y1 - y0 < 6 or (x1 - x0) * (y1 - y0) > 0.6 * DESIGN_W * DESIGN_H:
        return None
    return x0, y0, x1, y1


def parse_job(data) -> Job | None:
    """A model's plan as a :class:`Job`; None when it has no usable step. Unknown kinds are plain steps, an "add" with
    no count or a "timer" with no time too: never a step that cannot finish."""
    if not isinstance(data, dict):
        return None
    steps = []
    for raw in data.get("steps") or []:
        if not isinstance(raw, dict) or len(steps) >= MAX_STEPS:
            continue
        say = re.sub(r"\b([Tt]he) one\b", r"\1", _clean(raw.get("say"), 90)).rstrip(". ")  # a manual's "1x" read aloud
        if not say:
            continue
        kind = str(raw.get("kind") or "do").lower()
        count = raw.get("count") if isinstance(raw.get("count"), int) and not isinstance(raw.get("count"), bool) else 0
        seconds = raw.get("seconds")
        seconds = float(seconds) if isinstance(seconds, (int, float)) and not isinstance(seconds, bool) else 0.0
        if kind not in ("do", "add", "timer", "background") or (kind == "add" and not 1 <= count <= 40) or (
                kind == "timer" and not 0 < seconds <= 3600) or (kind == "background" and not 0 < seconds <= 24 * 3600):
            kind = "do"
        parts = tuple(_clean(p, 40) for p in (raw.get("parts") or []) if _clean(p, 40))[:4]
        steps.append(Step(kind=kind, say=say, parts=parts, count=count if kind == "add" else 0,
                          seconds=seconds if kind in ("timer", "background") else 0.0,
                          check=_clean(raw.get("check"), 200) if kind == "do" else "",
                          label=_clean(raw.get("label"), 12) if kind == "background" else ""))
    if not steps:
        return None
    return Job(title=_clean(data.get("title"), 28) or "Your job", steps=tuple(steps))


# ---------------------------------------------------------------------------------------------------- layout


def _overlap(a, b) -> float:
    return max(0.0, min(a[2], b[2]) - max(a[0], b[0])) * max(0.0, min(a[3], b[3]) - max(a[1], b[1]))


def free_spot(w: float, h: float, taken: list, area=WORK) -> tuple[float, float]:
    """Top-left of a w x h block in ``area`` that covers as little of ``taken`` as it can, as far from the person (the
    top of the page: arms come from the bottom edge) and as far left as that allows."""
    x0, y0, x1, y1 = area
    best, best_cost = (x0, y0), math.inf
    for y in np.arange(y0, max(y0 + 1.0, y1 - h + 1.0), 12.0):
        for x in np.arange(x0, max(x0 + 1.0, x1 - w + 1.0), 16.0):
            rect = (x, y, x + w, y + h)
            cost = sum(_overlap(rect, (t[0] - 8, t[1] - 8, t[2] + 8, t[3] + 8)) for t in taken)
            if cost < best_cost - 1e-6:
                best, best_cost = (float(x), float(y)), cost
    return best


def label_spot(box, w: float, h: float, taken: list, area=WORK) -> tuple[float, float, float, float]:
    """A w x h label beside ``box``: above, below, right or left, the first that covers nothing already placed, else the
    one that covers least; always in ``area``."""
    x0, y0, x1, y1 = box
    gap = 8.0
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    best, best_cost = None, math.inf
    for lx, ly in ((cx - w / 2, y0 - gap - h), (cx - w / 2, y1 + gap), (x1 + gap, cy - h / 2), (x0 - gap - w, cy - h / 2)):
        lx = min(max(lx, area[0]), max(area[0], area[2] - w))
        ly = min(max(ly, area[1]), max(area[1], area[3] - h))
        rect = (lx, ly, lx + w, ly + h)
        cost = sum(_overlap(rect, t) for t in taken)
        if cost < best_cost:
            best, best_cost = rect, cost
        if cost == 0:
            break
    return best


def _with_article(thing: str) -> str:
    """ "a bicycle", "an egg", and as it is for what takes none ("the Eiffel Tower", "scissors")."""
    t = thing.strip()
    low = t.lower()
    if low.startswith(("a ", "an ", "the ", "some ")) or (low.endswith("s") and not low.endswith("ss")):
        return t
    if t[:1].isupper() or low in ("eiffel tower", "golden gate bridge", "colosseum", "taj mahal", "pyramid of giza"):
        return f"the {t}"
    return f"{'an' if low[:1] in 'aeiou' else 'a'} {t}"


def _timer_name(label: str) -> str:
    return re.sub(r"[^A-Z0-9]+", " ", label.upper()).strip().split(" ")[0] if label else ""


def _union(things: list[Thing]) -> tuple[float, float, float, float]:
    return (min(t.box[0] for t in things), min(t.box[1] for t in things), max(t.box[2] for t in things),
            max(t.box[3] for t in things))


def _ellipse(box, n: int = 32) -> list[tuple[float, float]]:
    x0, y0, x1, y1 = box
    cx, cy, rx, ry = (x0 + x1) / 2, (y0 + y1) / 2, (x1 - x0) / 2 + OUTLINE_PAD, (y1 - y0) / 2 + OUTLINE_PAD
    return [(cx + rx * math.cos(a), cy + ry * math.sin(a)) for a in np.linspace(0, 2 * math.pi, n, endpoint=False)]


def poly_area(pts) -> float:
    p = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
    x, y = p[:, 0], p[:, 1]
    return 0.5 * abs(float(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1))))


def fmt_seconds(seconds: float) -> str:
    s = int(math.ceil(max(seconds, 0.0)))
    return f"{s // 60}:{s % 60:02d}" if s >= 60 else str(s)


# ---------------------------------------------------------------------------------------------------- the tile


class Helper(Game):
    """idle -> (mic) -> seeing -> [asking -> (mic) -> seeing ...] -> planning -> steps -> done, or seeing -> answer (a
    question: said, its notes on the table, a picture when Muse Image has drawn it). See the module doc."""

    key = "helper"

    def __init__(self, now, rng, *, listens: bool = True, thinks: bool = True, timers: Timers | None = None):
        super().__init__(now, rng)
        self.listens, self.thinks = listens, thinks
        self.timers = timers if timers is not None else Timers()
        self._clocks_started: set[int] = set()  # this job's background steps whose clock has started
        self.state = "idle"  # idle | seeing | asking | planning | steps | done | answer | modelling | model | placing | map
        self.want: Ask | None = None  # the session makes the call and answers with answered()
        self.want_t = now
        self.want_outlines: list[Thing] = []
        self.start_game: str | None = None  # a tile asked for: the session starts it
        self.hush = False  # stop what the phone is saying: the person is about to talk, or moved on
        self.log: list[str] = []
        self.epoch = 0
        self.last = now
        # the mic
        self.mic = Bubble("mic", "HOLD\nTO TALK", MIC_C, BUBBLE_R)
        self.mic_open = False
        self._mic_on = 0.0  # how long a fingertip has been on it
        self._mic_touch = -1e9
        self.opened_t = -1e9  # when the mic last opened
        self.released_t: float | None = None  # let go, and no words arrived yet
        self._dealt_with = False  # words of this hold were acted on already
        self.utter: list[str] = []  # transcripts of what is being said
        self.utter_t = -1e9
        # the job
        self.request = ""
        self.qa: list[tuple[str, str]] = []
        self.question = ""
        self.job_name = ""
        self.manual = ""
        self.things: list[Thing] = []
        self._uid = UID_BASE
        self.job: Job | None = None
        self.k = 0
        self.parts: list[list[Thing]] = []  # what this step uses: part by part, the objects of each
        self.reply = ""  # the model's last word on this step
        self.reply_role = "info"
        # a 3D model
        self.subject = ""
        self.wire: Wire | None = None
        self.yaw, self.tilt, self.zoom, self.spin = 0.6, 0.4, 1.0, IDLE_SPIN
        self.pens: list = []  # this frame's
        self._drag: tuple[int, tuple[float, float]] | None = None  # (hand, where it was last frame)
        self._zoom_from: tuple[float, float] | None = None  # (distance between the hands, zoom) when both pinched
        # a 3D map
        self.map_place: Place | None = None
        self.map_at = (0.0, 20.0)  # lon, lat of its centre
        self.map_zoom, self.map_bearing, self.map_pitch = 2.0, 0.0, 0.0
        self.flight = 0
        self._fly_until = -1e9
        self._map_from: tuple[float, float, float, float] | None = None  # distance and angle of both hands, zoom, bearing
        # a question's answer
        self.notes: list[str] = []
        self.want_picture: str | None = None  # the session has Muse Image draw it and answers with pictured()
        self.painting = False
        self.picture = None  # lightwork.painter.Painted
        self._pic_rect: tuple[float, float, float, float] | None = None
        self.next = Bubble("next", "NEXT", NEXT_C, BUBBLE_R)
        self.back = Bubble("back", "BACK", BACK_C, BUBBLE_R)
        self._spent: set[str] = set()  # a button that fired waits for the finger to leave
        self.advance_at: float | None = None
        self.timer_end: float | None = None
        self.trips = 0
        self._inside = False
        self._left_t = -1e9
        self._trip_t = -1e9
        self.worked = False  # a hand was at work during this step
        self.checks = 0
        self.checking = False
        self.hand_t = -1e9  # last time a hand was on the page
        # the layout
        self._dirty = True
        self._block: tuple[float, float] = (WORK[0], WORK[1])
        self._labels: list[tuple[int, tuple[float, float], str]] = []  # (first thing's uid, where, text)
        self._taken: list = []  # what the digits of a step must keep clear of

    # ---- per frame

    def update(self, now, points, tips, segments=(), pens=()):
        dt = min(max(now - self.last, 0.0), 0.1)
        self.last = now
        if points:
            self.hand_t = now
        self.pens = list(pens)
        self._mic_step(now, tips, dt)
        self._utterance(now)
        if self.state == "steps":
            self._step(now, points, tips, dt)
        elif self.state == "model":
            self._turn(pens, dt)
        elif self.state == "map":
            self._move_map(pens, now, dt)

    def _mic_step(self, now: float, tips: list, dt: float) -> None:
        if not self.listens:
            return
        if _near(tips, self.mic.c, self.mic.r + 10):
            self._mic_on += dt
            self._mic_touch = now
        else:
            self._mic_on = 0.0
        if not self.mic_open and self._mic_on >= MIC_OPEN_S:
            self.mic_open, self.hush, self.released_t, self._dealt_with = True, True, None, False
            self.opened_t = now
            self.utter = []
            self.sfx.append("listen")
            self.log.append("helper: mic open")
        elif self.mic_open and now - self._mic_touch > MIC_TAIL_S:
            self.mic_open = False
            self.released_t = None if self._dealt_with else now
            self.sfx.append("press")
            self.log.append("helper: mic closed")

    def dealt_with(self) -> None:
        """What was said has been acted on elsewhere ("next", a timer): nothing more to wait for, nothing to apologise
        for after letting go."""
        self.utter, self.released_t, self._dealt_with = [], None, True

    def heard(self, text: str, now: float) -> None:
        """A finished sentence from the transcriber (the session has already taken games and commands out)."""
        self.utter.append(text)
        self.utter_t = now
        self.released_t = None if self.released_t is None else max(self.released_t, now - UTTER_END_S)

    def _utterance(self, now: float) -> None:
        if not self.utter:
            if self.released_t is not None and now - self.released_t > NO_WORDS_S:
                self.released_t = None
                self.say.append("Sorry, I didn't catch that. Hold the microphone and try again.")
                self.sfx.append("error")
            return
        if self.mic_open or now - self.utter_t < UTTER_END_S:
            return
        if self.released_t is not None and now - self.released_t < UTTER_END_S:
            return
        text, self.utter, self.released_t = " ".join(self.utter), [], None
        self.log.append(f"helper: request {text!r} in state {self.state}")
        self._said(text, now)

    def _said(self, text: str, now: float) -> None:
        if self.state == "asking":
            self.qa.append((self.question, text))
            self._see(now)
        elif self.state in ("seeing", "planning"):  # more words while it thinks: they count, it starts over
            self.request = _clean(f"{self.request}. {text}", 400)
            self._see(now)
        elif self.state == "steps" and not any(w in _words(text) for w in NEW_JOB_WORDS):
            self._ask_about_step(text, now)
        else:
            self._new_job(text, now)

    def _ask(self, ask: Ask, now: float) -> None:
        self.want, self.want_t = ask, now

    def go_to(self, place: str, now: float) -> None:
        """Have the model find a place on the map."""
        self.epoch += 1
        self.request, self.qa, self.manual, self.job_name, self.job, self.wire = "", [], "", "", None, None
        self.parts, self.reply, self.things, self.want_outlines, self.map_place = [], "", [], [], None
        self._clear_answer()
        self.subject = _clean(place, 40)
        self.state = "placing"
        self.sfx.append("scan")
        self._dirty = True
        self._ask(Ask("place", PLACE_PROMPT.format(place=self.subject), PLACE_SCHEMA, look=False, max_out=2000,
                      timeout_s=45.0, epoch=self.epoch, subject=f"place {self.subject}"), now)

    def _on_place(self, data: dict, now: float) -> None:
        place = parse_place(data, self.subject)
        if place is None:
            self._failed(Ask("place", "", {}), now)
            return
        self.map_place = place
        self.state = "map"
        self.map_at, self.map_zoom, self.map_bearing, self.map_pitch = (place.lon, place.lat), place.zoom, place.bearing, MAP_PITCH
        self.flight += 1
        self._fly_until = now + FLY_S
        self._drag, self._map_from = None, None
        self.log.append(f"helper: flying to {place.name}: {place.lat:.4f}, {place.lon:.4f}, zoom {place.zoom:.1f}")
        self.say.append(place.say or f"Here's {place.name}.")
        self.sfx.append("answer")
        self._dirty = True

    def _move_map(self, pens, now: float, dt: float) -> None:
        """One pinched hand drags the map with it, two turn and zoom it; left alone the view slowly turns. Nothing moves
        it during the flight: it would stop the flight dead."""
        if now < self._fly_until:
            return
        down = [p for p in pens if p.down and p.at[0] < STRIP_X]
        if len(down) >= 2:
            a, b = down[0].at, down[1].at
            d = max(math.dist(a, b), 20.0)
            angle = math.degrees(math.atan2(b[1] - a[1], b[0] - a[0]))
            if self._map_from is None:
                self._map_from = (d, angle, self.map_zoom, self.map_bearing)
            d0, a0, z0, b0 = self._map_from
            self.map_zoom = min(max(z0 + math.log2(d / d0), PLACE_ZOOM[0]), PLACE_ZOOM[1])
            self.map_bearing = (b0 - (angle - a0)) % 360  # the map turns with the hands
            self._drag = None
            return
        self._map_from = None
        if not down:
            self._drag = None
            self.map_bearing = (self.map_bearing + ORBIT_DEG_S * dt) % 360
            return
        p = down[0]
        if self._drag is not None and self._drag[0] == p.hand:
            dx, dy = p.at[0] - self._drag[1][0], p.at[1] - self._drag[1][1]
            if math.hypot(dx, dy) <= DRAG_JUMP:
                self._pan(dx, dy)
        self._drag = (p.hand, p.at)

    def _pan(self, dx: float, dy: float) -> None:
        """The ground under the finger stays under it: with the view pitched, a px up the table is more ground."""
        b = math.radians(self.map_bearing)
        dy = dy / max(math.cos(math.radians(self.map_pitch)), 0.35)
        east = -(dx * math.cos(b) - dy * math.sin(b))
        north = dx * math.sin(b) + dy * math.cos(b)
        world = TILE_PX * 2 ** self.map_zoom
        x, y = _to_world(*self.map_at)
        self.map_at = _from_world(x + east / world, y - north / world)

    def show_model(self, subject: str, now: float) -> None:
        """Have the model draw a thing as lines (or, for one drawn before, the session answers from its cache)."""
        self.epoch += 1
        self.request, self.qa, self.manual, self.job_name, self.job, self.wire = "", [], "", "", None, None
        self.parts, self.reply, self.things, self.want_outlines = [], "", [], []
        self._clear_answer()
        self.subject = _clean(subject, 40)
        self.state = "modelling"
        self.sfx.append("scan")
        self.say.append(f"Let me build {_with_article(self.subject)}.")
        self._dirty = True
        self._ask(Ask("model", MODEL_PROMPT.format(what=self.subject), MODEL_SCHEMA, look=False, max_out=20000,
                      timeout_s=120.0, epoch=self.epoch, subject=self.subject), now)

    def _new_job(self, text: str, now: float) -> None:
        self.request, self.qa, self.manual, self.job_name, self.job = _clean(text, 400), [], "", "", None
        self.parts, self.reply = [], ""
        self._clear_answer()
        self._clocks_started = set()
        self._see(now)

    def _history(self) -> str:
        return "".join(f'\nYou asked: "{q}" They answered: "{a}"' for q, a in self.qa)

    def _see(self, now: float) -> None:
        if not self.thinks:
            self.state = "idle"
            self.say.append("I can only start games: I have no model to look with.")
            self.sfx.append("error")
            return
        self.epoch += 1
        self.state = "seeing"
        self.sfx.append("scan")
        self._dirty = True
        self._ask(Ask("see", SEE_PROMPT.format(request=self.request, history=self._history()), SEE_SCHEMA,
                      max_out=4000, timeout_s=45.0, epoch=self.epoch), now)

    def _names(self) -> list[str]:
        """The things' names, each once, in the model's order."""
        return list(dict.fromkeys(t.name for t in self.things))

    def _count(self, name: str) -> int:
        return sum(_name_key(t.name) == _name_key(name) for t in self.things)

    def _plan(self, now: float) -> None:
        self.epoch += 1
        self.state = "planning"
        self._dirty = True
        names = [f"{k} ({self._count(k)} of them)" if self._count(k) > 1 else k for k in self._names()]
        things = "; ".join(names) or "nothing the camera could name"
        manual = f"\nThe manual or picture on the table shows: {self.manual}" if self.manual else ""
        job = self.job_name or "what they asked for"
        self._ask(Ask("plan", PLAN_PROMPT.format(request=self.request, history=self._history(), job=job, things=things,
                                                 manual=manual), PLAN_SCHEMA, look=False, max_out=6000, timeout_s=90.0,
                      epoch=self.epoch), now)

    # ---- answers

    def answered(self, ask: Ask, data, err: str | None, now: float, *, by: str = "") -> None:
        """A model call came back: ``data`` its JSON (None when it failed, ``err`` why). Stale ones are dropped."""
        if ask.epoch != self.epoch:
            self.log.append(f"helper: late {ask.role} answer dropped")
            return
        self.log.append(f"helper {ask.role}: {err or data} {by}".rstrip())
        if ask.role == "check":
            self.checking = False
        if err or not isinstance(data, dict):
            self._failed(ask, now)
            return
        getattr(self, f"_on_{ask.role}")(data, now)

    def _failed(self, ask: Ask, now: float) -> None:
        self.sfx.append("error")
        if ask.role == "place":
            self.state = "idle"
            self.say.append(f"Sorry, I couldn't find {self.subject}. Hold the microphone and try again.")
            self._dirty = True
        elif ask.role == "model":
            self.state = "idle"
            self.say.append(f"Sorry, I couldn't build {_with_article(self.subject)}. Hold the microphone and try again.")
            self._dirty = True
        elif ask.role in ("see", "plan"):
            self.state = "asking" if self.qa else "idle"
            self.question = self.question if self.qa else ""
            self.say.append("Sorry, I couldn't work that out. Hold the microphone and say it again.")
            self._dirty = True
        elif ask.role == "check":
            self._set_reply("I COULD NOT LOOK JUST NOW", "warn")
        elif ask.role == "answer":
            self.say.append("Sorry, I couldn't answer that.")
        # a failed locate leaves the parts where they last were

    def _clear_answer(self) -> None:
        self.notes, self.want_picture, self.painting, self.picture, self._pic_rect = [], None, False, None, None

    def _on_question(self, data: dict, now: float) -> None:
        """The answer is said at once, so the wait for the picture is not empty."""
        self._merge_things(data.get("things"), now)
        self._clear_answer()
        self.state = "answer"
        self.say.append(_clean(data.get("answer"), 400))
        self.sfx.append("answer")
        raw = data.get("notes") if isinstance(data.get("notes"), list) else []
        self.notes = [line for n in raw for line in textwrap.wrap(_clean(n, 60).upper(), NOTE_CHARS)][:MAX_NOTE_LINES]
        prompt = _clean(data.get("picture"), 500)
        if prompt:
            self.want_picture, self.painting = prompt, True
        self._dirty = True

    def pictured(self, painted, now: float) -> None:
        """Muse Image is done (``painted`` is None: it failed, the answer stands without one)."""
        self.painting = False
        self.picture = painted
        self.sfx.append("found" if painted is not None else "error")
        self._dirty = True

    def _on_model(self, data: dict, now: float) -> None:
        wire = parse_wire(data, self.subject)
        if wire is None:
            self._failed(Ask("model", "", {}), now)
            return
        self.wire = wire
        self.state = "model"
        self.yaw, self.tilt, self.zoom, self.spin = 0.6, 0.4, 1.0, IDLE_SPIN
        self._drag, self._zoom_from = None, None
        self.log.append(f"helper: {wire.name}: {len(wire.lines)} lines, {wire.points} points")
        self.say.append(f"Here's {_with_article(self.subject)}. Pinch and drag to turn it.")
        self.sfx.append("answer")
        self._dirty = True

    def _turn(self, pens, dt: float) -> None:
        """One hand pinched turns the model, both zoom it; let go and it runs on for a moment, then turns slowly."""
        down = [p for p in pens if p.down and p.at[0] < STRIP_X]
        if len(down) >= 2:
            d = max(math.dist(down[0].at, down[1].at), 20.0)
            if self._zoom_from is None:
                self._zoom_from = (d, self.zoom)
            self.zoom = min(max(self._zoom_from[1] * d / self._zoom_from[0], ZOOM_RANGE[0]), ZOOM_RANGE[1])
            self._drag, self.spin = None, 0.0
            return
        self._zoom_from = None
        if down:
            p = down[0]
            if self._drag is not None and self._drag[0] == p.hand:
                dx, dy = p.at[0] - self._drag[1][0], p.at[1] - self._drag[1][1]
                if math.hypot(dx, dy) <= DRAG_JUMP:
                    self.yaw += dx * TURN_PER_PX
                    self.tilt = min(max(self.tilt + dy * TILT_PER_PX, TILT_RANGE[0]), TILT_RANGE[1])
                    if dt > 0:
                        self.spin = 0.7 * self.spin + 0.3 * dx * TURN_PER_PX / dt
            self._drag = (p.hand, p.at)
            return
        self._drag = None
        self.yaw += self.spin * dt
        self.spin += (IDLE_SPIN - self.spin) * min(dt / SPIN_DECAY_S, 1.0)

    def _on_see(self, data: dict, now: float) -> None:
        game = str(data.get("game") or "none").lower()
        if game in GAME_NAMES:
            self.start_game = game
            return
        intent = str(data.get("intent") or "job").lower()
        if intent == "question" and _clean(data.get("answer"), 400):
            self._on_question(data, now)
            return
        if intent == "model" and _clean(data.get("subject"), 40):
            self.show_model(_clean(data.get("subject"), 40), now)
            return
        if intent == "place" and _clean(data.get("subject"), 40):
            self.go_to(_clean(data.get("subject"), 40), now)
            return
        self.job_name = _clean(data.get("job"), 28) or self.job_name
        self.manual = _clean(data.get("manual"), 400)
        self._merge_things(data.get("things"), now)
        say = _clean(data.get("say"), 120)
        question = _clean(data.get("question"), 120)
        if question and len(self.qa) < MAX_QUESTIONS:
            self.state, self.question = "asking", question
            self.say.append(f"{say} {question}".strip() if not self.qa else question)
            self.sfx.append("answer")
        else:
            self.say.append(f"{say} Let me plan that.".strip() if not self.qa else "Okay, let me plan that.")
            self.sfx.append("found")
            self._plan(now)
        self._dirty = True

    def _merge_things(self, raw, now: float) -> None:
        """A fresh look: what it lists is what is there. A thing that stayed keeps its number and its outline."""
        self.things = self._updated(self.things, raw)
        self.want_outlines = [t for t in self.things if t.outline is None and not t.asked]

    def _updated(self, old: list[Thing], raw) -> list[Thing]:
        """``raw`` (model entries) as things: each takes over the nearest old thing of its name, if there is one."""
        pool = list(old)
        out: list[Thing] = []
        for o in (raw if isinstance(raw, list) else [])[:MAX_THINGS]:
            if not isinstance(o, dict) or o.get("visible") is False:
                continue
            box = box_of(o)
            name = _clean(o.get("name"), 40).lower()
            if box is None or not name:
                continue
            centre = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
            same = [t for t in pool if _name_key(t.name) == _name_key(name)]
            before = min(same, key=lambda t: math.dist(t.centre, centre), default=None)
            if before is None:
                self._uid += 1
                out.append(Thing(self._uid, name, box))
                continue
            pool.remove(before)
            if math.dist(before.centre, centre) > OUTLINE_MOVE:  # a model's boxes vary: only a real move redraws it
                before.box, before.outline, before.asked = box, None, False
            out.append(before)
        return out

    def place(self, t: Thing, box) -> None:
        """Where a thing really is, found in the picture (the session's ``snap_box``)."""
        t.box = tuple(float(v) for v in box)
        self._dirty = True

    def outlined(self, uid: int, pts) -> None:
        """SAM's outline of a thing (design px, or None: no usable mask, it keeps the box's). It must fit the box asked
        for: a mask of the neighbour or the shadow is worse than an ellipse."""
        t = next((t for t in self.things if t.uid == uid), None)
        if t is None or pts is None or len(pts) < 3:
            return
        x0, y0, x1, y1 = t.box
        w, h = x1 - x0, y1 - y0
        cx, cy = np.asarray(pts, dtype=np.float64).reshape(-1, 2).mean(axis=0)
        grow = max(0.3 * max(w, h), 20.0)  # a small thing's box can be off by more than its own size
        drawn = (w + 2 * OUTLINE_PAD) * (h + 2 * OUTLINE_PAD)  # the outline is pushed out from the object like the box's
        # at most 1.8 x the box, or the box plus 20 px all round (a small thing's box can be smaller than the thing)
        most = max(1.8 * drawn, (w + 40.0) * (h + 40.0))
        if (0.05 * drawn <= poly_area(pts) <= most and x0 - grow <= cx <= x1 + grow
                and y0 - grow <= cy <= y1 + grow):
            t.outline = [(float(a), float(b)) for a, b in pts]
            p = np.asarray(t.outline)  # where it really is: a model's box is off by a few per cent of the view
            t.box = (float(p[:, 0].min()), float(p[:, 1].min()), float(p[:, 0].max()), float(p[:, 1].max()))
            self._dirty = True
        else:
            self.log.append(f"helper: SAM outline does not fit {t.name}: area {poly_area(pts):.0f} against {drawn:.0f}, "
                            f"centre {cx:.0f},{cy:.0f} against box {x0:.0f},{y0:.0f},{x1:.0f},{y1:.0f}; kept the box")

    def _on_plan(self, data: dict, now: float) -> None:
        job = parse_job(data)
        if job is None:
            self._failed(Ask("plan", "", {}), now)
            return
        self.job = job
        self.log.append(f"helper: {job.title}: {len(job.steps)} steps")
        self.sfx.append("answer")
        self.say.append(f"{job.title}: {len(job.steps)} step{'' if len(job.steps) == 1 else 's'}.")
        self.state = "steps"
        self._enter(0, now)

    def _on_locate(self, data: dict, now: float) -> None:
        step = self._step_now()
        if step is None:
            return
        found = [o for o in (data.get("found") if isinstance(data.get("found"), list) else []) if isinstance(o, dict)]
        keys = {_name_key(_clean(o.get("name"), 40)) for o in found}
        told = [t for t in self.things if any(self._same(t.name, k) for k in keys)]
        fresh = self._updated(told, found)  # what it found now replaces what it was asked about
        self.things = [t for t in self.things if t not in told] + fresh
        self.want_outlines += [t for t in fresh if t.outline is None and not t.asked and t not in self.want_outlines]
        self._light(step)
        self._dirty = True

    @staticmethod
    def _same(name: str, key: str) -> bool:
        return name_match(name, key) >= 0.5

    def _on_check(self, data: dict, now: float) -> None:
        say = _clean(data.get("say"), 120)
        if data.get("ok") is True:
            self._set_reply(say or "LOOKS RIGHT", "target")
            self.advance_at = now + ADVANCE_S
            self.sfx.append("answer")
        else:
            self._set_reply(say or "NOT YET", "warn")
            self.sfx.append("error")
        if say:
            self.say.append(say)

    def _on_answer(self, data: dict, now: float) -> None:
        say = _clean(data.get("say"), 160)
        if say:
            self._set_reply(say, "info")
            self.say.append(say)
            self.sfx.append("answer")

    def _set_reply(self, text: str, role: str) -> None:
        self.reply, self.reply_role = text, role
        self._dirty = True

    # ---- steps

    def _step_now(self) -> Step | None:
        return self.job.steps[self.k] if self.job is not None and 0 <= self.k < len(self.job.steps) else None

    def _light(self, step: Step) -> None:
        """What this step uses, part by part in its order, among the things seen: all of the best-matching name (all
        three eggs). A name can serve only one part."""
        self.parts = []
        used: set[str] = set()
        for part in step.parts:
            rest = [t for t in self.things if _name_key(t.name) not in used]
            best = max(rest, key=lambda t: name_match(t.name, part), default=None)
            if best is not None and name_match(best.name, part) >= 0.5:
                key = _name_key(best.name)
                used.add(key)
                self.parts.append([t for t in self.things if _name_key(t.name) == key])

    @property
    def lit(self) -> list[Thing]:
        return [t for part in self.parts for t in part]

    @property
    def parts_known(self) -> bool:
        """Every part of this step was found: a look that shows them unmoved needs no model."""
        step = self._step_now()
        return step is not None and len(self.parts) == len(step.parts)

    def _enter(self, k: int, now: float) -> None:
        self.epoch += 1
        self.k = k
        step = self.job.steps[k]
        self.hush = True
        self.reply, self.reply_role = "", "info"
        self.advance_at = None
        self.timer_end = now + step.seconds if step.kind == "timer" else None
        self.trips, self._inside, self._left_t, self._trip_t = 0, False, -1e9, -1e9
        self.worked, self.checks, self.checking = False, 0, False
        self._light(step)
        self.say.append(f"Step {k + 1}. {step.say}.")
        if step.parts and (k > 0 or not self.parts_known):  # things moved, or the plan named more than was seen
            names = "; ".join(step.parts)
            self._ask(Ask("locate", LOCATE_PROMPT.format(title=self.job.title, say=step.say, names=names),
                          LOCATE_SCHEMA, effort="minimal", max_out=2000, timeout_s=30.0, epoch=self.epoch), now)
        self._dirty = True

    def _go(self, k: int, now: float) -> None:
        step = self._step_now()
        if (k > self.k and step is not None and step.kind == "background" and self.k not in self._clocks_started):
            self._clocks_started.add(self.k)  # they say it is in the oven: the clock starts, and they carry on
            self.say.append(self.timers.start_timer(_timer_name(step.label), step.seconds, now))
        if k >= len(self.job.steps):
            self.epoch += 1
            self.state, self.parts, self.reply = "done", [], ""
            self.want = None
            self.hush = True
            self.sfx.append("answer")
            self.say.append(f"{self.job.title}: done. Hold the microphone for the next job.")
            self._dirty = True
            return
        self._enter(max(k, 0), now)

    def _press(self, b: Bubble, tips: list, dt: float) -> bool:
        """A button held for its dwell, once: it waits for the finger to leave before it can fire again."""
        near = _near(tips, b.c, b.r + 6)
        if b.key in self._spent:
            b.dwell = 0.0
            if not near:
                self._spent.discard(b.key)
            return False
        if b.step(tips, dt):
            self._spent.add(b.key)
            self.sfx.append("press")
            return True
        return False

    def _step(self, now: float, points: list, tips: list, dt: float) -> None:
        if self.k > 0 and self._press(self.back, tips, dt):
            self._go(self.k - 1, now)
            return
        if self._press(self.next, tips, dt):
            self._go(self.k + 1, now)
            return
        step = self._step_now()
        if self.advance_at is not None:
            if now >= self.advance_at:
                self._go(self.k + 1, now)
            return
        if step.kind == "timer":
            if now >= self.timer_end:
                self.say.append("Time.")
                self.sfx.append("answer")
                self.advance_at = now + ADVANCE_S
        elif step.kind == "add":
            self._trips(step, now, points)
        elif step.check:
            self._watch(step, now, points)

    def _target(self) -> tuple[float, float, float, float] | None:
        """The box around where this step's things go."""
        return _union(self.parts[-1]) if self.parts else None

    def _trips(self, step: Step, now: float, points: list) -> None:
        """A hand arriving at the last part (or, with none located, on the work area), after it had left: one trip."""
        t = self._target()
        x0, y0, x1, y1 = (t[0] - 18, t[1] - 18, t[2] + 18, t[3] + 18) if t is not None else (
            WORK[0], WORK[1], STRIP_X, WORK[3])
        inside = any(x0 <= p[0] <= x1 and y0 <= p[1] <= y1 for p in points)
        if inside and not self._inside and now - self._left_t >= OUTSIDE_S and now - self._trip_t >= TRIP_GAP_S:
            self.trips += 1
            self._trip_t = now
            self.sfx.append("found")
            if self.trips >= step.count:
                self.say.append(f"{self.trips}. That's all {step.count}.")
                self.advance_at = now + ADVANCE_S
            else:
                self.say.append(f"{self.trips}.")
        if self._inside and not inside:
            self._left_t = now
        self._inside = inside

    def _watch(self, step: Step, now: float, points: list) -> None:
        """Hands at work, then hands away: the camera checks the step."""
        if any(p[0] < STRIP_X for p in points):
            self.worked = True
        if (self.worked and not points and not self.checking and self.want is None and self.checks < MAX_CHECKS
                and now - self.hand_t >= CHECK_AFTER_S):
            self.worked, self.checking = False, True
            self.checks += 1
            self.sfx.append("listen")
            self._ask(Ask("check", CHECK_PROMPT.format(title=self.job.title, say=step.say, question=step.check),
                          CHECK_SCHEMA, max_out=2000, timeout_s=45.0, epoch=self.epoch), now)

    def _ask_about_step(self, text: str, now: float) -> None:
        step = self._step_now()
        self.sfx.append("listen")
        self._ask(Ask("answer", ANSWER_PROMPT.format(title=self.job.title, say=step.say, question=_clean(text, 200)),
                      ANSWER_SCHEMA, max_out=1500, timeout_s=30.0, epoch=self.epoch), now)

    def command(self, verb: str, now: float) -> bool:
        """A spoken command (``tabletop.command_of``). False when it means nothing now: it is then just words."""
        if verb == "stop" and self.state in ("asking", "seeing", "planning", "steps", "answer", "modelling", "model",
                                             "placing", "map"):
            self.epoch += 1
            self.state, self.want, self.job, self.parts, self.reply, self.wire = "idle", None, None, [], "", None
            self.map_place = None
            self._clear_answer()
            self.say.append("Stopped.")
            self._dirty = True
            return True
        if self.state != "steps":
            return False
        if verb == "next":
            self._go(self.k + 1, now)
        elif verb == "back":
            self._go(self.k - 1, now)
        elif verb == "repeat":
            self.say.append(f"Step {self.k + 1}. {self._step_now().say}.")
        elif verb == "undo" and self.trips > 0:
            self.trips -= 1
            self.advance_at = None
            self.say.append(f"Back to {self.trips}.")
        elif verb == "how many" and self._step_now().kind == "add":
            self.say.append(f"{self.trips} of {self._step_now().count}.")
        elif verb == "check" and self._step_now().check and self.want is None and not self.checking:
            self.worked = True
            self.hand_t = -1e9
            self._watch(self._step_now(), now, [])
        else:
            return False
        return True

    # ---- display

    def hud(self, now, best):
        if not self.listens:
            return "VOICE IS OFF: START WITH --VOICE", "GAMES ARE ON THE MENU", "warn"
        if self.mic_open:
            return "LISTENING: SPEAK, THEN LET GO", "", "target"
        if self.released_t is not None or self.utter:
            return "ONE MOMENT...", "", "info"
        title = (self.job.title if self.job is not None else self.job_name or self.request).upper()
        if self.state == "seeing":
            return "LOOKING AT THE TABLE...", title, "warn"
        if self.state == "asking":
            return "ONE QUESTION", "HOLD THE MIC TO ANSWER", "target"
        if self.state == "planning":
            return f"PLANNING: {title}", "ONE MOMENT...", "warn"
        if self.state == "done":
            return f"{title}: DONE", "HOLD THE MIC FOR THE NEXT JOB", "target"
        if self.state == "placing":
            return f"GOING TO {self.subject.upper()}"[:44], "MUSE WONDER IS FINDING IT...", "warn"
        if self.state == "map":
            return (f"{self.map_place.name.upper()} · MAP: OPENFREEMAP, OPENSTREETMAP"[:48],
                    "PINCH AND DRAG. TWO HANDS TURN AND ZOOM", "target")
        if self.state == "modelling":
            return f"BUILDING: {self.subject.upper()}"[:44], "MUSE WONDER IS DRAWING IT...", "warn"
        if self.state == "model":
            return self.wire.name.upper(), "PINCH AND DRAG TO TURN, TWO HANDS TO ZOOM", "target"
        if self.state == "answer":
            return ("DRAWING IT WITH MUSE IMAGE..." if self.painting else "THERE YOU GO"), "HOLD THE MIC TO ASK MORE", "target"
        if self.state == "steps":
            return f"{title} · STEP {self.k + 1} OF {len(self.job.steps)}", self._hint(), "info"
        return "HOLD THE MIC AND TELL ME WHAT TO DO", "A JOB, OR A GAME TO START", "info"

    def _hint(self) -> str:
        step = self._step_now()
        if self.checking:
            return "CHECKING..."
        if self.advance_at is not None:
            return "GOOD"
        if step.kind == "add":
            return f"{self.trips} OF {step.count}: I COUNT EACH TRIP"
        if step.kind == "timer":
            return "HOLD NEXT TO SKIP THE TIMER"
        if step.kind == "background":
            return "HOLD NEXT ONCE IT'S IN: THE CLOCK STARTS"
        if step.check:
            return "HANDS AWAY AND I'LL CHECK, OR HOLD NEXT"
        return "HOLD NEXT WHEN IT'S DONE"

    def _block_lines(self) -> list[tuple[str, str]]:
        """(text, role) lines of the text block."""
        if self.state == "idle":
            return [("HOLD THE MIC AND SAY:", "info"), ("HELP ME BUILD THIS LEGO", "source"),
                    ("WHAT CAN I COOK WITH THIS?", "source"), ("SET A TIMER FOR 10 MINUTES", "source"),
                    ("START ROCK PAPER SCISSORS", "source")]
        if self.state == "asking":
            return [(line, "target") for line in textwrap.wrap(self.question.upper(), CHARS)]
        if self.state == "done":
            return [(f"{self.job.title.upper()}: DONE", "target")]
        if self.state == "answer":
            return [(line, "target") for line in self.notes]
        if self.state == "steps":
            step = self._step_now()
            lines = [(line, "info") for line in textwrap.wrap(step.say.upper(), CHARS)]
            if step.kind == "background" and self.k not in self._clocks_started:
                name = _timer_name(step.label) or "CLOCK"
                lines.append((f"{name} {clock_text(step.seconds)} WHEN YOU HOLD NEXT"[:CHARS], "source"))
            return lines + [(line, self.reply_role) for line in textwrap.wrap(self.reply.upper(), CHARS)[:3]]
        return []

    def _shown(self) -> list[Thing]:
        if self.state in ("seeing", "asking", "planning", "answer"):
            return list(self.things)
        return self.lit if self.state == "steps" else []

    def _groups(self, things: list[Thing]) -> list[list[Thing]]:
        groups: dict[str, list[Thing]] = {}
        for t in things:
            groups.setdefault(_name_key(t.name), []).append(t)
        return list(groups.values())

    def _layout(self) -> None:
        """Where the text block and the labels go: computed when something changes, not per frame, so nothing jumps.
        Identical things share one label: "3 EGGS"."""
        self._dirty = False
        shown = self._shown()
        boxes = [t.box for t in self.things]  # text keeps clear of everything on the table, lit or not ...
        ring = self._ring()
        if ring is not None:  # ... and of the ring a counted or timed step draws around its target
            boxes.append((ring[0] - ring[2], ring[1] - ring[2], ring[0] + ring[2], ring[1] + ring[2]))
        if self.state == "answer" and (self.painting or self.picture is not None):
            if self._pic_rect is None:  # the picture comes first, and stays where it is
                x, y = free_spot(PIC_W, PIC_H, boxes)
                self._pic_rect = (x, y, x + PIC_W, y + PIC_H)
            boxes.append(self._pic_rect)
        lines = self._block_lines()
        self._taken = list(boxes)
        if lines:
            w = TEXT_W * max(len(text) for text, _ in lines) + self._digit_w()
            h = LINE_H * len(lines) + 6
            self._block = free_spot(w, h, boxes)
            self._taken.append((self._block[0], self._block[1], self._block[0] + w, self._block[1] + h))
        self._labels = []
        for group in self._groups([] if self.state == "answer" else shown):  # an answer's outline needs no name
            n = len(group)
            name = plural(group[0].name, n)
            text = _clean(f"{n} {name}" if n > 1 else name, 24).upper()
            r = label_spot(_union(group), TEXT_W * len(text), LINE_H, self._taken)
            self._labels.append((group[0].uid, (r[0], (r[1] + r[3]) / 2), text))
            self._taken.append(r)

    def _digit_w(self) -> float:
        return 18.0 * 1.45 * len(str(self.k + 1)) + 12.0 if self.state == "steps" else 0.0

    def draw(self, c, now):
        if self._dirty:
            self._layout()
        if self.state == "answer":  # first: black in a picture hides what lies under it
            self._draw_picture(c, now)
        c.poly("hp-strip", [(STRIP_X, WORK[1]), (STRIP_X, DESIGN_H - 10)], "info")
        self._draw_mic(c, now)
        if self.state in ("seeing", "modelling", "placing"):
            self._draw_sweep(c, now)
        if self.state == "map":
            c.map_view("hp-map", MAP_RECT, self.map_at, self.map_zoom, self.map_bearing, self.map_pitch, flight=self.flight,
                       fly_ms=int(FLY_S * 1000))
        if self.state == "model":
            self._draw_model(c)
        self._draw_things(c, now)
        self._draw_block(c)
        if self.state == "steps":
            self.next.draw(c, "target")
            self._chevron(c, "hp-next-ico", self.next.c, 1.0, "target")
            if self.k > 0:
                self.back.draw(c, "info")
                self._chevron(c, "hp-back-ico", self.back.c, -1.0, "info")
            self._draw_step(c, now)

    def _draw_model(self, c) -> None:
        """Near lines cyan, far ones blue (dim on the projector): it reads as 3D even when it stands still. Fingertips in
        a pinch are marked: it has you."""
        for i, line in enumerate(self.wire.lines):
            pts, depth = project(line, self.yaw, self.tilt, self.zoom)
            c.poly(f"hp-w{i}", pts, "done" if depth > 0.15 else "source", draw_ms=900)
        for p in self.pens:
            if p.down and p.at[0] < STRIP_X:
                c.dot(f"hp-pinch{p.hand}", p.at, "target")

    def _draw_picture(self, c, now: float) -> None:
        """The picture, or, while Muse Image draws it, the place it will go: its corners and a line going over it."""
        if self._pic_rect is None:
            return
        x0, y0, x1, y1 = self._pic_rect
        if self.picture is not None:
            scale = min((x1 - x0) / self.picture.width, (y1 - y0) / self.picture.height)
            w, h = self.picture.width * scale, self.picture.height * scale
            ox, oy = x0 + ((x1 - x0) - w) / 2, y0 + ((y1 - y0) - h) / 2
            c.picture("hp-pic", self.picture.src, ox, oy, ox + w, oy + h, draw_ms=900)
        elif self.painting:
            arm = 22.0
            for k, (x, y, dx, dy) in enumerate(((x0, y0, 1, 1), (x1, y0, -1, 1), (x1, y1, -1, -1), (x0, y1, 1, -1))):
                c.poly(f"hp-pic-c{k}", [(x + dx * arm, y), (x, y), (x, y + dy * arm)], "source", glow=True)
            f = (now * 0.5) % 2.0
            f = f if f < 1.0 else 2.0 - f
            sweep = y0 + (y1 - y0) * f
            c.poly("hp-pic-sweep", [(x0 + 8, sweep), (x1 - 8, sweep)], "source", glow=True)

    @staticmethod
    def _chevron(c, id: str, at, side: float, role: str) -> None:
        x, y = at
        c.poly(id, [(x - side * 9, y - 16), (x + side * 9, y), (x - side * 9, y + 16)], role)

    def _draw_mic(self, c, now: float) -> None:
        on = self.mic_open
        role = "target" if on else ("info" if self.listens else "error")
        self.mic.draw(c, role)
        x, y = self.mic.c
        mic_icon(c, "hp-mic", x, y, 20.0, role)
        if on:  # rings going out: it hears you
            for k in range(2):
                f = (now * 1.2 + k / 2) % 1.0
                c.circle(f"hp-mic-ring{k}", (x, y), self.mic.r + 4 + 18 * f, "target", n=28)

    @staticmethod
    def _draw_sweep(c, now: float) -> None:
        """A line of light going over the table while the model looks."""
        f = (now * 0.5) % 2.0
        f = f if f < 1.0 else 2.0 - f
        y = WORK[1] + (WORK[3] - WORK[1]) * f
        c.poly("hp-sweep", [(WORK[0], y), (STRIP_X - 10, y)], "source", glow=True)

    def _draw_things(self, c, now: float) -> None:
        """Outlines, white while looking; on a step what to take is cyan, where it goes green, what lies between yellow."""
        roles: dict[int, str] = {}
        if self.state == "steps":
            for i, part in enumerate(self.parts):
                role = "target" if i == len(self.parts) - 1 else ("source" if i == 0 else "warn")
                roles.update({t.uid: role for t in part})
        for t in self._shown():
            role = roles.get(t.uid, "info")
            if t.outline is not None:  # a new prim: SAM's outline draws itself in place of the first one
                c.poly(f"hp-t{t.uid}-s", t.outline, role, closed=True, glow=True, draw_ms=450)
            else:
                c.poly(f"hp-t{t.uid}", _ellipse(t.box), role, closed=True, glow=True, draw_ms=450)
        for uid, at, text in self._labels:
            c.label(f"hp-t{uid}-txt", at, text, "info", type_ms=28 * len(text))
        if self.state == "steps" and len(self.parts) >= 2:
            self._draw_arrow(c, self.parts[0], self.parts[-1], now)

    @staticmethod
    def _draw_arrow(c, sources: list[Thing], targets: list[Thing], now: float) -> None:
        """From what to take (the nearest one) to where it goes, with chevrons running along it."""
        a, b = min(((s, t) for s in sources for t in targets), key=lambda p: math.dist(p[0].centre, p[1].centre))
        ax, ay = a.centre
        bx, by = b.centre
        d = math.dist((ax, ay), (bx, by))
        if d < 1:
            return
        ux, uy = (bx - ax) / d, (by - ay) / d

        def reach(box) -> float:  # from a box's centre to its edge along the arrow, plus the outline's margin
            w, h = (box[2] - box[0]) / 2, (box[3] - box[1]) / 2
            return min(w / max(abs(ux), 1e-6), h / max(abs(uy), 1e-6)) + 12.0

        s0, s1 = reach(a.box), d - reach(b.box)
        if s1 - s0 < 24:  # they nearly touch: the two outlines say it
            return
        c.poly("hp-arrow", [(ax + ux * s0, ay + uy * s0), (ax + ux * s1, ay + uy * s1)], "target")
        n = max(1, min(3, int((s1 - s0) / 60)))
        for k in range(n):
            s = s0 + (s1 - s0) * ((now * 0.6 + k / n) % 1.0)
            x, y = ax + ux * s, ay + uy * s
            c.poly(f"hp-chev{k}", [(x - ux * 10 - uy * 8, y - uy * 10 + ux * 8), (x, y),
                                   (x - ux * 10 + uy * 8, y - uy * 10 - ux * 8)], "target")

    def _draw_block(self, c) -> None:
        lines = self._block_lines()
        if not lines:
            return
        x, y = self._block
        if self.state == "steps":
            c.number("hp-k", str(self.k + 1), x, y + 4, 18.0, "target")
            x += self._digit_w()
        for i, (text, role) in enumerate(lines):
            c.label(f"hp-l{i}", (x, y + LINE_H / 2 + LINE_H * i), text, role, type_ms=22 * len(text))

    def _draw_step(self, c, now: float) -> None:
        step = self._step_now()
        if step.kind == "timer" and self.timer_end is not None:
            left = max(0.0, self.timer_end - now)
            role = "target" if left > 0 else "warn"
            frac, text, id = left / max(step.seconds, 1.0), fmt_seconds(left), "hp-timer"
        elif step.kind == "add":
            role = "target"
            frac, text, id = self.trips / max(step.count, 1), f"{self.trips}/{step.count}", "hp-count"
        else:
            return
        ring = self._ring()
        if ring is not None:
            cx, cy, r = ring
        else:  # no part to go around: a ring of its own, where there is room
            x, y = free_spot(130.0, 130.0, self._taken)
            cx, cy, r = x + 65.0, y + 65.0, 58.0
        c.arc(f"{id}-arc", (cx, cy), r, frac, role)
        w = 17.0
        width = w * 1.45 * len(text)
        if ring is None:
            x, y = cx - width / 2, cy - w
        else:
            x, y, _, _ = label_spot((cx - r, cy - r, cx + r, cy + r), width, 2 * w, self._taken)
        self._digits(c, id, text, x, y, w, role)

    def _ring(self) -> tuple[float, float, float] | None:
        """Centre and radius of the ring a counted or timed step draws around its target, if it has one."""
        step = self._step_now() if self.state == "steps" else None
        t = self._target()
        if step is None or step.kind not in ("add", "timer") or t is None:
            return None
        return (t[0] + t[2]) / 2, (t[1] + t[3]) / 2, 0.5 * math.hypot(t[2] - t[0], t[3] - t[1]) + 14

    @staticmethod
    def _digits(c, id: str, text: str, x: float, y: float, w: float, role: str) -> None:
        """Big digits, with ":" or "/" among them."""
        for i, ch in enumerate(text):
            ox = x + i * w * 1.45
            if ch == ":":
                c.dot(f"{id}-c{i}a", (ox + w / 2, y + 0.6 * w), role)
                c.dot(f"{id}-c{i}b", (ox + w / 2, y + 1.4 * w), role)
            elif ch == "/":
                c.poly(f"{id}-s{i}", [(ox + 0.85 * w, y), (ox + 0.15 * w, y + 2 * w)], role)
            else:
                c.number(f"{id}-{i}", ch, ox, y, w, role)
