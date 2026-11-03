"""``lightwork play``: projected playground, an interactive dashboard (see ``lightwork.playground``).

Needs the projector page (``lightwork tank up`` in another terminal). Every session starts with a calibration
(``lightwork.sessioncal``): camera gain for the room, the projector-camera mapping (keep hands out of view), then a
hand check that projects three targets to touch (or, with ``--ceiling``, point at). The dashboard follows:

- table/wall (default): hold a fingertip on a button to press it; a drop zone outlines objects put into it. Hands far
  from the camera (a wall) are also searched in overlapping tiles over the projected page (``--hand-tiles``).
- ``--table``: objects anywhere on the table are outlined, numbered and measured (mm), and followed when they move;
  two hands are tracked at full camera resolution, their skeletons projected onto them (``lightwork.tabletop``).
  Objects are looked for only in scans: once the view is still after a change, our light over the table goes dark for
  about a quarter of a second and the camera takes two clean frames, so it never mistakes our own outlines and labels
  for objects. LEARN relearns the empty table; SCALE measures a bank card on the table for exact sizes (else sizes
  come from the hands, marked "~"). ``--names meta`` names each object once with Muse Spark (the cheapest, contributor
  model by default; several objects per call; recorded under ``.lightwork/play-calls/``, spend shown), ``--names
  ollama`` with a local vision model (``lightwork.namer``). With Meta names, SAM 3.1 outlines each object from its
  mask (``--sam``, ``lightwork.outliner``). The projector page plays short sounds (scan, found, gone, press).
- ``--ceiling``: mid-air hand between camera and ceiling; pinch to click, drag or paint, open hand to clear;
  a gesture panel (finger count, pose name) and the hand skeleton projected where the camera sees the hand.

``--rotate 180`` turns the whole projection, calibration pattern included, to read from the far side of the table.

During the session the gain follows the room and the mapping is re-flashed every ``--recal-every`` seconds when no hand
is around; drift is adopted and a bumped rig repeats the hand check. Table mode keeps the camera gain chosen at the
start (a gain step looks like a light change to the scans) and re-flashes only when the projected border moved
(``--recal-every`` defaults to 0 there). ``systemctl --user kill -s USR1 lightwork-play``
(or ``kill -USR1``) runs the full calibration again. Open http://127.0.0.1:8000/screen for the camera view.
Hands and object scans are local (MediaPipe hand landmarks + OpenCV). Meta calls (names, SAM outlines, spoken
questions) only with --names meta / --sam / --voice, recorded under .lightwork/play-calls/. Not a procedure run.
"""

from __future__ import annotations

import math
import time

COMMAND = "play"
HELP = ("Projected playground on a table, wall or ceiling: local hands and scans; on a table, optional Meta names, "
        "SAM outlines, a bench check and spoken questions (recorded calls).")

LEARN_DELAY_S = 0.7  # after LEARN is pressed: let the finger leave and the scene settle first
STATUS_EVERY_S = 5.0
START_GAIN = 0  # opening gain when the gain is automatic (reads back exactly; chosen per room next)
GAIN_CHECK_S = 15.0
QUIET_S = 2.0  # no hand seen this long before a background gain nudge or interval flash
ADOPT_PX = 2.0  # page px of drift worth adopting a new mapping
BUMP_PX = 12.0  # page px of drift that means the rig moved: check the hand again
HANDS_AWAY_S = 0.8  # requested recalibrations wait this long for hands to leave ...
HANDS_AWAY_MIN_S = 4.0  # ... and at least this long after the request: a person in the light is no hand to the tracker
HANDS_AWAY_MAX_S = 8.0  # ... but not longer
CALIB_RETRIES = 2  # a recalibration that fails during the session is tried again this often, then the mapping is kept
HAND_MARGIN = 0.1  # hand search tiles cover the projected page plus this share of its size: a palm beside the edge


def configure(p) -> None:
    from ..contracts.server import PORT

    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--ceiling", action="store_true", help="mid-air mode for a ceiling: pinch, paint, gestures")
    mode.add_argument("--table", action="store_true",
                      help="table mode: objects outlined with their size and tracked, two hands tracked")
    p.add_argument("--rotate", type=int, choices=(0, 180), default=0,
                   help="turn the projection: 180 reads right from the far side of a table")
    p.add_argument("--page-mm", type=float,
                   help="table: measured width of the projected page on the table, mm (exact sizes without a card)")
    p.add_argument("--names", choices=("off", "meta", "open", "ollama"), default="off",
                   help="table: name each object once: meta = Muse Spark (billed, recorded), open = Meta open weights "
                        "via the partners (Baseten/Fireworks/Modal, or own hardware; see `lightwork partners`), "
                        "ollama = local model (not Meta)")
    p.add_argument("--names-model", help="model for --names (default: muse-spark-1.3-contributor / qwen3.5:4b)")
    p.add_argument("--sam", choices=("auto", "on", "off"), default="auto",
                   help="table: outline objects from SAM 3.1 masks (billed, recorded; auto = with --names meta/open)")
    p.add_argument("--task", help="table: a bench check: JSON {title, items}, a text file (title, then one item per "
                                  "line) or inline \"Title: item, item, ...\"; listed items turn green, others red")
    p.add_argument("--do", metavar="JOB",
                   help="table: a job to walk through: a built-in plan (pancakes, serial-dilution, seedlings, "
                        "kit-return), a plan file, or what you want to do in words (\"pancakes for four\"), planned "
                        "by the brain. With --voice it can also be asked for aloud (\"help me ...\")")
    p.add_argument("--game", choices=("menu", "catch", "follow", "draw"),
                   help="table: start with a hand game (nothing to set up): menu, catch (catch the light), follow "
                        "(follow the lights) or draw (draw and guess, Muse Spark guesses). Also by voice: \"let's play\"")
    p.add_argument("--brain", choices=("auto", "meta", "open", "off"), default="auto",
                   help="table: who plans jobs and checks steps: meta = Meta Model API (Muse Spark), open = Meta open "
                        "weights via the partners, auto = open when a partner is configured, else meta")
    p.add_argument("--planner", default=None,
                   help="Meta model that writes plans (default muse-spark-1.3; muse-wonder-1.0 is slower)")
    p.add_argument("--voice", action="store_true",
                   help="table: ask out loud (Muse Voice Transcribe via the /screen page's microphone, press Start "
                        "there): point at an object and ask \"what is this?\"; Muse Spark answers on the projector")
    p.add_argument("--mic", default="auto",
                   help="voice input: auto = the C920's microphone above the table, read by the server (no page "
                        "needed); browser = the /screen page's Start button; or part of a PipeWire source name")
    p.add_argument("--ollama-url", default="http://127.0.0.1:11434", help="Ollama server for --names ollama")
    p.add_argument("--camera", help="by-id path, /dev/videoN or index (default: C920)")
    p.add_argument("--port", type=int, default=PORT)
    p.add_argument("--page-wait", type=float, default=60.0, help="seconds to wait for the projector page")
    p.add_argument("--exposure", type=int, default=None,
                   help="C920 exposure_time_absolute in 100 us (default 312; 78 with --game: a quarter of the motion "
                        "blur on a moving hand, written again once streaming, where the C920 applies it)")
    p.add_argument("--gain", type=int, help="fixed C920 gain 0-255 (default: chosen for the room and adjusted)")
    p.add_argument("--recal-every", type=float, default=None,
                   help="seconds between mapping checks (0 = off; default 300, table 0: the border watch still checks)")
    p.add_argument("--no-hand-check", action="store_true", help="skip the touch/point targets after calibrating")
    p.add_argument("--no-frame", action="store_true", help="no projected border (and no border watch)")
    p.add_argument("--threshold", type=int, default=28, help="drop-zone change threshold, 0-255")
    p.add_argument("--min-area", type=int, default=500, help="smallest object in the drop zone (table: on the table), camera px")
    p.add_argument("--no-hands", action="store_true", help="no hand tracking (table drop zone only)")
    p.add_argument("--hand-tiles", type=int, default=3,
                   help="table/wall: most tiles the hand search splits the projected page into, so hands far from the "
                        "camera are found (0 = whole frame only)")
    p.add_argument("--quiet", action="store_true", help="do not speak on the projector phone")


def _wait_projector(hub, timeout: float):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        page = hub.hello("projector")
        if page is not None:
            time.sleep(0.4)  # let a fresh page settle its size
            return hub.hello("projector")
        time.sleep(0.1)
    return None


def _sigterm(*_):
    raise KeyboardInterrupt


def _tracker(args):
    if args.no_hands and not args.ceiling:
        return None
    from ..vision.hands import HandsUnavailable

    try:
        if args.ceiling:
            from ..playground import PoseTracker

            return PoseTracker()
        if args.table:
            from ..vision.hands import CropHandTracker

            if args.game:  # games: hands move fast; search harder, follow longer, accept blurrier hands
                t = CropHandTracker(num_hands=2, search_per_frame=4, min_confidence=0.4)
                t.LOST_FRAMES = 6
                return t
            return CropHandTracker(num_hands=2)
        from ..vision.hands import HandTracker, TiledHandTracker

        return TiledHandTracker(max_tiles=args.hand_tiles) if args.hand_tiles > 0 else HandTracker()
    except HandsUnavailable as exc:
        print(f"hands off: {exc}")
        if "RLIMIT_RTTIME" in str(exc):
            mode = " --ceiling" if args.ceiling else ""
            print("  this terminal (e.g. an IDE's) inherited RLIMIT_RTTIME=0; start it from the user manager:")
            print(f"  systemd-run --user -P --wait --same-dir -p KillSignal=SIGINT $(command -v uv) run lightwork play{mode}")
        return None


def _namer(args):
    """The object namer for --table --names, or None (names off, or the backend is not usable)."""
    if not args.table or args.names == "off":
        return None
    from ..namer import META_MODEL, OLLAMA_MODEL, MetaNamer, OllamaNamer

    if args.names == "meta":
        from ..config import load_settings

        namer = MetaNamer(load_settings(), model=args.names_model or META_MODEL)
    elif args.names == "open":
        from ..brain import make_brain
        from ..config import load_settings
        from ..namer import BrainNamer

        namer = BrainNamer(make_brain("open", load_settings()))
    else:
        namer = OllamaNamer(model=args.names_model or OLLAMA_MODEL, url=args.ollama_url)
    problem = namer.check()
    if problem:
        print(f"object names off: {problem}")
        namer.close()
        return None
    where = f"; calls recorded in {namer.run_dir}" if args.names in ("meta", "open") else ""
    print(f"object names: {namer.label}{where}")
    return namer


def _asker(args):
    """Muse Spark answers to spoken questions (--table --voice), or None."""
    if not args.table or not args.voice:
        return None
    from ..asker import BrainAsker, MetaAsker
    from ..config import load_settings

    if args.brain == "open":  # answers from Meta's open weights too
        from ..brain import make_brain

        asker = BrainAsker(make_brain("open", load_settings()))
    else:
        asker = MetaAsker(load_settings())
    problem = asker.check()
    if problem:
        print(f"spoken questions off: {problem}")
        asker.close()
        return None
    print(f"spoken questions: Muse Voice Transcribe -> {asker.label}; calls recorded in {asker.run_dir}")
    return asker


def _voice(args, heard, control, listening=None):
    """The Muse Voice Transcribe relay for --voice: sentences go to ``heard`` (a queue), the HUD to /screen, what is
    being heard right now to ``listening`` (a dict the table shows), and the microphone is read by the server itself
    (``--mic``) unless a browser page streams one."""
    if not args.table or not args.voice:
        return None, None
    from ..config import load_settings
    from ..voice import VoiceService

    def on_event(kind: str, payload: dict) -> None:
        text = str(payload.get("transcript") or "").strip()
        if text:
            heard.put(text)

    def on_hud(hud: dict) -> None:
        control.push_state({"hud": hud})
        if listening is not None:
            text = str(hud.get("heard") or "").strip()
            if text and text != listening.get("text"):
                listening.update(text=text, t=time.monotonic())

    voice = VoiceService(load_settings(), ["Lightwork", "table", "object", "missing", "next", "done", "repeat", "start",
                                           "stop", "check", "undo", "help me", "pancakes", "dilution", "scoop",
                                           "play", "catch the light", "follow the lights", "draw", "guess", "menu"],
                         on_event, on_hud=on_hud)
    voice.start()
    mic = None
    if args.mic != "browser":
        from ..voice.capture import ServerMic, find_source, sources

        source = find_source("C920" if args.mic == "auto" else args.mic)
        if source is None:
            print(f"voice: no microphone matching {args.mic!r} (sources: {', '.join(sources()) or 'none'}); "
                  "open /screen and press Start for the browser's")
        else:
            mic = ServerMic(voice.feed_pcm, source, yield_to=voice.browser_active)
            problem = mic.start()
            if problem:
                print(f"voice: {problem}; open /screen and press Start for the browser's microphone")
                mic = None
            else:
                print(f"voice: listening on {source} (no page needed; a /screen Start takes over)")
    else:
        print("voice: open /screen and press Start for the microphone")
    return voice, mic


def _brain(args):
    """Plans and checks for --table (--do / --voice): Meta Model API or Meta open weights, or None."""
    if not args.table or args.brain == "off" or not (args.do or args.voice or args.game):
        return None
    from ..brain import make_brain
    from ..config import load_settings

    brain = make_brain(args.brain, load_settings(), model=args.planner)
    problem = brain.check()
    if problem and args.brain == "auto":  # no partner and no Meta key: try the other kind
        brain.close()
        brain = make_brain("meta", load_settings(), model=args.planner)
        problem = brain.check()
    if problem:
        print(f"brain off ({problem}): built-in plans only, checks taken on your word")
        brain.close()
        return None
    print(f"brain: {brain.label}; calls recorded in {brain.run_dir}")
    return brain


def _plan(args, brain):
    """The --do job: a built-in plan or a file now; words go to the brain once the session runs (None here)."""
    if not args.table or not args.do:
        return None, None
    from ..guide import find_builtin, load_plan, scale_plan

    try:
        return load_plan(args.do), None
    except ValueError:
        pass
    found = find_builtin(args.do)  # "pancakes for 4": the built-in plan, scaled (instant, the same every time)
    if found is not None:
        return scale_plan(*found), None
    if brain is None:
        raise SystemExit(f"--do {args.do!r}: not a plan file or built-in plan, and no brain to plan it")
    return None, args.do


def _outliner(args):
    """SAM outlines for --table (``--sam``), or None."""
    if not args.table or args.sam == "off" or (args.sam == "auto" and args.names not in ("meta", "open")):
        return None
    from ..config import load_settings
    from ..outliner import SamOutliner

    outliner = SamOutliner(load_settings())
    problem = outliner.check()
    if problem:
        print(f"SAM outlines off: {problem}")
        outliner.close()
        return None
    print(f"object outlines: {outliner.label}; calls recorded in {outliner.run_dir}")
    return outliner


def run(args) -> int:
    import signal

    from ..contracts.render import SceneMsg, View
    from ..server.app import create_app
    from ..server.hub import make_hubs
    from ..server.runner import start_server
    from ..vision.calib.solve import CalibrationError
    from ..vision.camera import load_lock
    from ..vision.capture import CameraNotFound, open_camera

    signal.signal(signal.SIGTERM, _sigterm)  # `systemctl --user stop` cleans up like Ctrl-C
    if args.recal_every is None:
        args.recal_every = 0.0 if args.table else 300.0
    if args.exposure is None:
        args.exposure = 78 if args.game else 312
    gain = args.gain if args.gain is not None else START_GAIN
    lock = dict(load_lock())
    lock.update(exposure_time_absolute=args.exposure, gain=gain)
    try:
        cam = open_camera(args.camera, lock=lock)
    except CameraNotFound as exc:
        print(f"camera: {exc}")
        return 2
    import queue

    task = None
    if args.task:
        from ..benchtask import load_task

        try:
            task = load_task(args.task)
        except (OSError, ValueError) as exc:
            print(f"task: {exc}")
            cam.close()
            return 2
        print(f"bench check: {task.title}: {', '.join(task.items)}")
    hubs = make_hubs()
    hubs["render_hub"].set_view(View(rotate=args.rotate))  # before any pattern: the mapping must include the turn
    heard: queue.SimpleQueue[str] = queue.SimpleQueue()
    listening: dict = {"text": "", "t": -1e9}
    voice, mic = _voice(args, heard, hubs["control_hub"], listening)
    if voice is not None:
        hubs["render_hub"].on_tts(voice.on_tts)  # the mic is muted while the projector phone speaks
    try:
        server = start_server(create_app(mode="play", voice=voice, **hubs), port=args.port)
    except RuntimeError as exc:
        cam.close()
        print(f"server: {exc} (is `lightwork live`, `calibrate`, `replay` or another `play` still running?)")
        return 2
    tracker = namer = outliner = asker = brain = None
    try:
        print(f"camera: {cam.status}")
        tracker = _tracker(args)
        namer = _namer(args)
        outliner = _outliner(args)
        asker = _asker(args)
        brain = _brain(args)
        plan, plan_words = _plan(args, brain)
        if plan is not None:
            print(f"job: {plan.title}: {', '.join(it.name for it in plan.items)}; {len(plan.steps)} steps")
        if args.ceiling and tracker is None:
            print("ceiling mode needs hand tracking")
            return 2
        first = cam.wait_frame(0.0, 3.0)
        if first is None:
            print("camera gave no frame")
            return 2
        frame_size = (first.image.shape[1], first.image.shape[0])
        hubs["control_hub"].push_state({
            "run": {"pack": "playground (ceiling)" if args.ceiling else "playground (table)" if args.table
                    else "playground", "mode": "live",
                    "backend": "local hand landmarks" + ("" if args.ceiling else " + change detection"),
                    "provenance": "local"},
            "banner": "PLAYGROUND · local only · nothing recorded",
            "camera": {"w": frame_size[0], "h": frame_size[1]},
        })
        print(f"waiting for the projector page on http://localhost:{args.port}/projector (run `lightwork tank up`)")
        page = _wait_projector(hubs["render_hub"], args.page_wait)
        if page is None:
            print("no projector page connected")
            return 2
        print(f"projector page {page.innerWidth}x{page.innerHeight}{', turned 180 degrees' if args.rotate else ''}; "
              f"screen view: http://127.0.0.1:{args.port}/screen")
        play = Play(args, cam, hubs, tracker, frame_size, gain, namer=namer, outliner=outliner, asker=asker,
                    task=task, heard=heard, plan=plan, plan_words=plan_words, brain=brain, listening=listening)
        # `systemctl --user kill -s USR1 lightwork-play` (or `kill -USR1 <pid>`): full calibration again
        signal.signal(signal.SIGUSR1, lambda *_: setattr(play, "recal", True))
        # `... -s USR2 ...`: --table debug snapshot under .lightwork/play-debug/ (frame, reference, masks, tracks)
        signal.signal(signal.SIGUSR2, lambda *_: setattr(play, "dump", True))
        play.run(page)
        return 0
    except CalibrationError as exc:
        print(f"calibration failed: {exc}")
        return 1
    except KeyboardInterrupt:
        return 0
    finally:
        # uv forwards the signal it got too: a second interrupt here would abort cleanup mid-way
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        try:
            hubs["render_hub"].publish_scene(SceneMsg(seq=0, mode="normal", prims=[]))
            time.sleep(0.3)
        except Exception:  # noqa: BLE001 - best-effort blanking on the way out
            pass
        if tracker is not None:
            tracker.close()
        if namer is not None:
            print(namer.summary())
            namer.close()
        if outliner is not None:
            print(outliner.summary())
            outliner.close()
        if asker is not None:
            print(asker.summary())
            asker.close()
        if brain is not None:
            print(brain.summary())
            brain.close()
        if mic is not None:
            print(f"microphone: {mic.status()}")
            mic.close()
        if voice is not None:
            voice.close()
        server.stop()
        cam.close()
        print("stopped")


class Play:
    """Session calibration, hand check and the frame loop for every mode."""

    table = None  # TableSession (--table)
    last_image = None  # newest camera image, for carrying the learned table over to a new mapping
    dump = False  # SIGUSR2: write a --table debug snapshot

    def __init__(self, args, cam, hubs, tracker, frame_size, gain: int, namer=None, outliner=None, asker=None,
                 task=None, heard=None, plan=None, plan_words=None, brain=None, listening=None):
        from ..sessioncal import CameraGain

        self.args, self.cam, self.tracker, self.frame_size = args, cam, tracker, frame_size
        self.namer, self.outliner, self.asker, self.task, self.heard = namer, outliner, asker, task, heard
        self.plan, self.plan_words, self.brain = plan, plan_words, brain
        self.listening = listening  # what the transcriber hears right now: shown on the table while you speak
        self._shown_heard = ""
        self.render, self.preview, self.control = hubs["render_hub"], hubs["preview_hub"], hubs["control_hub"]
        self.gain = CameraGain(cam, gain)
        self.auto_gain = args.gain is None and self.gain.available
        self.page = None
        self.geo = self.dash = self.zone = self.overlay = self.footprint = None
        self.check = None  # HandCheck while the targets are up
        self.check_result = None
        self.pending: str | None = None  # "full" | "retry" | "interval": waits for hands to leave
        self.pending_since = 0.0
        self.calib_t = 0.0
        self.last_drift: float | None = None
        self.learn_at: float | None = None
        self.zone_pause_until = 0.0
        self.last_hand_t = -1e9
        self.next_gain_check = 0.0
        self.last_prims = None
        self.frame: list = []  # projected border prims
        self.border = None  # BorderWatch on the projected border
        self.border_next = 0.0
        self.say = ""
        self.recal = False
        self.tiles_shown = None  # hand search tiles last printed
        self.calib_failures = 0  # failed recalibrations in a row during the session

    # ------------------------------------------------------------------------------------------------ calibration

    def _speak(self, text: str) -> None:
        self.say = text
        if not self.args.quiet:
            self.render.say(text)

    def _adopt(self, H, page) -> None:
        """Use mapping ``H``; dashboard state survives, the drop zone is rebuilt in the new camera px."""
        from ..contracts.render import Calib
        from ..playground import (
            CeilingDashboard,
            Dashboard,
            DropZone,
            Geo,
            OverlayTimeline,
        )
        from ..sessioncal import footprint_mask

        self.page = page
        self.geo = Geo(H, page.innerWidth, page.innerHeight)
        self.render.set_calib(Calib(H_cam_to_page=[list(r) for r in H], page_size=[page.innerWidth, page.innerHeight]))
        if self.args.table:
            self._adopt_table()
        elif self.dash is None:
            self.dash = CeilingDashboard(self.geo) if self.args.ceiling else Dashboard(self.geo)
        else:
            self.dash.geo = self.geo
        if not self.args.ceiling and not self.args.table:
            self.zone = DropZone(self.geo, self.dash.zone_rect, self.frame_size,
                                 threshold=self.args.threshold, min_area=self.args.min_area)
            self.overlay = OverlayTimeline(self.geo, self.frame_size)
            self.learn_at = time.monotonic() + 1.0
        self.footprint = footprint_mask(self.geo, self.frame_size)
        self._hand_region()
        self._watch_border()
        self.last_prims = None

    def _adopt_table(self) -> None:
        from ..config import load_settings
        from ..tabletop import TableSession

        if self.table is None:
            a = self.args
            self.table = TableSession(self.geo, self.frame_size, threshold=a.threshold, min_area=a.min_area,
                                      page_mm=a.page_mm, state_dir=load_settings().state_dir, namer=self.namer,
                                      outliner=self.outliner, link=self.render, asker=self.asker, task=self.task,
                                      plan=self.plan, brain=self.brain, voice=bool(a.voice), planner_model=a.planner,
                                      game=a.game)
            self.table.learn_at = time.monotonic() + 0.5
            if self.plan_words:  # --do in words: planned once the table is up
                self.table._pending_say += self.table.request_plan(self.plan_words, time.monotonic())
            scale = self.table.scale
            print("table scale: " + (f"{scale.source}, {scale.mm_per_px:.3f} mm per page px" if scale else
                                     "from the hands until a card is measured (SCALE)"))
        else:
            self.table.set_geo(self.geo, self.last_image)
        if self.args.game and self.gain.available:  # the C920 ignores some exposures at stream start: write it
            err = self.gain.ctl.set("exposure_time_absolute", int(self.args.exposure))  # again now (3 Nov: 78 read
            print(f"game exposure {self.args.exposure / 10:.1f} ms" + (f": {err}" if err else ""))  # back as 156)
        if self.table.zone_error:
            print(f"table: {self.table.zone_error}")
        else:
            x, y, w, h = self.table.zone.roi
            print(f"table area: {w}x{h} camera px at {x},{y}, detection at 1/{self.table.zone.down}")

    def _hand_region(self) -> None:
        """Point the tiled hand search at the projected page (plus ``HAND_MARGIN``) in camera px."""
        from ..geometry import apply_h

        if not hasattr(self.tracker, "set_region"):
            return
        g = self.geo
        cam = apply_h(g.Hi, [(0, 0), (g.page_w, 0), (g.page_w, g.page_h), (0, g.page_h)])
        (x0, y0), (x1, y1) = cam.min(axis=0), cam.max(axis=0)
        m = HAND_MARGIN * max(x1 - x0, y1 - y0)
        fw, fh = self.frame_size
        region = (max(0, int(x0 - m)), max(0, int(y0 - m)), min(fw, int(x1 + m)), min(fh, int(y1 + m)))
        tiles = self.tracker.set_region(region)
        if tiles and tiles != self.tiles_shown:
            self.tiles_shown = tiles
            sizes = [(t[2] - t[0], t[3] - t[1]) for t in tiles]
            where = (f"{len(tiles)} full-resolution tiles over the page plus margin, each hand then followed in its own"
                     " crop" if self.args.table else
                     f"whole frame + {len(tiles)} tile{'s' if len(tiles) > 1 else ''} over the page")
            print(f"hand search: {where} ({', '.join(f'{w}x{h}' for w, h in sizes)} px): palms "
                  f"{max(fw, fh) / max(max(s) for s in sizes):.1f}x larger to the detector")

    def _watch_border(self) -> None:
        """(Re)draw the border for the current mapping and take a new baseline once it is seen."""
        from ..sessioncal import BorderWatch, frame_prims

        if self.args.no_frame:
            return
        self.frame = frame_prims(self.geo)
        self.border = BorderWatch(self.geo)
        self.border_next = time.monotonic() + 1.0  # the page redraws with the new mapping first

    def calibrate(self, page, *, gain: bool) -> None:
        """Gain (optional), then the pattern flash. Falls back to the saved calibration only before the first mapping."""
        from ..sessioncal import choose_gain, flash_geometry
        from ..vision.calib.flow import load_calibration
        from ..vision.calib.solve import CalibrationError

        print("calibrating: keep hands out of view")
        if gain and self.auto_gain:
            if self.render.show_mode("dark") is not None:
                time.sleep(0.15)
            level = choose_gain(self.gain, self.cam, self.footprint)
            print(f"camera gain {self.gain.gain}: surface level {level:.0f}")
        try:
            flash = flash_geometry(self.cam, self.render, page, self.gain, log=print)
        except CalibrationError as exc:
            if self.geo is not None or self.args.rotate:  # a saved calibration was solved from an unturned page
                raise
            saved = load_calibration()
            if saved.page is None or (saved.page.w, saved.page.h) != (page.innerWidth, page.innerHeight):
                raise
            print(f"pattern flash failed ({exc}); using the saved calibration, unverified")
            self._adopt(saved.H_cam_to_page, page)
            return
        r = flash.solution.residuals
        print(f"mapping: {r.n_markers} markers, rms {r.rms_px:.2f} page px, {flash.t_s:.2f} s")
        self._adopt(flash.solution.H.tolist(), page)
        self.calib_t = time.monotonic()
        self.last_drift = None

    def start_check(self, attempt: int = 1) -> None:
        from ..sessioncal import TARGETS, HandCheck

        if self.args.no_hand_check or self.tracker is None:
            return
        mode = "point" if self.args.ceiling else "touch"
        self.check = HandCheck(self.geo, mode, attempt)
        self._speak("Touch the circles, one at a time." if mode == "touch" else "Point at the circles, one at a time.")
        print(f"hand check: {'touch' if mode == 'touch' else 'point at'} the {len(TARGETS)} projected targets")

    def finish_check(self, res, now: float) -> None:
        from ..sessioncal import PASS_RESIDUAL

        print(res.message + (f" errors {list(res.errors)}" if res.errors else ""))
        self.check_result, attempt, self.check = res, self.check.attempt, None
        if res.status in ("pass", "aligned"):
            self._set_offset(res.offset)
        elif res.status == "fail" and attempt < 2:
            if self._border_steady():
                # the border shows the mapping did not move: the hand tracking was the problem, not the calibration
                print("border unchanged since calibration: mapping verified; repeating the targets")
                self._speak("The mapping is fine. Stand to the side so the camera sees your hand, and try again.")
                self.start_check(attempt=2)
                return
            self._speak(res.message + " Keep clear of the projection, recalibrating.")
            self._request("retry", now)
            return
        elif res.status == "fail":
            # scattered errors average to noise, not to the fingertip offset: correct the cursor only when they agree
            agree = len(res.errors) >= 2 and res.residual <= PASS_RESIDUAL
            self._set_offset(res.offset if agree else (0.0, 0.0))
            note = f"cursor corrected by {res.offset[0]:+.0f}, {res.offset[1]:+.0f} px" if agree else "no cursor correction"
            if self._border_steady():
                print(f"continuing: mapping verified by the border; {note}")
            else:
                print(f"continuing unverified ({note}): are the camera and projector facing the same flat surface?")
        self._speak(res.message)
        self.learn_at = now + 1.0 if self.zone is not None else None

    def _set_offset(self, offset) -> None:
        if self.table is not None:
            self.table.slots.offset = offset
        else:
            self.dash.offset = offset

    def _check_tip(self, poses) -> tuple[float, float] | None:
        """The fingertip the hand check follows: of all hands, the one nearest the current target."""
        if not poses:
            return None
        if self.check is None or self.geo is None:
            return poses[0].tip
        from ..sessioncal import TARGETS

        target = TARGETS[min(self.check.i, len(TARGETS) - 1)]
        return min((p.tip for p in poses), key=lambda tip: math.dist(self.geo.c2d(tip), target))

    def _border_steady(self) -> bool:
        """The projected border is seen where it was drawn, against a baseline: the mapping is independently verified."""
        b = self.border
        if b is None or b.baseline is None or b.last is None or b.last.n_seen < 3:
            return False
        shift = b.shift(b.last)
        return shift is not None and shift < 1.0

    def _request(self, kind: str, now: float) -> None:
        if self.pending != "full":  # a full calibration covers the rest
            self.pending, self.pending_since = kind, now

    def _run_pending(self) -> None:
        from ..sessioncal import drift_px, flash_geometry
        from ..vision.calib.solve import CalibrationError

        kind, self.pending = self.pending, None
        page = self.render.hello("projector") or self.page
        if kind in ("full", "retry"):
            if kind == "full":
                self.check = None
            try:
                self.calibrate(page, gain=kind == "full")
            except CalibrationError as exc:  # someone in the light, or the rig moved out of view: keep the mapping
                self.calib_failures += 1
                if self.calib_failures <= CALIB_RETRIES:
                    print(f"calibration failed ({exc}); keeping the current mapping and trying again")
                    self._speak("Calibration failed. Keep clear of the projection, trying again.")
                    self._request(kind, time.monotonic())
                else:
                    print(f"calibration failed {self.calib_failures} times ({exc}); continuing with the current mapping"
                          " (is the whole projection in the camera's view?)")
                    self._speak("Calibration failed. Continuing with the previous mapping.")
                    self.calib_failures = 0
                    self.calib_t = time.monotonic()
            else:
                self.calib_failures = 0
                self.start_check(attempt=2 if kind == "retry" else 1)
        else:  # interval
            try:
                flash = flash_geometry(self.cam, self.render, page, self.gain, log=print)
            except CalibrationError as exc:
                print(f"interval check: pattern flash failed ({exc}); keeping the current mapping")
                self.calib_t = time.monotonic()
                return
            H = flash.solution.H.tolist()
            self.last_drift = drift_px(self.geo.H, H, page.innerWidth, page.innerHeight)
            self.calib_t = time.monotonic()
            if self.last_drift < ADOPT_PX:
                print(f"interval check: drift {self.last_drift:.1f} page px, mapping kept")
                self._watch_border()
            else:
                print(f"interval check: drift {self.last_drift:.1f} page px, new mapping adopted")
                self._adopt(H, page)
                if self.last_drift >= BUMP_PX:
                    self._speak("The rig moved. Check the surface again.")
                    self.start_check()
        self.zone_pause_until = time.monotonic() + 0.6  # the camera still sees the pattern for a moment

    # ------------------------------------------------------------------------------------------------ frame loop

    def run(self, page) -> None:
        from ..contracts.render import SceneMsg
        from ..playground import Canvas
        from ..server.hub import encode_preview
        from ..sessioncal import nudge_gain

        self.calibrate(page, gain=True)
        self.start_check()
        last_t, last_step = 0.0, time.monotonic()
        n_frames, hand_frames = 0, 0
        fps_t0 = last_status = last_page_check = time.monotonic()
        while True:
            frame = self.cam.wait_frame(last_t, 1.0)
            if frame is None:
                continue
            newest = self.cam.latest()  # a slow frame must not queue the next ones up: always work on the newest
            if newest is not None and newest.t_mono > frame.t_mono:
                frame = newest
            last_t = frame.t_mono
            self.last_image = frame.image
            now = time.monotonic()
            n_frames += 1
            dt, last_step = now - last_step, now

            pose = obs = None
            poses: list = []
            if self.args.ceiling:
                pose = self.tracker.pose(frame)
                tip_cam = None if pose is None else (float(pose.img[8][0]), float(pose.img[8][1]))
            elif self.args.table:
                if self.tracker is not None and getattr(self.tracker, "motion_aware", False):
                    if getattr(self, "_motion", None) is None:
                        from ..vision.hands import MotionBoxes

                        self._motion = MotionBoxes()
                    poses = self.tracker.poses(frame, motion=self._motion(frame.image))
                else:
                    poses = self.tracker.poses(frame) if self.tracker is not None else []
                tip_cam = self._check_tip(poses)
            else:
                obs = self.tracker.process(frame) if self.tracker is not None else None
                tip_cam = obs.tip_xy if obs is not None and obs.present else None
            if tip_cam is not None:
                self.last_hand_t = now
                hand_frames += 1

            if self.check is not None:
                res = self.check.step(None if tip_cam is None else self.geo.c2d(tip_cam), now)
                c = Canvas(self.geo)
                self.check.draw(c)
                prims, obj_prims = c.prims, []
                if res is not None:
                    self.finish_check(res, now)
            elif self.pending in ("full", "retry"):
                prims, obj_prims = [], []  # replaced by the hands-away notice below; no zone learning meanwhile
            else:
                if self.args.ceiling:
                    fired = self.dash.update(pose, now, dt)
                elif self.args.table:
                    fired = self._table_step(frame, poses, now, dt)
                else:
                    fired = self._table(frame, obs, now, dt)
                if fired:
                    print("pressed " + ", ".join(fired))
                if self.args.table:
                    prims, obj_prims = self.table.scene(now, also_lit=self.frame), []
                else:
                    prims, obj_prims = self.dash.scene(self.zone, time.strftime("%H:%M:%S"))
            if self.pending in ("full", "retry"):
                c = Canvas(self.geo)
                c.label("hands-away", (250, 210), "KEEP CLEAR: CALIBRATING", "warn")
                prims, obj_prims = c.prims, []

            prims = prims + self.frame
            if prims != self.last_prims:
                self.render.publish_scene(SceneMsg(seq=0, mode="normal", prims=prims))
                if self.overlay is not None:
                    self.overlay.set(obj_prims, time.monotonic())
                self.last_prims = prims

            if n_frames % 2 == 0 and self.preview.has_viewers:
                self.preview.push(frame.idx, frame.t_mono, encode_preview(frame.image))

            quiet = now - self.last_hand_t >= QUIET_S and self.check is None
            if self.recal:
                self.recal = False
                print("full calibration requested")
                self._request("full", now)
            if now - last_page_check >= 1.0:
                last_page_check = now
                current = self.render.hello("projector")
                if current is not None and (current.innerWidth, current.innerHeight) != (self.geo.page_w, self.geo.page_h):
                    print(f"projector page resized to {current.innerWidth}x{current.innerHeight}")
                    self._request("full", now)
            if (self.args.recal_every > 0 and self.pending is None and self.check is None
                    and now - self.calib_t >= self.args.recal_every):
                self._request("interval", now)
            if (self.border is not None and self.pending is None and tip_cam is None and now >= self.border_next
                    and now - self.last_hand_t >= 1.0 and now >= self.zone_pause_until):
                self.border_next = now + 1.0
                had_baseline = self.border.baseline is not None
                event = self.border.update(frame.image)
                if not had_baseline and self.border.baseline is not None:
                    r = self.border.baseline
                    print(f"border seen on {r.n_seen}/4 sides (contrast {r.contrast:.0f}): watching it")
                if event == "moved":
                    print(f"projection border shifted {self.border.shift(self.border.last):.1f} page px: checking the mapping")
                    self._request("interval", now)
                elif event == "lost" and self.args.table:  # objects or a dim room hide it; a bump shows as "moved"
                    print("projection border no longer seen (not checking: in table mode only a shifted border is)")
                elif event == "lost":
                    print("projection border no longer seen: checking the mapping")
                    self._request("interval", now)
            if self.pending is not None:
                waited = now - self.pending_since
                hands_away = now - self.last_hand_t >= HANDS_AWAY_S and waited >= HANDS_AWAY_MIN_S
                forced = self.pending != "interval" and waited >= HANDS_AWAY_MAX_S
                if (self.pending == "interval" and quiet) or (self.pending != "interval" and (hands_away or forced)):
                    self._run_pending()
                    self.last_prims = None
            elif self.auto_gain and quiet and now >= self.next_gain_check and not self.args.table:
                self.next_gain_check = now + GAIN_CHECK_S
                changed, level = nudge_gain(self.gain, self.cam, self.footprint)
                if changed:
                    print(f"surface level {level:.0f}: camera gain now {self.gain.gain}")

            if now - last_status >= 1.0:
                last_status = now
                self.control.push_state({"steps": self._readouts(now), "last_say": self.say or None})
                if now - fps_t0 >= STATUS_EVERY_S:
                    print(f"{n_frames / (now - fps_t0):.1f} fps; hand in {hand_frames}/{n_frames} frames; {self._summary()}")
                    n_frames, hand_frames, fps_t0 = 0, 0, now

    def _table_step(self, frame, poses, now: float, dt: float) -> list[str]:
        fired, say = self.table.step(frame, poses, now, dt, detect=now >= self.zone_pause_until)
        if self.listening is not None:  # live words on the table while you speak: it is hearing you
            text, t = self.listening.get("text", ""), self.listening.get("t", -1e9)
            if text and text != self._shown_heard and now - t < 3.0:
                self._shown_heard = text
                self.table.dash.notify(("HEARD: " + text.upper())[-44:], now)
        while self.heard is not None and not self.heard.empty():
            text = self.heard.get_nowait()
            print(f"heard: {text!r}")
            say += self.table.hear(text, now)
        if self.dump:
            from ..config import load_settings

            self.dump = False
            path = load_settings().state_dir / "play-debug" / time.strftime("%Y%m%dT%H%M%S")
            self.table.dump(path, frame.image)
            print(f"debug snapshot: {path}")
        for line in self.table.log:
            print(line)
        self.table.log.clear()
        for name in self.table.sfx:
            self.render.sfx(name)
        self.table.sfx.clear()
        for line in say:
            print(f"table: {line}")
            self._speak(line)
        return fired

    def _table(self, frame, obs, now: float, dt: float) -> list[str]:
        from ..playground import spoken

        present = bool(obs and obs.present)
        self.dash.track(obs.tip_xy if present else None, now)
        fired = self.dash.step(now, dt)
        if "learn" in fired:
            self.learn_at = now + LEARN_DELAY_S
            self.zone.status = "learning"
        hand_near = self.dash.near_zone()
        if self.learn_at is not None and now >= self.learn_at:
            if hand_near:
                self.learn_at = now + 0.3
            else:
                self.zone.learn(frame.image)
                self.learn_at = None
                print("drop zone learned (empty reference)")
        elif self.learn_at is None and frame.idx % 2 == 0 and now >= self.zone_pause_until:
            overlay = self.overlay.at(frame.t_mono)
            relearns = self.zone.relearns
            changed = self.zone.update(frame.image, frame.t_mono, overlay=overlay, hand_near=hand_near,
                                       hand_visible=present)
            if self.zone.relearns != relearns:
                print(f"zone relearned: light change or something stuck on its edge ({self.zone.relearns} so far)")
                self._speak("Zone relearned.")
            if changed:
                self._speak(spoken(self.zone))
                print(f"zone: {self.say}")
        return fired

    # ------------------------------------------------------------------------------------------------ status

    def _summary(self) -> str:
        from ..playground import zone_text

        d = self.dash
        cal = f"gain {self.gain.gain}"
        if self.border is not None and self.border.last is not None:
            shift = self.border.shift(self.border.last)
            cal += f", border {self.border.last.n_seen}/4" + ("" if shift is None else f" shift {shift:.1f} px")
        if self.args.table:
            return f"{self._table_line()}; {cal}"
        if self.args.ceiling:
            return (f"gesture {d.gesture or '-'}, {d.fingers} fingers, pinch {'on' if d.pinching else 'off'}, "
                    f"{d.clicks} pinches, {len(d.strokes)} strokes; {cal}")
        z = self.zone
        return f"{zone_text(z).lower()}; light {'/'.join(f'{g:.2f}' for g in z.gain[::-1])}, {z.relearns} relearns; {cal}"

    def _border_text(self) -> str:
        if self.border is None:
            return "off"
        r = self.border.last
        if r is None:
            return "waiting for a hand-free frame"
        shift = self.border.shift(r)
        sides = " ".join(f"{n}:{'-' if o is None else f'{o:+.1f}'}" for n, o in zip(("T", "R", "B", "L"), r.sides, strict=False))
        base = "no baseline yet" if shift is None else f"shift {shift:.1f} px"
        return f"{r.n_seen}/4 sides ({sides}) · {base} · contrast {r.contrast:.0f}"

    def _table_line(self) -> str:
        from ..tabletop import table_text

        t = self.table
        if t is None:
            return "table not set up"
        hands = ", ".join(s.side or str(s.k + 1) for s in t.slots.active) or "none"
        scale = t.scale
        sc = "none yet" if scale is None else f"{scale.source} {scale.mm_per_px:.3f} mm/px"
        z = t.zone
        light = "" if z is None else (f"; light {'/'.join(f'{g:.2f}' for g in z.gain[::-1])}, {z.relearns} relearns, "
                                      f"{z.healed} healed")
        names = f"; names {t.namer.summary()}" if t.namer is not None else ""
        names += f"; outlines {t.outliner.summary()}" if t.outliner is not None else ""
        names += f"; {t.scans} scans, {t.scan_aborts} dropped"
        n, raw, drawn, redrawn = t.slots.jitter()
        if n:
            k, unit = (scale.mm_per_px, "mm") if scale is not None else (1.0, "page px")
            hands += (f" (held still {n} frames: landmarks move {raw * k:.1f} {unit} a frame, drawn {drawn * k:.2f} "
                      f"{unit}, redrawn in {redrawn:.0%})")
        return f"objects: {table_text(z, t.tracks.tracks, scale)}; hands {hands}; scale {sc}{light}{names}"

    def _table_rows(self) -> list[dict]:
        from ..tabletop import table_text

        t = self.table
        if t is None:
            return []
        z = t.zone
        scale = t.scale
        hands = [s.side or str(s.k + 1) for s in t.slots.active]
        rows = [
            {"id": "hands", "say": f"{len(hands)} tracked" + (f" ({', '.join(hands)})" if hands else ""),
             "state": "passed" if hands else "pending"},
            {"id": "objects", "say": table_text(z, t.tracks.tracks, scale),
             "state": "passed" if z is not None and z.status == "ready" else "provisional"},
            {"id": "scale", "say": "none yet" if scale is None else
             f"{scale.source}: {scale.mm_per_px:.3f} mm per page px" + (" (approximate)" if scale.approx else ""),
             "state": "passed" if scale is not None and not scale.approx else "provisional"},
        ]
        if t.namer is not None:
            err = f" · last error: {t.namer.error}" if t.namer.error else ""
            rows.append({"id": "names", "say": t.namer.summary() + err, "state": "provisional" if err else "passed"})
        g = t.guide
        if g is not None:
            l1, l2, _ = g.hud(time.monotonic())
            rows.append({"id": "job", "say": f"{g.plan.title} ({g.plan.source}) · {l1.lower()} · {l2.lower()}",
                         "state": "passed" if g.phase == "done" else "provisional"})
        if t.brain is not None:
            rows.append({"id": "brain", "say": t.brain.summary(), "state": "passed"})
            for k, line in enumerate(list(t.brain.trace)[-4:]):
                rows.append({"id": f"call {k + 1}", "say": line, "state": "passed"})
        if z is not None:
            rows.append({"id": "light", "say": "table light " + "/".join(f"{g:.2f}" for g in z.gain[::-1])
                         + f" (R/G/B) · {z.relearns} auto relearns · {z.healed} removed objects healed",
                         "state": "passed"})
        return rows

    def _readouts(self, now: float) -> list[dict]:
        from ..playground import zone_text

        d = self.dash
        hc = self.check_result
        hand_check = "running" if self.check is not None else (
            "not run" if hc is None else f"{hc.status} · scatter {hc.residual:.0f} px · offset {hc.offset[0]:+.0f},{hc.offset[1]:+.0f}")
        drift = "" if self.last_drift is None else f", last drift {self.last_drift:.1f} px"
        if self.args.table:
            return [
                {"id": "calibration", "say": f"gain {self.gain.gain} ({'auto' if self.auto_gain else 'fixed'}) · "
                 f"mapping {now - self.calib_t:.0f} s old{drift}", "state": "passed"},
                {"id": "hand check", "say": hand_check,
                 "state": "passed" if hc is not None and hc.status in ("pass", "aligned") else "provisional"},
                *self._table_rows(),
            ]
        rows = [
            {"id": "calibration", "say": f"gain {self.gain.gain} ({'auto' if self.auto_gain else 'fixed'}) · "
             f"mapping {now - self.calib_t:.0f} s old{drift}", "state": "passed"},
            {"id": "border", "say": self._border_text(), "state": "passed" if self.border is not None
             and self.border.baseline is not None and self.border.last is not None and self.border.last.n_seen >= 3
             else "provisional"},
            {"id": "hand check", "say": hand_check,
             "state": "passed" if hc is not None and hc.status in ("pass", "aligned") else "provisional"},
            {"id": "hand", "say": "hand tracked" if d.tip else ("no hand" if self.tracker else "tracking off"),
             "state": "passed" if d.tip else "pending"},
            {"id": "count", "say": str(d.count), "state": "passed"},
            {"id": "level", "say": f"{round(d.level * 100)}%", "state": "passed"},
        ]
        if self.args.ceiling:
            rows += [
                {"id": "gesture", "say": f"{d.gesture or '-'} · {d.fingers} fingers", "state": "passed" if d.pose else "pending"},
                {"id": "pinch", "say": f"{'on' if d.pinching else 'off'} · {d.clicks} so far", "state": "passed"},
                {"id": "paint", "say": f"{len(d.strokes)} strokes", "state": "passed"},
            ]
        else:
            z = self.zone
            rows.append({"id": "zone", "say": zone_text(z).capitalize() + "".join(
                f" · {i} {o.colour} ({o.area} px)" for i, o in enumerate(z.objects, 1)),
                "state": "passed" if z.status == "ready" else "provisional"})
            rows.append({"id": "light", "say": "zone light " + "/".join(f"{g:.2f}" for g in z.gain[::-1])
                         + f" (R/G/B) · {z.relearns} auto relearns", "state": "passed"})
        return rows
