# Lightwork

**A helper that works on your real table.** Say what you want. It looks at what's there, asks only what it can't see, and shows you what to do with light on the objects themselves. A phone projector draws on the table, a webcam watches from above, and your hands are the controls.

Solo Indie entry for the Meta Global AI Developer Hackathon. All code was written during the event.

**Video:** _link added at submission._ Every wait that is cut or sped up in it is labelled, and everything it shows runs from this repository.

## What it does

You hold a fingertip on the projected mic bubble and speak. The microphone is open only while you hold it, so talk at the table goes nowhere.

| Say | What happens |
|---|---|
| "Help me make pancakes with this" | The table goes dark for a moment and the camera takes a clean picture. Muse Wonder counts your eggs but asks how many grams are in the bowl, because a camera can't weigh. Then it walks you through the steps: what to take is outlined (SAM 3.1), an arrow shows where it goes, and the camera checks each step before moving on. An oven step starts a clock and the job carries on. |
| "Help me build this Lego" | The same, working from the bricks and any manual page on the table. A wrong brick doesn't pass the camera check. |
| "What is this book about?" | It reads what the camera can read (covers, headings and big diagrams, not small print), answers out loud, puts the key facts on the table, and Muse Image draws a labelled diagram beside it. |
| "Show me a cargo ship" | Muse Wonder writes a program that builds it part by part, inside included. Pinch and drag to turn it, pinch the bar on the left to slice it open, use two hands to zoom in and move through it. "Show me the engine" fades the rest and flies there. |
| "Take me to Paris" | Muse Wonder gives the coordinates, then a 3D city flies in under your hands. Pinch and drag to move, two hands to turn and zoom. |
| "Set a timer for 10 minutes" | Spoken timers and a stopwatch, with no model call, on every tile. |
| "Start rock paper scissors" | The table's games start by name: rock paper scissors, catch the light, follow the lights, slice the fruit, draw and guess, chess. |
| "Let's play chess" | The table draws a board of light and shows where each real piece goes. You play White: move a piece and take your hand away, and the camera reads the move, animates it and the table replies. It has no hands, so it lights up its move and you make it for it, and the camera checks. The moves run down one side, the clocks stand on the other. |

## How it works

```text
fingertip on the mic  ->  Muse Voice Transcribe  ->  games and timers start at once, with no model call
                                                 ->  anything else:

projection goes dark for a moment  ->  clean camera picture, rectified to the table, the room left out
                                   ->  Muse Wonder: what do they want, what is here, what must I still ask?
    a job      ->  steps  ->  SAM 3.1 outlines the real objects  ->  camera checks after each step
    a question ->  spoken answer  ->  Muse Image diagram served to the projector page by URL
    a model    ->  Muse Wonder writes a program  ->  it runs in a sandbox  ->  Three.js on the phone draws it
    a place    ->  Muse Wonder gives coordinates  ->  MapLibre on the phone flies there
```

- **The blackout** is what makes the picture honest: our own light would otherwise look like objects to the model. The page confirms it has gone dark before the camera looks.
- **Hands** run locally at 30 fps (MediaPipe on the laptop's GPU). That helper only tracks fingertips and pinches. Meta's models do the seeing, planning, checking, reading, drawing and listening.
- **Model programs are treated as untrusted input** (`lightwork/wireframe.py`): the syntax is checked against a short list, with no imports, no attribute access beyond `math` and a few list methods, and it runs in a child Python with an empty environment and limits on CPU, memory and files.
- **Chess uses no model.** The rules and the engine are ours, in plain Python (`lightwork/chess.py`, checked against the published perft counts), running in their own process so a search never slows the camera loop. A move is read by comparing clean pictures of the board square by square, and the legal moves say which pattern of changed squares means what.
- **Every model call is recorded** under `.lightwork/play-calls/<session>/`: the request, the response, the picture sent, timing and cost, held to a local spend limit (`LW_BUDGET_USD`). Each mic hold's audio and transcript are recorded there too.

### Meta models and their jobs

| Model | Job |
|---|---|
| Muse Wonder (`muse-wonder-1.0`) | Sees the table, works out the intent, plans steps, checks them, answers questions, writes the 3D model programs, finds places |
| Muse Image (`muse-image-1.0`) | Diagrams for answers, drawn as bright lines on black, because black is no light on the table |
| SAM 3.1 (`sam-3.1`) | Outlines of the real objects from boxes |
| Muse Voice Transcribe (`muse-voice-transcribe-1.0`) | Everything said into the mic |

### Non-Meta components

No hosted non-Meta model is called at runtime. These are the local helpers, libraries and data:

| Component | Role | Licence |
|---|---|---|
| MediaPipe HandLandmarker | Local fingertip and pinch tracking; never the proof that a step is done | Apache-2.0; model downloaded separately |
| Three.js 0.169 (vendored) | Draws the 3D models on the phone, slices them, moves through them | MIT |
| MapLibre GL JS 4.7.1 (from unpkg, only when a map is asked for) | Draws the 3D map on the phone | BSD-3-Clause |
| OpenFreeMap "liberty" style, OpenMapTiles, OpenStreetMap data (fetched only when a map is asked for) | The map's look and data; attribution stays on screen | Their terms; OSM is ODbL |
| Browser `speechSynthesis` | The table's voice: the phone's built-in one | Phone's own |
| OpenCV | Camera capture, ArUco calibration, image processing; classic CV, not a model | Apache-2.0 |
| PipeWire `pw-record` | Reads the microphone on Linux | MIT |
| NumPy, FastAPI, Uvicorn, HTTPX, Pydantic | Arrays, local server, model HTTP, schemas | BSD-3-Clause, MIT |
| `meta-sam-parser` | Decodes Meta's SAM responses; a library, not a model | SAM License |

Details and sources: [THIRD_PARTY.md](THIRD_PARTY.md).

## Run the demo

**You need:** Linux with PipeWire (this is what it was built on), a webcam looking straight down at a table (a Logitech C920 here), a projector that can show a browser page on that table (a Unihertz Tank 3 Pro phone here), a Meta Model API key, and [uv](https://docs.astral.sh/uv/getting-started/installation/) with Python 3.12. The map also needs internet on the device showing the page.

```sh
uv sync --frozen --extra live
cp .env.example .env              # put your key in MODEL_API_KEY
uv run lightwork doctor
```

Start the table, with hands out of view while it calibrates:

```sh
uv run lightwork play --table --rotate 180 --names meta --names-model muse-wonder-1.0 \
    --page-wait 300 --no-hand-check --voice --game menu
```

`--rotate 180` is for when you sit opposite the projector; leave it out otherwise. Then, in another terminal, show the page:

```sh
uv run lightwork tank up --serial <your phone's adb serial>    # an Android phone over USB
```

For any other projector, open `http://localhost:8000/projector` full screen on it and tap once, which unlocks speech and sound. Hold a finger on **GENERAL HELPER**, then on the mic, and speak.

On this laptop MediaPipe only runs under the user manager, because PipeWire's JACK client lowers the real-time limit otherwise. If `play` is killed at start, run it through `systemd-run --user --same-dir -E PYTHONUNBUFFERED=1 "$(command -v uv)" run lightwork play ...`.

More: [the operator guide](docs/usage.md).

### No projector or no key?

The recorded calls of real runs are in [`recordings/`](recordings/): the pictures sent, the replies, timing and cost. The older bench mode can be replayed offline with no key, camera or network, using a clearly labelled **synthetic** fixture that only demonstrates deterministic replay, not model accuracy:

```sh
uv run lightwork verify tests/fixtures/runs/synthetic-kit
uv run lightwork replay tests/fixtures/runs/synthetic-kit     # open http://127.0.0.1:8000/screen
```

## Limits, plainly

- **It takes time.** On Muse Wonder at low reasoning: a look about 5 to 12 s, a plan 5 to 20 s, a camera check 3 to 4 s, a Muse Image diagram 11 to 13 s, a 3D model about 1 to 2 minutes the first time and instant after that, because it is kept by name. Hands, timers and games don't wait for any of that.
- **The camera sees about 1.6 px per mm** at this height: enough for a cover, a heading or a big diagram, not small print. The model is told to say so rather than guess.
- **A camera check judges what shows from above.** It can't weigh, and it can be wrong; NEXT always overrides it.
- **A 3D model is Muse Wonder's idea of the thing, not an engineering drawing.** Its proportions and insides come from what the model knows, and it can get them wrong.
- **Light matters.** A 100 lumen projector needs a dim room, and hand tracking misses more when palms leave the camera's view.
- **Chess reads what changed, not which piece is which.** It always promotes to a queen, and a move it misreads can be taken back with UNDO. Lift pieces off the table while it calibrates: they cover the markers.
- **Privacy.** Only the rectified table goes to the model, never the room, but a hand or anything else on the table is in the picture. Don't assume hosted inference has zero retention. Each hold's audio is recorded locally.

## The older bench mode

The repo began as a stricter bench tool: `lightwork author`, `layout`, `calibrate`, `live`, `verify`, `audit`, `replay` turn a written procedure into a validated step graph with a deterministic evidence reducer and a replayable event log. It is still here and tested, and [docs/usage.md](docs/usage.md) covers it. It has no accepted real-bench runs, so no accuracy is claimed for it. The optional open-weight path is in [docs/open-weight.md](docs/open-weight.md); no tuned model is claimed.

## Licence

Lightwork's code is [Apache-2.0](LICENSE). Dependencies, model weights, hosted services and procedure texts keep their own terms: see [THIRD_PARTY.md](THIRD_PARTY.md).
