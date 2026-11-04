# Third-party components and content

Lightwork's own code is [Apache-2.0](LICENSE). That grant does not relicense dependencies, model weights, hosted APIs, procedure text, recordings or browser voices. Packages are installed through [pyproject.toml](pyproject.toml) and the pinned [uv.lock](uv.lock), not copied wholesale into this repository. Preserve the upstream licence and notice files when redistributing wheels, runtimes or models.

## Runtime components

Licence identifiers below were checked against installed distribution metadata/licence files on 29 October 2026, except optional MediaPipe and llama.cpp, whose primary project sources are cited. Versions describe that inspection, not a promise that all optional components are installed. A combined wheel licence can include bundled native components with their own notices.

| Component | Role in Lightwork | Licence/source |
|---|---|---|
| MediaPipe code (optional `live` extra) | On-device hand/fingertip tracking; advisory, never sufficient to pass a step | [Apache-2.0 project licence](https://github.com/google-ai-edge/mediapipe/blob/master/LICENSE) |
| MediaPipe HandLandmarker model bundle | Palm and hand-landmark inference; downloaded locally, not bundled in the repo | [Official task guide](https://developers.google.com/edge/mediapipe/solutions/vision/hand_landmarker) links the bundle and [model card](https://storage.googleapis.com/mediapipe-assets/Model%20Card%20Hand%20Tracking%20(Lite_Full)%20with%20Fairness%20Oct%202021.pdf); model card page 2 states Apache-2.0 |
| `opencv-contrib-python` 5.0.0.93 | Camera capture, image processing, ArUco, homography, optical measurements; not an inference model | Apache-2.0 metadata; retain the wheel's bundled third-party notices |
| `numpy` 2.5.4 | Image/numeric arrays | Installed expression: BSD-3-Clause AND 0BSD AND MIT AND Zlib AND CC0-1.0; retain bundled notices |
| `fastapi` 0.162.0 | Local HTTP/WebSocket application | MIT metadata |
| `uvicorn` 0.54.0 | ASGI server | BSD-3-Clause metadata |
| `httpx` 0.28.1 | Model HTTP transport | BSD-3-Clause metadata; not MIT in this inspected version |
| `pydantic` 2.14.2 | Contracts and validation | MIT metadata |
| `meta-sam-parser` 0.1.0 | Parse/decode SAM response masks; no separate inference model | **SAM License**, dated November 19, 2025, in the installed package's `dist-info/licenses/LICENSE`; [upstream repository](https://github.com/meta-models/meta-sam) |
| Browser `speechSynthesis` | Spoken output only; commands are transcribed by Muse Voice | Browser/OS/voice-provider terms. No voice engine is distributed or relicensed here; installed voices can differ by device and may use network services |
| Three.js 0.169.0 (`lightwork/web/vendor/three.module.min.js`, vendored) | Draws Muse Wonder's 3D models on the projector page: solid parts, the slicing plane, the camera that goes inside. Not a model; no model comes from it | [MIT](https://github.com/mrdoob/three.js/blob/r169/LICENSE), notice kept in the file |
| MapLibre GL JS 4.7.1 (loaded from unpkg only when a map is asked for) | Draws the 3D map on the projector page. Not a model | [BSD-3-Clause](https://github.com/maplibre/maplibre-gl-js/blob/v4.7.1/LICENSE.txt) |
| OpenFreeMap "liberty" style and tiles, OpenMapTiles, OpenStreetMap data (fetched only when a map is asked for) | The map's look and its data, buildings included | [OpenFreeMap](https://openfreemap.org/), [OpenMapTiles](https://openmaptiles.org/) and [OpenStreetMap](https://www.openstreetmap.org/copyright) (ODbL) terms; attribution stays on screen with the map |
| llama.cpp / `llama-server` | OpenAI-compatible serving of Glimmer | [MIT](https://github.com/ggml-org/llama.cpp/blob/master/LICENSE); does not license the weights |

`meta-sam-parser` is a Meta library, listed here with the non-model dependencies for completeness. Its licence is not interchangeable with Apache-2.0: distribution must include the SAM agreement, and published research using SAM Materials must acknowledge that use. **Lightwork uses Meta SAM Materials through `meta-sam-parser` and hosted SAM inference.** Consult the agreement itself for restrictions and other obligations.

### Installed server/transport dependencies

The installed runtime also includes:

| Component | Role | Licence recorded in metadata/files |
|---|---|---|
| Starlette | FastAPI routing/ASGI | BSD-3-Clause |
| AnyIO, h11 | Async and HTTP primitives | MIT |
| websockets | WebSocket transport | BSD-3-Clause |
| httptools, watchfiles | Uvicorn HTTP parsing/development file watching | MIT |
| python-dotenv | Uvicorn environment-file support | BSD-3-Clause |
| PyYAML | Server configuration support | MIT |
| uvloop | Optional native event loop | Both MIT and Apache-2.0 licence files are installed; preserve both and bundled notices |

This is a role/licence inventory, not a replacement for all wheel notices or a complete software bill of materials. The lockfile and installed distribution metadata determine the exact transitive set for each platform. Optional packages and system tools can bring additional dependencies. `uv`, Python, Android `adb` and `pdftotext` are separately installed tools, not bundled binaries; their own distributors' terms apply.

## Meta models and hosted services

- Muse Wonder, Muse Spark, SAM and Muse Voice are accessed through Meta's service. Lightwork does not redistribute their hosted weights or grant rights to the service. Applicable Meta access and usage terms remain in force.
- Muse Glimmer weights and derived GGUF/adapter assets have their own model licence and provenance. See [open-weight lineage and reproduction](docs/open-weight.md). Do not infer weight rights from llama.cpp's MIT licence or this repository's Apache licence.
- The HandLandmarker weights are the explicitly disclosed non-Meta local helper; they do not replace the Meta completion evidence.
- Development stand-ins are isolated, opt-in development tools. Their responses and `runs-dev/` artifacts are not submitted evidence and must not enter the public run history.

## Procedure attribution

The following texts are **CC BY 4.0**, separately from the code. Preserve attribution, the [licence link](https://creativecommons.org/licenses/by/4.0/) and a description of changes when adapting them.

- [Kit return-check](procedures/kit-return-check/source.md), [LEGO build](procedures/lego-build/source.md), [breadboard LED](procedures/breadboard-led/source.md), and [tea dilution](procedures/tea-dilution/source.md): original procedure text and trial design by the Lightwork contributors, 2026. These do not reproduce product manuals. LEGO is a descriptive reference, not sponsorship; no LEGO instruction illustrations are included.
- [Serial dilution across row A](procedures/serial-dilution-row-a/source.md): adapted from **Baldwin G, Haddock-Angelli T, Beal J, Dwijayanti A, Storch M, Farny N, Telmer C, Vignoni A, Tennant R, Rutten P (2019), “Calibration Protocol – Plate Reader Fluorescence Calibration”, V.3**, protocols.io, [doi:10.17504/protocols.io.6zrhf56](https://doi.org/10.17504/protocols.io.6zrhf56), CC BY 4.0. Changes: blue food dye instead of fluorescein, water instead of PBS, one row only, camera reading rather than a plate reader, and fresh tips per transfer. Instructions are newly worded and no source figures are reproduced. The pack's source file gives the full adaptation context.

No iFixit text is included. Do not import instructions whose terms prohibit this use, or assume an accessible webpage is freely reusable. Model-authored text still requires source-rights review; a model response does not erase the underlying attribution obligation.

## Recordings and training data

The synthetic fixture is deliberately fabricated test data, not Meta output. Real frames, audio, trial labels, authoring sources and uploaded training material require the operator's rights/privacy review. A CC BY licence on a procedure does not automatically license photos taken while following it. Dataset build requires a separate explicit image/label licence and attribution; [training methodology](docs/training.md) explains the evidence filter and immutable provenance.
