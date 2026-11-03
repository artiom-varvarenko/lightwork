"""``lightwork partners``: which Meta open-weight providers are set up, and one recorded test call to each.

Lightwork runs Meta's open-weight models through the hackathon's inference partners or on our own hardware
(:mod:`lightwork.brain`): Muse Spark Open on Baseten or Fireworks AI, Muse Glimmer on a Modal GPU
(``scripts/modal_glimmer.py``) or on a Mac with llama.cpp. This sends each configured provider the same small vision
question with a generated test card and prints model, latency and answer. Every call is recorded under
``.lightwork/partner-checks/<time>/`` (request with the image by hash, raw response, ``meta.json``): the provenance the
open-weight prize's partner check asks for.

Keys go in ``.env``: ``BASETEN_API_KEY`` (``BASETEN_MODEL=meta-models/muse-spark-open-1026`` with the hackathon key),
``FIREWORKS_API_KEY``, ``MODAL_LLM_URL`` (+ ``MODAL_LLM_KEY``), ``LW_CHAT_BASE_URL`` (own hardware).
"""

from __future__ import annotations

COMMAND = "partners"
HELP = "Meta open-weight models via Baseten, Fireworks, Modal or own hardware: list them and make one recorded test call"

QUESTION = ("This is a test card on a white table, seen from above. What shape is drawn on it, what colour is it, and "
            "what word is written under it? Answer in one short sentence.")


def configure(p) -> None:
    p.add_argument("--only", help="test one provider: baseten | fireworks | modal | local")
    p.add_argument("--image", help="a JPEG to send instead of the generated test card")
    p.add_argument("--list", action="store_true", help="only list the configured providers")


def _card() -> bytes:
    import cv2
    import numpy as np

    img = np.full((360, 480, 3), 238, np.uint8)
    cv2.circle(img, (240, 150), 90, (40, 70, 220), -1)  # a red disc
    cv2.putText(img, "LIGHTWORK", (95, 310), cv2.FONT_HERSHEY_SIMPLEX, 1.6, (30, 30, 30), 4, cv2.LINE_AA)
    _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return buf.tobytes()


def run(args) -> int:
    from datetime import UTC, datetime
    from pathlib import Path

    from ..brain import BrainError, OpenBrain, providers
    from ..config import load_settings

    settings = load_settings()
    provs = providers(settings=settings)
    if args.only:
        provs = [p for p in provs if p.name == args.only]
    if not provs:
        print("no open-weight provider configured. Put one of these in .env:")
        print("  BASETEN_API_KEY=...      (Muse Spark Open; hackathon key: BASETEN_MODEL=meta-models/muse-spark-open-1026)")
        print("  FIREWORKS_API_KEY=...    (Muse Spark Open)")
        print("  MODAL_LLM_URL=https://<workspace>--lightwork-glimmer-serve.modal.run/v1   (scripts/modal_glimmer.py)")
        print("  LW_CHAT_BASE_URL=http://127.0.0.1:8080/v1   (Muse Glimmer with llama.cpp on your own machine)")
        return 2
    for p in provs:
        print(f"{p.name:10s} {p.label}: {p.model} at {p.base_url}")
    if args.list:
        return 0
    jpeg = Path(args.image).read_bytes() if args.image else _card()
    run_dir = settings.state_dir / "partner-checks" / datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    good = 0
    for p in provs:
        brain = OpenBrain([p], run_dir=run_dir)
        try:
            r = brain.ask("partner-check", QUESTION, [jpeg], effort="low", max_out=1500, timeout_s=180.0)
            good += 1
            print(f"ok   {p.name:10s} {r.model} {r.ms / 1000:.1f} s: {' '.join(r.text.split())[:160]}")
        except BrainError as exc:
            print(f"FAIL {p.name:10s} {p.model}: {exc}")
        finally:
            brain.close()
    print(f"recorded in {run_dir}")
    return 0 if good else 1
