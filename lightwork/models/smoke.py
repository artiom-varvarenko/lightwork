"""Live smoke recorder (no Lightwork CLI/server needed).

    uv run python -m lightwork.models.smoke --backend chat --role verdict --image crop.jpg --out /tmp/model-smoke

Loads .env via config.load_settings. Writes only supplied work-area media, canonical requests, raw responses and C5
metadata. No dev cache; every invocation calls the configured model. One schema/length retry is recorded as a
separate fingerprinted call, just as the reducer does. Exit 0 = usable answer, 1 = failed/inconclusive, 2 = setup.
URL authoring sources are fetched/extracted by the authoring workstream before passing text to build_request;
--text-file consumes that text, while --pdf sends the PDF directly to Wonder.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

from lightwork.config import load_settings
from lightwork.contracts.canonical import sha256_hex

from .budget import Budget
from .client import ModelClient, run_provenance, timeout_for
from .errors import ModelAccessError, ModelError
from .parse import parse_response
from .requests import META_MODELS, ROLES, build_request


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--backend", choices=("meta", "chat"), required=True)
    p.add_argument("--role", choices=ROLES, default="verdict")
    p.add_argument("--model", help="Override the configured model id")
    p.add_argument("--image", type=Path, help="JPEG/PNG work-area crop, no room/faces")
    p.add_argument("--out", type=Path, help="Record root containing calls/ and frames/")
    p.add_argument("--timeout", type=float)
    p.add_argument("--instruction", default="Put the remote in its slot.")
    p.add_argument("--question", default="Is the remote inside the right-hand slot of the case?")
    p.add_argument("--expected", default="")
    p.add_argument("--detail", default="Each item needs its designated slot so the kit is complete.")
    p.add_argument("--nouns", nargs="+", default=["remote control"])
    p.add_argument("--shape", choices=("point", "box", "count"), default="box")
    p.add_argument("--concept")
    p.add_argument("--box", nargs=4, type=int, metavar=("X0", "Y0", "X1", "Y1"), help="Half-open box in sent-image px")
    p.add_argument("--point", nargs=2, action="append", type=int, help="Positive point in sent-image px (repeatable)")
    p.add_argument("--text-file", type=Path, help="Original/licensed procedure text, including extracted URL text")
    p.add_argument("--pdf", type=Path)
    p.add_argument("--notes", default="")
    return p


def _input(args) -> tuple[dict, str | None, dict[str, bytes], str]:
    frame_sha = None
    frames: dict[str, bytes] = {}
    extension = ".jpg"
    if args.role in ("verdict", "ground", "segment"):
        if args.image is None:
            raise ValueError(f"{args.role} needs --image")
        data = args.image.read_bytes()
        image = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("--image is not a decodable image")
        frame_sha = sha256_hex(data)
        frames[frame_sha] = data
        extension = ".png" if data.startswith(b"\x89PNG") else ".jpg"
    if args.role == "verdict":
        ctx = {"step_id": "smoke", "instruction": args.instruction, "question": args.question,
               "expected": args.expected}
    elif args.role == "ground":
        ctx = {"nouns": args.nouns, "shape": args.shape}
    elif args.role == "why":
        ctx = {"step_id": "smoke", "instruction": args.instruction, "detail": args.detail}
    elif args.role == "segment":
        if args.concept is not None:
            if args.box is not None or args.point:
                raise ValueError("choose a SAM noun OR box/point prompts, not both")
            ctx = {"concept": args.concept}
        else:
            if args.box is None and not args.point:
                raise ValueError("segment needs --concept, --box or --point")
            prompt = {"id": 0}
            if args.box is not None:
                prompt["box"] = args.box
            if args.point:
                prompt["points"] = args.point
            ctx = {"object_prompts": [prompt], "w": image.shape[1], "h": image.shape[0]}
    else:
        if bool(args.text_file) == bool(args.pdf):
            raise ValueError("author needs exactly one of --text-file or --pdf")
        ctx = {"notes": args.notes}
        if args.pdf:
            data = args.pdf.read_bytes()
            frame_sha = sha256_hex(data)
            frames[frame_sha] = data
            extension = ".pdf"
            ctx["filename"] = args.pdf.name
        else:
            ctx["text"] = args.text_file.read_text(encoding="utf-8")
    return ctx, frame_sha, frames, extension


def run(args) -> int:
    settings = load_settings()
    model = args.model or (META_MODELS[args.role] if args.backend == "meta" else settings.chat_model)
    if not model:
        raise ValueError("chat smoke needs LW_CHAT_MODEL or --model")
    ctx, frame_sha, frames, extension = _input(args)
    provenance, _ = run_provenance(settings, args.backend, {args.role: model})
    out = args.out or (settings.root / "runs-dev" / "smoke" if provenance == "dev-standin"
                       else settings.state_dir / "smoke")
    out.mkdir(parents=True, exist_ok=True)
    for sha, data in frames.items():
        (out / "frames").mkdir(exist_ok=True)
        (out / "frames" / f"{sha}{extension}").write_bytes(data)
    existing = [int(p.name.split("-", 1)[0]) for p in (out / "calls").glob("[0-9]*-*") if p.is_dir()]
    index = max(existing, default=0) + 1
    with ModelClient(settings, run_dir=out, provenance=provenance, budget=Budget.from_settings(settings),
                     cache=None) as client:
        for attempt in range(2):
            endpoint, request = build_request(args.role, model=model, backend=args.backend, frame_sha=frame_sha, ctx=ctx)
            result = client.execute(intent_id=f"smoke/{args.role}/{index}", role=args.role, endpoint=endpoint,
                                    request=request, frames=frames, index=index,
                                    timeout_s=args.timeout or timeout_for(args.role, args.backend))
            call_dir = out / result["call_dir"]
            response = next(call_dir.glob("response.*"))
            parsed = parse_response(args.role, response.read_bytes(), response.suffix[1:], endpoint)
            print(json.dumps({"output": str(out), "provenance": provenance, **result, "parsed": parsed},
                             sort_keys=True, ensure_ascii=False))
            if result["status"] != "ok":
                return 1
            if parsed["status"] == "ok":
                return 0
            if attempt or args.role == "segment":
                return 1
            if parsed["status"] == "invalid":
                ctx["retry"] = {"reason": "invalid", "previous": parsed["result"]["text"], "error": parsed["error"]}
            elif parsed["status"] == "incomplete":
                ctx["retry"] = {"reason": "length"}
            else:
                return 1
            index += 1
    return 1


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        return run(args)
    except ModelAccessError as e:
        print(str(e), file=sys.stderr)
        print(json.dumps(e.result, sort_keys=True), file=sys.stderr)
        return 2
    except (ModelError, OSError, ValueError) as e:
        print(str(e), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
