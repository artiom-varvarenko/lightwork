"""Voice acceptance runs that need hardware or the key (briefs/webvoice.md §5).

    uv run python -m lightwork.voice.check mic [--seconds 60] [--wav /tmp/v1.wav]   # web:V1, open /voice-check
    uv run python -m lightwork.voice.check live [--dry-run]                          # browser mic -> Muse Voice
    uv run python -m lightwork.voice.check stream cmds.wav --expect cmds.tsv         # web:V3 (+ V5 latency)

``mic`` needs no key: it measures the browser's ingress rate (pass: 48,000 B/s ±1 %) and saves the first 10 s as a
WAV to listen to. ``live`` relays the browser mic to Muse Voice Transcribe and prints HUD changes and events.
``stream`` plays a mono 16-bit 24 kHz WAV through the relay in real time. ``--expect`` is a TSV of
``<start_s>\\t<label>`` per utterance, label = a command or ``-`` for a distractor; each event is attributed to the
last utterance that started before it. Pass: >= 96 % of commands right, 0 wrong commands, 0 distractors accepted.
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
import wave
from pathlib import Path

from ..config import load_settings
from .grammar import COMMANDS
from .pcm import BYTES_PER_S, FRAME_BYTES, FRAME_S, wav_bytes
from .service import VoiceService

PAGE = """<!doctype html><meta charset="utf-8"><title>Lightwork voice check</title>
<body style="font:20px sans-serif;background:#111;color:#eee;padding:2em">
<button id="go" style="font-size:28px">Start mic</button> <span id="st">idle</span>
<script type="module">
import { startMic } from "/static/js/mic.js";
document.getElementById("go").onclick = () => startMic({
  onState: (s) => (document.getElementById("st").textContent = `${s.state} ${s.label || ""} ${s.error || ""}`),
});
</script>"""


class MicProbe:
    """``feed_pcm`` sink that measures the ingress rate and keeps the first ``keep_s`` seconds."""

    def __init__(self, keep_s: float = 10.0):
        self.keep = int(keep_s * BYTES_PER_S)
        self.pcm = bytearray()
        self.t0: float | None = None
        self.t_last = 0.0
        self.total = 0
        self.lock = threading.Lock()

    def feed_pcm(self, frame: bytes) -> None:
        now = time.monotonic()
        with self.lock:
            if self.t0 is None:
                self.t0 = now
            else:
                self.total += len(frame)  # bytes after the first frame, over t_last - t0
            self.t_last = now
            if len(self.pcm) < self.keep:
                self.pcm += frame[: self.keep - len(self.pcm)]

    def rate(self) -> float:
        with self.lock:
            dt = self.t_last - (self.t0 or self.t_last)
            return self.total / dt if dt > 0 else 0.0


def _serve(sink, port: int):
    import uvicorn
    from fastapi.responses import HTMLResponse

    from ..server.app import create_app

    app = create_app(voice=sink)
    app.add_api_route("/voice-check", lambda: HTMLResponse(PAGE), methods=["GET"])
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    print(f"open http://localhost:{port}/voice-check and press Start mic", flush=True)
    return server, thread


def cmd_mic(args) -> int:
    probe = MicProbe()
    server, thread = _serve(probe, args.port)
    try:
        while probe.t0 is None:
            time.sleep(0.1)
        while time.monotonic() - probe.t0 < args.seconds:
            time.sleep(min(10.0, args.seconds))
            print(f"{time.monotonic() - probe.t0:5.1f} s  {probe.rate():9.1f} B/s", flush=True)
    except KeyboardInterrupt:
        pass
    finally:
        server.should_exit = True
        thread.join(5)
    rate = probe.rate()
    Path(args.wav).write_bytes(wav_bytes(bytes(probe.pcm)))
    ok = abs(rate - BYTES_PER_S) <= 0.01 * BYTES_PER_S
    print(f"V1 ingress {rate:.1f} B/s (target 48000 ±1%): {'PASS' if ok else 'FAIL'}; first 10 s -> {args.wav}")
    return 0 if ok else 1


def _service(args, on_event) -> VoiceService:
    settings = load_settings()
    if not settings.model_api_key:
        raise SystemExit("MODEL_API_KEY is not set (.env or environment)")

    def hud(h: dict) -> None:
        print(f"HUD voice={h['voice']} turn={h['turn']} heard={h['heard']!r} command={h['command']} "
              f"error={h['error']}", flush=True)

    return VoiceService(settings, args.keyword, on_event, dry_run=args.dry_run, on_hud=hud,
                        mode="file" if args.file_mode else "realtime")


def cmd_live(args) -> int:
    svc = _service(args, lambda t, p: print(f"EVENT {t} {json.dumps(p)}", flush=True))
    svc.start()
    server, thread = _serve(svc, args.port)
    try:
        while thread.is_alive():
            time.sleep(0.5)
    except KeyboardInterrupt:
        pass
    finally:
        server.should_exit = True
        thread.join(5)
        svc.close()
    print(f"latencies_ms={svc.stats()['latencies_ms']}")
    return 0


def _read_wav(path: str) -> bytes:
    with wave.open(path, "rb") as w:
        if (w.getnchannels(), w.getsampwidth(), w.getframerate()) != (1, 2, 24_000):
            raise SystemExit(f"{path}: need mono 16-bit 24 kHz; convert with "
                             f"ffmpeg -i in.wav -ac 1 -ar 24000 -c:a pcm_s16le out.wav")
        return w.readframes(w.getnframes())


def score(expect: list[tuple[float, str]], events: list[tuple[float, str, dict]]) -> dict:
    """Attribute events to utterances (last start <= event position) and count V3 outcomes."""
    got: list[list[str | None]] = [[] for _ in expect]
    for pos, type_, payload in events:
        idx = max((i for i, (start, _) in enumerate(expect) if start <= pos), default=None)
        if idx is not None:
            got[idx].append(payload["command"] if type_ == "voice" else None)
    commands = [i for i, (_, label) in enumerate(expect) if label != "-"]
    correct = sum(1 for i in commands if got[i].count(expect[i][1]) == 1
                  and all(c in (None, expect[i][1]) for c in got[i]))
    wrong = sum(1 for i in commands for c in got[i] if c is not None and c != expect[i][1])
    wrong += sum(1 for i in commands if got[i].count(expect[i][1]) > 1)
    distractors = sum(1 for i, (_, label) in enumerate(expect) if label == "-" for c in got[i] if c is not None)
    ok = correct >= 0.96 * len(commands) and wrong == 0 and distractors == 0
    return {"commands": len(commands), "correct": correct, "wrong": wrong,
            "distractors": len(expect) - len(commands), "distractors_accepted": distractors, "pass": ok,
            "per_utterance": [{"start_s": s, "expect": label, "got": g}
                              for (s, label), g in zip(expect, got, strict=True)]}


def cmd_stream(args) -> int:
    pcm = _read_wav(args.wav)
    expect: list[tuple[float, str]] = []
    if args.expect:
        for line in Path(args.expect).read_text().splitlines():
            if line.strip() and not line.startswith("#"):
                start, label = line.split("\t")[:2]
                if label != "-" and label not in COMMANDS:
                    raise SystemExit(f"{args.expect}: unknown label {label!r}")
                expect.append((float(start), label))
    events: list[tuple[float, str, dict]] = []
    t0 = [0.0]

    def on_event(type_: str, payload: dict) -> None:
        pos = time.monotonic() - t0[0]
        events.append((pos, type_, payload))
        print(f"{pos:7.2f} s  EVENT {type_} {json.dumps(payload)}", flush=True)

    svc = _service(args, on_event)
    svc.start()
    deadline = time.monotonic() + 15
    while svc.hud["voice"] not in ("live", "fallback") and time.monotonic() < deadline:
        time.sleep(0.05)
    if svc.hud["voice"] not in ("live", "fallback"):
        svc.close()
        print(f"voice did not come up: {svc.hud}")
        return 2
    pcm += bytes(int(4 * BYTES_PER_S))  # trailing silence so the last turn completes
    t0[0] = t = time.monotonic()
    for i in range(0, len(pcm), FRAME_BYTES):
        t += FRAME_S
        time.sleep(max(0.0, t - time.monotonic()))
        svc.feed_pcm(pcm[i:i + FRAME_BYTES].ljust(FRAME_BYTES, b"\0"))
    svc.close()
    lat = sorted(svc.stats()["latencies_ms"])
    if lat:
        print(f"V5 end-of-speech -> speechComplete: p50 {lat[len(lat) // 2]} ms, "
              f"p95 {lat[min(len(lat) - 1, int(0.95 * len(lat)))]} ms (n={len(lat)})")
    if not expect:
        return 0
    result = score(expect, events)
    print(json.dumps({k: v for k, v in result.items() if k != "per_utterance"}))
    for row in result["per_utterance"]:
        print(f"  {row['start_s']:7.2f}  {row['expect']:7}  {row['got']}")
    return 0 if result["pass"] else 1


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m lightwork.voice.check", description=__doc__.split("\n\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("mic", help="web:V1 browser ingress rate (no key)")
    m.add_argument("--seconds", type=float, default=60.0)
    m.add_argument("--wav", default="/tmp/lightwork-v1.wav")
    m.add_argument("--port", type=int, default=8000)
    for name in ("live", "stream"):
        s = sub.add_parser(name)
        s.add_argument("--keyword", action="append", default=[], help="extra noun keyword (repeatable)")
        s.add_argument("--dry-run", action="store_true")
        s.add_argument("--file-mode", action="store_true", help="D14 file endpoint instead of realtime")
        if name == "live":
            s.add_argument("--port", type=int, default=8000)
        else:
            s.add_argument("wav")
            s.add_argument("--expect")
    args = p.parse_args(argv)
    return {"mic": cmd_mic, "live": cmd_live, "stream": cmd_stream}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
