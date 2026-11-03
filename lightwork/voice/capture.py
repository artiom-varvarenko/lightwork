"""A microphone read by the server itself, for the voice relay: no browser page and no Start button needed.

``pw-record`` (PipeWire) streams the chosen source - by default the C920's microphone, which sits on the tripod
right above the table - as PCM16 LE mono 24 kHz to stdout; a thread cuts it into 80 ms frames for
:meth:`lightwork.voice.service.VoiceService.feed_pcm`. A browser mic (``/ws/mic``) takes over while it streams:
two sources must never interleave. Recorded on 3 Nov: 2.28 s from the C920 through ``pw-record ... -`` arrived as
raw samples (no header). Run inside the user service: a real-time thread in a shell with RLIMIT_RTTIME=0 is killed.
"""

from __future__ import annotations

import shutil
import subprocess
import threading
import time

FRAME_BYTES = 3840  # 80 ms of PCM16 LE mono at 24 kHz, what the relay paces


def sources() -> list[str]:
    """PipeWire/PulseAudio capture sources (no monitors)."""
    try:
        out = subprocess.run(["pactl", "list", "short", "sources"], capture_output=True, text=True, timeout=3).stdout
    except (OSError, subprocess.SubprocessError):
        return []
    names = [line.split("\t")[1] for line in out.splitlines() if len(line.split("\t")) > 1]
    return [n for n in names if not n.endswith(".monitor")]


def find_source(pattern: str = "C920") -> str | None:
    """The first capture source whose name contains ``pattern`` (case-insensitive)."""
    return next((n for n in sources() if pattern.lower() in n.lower()), None)


class ServerMic:
    def __init__(self, sink, source: str, *, yield_to=None, rate: int = 24000):
        self.sink = sink  # callable(bytes): one 80 ms frame
        self.source = source
        self.yield_to = yield_to  # callable() -> bool: a browser mic is streaming, skip ours
        self.rate = rate
        self.frames = 0
        self.skipped = 0
        self.error: str | None = None
        self.proc: subprocess.Popen | None = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()

    def start(self) -> str | None:
        """Start capturing; returns a problem or None."""
        exe = shutil.which("pw-record")
        if exe is None:
            return "pw-record not found (PipeWire)"
        cmd = [exe, "--target", self.source, "--format", "s16", "--rate", str(self.rate), "--channels", "1", "-"]
        try:
            self.proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=0)
        except OSError as exc:
            return f"pw-record: {exc}"
        self._thread = threading.Thread(target=self._loop, name="server-mic", daemon=True)
        self._thread.start()
        return None

    def _loop(self) -> None:
        buf = bytearray()
        out = self.proc.stdout
        while not self._stop.is_set():
            chunk = out.read(FRAME_BYTES - len(buf))
            if not chunk:
                self.error = f"pw-record ended (code {self.proc.poll()})"
                return
            buf += chunk
            if len(buf) < FRAME_BYTES:
                continue
            frame, buf = bytes(buf), bytearray()
            if self.yield_to is not None and self.yield_to():
                self.skipped += 1
                continue
            try:
                self.sink(frame)
                self.frames += 1
            except Exception as exc:  # noqa: BLE001 - a sink error must not kill capture
                self.error = f"{type(exc).__name__}: {exc}"

    def status(self) -> str:
        s = f"{self.source}: {self.frames * 0.08:.0f} s of audio"
        return s + (f" ({self.error})" if self.error else "")

    def close(self) -> None:
        self._stop.set()
        if self.proc is not None and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                self.proc.kill()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        time.sleep(0)
