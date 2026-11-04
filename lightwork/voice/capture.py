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
from collections import deque

from .pcm import level_dbfs

FRAME_BYTES = 3840  # 80 ms of PCM16 LE mono at 24 kHz, what the relay paces
PRE_ROLL_FRAMES = 4  # the mic gate hands on this much from just before it opened: words start before the finger lands
MAX_HELD_BYTES = 60 * 48000  # the audio of one hold kept for its report and recording: a minute
RESTART_S = 1.0  # after pw-record ended by itself


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


def hold_report(pcm: bytes, seconds: float) -> str:
    """One line on what one hold of the mic delivered: enough to tell silence from a dead microphone afterwards."""
    if not pcm:
        return f"{seconds:.1f} s open, NO AUDIO reached the gate (is pw-record running?)"
    frames = [pcm[i:i + FRAME_BYTES] for i in range(0, len(pcm), FRAME_BYTES)]
    levels = [level_dbfs(f) for f in frames]
    return (f"{seconds:.1f} s open, {len(pcm) / 48000:.1f} s of audio, level {sum(levels) / len(levels):.0f} dBFS on "
            f"average, {max(levels):.0f} at its loudest")


class MicGate:
    """Between a microphone and the relay: frames pass only while it is open, so table talk goes nowhere (3 Nov: an
    always-open mic took everything said at the table, Russian chat included, as input). While it is shut the relay's
    pacer sends silence, which also ends the turn that was being spoken. Thread-safe: frames come from the mic's
    thread, ``set_open`` from the frame loop."""

    def __init__(self, sink, pre_roll: int = PRE_ROLL_FRAMES, *, keep_pre_roll=None, on_hold=None):
        self.sink = sink  # callable(bytes): one 80 ms frame
        self.keep_pre_roll = keep_pre_roll  # callable() -> bool; False: the moment before holds the phone's voice, drop it
        self.on_hold = on_hold  # callable(pcm, seconds): the audio of one hold, when it closes
        self.opened = 0  # times it opened
        self._open = False
        self._opened_t = 0.0
        self._closed_t = -1e9
        self._held = bytearray()
        self._recent: deque[bytes] = deque(maxlen=max(pre_roll, 1))
        self._lock = threading.Lock()

    @property
    def is_open(self) -> bool:
        with self._lock:
            return self._open

    def recently_open(self, within_s: float) -> bool:
        """Open now, or shut for less than this long: a transcript arriving now can be of words said into it."""
        with self._lock:
            return self._open or time.monotonic() - self._closed_t < within_s

    def _pass(self, frame: bytes) -> None:
        self.sink(frame)
        if len(self._held) < MAX_HELD_BYTES:
            self._held += frame

    def set_open(self, on: bool) -> None:
        hold = None
        with self._lock:
            if on and not self._open:
                self.opened += 1
                self._opened_t = time.monotonic()
                self._held.clear()
                if self.keep_pre_roll is None or self.keep_pre_roll():
                    for frame in self._recent:
                        self._pass(frame)
                self._recent.clear()
            elif self._open and not on:
                self._closed_t = time.monotonic()
                hold = (bytes(self._held), self._closed_t - self._opened_t)
                self._held.clear()
            self._open = on
        if hold is not None and self.on_hold is not None:
            self.on_hold(*hold)

    def __call__(self, frame: bytes) -> None:
        with self._lock:
            if self._open:
                self._pass(frame)
            else:
                self._recent.append(frame)


class ServerMic:
    def __init__(self, sink, source: str, *, yield_to=None, rate: int = 24000):
        self.sink = sink  # callable(bytes): one 80 ms frame
        self.source = source
        self.yield_to = yield_to  # callable() -> bool: a browser mic is streaming, skip ours
        self.rate = rate
        self.frames = 0
        self.skipped = 0
        self.restarts = 0
        self.error: str | None = None
        self.proc: subprocess.Popen | None = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()

    def start(self) -> str | None:
        """Start capturing; returns a problem or None."""
        problem = self._spawn()
        if problem:
            return problem
        self._thread = threading.Thread(target=self._loop, name="server-mic", daemon=True)
        self._thread.start()
        return None

    def _spawn(self) -> str | None:
        exe = shutil.which("pw-record")
        if exe is None:
            return "pw-record not found (PipeWire)"
        # `systemctl kill -s USR1|USR2 lightwork-play` (recalibrate, snapshot) goes to every process of the unit, and those
        # signals end a program that has no handler. 4 Nov: one snapshot left the table deaf for the rest of the run. The
        # shell's "ignore" survives its exec, and the pid stays pw-record's.
        cmd = ["/bin/sh", "-c", 'trap "" USR1 USR2; exec "$@"', "pw-record", exe, "--target", self.source,
               "--format", "s16", "--rate", str(self.rate), "--channels", "1", "-"]
        try:
            self.proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=0)
        except OSError as exc:
            return f"pw-record: {exc}"
        return None

    def _loop(self) -> None:
        while not self._stop.is_set():
            self._read()
            if self._stop.is_set():
                return
            try:
                code = self.proc.wait(timeout=1.0)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                code = "killed"
            self.error = f"pw-record ended (code {code})"
            print(f"microphone: {self.error}; starting it again")  # a table that cannot hear must not stay that way
            if self._stop.wait(RESTART_S):
                return
            self.restarts += 1
            problem = self._spawn()
            if problem:
                self.error = problem
                return

    def _read(self) -> None:
        buf = bytearray()
        out = self.proc.stdout
        while not self._stop.is_set():
            chunk = out.read(FRAME_BYTES - len(buf))
            if not chunk:
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
        if self.restarts:
            s += f", restarted {self.restarts} time{'' if self.restarts == 1 else 's'}"
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
