"""PCM16 LE mono 24 kHz helpers: the mic ring buffer, an RMS voice-activity detector and WAV packing."""

from __future__ import annotations

import io
import threading
import wave
from dataclasses import dataclass

import numpy as np

RATE = 24_000
BYTES_PER_S = RATE * 2
FRAME_S = 0.08
FRAME_BYTES = int(BYTES_PER_S * FRAME_S)  # 3840
MAX_BACKLOG_BYTES = BYTES_PER_S // 2  # 0.5 s


class PcmRing:
    """Thread-safe FIFO of PCM bytes that keeps only the newest ``max_bytes`` (sample-aligned)."""

    def __init__(self, max_bytes: int = MAX_BACKLOG_BYTES):
        if max_bytes % 2:
            raise ValueError("max_bytes must be sample-aligned")
        self._max = max_bytes
        self._buf = bytearray()
        self._lock = threading.Lock()
        self.dropped_bytes = 0

    def push(self, data: bytes) -> None:
        if len(data) % 2:
            data = data[:-1]
        with self._lock:
            self._buf += data
            over = len(self._buf) - self._max
            if over > 0:
                del self._buf[:over]
                self.dropped_bytes += over

    def pop(self, n: int) -> bytes:
        """Exactly ``n`` bytes: the oldest buffered audio, zero-padded on underflow."""
        with self._lock:
            out = bytes(self._buf[:n])
            del self._buf[:n]
        return out + bytes(n - len(out))

    def clear(self) -> None:
        with self._lock:
            self._buf.clear()

    def __len__(self) -> int:
        with self._lock:
            return len(self._buf)


def level_dbfs(frame: bytes) -> float:
    samples = np.frombuffer(frame, dtype="<i2").astype(np.float64)
    if samples.size == 0:
        return -120.0
    rms = float(np.sqrt(np.mean(samples * samples)))
    return 20.0 * float(np.log10(rms / 32768.0)) if rms > 0 else -120.0


@dataclass(frozen=True)
class Utterance:
    pcm: bytes
    t_start: float  # time of the first voiced frame
    t_end: float  # time of the last voiced frame


class RmsVad:
    """Energy VAD over 80 ms frames with an adaptive noise floor.

    A frame is voiced when it is ``margin_db`` above the noise floor and above ``min_db``. Speech starts after
    ``start_frames`` voiced frames in a row and ends after ``hang_frames`` unvoiced ones. Utterances whose voiced span
    is outside ``[min_s, max_s]`` are discarded (commands are single words). ``last_end_t`` is the time of the last
    voiced frame of the most recent speech segment, kept or not (V5 latency reference).
    """

    def __init__(self, *, margin_db=12.0, min_db=-50.0, start_frames=2, hang_frames=6, min_s=0.3, max_s=4.0,
                 preroll_frames=2):
        self.margin_db = margin_db
        self.min_db = min_db
        self.start_frames = start_frames
        self.hang_frames = hang_frames
        self.min_s = min_s
        self.max_s = max_s
        self.preroll_frames = preroll_frames
        self.floor_db = -70.0
        self.last_end_t: float | None = None
        self._pre: list[bytes] = []
        self._run = 0
        self._quiet = 0
        self._speaking = False
        self._frames: list[bytes] = []
        self._t_start = 0.0
        self._t_last = 0.0
        self._too_long = False

    def push(self, frame: bytes, t: float) -> Utterance | None:
        db = level_dbfs(frame)
        voiced = db >= max(self.floor_db + self.margin_db, self.min_db)
        if not voiced and not self._speaking:
            self.floor_db = max(-70.0, 0.95 * self.floor_db + 0.05 * db)
        if not self._speaking:
            self._run = self._run + 1 if voiced else 0
            self._pre.append(frame)
            if self._run >= self.start_frames:
                self._speaking = True
                self._frames = list(self._pre)
                self._t_start = t - (self.start_frames - 1) * FRAME_S
                self._t_last = t
                self._quiet = 0
                self._too_long = False
            del self._pre[: max(0, len(self._pre) - self.preroll_frames - self.start_frames)]
            return None
        if not self._too_long:
            self._frames.append(frame)
        if voiced:
            self._t_last = t
            self._quiet = 0
            if self._t_last - self._t_start + FRAME_S > self.max_s:
                self._too_long = True
                self._frames = []
            return None
        self._quiet += 1
        if self._quiet < self.hang_frames:
            return None
        self._speaking = False
        self._run = 0
        self._pre = []
        self.last_end_t = self._t_last
        span = self._t_last - self._t_start + FRAME_S
        frames, self._frames = self._frames, []
        if self._too_long or span < self.min_s:
            return None
        return Utterance(b"".join(frames), self._t_start, self._t_last)


def wav_bytes(pcm: bytes, rate: int = RATE) -> bytes:
    out = io.BytesIO()
    with wave.open(out, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(pcm)
    return out.getvalue()
