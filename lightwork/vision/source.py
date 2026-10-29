"""Frame sources: one interface for the live camera (workstream A), video files and image sequences.

``read()`` returns the next :class:`Frame` or ``None`` at the end. ``t_mono`` is seconds on the producer's monotonic
clock; file sources synthesize it from ``fps`` (or take it from a timestamps list) so runs replay deterministically.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable

import cv2
import numpy as np


@dataclass(frozen=True)
class Frame:
    image: np.ndarray  # BGR uint8
    t_mono: float
    idx: int


@runtime_checkable
class FrameSource(Protocol):
    controlled: bool  # True when exposure/WB/focus are locked (live C920 on Linux)

    def read(self) -> Frame | None: ...

    def close(self) -> None: ...


class ImageSequenceSource:
    """Frames from in-memory images or image files, in order; optional per-frame timestamps."""

    controlled = True

    def __init__(self, images: Sequence[np.ndarray | str | Path], fps: float = 30.0, timestamps: Sequence[float] | None = None, loop: bool = False):
        if timestamps is not None and len(timestamps) != len(images):
            raise ValueError("timestamps must match images")
        self._images = list(images)
        self._ts = list(timestamps) if timestamps is not None else None
        self._fps = fps
        self._loop = loop
        self._i = 0
        self._n = 0

    def read(self) -> Frame | None:
        if self._i >= len(self._images):
            if not self._loop or not self._images:
                return None
            self._i = 0
        item = self._images[self._i]
        img = item if isinstance(item, np.ndarray) else cv2.imread(str(item), cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(str(item))
        t = self._ts[self._i] if self._ts is not None else self._n / self._fps
        f = Frame(image=img, t_mono=t, idx=self._n)
        self._i += 1
        self._n += 1
        return f

    def close(self) -> None:
        self._images = []

    def __iter__(self) -> Iterator[Frame]:
        while (f := self.read()) is not None:
            yield f


class VideoFileSource:
    """Frames from a video file via OpenCV; ``t_mono`` from the container position (or idx / fps)."""

    controlled = True

    def __init__(self, path: str | Path):
        self._cap = cv2.VideoCapture(str(path))
        if not self._cap.isOpened():
            raise FileNotFoundError(str(path))
        self._fps = self._cap.get(cv2.CAP_PROP_FPS) or 30.0
        self._n = 0

    def read(self) -> Frame | None:
        ok, img = self._cap.read()
        if not ok:
            return None
        f = Frame(image=img, t_mono=self._n / self._fps, idx=self._n)
        self._n += 1
        return f

    def close(self) -> None:
        self._cap.release()

    def __iter__(self) -> Iterator[Frame]:
        while (f := self.read()) is not None:
            yield f
