"""Live run recorder (C6, PLAN.md §4.2).

The orchestrator appends input events. With a reducer attached, :meth:`Recorder.ingest` appends an input, feeds it to
the reducer and appends the derived events (``src="reducer"``, the input's ``t``) under one lock: exactly the order
``verify`` re-derives. ``t`` never decreases along ``seq``; an input that arrives late takes the previous event's ``t``.

Crash safety: each event line is fsynced before ``append`` returns, frames and previews are written atomically, and
``manifest.json`` is written only by :meth:`Recorder.close`; a run directory without it is an incomplete run.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from pydantic import BaseModel

from ..contracts.calls import call_dirname
from ..contracts.canonical import canonical, canonical_sha, q_s, sha256_hex
from ..contracts.events import DERIVED_TYPES, Event, Payload, genesis_hash, read_events, seal
from ..contracts.run import Manifest, Provenance
from .manifest import RUN_ID_RE, build_manifest, size_class
from .store import EVENTS, MANIFEST, RUN_INPUT_FILES, DirStore, write_atomic

log = logging.getLogger(__name__)

RUN_DIRS: dict[str, str] = {"meta": "runs", "dev-standin": "runs-dev", "synthetic": "runs-synthetic"}
PREVIEW_MAX_H = 480
PREVIEW_MIN_INTERVAL_S = 1.0
PREVIEW_QUALITY = 70


def new_run_id(pack: str, now: datetime | None = None) -> str:
    return f"{(now or datetime.now(UTC)):%Y%m%dT%H%M%SZ}_{pack}"


class Recorder:
    """One run under ``<root>/runs/``, ``runs-dev/`` or ``runs-synthetic/`` according to its provenance.

    ``run_started`` is the C1 payload; ``run_id`` (``<YYYYMMDDTHHMMSSZ>_<pack>``, from the UTC clock) and ``pack``
    are filled in when absent. Thread-safe.
    """

    def __init__(
        self,
        root: Path,
        pack: str,
        run_started: Mapping[str, Any],
        *,
        provenance: Provenance,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if provenance not in RUN_DIRS:
            raise ValueError(f"unknown provenance {provenance!r}")
        started = {"pack": pack, **run_started}
        started.setdefault("run_id", new_run_id(pack))
        m = RUN_ID_RE.match(started["run_id"])
        if started["pack"] != pack or not m or m.group(7) != pack:
            raise ValueError(f"run id {started['run_id']!r} / pack {started['pack']!r} do not match pack {pack!r}")
        self.root = Path(root)
        self.run_id: str = started["run_id"]
        self.provenance: Provenance = provenance
        self.run_dir = self.root / RUN_DIRS[provenance] / self.run_id
        self.reducer: Any = None
        self.manifest: Manifest | None = None
        self.run_dir.mkdir(parents=True, exist_ok=False)
        for sub in ("frames", "preview", "calls"):
            (self.run_dir / sub).mkdir()
        self._lock = threading.RLock()
        self._clock = clock
        self._t0 = clock()
        self._f = open(self.run_dir / EVENTS, "ab")
        self._seq = 0
        self._prev_h = genesis_hash(self.run_id)
        self._last_t = 0.0
        self._calls = 0
        self._preview_at: tuple[float, int] | None = None  # (t, seq) of the last stored preview
        self.started = self.append("run_started", "orchestrator", started, 0.0)

    @property
    def store(self) -> DirStore:
        return DirStore(self.run_dir)

    def now(self) -> float:
        """Seconds since the run started (quantized), on the recorder's monotonic clock."""
        return q_s(self._clock() - self._t0)

    def run_time(self, t_mono: float) -> float:
        """Convert a capture timestamp on the same monotonic clock (``Frame.t_mono``) to run time."""
        return q_s(t_mono - self._t0)

    # ---------------------------------------------------------------- events

    def _assert_open(self) -> None:
        if self._f.closed:
            raise RuntimeError(f"run {self.run_id} is closed")

    def append(self, type_: str, src: str, payload: Mapping[str, Any] | Payload, t: float) -> Event:
        """Seal and durably append one event. Derived types come from ``src="reducer"``, inputs never do."""
        if (type_ in DERIVED_TYPES) != (src == "reducer"):
            raise ValueError(f"{type_!r} with src {src!r}: derived events have src 'reducer', inputs never do")
        with self._lock:
            self._assert_open()
            t = max(q_s(t), self._last_t)
            ev = seal(self._prev_h, self._seq + 1, t, type_, src, payload)
            self._f.write(ev.line().encode("utf-8") + b"\n")
            self._f.flush()
            os.fsync(self._f.fileno())
            self._seq, self._prev_h, self._last_t = ev.seq, ev.h, t
            return ev

    def attach(self, reducer: Any) -> list[Event]:
        """Attach the run's reducer right after the run started and derive from ``run_started``."""
        with self._lock:
            if self.reducer is not None or self._seq != 1:
                raise RuntimeError("attach the reducer once, before any event after run_started")
            self.reducer = reducer
            return self._derive(self.started)

    def ingest(self, type_: str, src: str, payload: Mapping[str, Any] | Payload, t: float) -> list[Event]:
        """Append an input event and the derived events the attached reducer returns for it."""
        if type_ in DERIVED_TYPES:
            raise ValueError(f"{type_!r} is a derived type")
        with self._lock:
            if self.reducer is None:
                raise RuntimeError("no reducer attached")
            ev = self.append(type_, src, payload, t)
            return [ev, *self._derive(ev)]

    def _derive(self, ev: Event) -> list[Event]:
        return [self.append(type_, "reducer", payload, ev.t) for type_, payload in self.reducer.feed(ev)]

    # ---------------------------------------------------------------- files

    def put_run_file(self, name: str, obj: BaseModel) -> str:
        """Write ``stepgraph.json``, ``layout.json`` or ``calib.json``; returns the canonical sha for its
        ``*_loaded`` event."""
        model = RUN_INPUT_FILES.get(name)
        if model is None or not isinstance(obj, model):
            raise TypeError(f"{name}: expected one of {sorted(RUN_INPUT_FILES)} with its contract model")
        with self._lock:
            self._assert_open()
            write_atomic(self.run_dir / name, (canonical(obj) + "\n").encode("utf-8"))
        return canonical_sha(obj)

    def put_frame(self, jpeg: bytes) -> str:
        """Store a keyframe as ``frames/<sha>.jpg`` (content-addressed, idempotent); returns its sha."""
        sha = sha256_hex(jpeg)
        with self._lock:
            self._assert_open()
            path = self.run_dir / "frames" / f"{sha}.jpg"
            if not path.exists():
                write_atomic(path, jpeg)
        return sha

    def put_preview(self, crop_bgr: np.ndarray, t: float) -> str | None:
        """Store a work-area crop (D10) as ``preview/<seq>.jpg``, ``<seq>`` = the last appended event, scaled to at
        most 480 px tall. At most one preview per seq and per second; returns the relative path, or None if skipped."""
        t = q_s(t)
        with self._lock:
            self._assert_open()
            last = self._preview_at
            if last is not None and (last[1] == self._seq or t - last[0] < PREVIEW_MIN_INTERVAL_S):
                return None
            h, w = crop_bgr.shape[:2]
            if h > PREVIEW_MAX_H:
                size = (max(1, round(w * PREVIEW_MAX_H / h)), PREVIEW_MAX_H)
                crop_bgr = cv2.resize(crop_bgr, size, interpolation=cv2.INTER_AREA)
            ok, buf = cv2.imencode(".jpg", crop_bgr, [cv2.IMWRITE_JPEG_QUALITY, PREVIEW_QUALITY])
            if not ok:
                raise ValueError("preview JPEG encoding failed")
            rel = f"preview/{self._seq}.jpg"
            write_atomic(self.run_dir / rel, buf.tobytes())
            self._preview_at = (t, self._seq)
            return rel

    def alloc_call(self, role: str, model: str) -> tuple[int, str]:
        """Next call index and its directory relative to the run (``calls/<NNNN>-<role>-<model>``, C5)."""
        with self._lock:
            self._assert_open()
            self._calls += 1
            return self._calls, f"calls/{call_dirname(self._calls, role, model)}"

    # ---------------------------------------------------------------- close

    def close(self, reason: str) -> Path:
        """Append ``run_stopped`` and write the manifest. Non-Meta or generated calls quarantine the run in
        ``runs-dev/`` or ``runs-synthetic/``. A reducer failure leaves no manifest. Idempotent."""
        with self._lock:
            if self._f.closed:
                return self.run_dir
            try:
                if self.reducer is None:
                    self.append("run_stopped", "orchestrator", {"reason": reason}, self.now())
                else:
                    self.ingest("run_stopped", "orchestrator", {"reason": reason}, self.now())
            finally:
                self._f.close()
            self._seal()
            return self.run_dir

    def _seal(self) -> None:
        provenance = self._recorded_provenance()
        if provenance != self.provenance:
            target = self.root / RUN_DIRS[provenance] / self.run_id
            target.parent.mkdir(parents=True, exist_ok=True)
            os.replace(self.run_dir, target)
            log.warning("run %s has %s evidence: moved to %s", self.run_id, provenance, target)
            self.run_dir = target
        self.provenance = provenance
        events = read_events(self.run_dir / EVENTS)
        self.manifest = build_manifest(self.run_dir, events, provenance)
        write_atomic(self.run_dir / MANIFEST, (canonical(self.manifest) + "\n").encode("utf-8"))
        size = self.manifest.size_bytes
        log_size = log.info if size_class(size) == "ok" else log.warning
        log_size("run %s is %.1f MB (%s; target 10 MB, hard limit 25 MB)", self.run_id, size / 2**20, size_class(size))

    def _recorded_provenance(self) -> Provenance:
        found = {self.provenance}
        if self.started.payload["platform"] == "synthetic":
            found.add("synthetic")
        for call in (self.run_dir / "calls").iterdir():
            if not call.is_dir():
                continue
            try:
                meta = json.loads((call / "meta.json").read_bytes())
                found.add(meta.get("provenance", "dev-standin"))
                if str(meta.get("response_id") or "").startswith(("resp_synthetic", "chatcmpl-synthetic")):
                    found.add("synthetic")
            except (OSError, ValueError, AttributeError, TypeError):
                found.add("dev-standin")
        if "synthetic" in found:
            return "synthetic"
        return "meta" if found == {"meta"} else "dev-standin"
