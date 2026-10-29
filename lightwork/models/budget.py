"""D21 hard-cap ledger at ``.lightwork/spend.json``.

Reservations are persisted under an OS file lock, so concurrent clients/processes cannot overbook the cap. A
crashed process leaves its reservation held rather than silently spending again. Investigate the recorded call
before releasing a stranded reservation. Unknown billed usage stays reserved as ``unresolved_usd``; it is never
invented as actual spend. ``pricing.estimate_usd`` uses the model's context limit, not a heuristic image estimate.
"""

from __future__ import annotations

import json
import math
import os
import tempfile
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

from .errors import BudgetExceeded
from .pricing import estimate_usd

SPEND_FILE = "spend.json"


class Budget:
    def __init__(self, path: Path, cap_usd: float):
        self.path = Path(path)
        self.cap_usd = float(cap_usd)
        if not math.isfinite(self.cap_usd) or self.cap_usd < 0:
            raise ValueError("budget cap must be finite and non-negative")

    @classmethod
    def from_settings(cls, settings) -> Budget:
        return cls(settings.state_dir / SPEND_FILE, settings.budget_usd)

    @contextmanager
    def _locked(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.with_suffix(".lock").open("a+b") as lock:
            if os.name == "nt":
                import msvcrt
                if lock.tell() == 0:
                    lock.write(b"\0")
                    lock.flush()
                lock.seek(0)
                msvcrt.locking(lock.fileno(), msvcrt.LK_LOCK, 1)
            else:
                import fcntl
                fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                if os.name == "nt":
                    lock.seek(0)
                    msvcrt.locking(lock.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    fcntl.flock(lock.fileno(), fcntl.LOCK_UN)

    @property
    def spent_usd(self) -> float:
        with self._locked():
            return self._load()["spent_usd"]

    def reserve(self, model: str, request: dict) -> str | None:
        estimate = estimate_usd(model, request)
        if estimate == 0:
            return None
        with self._locked():
            data = self._load()
            held = sum(data["reservations"].values()) + data["unresolved_usd"]
            if round(data["spent_usd"] + held + estimate, 6) > self.cap_usd:
                raise BudgetExceeded(
                    f"LW_BUDGET_USD cap ${self.cap_usd:.2f}: spent ${data['spent_usd']:.6f}, "
                    f"held ${held:.6f}, maximum new charge ${estimate:.6f} ({self.path})"
                )
            token = uuid.uuid4().hex
            data["reservations"][token] = estimate
            self._save(data)
            return token

    def settle(self, reserved: str | None, actual_usd: float | None) -> None:
        if reserved is None:
            return
        if actual_usd is not None and (not math.isfinite(actual_usd) or actual_usd < 0):
            raise ValueError("actual cost must be finite and non-negative, or None for unknown billing")
        with self._locked():
            data = self._load()
            estimate = data["reservations"].pop(reserved)
            if actual_usd is None:
                data["unresolved_usd"] = round(data["unresolved_usd"] + estimate, 6)
            else:
                data["spent_usd"] = round(data["spent_usd"] + actual_usd, 6)
                data["calls"] += int(actual_usd > 0)
            self._save(data)

    def _load(self) -> dict:
        if not self.path.exists():
            return {"spent_usd": 0.0, "unresolved_usd": 0.0, "reservations": {}, "calls": 0, "updated_utc": None}
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            data.setdefault("reservations", {})
            data.setdefault("unresolved_usd", 0.0)
            amounts = [data["spent_usd"], data["unresolved_usd"], *data["reservations"].values()]
            if any(not isinstance(v, (int, float)) or isinstance(v, bool) or not math.isfinite(v) or v < 0 for v in amounts):
                raise ValueError("invalid money amount")
            if not isinstance(data["calls"], int) or data["calls"] < 0:
                raise ValueError("invalid call count")
            return data
        except (ValueError, KeyError, TypeError, AttributeError) as e:
            raise ValueError(f"corrupt spend counter {self.path}: {e}; reconcile it against recorded calls") from None

    def _save(self, data: dict) -> None:
        data["updated_utc"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=self.path.parent, delete=False) as tmp:
            try:
                json.dump(data, tmp, sort_keys=True, allow_nan=False)
                tmp.write("\n")
                tmp.flush()
                os.fsync(tmp.fileno())
                tmp.close()
                os.replace(tmp.name, self.path)
            finally:
                Path(tmp.name).unlink(missing_ok=True)
