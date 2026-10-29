"""D21 dev response cache: recorded responses keyed by (server host, request fingerprint).

Serves repeated dev requests without a new call. Calls served from it carry ``attempts == []`` in ``meta.json`` and
cost nothing; ``lightwork audit`` rejects meta runs that contain such calls. The live orchestrator never passes a
cache while recording a run.
"""

from __future__ import annotations

import json
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path

from lightwork.contracts.calls import CallMeta

_SAFE = re.compile(r"[^A-Za-z0-9._\-]+")


@dataclass(frozen=True)
class Cached:
    raw: bytes
    kind: str  # "sse" | "json"
    meta: CallMeta  # of the call that produced it


class DevCache:
    def __init__(self, root: Path):
        self.root = Path(root)

    @classmethod
    def from_settings(cls, settings) -> DevCache:
        return cls(settings.state_dir / "cache")

    def _dir(self, host: str, fingerprint: str) -> Path:
        if not re.fullmatch(r"[0-9a-f]{64}", fingerprint):
            raise ValueError(f"bad fingerprint {fingerprint!r}")
        return self.root / _SAFE.sub("_", host) / fingerprint

    def get(self, host: str, fingerprint: str) -> Cached | None:
        d = self._dir(host, fingerprint)
        for kind in ("sse", "json"):
            body = d / f"response.{kind}"
            if body.is_file():
                meta = CallMeta.model_validate(json.loads((d / "meta.json").read_text(encoding="utf-8")))
                if meta.fingerprint != fingerprint or meta.base_host != host or meta.status != "ok":
                    raise ValueError(f"cache entry metadata mismatch: {d}")
                return Cached(body.read_bytes(), kind, meta)
        return None

    def put(self, host: str, fingerprint: str, raw: bytes, kind: str, meta: CallMeta) -> None:
        d = self._dir(host, fingerprint)
        if d.exists():
            return
        if kind not in ("sse", "json") or meta.fingerprint != fingerprint or meta.base_host != host:
            raise ValueError("cache response metadata does not match its key")
        d.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=f".{d.name}.", dir=d.parent) as temp:
            tmp = Path(temp)
            (tmp / f"response.{kind}").write_bytes(raw)
            (tmp / "meta.json").write_text(meta.model_dump_json() + "\n", encoding="utf-8")
            try:
                os.rename(tmp, d)
            except OSError:
                if not d.exists():
                    raise
