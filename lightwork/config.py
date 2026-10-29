"""Settings from the environment and an optional ``.env`` (environment wins).

Keys (see ``.env.example``): MODEL_API_KEY, META_API_BASE, LW_BACKEND, LW_CHAT_BASE_URL, LW_CHAT_API_KEY,
LW_CHAT_MODEL, LW_DEV_ALLOW_NON_META, LW_BUDGET_USD. Replay, verify, audit and export never need any of them.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

STATE_DIR_NAME = ".lightwork"


def parse_dotenv(text: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].lstrip()
        key, sep, value = line.partition("=")
        if not sep:
            continue
        key = key.strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        elif " #" in value:
            value = value.split(" #", 1)[0].rstrip()
        out[key] = value
    return out


def _truthy(v: str | None) -> bool:
    return (v or "").strip().lower() in ("1", "true", "yes", "on")


@dataclass(frozen=True)
class Settings:
    model_api_key: str | None = field(default=None, repr=False)
    meta_api_base: str = "https://api.meta.ai/v1"
    backend: str = "meta"  # meta | glimmer
    chat_base_url: str | None = None
    chat_api_key: str | None = field(default=None, repr=False)
    chat_model: str | None = None
    dev_allow_non_meta: bool = False
    budget_usd: float = 120.0
    root: Path = field(default_factory=Path.cwd)

    @property
    def state_dir(self) -> Path:
        return self.root / STATE_DIR_NAME


def load_settings(root: Path | None = None, environ: dict[str, str] | None = None) -> Settings:
    root = Path(root) if root else Path.cwd()
    env: dict[str, str] = {}
    dotenv = root / ".env"
    if dotenv.is_file():
        env.update(parse_dotenv(dotenv.read_text(encoding="utf-8")))
    env.update(os.environ if environ is None else environ)

    def get(k: str) -> str | None:
        v = env.get(k)
        return v if v not in (None, "") else None

    return Settings(
        model_api_key=get("MODEL_API_KEY"),
        meta_api_base=(get("META_API_BASE") or "https://api.meta.ai/v1").rstrip("/"),
        backend=get("LW_BACKEND") or "meta",
        chat_base_url=(get("LW_CHAT_BASE_URL") or "").rstrip("/") or None,
        chat_api_key=get("LW_CHAT_API_KEY"),
        chat_model=get("LW_CHAT_MODEL"),
        dev_allow_non_meta=_truthy(get("LW_DEV_ALLOW_NON_META")),
        budget_usd=float(get("LW_BUDGET_USD") or 120),
        root=root,
    )
