"""Provenance of a model call (PLAN.md D15): ``meta`` for Meta models only, everything else ``dev-standin``.

- Meta backend: the host is Meta's (``*.meta.ai``) and the model id is on the Meta allow-list.
- Chat backend: llama.cpp ``/props`` identifies the loaded Glimmer GGUF. Servers without ``/props`` (Ollama,
  LM Studio) use a narrow allow-list of Glimmer ids; contradictory loaded-weight metadata always overrides an id.
Anything else (including hosted dev-provider aliases) is a dev stand-in. Like model ids on the API, these are
server identity assertions, not cryptographic attestations of the running weights.
"""

from __future__ import annotations

import re
from typing import Literal

from lightwork.contracts.calls import ServerProps

Provenance = Literal["meta", "dev-standin"]

META_MODEL_IDS = frozenset(
    {
        "muse-wonder-1.0",
        "muse-spark-1.3",
        "muse-spark-1.2",
        "muse-spark-1.1",
        "muse-spark-1.3-contributor",
        "muse-spark-1.2-contributor",
        "sam-3.1",
        "muse-voice-transcribe-1.0",
        "muse-image-1.0",
    }
)
CHAT_META_IDS = frozenset({"muse-glimmer", "muse-glimmer:30b", "muse-glimmer:30b-mlx",
                           "meta/muse-glimmer-30b", "meta-llama/muse-glimmer-30b"})
_GLIMMER_GGUF = re.compile(r"(?i)glimmer[^/\\]*\.gguf$")  # basename of the loaded model file


def is_meta_host(host: str) -> bool:
    name = host.rsplit("@", 1)[-1].split(":", 1)[0].lower()
    return name == "meta.ai" or name.endswith(".meta.ai")


def is_glimmer_server(props: ServerProps | None) -> bool:
    return props is not None and bool(props.model_path) and bool(_GLIMMER_GGUF.search(props.model_path))


def classify(backend: str, host: str, model: str, props: ServerProps | None) -> Provenance:
    if backend == "meta":
        return "meta" if is_meta_host(host) and model in META_MODEL_IDS else "dev-standin"
    if backend == "chat":
        if props is not None and props.model_path:
            return "meta" if is_glimmer_server(props) else "dev-standin"
        return "meta" if model.lower() in CHAT_META_IDS else "dev-standin"
    raise ValueError(f"unknown backend {backend!r}")
