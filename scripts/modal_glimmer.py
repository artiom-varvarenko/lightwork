"""Muse Glimmer (Meta open weights, 30B) on a Modal GPU, behind vLLM's OpenAI-compatible API, for Lightwork's
open-weight brain (``MODAL_LLM_URL``; see ``lightwork partners``).

Follows Meta's vLLM recipe for Glimmer (https://dev.meta.ai/docs/muse-glimmer/deploy#vllm): the
``vllm/vllm-openai:muse-glimmer`` image, the bf16 checkpoint ``meta-models/Muse-Glimmer-30B`` (about 60 GB, kept in a
Modal Volume after the first start), one 80 GB GPU, the ``muse_glimmer`` reasoning and tool parsers.

    pip install modal && modal setup                       # once: log in (opens a browser)
    modal secret create lightwork-glimmer LW_MODAL_KEY=$(openssl rand -hex 16)
    modal deploy scripts/modal_glimmer.py                  # prints the URL
    # .env: MODAL_LLM_URL=https://<workspace>--lightwork-glimmer-serve.modal.run/v1  MODAL_LLM_KEY=<the same key>
    uv run lightwork partners --only modal                 # first call waits for the weights (several minutes)

Costs Modal GPU time while a container is up (scaledown after 10 idle minutes). Stop it with
``modal app stop lightwork-glimmer``. Not imported by Lightwork itself; Modal is not a dependency.
"""

import subprocess

import modal

MODEL = "meta-models/Muse-Glimmer-30B"
PORT = 8000
GPU = "H100"  # 80 GB: the recipe needs about 72 GB of VRAM for the bf16 checkpoint

image = modal.Image.from_registry("vllm/vllm-openai:muse-glimmer", add_python="3.12").entrypoint([])
hf_cache = modal.Volume.from_name("lightwork-hf-cache", create_if_missing=True)
app = modal.App("lightwork-glimmer")


@app.function(image=image, gpu=GPU, volumes={"/root/.cache/huggingface": hf_cache},
              secrets=[modal.Secret.from_name("lightwork-glimmer")], timeout=60 * 60, scaledown_window=10 * 60)
@modal.concurrent(max_inputs=8)
@modal.web_server(port=PORT, startup_timeout=30 * 60)
def serve():
    import os

    cmd = ["vllm", "serve", MODEL, "--served-model-name", "muse-glimmer", "--host", "0.0.0.0", "--port", str(PORT),
           "--tensor-parallel-size", "1", "--enable-auto-tool-choice", "--tool-call-parser", "muse_glimmer",
           "--reasoning-parser", "muse_glimmer", "--generation-config", "auto",
           "--api-key", os.environ["LW_MODAL_KEY"]]
    subprocess.Popen(cmd)
