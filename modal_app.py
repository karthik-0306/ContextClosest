"""
modal_app.py

Serverless deployment of the Hybrid Multimodal Search Engine on Modal
(https://modal.com). Runs the full PyTorch path (fine-tuned SigLIP 2 + LoRA +
SPLADE) — no model shrinking or re-indexing needed.

Deploy:
    pip install modal
    modal token new                       # browser login, one time
    modal secret create qdrant QDRANT_URL="..." QDRANT_API_KEY="..."
    modal deploy modal_app.py

The command prints a public URL like
    https://<user>--contextclosest-search-web.modal.run
Open it in a browser, or hit /api/health.

The deployed app searches the same Qdrant Cloud index that the Spark + Kafka
pipeline writes to, so products streamed in from your machine appear on the
public site as soon as the consumer indexes them.

Optional Redis cache: create a free hosted Redis (e.g. Upstash), then
    modal secret create redis REDIS_URL="rediss://default:<password>@<host>:6379"
and set ENABLE_REDIS = True below.  Without it the app runs uncached.
Kafka telemetry is turned off on Modal (KAFKA_BOOTSTRAP is empty) because the
broker runs on your machine.
"""

import modal

# Uses the Modal secret "redis" (REDIS_URL of a hosted Redis, see above).
# Set to False to run the deployment without a cache.
ENABLE_REDIS = True

APP_DIR = "/root/app"
BASE_MODEL = "google/siglip2-base-patch16-224"
SPLADE_MODEL = "prithivida/Splade_PP_en_v1"

# Caches baked into the image so cold starts don't re-download ~1.5 GB of weights.
CACHE_ENV = {
    "HF_HOME": "/opt/hf-cache",
    "FASTEMBED_CACHE_PATH": "/opt/fastembed-cache",
    "HF_HUB_DISABLE_TELEMETRY": "1",
}


def _prefetch_models() -> None:
    """Runs at image build time to bake model weights into the image layer."""
    from transformers import AutoModel, AutoProcessor
    from fastembed import SparseTextEmbedding

    AutoProcessor.from_pretrained(BASE_MODEL)
    AutoModel.from_pretrained(BASE_MODEL)
    SparseTextEmbedding(model_name=SPLADE_MODEL)


image = (
    modal.Image.debian_slim(python_version="3.11")
    # CPU-only torch (avoids pulling ~2 GB of CUDA wheels)
    .pip_install("torch", index_url="https://download.pytorch.org/whl/cpu")
    .pip_install_from_requirements("requirements-modal.txt")
    .env({
        **CACHE_ENV,
        "PYTHONPATH": APP_DIR,
        "LORA_ADAPTER_DIR": f"{APP_DIR}/deploy/lora_adapter",
        "TORCH_NUM_THREADS": "2",   # matches cpu=2.0 below
        "KAFKA_BOOTSTRAP": "",      # no broker on Modal: search telemetry off
    })
    .run_function(_prefetch_models)
    # Only ship what the server needs — never Raw_Data/ or Processed_Data/.
    .add_local_dir("static", remote_path=f"{APP_DIR}/static")
    .add_local_dir("deploy", remote_path=f"{APP_DIR}/deploy")
    .add_local_dir("search", remote_path=f"{APP_DIR}/search")
    .add_local_dir("indexing", remote_path=f"{APP_DIR}/indexing")
    .add_local_file("app.py", remote_path=f"{APP_DIR}/app.py")
    .add_local_file("config.py", remote_path=f"{APP_DIR}/config.py")
)

app = modal.App("contextclosest-search")

secrets = [modal.Secret.from_name("qdrant")]  # provides QDRANT_URL, QDRANT_API_KEY
if ENABLE_REDIS:
    secrets.append(modal.Secret.from_name("redis"))  # provides REDIS_URL


@app.function(
    image=image,
    secrets=secrets,
    cpu=2.0,
    memory=4096,
    scaledown_window=300,   # stay warm 5 min after the last request, then scale to zero
    min_containers=0,       # $0 while idle
    timeout=300,            # generous for the first (cold) request while models load
)
@modal.concurrent(max_inputs=8)
@modal.asgi_app()
def web():
    from app import app as fastapi_app
    return fastapi_app
