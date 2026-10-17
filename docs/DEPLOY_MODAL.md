# Deploying to Modal

Modal runs the container only while requests are being served and **scales to
zero** when idle, so the free monthly credit (Starter plan) covers a portfolio
demo. The full PyTorch path runs unchanged: fine-tuned SigLIP 2 + LoRA + SPLADE.

## How the deployment fits the pipeline

```
your machine                                   Modal
────────────                                   ─────
Spark ETL → Kafka → kafka_consumer.py ──┐      FastAPI app (modal_app.py)
                                         ├──►   Qdrant Cloud  ◄── searches
kafka_producer.py --upload-images ──────┼──►   S3 bucket     ◄── browser loads images
                                         └──►   Redis (optional, hosted) ◄── cache
```

- The Modal app and the streaming indexer use the **same Qdrant Cloud
  collection**, so products you add with the Spark → Kafka pipeline appear on
  the public site as soon as the consumer indexes them. No redeploy is needed.
- Images come from the public S3 bucket; `kafka_producer.py --upload-images`
  uploads images of new products.
- Kafka telemetry is off on Modal (`KAFKA_BOOTSTRAP` is empty in the image)
  because the broker runs on your machine.

## Prerequisites

- Qdrant Cloud collection `abo_products` populated (`scripts/cloud_reindex.py`)
- Product images in a **public** S3 bucket (`scripts/upload_to_s3.py`)
- `QDRANT_URL` and `QDRANT_API_KEY` (from your `.env`)

## What is in the repo for this

| File | Purpose |
|------|---------|
| `modal_app.py` | Modal image, secrets, and the FastAPI `app` served as a web endpoint |
| `requirements-modal.txt` | Runtime dependencies (torch is installed separately, CPU build) |
| `deploy/lora_adapter/` | 4.6 MB LoRA adapter shipped in the image |

The image contains only `app.py`, `config.py`, `search/`, `indexing/`, `static/`
and `deploy/` — never `Raw_Data/` or `Processed_Data/`. Model weights are baked
into the image at build time, so cold starts do not download them.

## Steps

### 1. Sign up
https://modal.com/signup — continue with GitHub or Google. No credit card needed.

### 2. Install the CLI and log in
```bash
pip install modal
modal token new
```

### 3. Create the Qdrant secret
```bash
modal secret create qdrant QDRANT_URL="https://<your-cluster>.cloud.qdrant.io" QDRANT_API_KEY="<your-key>"
```
(or Modal dashboard → Secrets → New secret, named exactly `qdrant`.)

### 4. Deploy
```bash
modal deploy modal_app.py
```
The first build installs torch and the dependencies and bakes in the model
weights (a few minutes). It prints a URL like:
```
https://<username>--contextclosest-search-web.modal.run
```

### 5. Verify
- `…/api/health` → `{"status":"ok", ...}` (`redis_available` is false unless you enable Redis below)
- Open the URL and run a search. The first request after idle is a 20–40 s cold start.

## Optional: Redis cache on Modal

1. Create a free hosted Redis, for example on [Upstash](https://upstash.com), and copy its TLS URL.
2. `modal secret create redis REDIS_URL="rediss://default:<password>@<host>:6379"`
3. Set `ENABLE_REDIS = True` at the top of `modal_app.py` and run `modal deploy modal_app.py`.
4. Put the same `REDIS_URL` in your local `.env`. The streaming indexer then
   clears the deployed app's cached results whenever it adds products, and the
   trending bar works on the public site.

## Redeploying after code changes

```bash
git push origin main        # keep GitHub in sync
modal deploy modal_app.py
```

## Notes

- **Cost:** you pay only for seconds served. `scaledown_window=300` keeps the
  container warm 5 minutes after the last request; lower it to save credit,
  raise it to reduce cold starts.
- **CPU threads:** the image sets `TORCH_NUM_THREADS=2` to match `cpu=2.0`.
- **Logs:** `modal app logs contextclosest-search`.
- **Stop it:** `modal app stop contextclosest-search`.
- If the credit runs out without billing set up, Modal stops the app instead of charging.
