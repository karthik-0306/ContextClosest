# ContextClosest — Hybrid Multimodal Search Engine for Large-Scale Product Catalogs

**Complete Project Documentation**

| | |
|---|---|
| **Project name** | ContextClosest |
| **One-line description** | A hybrid (dense + sparse) multimodal product search engine over the Amazon Berkeley Objects catalog, built on a LoRA-fine-tuned SigLIP 2 model and SPLADE, fused with Reciprocal Rank Fusion in Qdrant. |
| **Repository** | https://github.com/karthik-0306/ContextClosest |
| **Live demo** | https://karthik-0306--contextclosest-search-web.modal.run/ |
| **Timeline** | 10 Aug 2026 – 02 Sep 2026 |
| **Dataset** | Amazon Berkeley Objects (ABO), listings + `images-small` variant |
| **Core models** | `google/siglip2-base-patch16-224` (+ LoRA), `prithivida/Splade_PP_en_v1` |
| **Serving** | FastAPI + Uvicorn, deployed serverless on Modal |
| **Storage** | Qdrant Cloud (vectors + payload), AWS S3 (product images) |

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Problem Statement & Motivation](#2-problem-statement--motivation)
3. [System Architecture](#3-system-architecture)
4. [Phase 1 — Data: The ABO Dataset & ETL Pipeline](#4-phase-1--data-the-abo-dataset--etl-pipeline)
5. [Phase 2 — Domain Adaptation: Fine-Tuning SigLIP 2 with LoRA](#5-phase-2--domain-adaptation-fine-tuning-siglip-2-with-lora)
6. [Phase 3 — Hybrid Vector Indexing (Qdrant)](#6-phase-3--hybrid-vector-indexing-qdrant)
7. [Phase 4 — Search API & Web Interface](#7-phase-4--search-api--web-interface)
8. [Query-Time Retrieval Flow](#8-query-time-retrieval-flow)
9. [Deployment & Infrastructure](#9-deployment--infrastructure)
10. [The Deployment Migration Story (Render → HF → Modal)](#10-the-deployment-migration-story-render--hf--modal)
11. [File-by-File Reference](#11-file-by-file-reference)
12. [Complete Table of Numerical Values](#12-complete-table-of-numerical-values)
13. [Technology Stack & Versions](#13-technology-stack--versions)
14. [How to Run / Reproduce](#14-how-to-run--reproduce)
15. [Known Limitations](#15-known-limitations)
16. [Glossary](#16-glossary)

---

## 1. Executive Summary

**ContextClosest** is an end-to-end information-retrieval system that lets a user search a
56,427-product e-commerce catalog with natural-language queries such as *"blue red shoes"*
or *"wooden dining table"*. It combines two complementary retrieval signals:

- **Dense retrieval** — a **SigLIP 2** vision-language model, **fine-tuned with LoRA** on the
  project's own product data, embeds every product *image* into a 768-dimensional vector.
  At query time the same model embeds the *text* query into the same 768-d space, and
  nearest-neighbour search (cosine) returns visually/semantically similar products.
- **Sparse retrieval** — **SPLADE** (`Splade_PP_en_v1`) produces a learned sparse
  bag-of-terms vector for each product's concatenated text fields, and for the query.
  This recovers exact keyword / brand / attribute matches that dense vectors blur.

The two ranked lists are merged with **Reciprocal Rank Fusion (RRF)**, executed natively
inside the **Qdrant** vector database. A **FastAPI** backend serves the search endpoint
and a dark-mode single-page frontend. The whole service is deployed **serverless on Modal**,
reading vectors from **Qdrant Cloud** and product images from a public **AWS S3** bucket.

The project is organised into four phases: **(1)** data cleaning/ETL, **(2)** LoRA
fine-tuning of SigLIP 2, **(3)** hybrid vector indexing, **(4)** the search API + UI, plus
a cloud-deployment layer.

---

## 2. Problem Statement & Motivation

Large e-commerce catalogs are massive repositories of multimodal data (titles, attributes,
images). The search bar is the primary interface, but bridging a user's natural-language
intent to the catalog is hard:

- **Lexical search (TF-IDF / BM25)** is efficient and great at exact matches (a brand like
  *"Adidas"*, a model code) but suffers the **vocabulary-mismatch problem**: a query for
  *"comfortable running shoes"* misses a product titled *"athletic breathable sneakers"*
  because there is no word overlap.
- **Dense retrieval** (BERT-style encoders, or multimodal CLIP/SigLIP) maps text and images
  into a shared continuous space where semantic similarity is measurable via cosine
  similarity. But dense models **fail at precise lexical constraints** — a search for a
  *"blue Adidas shoe"* can return a *"blue Nike shoe"* because overall semantic similarity
  overwhelms the brand token.
- **Sparse neural retrieval (SPLADE)** uses a transformer to learn *sparse* representations
  that keep exact-term signal **and** perform learned term expansion, keeping the efficiency
  of inverted indices while injecting semantics.

**The hybrid approach** runs the query through both a dense multimodal encoder and a sparse
neural encoder, producing two candidate lists, then fuses them with **RRF** — which needs no
weight tuning and boosts items that rank well in *both* the semantic and lexical domains.

**Why fine-tune?** The base SigLIP 2 model is trained on web image-text pairs. Product
catalogs have their own visual and lexical distribution (attribute-style phrasing, brand
names, catalog photography). **LoRA** (Low-Rank Adaptation) adapts the foundation model to
this domain **without catastrophic forgetting and at a fraction of the compute** of full
fine-tuning — only ~1.2 M parameters are trained versus ~375 M in the full model.

---

## 3. System Architecture

```
                              ┌──────────────────────────────────────────────┐
   OFFLINE (build time)       │                  ONLINE (serving)            │
                              │                                              │
 Raw ABO listings (147,702)   │            User text query                   │
 + image metadata             │                  │                           │
        │                     │                  ▼                           │
        ▼                     │      ┌───────────────────────┐               │
 data_pipeline/clean.py       │      │  HybridSearchEngine    │              │
  · filter · dedup · join     │      │  (search/engine.py)    │              │
        │                     │      │                        │              │
        ▼                     │      │  ┌──────────────────┐  │              │
 products.parquet (56,427)    │      │  │ InferenceEngine  │  │              │
        │                     │      │  │ SigLIP 2 text →  │  │  768-d dense │
        ├───────────────┐     │      │  │   768-d vector   │  │              │
        ▼               ▼     │      │  │ SPLADE text →    │  │  sparse vec  │
 training/train.py   package_ │      │  │   sparse vector  │  │              │
 (Kaggle T4 GPU)     for_     │      │  └────────┬─────────┘  │              │
  LoRA adapter       kaggle   │      │           ▼            │              │
  (~4.6 MB)          .zip     │      │   Qdrant query_points  │              │
        │                     │      │   Prefetch(dense) +    │              │
        ▼                     │      │   Prefetch(sparse) →   │              │
 indexing/ (Kaggle):          │      │   FusionQuery(RRF)     │              │
  · SigLIP2 image embeds      │      └───────────┬───────────┘               │
    → image_embeddings.npy    │                  │                           │
  · SPLADE text embeds        │                  ▼                           │
    → sparse_text_embeddings  │        Qdrant Cloud  ── collection            │
        │                     │        "abo_products" (56,427 points)         │
        ▼                     │                  │                           │
 scripts/cloud_reindex.py ────┼──────────────────┘                           │
  upsert vectors + payload    │        payload.image_url → AWS S3            │
 scripts/upload_to_s3.py ─────┼──────────────────────────► (eu-north-1)     │
  upload images + parquet     │                                              │
                              │        FastAPI (app.py) + static/ frontend   │
                              │        deployed on Modal (modal_app.py)      │
                              └──────────────────────────────────────────────┘
```

**Two vectors per product** are stored in one Qdrant point:

| Vector name | Type | Size | Distance | Source |
|---|---|---|---|---|
| `dense_image` | dense | 768 | Cosine | Fine-tuned SigLIP 2 **image** tower |
| `sparse_text` | sparse | (vocab-sized, ~30 k) | dot product | SPLADE over `title + brand + color + material + category` |

At query time the **text** query is embedded by the SigLIP 2 **text** tower (shared space
with the image tower) and by SPLADE, then both are searched and fused.

---

## 4. Phase 1 — Data: The ABO Dataset & ETL Pipeline

### 4.1 The dataset

**Amazon Berkeley Objects (ABO)** is a public Amazon product dataset.

| Metric | Value |
|---|---|
| Raw product listings (total) | **147,702** |
| Full image catalog (all listings) | **398,212** image files |
| Image variant used | `abo-images-small` (max side ≈ 256 px) |
| Listing format | gzipped newline-delimited JSON (`listings/metadata/*.json.gz`) |
| Image metadata | `images/metadata/images.csv.gz` (maps `image_id` → relative path) |
| Languages present | multilingual; project keeps English only |

Each raw listing is a nested JSON record with multilingual fields
(`item_name`, `brand`, `color`, `material`, `product_type`, `node` category tree,
`main_image_id`, `country`, …).

### 4.2 The cleaning / ETL pipeline (`data_pipeline/clean.py`)

The loader (`data_pipeline/loader.py`) **streams** the gzipped JSON line-by-line (Python
generators) so the raw data never has to fit in RAM. Each record passes through this filter
chain, and rejects are counted:

| Filter / step | Rule | Records dropped |
|---|---|---|
| **Swatch removal** | any `item_name` value contains the word "swatch" (case-insensitive) | **439** |
| **Product-type filter** | `product_type` missing, **or** in the drop-list `("cellular_phone_case",)` | **64,853** |
| **English-title filter** | no English `item_name` variant found | **24,829** |
| **Image-presence filter** | `main_image_id` missing or not in the image-metadata lookup | **432** |
| **Cross-marketplace dedup** | same `item_id` (ASIN) seen from a lower-priority marketplace | **722** |
| **KEPT (final catalog)** | | **56,427** |

Sanity check: 439 + 64,853 + 24,829 + 432 + 722 = 91,275; and 147,702 − 91,275 = **56,427**. ✓

**English value selection** — for any multilingual field, the pipeline picks the first
available value following the language priority:

```
en_US  >  en_CA  >  en_GB  >  en_AU  >  en_IN   (then any other en_* tag)
```

For `color`, "standardized" values are preferred over raw strings. `material` falls back to
`fabric_type` when absent.

**Cross-marketplace deduplication** — the same product (ASIN) can appear in multiple Amazon
marketplaces. The record from the highest-priority country is kept:

```
US = 1  >  CA = 2  >  GB = 3  >  AU = 4  >  IN = 5   (any other country = 99)
```

**Category** is reduced to the **leaf** node name of the ABO category tree
(e.g. `.../Shoes/Sneakers` → `Sneakers`).

### 4.3 Outputs

| Output | What it is | Approx. size |
|---|---|---|
| `Processed_Data/products.parquet` | The clean catalog — **56,427 rows × 9 columns** | ~4.7 MB |
| `Processed_Data/images/` | Only the images actually referenced — **43,631 unique files** (products share images across size/colour variants) | — |
| `Processed_Data/images_meta.parquet` | Filtered image metadata (only kept `image_id`s) | — |
| `Processed_Data/stats/dataset_stats.json` | The filter counts above | <1 KB |

**Product schema (9 fields, `data_pipeline/schema.py`):**

`item_id` (ASIN), `product_type` (lowercased), `title` (English), `color` (standardized),
`material` (or fabric fallback), `brand`, `category` (leaf), `main_image_id`,
`main_image_path` (relative path like `3a/3a4e88ef.jpg`).

### 4.4 Packaging for the GPU (`data_pipeline/package_for_kaggle.py`)

Bundles `products.parquet` + the **43,631 referenced images** into `kaggle_dataset.zip`
(ZIP_DEFLATED) so the Kaggle fine-tuning notebook has everything with no extra setup.
Only referenced images are included — the full 398,212-file catalog is far larger than
needed.

---

## 5. Phase 2 — Domain Adaptation: Fine-Tuning SigLIP 2 with LoRA

Run on a **Kaggle T4 GPU (16 GB VRAM)**. Local execution (`training/train.py`) is supported
only for code validation / smoke testing — on CPU it is intentionally too slow for real
training.

### 5.1 Base model

| | |
|---|---|
| Checkpoint | `google/siglip2-base-patch16-224` |
| Family | SigLIP 2 (Sigmoid Loss for Language-Image Pre-training, v2) |
| Image resolution | **224 × 224**, patch size **16** |
| Embedding dimension | **768** |
| Text tokenizer | Gemma multilingual tokenizer, vocab **256,000** |
| Approx. total parameters | ~375 M |

**Why SigLIP over CLIP?** SigLIP replaces CLIP's softmax contrastive loss with a **pairwise
sigmoid loss**: every (image, caption) pair is scored independently as
positive/negative rather than normalised across the batch. This is more stable at **small
batch sizes** (CLIP-style training typically needs very large batches).

### 5.2 LoRA adapter configuration (`training/lora_config.py`)

| Hyperparameter | Value | Note |
|---|---|---|
| `r` (rank) | **16** | trainable params per adapted layer |
| `lora_alpha` | **32** | effective LoRA LR scales as `alpha / r = 2.0` |
| `lora_dropout` | **0.05** | regularisation on LoRA path |
| `target_modules` | `["q_proj", "v_proj"]` | query & value projections only |
| Applied to | **both** vision tower and text tower attention blocks | `k_proj`, `out_proj` left frozen |
| `bias` | `"none"` | base biases stay frozen |
| Trainable parameters | **≈ 1.18 M** (`16 × (768 + 768)` per module × 2 modules/layer × ~24 layers) | ≈ 0.3 % of the model |
| Saved adapter size | **≈ 4.6 MB** (`adapter_model.safetensors`, 4,732,520 bytes) | |
| PEFT version | 0.19.1 | |

### 5.3 Training configuration (`training/lora_config.py::TrainingConfig`)

| Hyperparameter | Value |
|---|---|
| Epochs | **3** |
| Batch size | **32** image-text pairs |
| Gradient accumulation steps | **1** (effective batch = 32) |
| Optimizer | **AdamW** |
| Peak learning rate | **2 × 10⁻⁴** |
| LR schedule | **cosine** decay with warmup |
| Warmup ratio | **0.1** (10 % of total steps) |
| Weight decay | **0.01** |
| Max gradient norm (clip) | **1.0** |
| Mixed precision | **fp16** (`torch.cuda.amp`) |
| Gradient checkpointing | **enabled** (~20 % throughput cost, lower VRAM) |
| Logging interval | every **50** optimizer steps |
| Validation split | **10 %** (`val_split_ratio = 0.1`) → ≈ **50,784 train / 5,643 val** |
| Random seed | **42** (split + weight init) |
| Checkpoint / eval strategy | every **epoch** |
| Output dir | `training/checkpoints/` |

### 5.4 Training data construction

- **No raw titles are used for training text.** Raw Amazon titles contain marketing noise,
  size codes and language artifacts. Instead, `data_pipeline/schema.py::build_caption()`
  synthesises a **query-style caption** from the clean attributes, e.g.:

  ```
  {"color":"brown","material":"suede","product_type":"shoes",
   "brand":"The Fix","category":"Loafer"}
   →  "a brown suede shoes in the loafer category by The Fix"
  ```

  This narrows the train/inference distribution gap (users type attribute-style queries).

- **Two caption variants per product**: one **with** the brand (`… by <brand>`), one
  **without**. The dataset alternates variant by epoch (`epoch % 2`) so the model sees both
  phrasings and does not overfit a single sentence structure. Effective unique text inputs
  seen ≈ 2× the product count. This also acts as **brand-dropout** regularisation.

- Corrupt/missing images are silently replaced with a 224×224 grey placeholder so one bad
  file cannot crash a multi-hour run.

### 5.5 Objective & checkpointing

- **Loss**: SigLIP sigmoid loss (`model(**batch, return_loss=True).loss`) — maximise the
  similarity of the true (image, caption) pair, minimise all other in-batch cross-pairs.
- **Evaluation**: average sigmoid loss on the val split, computed every epoch with a fixed
  caption variant (with-brand) so the metric is comparable across epochs.
- **Checkpoints**: per-epoch adapter saves, plus a separately tracked
  `best_checkpoint/` (lowest val loss) and a final `lora_adapter_final/`.
- The chosen adapter ships in the repo at **`deploy/lora_adapter/`**
  (`adapter_config.json` + `adapter_model.safetensors`), and locally at
  `Processed_Data/models/lora_adapter/`.

---

## 6. Phase 3 — Hybrid Vector Indexing (Qdrant)

### 6.1 Dense image embeddings

- Generated on Kaggle GPU by running every product's **main image** through the
  **merged fine-tuned SigLIP 2 image tower** (`get_image_features`), then **L2-normalised**.
- Stored as `Processed_Data/image_embeddings.npy` —
  **56,427 × 768 float32 = 173,343,872 bytes (≈ 165 MB)**.

### 6.2 Sparse text embeddings

- Model: **`prithivida/Splade_PP_en_v1`** (SPLADE++ English v1), run via **`fastembed`**
  (`SparseTextEmbedding`), which executes an ONNX build under the hood.
- Input document per product (`indexing/build_index.py::build_search_text`):

  ```
  title + " " + brand + " " + color + " " + material + " " + category
  ```

- Output is a list of `{indices: [...], values: [...]}` sparse vectors, stored as
  `Processed_Data/sparse_text_embeddings.json` (**≈ 92 MB**).

### 6.3 The Qdrant collection

| Property | Value |
|---|---|
| Collection name | **`abo_products`** |
| Dense vector `dense_image` | size **768**, distance **Cosine** |
| Sparse vector `sparse_text` | `SparseVectorParams()` (dot-product scoring) |
| Points | **56,427** |
| Payload | the entire cleaned product row (all 9 fields; `image_url` added for cloud) |
| Local build point ID | `uuid5(NAMESPACE_OID, item_id)` — deterministic per ASIN |
| Cloud build point ID | integer row index `i` (`scripts/cloud_reindex.py`) |
| Local upload batch size | **256** (`indexing/build_index.py`) |
| Cloud upload batch size | **128** (`scripts/cloud_reindex.py`, hardened with retry + 120 s timeout) |

`indexing/build_index.py` can run fully local: it uses pre-computed Kaggle `.npy` / `.json`
if present, otherwise regenerates embeddings on CPU (slow). It always recreates the
collection from scratch.

### 6.4 Moving to the cloud

- **`scripts/cloud_reindex.py`** reads `products.parquet` + `image_embeddings.npy` +
  `sparse_text_embeddings.json`, and upserts all 56,427 points into **Qdrant Cloud**.
  It rewrites each payload: removes the local `main_image_path` and adds a public
  `image_url`:

  ```
  https://<AWS_BUCKET_NAME>.s3.<AWS_REGION>.amazonaws.com/images/<image_filename>
  ```

- **`scripts/upload_to_s3.py`** uploads `products.parquet` + every `*.jpg` under
  `Processed_Data/images/` to the S3 bucket under `images/<filename>`, using a
  **20-thread** pool and skipping objects that already exist.

- **`scripts/fix_image_urls.py`** (one-off repair) — a region mismatch (`us-east-1` in
  `.env` vs the bucket's real `eu-north-1`) had been baked into every `image_url`, causing
  S3 `301` redirects that browsers don't follow for `<img>`. This script scrolls the
  collection and `set_payload`s a corrected URL only where wrong (**3,341** payloads fixed
  in the incident, idempotent, safe to re-run).

---

## 7. Phase 4 — Search API & Web Interface

### 7.1 Backend (`app.py`, `search/engine.py`)

**FastAPI** app with a **lifespan** context manager that constructs one
`HybridSearchEngine` at startup (models loaded once, never per request).

| Route | Method | Purpose |
|---|---|---|
| `/` | GET | serves `static/index.html` |
| `/api/search` | GET | `q` (required, min length 1), `top_k` (default **12**, range **1–50**) → hybrid search |
| `/api/health` | GET | `{"status": "ok" | "loading", "collection": "abo_products"}` |
| `/static/*` | GET | CSS / JS |
| `/images/*` | GET | local image mount — **only** when `Processed_Data/images/` exists (dev); in the cloud images come from S3 URLs in the payload |

**Response shape** of `/api/search`:

```json
{
  "results": [
    {"score": 0.5, "item_id": "...", "title": "...", "brand": "...",
     "color": "...", "material": "...", "product_type": "...",
     "category": "...", "image_url": "https://...s3.eu-north-1..."}
  ],
  "count": 12,
  "elapsed_ms": 291.5
}
```

### 7.2 `HybridSearchEngine.search()` — the core

1. `query.strip()`; empty → empty result.
2. **Dense**: `InferenceEngine.embed_text_dense([query])` → SigLIP 2 text tower →
   L2-normalised **768-d** vector.
3. **Sparse**: `InferenceEngine.embed_text_sparse([query])` → SPLADE → `{indices, values}`
   → `SparseVector`.
4. **Qdrant `query_points`** with two prefetches and RRF fusion:

   ```python
   prefetch=[
       Prefetch(query=dense_vector,  using="dense_image", limit=top_k*3),
       Prefetch(query=sparse_vector, using="sparse_text", limit=top_k*3),
   ],
   query=FusionQuery(fusion=Fusion.RRF),
   limit=top_k,
   with_payload=True,
   ```

   Each sub-retriever returns `top_k × 3` candidates (e.g. **36** for the default 12);
   RRF re-ranks the union; the top `top_k` are returned.
5. Format each hit: round score to 4 dp, pull payload fields, choose `image_url` (S3) if
   present else `/images/<main_image_path>`.

### 7.3 `InferenceEngine` (`indexing/inference.py`)

- **Dense**: `AutoModel.from_pretrained("google/siglip2-base-patch16-224")` wrapped with the
  LoRA adapter via `PeftModel.from_pretrained`; runs on CUDA if available else CPU;
  `eval()` mode. Text is processed with `padding="max_length", truncation=True`;
  `get_text_features` → normalise → list.
- **Sparse**: `fastembed.SparseTextEmbedding("prithivida/Splade_PP_en_v1")`.
- `embed_images_dense()` (offline indexer only) mirrors the text path with
  `get_image_features`.
- The adapter directory is resolved by `config.py` in this order:
  `LORA_ADAPTER_DIR` env var → `Processed_Data/models/lora_adapter/` → `deploy/lora_adapter/`.

### 7.4 Frontend (`static/`)

| File | Role |
|---|---|
| `index.html` | Single page: hero, stat chips (56,427 / 768 / RRF / SigLIP 2), search bar, results grid, footer |
| `styles.css` | 608-line dark-mode design system — bg `#0d1117`/`#161b22`/`#1c2333`, indigo→violet accent gradient `#6366f1 → #8b5cf6`, Inter font, glassmorphism, skeleton loaders, animated cards |
| `app.js` | Vanilla JS: fires search on button click / Enter (no search-as-you-type — CPU inference is too slow), shows 12 skeleton cards, `fetch('/api/search?q=…&top_k=12')`, renders product cards with a relevance bar (`score × 100`), image `onerror` → 📦 emoji fallback, HTML-escapes all API strings (XSS guard), intro / empty / error states |

---

## 8. Query-Time Retrieval Flow

```
"blue red shoes"
      │
      ├──────────────► SigLIP 2 text tower  ──►  768-d dense vector (L2-normed)
      │                                              │
      └──────────────► SPLADE (fastembed)   ──►  sparse {indices, values}
                                                     │
                                                     ▼
                       Qdrant  collection "abo_products"  (56,427 points, Qdrant Cloud)
                       ┌─────────────────────────┬─────────────────────────┐
                       │ Prefetch dense_image    │ Prefetch sparse_text    │
                       │ cosine, limit 36        │ dot-product, limit 36   │
                       └───────────┬─────────────┴───────────┬─────────────┘
                                   └────────►  RRF fusion  ◄─┘
                                              (Fusion.RRF)
                                                   │
                                                   ▼
                                        top 12 fused results
                                                   │
                                                   ▼
                    JSON {results, count, elapsed_ms}  →  frontend cards
                    image_url → https://processed-abo-dataset.s3.eu-north-1.amazonaws.com/images/<f>.jpg
```

**Reciprocal Rank Fusion**: for a document `d` appearing at rank `r_i` in result list `i`,
its fused score is `Σ_i 1 / (k + r_i)` (Qdrant's default `k`). RRF needs **no per-signal
weight tuning** and rewards documents that rank well in *both* lists. In practice the
observed `score` values (0.5, 0.333, 0.25, …) are the reciprocal-rank contributions.

**Measured end-to-end latency** (live Modal deployment, small sample):
**≈ 230–740 ms** per query, ~290 ms typical.

---

## 9. Deployment & Infrastructure

### 9.1 Current production topology

```
             ┌────────────────────────────┐
   Browser ──►  Modal serverless container │  https://karthik-0306--contextclosest-search-web.modal.run
             │  (FastAPI + Uvicorn,        │
             │   SigLIP 2 + SPLADE in RAM) │
             └───────┬────────────┬────────┘
                     │            │
         vectors +   │            │  image bytes (in <img> from browser)
         payload     ▼            ▼
             Qdrant Cloud     AWS S3  bucket: processed-abo-dataset
             collection       region: eu-north-1
             "abo_products"   images/*  (public-read bucket policy)
             56,427 points    ~43,631 jpgs + products.parquet
```

### 9.2 Modal configuration (`modal_app.py`)

| Setting | Value |
|---|---|
| Modal app name | `contextclosest-search` |
| Base image | `modal.Image.debian_slim(python_version="3.11")` |
| torch | CPU wheel (`--index-url https://download.pytorch.org/whl/cpu`) |
| Other deps | `requirements-modal.txt` |
| Model weights | **baked into the image** at build via `_prefetch_models()` (SigLIP 2 + SPLADE) so cold starts don't re-download ~1.5 GB |
| `cpu` | **2.0** cores |
| `memory` | **4096** MB |
| `scaledown_window` | **300 s** (stay warm 5 min after last request, then scale to zero) |
| `min_containers` | **0** ($0 while idle) |
| `timeout` | **300 s** (generous for the cold first request) |
| Concurrency | `@modal.concurrent(max_inputs=8)` |
| Secret | `qdrant` → injects `QDRANT_URL`, `QDRANT_API_KEY` |
| Env in image | `LORA_ADAPTER_DIR=/root/app/deploy/lora_adapter`, `PYTHONPATH=/root/app`, HF/fastembed cache dirs |
| Files shipped | `static/`, `deploy/`, `search/`, `indexing/`, `app.py`, `config.py` (never `Raw_Data/` or `Processed_Data/`) |

**Modal free "Starter" tier**: **$30 / month** of compute credit, no credit card required to
start; if the credit is exhausted without billing added, the app **stops** rather than
incurring charges. Cost model is pay-per-second-served, so a scale-to-zero demo stays well
inside the credit.

### 9.3 Qdrant Cloud

- Free cluster hosting the single `abo_products` collection (56,427 points).
- Accessed by `search/engine.py` via `QDRANT_URL` + `QDRANT_API_KEY` env vars; falls back
  to a local on-disk Qdrant (`Processed_Data/qdrant_db/`) if those are unset.

### 9.4 AWS S3

- Bucket: **`processed-abo-dataset`**, region **`eu-north-1`** (Stockholm).
- `images/` prefix is **public-read** via bucket policy (`s3:GetObject`, `Principal: *`).
- Holds ~43,631 product JPEGs + `dataset/products.parquet`.
- The browser loads each result image **directly** from S3 via the `image_url` in the
  Qdrant payload — the app server never proxies images.

### 9.5 Secrets & config (`.env`, not committed)

`QDRANT_URL`, `QDRANT_API_KEY`, `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`,
`AWS_BUCKET_NAME`, `AWS_REGION`. `.env.example` documents the shape. `.gitignore` blocks
`.env`, `Raw_Data/`, `Processed_Data/`, all model/embedding binaries (`*.npy`, `*.onnx`,
`*.safetensors`, …) — with an explicit exception for the 4.6 MB `deploy/lora_adapter/`.

---

## 10. The Deployment Migration Story (Render → HF → Modal)

The serving layer went through three hosts before landing on Modal. This is worth recording
because it explains several files (and their removal).

| Date | Attempt | Outcome |
|---|---|---|
| 31 Aug 2026 | **Render (free, 512 MB)** with an **ONNX** inference path (`RENDER=True` branch in `inference.py`, `scripts/export_onnx.py`, `render.yaml`, `requirements-render.txt`) | **Failed.** The ONNX model was git-ignored and its weights file (441 MB+) exceeded GitHub limits, so the container crashed on startup: `NoSuchFile: models/onnx/siglip_text.onnx` → `Exited with status 3`. Even fixed, the SigLIP 2 text encoder (256 k Gemma vocab ≈ 1 GB, ~300 MB int8) + SPLADE can't fit 512 MB. |
| 01 Sep 2026 | **Hugging Face Spaces (Docker, free)** — `Dockerfile`, `requirements-hf.txt`, README Space front-matter | **Blocked.** HF made Gradio/Docker Spaces **require a paid plan (PRO $9/mo)** around 08 Jul 2026; only Static Spaces remain free. Pre-existing Docker Spaces are grandfathered, but new ones (and duplicates) are paid. |
| 01 Sep 2026 | **Modal (free Starter, $30/mo credit)** — `modal_app.py`, `requirements-modal.txt` | **Success.** Runs the full unchanged PyTorch path; 16 GB-class memory available; scales to zero. |
| 02 Sep 2026 | S3 **region fix** (`us-east-1` → `eu-north-1`) via `scripts/fix_image_urls.py` | Images now load. |

All Render/HF files were removed from the repo. `deploy/lora_adapter/` and the Qdrant
Cloud + S3 setup survived every migration unchanged.

---

## 11. File-by-File Reference

### Root

| File | What it does |
|---|---|
| `app.py` | FastAPI application. Lifespan loads one `HybridSearchEngine`. Routes: `/` (frontend), `/api/search`, `/api/health`, `/static`, conditional `/images`. Run `python app.py` → serves on `:8000`. |
| `config.py` | Single source of truth for all paths + constants. `Config(project_root)` builds every data path. Holds `BASE_MODEL_NAME`, `DROPPED_PRODUCT_TYPES`, `LANGUAGE_PRIORITY`, `COUNTRY_PRIORITY`. Resolves the LoRA adapter dir (`LORA_ADAPTER_DIR` env → processed → `deploy/`). |
| `modal_app.py` | Modal deployment entry point. Defines the container image (CPU torch + `requirements-modal.txt`), bakes model weights in at build, mounts the LoRA adapter, injects the `qdrant` secret, serves `app.py`'s FastAPI app as a Modal web endpoint. `modal deploy modal_app.py`. |
| `requirements.txt` | Full local/dev + training deps: pandas, pyarrow, torch, torchvision, transformers, peft, accelerate, fastembed, qdrant-client, fastapi, uvicorn, onnxruntime, optimum, Pillow, boto3, python-dotenv. |
| `requirements-modal.txt` | Slim serving deps for the Modal image (no torch here — installed separately as the CPU wheel): fastapi, uvicorn, transformers ≥4.49, peft ≥0.19, sentencepiece, Pillow, fastembed, qdrant-client, python-dotenv. |
| `test_search.py` | Ad-hoc smoke test — embeds `"stylish black leather boots"` and runs one hybrid RRF query against a **local** Qdrant, prints top 3. |
| `.env.example` | Template for the 6 required secrets (Qdrant × 2, AWS × 4). |
| `.gitignore` | Blocks data dirs, `.env*`, and model/embedding binaries; explicitly re-includes `deploy/lora_adapter/*.safetensors`. |
| `README.md` | Public project readme — architecture diagram, the 4 phases, quick-start, deployment pointer, tech-stack + dataset-stats tables. |

### `data_pipeline/` — Phase 1 ETL

| File | What it does |
|---|---|
| `download.py` | Verifies the raw ABO assets exist locally (listings dir has `*.json.gz`, `images.csv.gz` present, `images/small/` present). Raises `FileNotFoundError` if not. No actual downloading. |
| `loader.py` | `ABOLoader`. `load_image_metadata()` reads the gzipped image CSV into an `image_id → path` dict. `iter_listings()` **streams** raw JSON records line-by-line from all `*.json.gz` archives (generator, RAM-safe). |
| `schema.py` | `Product` frozen dataclass (the 9-field contract) + `build_caption(product, include_brand)` which turns clean attributes into a query-style training caption ("a brown suede shoes in the loafer category by The Fix"). |
| `clean.py` | `ABOProcessor` — the full filter/dedup/join pipeline (Section 4.2). `main()` writes `products.parquet`, copies the 43,631 active images into `Processed_Data/images/`, writes `images_meta.parquet` and `stats/dataset_stats.json`. |
| `package_for_kaggle.py` | Zips `products.parquet` + the 43,631 referenced images into `kaggle_dataset.zip` (deflated) for upload to Kaggle as a dataset. |

### `training/` — Phase 2 fine-tuning

| File | What it does |
|---|---|
| `lora_config.py` | Two dataclasses: `LoRAConfig` (r=16, alpha=32, dropout=0.05, target `q_proj`/`v_proj`, bias none) and `TrainingConfig` (3 epochs, batch 32, LR 2e-4, cosine, warmup 0.1, wd 0.01, clip 1.0, fp16, grad-checkpointing, 10 % val, seed 42). All "magic numbers" live here. |
| `train.py` | The fine-tuning loop. `ABOProductDataset` serves (image, caption) pairs with per-epoch caption-variant alternation. `build_lora_model()` wraps SigLIP 2 with LoRA. `run_training()`: AdamW + cosine schedule + fp16 GradScaler + grad accumulation + grad clipping; per-epoch eval on sigmoid loss; saves per-epoch, `best_checkpoint/`, and `lora_adapter_final/`. Meant for a Kaggle T4 GPU. |
| `__init__.py` | Package marker. |

### `indexing/` — Phase 3

| File | What it does |
|---|---|
| `inference.py` | `InferenceEngine` — loads the fine-tuned SigLIP 2 (base + LoRA via PEFT) for dense text/image embeddings and `fastembed` SPLADE for sparse. `embed_text_dense`, `embed_text_sparse`, `embed_images_dense`. Used by both the indexer and the live search engine. |
| `build_index.py` | Builds the **local** Qdrant collection `abo_products`. Loads pre-computed Kaggle `image_embeddings.npy` / `sparse_text_embeddings.json` if present (else regenerates on CPU). Creates the hybrid collection (dense 768 cosine + sparse), upserts 56,427 points (batch 256, `uuid5` IDs, full row as payload). |
| `__init__.py` | Package marker. |

### `search/` — Phase 4 core

| File | What it does |
|---|---|
| `engine.py` | `HybridSearchEngine` — constructs the `InferenceEngine`, connects to Qdrant Cloud (env) or local. `search(query, top_k)`: dense + sparse embed → `query_points` with two `Prefetch`es (`limit = top_k*3`) + `FusionQuery(Fusion.RRF)` → format hits (score, all payload fields, `image_url`). |
| `__init__.py` | Package marker / docstring. |

### `scripts/` — cloud data ops

| File | What it does |
|---|---|
| `cloud_reindex.py` | Upserts all 56,427 points (vectors from `.npy`/`.json`, payload from parquet) into **Qdrant Cloud**. Rewrites payload: drops `main_image_path`, adds S3 `image_url`. Integer point IDs. Batch 128, 120 s client timeout, exponential-backoff retry (5 attempts). |
| `upload_to_s3.py` | Uploads `products.parquet` + every `Processed_Data/images/**/*.jpg` to S3 under `images/<name>`. 20-thread pool, `head_object` skip-if-exists, sets `ContentType`. |
| `fix_image_urls.py` | One-off repair: scrolls `abo_products`, `set_payload`s a region-corrected `image_url` only where wrong. Idempotent. Defaults target region to `eu-north-1`. |

### `deploy/`

| Path | What it is |
|---|---|
| `deploy/lora_adapter/adapter_config.json` | PEFT config for the shipped adapter (base `google/siglip2-base-patch16-224`, r=16, α=32, target `q_proj`/`v_proj`). |
| `deploy/lora_adapter/adapter_model.safetensors` | The trained LoRA weights, ~4.6 MB — the only model binary committed to git. |

### `static/` — frontend

| File | What it is |
|---|---|
| `index.html` | The single page (hero, stat chips, search bar, results grid, footer with model links). |
| `styles.css` | 608-line dark-mode design system (tokens, glassmorphism, cards, skeletons, responsive grid). |
| `app.js` | Client logic — search on click/Enter, skeletons, `fetch` `/api/search`, card rendering with relevance bars, image fallback, XSS-safe escaping, UI states. |

### `docs/`

| File | What it is |
|---|---|
| `introduction.tex` | LaTeX intro / background section (traditional vs dense vs sparse vs hybrid retrieval; project scope & objectives). |
| `DEPLOY_MODAL.md` | Step-by-step Modal deployment guide. |
| `coursework_paper.tex` | (untracked) coursework write-up draft. |
| `PROJECT_DOCUMENTATION.md` | This document. |

### Removed (earlier deployment attempts, no longer in the repo)

`render.yaml`, `requirements-render.txt`, `scripts/export_onnx.py`, `Dockerfile`,
`.dockerignore`, `requirements-hf.txt`, `docs/DEPLOY_HF.md`, and the ONNX/`RENDER` branch of
`inference.py` — see Section 10.

---

## 12. Complete Table of Numerical Values

### Dataset & ETL

| Quantity | Value |
|---|---|
| ABO raw listings | 147,702 |
| ABO full image catalog | 398,212 files |
| Dropped — swatch titles | 439 |
| Dropped — missing/excluded product_type | 64,853 |
| Dropped — no English title | 24,829 |
| Dropped — no valid image | 432 |
| Dropped — cross-marketplace duplicate | 722 |
| **Final catalog (kept)** | **56,427** |
| Unique active images | 43,631 |
| Product schema fields | 9 |
| Language priority list | 5 (`en_US`, `en_CA`, `en_GB`, `en_AU`, `en_IN`) |
| Country dedup priorities | US 1, CA 2, GB 3, AU 4, IN 5, other 99 |
| Dropped product-type list | 1 (`cellular_phone_case`) |
| `products.parquet` size | ≈ 4.7 MB |
| `kaggle_dataset.zip` contents | parquet + 43,631 images |

### Model / fine-tuning

| Quantity | Value |
|---|---|
| Base model | `google/siglip2-base-patch16-224` |
| Image resolution | 224 × 224, patch 16 |
| Embedding dim | 768 |
| Text vocab (Gemma tokenizer) | 256,000 |
| Base params (approx) | ~375 M |
| LoRA rank `r` | 16 |
| LoRA `alpha` | 32 (scale α/r = 2.0) |
| LoRA dropout | 0.05 |
| LoRA target modules | `q_proj`, `v_proj` (vision + text) |
| LoRA bias | none |
| Trainable params (approx) | ≈ 1.18 M (~0.3 %) |
| Adapter file size | ≈ 4.6 MB (4,732,520 bytes) |
| Epochs | 3 |
| Batch size | 32 |
| Gradient accumulation | 1 |
| Peak learning rate | 2 × 10⁻⁴ |
| Warmup ratio | 0.1 |
| Weight decay | 0.01 |
| Max grad norm | 1.0 |
| LR scheduler | cosine + warmup |
| Precision | fp16 mixed |
| Gradient checkpointing | on (~20 % throughput cost) |
| Logging interval | 50 steps |
| Validation split | 10 % (≈ 5,643 val / ≈ 50,784 train) |
| Seed | 42 |
| Caption variants per product | 2 (with / without brand) |
| Target GPU | Kaggle T4, 16 GB VRAM |
| Loss | SigLIP sigmoid (pairwise) |

### Embeddings & index

| Quantity | Value |
|---|---|
| Dense vector | 768-d float32, L2-normalised, cosine |
| `image_embeddings.npy` | 56,427 × 768 × 4 B = 173,343,872 B (≈ 165 MB) |
| Sparse model | `prithivida/Splade_PP_en_v1` |
| `sparse_text_embeddings.json` | ≈ 92 MB |
| Sparse doc fields | title + brand + color + material + category |
| Qdrant collection | `abo_products` |
| Points | 56,427 |
| Local index batch size | 256 |
| Cloud upsert batch size | 128 |
| Cloud client timeout | 120 s |
| Cloud upsert retries | 5 (exponential backoff) |
| `fix_image_urls.py` payloads corrected (incident) | 3,341 |

### API & retrieval

| Quantity | Value |
|---|---|
| `top_k` default | 12 |
| `top_k` range | 1–50 |
| Query min length | 1 |
| Per-signal prefetch limit | `top_k × 3` (36 at default) |
| Fusion | RRF (`Fusion.RRF`), no weights |
| Score rounding | 4 decimal places |
| Measured latency (Modal) | ≈ 230–740 ms (~290 ms typical) |
| Frontend result cards | 12 |
| Frontend skeleton cards | 12 |

### Deployment

| Quantity | Value |
|---|---|
| Modal app | `contextclosest-search` |
| Modal CPU | 2.0 cores |
| Modal memory | 4096 MB |
| Modal `scaledown_window` | 300 s |
| Modal `min_containers` | 0 |
| Modal request timeout | 300 s |
| Modal concurrency | 8 inputs / container |
| Modal free credit | $30 / month |
| Python version (image) | 3.11 |
| S3 bucket | `processed-abo-dataset` |
| S3 region | `eu-north-1` |
| S3 upload threads | 20 |
| Qdrant Cloud collection points | 56,427 |
| Required env vars | 6 |

### Frontend design tokens

| Token | Value |
|---|---|
| Background primary / secondary / card | `#0d1117` / `#161b22` / `#1c2333` |
| Accent gradient | `#6366f1` → `#8b5cf6` (indigo → violet) |
| Font | Inter (weights 300–800) |
| Border radii | 8 / 12 / 16 px |
| `styles.css` size | 608 lines |

---

## 13. Technology Stack & Versions

| Layer | Technology | Version (as deployed) |
|---|---|---|
| Language | Python | 3.11 |
| Dense model | SigLIP 2 (`google/siglip2-base-patch16-224`) + LoRA | — |
| Fine-tuning | HuggingFace `transformers` + `peft` | transformers ≥ 4.49 (5.16.1 observed), peft 0.19.1 |
| Sparse model | SPLADE (`prithivida/Splade_PP_en_v1`) via `fastembed` | fastembed 0.8.0 |
| Deep learning | PyTorch (CPU wheel in prod) | 2.x |
| Vector DB | Qdrant (`qdrant-client`) | 1.19.x |
| Backend | FastAPI + Uvicorn | fastapi 0.141.x, uvicorn 0.52.x |
| Frontend | Vanilla HTML/CSS/JS (no framework), Inter font | — |
| Serverless host | Modal | — |
| Object storage | AWS S3 (`boto3`) | — |
| Data processing | pandas, pyarrow, gzip streaming | pandas ≥ 2.0, pyarrow ≥ 12.0 |
| GPU training | Kaggle T4 | — |

---

## 14. How to Run / Reproduce

### Local development

```bash
conda create -n ai_env python=3.11 && conda activate ai_env
pip install -r requirements.txt
cp .env.example .env          # fill in Qdrant + AWS creds

# Phase 1 — build the clean catalog (needs Raw_Data/)
python data_pipeline/download.py       # verify raw assets
python data_pipeline/clean.py          # → products.parquet, images/, stats

# Phase 2 — fine-tune (real run: Kaggle T4; local = smoke test only)
python data_pipeline/package_for_kaggle.py   # → kaggle_dataset.zip
python training/train.py

# Phase 3 — build the local vector index
python indexing/build_index.py         # → local Qdrant at Processed_Data/qdrant_db/

# Phase 4 — run the app
python app.py                          # → http://localhost:8000
```

### Push to the cloud

```bash
python scripts/upload_to_s3.py         # images + parquet → S3
python scripts/cloud_reindex.py        # vectors + payload → Qdrant Cloud
```

### Deploy (Modal)

```bash
pip install modal
modal token new
modal secret create qdrant QDRANT_URL="..." QDRANT_API_KEY="..."
modal deploy modal_app.py
```

Full guide: `docs/DEPLOY_MODAL.md`.

---

## 15. Known Limitations

- **No relevance benchmark.** There is no labelled query→product test set, so there are no
  recall@k / nDCG numbers and no measured "hybrid vs dense-only" comparison. Quality claims
  are qualitative.
- **Dataset has no external brands.** ABO is Amazon's own catalog — every brand is
  Amazon private-label (Amazon Essentials, `find.`, RED WAGON, Rivet, Alkove, …). Queries
  like *"Nike shoes"* can never match.
- **Conversational queries dilute the signal.** Filler words ("hey I was looking for…")
  are embedded verbatim; there is no query-cleaning / rewriting step.
- **Cold starts.** The Modal container scales to zero; the first request after an idle
  period pays a ~20–40 s model-load penalty.
- **CPU inference.** Production runs SigLIP 2 on CPU (a few hundred ms per query); no GPU
  serving.
- **Single image per product.** Only `main_image_id` is embedded; ABO's alternate views
  and 3D models are unused.
- **`build_index.py` sparse fallback** references `engine.embed_text_sparse` even in the
  pre-computed-`.npy` branch where `engine` may be an image-only setup — the pre-computed
  JSON path is the supported one.

---

## 16. Glossary

| Term | Meaning |
|---|---|
| **ABO** | Amazon Berkeley Objects — a public Amazon product dataset (147,702 listings, 398,212 images). |
| **SigLIP / SigLIP 2** | Sigmoid Loss for Language-Image Pre-training. A CLIP-style dual-encoder that uses a pairwise sigmoid loss instead of softmax, giving better small-batch behaviour. |
| **Dense retrieval** | Encoding text/images to fixed-length continuous vectors and ranking by vector similarity (cosine). Captures semantics; blurs exact tokens. |
| **Sparse (neural) retrieval** | Learned high-dimensional but mostly-zero vectors over a vocabulary. Keeps exact-term signal + learned term expansion. Runs on an inverted index. |
| **SPLADE** | Sparse Lexical AND Expansion model — a transformer that produces sparse term-weight vectors with learned expansion. Here: `Splade_PP_en_v1`. |
| **LoRA** | Low-Rank Adaptation — freezes the base model and trains small rank-`r` update matrices on chosen layers. Cheap, fast, avoids catastrophic forgetting. |
| **PEFT** | Parameter-Efficient Fine-Tuning — the HuggingFace library / family of methods (LoRA, prefix-tuning, …). |
| **RRF** | Reciprocal Rank Fusion — combine ranked lists by summing `1/(k + rank)` per document. No weight tuning; rewards cross-list agreement. |
| **Qdrant** | Open-source vector database; supports named dense + sparse vectors per point and server-side fusion queries. |
| **Prefetch (Qdrant)** | A sub-query whose top results feed a parent fusion/rerank query. |
| **fastembed** | Lightweight Qdrant library that runs embedding models (incl. SPLADE) via ONNX with no PyTorch dependency. |
| **Modal** | Serverless compute platform for Python; deploy functions/containers that scale to zero. |
| **Cold start** | The latency of the first request after a serverless container has been shut down (image pull + process start + model load). |
| **Catastrophic forgetting** | When fine-tuning erases a model's pretrained general capabilities. LoRA + frozen base weights mitigates it. |
| **Vocabulary mismatch** | Lexical search failing because query and document use different words for the same concept. |
