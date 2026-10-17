"""
data_pipeline/kafka_consumer.py

Real-Time Catalog Ingestion Consumer (Apache Kafka -> Qdrant).

Instead of re-running the offline batch indexer every time a product is
added, this long-running consumer listens to the 'catalog-updates' topic and
indexes each new product incrementally, so it becomes searchable within
seconds.

Architecture:
    Spark ETL output --(kafka_producer.py)--> Kafka topic 'catalog-updates'
        └─► THIS consumer (micro-batches: 8 messages or 3 s, whichever first)
                ├─► SigLIP 2 + LoRA dense IMAGE embedding (same as batch index)
                ├─► SPLADE sparse text embedding
                ├─► Qdrant upsert (deterministic point id per item_id)
                ├─► Kafka offset commit (only after a successful upsert)
                └─► Redis result-cache invalidation

Delivery semantics: at-least-once.  Offsets are committed manually after the
Qdrant upsert succeeds; if the consumer crashes mid-batch the messages are
re-delivered, and because point ids are derived from item_id the replay
overwrites the same points instead of creating duplicates.

Message schema (JSON) — required: item_id, title.  Optional: brand, color,
material, product_type, category, main_image_id, main_image_path, image_url,
event_ts (producer publish time).

Usage:
    python data_pipeline/kafka_consumer.py              # clean demo output, runs until Ctrl-C
    python data_pipeline/kafka_consumer.py --verbose    # detailed per-message / per-batch logs
    python data_pipeline/kafka_consumer.py --from-beginning
"""

import argparse
import io
import json
import logging
import os
import sys
import time
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv

from config import Config

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("kafka_consumer")
for _noisy in ("httpx", "kafka"):
    logging.getLogger(_noisy).setLevel(logging.WARNING)

# ── Constants ──────────────────────────────────────────────────────────────────
KAFKA_BOOTSTRAP = os.getenv("KAFKA_BOOTSTRAP", "localhost:9092")
CATALOG_TOPIC   = "catalog-updates"
CONSUMER_GROUP  = "contextclosest-indexer"
COLLECTION_NAME = "abo_products"

# Micro-batch: flush after BATCH_SIZE messages or BATCH_MAX_WAIT_S seconds,
# whichever comes first — amortises model overhead without stalling the tail.
BATCH_SIZE: int = 8
BATCH_MAX_WAIT_S: float = 3.0

PROJECT_ROOT = Path(__file__).resolve().parent.parent

PAYLOAD_FIELDS = ("item_id", "product_type", "title", "color", "material",
                  "brand", "category", "main_image_id", "image_url")


def configure_output(verbose: bool) -> None:
    """Clean demo output by default; full logs with --verbose."""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")   # ✓ on Windows consoles
        except (AttributeError, ValueError):
            pass
    if verbose:
        return
    logging.getLogger().setLevel(logging.WARNING)
    for name in ("kafka", "httpx", "transformers", "huggingface_hub", "fastembed", "search.cache"):
        logging.getLogger(name).setLevel(logging.ERROR)
    import warnings

    warnings.filterwarnings("ignore")
    os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")


def point_id_for(item_id: str) -> str:
    """Deterministic Qdrant point id — replays overwrite instead of duplicating."""
    return str(uuid.uuid5(uuid.NAMESPACE_OID, str(item_id)))


def build_search_text(record: dict) -> str:
    """Same text recipe as indexing/build_index.py, so sparse vectors match."""
    parts = [record.get(k) for k in ("title", "brand", "color", "material", "category")]
    return " ".join(str(p) for p in parts if p and str(p) != "None")


class StreamingIndexer:
    """Holds the models, the Qdrant client and the Redis cache for the consumer."""

    def __init__(self, config: Config) -> None:
        from qdrant_client import QdrantClient

        from indexing.inference import InferenceEngine
        from search.cache import RedisCache

        self._config = config

        logger.info("Loading InferenceEngine (SigLIP 2 + LoRA, SPLADE)...")
        self.engine = InferenceEngine(
            base_model_name=config.BASE_MODEL_NAME,
            lora_adapter_path=str(config.lora_adapter_dir),
        )

        qdrant_url, qdrant_key = os.getenv("QDRANT_URL"), os.getenv("QDRANT_API_KEY")
        if qdrant_url and qdrant_key:
            logger.info("Connecting to Qdrant Cloud at %s", qdrant_url)
            self.qdrant = QdrantClient(url=qdrant_url, api_key=qdrant_key, timeout=60)
        else:
            logger.info("Connecting to local Qdrant at %s", config.qdrant_db_dir)
            self.qdrant = QdrantClient(path=str(config.qdrant_db_dir))

        if not self.qdrant.collection_exists(COLLECTION_NAME):
            logger.error("Qdrant collection '%s' not found. Build the index first.", COLLECTION_NAME)
            sys.exit(1)
        count = self.qdrant.count(COLLECTION_NAME, exact=True).count
        logger.info("Collection '%s' currently holds %d products.", COLLECTION_NAME, count)

        self.cache = RedisCache.from_env()

    # ── Image loading ──────────────────────────────────────────────────────────

    def load_image(self, record: dict):
        """Loads the product image from the local ABO folder, else from image_url."""
        from PIL import Image

        rel_path = record.get("main_image_path")
        if rel_path:
            local = self._config.images_base_dir / rel_path
            if local.exists():
                return Image.open(local).convert("RGB")

        url = record.get("image_url")
        if url and url.startswith("http"):
            try:
                import requests

                resp = requests.get(url, timeout=10)
                resp.raise_for_status()
                return Image.open(io.BytesIO(resp.content)).convert("RGB")
            except Exception as exc:  # noqa: BLE001
                logger.warning("Could not download image for %s: %s", record.get("item_id"), exc)
        return None

    # ── Batch processing ───────────────────────────────────────────────────────

    def process_batch(self, records: list[dict], on_product_done=None) -> int:
        """
        Embeds a micro-batch and upserts it into Qdrant.

        ``on_product_done`` is called once per product after its image is
        embedded (drives the progress bar).  Returns the number of stale
        cached search results invalidated in Redis.
        """
        from qdrant_client.models import PointStruct, SparseVector

        t0 = time.perf_counter()

        dense_by_idx: dict[int, list[float]] = {}
        for i, record in enumerate(records):
            image = self.load_image(record)
            if image is not None:
                dense_by_idx[i] = self.engine.embed_images_dense([image])[0]
            if on_product_done:
                on_product_done()
        t_dense = time.perf_counter()

        sparse_vecs = self.engine.embed_text_sparse([build_search_text(r) for r in records])
        t_sparse = time.perf_counter()

        ingested_at = time.time()
        points = []
        for i, (record, sparse) in enumerate(zip(records, sparse_vecs)):
            payload = {k: record.get(k) or "" for k in PAYLOAD_FIELDS}
            payload["source"] = "kafka-stream"
            payload["ingested_at"] = ingested_at

            vector = {"sparse_text": SparseVector(indices=sparse["indices"], values=sparse["values"])}
            if i in dense_by_idx:
                vector["dense_image"] = dense_by_idx[i]
            else:
                # No image: index for keyword (sparse) search only.  Never put a
                # TEXT embedding in the image-vector slot — it lives in a
                # different similarity range and would distort ranking.
                logger.warning("No image for %s — indexed for keyword search only.", record["item_id"])

            points.append(PointStruct(id=point_id_for(record["item_id"]), vector=vector, payload=payload))

        for attempt in range(4):
            try:
                self.qdrant.upsert(collection_name=COLLECTION_NAME, points=points, wait=True)
                break
            except Exception as exc:  # noqa: BLE001
                if attempt == 3:
                    raise
                logger.warning("Qdrant upsert failed (%s); retrying in %ds", exc, 2 ** attempt)
                time.sleep(2 ** attempt)
        t_upsert = time.perf_counter()

        invalidated = self.cache.invalidate_results()

        logger.info(
            "Batch of %d indexed | image-embed %.0f ms | SPLADE %.0f ms | Qdrant upsert %.0f ms "
            "| %d stale cached results invalidated",
            len(records), (t_dense - t0) * 1000, (t_sparse - t_dense) * 1000,
            (t_upsert - t_sparse) * 1000, invalidated,
        )
        return invalidated

    def index_size(self) -> int:
        return self.qdrant.count(COLLECTION_NAME, exact=True).count


def say(message: str) -> None:
    """Clean, demo-friendly status line (printed in both modes)."""
    print(message, flush=True)


def run_consumer(from_beginning: bool = False, verbose: bool = False) -> None:
    try:
        from kafka import KafkaConsumer  # type: ignore
    except ImportError:
        logger.error("kafka-python is not installed: pip install -r requirements-bigdata.txt")
        sys.exit(1)
    from tqdm import tqdm

    say("⏳ Loading AI models (SigLIP 2 + LoRA, SPLADE)...")
    indexer = StreamingIndexer(Config(PROJECT_ROOT))
    say("✓ AI models loaded")
    say(f"✓ Connected to Qdrant vector database — {indexer.index_size():,} products indexed")
    say("✓ Connected to Redis cache" if indexer.cache.is_available else "✗ Redis not reachable (cache disabled)")

    group = f"{CONSUMER_GROUP}-{int(time.time())}" if from_beginning else CONSUMER_GROUP
    consumer = KafkaConsumer(
        CATALOG_TOPIC,
        bootstrap_servers=[KAFKA_BOOTSTRAP],
        group_id=group,
        value_deserializer=lambda b: json.loads(b.decode("utf-8")),
        auto_offset_reset="earliest",
        enable_auto_commit=False,      # commit only after Qdrant upsert succeeds
    )
    say(f"✓ Kafka consumer listening on topic '{CATALOG_TOPIC}' — waiting for new products...\n")

    batch: list[dict] = []
    batch_started = 0.0
    total = 0
    bar = None               # progress bar for the current producer run
    bar_run = None

    def start_bar(record: dict) -> None:
        nonlocal bar, bar_run
        bar_run = record.get("run_id")
        bar = tqdm(total=record.get("run_total"), desc="Indexing new products",
                   unit="product", file=sys.stdout, disable=verbose,
                   bar_format="{desc}: {percentage:3.0f}%|{bar:30}| {n_fmt}/{total_fmt} [{elapsed}]")

    def finish_bar() -> None:
        nonlocal bar, bar_run
        if bar is None:
            return
        done = bar.n
        bar.close()
        bar, bar_run = None, None
        say(f"✓ {done} new products are now searchable — index holds {indexer.index_size():,} products")
        say("✓ Redis search cache refreshed")
        say(f"✓ Listening for more updates on '{CATALOG_TOPIC}'...\n")

    def flush() -> None:
        nonlocal total, batch_started
        if not batch:
            return
        indexer.process_batch(batch, on_product_done=(lambda: bar.update(1)) if bar else None)
        consumer.commit()
        total += len(batch)
        logger.info("Offsets committed. Products indexed this session: %d", total)
        batch.clear()
        batch_started = 0.0
        if bar is not None and bar.total and bar.n >= bar.total:
            finish_bar()

    try:
        while True:
            polled = consumer.poll(timeout_ms=500, max_records=BATCH_SIZE)
            for tp, messages in polled.items():
                for msg in messages:
                    record = msg.value
                    if not isinstance(record, dict) or not record.get("item_id") or not record.get("title"):
                        logger.warning("Skipping malformed message at %s offset %d", tp, msg.offset)
                        continue
                    logger.info("Received  partition=%d offset=%-4d item_id=%s  '%s'",
                                tp.partition, msg.offset, record["item_id"], record["title"][:50])
                    if bar is None or record.get("run_id") != bar_run:
                        if bar is not None:
                            flush()
                            finish_bar()
                        start_bar(record)
                    if not batch:
                        batch_started = time.time()
                    batch.append(record)
                    if len(batch) >= BATCH_SIZE:
                        flush()
            if batch and time.time() - batch_started >= BATCH_MAX_WAIT_S:
                flush()
            if bar is not None and not bar.total and not batch:
                finish_bar()     # producer announced no total: close once the stream drains
    except KeyboardInterrupt:
        flush()
        finish_bar()
    finally:
        consumer.close()
        say(f"Consumer stopped. Products indexed this session: {total}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Kafka -> Qdrant streaming indexer.")
    parser.add_argument("--from-beginning", action="store_true",
                        help="Use a fresh consumer group and replay the whole topic.")
    parser.add_argument("--verbose", action="store_true",
                        help="Detailed logs instead of the clean progress display.")
    args = parser.parse_args()
    configure_output(args.verbose)
    run_consumer(from_beginning=args.from_beginning, verbose=args.verbose)


if __name__ == "__main__":
    main()
