"""
data_pipeline/kafka_producer.py

Publishes new catalog products to the Kafka topic 'catalog-updates'.

The usual input is the Parquet written by spark_clean.py in incremental mode
(only products that are not indexed yet).  Each product becomes one JSON
message keyed by its item_id, so every update to the same product lands on the
same partition and is consumed in order.  The streaming indexer
(kafka_consumer.py) embeds and indexes them.

With --upload-images, each product's image is first uploaded to the public S3
bucket (skipped when it is already there) and its image_url is set, so the
deployed site can display the new product.

Usage:
    python data_pipeline/kafka_producer.py --parquet Processed_Data/new_products.parquet
    python data_pipeline/kafka_producer.py --parquet Processed_Data/new_products.parquet --upload-images
    python data_pipeline/kafka_producer.py --file new_products.jsonl
    python data_pipeline/kafka_producer.py --product '{"item_id": "B0XXXX", "title": "...", "image_url": "https://..."}'

Message fields: item_id and title (required), brand, color, material,
product_type, category, main_image_id, main_image_path, image_url.
The producer adds event_ts (publish time) and run_id / run_total, which the
consumer uses for its progress bar.
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv

from config import Config

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("kafka_producer")

KAFKA_BOOTSTRAP = os.getenv("KAFKA_BOOTSTRAP", "localhost:9092")
CATALOG_TOPIC   = "catalog-updates"


def configure_output(verbose: bool) -> None:
    """Clean console output by default; full logs with --verbose."""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")   # ✓ on Windows consoles
        except (AttributeError, ValueError):
            pass
    if not verbose:
        import warnings

        warnings.filterwarnings("ignore")
        logging.getLogger().setLevel(logging.WARNING)
        logging.getLogger("kafka").setLevel(logging.ERROR)


def get_producer():
    """Creates a KafkaProducer, or exits with a helpful message."""
    try:
        from kafka import KafkaProducer  # type: ignore
    except ImportError:
        logger.error("kafka-python is not installed: pip install -r requirements-bigdata.txt")
        sys.exit(1)

    try:
        return KafkaProducer(
            bootstrap_servers=[KAFKA_BOOTSTRAP],
            key_serializer=lambda k: str(k).encode("utf-8"),
            value_serializer=lambda v: json.dumps(v).encode("utf-8"),
            acks="all",   # wait for the broker so we know each message is durable
        )
    except Exception as exc:  # noqa: BLE001
        print(f"✗ Cannot connect to Kafka at {KAFKA_BOOTSTRAP}: {exc}\n"
              "  Start it with: docker compose -f docker-compose-bigdata.yml up -d")
        sys.exit(1)


# ── Input ──────────────────────────────────────────────────────────────────────

def load_products(args) -> list[dict]:
    """Reads products from --parquet, --file or --product."""
    if args.parquet:
        import pandas as pd

        df = pd.read_parquet(args.parquet)
        items = [
            {k: (None if pd.isna(v) else v) for k, v in row.items()}
            for row in df.to_dict(orient="records")
        ]
        source = "the Spark output"
    elif args.file:
        path = Path(args.file)
        if not path.exists():
            print(f"✗ File not found: {path}")
            sys.exit(1)
        content = path.read_text(encoding="utf-8").strip()
        if content.startswith("["):
            items = json.loads(content)
        else:
            items = [json.loads(line) for line in content.splitlines() if line.strip()]
        source = path.name
    else:
        try:
            items = [json.loads(args.product)]
        except json.JSONDecodeError as exc:
            print(f"✗ Invalid JSON: {exc}")
            sys.exit(1)
        source = "--product"

    valid = [p for p in items if isinstance(p, dict) and p.get("item_id") and p.get("title")]
    if len(valid) < len(items):
        logger.warning("Skipped %d products without item_id or title.", len(items) - len(valid))
    print(f"✓ Loaded {len(valid)} products from {source}", flush=True)
    return valid


# ── Images ─────────────────────────────────────────────────────────────────────

class ImageUploader:
    """
    Uploads product images to the public S3 bucket, once per file.

    Images are stored as images/<file name>, the same key scheme the batch
    upload (scripts/upload_to_s3.py) and the Spark job's image_url use.
    """

    def __init__(self) -> None:
        import boto3

        self.bucket = os.getenv("AWS_BUCKET_NAME")
        region = os.getenv("AWS_REGION", "us-east-1")
        if not all([self.bucket, os.getenv("AWS_ACCESS_KEY_ID"), os.getenv("AWS_SECRET_ACCESS_KEY")]):
            print("✗ --upload-images needs AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY and AWS_BUCKET_NAME in .env")
            sys.exit(1)
        self.s3 = boto3.client(
            "s3",
            region_name=region,
            aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
        )
        self.base_url = f"https://{self.bucket}.s3.{region}.amazonaws.com/images/"
        self.images_dir = Config(PROJECT_ROOT).images_base_dir
        self.uploaded = 0
        self.existing = 0

    def _exists(self, key: str) -> bool:
        from botocore.exceptions import ClientError

        try:
            self.s3.head_object(Bucket=self.bucket, Key=key)
            return True
        except ClientError as exc:
            if exc.response.get("Error", {}).get("Code") in ("404", "NoSuchKey", "NotFound"):
                return False
            raise

    def ensure(self, product: dict) -> Optional[str]:
        """Returns the product's public image URL, uploading the image if needed."""
        rel_path = product.get("main_image_path")
        if not rel_path:
            return product.get("image_url")
        local = self.images_dir / rel_path
        if not local.exists():
            logger.warning("Image not found locally for %s: %s", product["item_id"], local)
            return product.get("image_url")

        key = f"images/{local.name}"
        if self._exists(key):
            self.existing += 1
        else:
            self.s3.upload_file(str(local), self.bucket, key, ExtraArgs={"ContentType": "image/jpeg"})
            self.uploaded += 1
        return self.base_url + local.name


# ── Publishing ─────────────────────────────────────────────────────────────────

def publish(producer, product: dict) -> None:
    """
    Publishes one product.  The key is the item_id (per-product ordering) and
    event_ts records the publish time.
    """
    product = {**product, "event_ts": time.time()}
    meta = producer.send(CATALOG_TOPIC, key=product["item_id"], value=product).get(timeout=10)
    logger.info("Published item_id=%-12s -> partition=%d offset=%d  '%s'",
                product["item_id"], meta.partition, meta.offset, product["title"][:55])


def publish_many(producer, items: list[dict], delay: float, verbose: bool) -> None:
    """Publishes products with a progress bar; tags them with one run id."""
    from tqdm import tqdm

    run = {"run_id": f"run-{int(time.time() * 1000)}", "run_total": len(items)}
    bar_format = "{desc}: {percentage:3.0f}%|{bar:30}| {n_fmt}/{total_fmt} [{elapsed}]"
    for item in tqdm(items, desc="Sending products to Kafka", unit="product",
                     file=sys.stdout, disable=verbose, bar_format=bar_format):
        publish(producer, {**item, **run})
        if delay:
            time.sleep(delay)
    print(f"✓ {len(items)} products sent to Kafka topic '{CATALOG_TOPIC}'", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Publish new catalog products to Kafka.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--parquet", help="Parquet output of spark_clean.py (file or directory).")
    source.add_argument("--file", help="JSON list or JSON-lines file of products.")
    source.add_argument("--product", help="A single product as a JSON string.")
    parser.add_argument("--upload-images", action="store_true",
                        help="Upload each product's image to the public S3 bucket and set image_url.")
    parser.add_argument("--delay", type=float, default=0.0,
                        help="Seconds to wait between messages (default 0).")
    parser.add_argument("--dry-run", action="store_true",
                        help="Load products (and upload images) but do not publish to Kafka.")
    parser.add_argument("--verbose", action="store_true",
                        help="Log every message (partition / offset) instead of a progress bar.")
    args = parser.parse_args()
    configure_output(args.verbose)

    items = load_products(args)
    if not items:
        return

    if args.upload_images:
        uploader = ImageUploader()
        for item in items:
            url = uploader.ensure(item)
            if url:
                item["image_url"] = url
        print(f"✓ Images on S3: {uploader.uploaded} uploaded, {uploader.existing} already there", flush=True)

    if args.dry_run:
        print(f"✓ Dry run: {len(items)} products ready, nothing published", flush=True)
        return

    producer = get_producer()
    print(f"✓ Connected to Kafka broker at {KAFKA_BOOTSTRAP}", flush=True)
    publish_many(producer, items, delay=args.delay, verbose=args.verbose)
    producer.flush()
    producer.close()


if __name__ == "__main__":
    main()
