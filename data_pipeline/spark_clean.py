"""
data_pipeline/spark_clean.py

Distributed ETL Pipeline for the ABO Product Catalog — Apache Spark.

This is the Spark version of data_pipeline/clean.py.  It applies the same
cleaning rules, but as distributed DataFrame transformations:

  1. Read        — Parallel ingestion of gzipped JSON-lines listing files.
  2. Filter      — Drop swatches, unsupported product types (phone cases),
                   listings without an English title, listings without an image.
  3. Join        — Join with the ABO image metadata (images.csv.gz) to resolve
                   each listing's main image path.
  4. Deduplicate — Window function (row_number over item_id, ordered by
                   marketplace priority US > CA > GB > AU > IN) keeps the best
                   listing per ASIN without collecting data to the driver.
  5. Anti-join   — (optional) drop products that are already in the live
                   search index, so only genuinely NEW products move on to the
                   Kafka streaming stage.  --against-index reads the indexed
                   item_ids from Qdrant; --existing takes them from a Parquet file.
  6. Write       — Snappy-compressed Parquet + a JSON funnel report.

Two modes:
    # Full catalog (147,702 raw listings) — the batch layer:
    python data_pipeline/spark_clean.py

    # Incremental: clean a new feed and keep only products not yet indexed:
    python data_pipeline/spark_clean.py \\
        --input  path/to/new_feed_dir \\
        --against-index \\
        --output Processed_Data/new_products.parquet

The feed uses the raw ABO listing format (JSON lines, optionally .gz); each
product's main_image_id must appear in the image metadata (--images_meta).
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv

from config import Config

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


# ── Cleaning constants (mirror config.py) ──────────────────────────────────────
LANGUAGE_PRIORITY = ["en_US", "en_CA", "en_GB", "en_AU", "en_IN"]
COUNTRY_PRIORITY  = {"US": 1, "CA": 2, "GB": 3, "AU": 4, "IN": 5}
DROPPED_TYPES     = ["cellular_phone_case"]


# ── Native Spark SQL column expressions (default, run entirely in the JVM) ─────

def _native_columns(raw_df):
    """
    Builds the flattened columns with Spark's built-in higher-order functions
    (filter / exists / get).  No Python worker round-trip per row, so this is
    far faster than the equivalent Python UDFs below.
    """
    from pyspark.sql import functions as F

    def elem_fields(name):
        return raw_df.schema[name].dataType.elementType.fieldNames()

    def english(name, use_standardized=False):
        if name not in raw_df.columns:
            return F.lit(None).cast("string")
        fields = elem_fields(name)
        arr = F.col(name)

        def value_of(x):
            v = x["value"].cast("string")
            if use_standardized and "standardized_values" in fields:
                v = F.coalesce(F.get(x["standardized_values"], 0).cast("string"), v)
            return v

        def last_match(pred):
            # Python's dict overwrite keeps the LAST value per language; match it.
            matches = F.filter(arr, lambda x: pred(x) & value_of(x).isNotNull())
            return value_of(F.get(matches, F.size(matches) - 1))

        candidates = [last_match(lambda x, l=lang: x["language_tag"] == l) for lang in LANGUAGE_PRIORITY]
        candidates.append(last_match(lambda x: x["language_tag"].startswith("en_")))
        return F.coalesce(*candidates)

    def col_or_null(name):
        return F.col(name) if name in raw_df.columns else F.lit(None)

    is_swatch = (
        F.coalesce(F.exists("item_name", lambda x: F.lower(x["value"].cast("string")).contains("swatch")),
                   F.lit(False))
        if "item_name" in raw_df.columns else F.lit(False)
    )
    product_type = (F.lower(F.get(F.col("product_type"), 0)["value"].cast("string"))
                    if "product_type" in raw_df.columns else F.lit(None).cast("string"))
    if "node" in raw_df.columns:
        leaf = F.element_at(
            F.split(F.regexp_replace(F.get(F.col("node"), 0)["node_name"], "^/+|/+$", ""), "/"), -1)
        category = F.when(leaf == "", None).otherwise(leaf)
    else:
        category = F.lit(None).cast("string")

    return [
        F.col("item_id"),
        col_or_null("country").alias("country"),
        is_swatch.alias("is_swatch"),
        product_type.alias("product_type"),
        english("item_name").alias("title"),
        english("color", use_standardized=True).alias("color"),
        F.coalesce(english("material"), english("fabric_type")).alias("material"),
        english("brand").alias("brand"),
        category.alias("category"),
        col_or_null("main_image_id").alias("main_image_id"),
    ]


# ── Python UDF variant (--udf) — kept for the performance comparison ───────────

def _as_dict(item):
    """Spark passes array<struct> elements as Row objects; normalise to dict."""
    if item is None:
        return {}
    if hasattr(item, "asDict"):
        return item.asDict()
    return item if isinstance(item, dict) else {}


def _extract_english(field_list, use_standardized=False):
    """Mirrors ABOProcessor._get_english_value() — best English value or None."""
    if not field_list:
        return None
    val_map = {}
    for raw in field_list:
        item = _as_dict(raw)
        lang = item.get("language_tag") or ""
        if use_standardized:
            std = item.get("standardized_values")
            if std and std[0]:
                val_map[lang] = str(std[0])
                continue
        if item.get("value") is not None:
            val_map[lang] = str(item["value"])
    for lang in LANGUAGE_PRIORITY:
        if lang in val_map:
            return val_map[lang]
    for lang, val in val_map.items():
        if lang.startswith("en_"):
            return val
    return None


def _has_swatch(names):
    """True if any language variant of the item name mentions 'swatch'."""
    for raw in names or []:
        val = _as_dict(raw).get("value")
        if val and "swatch" in str(val).lower():
            return True
    return False


def _first_value(field_list):
    """product_type is array<struct<value>>; return the first value, lowercased."""
    for raw in field_list or []:
        val = _as_dict(raw).get("value")
        if val:
            return str(val).lower()
    return None


def _leaf_category(nodes):
    """Last segment of the first browse-node path, e.g. '/Shoes/Loafers' -> 'Loafers'."""
    if not nodes:
        return None
    first = _as_dict(nodes[0])
    path = first.get("node_name") or first.get("path") or ""
    segments = [s for s in path.strip("/").split("/") if s]
    return segments[-1] if segments else None


def _udf_flatten(raw_df, col_or_null):
    """Same flattening as _native_columns, implemented with Python UDFs."""
    from pyspark.sql import functions as F
    from pyspark.sql import types as T

    extract_en      = F.udf(lambda f: _extract_english(f, False), T.StringType())
    extract_en_std  = F.udf(lambda f: _extract_english(f, True), T.StringType())
    has_swatch_udf  = F.udf(_has_swatch, T.BooleanType())
    first_value_udf = F.udf(_first_value, T.StringType())
    leaf_cat_udf    = F.udf(_leaf_category, T.StringType())

    return raw_df.select(
        F.col("item_id"),
        col_or_null("country").alias("country"),
        has_swatch_udf(col_or_null("item_name")).alias("is_swatch"),
        first_value_udf(col_or_null("product_type")).alias("product_type"),
        extract_en(col_or_null("item_name")).alias("title"),
        extract_en_std(col_or_null("color")).alias("color"),
        F.coalesce(
            extract_en(col_or_null("material")),
            extract_en(col_or_null("fabric_type")),
        ).alias("material"),
        extract_en(col_or_null("brand")).alias("brand"),
        leaf_cat_udf(col_or_null("node")).alias("category"),
        col_or_null("main_image_id").alias("main_image_id"),
    )


def _configure_windows_env() -> None:
    """Spark on Windows needs HADOOP_HOME/bin on PATH and an explicit worker Python."""
    import pyspark

    # A system-wide SPARK_HOME (e.g. C:\spark with Spark 3.5) that differs from the
    # pip-installed pyspark version breaks the JVM gateway; always use pyspark's own.
    os.environ["SPARK_HOME"] = str(Path(pyspark.__file__).resolve().parent)
    os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
    os.environ.setdefault("PYSPARK_DRIVER_PYTHON", sys.executable)
    hadoop_home = os.environ.get("HADOOP_HOME")
    if hadoop_home:
        hadoop_bin = str(Path(hadoop_home) / "bin")
        if hadoop_bin not in os.environ.get("PATH", ""):
            os.environ["PATH"] = hadoop_bin + os.pathsep + os.environ.get("PATH", "")


def build_spark(app_name: str, extra_conf: dict | None = None,
                master: str = "local[*]", shuffle_partitions: int = 8):
    """Creates a local SparkSession (default: every available CPU core)."""
    _configure_windows_env()
    if not VERBOSE:
        # conf/log4j2.properties limits Spark's JVM logging to errors.
        os.environ["SPARK_CONF_DIR"] = str(Path(__file__).resolve().parent.parent / "conf")
    from pyspark.sql import SparkSession

    builder = (
        SparkSession.builder
        .appName(app_name)
        .master(master)
        .config("spark.driver.memory", "4g")
        .config("spark.sql.shuffle.partitions", str(shuffle_partitions))
        .config("spark.ui.showConsoleProgress", "false")
    )
    for key, value in (extra_conf or {}).items():
        builder = builder.config(key, value)
    spark = builder.getOrCreate()
    spark.sparkContext.setLogLevel("WARN" if VERBOSE else "ERROR")
    return spark


# Clean demo output by default; scripts set this from their --verbose flag.
VERBOSE = False


def configure_output(verbose: bool) -> None:
    """Clean demo output by default; full logs with --verbose."""
    global VERBOSE
    VERBOSE = verbose
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")   # ✓ on Windows consoles
        except (AttributeError, ValueError):
            pass
    if not verbose:
        logging.getLogger().setLevel(logging.WARNING)
        logging.getLogger("py4j").setLevel(logging.ERROR)


def quiet_exit() -> None:
    """
    Exits without PySpark's Windows shutdown noise.

    On Windows PySpark registers an atexit hook that runs ``taskkill /t`` on the
    spark-submit process tree, which prints "SUCCESS: The process ... has been
    terminated" for every process.  Do the same cleanup silently instead.
    """
    sys.stdout.flush()
    sys.stderr.flush()
    if os.name != "nt" or VERBOSE:
        return
    import subprocess

    from pyspark import SparkContext

    proc = getattr(SparkContext._gateway, "proc", None)
    if proc is not None:
        subprocess.run(["taskkill", "/f", "/t", "/pid", str(proc.pid)],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    os._exit(0)


def _resolve_input(path: str) -> str:
    """Accepts a file or a directory of *.json.gz (Spark reads every file in it)."""
    p = Path(path)
    # Hadoop's glob matcher does not handle Windows drive paths; a plain
    # directory (or file) path in POSIX form works everywhere.
    return p.resolve().as_posix() if p.exists() else path


def run_spark_etl(
    input_path: str,
    output_path: str,
    images_meta_path: str,
    existing_path: str | None = None,
    s3_base_url: str | None = None,
    use_udfs: bool = False,
) -> dict:
    """
    Runs the full Spark ETL and returns the funnel statistics.

    Args:
        input_path:       Raw listings (file, directory or glob of *.json.gz).
        output_path:      Output Parquet directory.
        images_meta_path: ABO images.csv.gz (image_id -> relative path).
        existing_path:    Optional Parquet with an ``item_id`` column listing the
                          products already in the search index (anti-join).
        s3_base_url:      Optional base URL; adds ``image_url`` = base + file name.
        use_udfs:         Use the Python-UDF implementation instead of native
                          Spark expressions (for the performance comparison).
    """
    from pyspark.sql import functions as F
    from pyspark.sql.window import Window

    print("⏳ Starting Apache Spark (local mode)...", flush=True)
    spark = build_spark("ContextClosest-ABO-ETL")
    print(f"✓ Spark {spark.version} running on {spark.sparkContext.defaultParallelism} cores "
          f"— monitor at {spark.sparkContext.uiWebUrl}", flush=True)
    print("⏳ Cleaning raw listings: flatten JSON -> filter -> join images -> dedup -> anti-join...",
          flush=True)
    logger.info("Spark %s running locally on %s cores — UI: %s",
                spark.version, spark.sparkContext.defaultParallelism,
                spark.sparkContext.uiWebUrl)

    t_start = time.perf_counter()

    # ── 1. Read raw listings ───────────────────────────────────────────────────
    source = _resolve_input(input_path)
    logger.info("Reading raw listings from: %s", source)
    raw_df = spark.read.json(source)

    def col_or_null(name: str):
        return F.col(name) if name in raw_df.columns else F.lit(None)

    # ── 2. Flatten nested multilingual fields ─────────────────────────────────
    if not use_udfs:
        logger.info("Flattening with native Spark SQL expressions.")
        flat_df = raw_df.select(*_native_columns(raw_df)).cache()
    else:
        logger.info("Flattening with Python UDFs (slower; comparison mode).")
        flat_df = _udf_flatten(raw_df, col_or_null).cache()

    funnel = {"raw_listings": flat_df.count()}
    country_prio = F.coalesce(
        F.create_map(*[F.lit(x) for kv in COUNTRY_PRIORITY.items() for x in kv])[F.col("country")],
        F.lit(99),
    )

    # ── 3. Rule-based filters (same order as the Pandas pipeline) ─────────────
    df = flat_df.filter(~F.col("is_swatch"))
    funnel["after_swatch_filter"] = df.count()

    df = df.filter(F.col("product_type").isNotNull() & ~F.col("product_type").isin(DROPPED_TYPES))
    funnel["after_product_type_filter"] = df.count()

    df = df.filter(F.col("title").isNotNull())
    funnel["after_english_filter"] = df.count()

    # ── 4. Join with image metadata ───────────────────────────────────────────
    images_df = (
        spark.read.option("header", "true").csv(images_meta_path)
        .select(F.col("image_id"), F.col("path").alias("main_image_path"))
    )
    df = df.join(images_df, df["main_image_id"] == images_df["image_id"], "inner").drop("image_id")
    funnel["after_image_join"] = df.count()

    # ── 5. Cross-marketplace deduplication (Window function) ──────────────────
    window = Window.partitionBy("item_id").orderBy(country_prio.asc())
    df = (
        df.withColumn("rn", F.row_number().over(window))
        .filter(F.col("rn") == 1)
        .drop("rn", "is_swatch", "country")
    )
    funnel["unique_products"] = df.count()

    # ── 6. Anti-join against products already in the search index ─────────────
    if existing_path:
        existing_df = spark.read.parquet(existing_path).select("item_id").distinct()
        funnel["already_indexed_snapshot"] = existing_df.count()
        df = df.join(F.broadcast(existing_df), on="item_id", how="left_anti")
        funnel["new_products"] = df.count()

    if s3_base_url:
        df = df.withColumn(
            "image_url",
            F.concat(F.lit(s3_base_url), F.element_at(F.split("main_image_path", "/"), -1)),
        )

    columns = ["item_id", "product_type", "title", "color", "material", "brand",
               "category", "main_image_id", "main_image_path"]
    if s3_base_url:
        columns.append("image_url")

    # ── 7. Write Parquet ──────────────────────────────────────────────────────
    final_count = funnel.get("new_products", funnel["unique_products"])
    out_partitions = 1 if final_count < 50_000 else 4
    df.select(*columns).coalesce(out_partitions).write.mode("overwrite").parquet(output_path)

    funnel["elapsed_seconds"] = round(time.perf_counter() - t_start, 2)
    spark.stop()

    _print_funnel(funnel)
    report_path = Path(output_path).parent / (Path(output_path).stem + "_funnel.json")
    report_path.write_text(json.dumps(funnel, indent=2))
    final_count = funnel.get("new_products", funnel["unique_products"])
    label = "NEW products" if "new_products" in funnel else "clean products"
    print(f"✓ {final_count:,} {label} written to {output_path}", flush=True)
    logger.info("Funnel report: %s", report_path)
    return funnel


def _print_funnel(funnel: dict) -> None:
    """Prints the record funnel so every cleaning stage is visible."""
    labels = [
        ("raw_listings",              "Raw listings read"),
        ("after_swatch_filter",       "After swatch filter"),
        ("after_product_type_filter", "After product-type filter"),
        ("after_english_filter",      "After English-title filter"),
        ("after_image_join",          "After image-metadata join"),
        ("unique_products",           "After marketplace dedup"),
        ("already_indexed_snapshot",  "Products already in index"),
        ("new_products",              "NEW products (anti-join)"),
    ]
    lines = ["", "=" * 52, "  Spark ETL funnel", "=" * 52]
    for key, label in labels:
        if key in funnel:
            lines.append(f"  {label:<30}{funnel[key]:>12,}")
    lines.append(f"  {'Wall-clock time':<30}{funnel['elapsed_seconds']:>10.1f} s")
    lines.append("=" * 52)
    print("\n".join(lines), flush=True)


def export_indexed_ids(output_path: Path) -> Path:
    """Writes the item_ids currently in the Qdrant collection to a Parquet file."""
    import pandas as pd
    from qdrant_client import QdrantClient

    url, key = os.getenv("QDRANT_URL"), os.getenv("QDRANT_API_KEY")
    config = Config(Path(__file__).resolve().parent.parent)
    client = (QdrantClient(url=url, api_key=key, timeout=120) if url and key
              else QdrantClient(path=str(config.qdrant_db_dir)))

    ids: list[str] = []
    offset = None
    while True:
        points, offset = client.scroll("abo_products", limit=2000, offset=offset,
                                       with_payload=["item_id"], with_vectors=False)
        ids.extend(p.payload["item_id"] for p in points if p.payload)
        if offset is None:
            break
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"item_id": ids}).to_parquet(output_path, index=False)
    print(f"✓ {len(ids):,} item_ids currently in the search index", flush=True)
    return output_path


def main() -> None:
    config = Config(Path(__file__).resolve().parent.parent)
    bucket = os.getenv("AWS_BUCKET_NAME")
    region = os.getenv("AWS_REGION", "us-east-1")
    default_s3 = f"https://{bucket}.s3.{region}.amazonaws.com/images/" if bucket else None

    parser = argparse.ArgumentParser(description="PySpark ETL for the ABO product catalog.")
    parser.add_argument("--input", default=str(config.listings_dir),
                        help="Raw listings: file, directory of *.json.gz, or glob.")
    parser.add_argument("--output", default=str(config.processed_dir / "spark_products.parquet"),
                        help="Output Parquet directory.")
    parser.add_argument("--images_meta", default=str(config.images_meta_path),
                        help="Path to images.csv.gz.")
    parser.add_argument("--against-index", action="store_true",
                        help="Anti-join against the item_ids currently in the Qdrant index.")
    parser.add_argument("--existing", default=None,
                        help="Parquet of item_ids already indexed (enables the anti-join).")
    parser.add_argument("--s3_base", default=default_s3,
                        help="Base URL used to build image_url (default from .env).")
    parser.add_argument("--udf", action="store_true",
                        help="Use Python UDFs instead of native Spark expressions (slower, for comparison).")
    parser.add_argument("--verbose", action="store_true", help="Show full Spark / pipeline logs.")
    args = parser.parse_args()
    configure_output(args.verbose)

    existing = args.existing
    if args.against_index:
        existing = str(export_indexed_ids(config.processed_dir / "indexed_item_ids.parquet"))

    run_spark_etl(
        input_path=args.input,
        output_path=args.output,
        images_meta_path=args.images_meta,
        existing_path=existing,
        s3_base_url=args.s3_base,
        use_udfs=args.udf,
    )
    quiet_exit()


if __name__ == "__main__":
    main()
