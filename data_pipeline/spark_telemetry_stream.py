"""
data_pipeline/spark_telemetry_stream.py

Real-time search analytics — Kafka -> Spark Structured Streaming -> Parquet.

Every request to /api/search publishes a telemetry event to the Kafka topic
'search-telemetry' (see app.py):

    {"query": str, "count": int, "elapsed_ms": float, "cache_hit": bool, "ts": float}

This job consumes that stream in micro-batches and computes, per tumbling
event-time window:

    searches            number of searches
    unique_queries      approximate distinct query strings
    zero_result_rate    share of searches that returned nothing (catalog gaps)
    cache_hit_rate      share answered from the Redis result cache
    avg_latency_ms / p95_latency_ms   server-side latency

plus a running "top queries" leaderboard.  Window metrics are appended to
Parquet (Processed_Data/telemetry/window_metrics) for downstream analysis.

Usage:
    python data_pipeline/spark_telemetry_stream.py                 # 1-minute windows
    python data_pipeline/spark_telemetry_stream.py --window "30 seconds" --from-beginning
"""

import argparse
import logging
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from data_pipeline.spark_clean import build_spark

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("telemetry_stream")

KAFKA_BOOTSTRAP = os.getenv("KAFKA_BOOTSTRAP", "localhost:9092")
TOPIC           = "search-telemetry"
OUT_DIR         = PROJECT_ROOT / "Processed_Data" / "telemetry"


def main() -> None:
    parser = argparse.ArgumentParser(description="Spark Structured Streaming search analytics.")
    parser.add_argument("--window", default="1 minute", help="Tumbling window size.")
    parser.add_argument("--trigger", default="10 seconds", help="Micro-batch interval.")
    parser.add_argument("--cores", type=int, default=2, help="Local Spark cores to use.")
    parser.add_argument("--from-beginning", action="store_true",
                        help="Process the whole topic history, not just new events.")
    args = parser.parse_args()

    import pyspark
    from pyspark.sql import functions as F
    from pyspark.sql import types as T

    kafka_pkg = f"org.apache.spark:spark-sql-kafka-0-10_2.13:{pyspark.__version__}"
    # 2 cores is plenty for the telemetry volume and leaves the CPU to PyTorch
    # inference in the API and the streaming indexer running on the same machine.
    spark = build_spark("ContextClosest-Telemetry-Stream", {"spark.jars.packages": kafka_pkg},
                        master=f"local[{args.cores}]", shuffle_partitions=args.cores)
    logger.info("Spark %s | UI: %s", spark.version, spark.sparkContext.uiWebUrl)

    schema = T.StructType([
        T.StructField("query", T.StringType()),
        T.StructField("count", T.IntegerType()),
        T.StructField("elapsed_ms", T.DoubleType()),
        T.StructField("cache_hit", T.BooleanType()),
        T.StructField("ts", T.DoubleType()),
    ])

    events = (
        spark.readStream.format("kafka")
        .option("kafka.bootstrap.servers", KAFKA_BOOTSTRAP)
        .option("subscribe", TOPIC)
        .option("startingOffsets", "earliest" if args.from_beginning else "latest")
        .option("failOnDataLoss", "false")
        .load()
        .select(F.from_json(F.col("value").cast("string"), schema).alias("e"))
        .select("e.*")
        .withColumn("event_time", F.col("ts").cast("timestamp"))
        .withColumn("query", F.lower(F.trim("query")))
    )

    # ── Windowed KPIs ──────────────────────────────────────────────────────────
    window_metrics = (
        events.withWatermark("event_time", "2 minutes")
        .groupBy(F.window("event_time", args.window))
        .agg(
            F.count("*").alias("searches"),
            F.approx_count_distinct("query").alias("unique_queries"),
            F.round(F.avg((F.col("count") == 0).cast("double")), 3).alias("zero_result_rate"),
            F.round(F.avg(F.col("cache_hit").cast("double")), 3).alias("cache_hit_rate"),
            F.round(F.avg("elapsed_ms"), 1).alias("avg_latency_ms"),
            F.round(F.percentile_approx("elapsed_ms", 0.95), 1).alias("p95_latency_ms"),
        )
        .select(
            F.col("window.start").alias("window_start"),
            F.col("window.end").alias("window_end"),
            "searches", "unique_queries", "zero_result_rate",
            "cache_hit_rate", "avg_latency_ms", "p95_latency_ms",
        )
    )

    metrics_path = str((OUT_DIR / "window_metrics").resolve().as_posix())

    def write_window_batch(batch_df, batch_id: int) -> None:
        if batch_df.isEmpty():
            return
        rows = batch_df.orderBy("window_start").withColumn("batch_id", F.lit(batch_id))
        rows.write.mode("append").parquet(metrics_path)
        print(f"\n=== Search KPIs per {args.window} window (micro-batch {batch_id}) ===")
        rows.drop("batch_id").show(truncate=False)

    q1 = (
        window_metrics.writeStream
        .outputMode("update")
        .foreachBatch(write_window_batch)
        .option("checkpointLocation", str((OUT_DIR / "_chk_windows").resolve().as_posix()))
        .trigger(processingTime=args.trigger)
        .start()
    )

    # ── Running top-queries leaderboard ────────────────────────────────────────
    top_queries = (
        events.groupBy("query")
        .agg(
            F.count("*").alias("searches"),
            F.round(F.avg("elapsed_ms"), 1).alias("avg_latency_ms"),
            F.max("count").alias("max_results"),
        )
        .orderBy(F.desc("searches"))
        .limit(10)
    )
    q2 = (
        top_queries.writeStream
        .outputMode("complete")
        .format("console")
        .option("truncate", "false")
        .option("checkpointLocation", str((OUT_DIR / "_chk_top").resolve().as_posix()))
        .trigger(processingTime=args.trigger)
        .start()
    )

    logger.info("Streaming from Kafka topic '%s'. Window metrics -> %s (Ctrl-C to stop)",
                TOPIC, metrics_path)
    try:
        spark.streams.awaitAnyTermination()
    except KeyboardInterrupt:
        logger.info("Stopping streams...")
        q1.stop()
        q2.stop()
        spark.stop()


if __name__ == "__main__":
    main()
