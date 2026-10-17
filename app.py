"""
app.py

FastAPI web application for the Hybrid Multimodal Search Engine.

Big data integration (both optional — search works without them):
  - Redis: result cache, query-embedding cache and trending searches
    (search/cache.py, configured by REDIS_URL or REDIS_HOST/REDIS_PORT).
  - Kafka: every search publishes a telemetry event to 'search-telemetry'
    from a background thread, consumed by data_pipeline/spark_telemetry_stream.py.
    The broker is KAFKA_BOOTSTRAP (default localhost:9092); set it to an empty
    value to turn telemetry off, as the Modal deployment does.

Run locally:
    python app.py        # then open http://localhost:8000
"""

import asyncio
import json
import logging
import os
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path

import uvicorn
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

# Ensure the project root is on the Python path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from dotenv import load_dotenv

from config import Config
from search.engine import HybridSearchEngine

# Load QDRANT_URL / QDRANT_API_KEY etc. from .env so the local server searches
# the same Qdrant Cloud index that the Kafka streaming indexer writes to.
# (On Modal the variables come from secrets and there is no .env — no-op.)
load_dotenv(Path(__file__).resolve().parent / ".env")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Global state — models are loaded once at startup, not per request
# ---------------------------------------------------------------------------
_engine: HybridSearchEngine | None = None
_kafka_producer = None   # Optional Kafka producer — None when Kafka is offline


# ---------------------------------------------------------------------------
# Kafka Helper
# ---------------------------------------------------------------------------

def _init_kafka_producer():
    """
    Connects a fire-and-forget Kafka producer for search telemetry.
    Returns None when KAFKA_BOOTSTRAP is empty, kafka-python is missing or the
    broker is unreachable, so the rest of the application is unaffected.
    """
    bootstrap = os.getenv("KAFKA_BOOTSTRAP", "localhost:9092").strip()
    if not bootstrap:
        logger.info("KAFKA_BOOTSTRAP is empty — search telemetry disabled.")
        return None
    try:
        from kafka import KafkaProducer  # type: ignore
        producer = KafkaProducer(
            bootstrap_servers=[bootstrap],
            value_serializer=lambda v: json.dumps(v).encode("utf-8"),
            acks=0,                 # fire-and-forget: no ack wait, zero latency
            request_timeout_ms=500,
            max_block_ms=1000,      # never block a search request on metadata
        )
        logger.info("Kafka producer connected to %s", bootstrap)
        return producer
    except ImportError:
        logger.warning(
            "kafka-python not installed.  Install: pip install kafka-python\n"
            "Search telemetry streaming is disabled."
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "Kafka broker unavailable (%s).  Search telemetry disabled.\n"
            "Start Kafka with: docker compose -f docker-compose-bigdata.yml up -d",
            exc,
        )
    return None


def _publish_telemetry(query: str, count: int, elapsed_ms: float, cache_hit: bool) -> None:
    """
    Publishes a search telemetry event to the Kafka 'search-telemetry' topic.
    This is called fire-and-forget in a background thread; it never blocks
    the HTTP response path.

    Schema (JSON):
        query      - the search string
        count      - number of results returned
        elapsed_ms - server-side latency in milliseconds
        cache_hit  - True if the result came from the Redis cache
        ts         - Unix timestamp (float)
    """
    if _kafka_producer is None:
        return
    try:
        event = {
            "query": query,
            "count": count,
            "elapsed_ms": elapsed_ms,
            "cache_hit": cache_hit,
            "ts": time.time(),
        }
        _kafka_producer.send("search-telemetry", value=event)
    except Exception as exc:  # noqa: BLE001
        logger.debug("Kafka publish error: %s", exc)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI lifespan context manager.
    Loads the search engine (models + Qdrant connection) once at startup,
    initialises the optional Kafka producer, and cleans up on shutdown.
    """
    global _engine, _kafka_producer
    logger.info("=== Starting Hybrid Search Engine ===")
    config = Config(Path(__file__).resolve().parent)
    _engine = HybridSearchEngine(config)

    # Kafka telemetry producer (optional)
    _kafka_producer = _init_kafka_producer()

    logger.info("=== Search Engine Ready — visit http://localhost:8000 ===")
    yield
    logger.info("=== Shutting down ===")
    if _kafka_producer is not None:
        _kafka_producer.close(timeout=2)
    _engine = None
    _kafka_producer = None


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(
    title="ContextClosest API",
    description="Hybrid multimodal search (SigLIP 2 + SPLADE, RRF in Qdrant) over 56,427 Amazon products, with Redis caching and trending searches.",
    version="1.0.0",
    lifespan=lifespan,
)

# Serve product images from Processed_Data/images/
_project_root = Path(__file__).resolve().parent
_images_dir = _project_root / "Processed_Data" / "images"

# Mount local images only when present (local dev). In the cloud the frontend
# loads images directly from the public S3 URLs stored in the Qdrant payloads.
if _images_dir.exists():
    app.mount("/images", StaticFiles(directory=str(_images_dir)), name="images")
    logger.info("Mounted product images from %s", _images_dir)
else:
    logger.info("No local images dir — using S3 URLs from Qdrant Cloud.")

# Serve frontend static files (CSS, JS)
_static_dir = _project_root / "static"
app.mount("/static", StaticFiles(directory=str(_static_dir)), name="static")


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/", include_in_schema=False)
async def serve_frontend() -> FileResponse:
    """Serves the main frontend HTML page."""
    index_path = _static_dir / "index.html"
    if not index_path.exists():
        raise HTTPException(status_code=404, detail="Frontend not found.")
    return FileResponse(str(index_path))


@app.get("/api/search")
async def search(
    q: str = Query(..., min_length=1, description="Text search query"),
    top_k: int = Query(default=12, ge=1, le=50, description="Number of results"),
) -> JSONResponse:
    """
    Runs a hybrid text search (dense SigLIP 2 + sparse SPLADE) against the
    Qdrant collection and returns ranked product results.

    Args:
        q: The natural language query string.
        top_k: Number of results to return (1–50, default 12).

    Returns:
        JSON object with keys: results (list), count (int), elapsed_ms (float).
    """
    if _engine is None:
        raise HTTPException(status_code=503, detail="Search engine is not ready yet.")

    try:
        response = _engine.search(query=q, top_k=top_k)

        # Publish telemetry to Kafka asynchronously (fire-and-forget).
        # asyncio.get_event_loop().run_in_executor ensures the kafka send
        # never blocks the HTTP response path.
        loop = asyncio.get_event_loop()
        loop.run_in_executor(
            None,
            _publish_telemetry,
            q,
            response.get("count", 0),
            response.get("elapsed_ms", 0.0),
            response.get("cache_hit", False),
        )

        return JSONResponse(content=response)
    except Exception as exc:
        logger.exception("Search error for query '%s': %s", q, exc)
        raise HTTPException(status_code=500, detail=f"Search failed: {str(exc)}")


@app.get("/api/health")
async def health() -> JSONResponse:
    """Health check endpoint — also reports Redis and Kafka status."""
    cache_stats = _engine.cache_stats() if _engine else {"redis_available": False}
    return JSONResponse(
        content={
            "status": "ok" if _engine is not None else "loading",
            "collection": "abo_products",
            "kafka_producer": _kafka_producer is not None,
            **cache_stats,
        }
    )


@app.get("/api/trending")
async def trending(
    n: int = Query(default=5, ge=1, le=50, description="Number of trending queries"),
) -> JSONResponse:
    """
    Returns the top-n search queries of the last hour (only searches that
    returned products), from a sliding window of Redis Sorted Sets.

    Args:
        n: Number of trending queries to return (1–50, default 5).

    Returns:
        JSON object with key ``trending`` containing a list of
        {query: str, count: int} dicts ordered by descending count.
    """
    if _engine is None:
        raise HTTPException(status_code=503, detail="Search engine is not ready.")
    return JSONResponse(content={"trending": _engine.get_trending(n)})


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=False,  # Reload disabled — models take time to load
        log_level="info",
    )
