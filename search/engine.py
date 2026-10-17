"""
search/engine.py

Provides the HybridSearchEngine class that orchestrates text query embedding
and hybrid vector search against the Qdrant database (Local or Cloud).

Big Data Integration — Redis:
  Queries are answered from a two-tier Redis cache before any expensive
  PyTorch inference or Qdrant RPC is attempted:
    1. Result Cache  — full serialised JSON result, 1-hour TTL.
    2. Embedding Cache — 768-dim dense + SPLADE sparse vectors, 24-hour TTL.
  Searches that return products are counted in a sliding one-hour window
  of Redis Sorted Sets for real-time trending queries (/api/trending).
  If Redis is unavailable the engine falls back transparently.
"""

import os
import logging
import time
from pathlib import Path
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Fusion,
    FusionQuery,
    Prefetch,
    SparseVector,
)

from search.cache import RedisCache

logger = logging.getLogger(__name__)

COLLECTION_NAME = "abo_products"

# Score thresholds: products must pass BOTH before entering RRF.
# Dense  — cosine similarity between the query text vector and the stored
#          image vector.  Range: [-1, 1].  0.20 filters out clearly
#          unrelated products while still allowing cross-modal soft matches.
# Sparse — dot-product score from SPLADE keyword overlap.  Range: [0, ∞).
#          5.0 requires at least a meaningful keyword match (e.g. a shared
#          content word with weight > ~2 each).  Lower values let noisy
#          partial matches through; higher values become too strict for
#          short or generic queries.
# Products added by the Kafka streaming indexer show a "NEW" badge for a week.
NEW_BADGE_SECONDS: int = 7 * 24 * 3600

DENSE_SCORE_THRESHOLD: float = 0.20
SPARSE_SCORE_THRESHOLD: float = 15.0


class HybridSearchEngine:
    """
    Orchestrates text-query embedding and hybrid (dense + sparse) search
    against a Qdrant collection using Reciprocal Rank Fusion (RRF).

    Score thresholds are applied independently at each prefetch stage so
    that only products that clear a minimum relevance bar enter the RRF
    fusion step.  If no products survive either threshold the search returns
    an empty result set rather than surfacing irrelevant items.
    """

    def __init__(self, config: Any) -> None:
        """
        Initializes the search engine by loading the inference models,
        connecting to the Qdrant database, and initialising the Redis cache.
        """
        from indexing.inference import InferenceEngine

        self._config = config

        logger.info("Loading InferenceEngine (SigLIP 2 + SPLADE)...")
        self._inference = InferenceEngine(
            base_model_name=config.BASE_MODEL_NAME,
            lora_adapter_path=str(config.lora_adapter_dir),
        )

        qdrant_url = os.getenv("QDRANT_URL")
        qdrant_api_key = os.getenv("QDRANT_API_KEY")

        if qdrant_url and qdrant_api_key:
            logger.info("Connecting to Qdrant Cloud at %s", qdrant_url)
            self._client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)
        else:
            logger.info("Connecting to local Qdrant at %s", config.qdrant_db_dir)
            self._client = QdrantClient(path=str(config.qdrant_db_dir))

        # Validate collection exists
        if not self._client.collection_exists(COLLECTION_NAME):
            raise RuntimeError(
                f"Qdrant collection '{COLLECTION_NAME}' not found."
            )

        info = self._client.get_collection(COLLECTION_NAME)
        logger.info(
            "Connected to collection '%s' with %d points.",
            COLLECTION_NAME,
            info.points_count,
        )

        # ── Redis Cache (graceful fallback if unavailable) ─────────────────────
        self._cache = RedisCache.from_env()

    def search(
        self,
        query: str,
        top_k: int = 10,
        dense_threshold: float = DENSE_SCORE_THRESHOLD,
        sparse_threshold: float = SPARSE_SCORE_THRESHOLD,
    ) -> dict:
        """
        Runs a hybrid text search against the Qdrant collection.

        Products must satisfy the dense OR sparse threshold to enter RRF
        fusion.  The final result list may contain fewer than top_k items
        (including zero) when no products meet either threshold.

        Args:
            query: Natural language search string.
            top_k: Maximum number of results to return.
            dense_threshold: Minimum cosine similarity (0-1) for the dense
                prefetch.  Products scoring below this are excluded from RRF.
            sparse_threshold: Minimum dot-product score for the sparse
                prefetch.  Products scoring below this are excluded from RRF.

        Returns:
            Dict with keys:
                results    - list of ranked product dicts (may be empty)
                elapsed_ms - wall-clock time for the full search
                count      - number of results returned
                no_results - True when the result list is empty
        """
        if not query or not query.strip():
            return {"results": [], "elapsed_ms": 0.0, "count": 0, "no_results": True}

        query = query.strip()
        t0 = time.perf_counter()

        # ── Redis Tier-1: Full Result Cache ────────────────────────────────────
        variant = f"k={top_k}|d={dense_threshold}|s={sparse_threshold}"
        cached = self._cache.get_result(query, variant)
        if cached is not None:
            # Report the latency of THIS request, not the original miss.
            elapsed_ms = (time.perf_counter() - t0) * 1000
            cached["cache_hit"] = True
            cached["elapsed_ms"] = round(elapsed_ms, 2)
            if cached.get("count"):
                self._cache.record_query(query)   # trending: only queries with results
            logger.info(
                "Cache HIT (result) for query '%s' — served in %.2f ms.", query, elapsed_ms
            )
            return cached

        # ── Redis Tier-2: Embedding Cache ──────────────────────────────────────
        cached_embeds = self._cache.get_embeddings(query)
        if cached_embeds:
            logger.debug("Embedding cache HIT for query: '%s'", query)
            dense_vector: list[float] = cached_embeds["dense"]
            sparse_raw: dict = cached_embeds["sparse"]
        else:
            # 1. Generate dense text embedding (SigLIP 2 text encoder)
            logger.debug("Embedding query (dense): '%s'", query)
            dense_vector = self._inference.embed_text_dense([query])[0]

            # 2. Generate sparse SPLADE embedding
            logger.debug("Embedding query (sparse): '%s'", query)
            sparse_raw = self._inference.embed_text_sparse([query])[0]

            # Store embeddings for future repeated queries.
            self._cache.set_embeddings(query, dense_vector, sparse_raw)

        sparse_vector = SparseVector(
            indices=sparse_raw["indices"],
            values=sparse_raw["values"],
        )

        # 3. Hybrid search with score thresholds on each prefetch leg.
        #    score_threshold on a Prefetch tells Qdrant to discard any
        #    candidate whose raw score (cosine / dot-product) falls below
        #    that value - they never participate in RRF at all.
        results = self._client.query_points(
            collection_name=COLLECTION_NAME,
            prefetch=[
                Prefetch(
                    query=dense_vector,
                    using="dense_image",
                    limit=top_k * 5,            # cast a wide net before threshold
                    score_threshold=dense_threshold,
                ),
                Prefetch(
                    query=sparse_vector,
                    using="sparse_text",
                    limit=top_k * 5,
                    score_threshold=sparse_threshold,
                ),
            ],
            query=FusionQuery(fusion=Fusion.RRF),
            limit=top_k,
            with_payload=True,
        )

        elapsed_ms = (time.perf_counter() - t0) * 1000

        # 4. Format results
        formatted = []
        for point in results.points:
            payload = point.payload or {}

            # Use public S3 URL if on cloud, otherwise fallback to local path
            if "image_url" in payload:
                image_url = payload["image_url"]
            else:
                image_path = payload.get("main_image_path", "")
                image_url = f"/images/{image_path}" if image_path else ""

            formatted.append(
                {
                    "score": round(point.score, 4),
                    "item_id": payload.get("item_id", ""),
                    "title": payload.get("title", ""),
                    "brand": payload.get("brand") or "",
                    "color": payload.get("color") or "",
                    "material": payload.get("material") or "",
                    "product_type": payload.get("product_type") or "",
                    "category": payload.get("category") or "",
                    "image_url": image_url,
                    # Set by the Kafka streaming indexer for live-ingested products.
                    "is_new": payload.get("source") == "kafka-stream"
                    and time.time() - float(payload.get("ingested_at") or 0) < NEW_BADGE_SECONDS,
                    "ingested_at": payload.get("ingested_at"),
                }
            )

        no_results = len(formatted) == 0

        if no_results:
            logger.info(
                "Query '%s' -> no results above thresholds "
                "(dense>=%.2f, sparse>=%.1f) in %.1f ms",
                query,
                dense_threshold,
                sparse_threshold,
                elapsed_ms,
            )
        else:
            logger.info(
                "Query '%s' -> %d results in %.1f ms", query, len(formatted), elapsed_ms
            )

        result = {
            "results": formatted,
            "elapsed_ms": round(elapsed_ms, 1),
            "count": len(formatted),
            "no_results": no_results,
            "cache_hit": False,
        }

        # Store the fresh result in the Redis result cache for future requests.
        self._cache.set_result(query, result, variant)
        if not no_results:
            self._cache.record_query(query)   # trending: only queries with results

        return result

    def cache_stats(self) -> dict:
        """Returns Redis cache health and memory statistics."""
        return self._cache.get_stats()

    def get_trending(self, n: int = 10) -> list[dict]:
        """Returns the top-n trending queries from the Redis sorted set."""
        return self._cache.get_trending(n)
