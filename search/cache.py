"""
search/cache.py

Redis serving layer for the hybrid search engine.

  1. Result cache    — the full JSON response per (query, top_k, thresholds),
                       1-hour TTL.  Cleared by the Kafka streaming indexer
                       whenever new products are indexed.
  2. Embedding cache — the query's dense (SigLIP 2) and sparse (SPLADE)
                       vectors, 24-hour TTL.  Query embeddings do not depend on
                       the catalog, so they survive catalog updates.
  3. Trending        — a sliding one-hour window of 5-minute sorted-set buckets;
                       only searches that returned products are counted.

Connection (see RedisCache.from_env): REDIS_URL, e.g. rediss://... for a hosted
Redis such as Upstash, or REDIS_HOST / REDIS_PORT / REDIS_PASSWORD, defaulting
to a local Redis on 127.0.0.1:6379.

If Redis is unavailable every call is a no-op: search keeps working without
caching or trending.
"""

import hashlib
import json
import logging
import os
import time
from typing import Optional

logger = logging.getLogger(__name__)

# ── Cache TTL settings ─────────────────────────────────────────────────────────
RESULT_CACHE_TTL_SECONDS: int = 3600   # 1 hour  — full search result cache
EMBED_CACHE_TTL_SECONDS: int  = 86400  # 24 hours — dense+sparse embedding cache
TRENDING_PREFIX: str = "contextclosest:trending:"
TRENDING_WINDOW_SECONDS: int = 3600     # trending = searches in the last hour
TRENDING_BUCKET_SECONDS: int = 300      # 5-minute buckets
MAX_TRENDING_DISPLAY: int = 5           # top-N queries shown in the UI


RESULT_KEY_PREFIX: str = "contextclosest:result:"


def _query_key(query: str, variant: str = "") -> str:
    """
    Returns a stable cache key for a search result.

    ``variant`` encodes the search parameters (top_k, thresholds) so that the
    same query text with different parameters never shares a cached result.
    """
    raw = f"{query.lower().strip()}|{variant}"
    digest = hashlib.sha256(raw.encode()).hexdigest()[:16]
    return f"{RESULT_KEY_PREFIX}{digest}"


def _embed_key(query: str) -> str:
    """Returns a stable cache key for a query embedding."""
    digest = hashlib.sha256(query.lower().strip().encode()).hexdigest()[:16]
    return f"contextclosest:embed:{digest}"


class RedisCache:
    """
    Thin wrapper around redis-py: result cache, embedding cache and the
    trending-searches leaderboard.

    All public methods are safe to call even when Redis is unavailable —
    they return None / empty values and log a debug-level warning instead of
    raising exceptions.
    """

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 6379,
        db: int = 0,
        password: Optional[str] = None,
        url: Optional[str] = None,
    ) -> None:
        """
        Connects to Redis; ``url`` (redis:// or rediss://) wins over host/port.
        """
        self._client = None
        self._available = False
        where = url.split("@")[-1] if url else f"{host}:{port}"   # never log a password

        try:
            import redis  # optional dependency — graceful fallback if absent

            options = dict(socket_connect_timeout=2, socket_timeout=2, decode_responses=False)
            if url:
                self._client = redis.Redis.from_url(url, **options)
            else:
                self._client = redis.Redis(host=host, port=port, db=db, password=password, **options)
            self._client.ping()
            self._available = True
            logger.info("Redis cache connected at %s — caching and trending enabled.", where)
        except ImportError:
            logger.warning("redis-py not installed (pip install redis); caching and trending disabled.")
        except Exception as exc:  # noqa: BLE001
            logger.warning("Redis unavailable at %s (%s); search works without caching.", where, exc)

    @classmethod
    def from_env(cls) -> "RedisCache":
        """Builds a cache from REDIS_URL, or REDIS_HOST / REDIS_PORT / REDIS_PASSWORD."""
        return cls(
            host=os.getenv("REDIS_HOST", "127.0.0.1"),
            port=int(os.getenv("REDIS_PORT", "6379")),
            password=os.getenv("REDIS_PASSWORD") or None,
            url=os.getenv("REDIS_URL") or None,
        )

    # ── Property ───────────────────────────────────────────────────────────────

    @property
    def is_available(self) -> bool:
        """True when Redis is connected and responsive."""
        return self._available

    # ── Result Cache ───────────────────────────────────────────────────────────

    def get_result(self, query: str, variant: str = "") -> Optional[dict]:
        """
        Returns the cached search result dict for *query*, or None on a miss.

        Args:
            query:   The raw search query string.
            variant: Search-parameter fingerprint (see ``_query_key``).

        Returns:
            Deserialised result dict or None.
        """
        if not self._available:
            return None
        try:
            raw = self._client.get(_query_key(query, variant))
            if raw:
                logger.debug("Cache HIT  (result)  for query: '%s'", query)
                return json.loads(raw.decode("utf-8"))
            logger.debug("Cache MISS (result)  for query: '%s'", query)
            return None
        except Exception as exc:  # noqa: BLE001
            logger.debug("Redis get_result error: %s", exc)
            return None

    def set_result(self, query: str, result: dict, variant: str = "") -> None:
        """
        Stores *result* in the cache for *query* with a 1-hour TTL.

        Args:
            query:   The raw search query string.
            variant: Search-parameter fingerprint (see ``_query_key``).
            result: The full search result dict returned by HybridSearchEngine.
        """
        if not self._available:
            return
        try:
            payload = json.dumps(result, ensure_ascii=False).encode("utf-8")
            self._client.setex(_query_key(query, variant), RESULT_CACHE_TTL_SECONDS, payload)
            logger.debug("Cache SET  (result)  for query: '%s'", query)
        except Exception as exc:  # noqa: BLE001
            logger.debug("Redis set_result error: %s", exc)

    def invalidate_results(self) -> int:
        """
        Deletes every cached search result.

        Called by the Kafka streaming indexer after it upserts new products:
        cached result lists no longer reflect the catalog, so they must go.
        Query embeddings stay valid (they do not depend on the catalog) and
        are deliberately kept.

        Returns:
            Number of result keys deleted.
        """
        if not self._available:
            return 0
        try:
            deleted = 0
            batch = []
            for key in self._client.scan_iter(match=f"{RESULT_KEY_PREFIX}*", count=500):
                batch.append(key)
                if len(batch) >= 500:
                    deleted += self._client.delete(*batch)
                    batch.clear()
            if batch:
                deleted += self._client.delete(*batch)
            return deleted
        except Exception as exc:  # noqa: BLE001
            logger.debug("Redis invalidate_results error: %s", exc)
            return 0

    # ── Embedding Cache ────────────────────────────────────────────────────────

    def get_embeddings(self, query: str) -> Optional[dict]:
        """
        Returns cached dense + sparse embeddings for *query*, or None.

        Returns:
            Dict with keys ``dense`` (list[float]) and ``sparse`` (dict with
            ``indices`` and ``values`` lists), or None on a miss.
        """
        if not self._available:
            return None
        try:
            raw = self._client.get(_embed_key(query))
            if raw:
                logger.debug("Cache HIT  (embed)   for query: '%s'", query)
                return json.loads(raw.decode("utf-8"))
            logger.debug("Cache MISS (embed)   for query: '%s'", query)
            return None
        except Exception as exc:  # noqa: BLE001
            logger.debug("Redis get_embeddings error: %s", exc)
            return None

    def set_embeddings(self, query: str, dense: list, sparse: dict) -> None:
        """
        Stores dense and sparse embeddings for *query* with a 24-hour TTL.

        Args:
            query:  Raw search query string.
            dense:  768-dim dense vector (list of floats).
            sparse: SPLADE sparse dict with ``indices`` and ``values`` keys.
        """
        if not self._available:
            return
        try:
            payload = json.dumps(
                {"dense": dense, "sparse": sparse}, ensure_ascii=False
            ).encode("utf-8")
            self._client.setex(_embed_key(query), EMBED_CACHE_TTL_SECONDS, payload)
            logger.debug("Cache SET  (embed)   for query: '%s'", query)
        except Exception as exc:  # noqa: BLE001
            logger.debug("Redis set_embeddings error: %s", exc)

    # ── Trending Queries (sliding window of Redis Sorted Sets) ─────────────────
    #
    # Each 5-minute bucket is its own sorted set (ZINCRBY query -> count) that
    # expires automatically once it leaves the window.  "Trending" is the
    # ZUNIONSTORE of the buckets inside the window, so old searches fall off on
    # their own and the leaderboard always reflects what people search NOW.

    @staticmethod
    def _bucket_key(bucket: int) -> str:
        return f"{TRENDING_PREFIX}{bucket}"

    @staticmethod
    def _current_bucket() -> int:
        return int(time.time() // TRENDING_BUCKET_SECONDS)

    def record_query(self, query: str) -> None:
        """
        Counts *query* in the current time bucket (atomic ZINCRBY).

        Only call this for searches that returned results — queries with no
        matching products should never appear as "trending".
        """
        if not self._available:
            return
        try:
            key = self._bucket_key(self._current_bucket())
            pipe = self._client.pipeline()
            pipe.zincrby(key, 1, query.lower().strip())
            pipe.expire(key, TRENDING_WINDOW_SECONDS + TRENDING_BUCKET_SECONDS)
            pipe.execute()
        except Exception as exc:  # noqa: BLE001
            logger.debug("Redis record_query error: %s", exc)

    def get_trending(self, n: int = MAX_TRENDING_DISPLAY) -> list[dict]:
        """
        Returns the top-*n* queries searched within the sliding window.

        Returns:
            List of dicts with keys ``query`` (str) and ``count`` (int),
            sorted by count descending.
        """
        if not self._available:
            return []
        try:
            now = self._current_bucket()
            buckets = TRENDING_WINDOW_SECONDS // TRENDING_BUCKET_SECONDS
            keys = [self._bucket_key(b) for b in range(now - buckets + 1, now + 1)]
            tmp = f"{TRENDING_PREFIX}window"
            pipe = self._client.pipeline()
            pipe.zunionstore(tmp, keys)
            pipe.zrevrange(tmp, 0, n - 1, withscores=True)
            pipe.delete(tmp)
            items = pipe.execute()[1]
            return [{"query": q.decode("utf-8"), "count": int(c)} for q, c in items]
        except Exception as exc:  # noqa: BLE001
            logger.debug("Redis get_trending error: %s", exc)
            return []

    # ── Cache Stats ────────────────────────────────────────────────────────────

    def get_stats(self) -> dict:
        """
        Returns basic Redis memory and key-count statistics.

        Returns:
            Dict suitable for embedding in a health-check API response.
        """
        if not self._available:
            return {"redis_available": False}
        try:
            info = self._client.info("memory")
            db_size = self._client.dbsize()
            return {
                "redis_available": True,
                "redis_used_memory_human": info.get("used_memory_human", "N/A"),
                "redis_total_keys": db_size,
            }
        except Exception as exc:  # noqa: BLE001
            return {"redis_available": False, "error": str(exc)}
