"""In-Memory sematic cache (dev backend).

Stores entries in a list + a parallel numpy matrix of embeddings.
O(n) cosine search; fine for ~10k entries. Swap to VaalkeyCache for prod.

Embeddings from BAAI/bge-small-en-v1.5 are normalized, so cosine == dot.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np

from src.cache.base import BaseCache, CacheEntry
from src.cache.kb_version import read_kb_version
from src.components.embeddings import get_embeddings
from src.config import get_settings
from src.logger import get_logger

log = get_logger(__name__)

class InMemoryCache(BaseCache):
    def __init__(self) -> None:
        settings = get_settings()
        self._threshold: float = settings.cache_threshold
        self._ttl: timedelta = timedelta(days=settings.cache_ttl_days)
        self._embedder = get_embeddings()

        self._entries: list[CacheEntry] = []
        self._matrix: np.ndarray | None = None       # Shape (n_entries, embed_dim)

        self._hits = 0
        self._misses = 0 


    # ── public API ────────────────────────────────────────────────────────────
    def get(self, query: str) -> CacheEntry | None:
        if not self._entries:
            self._misses += 1
            return None

        kb_now = read_kb_version()
        now = datetime.now(timezone.utc)

        # Mask: True where entry is still valid (kb match + within TTL)
        valid_mask = np.array(
            [
                e.kb_version == kb_now and (now-e.created_at) < self._ttl
                for e in self._entries
            ]
        )
        if not valid_mask.any():
            self._misses += 1
            return None

        # Embed query and cosine-score against ALL entries, then mask invalid
        q_vec = np.array(self._embedder.embed_query(query))
        scores = self._matrix @ q_vec
        scores = np.where(valid_mask, scores, -np.inf)

        best_idx = int(np.argmax(scores))
        best_score = float(scores[best_idx])

        if best_score >= self._threshold:
            entry = self._entries[best_idx]
            entry.hit_count += 1
            self._hits += 1
            log.info("cache HIT score=%.3f | query=%r", best_score, query)
            return entry

        self._misses += 1
        log.info("cache MISS top_score=%.3f | query=%r", best_score, query)
        return None

    def set(
        self,
        query: str,
        answer: str,
        sources: list[dict] | None = None,
    ) -> None:
        embedding = self._embedder.embed_query(query)
        entry = CacheEntry(
            query=query,
            query_embedding=embedding,
            answer=answer,
            sources=sources,
            kb_version=read_kb_version()
        )
        self._entries.append(entry)

        vec = np.array(embedding).reshape(1, -1)
        self._matrix = vec if self._matrix is None else np.vstack([self._matrix, vec])
        log.info("cache SET  size=%d  query=%r", len(self._entries), query)

    def clear(self) -> None:
        self._entries.clear()
        self._matrix = None
        self._hits = 0
        self._misses = 0
        log.info("cache Cleared")

    def stats(self) -> dict:
        total = self._hits + self._misses
        return {
            "size": len(self._entries),
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": (self._hits/total) if total else 0.0,
        }    

