"""Cache contract — every backend (memory, Valkey, ...) implements BaseCache.

Callers depend on BaseCache, never on a concrete class. That's what makes
backends swappable without touching business logic.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from datetime import datetime, timezone

from pydantic import BaseModel, Field


class CacheEntry(BaseModel):
    """One cached (query, answer) pair plus metadata for invalidation."""
    query: str
    query_embedding: list[float]
    answer: str
    sources: list[dict] | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    kb_version: str
    hit_count: int = 0


class BaseCache(ABC):
    """Abstract semantic cache. Concrete backends implement the four methods."""

    @abstractmethod
    def get(self, query: str) -> CacheEntry | None:
        """Return cached entry if a semantically similar query exists, else None."""


    @abstractmethod
    def set(
        self,
        query: str,
        answer: str,
        sources: list[dict] | None = None,
    ) -> None:
        """Embed the query and store the (query, answer) entry."""

    @abstractmethod
    def clear(self) -> None:
        """Drop all entries."""

    @abstractmethod
    def stats(self) -> dict:
        """Return {size, hits, misses, hit_rate} for observability."""
        

    