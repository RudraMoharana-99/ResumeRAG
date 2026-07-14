"""Cache factory - callers use get_cache(), never construct backends directly.

Swapping to Valkey later: set Settings.cache_backend = "valkey", add the
backend branch here. Zero changes to callers.
"""

from __future__ import annotations

from functools import lru_cache

from src.cache.base import BaseCache
from src.cache.memory import InMemoryCache
from src.config import get_settings
from src.logger import get_logger


log = get_logger(__name__)

@lru_cache
def get_cache() -> BaseCache:
    settings = get_settings()
    backend = settings.cache_backend.lower()

    if backend == "memory":
        log.info("Initializing InMemoryCache (threshold=%.2f, ttl_days=%d)",
                 settings.cache_threshold, settings.cache_ttl_days)

        return InMemoryCache()

    raise ValueError(f"Unknown cache backend: {backend!r}")