"""Caching backends and deterministic cache keys behind the ``CacheBackend`` port."""

from __future__ import annotations

from egrag.caching.disk import CacheMetrics, DiskCacheBackend
from egrag.caching.keys import build_cache_key, build_nli_cache_key, content_hash
from egrag.caching.memory import InMemoryCacheBackend, NullCacheBackend

__all__ = [
    "CacheMetrics",
    "DiskCacheBackend",
    "InMemoryCacheBackend",
    "NullCacheBackend",
    "build_cache_key",
    "build_nli_cache_key",
    "content_hash",
]
