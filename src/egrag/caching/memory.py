"""In-process cache backends.

Caching is transparent and must never change results: a cold cache and a warm
cache must produce identical output. These backends hold no global state; each
instance owns its own store.
"""

from __future__ import annotations

import hashlib


class InMemoryCacheBackend:
    """A simple in-memory cache. Implements the ``CacheBackend`` protocol."""

    def __init__(self) -> None:
        self._store: dict[str, str] = {}

    def get(self, key: str) -> str | None:
        return self._store.get(key)

    def set(self, key: str, value: str) -> None:
        self._store[key] = value


class NullCacheBackend:
    """A cache that stores nothing. Implements the ``CacheBackend`` protocol."""

    def get(self, key: str) -> str | None:
        return None

    def set(self, key: str, value: str) -> None:
        return None


def content_key(*parts: str) -> str:
    """Build a stable, content-addressed cache key from string parts."""

    digest = hashlib.sha256()
    for part in parts:
        digest.update(part.encode("utf-8"))
        digest.update(b"\x00")
    return digest.hexdigest()


__all__ = ["InMemoryCacheBackend", "NullCacheBackend", "content_key"]
