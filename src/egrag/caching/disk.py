"""Disk cache backend with atomic writes, corruption detection, and metrics.

Each entry is a JSON file holding the value plus a SHA-256 checksum. Writes are
atomic (write to a temp file in the same directory, then ``os.replace``), so a
concurrent or interrupted write can never leave a partial, valid-looking entry.
On read, a checksum mismatch or parse error is treated as corruption: the entry
is quarantined (renamed ``.corrupt``) and reported as a miss. Disabling the
cache performs no writes. Secrets must never be cached (keys/values are hashed
content only; see :mod:`egrag.caching.keys`).
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path


@dataclass
class CacheMetrics:
    """Counters describing cache effectiveness within one process."""

    hits: int = 0
    misses: int = 0
    writes: int = 0
    corruptions: int = 0


def _checksum(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


class DiskCacheBackend:
    """A content-addressed disk cache. Implements the ``CacheBackend`` protocol."""

    def __init__(self, directory: str | Path, *, enabled: bool = True) -> None:
        self._dir = Path(directory)
        self._enabled = enabled
        self.metrics = CacheMetrics()
        if self._enabled:
            self._dir.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        # Keys are already safe (namespace + hex); guard against separators anyway.
        safe = key.replace("/", "_").replace("\\", "_").replace("..", "_")
        return self._dir / f"{safe}.json"

    def get(self, key: str) -> str | None:
        if not self._enabled:
            self.metrics.misses += 1
            return None
        path = self._path(key)
        if not path.is_file():
            self.metrics.misses += 1
            return None
        try:
            raw = path.read_text(encoding="utf-8")
            entry = json.loads(raw)
            value = entry["value"]
            checksum = entry["checksum"]
        except (OSError, ValueError, KeyError, TypeError):
            self._quarantine(path)
            self.metrics.corruptions += 1
            self.metrics.misses += 1
            return None
        if not isinstance(value, str) or _checksum(value) != checksum:
            self._quarantine(path)
            self.metrics.corruptions += 1
            self.metrics.misses += 1
            return None
        self.metrics.hits += 1
        return value

    def set(self, key: str, value: str) -> None:
        if not self._enabled:
            return
        payload = json.dumps({"checksum": _checksum(value), "value": value})
        fd, tmp_name = tempfile.mkstemp(dir=self._dir, prefix=".tmp-", suffix=".json")
        tmp = Path(tmp_name)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp, self._path(key))  # atomic
        except OSError:
            tmp.unlink(missing_ok=True)
            raise
        self.metrics.writes += 1

    @staticmethod
    def _quarantine(path: Path) -> None:
        try:
            path.replace(path.with_suffix(".corrupt"))
        except OSError:  # pragma: no cover - best-effort quarantine
            path.unlink(missing_ok=True)


__all__ = ["CacheMetrics", "DiskCacheBackend"]
