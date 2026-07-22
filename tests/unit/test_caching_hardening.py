"""Caching acceptance tests (7-14): keys, corruption, atomicity, disabling."""

from __future__ import annotations

from pathlib import Path

import pytest

from egrag.caching import DiskCacheBackend, build_cache_key

BASE = {
    "namespace": "claims",
    "content": "the elephant is large",
    "algorithm": "sentence-baseline",
    "model": "m",
    "model_revision": "r1",
    "prompt_version": "extraction_v1",
    "config": {"chunk_size": 512},
}


@pytest.mark.unit
def test_key_changes_with_content() -> None:
    """Acceptance 7: cache keys change when input content changes."""

    assert build_cache_key(**BASE) != build_cache_key(**{**BASE, "content": "different text"})


@pytest.mark.unit
def test_key_changes_with_model_revision() -> None:
    """Acceptance 8: cache keys change when the model revision changes."""

    assert build_cache_key(**BASE) != build_cache_key(**{**BASE, "model_revision": "r2"})


@pytest.mark.unit
def test_key_changes_with_prompt_version() -> None:
    """Acceptance 9: cache keys change when the prompt version changes."""

    assert build_cache_key(**BASE) != build_cache_key(**{**BASE, "prompt_version": "v2"})


@pytest.mark.unit
def test_key_changes_with_config() -> None:
    """Acceptance 10: cache keys change when relevant configuration changes."""

    assert build_cache_key(**BASE) != build_cache_key(**{**BASE, "config": {"chunk_size": 256}})


@pytest.mark.unit
def test_key_stable_for_equivalent_inputs() -> None:
    """Acceptance 11: cache keys are stable for equivalent inputs (order-independent)."""

    reordered = {**BASE, "config": {"chunk_size": 512}}
    assert build_cache_key(**BASE) == build_cache_key(**reordered)
    assert build_cache_key(**BASE) == build_cache_key(**BASE)


@pytest.mark.unit
def test_round_trip_and_metrics(tmp_path: Path) -> None:
    cache = DiskCacheBackend(tmp_path)
    assert cache.get("k") is None
    cache.set("k", "value")
    assert cache.get("k") == "value"
    assert cache.metrics.writes == 1
    assert cache.metrics.hits == 1
    assert cache.metrics.misses == 1


@pytest.mark.unit
def test_corrupted_entry_rejected_and_quarantined(tmp_path: Path) -> None:
    """Acceptance 12: corrupted cache entries are rejected/quarantined safely."""

    cache = DiskCacheBackend(tmp_path)
    cache.set("k", "value")
    # Corrupt the stored entry on disk.
    entry = next(tmp_path.glob("*.json"))
    entry.write_text('{"checksum": "deadbeef", "value": "tampered"}', encoding="utf-8")
    assert cache.get("k") is None  # detected as corrupt -> miss
    assert cache.metrics.corruptions == 1
    assert list(tmp_path.glob("*.corrupt"))  # quarantined


@pytest.mark.unit
def test_partial_entry_is_not_valid_looking(tmp_path: Path) -> None:
    """Acceptance 13: a partial/interrupted write is never a valid-looking entry."""

    cache = DiskCacheBackend(tmp_path)
    cache.set("k", "value")
    # Atomic writes leave no temp files behind; exactly one final entry exists.
    assert not list(tmp_path.glob(".tmp-*"))
    assert len(list(tmp_path.glob("*.json"))) == 1
    # A truncated/partial JSON file is treated as corruption, not a hit.
    entry = next(tmp_path.glob("*.json"))
    entry.write_text('{"checksum": "x", "val', encoding="utf-8")
    assert cache.get("k") is None


@pytest.mark.unit
def test_disabled_cache_performs_no_write(tmp_path: Path) -> None:
    """Acceptance 14: disabled caching performs no cache write."""

    cache = DiskCacheBackend(tmp_path / "cache", enabled=False)
    cache.set("k", "value")
    assert cache.get("k") is None
    assert not (tmp_path / "cache").exists() or not list((tmp_path / "cache").glob("*.json"))
    assert cache.metrics.writes == 0
