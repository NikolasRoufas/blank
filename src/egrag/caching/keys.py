"""Deterministic, content-addressed cache-key construction.

A cache key is a stable hash over every input that can change the cached value:
the content hash, the algorithm identifier, the model identifier and revision,
the prompt version, the schema version, and the relevant configuration values.
If any of these differ, the key differs — so stale cross-model, cross-revision,
cross-prompt, or cross-config reuse cannot happen. Equivalent inputs yield an
identical key. Secrets must never be passed in; only hash non-secret inputs.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

from egrag.domain.version import SCHEMA_VERSION


def content_hash(text: str) -> str:
    """Return a stable SHA-256 hex digest of text content."""

    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def build_cache_key(
    *,
    namespace: str,
    content: str,
    algorithm: str,
    model: str | None = None,
    model_revision: str | None = None,
    prompt_version: str | None = None,
    config: dict[str, Any] | None = None,
    schema_version: str = SCHEMA_VERSION,
) -> str:
    """Build a deterministic cache key from all value-affecting inputs."""

    payload = {
        "namespace": namespace,
        "content": content_hash(content),
        "algorithm": algorithm,
        "model": model or "",
        "model_revision": model_revision or "",
        "prompt_version": prompt_version or "",
        "schema_version": schema_version,
        # Sorted keys make the serialization order-independent and deterministic.
        "config": json.dumps(config or {}, sort_keys=True, separators=(",", ":")),
    }
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(serialized.encode("utf-8")).hexdigest()
    return f"{namespace}-{digest[:32]}"


def build_nli_cache_key(
    premise: str,
    hypothesis: str,
    *,
    model: str,
    model_revision: str | None = None,
    tokenizer_revision: str | None = None,
    label_mapping_version: str = "1",
    max_length: int | None = None,
    truncation: bool = True,
    entailment_threshold: float,
    contradiction_threshold: float,
    duplicate_threshold: float,
    schema_version: str = SCHEMA_VERSION,
) -> str:
    """Deterministic cache key for one directional NLI (premise -> hypothesis) call.

    The key includes every value-affecting input so that NLI results are never
    reused across models, revisions, tokenizers, label mappings, truncation
    settings, thresholds, or schema versions. The ``nli:`` namespace and the model
    fields guarantee these keys never collide with lexical-classifier cache keys.
    """

    payload = {
        "namespace": "nli",
        "premise": content_hash(premise),
        "hypothesis": content_hash(hypothesis),
        "model": model,
        "model_revision": model_revision or "",
        "tokenizer_revision": tokenizer_revision or "",
        "label_mapping_version": label_mapping_version,
        "max_length": max_length if max_length is not None else 0,
        "truncation": truncation,
        "entailment_threshold": entailment_threshold,
        "contradiction_threshold": contradiction_threshold,
        "duplicate_threshold": duplicate_threshold,
        "schema_version": schema_version,
    }
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(serialized.encode("utf-8")).hexdigest()
    return f"nli-{digest[:32]}"


__all__ = ["build_cache_key", "build_nli_cache_key", "content_hash"]
