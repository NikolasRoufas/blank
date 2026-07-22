"""Persistent caching wrapper for a structured-extraction model.

Wraps any :class:`StructuredModel` and memoizes its raw output behind a
content-addressed key (see :func:`egrag.caching.build_cache_key`). The cached
value is the raw model string, so a cold run and a warm run are byte-identical.
The key includes the model id, model revision, prompt version, schema version,
and the decoding controls (seed, determinism, max tokens), so output is never
reused across a different model/revision/prompt/decoding. Secrets are never
cached (only hashed content + identifiers).
"""

from __future__ import annotations

from egrag.adapters.extraction.interfaces import StructuredModel
from egrag.caching.keys import build_cache_key
from egrag.domain.ports import CacheBackend


class CachedStructuredModel:
    """A :class:`StructuredModel` that memoizes raw completions in a cache backend."""

    def __init__(
        self,
        inner: StructuredModel,
        cache: CacheBackend,
        *,
        model: str,
        model_revision: str | None = None,
        prompt_version: str | None = None,
        max_new_tokens: int | None = None,
    ) -> None:
        self._inner = inner
        self._cache = cache
        self._model = model
        self._model_revision = model_revision
        self._prompt_version = prompt_version
        self._max_new_tokens = max_new_tokens

    def _key(self, prompt: str, seed: int, deterministic: bool) -> str:
        return build_cache_key(
            namespace="extract",
            content=prompt,
            algorithm="structured-generation",
            model=self._model,
            model_revision=self._model_revision,
            prompt_version=self._prompt_version,
            config={
                "seed": seed,
                "deterministic": deterministic,
                "max_new_tokens": self._max_new_tokens,
            },
        )

    def complete(self, prompt: str, *, seed: int = 0, deterministic: bool = True) -> str:
        key = self._key(prompt, seed, deterministic)
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        output = self._inner.complete(prompt, seed=seed, deterministic=deterministic)
        self._cache.set(key, output)
        return output


__all__ = ["CachedStructuredModel"]
