# Caching

Caching is transparent: a cold cache and a warm cache produce the same result. It
exists to avoid recomputing extraction, NLI, and generation when the inputs are
unchanged. Backends live in `egrag.caching`; key construction is in
`egrag.caching.keys`.

## Backends

- `InMemoryCacheBackend` — a per-instance dict; no persistence.
- `NullCacheBackend` — stores nothing; useful to disable caching.
- `DiskCacheBackend` — one JSON file per entry holding the value and a SHA-256
  checksum. Writes go to a temporary file in the same directory and are moved into
  place with `os.replace`, so an interrupted write cannot leave a partial entry. On
  read, a checksum mismatch or parse error is treated as corruption: the file is
  renamed to `.corrupt` and reported as a miss. When disabled, it performs no
  writes. It counts hits, misses, writes, and corruptions (`CacheMetrics`).

## Keys

`build_cache_key` and `build_nli_cache_key` hash only non-secret inputs and include
every value that can change the output, so stale reuse across a different setup is
impossible:

- content (hashed);
- algorithm identifier;
- model name and model revision (and tokenizer revision for NLI);
- prompt version;
- schema version;
- decoding settings (seed, temperature, top-p, max tokens, stop);
- truncation and max length;
- NLI thresholds and label-mapping version.

## Adapter wrappers

Three wrappers add caching without changing behavior:

- `egrag.adapters.extraction.caching.CachedStructuredModel`
- `egrag.graph.caching.CachedPairClassifier`
- `egrag.generation.CachedTextGenerator`

They store raw strings (extraction, generation) or an explicit versioned JSON of
the three NLI probabilities. Nothing is pickled.

## Invalidation

A key changes — and the entry misses — when the model revision, prompt version, or
NLI thresholds change; changing a threshold recomputes NLI. This is covered by
`tests/unit/test_adapter_caching.py` (cold/warm equality, hit on repeat, miss on
revision/prompt/threshold change, corruption quarantine, disabled-cache no-write).

## Locations and cleanup

The disk cache directory is chosen by the caller (the experiment `matrix` command
defaults to `.egrag-cache`, which is git-ignored). To clear it, delete the
directory; entries are content-addressed, so nothing else depends on them.
