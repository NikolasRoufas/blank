# Persistent caching (§6)

Uses the existing `DiskCacheBackend` (atomic writes, SHA-256 checksum, corruption
quarantine, disable = no writes) and key builders. Three wrappers memoize raw
outputs (strings) / versioned JSON so cold and warm runs are identical.

## Key coverage (value-affecting inputs)

| Wrapper | Namespace | Key includes |
|---------|-----------|--------------|
| `CachedStructuredModel` | `extract` | content, model, model_revision, prompt_version, schema_version, seed, deterministic, max_new_tokens |
| `CachedPairClassifier` | `nli` | premise+hypothesis hashes, model, model_revision, tokenizer_revision, label_mapping_version, max_length, truncation, entailment/contradiction/duplicate thresholds, schema_version |
| `CachedTextGenerator` | `generation`/`generation-chat` | content (prompt or serialized messages), model, model_revision, prompt_version, schema_version, max_new_tokens, temperature, top_p, seed, deterministic, stop |

Secrets are never cached (only hashed content + identifiers). NLI values are an
explicit versioned JSON of the three probabilities (no pickling of typed objects).

## Tests (`tests/unit/test_adapter_caching.py`, 9)

- structured / NLI / generation: cold==warm and warm is a hit (inner called once);
- miss on model-revision change, prompt-version change, and NLI threshold change;
- corrupt entry → quarantined → miss → recompute (`metrics.corruptions ≥ 1`);
- disabled cache and `NullCacheBackend` → no writes, inner always called;
- chat generation cached separately.

## Live cache metrics (e2e smoke, `real-e2e-smoke.json`)

3 examples, full real pipeline, run twice:
- cold: `hits=0, misses=9, writes=9`
- warm: `hits=9, misses=9, writes=9` (9 new hits, 0 new writes)
- **cold/warm answers identical: true** (100% equality)

So extraction + NLI + generation are all served from cache on the warm run, and
transparency (cold == warm) holds end-to-end.
