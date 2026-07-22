# Component injection (§5)

`experiments/variants.py` now defines a frozen `RunComponents` and
`run_system(…, components=None)`. Injectable: `extractor`, `classifier`,
`reranker`, `token_counter`, `cache`, `retriever_factory`, `chunker`. Each field
defaults to `None` = the deterministic default (BM25, sentence chunker, sentence
extractor, lexical classifier, character token counter, no reranker/cache).

Design choices:
- A single frozen dataclass (not many loose params) — the same instances are
  passed to passage/claim/graph runners so a comparison stays fair.
- Existing callers are unchanged (`components` is optional; the deterministic
  runner and CLI still pass nothing and behave identically).
- The real `StructuredClaimExtractor` and `HuggingFaceNLIClassifier` satisfy the
  injected `EvidenceExtractor` / `PairClassifier` protocols, so the final matrix
  injects them here. Cache wrappers (`CachedStructuredModel`,
  `CachedPairClassifier`, `CachedTextGenerator`) compose transparently.
- Gold data is never a component and never enters `run_system` (unchanged).

Tests (`tests/integration/test_component_injection.py`, 4):
- existing no-`components` call still works;
- an injected extractor is actually used;
- the same component instances accumulate across claim + graph variants (shared);
- default classifier path still builds a graph when nothing is injected.

Verified end-to-end in the e2e smoke: `RunComponents(extractor=real_structured,
classifier=real_nli)` + real generator ran the full `full_egrag` pipeline over 3
examples with the real models (`real-e2e-smoke.json`).
