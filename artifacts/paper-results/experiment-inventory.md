# Experiment Capability Inventory

Inventory of what is **actually implemented and runnable locally** (verified by
inspecting the source and by the smoke/main runs). Nothing below is claimed
unless it exists in code and is exercised by tests. "Determinism" = byte-identical
outputs across repeated runs / seeds.

## Generators (the binding limitation)

| component | identifier | execution | optional deps | deterministic | status |
|---|---|---|---|---|---|
| `FakeTextGenerator` | n/a (templated) | local | none | yes | **fully implemented; used for all runs** |
| `HuggingFaceGenerator` | user model id | local | `transformers`,`torch` | seed-based | implemented but **unavailable** (deps not installed; no network) |
| `OpenAICompatibleGenerator` | user endpoint | remote/local server | `httpx` | server-dependent | implemented but **unavailable** (no endpoint; `httpx` absent) |

The experiment harness `_build_generator` **only accepts `fake`** and raises for
others, so all experiments here use the deterministic fake generator. There is
**no Ollama/vLLM/llama.cpp adapter**; the only server path is the generic
OpenAI-compatible adapter (unused here). Consequence: answer-text quality metrics
(EM/F1) are not meaningful — the fake emits a fixed templated string.

## Datasets & splits

| adapter | examples | split | status |
|---|---|---|---|
| `SyntheticGraphDataset` (`synthetic_graph`) | 2 | `test` | fully implemented |
| `TemporalConflictDataset` (`temporal_conflict`) | 1 | `test` | fully implemented |
| `JsonlDataset` (HotpotQA-/FEVER-style) | n/a | from file | implemented adapter, but **no local data file exists** → real multi-hop / FEVER benchmarks **unavailable** (not downloaded; no network) |

No train/dev split data exists locally; all built-in fixtures are `test`-tagged
mechanism checks. There are **no hop-count annotations**, so by-hop breakdowns are
unavailable.

## Retrieval / reranking / extraction

| component | impl | model | local | deps | deterministic | status |
|---|---|---|---|---|---|---|
| Chunker | `SentenceAwareChunker` | n/a | yes | none | yes | implemented |
| Retriever | `BM25Retriever` | n/a (pure-Python BM25) | yes | none | yes | implemented (used) |
| Retriever | `DenseRetriever` | sentence-transformers | yes | `sentence-transformers` | model-dependent | **unavailable** (dep absent) |
| Retriever | `HybridRetriever` | — | yes | `sentence-transformers` | — | **unavailable** (dep absent) |
| Reranker | `ScoreReranker` | n/a | yes | none | yes | implemented (identity on BM25 order) |
| Reranker | `CrossEncoderReranker` | cross-encoder | yes | `sentence-transformers` | model-dependent | **unavailable** (dep absent) |
| Extractor | `SentenceClaimExtractor` | n/a | yes | none | yes | implemented (used) |
| Extractor | structured/HF | user model | yes | `transformers` | seed-based | **unavailable** (dep absent) |

## Graph / reasoning

| component | impl | deterministic | status |
|---|---|---|---|
| Relation classifier | `LexicalPairClassifier` | yes | implemented (used) |
| Relation classifier | `HuggingFaceNLIClassifier` | model-based | **unavailable** (`transformers` absent) |
| Source reliability | `UniformReliability`, `ConfiguredPriorReliability`, `MetadataReliability` | yes | implemented (`MetadataReliability` used) |
| Propagation | `SignedBeliefPropagator` | yes | implemented (used) |
| Propagation | `NoPropagationBaseline` | yes | implemented (ablation) |
| Conflict resolver | `ConflictSetResolver` | yes | implemented (used) |
| Subgraph selector | `TopClaimsSelector`, `GreedyConnectedSelector`, `BeamSearchSelector` | yes | top + greedy used; beam available |
| Duplicates / temporal | `detect_lexical_duplicates`, `SupersessionResolver` | yes | implemented (used) |
| Token counters | `CharacterTokenCounter`, `WhitespaceTokenCounter`, `HuggingFaceTokenCounter` | yes / yes / model | character counter used; HF counter unavailable |
| GraphML export | `egrag.adapters.graph.networkx_export` | — | **unavailable** (`networkx` absent) |

## Experiment variants (all 10 implemented; run on every dataset)

`passage_rag`, `reranked_passage_rag`, `claim_only_rag`, `graph_no_propagation`,
`graph_with_propagation`, `graph_top_claim`, `graph_coherent_subgraph`,
`graph_no_temporal`, `graph_no_contradiction`, `full_egrag`.

Name mapping to the requested set: `egrag_full`→`full_egrag`,
`graph_top_claims`→`graph_top_claim`, `egrag_no_propagation`→`graph_no_propagation`,
`egrag_no_temporal`→`graph_no_temporal`, `egrag_no_contradiction`→`graph_no_contradiction`.

**Unavailable requested ablations (not silently substituted):**
- `egrag_no_coherent_selector` — closest implemented behavior is `graph_top_claim`
  (independent top-claim selection); there is no separate "coherent selector off"
  flag distinct from choosing the top-claims selector. Reported via `graph_top_claim`.
- `egrag_no_provenance_discount` — **absent from code/config/tests**: the
  provenance/duplicate lineage discount is not exposed as a runtime toggle, so this
  ablation cannot be run without adding a new flag (out of scope; not done).

## Metrics (separated by kind; see logs/metric-kinds.json)

- **deterministic**: exact_match, normalized_exact_match, token_f1, answer_accuracy,
  citation_precision/recall/completeness, evidence_precision/recall,
  invalid_citations, empty_prediction, and counts (num_passages, num_claims,
  num_graph_nodes, num_graph_edges, num_selected, token_estimate), total latency_ms.
- **heuristic** (lexical proxies, NOT ground truth): answer_evidence_entailment,
  contradiction_rate, unsupported_claim_rate (and conflict/coherence proxies).
- **model_based**: interface only (`EntailmentMetric`); no model bundled → not used.

**Requested metrics that are unavailable (not recorded by the harness):**
per-stage latency breakdown (retrieval/rerank/extraction/candidate/classification/
graph/propagation/selection/generation), peak memory, possible-pair / candidate-pair /
classified-pair counts, propagation iterations, conflict-resolution accuracy,
temporal-resolution accuracy (no gold temporal annotations beyond the 1 fixture),
selected-subgraph connectivity/coherence numeric metrics, reasoning-hop coverage.
Only **total** latency and node/edge/selected counts are captured per example.

## Configuration & export

- Experiment configs are `ExperimentConfig` objects (frozen under
  `frozen-configs/`); production pipeline YAMLs live in `configs/` (checksummed).
- Export formats: per-example JSONL, aggregate JSON, manifest JSON, evidence-package
  JSON, slim graph JSON, CSV tables, Markdown tables. No remote export.
