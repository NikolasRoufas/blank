# Final experiment matrix — PLAN ONLY (not executed)

Per the milestone, the final matrix is **not run** here. This is the plan to run
on the GPU PC after the prerequisites below.

## Prerequisites (blockers, must clear first)

1. **Usable structured generator** — fix `HuggingFaceGenerator` /
   `HuggingFaceStructuredModel` to apply the chat template + stop/JSON constraint,
   and/or use a larger instruct model. Re-run §4–6 smokes; require valid-output
   rate ≈ 1.0. (Today: 0% with Qwen2.5-0.5B + no chat template.)
2. **Wire real adapters + cache into the runner** — `runner._build_generator`
   currently only allows `"fake"`; `variants.py` hardcodes `SentenceClaimExtractor`
   and `LexicalPairClassifier`. To benchmark real EG-RAG, parametrize the runner to
   accept the real generator/extractor/NLI and wire `DiskCacheBackend`
   (keys already complete). This is a scoped interface change with its own tests —
   intentionally not done in this calibration milestone.
3. **HotpotQA distractor split** (recommended) — fullwiki gold coverage is only
   28.2%; download the distractor config for evidence-grounded metrics.

## Variants (8 required) + hypothesis

| Variant | Enabled | Disabled | Hypothesis tested |
|---------|---------|----------|-------------------|
| passage_rag | retrieval only | extraction, graph | baseline; passages as evidence |
| reranked_passage_rag | + rerank | graph | does reranking help passage RAG |
| claim_only_rag | extraction | graph, propagation | claims vs passages |
| graph_no_propagation | graph | belief propagation | does propagation help |
| graph_no_bridge | graph | bridge edges | do bridges aid multi-hop |
| graph_no_contradiction | graph | contradiction edges | do contradiction edges matter |
| graph_top_claim | graph, top selection | connected selection | selection strategy |
| full_egrag | all | — | full system |

(`graph_no_bridge` to be added to `VARIANTS`; registry currently has
`graph_with_propagation`/`graph_coherent_subgraph`/`graph_no_temporal` instead —
align names before the run. Include `graph_no_temporal` only if temporal evidence
is meaningful; HotpotQA has little, so expect ~no effect.)

## Shared settings (frozen)

From `frozen-configs/{fever,hotpotqa}.yaml`: top_k 5, budget 256, NLI
roberta-large-mnli @2a8f12d2 E0.4/C0.7/D0.8, structural contradiction gate,
deterministic seed 0, one frozen generator across all variants (fairness).

## Projected cost (GPU, with caching; precompute extraction+NLI once)

- HotpotQA dev-100 + FEVER dev-100, 8 variants, seed 0.
- Extraction + NLI computed once per example/pair and cached → variants cheap.
- On CUDA this is order minutes–low hours total (vs many hours/infeasible on CPU,
  see `timing/report.md`). Exact numbers to be measured on the GPU host.

## Planned output artifacts (per benchmark, per variant)

Per-example JSONL (answer, citations, evidence ids, metrics, latency), aggregate
metrics with failure counts, paired bootstrap comparisons (no significance claim),
`ExperimentManifest` (version, config hash, seeds, adapter identities, dataset
fingerprint, env), fairness audit. Preserve failed examples and malformed outputs.

## Metrics

- HotpotQA: EM, token-F1, supporting-fact P/R/F1, joint answer/support, citation
  P/R, required-hop coverage, required-claim recall, connected-subgraph rate,
  invalid-citation rate, latency.
- FEVER: label accuracy, evidence P/R/F1, complete evidence-set recovery,
  FEVER-score (where valid), citation P/R, abstention correctness,
  unsupported-output rate, latency.

Do not select a winner by answer EM alone; weigh evidence + citations + faithfulness.
