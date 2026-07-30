# Experiments and evaluation

The evaluation harness (`egrag.experiments`) is separate from the production
pipeline. It reuses production components to run controlled system variants over
datasets, scores predictions, compares variants, and writes reproducible
artifacts. The default path runs offline with deterministic fakes.

## Guarding against test-set tuning

- Configuration is a fixed input; the harness never reads metrics to choose it.
  `tuning_indicators()` flags a config whose name suggests test tuning.
- Datasets carry a `split`. `check_dataset_integrity` reports a question that
  appears in more than one split, so train/test overlap is visible.
- Gold answers and gold evidence are never passed to a variant. `run_system` takes
  only the question and documents; `assert_no_gold_parameters` enforces this.

## Datasets

Built-in synthetic fixtures need no download: `synthetic_graph` (small multi-claim
examples) and `temporal_conflict` (conflicting evidence with a supersession
signal). For HotpotQA-/FEVER-style data, convert a local corpus to JSONL where each
line has an `id`, `question` or `claim`, `documents`, `answers` or `label`, and
optional `gold_source_ids`, then load it with `--dataset ... --dataset-path`. The
real benchmark adapters (`FeverGoldEvidenceDataset`, `HotpotQADataset`) are
described in `docs/benchmarks.md`.

## System variants

`egrag.experiments.variants.VARIANTS` defines ten variants. They share the same
retriever, chunker, extractor, classifier, and generator, so a comparison isolates
one component.

| Variant | Family | Isolates |
|---------|--------|----------|
| `passage_rag` | passage | passages as evidence |
| `reranked_passage_rag` | passage | reranking (score reranker is identity on BM25 order) |
| `claim_only_rag` | claim | claim extraction, no graph |
| `graph_no_propagation` | graph | propagation off |
| `graph_with_propagation` | graph | propagation on |
| `graph_top_claim` | graph | top-claim selection |
| `graph_coherent_subgraph` | graph | greedy connected selection |
| `graph_no_temporal` | graph | temporal/supersession edges off |
| `graph_no_contradiction` | graph | contradiction edges off |
| `full_egrag` | graph | all on |

The `BRIDGES` relation is query-conditioned connectivity and never affects belief,
conflicts, or corroboration (`docs/bridge-relations.md`). There is no separate
bridge-ablation variant in the registry.

## Metrics

Metrics are grouped by `MetricKind`: `deterministic`, `heuristic`, `model_based`.
Heuristic semantic measures are lexical proxies and are labelled as such, not
reported as ground truth.

- Deterministic: exact match, normalized EM, token F1, answer accuracy, citation
  precision/recall/completeness, evidence precision/recall, invalid-citation count,
  empty-prediction, latency, and counts (passages, claims, graph nodes/edges,
  selected, token estimate).
- Heuristic: answer–evidence lexical entailment, negation-cue contradiction rate,
  unsupported-claim rate, conflict-resolution accuracy, subgraph coherence.
- Model-based: an `EntailmentMetric` interface is provided; no model is bundled.

`METRIC_LIMITATIONS` documents edge cases. Token F1 is 1.0 when prediction and gold
are both empty and 0.0 when exactly one is; citation/evidence metrics require gold
evidence and are skipped with a warning when it is missing, not scored as zero.

## Fair comparison

Compared variants use identical example IDs (`same_example_ids`) and are aligned by
example ID (`align_by_example_id`). Retrieval budget, evidence budget, and generator
settings are equal unless a difference is intentional; `check_fairness` detects
per-variant overrides and the runner refuses to run when `enforce_fairness` is set
and a violation is found.

## Running

```bash
uv run egrag experiment run \
  --name demo --dataset synthetic_graph \
  --variants passage_rag,claim_only_rag,full_egrag \
  --seeds 0 --output-dir runs/demo --top-k 3 --evidence-budget 256
uv run egrag experiment summarize --output-dir runs/demo
uv run egrag experiment compare --output-dir runs/demo \
  --variant-a full_egrag --variant-b passage_rag --metric citation_recall
uv run egrag experiment inspect-example --output-dir runs/demo --example-id syn-1
```

Runs are deterministic for a fixed seed (with the fake generator; a real model's
determinism depends on `do_sample=False`, which the adapter always sets unless
sampling is explicitly requested). The manifest records the framework and schema
versions, git commit, environment (Python, and — when actually used this run,
never fabricated — torch/transformers versions, CUDA runtime version, GPU name,
VRAM), resolved config, seeds, dataset fingerprint, and variant configurations.

`--generator` also accepts `huggingface` for a real local model (e.g. Qwen), with
`--generator-model`/`--generator-device`/`--generator-dtype`/
`--generator-quantization`/`--generator-disable-thinking`/`--require-cuda` to
configure it; see "Qwen scale experiments (A/B/C)" in `docs/reproduction.md` for
the exact commands and the model-selection rationale. Retrieval, extraction, and
relation-classification methodology are unaffected by this choice — only the
generator changes.

## Artifacts and resume

A run writes `resolved_config.json`, `manifest.json`, `reproducibility.json`,
`results.jsonl` (one round-trippable `ExampleResult` per line), `aggregate.json`,
`failures.log`, `timing.json`, `metric_warnings.json`, and per-example evidence
packages and graphs. `egrag experiment resume` skips completed
`(variant, seed, example)` triples and recomputes a corrupted trailing line rather
than trusting it.

## Statistics

`egrag.experiments.stats` provides mean/median/std/percentiles, a deterministic
percentile bootstrap CI, paired comparison over per-example deltas, and seed
aggregation. The bootstrap describes estimate variability; no hypothesis test or
significance is claimed.

## Local-model tests

Integration tests that load a real model are skipped unless
`EGRAG_RUN_LOCAL_MODELS=1` is set, so the default suite stays offline.

## Limitations

- The default generator is a deterministic fake; EM/F1 on the synthetic fixtures
  are low by construction and exercise the harness, not model quality. A real
  Hugging Face generator (e.g. Qwen) is supported (see "Running" above) and has
  been smoke-tested on an actual CUDA GPU (`scripts/cuda_smoke_test.py`,
  `artifacts/cuda-smoke/`) — but as of this writing, no full multi-seed,
  multi-variant Qwen scale run has been executed and reported; that is a
  deliberate next step, not a claimed result.
- Heuristic semantic metrics are lexical proxies.
- `reranked_passage_rag` matches `passage_rag` on these inputs because the score
  reranker is identity on BM25 order.
- Deterministic pilots measure evidence structure, not answer quality. Real-model
  answer-quality numbers require running the Qwen scale experiments end to end
  (`docs/reproduction.md`) and are not yet included in this repository's
  artifacts.
