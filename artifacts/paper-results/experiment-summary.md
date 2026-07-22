# EG-RAG Local Experiment Summary (factual)

This summarizes measured outputs only. No publication narrative, no LaTeX, no
superiority claims. See the cited artifacts for machine-readable values.

## Repository state

- **Evaluated locally; not publicly released.** Not a Git repository → no commit
  hash. Source identity = `source_fingerprint_sha256` in `environment.json`.
- Framework `egrag 0.1.0`, schema `1.5.0`.

## Environment

Apple M3 (8 cores), 16 GiB RAM, no GPU used; macOS 15.5 arm64; Python 3.12.13.
Optional deps present: only `pyyaml`. Absent: numpy, matplotlib, transformers,
torch, httpx, sentence-transformers, networkx, rank-bm25. Offline throughout.

## Pre-experiment gates (all PASS)

ruff format ✓ · ruff check ✓ · mypy ✓ (94 files) · pytest ✓ · `uv build` ✓ ·
core import ✓ · `egrag doctor` ✓ · `egrag experiment run` ✓.

## Datasets and splits

- `synthetic_graph` (test, n=2), `temporal_conflict` (test, n=1) — built-in
  synthetic fixtures.
- Real benchmarks (HotpotQA/FEVER): **adapter implemented, data unavailable**
  locally (no file, no network) → not run.

## Models / generator

- Only `FakeTextGenerator` (deterministic, templated) is runnable; HF and
  OpenAI-compatible backends are implemented but unavailable (deps/endpoint
  absent). Answer-text metrics are therefore uninformative by construction.

## System variants (all 10 implemented, run on both datasets)

passage_rag, reranked_passage_rag, claim_only_rag, graph_no_propagation,
graph_with_propagation, graph_top_claim, graph_coherent_subgraph,
graph_no_temporal, graph_no_contradiction, full_egrag.
Unavailable requested ablations: `egrag_no_coherent_selector` (reported via
`graph_top_claim`), `egrag_no_provenance_discount` (no runtime toggle — absent).

## Seeds / determinism

Seeds {42, 123, 2026}. The fake generator and all used components are
deterministic; per-seed outputs are byte-identical (`logs/determinism.json`),
so seed variance is zero. Determinism is genuine here (no sampling backend).

## Completed / failed examples

All runs completed; **0 failed examples**, 0 invalid outputs, 0 empty
predictions, 0 invalid citations (`failed-examples.jsonl` is empty).

## Headline measured numbers (seed 42; full table in main-results.csv)

`synthetic_graph` (n=2):
- passage_rag / reranked_passage_rag / claim_only_rag / graph_top_claim:
  token_f1 0.0769, citation_recall 1.00, evidence_recall 1.00, entailment 0.050,
  num_selected 2.5.
- graph_no_propagation / graph_with_propagation / graph_coherent_subgraph /
  graph_no_temporal / graph_no_contradiction / full_egrag:
  token_f1 0.0833, citation_recall 0.50, evidence_recall 0.50, entailment 0.0556,
  num_selected 1.0.

`temporal_conflict` (n=1): all variants citation_recall 1.00, evidence_recall
1.00, token_f1 0.00; graph variants entailment 0.1111 vs 0.10 for passage/claim.

## Paired comparison (full_egrag − passage_rag; 10k bootstrap, seed 12345)

- synthetic_graph: token_f1 Δ +0.0064 [0, 0.0128]; citation_recall Δ −0.50
  [−0.50, −0.50]; evidence_recall Δ −0.50 [−0.50, −0.50]; entailment Δ +0.0056.
- temporal_conflict: token_f1/recall Δ 0; entailment Δ +0.0111.
No significance claimed; intervals degenerate at n≤2.

## Robustness (evidence-budget sweep 128/256/512)

Metrics are **flat** across budgets for all three systems (examples fit the
smallest budget); only trivial latency differences. No budget sensitivity
observed (`robustness-results.csv`).

## Efficiency

Total per-example latency ≈ 0.1–0.3 ms (synthetic). Counts per variant in
`efficiency-results.csv`. Per-stage latency, peak memory, and pair counts are
**not instrumented** → unavailable.

## Negative / inconclusive findings (stated plainly)

- On these fixtures EG-RAG's graph selection **reduces** citation/evidence recall
  vs passage/claim baselines (more selective: 1 vs ~2.5 claims).
- Token-F1 / entailment gains from the graph are marginal (+0.006) and not
  powered.
- Budget sweep shows no effect at this corpus size.
- Answer-correctness metrics are ~0 due to the fake generator.

## Threats to validity

Tiny synthetic datasets (n≤2); fake generator (no real answer generation); lexical
baselines and lexical/heuristic semantic metrics (not ground truth); no real
benchmark data; CPU-only; deterministic so no variance estimate from sampling.

## Missing experiments (not run; reasons)

Real multi-hop/FEVER (no data), retrieval-noise & contradiction-density
robustness (no dataset variants), candidate-pair pruning study (counts not
recorded), per-stage latency / peak memory (not instrumented), real-generator
comparison (backends unavailable), `egrag_no_provenance_discount` ablation
(no toggle).

## Artifact paths

Everything under `artifacts/paper-results/`: `environment.json`,
`repository-state.md`, `experiment-inventory.md`, `fairness-audit.{json,md}`,
`development-selection.md`, `frozen-configs/` (+`checksums.json`), `smoke/`,
`manifests/`, `per-example/`, `aggregates/`, `comparisons/` (stats),
`tables/` (+`data-dictionary.md`), `figures/` (scripts+metadata),
`qualitative/all-cases.jsonl`, `logs/`, `failed-examples.jsonl`,
`main-results.csv`, `ablation-results.csv`, `robustness-results.csv`,
`efficiency-results.csv`, `statistical-comparisons.{json,md}`,
`result-validation-report.md`, `reproduction-commands.md`.

## One-command reproduction

```bash
PYTHONPATH=src .venv/bin/python scripts/run_paper_experiments.py
```
