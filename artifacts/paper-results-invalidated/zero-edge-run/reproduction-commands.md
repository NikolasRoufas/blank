# Exact Reproduction Commands

All commands are local and offline. `ruff` is invoked via `uvx ruff@0.15.18`
because this host's `.venv` ruff console script cannot load (filesystem issue);
it is the identical pinned version. The CLI is invoked via `python -m
egrag.cli.main` because the `.venv/bin/egrag` console script has the same load
issue; behavior is identical.

## 0. Environment

```bash
cd <repo root>            # not a git repo; source identity = environment.json source_fingerprint
uv sync                   # Python 3.12, core + dev deps
```

## 1. Quality gates (must pass before experiments)

```bash
uvx ruff@0.15.18 format --check .
uvx ruff@0.15.18 check .
PYTHONPATH=src .venv/bin/mypy src
PYTHONPATH=src .venv/bin/pytest
uv build
```

## 2. Smoke + core/CLI checks

```bash
PYTHONPATH=src .venv/bin/python -c "import egrag; from egrag.experiments import ExperimentRunner"
PYTHONPATH=src .venv/bin/python -m egrag.cli.main doctor
PYTHONPATH=src .venv/bin/python -m egrag.cli.main experiment run \
  --name smoke --dataset synthetic_graph \
  --variants passage_rag,claim_only_rag,full_egrag --output-dir runs/smoke --top-k 3
```

## 3. Full experiment set (one command — regenerates everything under artifacts/paper-results/)

```bash
PYTHONPATH=src .venv/bin/python scripts/run_paper_experiments.py
```

This runs: all 10 implemented variants on `synthetic_graph` and
`temporal_conflict` (seeds 42/123/2026, top-k 3, evidence budget 256); the
evidence-budget sweep (128/256/512) over passage/claim/full; then writes CSVs,
statistical comparisons (10,000-sample paired bootstrap, seed 12345), figure
scripts+metadata, qualitative cases, failure log, and the validation report.

## 4. Per-component CLI equivalents (for spot checks)

```bash
# one variant comparison on one dataset
PYTHONPATH=src .venv/bin/python -m egrag.cli.main experiment run \
  --name main_synth --dataset synthetic_graph \
  --variants passage_rag,reranked_passage_rag,claim_only_rag,graph_no_propagation,graph_top_claim,graph_coherent_subgraph,graph_no_temporal,graph_no_contradiction,graph_with_propagation,full_egrag \
  --seeds 42 --output-dir runs/main_synth --top-k 3 --evidence-budget 256
PYTHONPATH=src .venv/bin/python -m egrag.cli.main experiment summarize --output-dir runs/main_synth
PYTHONPATH=src .venv/bin/python -m egrag.cli.main experiment compare \
  --output-dir runs/main_synth --variant-a full_egrag --variant-b passage_rag \
  --metric citation_recall --samples 10000
PYTHONPATH=src .venv/bin/python -m egrag.cli.main experiment inspect-example \
  --output-dir runs/main_synth --example-id syn-1
```

## 5. Figures (requires matplotlib, not bundled)

```bash
python artifacts/paper-results/figures/quality_vs_budget.py
python artifacts/paper-results/figures/latency_by_variant.py
```
