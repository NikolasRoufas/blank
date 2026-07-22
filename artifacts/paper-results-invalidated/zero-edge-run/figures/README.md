# Figures

`matplotlib` and `numpy` are **not installed** and there is no network to install
them, so no raster/vector image was rendered. For each figure that has data, this
directory contains a deterministic generation **script** and a factual
**metadata** JSON (axes, metric, systems, source file, command). Running the
script in an environment with `matplotlib` reproduces the figure from the saved
CSV — no data values are edited.

## Available (data exists)

- `quality_vs_budget.py` / `.metadata.json` — token_f1 / citation_recall /
  evidence_recall vs evidence_token_budget; source `robustness-results.csv`.
  (Data shows the metrics are **flat** across budgets 128/256/512 because the
  synthetic examples fit the smallest budget.)
- `latency_by_variant.py` / `.metadata.json` — total `latency_ms` per system;
  source `efficiency-results.csv`.

## Unavailable (data not produced by the harness; not fabricated)

- **performance by hop count** — datasets carry no hop-count annotations.
- **retrieval-noise robustness** — no noise-injection dataset variant implemented.
- **contradiction-density robustness** — no density-sweep dataset implemented.
- **candidate-pair pruning** — possible/candidate/classified pair counts are not
  recorded by the runner.
- **per-stage latency breakdown** — only total latency is recorded; per-stage
  timing is not instrumented.

To render available figures (requires matplotlib):

```bash
python artifacts/paper-results/figures/quality_vs_budget.py
python artifacts/paper-results/figures/latency_by_variant.py
```
