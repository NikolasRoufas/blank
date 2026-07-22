# Development vs. Test Discipline

## No tuning was performed on any split

These experiments did **not** select, search, or adjust any hyperparameter using
results from any split (there is no separate dev split locally, and the built-in
fixtures are `test`-tagged mechanism checks). Every tunable value uses the
**framework default** as committed in the source / `configs/baseline.yaml`. No
value was changed after viewing any result.

## Values used (all defaults; recorded for reproducibility)

| parameter | value | source | selection criterion |
|---|---|---|---|
| retrieval top-k | 3 | experiment config | fixed a priori (small synthetic corpora); not tuned |
| evidence token budget | 256 (main); 128/256/512 (budget sweep) | experiment config | fixed sweep grid, not selected by score |
| reserved output tokens | 64 | experiment config | fixed |
| chunk size / overlap | 512 / 64 | `ChunkingConfig` defaults | default |
| candidate strategy / lexical overlap | pruned / 0.2 | `CandidateConfig` defaults | default |
| relation thresholds (entail/contradict/dup) | 0.5 / 0.5 / 0.8 | `ClassificationConfig` defaults | default |
| propagation damping / tolerance / max-iter | 0.5 / 1e-4 / 50 | `PropagationConfig` defaults | default |
| conflict margin / low-evidence | 0.1 / 0.15 | `ConflictResolutionConfig` defaults | default |
| selection weights | utility 1.0 / belief 1.0 | `SubgraphSelectionConfig` defaults | default |
| source reliability | metadata strategy, default 0.5 | `SourceReliabilityConfig` defaults | default |
| generator | fake, deterministic, seed ∈ {42,123,2026} | experiment config | only runnable backend |

## Frozen configurations

The exact `ExperimentConfig` for every run is frozen under
`artifacts/paper-results/frozen-configs/*.json` with SHA-256 checksums in
`frozen-configs/checksums.json` (which also checksums the shipped `configs/*.yaml`
pipeline configs). These were written at run time and not edited afterward.

## Incident policy

A genuine implementation defect **was** discovered during this work — a relative
`output_dir` was double-joined by the runner's path validation (see
`result-validation-report.md`). It was fixed with a regression test before the
final runs; no result produced with the buggy path was kept (the buggy run
produced no readable results and was discarded). All reported results come from
the corrected code.
