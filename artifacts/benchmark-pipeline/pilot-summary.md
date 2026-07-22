# Pilot Summary

## Status: pilots NOT executed (two independent blockers); pipeline pieces validated

Per the milestone's stop conditions, no benchmark pilot was run. Both pilots are
blocked, for different reasons:

- **HotpotQA pilot — blocked (data):** the dataset is cached only as parquet;
  `pyarrow`/`datasets`/`pandas` are not installed and cannot be installed offline.
  The adapter raises a typed `MissingDependencyError`. No HotpotQA numbers exist.
- **FEVER pilot — blocked (runtime/disk):** the FEVER adapter and data are ready
  (gold-evidence setting), but loading the real NLI + generator models wedged on
  disk I/O and free disk fell to 4.5 GiB. Running 8 variants × 25 examples with a
  real generator is not safely feasible on this host now (see `runtime-estimate.md`).

## What WAS built and validated (offline, deterministic)

- **FEVER adapter** (`FeverGoldEvidenceDataset`) — loads cached JSONL → canonical
  `DatasetExample`; gold label/evidence kept aside; validated (0 issues on slices).
- **HotpotQA adapter** (`HotpotQADataset`) — parquet→example mapping implemented;
  fails cleanly until a parquet reader exists.
- **Benchmark metrics** (`benchmark_metrics`) — HotpotQA EM/normalized-EM/token-F1/
  supporting-fact-P/R/F1/joint; FEVER label-accuracy/evidence-P/R/F1/alternative-
  evidence-set recovery/official FEVER score. Unit-tested incl. edge cases.
- **Tests:** `tests/integration/test_benchmark_pipeline.py` — 12 passed, 1 skipped
  (cache-gated real-FEVER load), offline.
- **Generator/NLI:** real adapters exist and are selected (`Qwen2.5-0.5B-Instruct`,
  `roberta-large-mnli`); a runtime estimator script is added. Live execution is the
  only part blocked.

## Pilot analysis questions (answerable now vs. pending)

1. Does real claim extraction preserve required evidence? — **pending live run.**
2. Does the generator follow evidence/citation instructions? — **pending**
   (the generation-validation contract exists and is unit-tested with the fake).
3. Does the graph activate on real examples? — **pending** for benchmarks; proven
   on controlled fixtures (bridge milestone).
4. Do bridges connect real multi-hop claims? — **pending** (HotpotQA blocked).
5. Does contradiction gating reject unrelated claims? — proven on controlled
   fixtures (guarantee); FEVER REFUTES pairs would exercise it on a live run.
6. Are baselines fair? — the harness fairness audit exists (`check_fairness`);
   wired for the variant set; runnable once the live pilot runs.
7. Is CPU runtime feasible? — **No on this host** (disk/I/O). Estimated 15–60 min
   for FEVER 25×8 warm given adequate disk; full matrix is GPU-scale.
8. Which components need calibration? — see candidate configs in `final-report.md`.

## Decision

Do not proceed to a live pilot in this environment. Unblock disk + parquet reader
(see `runtime-estimate.md` / `dataset-validation.md`), then run the pilot sequence
in `reproduction-commands.md`.
