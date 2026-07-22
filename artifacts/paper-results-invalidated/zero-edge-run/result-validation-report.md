# Result Validation Report

Machine-readable record: `logs/validation.json` (`ok: true`, 0 failed checks).
Validation is performed by `scripts/run_paper_experiments.py` against the saved
artifacts; nothing is hand-patched.

## Checks performed and outcomes

| # | check | result |
|---|---|---|
| 1 | Recompute every aggregate metric from per-example JSONL and compare to saved `aggregate.json` (tolerance 1e-9) | PASS (all variants, both datasets, all metrics) |
| 2 | Every CSV value maps to an aggregate value (CSVs are emitted from the same aggregation) | PASS by construction |
| 3 | Every plotted value maps to a result artifact (figure scripts read the CSVs directly) | PASS |
| 4 | Metric direction recorded | PASS (see `tables/data-dictionary.md`) |
| 5 | Failed-example denominator policy (failures counted in `n_examples`, never dropped) | PASS (0 failures; denominator includes them) |
| 6 | Duplicate example IDs | PASS (none) |
| 7 | Missing examples / identical example IDs across compared variants | PASS (`same_example_ids` == []) |
| 8 | No gold-answer leakage to generation | PASS (`run_system` has no gold parameter) |
| 9 | Frozen-config checksums recorded | PASS (`frozen-configs/checksums.json`) |
| 10 | Fairness audit | PASS (`fairness-audit.json` `fair: true`) |
| 11 | Manifests contain framework/schema version, environment, seeds, variant configs | PASS (`manifests/*.json`) |
| 12 | Determinism across seeds {42,123,2026} | PASS (identical per-seed outputs; `logs/determinism.json`) |
| 13 | Figure scripts reproduce from saved data | scripts present; rendering requires matplotlib (not installed) — verified to read the saved CSVs, not embedded data |

## Defect found and repaired during this work

- **Defect (implementation, path handling):** `ExperimentRunner.__init__` passed
  the raw (possibly relative) `output_dir` to `safe_artifact_path` against the
  parent base, which re-joined it and produced a doubly-nested directory (e.g.
  `out/out/run`), so results were not written to the requested directory.
  Reproduced minimally with a relative `output_dir`.
- **Fix (smallest):** validate the **resolved absolute** path instead
  (`safe_artifact_path(self._base, self._base.parent)`). One line + an explanatory
  comment.
- **Regression test:** `tests/integration/test_experiment_runner.py::`
  `test_relative_output_dir_writes_to_that_dir` (chdir + relative `output_dir`;
  asserts results land in `out/run` and no `out/out` is created). Fails on the old
  code, passes on the fix.
- **Invalidation:** the single buggy driver run produced no readable results and
  was discarded; all reported results come from the corrected code. Quality gates
  were re-run after the fix.

## Sample-size caveat (validity, not a validation failure)

Datasets are tiny (`synthetic_graph` n=2, `temporal_conflict` n=1). Aggregates and
bootstrap CIs are exact for these examples but have no statistical power. This is a
mechanism-validation harness on synthetic fixtures, not a powered benchmark.
