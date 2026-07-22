# Data Dictionary (CSV tables)

All values are copied directly from the per-example/aggregate artifacts by
`scripts/run_paper_experiments.py`; none are hand-edited. Decimal precision: 4
places. CI columns (`*_ci_low`, `*_ci_high`) are 95% percentile bootstrap
(10,000 resamples, seed 12345) over per-example values at seed 42. Winners are
**not** marked or bolded.

## Common identifier columns

| column | meaning |
|---|---|
| `dataset` | dataset adapter name (`synthetic_graph`, `temporal_conflict`) |
| `system` | experiment variant name |
| `evidence_token_budget` | evidence budget for the row (robustness table only) |
| `n_examples` | examples scored for this system (seed 42) |
| `n_failed` | failed examples for this system (included in the denominator) |

## Metric columns — direction

Higher is better: `exact_match`, `normalized_exact_match`, `token_f1`,
`answer_accuracy`, `citation_precision`, `citation_recall`,
`citation_completeness`, `evidence_precision`, `evidence_recall`,
`answer_evidence_entailment`.

Lower is better: `invalid_citations`, `empty_prediction`, `contradiction_rate`,
`unsupported_claim_rate`, `latency_ms`.

Descriptive (no direction): `num_passages`, `num_claims`, `num_graph_nodes`,
`num_graph_edges`, `num_selected`, `token_estimate`.

## Metric kinds (see ../logs/metric-kinds.json)

- **deterministic**: all `*_match`, `token_f1`, `answer_accuracy`, all
  citation/evidence metrics, `invalid_citations`, `empty_prediction`, counts,
  `latency_ms`.
- **heuristic** (lexical proxies, NOT ground truth): `answer_evidence_entailment`
  (token coverage), `contradiction_rate` (negation cue), `unsupported_claim_rate`.
- **model_based**: none used (interface only; no model bundled).

## Metric definitions (summary; full limitations in source `METRIC_LIMITATIONS`)

| metric | definition |
|---|---|
| `exact_match` | raw string equality of prediction to any gold answer |
| `normalized_exact_match` / `answer_accuracy` | equality after lowercasing, punctuation/article/whitespace normalization |
| `token_f1` | max token-overlap F1 over golds; 1.0 if both empty, 0.0 if exactly one empty |
| `citation_precision/recall/completeness` | cited source IDs vs gold source IDs |
| `evidence_precision/recall` | selected evidence source IDs vs gold source IDs |
| `invalid_citations` | count of cited claim IDs absent from the package |
| `empty_prediction` | 1.0 if the answer has no non-whitespace content |
| `answer_evidence_entailment` | fraction of answer tokens covered by selected evidence (lexical) |
| `contradiction_rate` | 1.0 if the answer negates shared evidence content (surface cue) |
| `unsupported_claim_rate` | fraction of answer sentences with lexical support < 0.5 |
| `num_passages/claims/graph_nodes/graph_edges/selected` | pipeline counts |
| `token_estimate` | recorded evidence token usage from the package budget |
| `latency_ms` | total per-example wall-clock (single value; no per-stage breakdown) |

## CSV files

- `main-results.csv` — answer/evidence/heuristic metrics per (dataset, system) with CIs.
- `ablation-results.csv` — graph ablations on `synthetic_graph` (subset of main).
- `robustness-results.csv` — evidence-budget sweep (128/256/512) on `synthetic_graph`.
- `efficiency-results.csv` — counts + total latency per (dataset, system).
