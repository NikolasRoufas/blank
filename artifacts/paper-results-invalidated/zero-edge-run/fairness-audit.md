# Fair-Comparison Audit

Machine-readable record: `fairness-audit.json` (`fair: true`).

## Shared across every compared system (no hidden differences)

| dimension | value |
|---|---|
| dataset / split | same per comparison (`synthetic_graph` or `temporal_conflict`, `test`) |
| corpus / query set | identical (each example's own documents + question) |
| example IDs | identical across all variants (verified by `same_example_ids`) |
| generator | `fake` (the only runnable backend), same instance settings |
| decoding | deterministic, seeds {42, 123, 2026} |
| max output tokens | identical (generation defaults) |
| retrieval candidate budget (top-k) | 3 for all |
| evidence token budget | 256 for all main runs (swept identically across systems in the budget study) |
| metric definitions | single shared implementation (`egrag.experiments.metrics`) |

## Allowed (intended) differences only

Each variant differs from `full_egrag` in exactly the component it isolates
(recorded in `fairness-audit.json → intended_isolations`): passage-vs-claim-vs-graph
family, reranking on/off, propagation on/off, selection strategy, temporal edges
on/off, contradiction edges on/off. No variant overrides the generator, top-k, or
evidence budget (`check_fairness` returned no issues).

## Invalid-comparison checks (all clear)

- Stronger generator only for EG-RAG: **no** — all use the same fake generator.
- Larger context/evidence budget for EG-RAG: **no** — identical budgets; the budget
  study varies the budget equally for all systems and reports it.
- Different example sets: **no** — identical example IDs per comparison.
- Dropped failed baseline examples: **no** — 0 failures; the aggregate denominator
  includes all examples and would include failures (see failure policy).
- Gold answers exposed to generation: **no** — `run_system` takes only the question
  and documents; it has no gold parameter (`assert_no_gold_parameters`).
- Thresholds changed after seeing test results: **no** — all defaults, frozen.

## Note on comparison validity scope

Comparisons are valid (same data, same settings), but **statistical power is
negligible**: `synthetic_graph` has 2 examples and `temporal_conflict` has 1.
These are mechanism checks, not powered benchmark comparisons. Paired bootstrap
intervals are reported but are degenerate at this sample size.
