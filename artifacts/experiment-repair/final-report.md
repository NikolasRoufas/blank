# Zero-Edge Repair — Final Report

## Exact root cause

The graph pipeline was correct; the **synthetic fixtures were outside the lexical
relation classifier's edge-forming range** (a fixture-design defect). The
`LexicalPairClassifier` scores pairs by Jaccard token overlap and returns
*neutral* below 0.2; `GraphBuilder` stores a SUPPORTS edge only when entailment ≥
0.5. The intended support claims were paraphrases with overlap ≈ 0.23 → neutral →
**no edge**. Contradictions were not expressed as shared-proposition polarity
flips, and the temporal fixture lacked structured `valid_from` timestamps.
Secondary: **entity normalization** emitted sentence-initial function words
("The", "An") as named entities, weakening shared-entity candidate pruning.
Process: there were **no mechanism metrics and no activation preflight**, so
edgeless graphs passed silently behind fake-generator answer metrics.

Controls confirmed the pipeline: a high-overlap support pair yields entailment 1.0
and one edge; a same-text+negation pair yields contradiction 0.7.

## Affected runs (invalidated, preserved unmodified)

All graph-family variants of `main_synthetic_graph`, `main_temporal_conflict`, and
the budget sweep in the original `artifacts/paper-results/` run. Copied to
`artifacts/paper-results-invalidated/zero-edge-run/` with `INVALIDATED.md`. The
experiment harness, manifests, output generation, validation, and passage/claim
variants were **not** affected and remain valid as software checks.

## Files changed

- `src/egrag/adapters/extraction/baseline.py` — entity normalization now strips
  leading capitalized function words (secondary-defect fix).
- `src/egrag/experiments/mechanisms.py` — **new**: gold-annotated mechanism
  fixtures (7 categories × 22 = 154 examples) + `GoldRelationClassifier` (oracle).
- `src/egrag/experiments/mechanism_eval.py` — **new**: graph builder/runner,
  mechanism metrics, oracle/end-to-end modes, activation + preflight.
- `src/egrag/experiments/__init__.py` — exports the new API.
- `scripts/run_mechanism_repair.py` — **new**: repaired-experiment driver.
- `tests/integration/test_mechanism_repair.py` — **new**: 17 regression tests.

No production thresholds were changed. No code was altered merely to make an
example pass. Gold labels are independent of predictions.

## Fixtures added (gold-annotated, schema-validated, varied)

22 each: support, contradiction, temporal (with a newer-unrelated distractor that
must not supersede), duplicate-source (same-source exact repeats + one independent
paraphrase), multi-hop (two required claims bridged by a shared entity),
unresolved-conflict (symmetric evidence), preferred-conflict (asymmetric
corroboration). Entities, values, dates, voice, and negation vary across examples.

## Metrics added

support/contradiction/supersession edge precision & recall; candidate-pair recall;
conflict-set recall; conflict-resolution accuracy; unresolved-conflict accuracy;
required-claim recall; required-hop coverage; selected-subgraph connectivity;
duplicate-cluster accuracy — each with explicit empty-set / denominator policy and
a `None`-excluded-from-aggregation rule.

## Tests added

17 regression tests (the required cases 1–19, several combined), all deterministic
and offline; plus an entity-normalization regression. Result on the affected
subsystem: **17 passed**.

## Thresholds changed

None. The oracle classifier emits entailment 0.7 (above the 0.5 support threshold,
below the 0.8 duplicate threshold) and contradiction 0.95 (above 0.5, below the
1.0 no-contradiction-ablation threshold) so it behaves like a sub-1.0 classifier;
production thresholds are untouched.

## Quality-gate results

See the printed status at the end of the session (ruff format/check, mypy, pytest,
build). Gates were re-run after every code change.

## Repaired experiment commands

```bash
PYTHONPATH=src .venv/bin/python scripts/run_mechanism_repair.py     # oracle + e2e + ablations + budget
PYTHONPATH=src .venv/bin/pytest tests/integration/test_mechanism_repair.py
```

## Oracle-mechanism results (per category, mean; vs gold)

All gold edge/conflict/required metrics = **1.0** (support, contradiction,
temporal supersession, duplicate cluster, multi-hop hops, unresolved, preferred
resolution); candidate_pair_recall = 1.0 throughout.

## End-to-end results (activation-based recovery)

- contradiction: 100% produced a contradiction edge from raw text.
- support: 0% produced a *support* edge; 100% produced an edge, recovered as a
  **duplicate** (near-identical sources). A surfaced lexical-classifier limitation.

## Component-activation counts (full_egrag oracle, 154 examples)

support 176, contradiction 66, supersession 22, duplicate 22 edges; 66 conflict
sets; 1997 propagation iterations; 154 connected selected subgraphs. Ablations
each zero out exactly their target (see `before-vs-after.md` /
`paper-results-repaired/diagnostics/component-activation.csv`).

## Remaining limitations

- Oracle saturation (1.0) validates **downstream reasoning**, not the lexical
  classifier; it is not a quality claim.
- End-to-end covers only support+contradiction; temporal/duplicate/multi-hop/
  conflict are oracle-only because the lexical extractor does not populate
  structured semantics/timestamps from text.
- Synthetic, templated fixtures (22/category); not natural-language benchmarks.
- No real generator; no real NLI model; CPU-only; deterministic.

## Suitability of repaired outputs

- **Software validation:** YES — mechanisms are exercised and tested.
- **Graph-mechanism evaluation (oracle):** YES for downstream reasoning given gold
  relations; mechanism metrics are gold-aligned.
- **Benchmark claims:** NO — synthetic fixtures, oracle relations, no real models
  or datasets; end-to-end extraction/classification is not yet adequate for
  support/temporal recovery from natural text.
