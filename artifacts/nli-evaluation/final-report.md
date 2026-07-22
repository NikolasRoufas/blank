# Real-NLI End-to-End Evaluation — Final Report

The real-NLI run is **executed** (the `local-models` runtime is now installed).
Results are reported as measured, including failures. Oracle, lexical, and
zero-edge results are unchanged.

## Runtime

- torch **2.12.1**, transformers **5.12.1**; MPS available but **not used**.
- Selected model: **`roberta-large-mnli`**, revision
  `2a8f12d27941090092df78e4ba6f0928eb5eac98`, loaded **offline from cache** (no
  re-download). Device: **CPU** (the adapter accepts `device=`/MPS, but untested
  MPS was deliberately not used for speed). Full-suite eval runtime ≈ **7.7 s**;
  model load ≈ 8–12 s; memory not separately instrumented (model ≈ 1.4 GB).

## Label mapping

Validated against the controlled cases — **correct** (entailment 0.994,
contradiction 0.998, neutral 0.997), mapped by label **name** (not index order).
Hard validation is a startup guard (`validate_label_mapping`).

## Frozen thresholds (development partition only, 308 dev pairs)

`entailment_threshold=0.4`, `contradiction_threshold=0.7`,
`duplicate_threshold=0.8` (fixed). Criterion: max macro-F1 with support/
contradiction precision ≥ 0.8. **The precision floor was NOT met** — dev
contradiction precision = **0.47** (support precision = 1.0): roberta over-predicts
CONTRADICTION on short synthetic claims; raising the contradiction threshold
mitigates but does not eliminate this. Frozen + checksummed before evaluation
(`end-to-end-real-nli/frozen-config.{json,sha256}`).

## Candidate-pair recall

**1.0** across all categories (measured separately; same candidate generation as
oracle). No relation failure is attributable to candidate pruning.

## Relation metrics (real-NLI, vs oracle upper bound)

| mechanism | real-NLI | oracle | gap |
|---|---|---|---|
| contradiction edge recall | 1.00 | 1.00 | 0 |
| unresolved-conflict accuracy | 1.00 | 1.00 | 0 |
| temporal supersession recall | 1.00 | 1.00 | 0 |
| duplicate-cluster accuracy | 1.00 | 1.00 | 0 |
| support edge recall | **0.00** | 1.00 | −1.00 |
| multi-hop required-hop coverage | **0.14** | 1.00 | −0.86 |
| preferred conflict-resolution accuracy | **0.18** | 1.00 | −0.82 |

Contradiction precision is **below** 1.0 (spurious contradictions on unrelated
short claims) — recall is high, precision is the weak point.

## Accepted edges by type (full suite, 154 examples)

support **40**, contradiction **135**, supersession **22**, duplicate **88**;
conflict sets **110**; propagation iterations 2050.

## Duplicate-vs-support behavior

Near-identical "support" claims score mutual entailment ≥ 0.8 → labeled
**DUPLICATE** (per the documented precedence), so support_edge_recall is 0 on the
support and duplicate fixtures. The duplicate threshold was **not** lowered to
shift counts. This is an honest fixture/representation interaction, not a bug.

## Temporal capability status

Supersession recall **1.0** — but this is **timestamp-driven** (structured
`valid_from` in the gold claims), **NLI-independent**. NLI additionally flags
old/new value pairs as contradictions. Temporal extraction **from raw text**
remains unavailable (no extractor); this is oracle/structured-metadata only.

## Multi-hop capability status

Support recall 0.0, required-hop coverage 0.14: NLI does not entail across
multi-hop bridge claims and does not infer `DEPENDS_ON`. Connectivity-based
selection cannot assemble the 2-hop subgraph without a bridging edge.

## Ablations (real NLI) — isolation holds

| variant | support | contradiction | supersession | duplicate | conflict sets | prop. iters |
|---|---|---|---|---|---|---|
| full_egrag | 40 | 135 | 22 | 88 | 110 | 2050 |
| graph_no_contradiction | 40 | **0** | 22 | 88 | **0** | 400 |
| graph_no_propagation | 40 | 135 | 22 | 88 | 110 | **0** |
| graph_top_claim | 40 | 135 | 22 | 88 | 110 | 2050 |
| graph_no_temporal | 40 | 157 | **0** | 88 | 110 | 2269 |

## Failed examples

No runtime failures (all 154 produced graphs; `validation.json` ok). "Failures"
here are *mechanism-recovery* failures (support/multi-hop), recorded in the
metrics, not dropped.

## Output directory

`artifacts/paper-results-repaired/end-to-end-real-nli/` (manifest, frozen-config
+ sha256, per-example.jsonl, aggregates.json, ablations.json,
component-activation.{json,csv}, validation.json). Threshold-selection raw
predictions: `artifacts/nli-evaluation/dev-predictions.json`. Comparison:
`artifacts/nli-evaluation/oracle-vs-real-nli.md`.

## Tests added / code changed

- `build_run` gained an optional `classification_config` (dev-frozen thresholds);
  precedence with the no-contradiction ablation preserved.
- Drivers `scripts/run_real_nli_eval.py` (new). NLI infra (label validation,
  directional, cache keys, precedence) was already present and tested.
- No oracle code/results changed.

## Quality-gate results

Reported at session end (ruff, mypy, pytest, build, core-only import,
label-mapping validation). Model-dependent tests run under
`requires_local_models`.

## Exact reproduction commands

```bash
uv sync --extra local-models
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH=src \
  .venv/bin/python scripts/run_real_nli_eval.py
PYTHONPATH=src .venv/bin/pytest -m requires_local_models
```

## Suitability of results

- **Adapter validation:** YES — real model loads offline, label mapping validated,
  batched directional inference, NLI cache keys, precedence policy all exercised.
- **Controlled end-to-end mechanism claims:** PARTIAL — EG-RAG recovers and reasons
  over **contradiction** (and timestamped **supersession**) end-to-end with an
  open-weight NLI model on controlled synthetic examples. It does **not** recover
  **support** or **multi-hop** on these fixtures (duplicate collapse; no
  dependency inference), and contradiction **precision** is imperfect.
- **Real benchmark claims:** NO — synthetic fixtures, gold claims (no real
  extraction), no real datasets, no real answer generator.

## Remaining limitations / next experiment

Support and multi-hop recovery need either less-duplicative support fixtures +
asymmetric entailment, or a relation model that captures bridging/dependency;
contradiction precision needs calibration or a better NLI. The next required
experiment before any EG-RAG-vs-RAG paper claim: **real datasets + real claim
extraction + a real answer generator**, then re-measure mechanism and answer
metrics.
