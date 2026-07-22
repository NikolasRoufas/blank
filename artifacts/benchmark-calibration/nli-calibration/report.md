# NLI calibration (§6, §10) — real model usable

Adapter: `HuggingFaceNLIClassifier` (`graph/classification.py`), model
`roberta-large-mnli`, revision `2a8f12d27941090092df78e4ba6f0928eb5eac98`,
device cpu, batch 16, max_length 256, truncation on, label-mapping version 1.
Raw smoke: `real-nli-smoke.json`.

## Label-mapping validation (hard guard) — PASS

`validate_label_mapping` ran the three controlled cases on the real model:
entailment / contradiction / neutral argmax all matched expected → **no issues**.
The label mapping is correct (not swapped/mis-ordered).

## Controlled directional relations (E0.4 / C0.7 / D0.8) — 4/4 correct

| Case | relation | entail a→b | entail b→a | contradiction |
|------|----------|-----------|-----------|---------------|
| supports | supports | 0.980 | 0.026 | 0.002 |
| contradicts | contradicts | 0.000 | 0.001 | 0.999 |
| neutral | neutral | 0.379 | 0.155 | 0.369 |
| duplicate | duplicate | 0.994 | 0.994 | 0.002 |

The neutral case sits near both thresholds (entail 0.379 < 0.4; contra 0.369 <
0.7) and is correctly left neutral — consistent with preferring **conservative
contradiction precision** (false contradictions propagate into beliefs, conflict
sets, and selection).

## Threshold calibration status

This milestone did **not** re-tune NLI thresholds. The validated dev-only
selection from the prior real-NLI milestone is retained and frozen:

- entailment ≈ **0.4**, contradiction ≈ **0.7**, duplicate **0.8** (fixed),
  selected on a development partition by macro-F1 over
  {supports, contradicts, neutral} subject to a support/contradiction precision
  floor of 0.8; see `scripts/run_real_nli_eval.py` and
  `artifacts/paper-results-repaired/end-to-end-real-nli/frozen-config.json`.

Structural contradiction gating (`StructuralContradictionGate`,
`structural_contradiction_ok`) demotes contradictions lacking a shared
entity/subject **without** changing any threshold — the recommended conservative
default. It is evaluated separately from raw NLI (prior milestone).

## Caching (§6)

Cache-key infrastructure is complete and content-addressed:
`build_nli_cache_key` includes premise/hypothesis hashes, model + model_revision
+ tokenizer_revision, label-mapping version, max_length, truncation, all three
thresholds, and schema version — so results are never reused across models,
revisions, prompts, thresholds, or schema. **However**, `DiskCacheBackend` is
**not yet wired** into `HuggingFaceNLIClassifier` or the runner (only `caching/`
and `domain/ports.py` reference it). Persistent-cache cold/warm equivalence and
miss-on-change proofs therefore require first wiring the backend into the NLI
classifier — recorded as a prerequisite for the GPU matrix, not claimed done.

## Latency (CPU M3)

load+validate 24.4 s; **~61 ms / single-direction pair** (batched). A directional
relation = 2 calls ≈ 122 ms. Feeds the §13 projection.
