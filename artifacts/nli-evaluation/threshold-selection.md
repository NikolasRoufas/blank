# NLI Threshold Selection — method (values PENDING the runtime)

The lexical thresholds are **not** reused for the NLI model. The selection method
below is implemented and ready, but the actual threshold values **cannot be
chosen yet** because the real NLI model could not be run (see `smoke/BLOCKED.md`).
No distributions or thresholds are fabricated.

## Method (frozen procedure, to run once the runtime is installed)

1. Split the 154 repaired fixtures into a **development** half and a held-out
   half by example index parity per category (deterministic; the test half is
   never used for selection).
2. On the development half only, run the real NLI classifier over all generated
   candidate pairs and collect entailment / contradiction / neutral score
   distributions, separated by gold relation type.
3. Evaluate relation precision/recall over candidate pairs across a grid of
   `entailment_threshold` and `contradiction_threshold` (and the
   `duplicate_threshold` for mutual entailment), holding the precedence policy of
   `egrag.graph.nli.decide_relation` fixed.
4. **Selection criterion (documented, fixed in advance):** maximize macro-F1
   across {supports, contradicts, neutral} subject to support-precision ≥ 0.8 and
   contradiction-precision ≥ 0.8 (a minimum-precision constraint with
   maximum-recall tie-breaking). The `duplicate_threshold` is fixed at the
   existing 0.8 unless development data shows duplicates and one-directional
   support are not separable at 0.8.
5. **Freeze** the selected thresholds (write them here and into a frozen config)
   **before** any evaluation on the held-out half.
6. Save raw development predictions to `dev-predictions.jsonl`.

## Constraints honored

- Thresholds are chosen on development data only, never on test examples.
- Thresholds are not lowered merely until fixtures pass; the precision floor
  prevents that.
- Candidate-pair recall is measured and reported **separately** (a pruned gold
  pair is a candidate-generation error, not an NLI error).

## Current values

**Unset / pending.** Until the runtime is installed and step 2 runs, the code
falls back to the documented defaults (`entailment_threshold=0.5`,
`contradiction_threshold=0.5`, `duplicate_threshold=0.8`) which are NOT claimed to
be tuned for any NLI model. `dev-predictions.jsonl` does not exist yet.
