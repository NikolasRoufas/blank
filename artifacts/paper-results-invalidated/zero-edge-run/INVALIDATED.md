# INVALIDATED — zero-edge run

**Date of invalidation:** 2026-06-25.

This directory is a frozen, unmodified copy of the original experiment run. It is
**invalid for any conclusion about graph reasoning** and is retained only for
provenance and comparison.

## Original run identity

- **Source-state identifier:** source fingerprint `4e6c12fd16ee12e6…`
  (full value in `environment.json`; the repository is not under Git).
- **Original experiment command:**
  `PYTHONPATH=src .venv/bin/python scripts/run_paper_experiments.py`
- Generator: deterministic `fake`; seeds 42/123/2026; top-k 3; evidence budget 256.

## Why it is invalid for graph reasoning

Every graph-family variant produced graphs with **zero edges** (no support,
contradiction, or supersession edges). Consequently:
- belief propagation ran on edgeless graphs (no message passing of consequence);
- conflict resolution received no conflict sets;
- temporal supersession never fired;
- all graph ablations were effectively identical because the components they
  ablate were inactive.

Root cause: the synthetic fixtures were lexically too dissimilar for the lexical
relation classifier to cross the 0.5 edge threshold, and the temporal fixture had
no structured timestamps (see `../../experiment-repair/root-cause-analysis.md`).

## What must NOT be cited from this run

- Any graph-mechanism result: edge counts, support/contradiction/supersession
  behavior, propagation effects, conflict-resolution accuracy, ablation
  differences, or "EG-RAG vs baseline" graph comparisons.

## What remains usable (software smoke checks only)

- The experiment **harness** behaved correctly: manifests, per-example JSONL,
  aggregates, resume idempotence, failure handling (0 failures), determinism,
  fairness audit, and CSV/validation generation are all valid as **software
  validation**. The runner, output generation, and validation code are **not**
  implicated in the defect and were not changed for that reason.
- Passage/claim variants (no graph) are unaffected as software checks but were
  never graph-mechanism evidence.

## Replacement

Corrected results: **`artifacts/paper-results-repaired/`** (oracle-mechanism and
end-to-end modes, with mechanism-level metrics and component-activation checks).
