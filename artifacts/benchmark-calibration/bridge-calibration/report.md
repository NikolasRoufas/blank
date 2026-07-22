# Bridge calibration (§11) — proxy-only on CPU; real run deferred

Bridges connect multi-hop claims to improve query utility without changing
belief/conflicts/corroboration (a documented EG-RAG invariant). Calibrating bridge
**activation** meaningfully requires real support edges between intermediate
claims — i.e. the real NLI classifier — and answer-side utility, i.e. a real
generator. Both are addressed in `live-smoke/summary.md`:

- Real NLI is usable but **not wired into the variant runner** (the runner uses
  `LexicalPairClassifier`).
- The real generator is currently unusable for the structured contract.

So on the deterministic CPU pipeline the bridge mechanism has almost no edges to
build on (HotpotQA lexical graphs: avg 0.1–1.4 edges; see
`reasoning-calibration/report.md`), and bridge **activation is effectively zero**
— not a calibrated result, an artifact of edge sparsity.

## What is established (preserved from prior milestone)

The bridge milestone validated, on controlled multi-hop cases with **real NLI**,
that bridges:
- produce **zero** direct belief change, **zero** conflict sets, **zero**
  corroboration increase (invariants hold), while improving connectivity;
see `artifacts/bridge-milestone/`. Those results are not re-derived or altered here.

## Proxy plan for the GPU matrix (HotpotQA, real NLI)

Where no gold bridge edge exists, use proxy metrics (clearly labelled): official
supporting facts as the gold chain, chain connectivity, intermediate-entity
correctness, and a fixed manual-audit subset (the 28-example full-coverage
HotpotQA subset is reserved for this — `samples/hotpot-dev-100.json`
`full_coverage_subset_ids`). Calibrate a small set only: minimum query utility,
bridge-confidence threshold, max bridge degree, generic-entity filters,
same-source policy. Re-confirm the three invariants on benchmark data.
