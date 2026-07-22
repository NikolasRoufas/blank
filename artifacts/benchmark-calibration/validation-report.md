# Validation report

## Scientific-integrity invariants (held)

- **Five quantities kept distinct** (belief, extraction_confidence,
  relation_confidence, source_reliability, query_utility) — domain models
  unchanged; no collapse introduced.
- **Contradictory evidence not discarded** — contradiction edges + conflict sets
  + answer uncertainty preserved; real-NLI contradiction smoke = 0.999 surfaced,
  not dropped.
- **Newer ≠ truer** — no supersession changes; temporal toggle unchanged.
- **Repetition ≠ corroboration** — duplicate policy intact (duplicate_threshold
  0.8; duplicates preserve provenance, do not add support).
- **Provenance retained** — every claim keeps ≥1 source span; ungrounded
  extractor spans are rejected (verified by the extractor's span check).
- **Untrusted source text** — renderers delimit evidence and forbid following
  in-evidence instructions; the generator hallucination ("2013") was surfaced as a
  faithfulness failure, not hidden.
- **No gold leakage** — gold answers/evidence live only on `DatasetExample`; never
  passed into `run_system`; samples fixed by seed before any system run; subsets
  selected by data availability (gold coverage), never by model success.

## Data integrity

- FEVER: 15,935 rows, 0 validation issues, 0 leakage (prior manifest).
- HotpotQA: 7,405 rows, 0 duplicate ids, 0 `validate_benchmark` issues; file sha
  matches recorded; fullwiki gold-coverage constraint (28.2%) documented.
- Frozen-config checksums recorded (`frozen-configs/checksums.json`).

## Honesty / negative results preserved

- Real generator and extractor smokes: **0% valid structured output** — recorded,
  not hidden; no fake substituted to fake success.
- Deterministic HotpotQA pilot: greedy-connected selector **halves** multi-hop
  recall under lexical edges — recorded as a negative finding.
- FEVER gold-evidence is non-discriminating for the graph — recorded.
- Prior milestone results (real-NLI, bridge, controlled-mechanism, zero-edge
  invalidation, benchmark-pipeline/integration) **not modified**.

## Gate status

Recorded in `final-report.md` (§ quality gates) and `environment-recovery/
final-verification.md`.
