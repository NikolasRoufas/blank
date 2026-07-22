# Candidate comparison (§8) — what could and could not be compared

## C1 / C2 / C3 with real generation: NOT RUN (blocked, honestly)

C1/C2/C3 differ mainly in generator settings (model, max_new_tokens, budget,
top-k) and NLI/bridge thresholds. A fair C1/C2/C3 comparison on benchmark
**answer/label** metrics requires a usable real generator. The real generator
smoke is **0/6 valid structured output** (`generator/report.md`), so running
C1/C2/C3 through it would produce only malformed outputs — not a comparison.
Per the rules, **no fake generator was substituted** to fake a winner. C1/C2/C3
remain defined candidates (see `resume-inventory.md`) pending a usable generator
on the GPU PC.

## Deterministic variant comparison (fairness-controlled) — RAN

All variants ran on **identical** examples (`hotpot-smoke-25`), same seed (0),
same retriever/extractor/classifier, same generator (`FakeTextGenerator`), same
budget (256). Only the intended component differs per variant → fairness holds by
construction (single shared `RunSettings`; no per-variant overrides). See
`fairness/audit.md`.

This compares **evidence-graph behavior**, not answer EM/F1 (fake generator).
Full table and the key finding (greedy connected selector halves multi-hop
recall under lexical edges) are in `reasoning-calibration/report.md` and
`pilots/hotpot-smoke-25-deterministic.json`.

**Do not read a "winner" from this.** It is a deterministic structural probe on
n=8 full-coverage examples with the lexical classifier; it intentionally does not
use answer EM as a selection criterion (the milestone forbids EM-only selection),
and the discriminating real-NLI + real-generator comparison is GPU work.

## Metrics that need a real generator (deferred, listed for the matrix)

HotpotQA: EM, token-F1, joint answer/support, citation precision/recall,
required-hop coverage via the generated answer. FEVER: label accuracy,
FEVER-score, abstention correctness, unsupported-output rate. All require a
generator that emits the valid structured contract — see `final-matrix-plan.md`.
