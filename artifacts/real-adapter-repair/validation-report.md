# Validation report (real-adapter-repair)

## Scientific-integrity rules — preserved (§7)

- **No fabricated citations** — `validate_attribution` still rejects unknown ids;
  bracket-stripping only maps `[c1]`→`c1` (an id shown in the evidence), it never
  invents one; unknown ids after stripping are still rejected (tested).
- **No citing rejected evidence** — still raises (regression test).
- **No hidden outside knowledge** — instructions still say "use ONLY supplied
  evidence"; evidence stays untrusted and delimited (chat path keeps it in the
  user role, instructions in the system role).
- **Contradictions not dropped** — unchanged; unresolved conflict still surfaces
  uncertainty (regression test).
- **No fake generator in real eval** — the smokes use the real models; the fake
  generator is only for deterministic tests. No fake substituted for a failed real
  model; the 0.5B faithfulness failures are reported, not hidden.
- **No answer-metric claims from fake runs** — none made.
- **No test-set tuning / no frozen-config edits from results** — frozen configs
  untouched; smokes use controlled examples, not benchmark test data.
- **JSON recovery does not mask failure** — accepts only exactly one unambiguous
  top-level object that passes the full schema; malformed/ambiguous/truncated/
  multi-object output still fails (tested). Recovery is surfaced as a diagnostic.
- **Insufficient evidence** — the service refuses to generate with no selected
  evidence; and the 0.5B model's hallucinated confident answer is recorded as a
  failure (not accepted as success).
- **Lazy imports** — transformers/torch never imported at module import; core-only
  import-isolation tests pass in a fresh interpreter.

## Gate results

| Gate | Result |
|------|--------|
| `ruff format --check` / `ruff check` | PASS |
| `mypy src` | PASS (104 files) |
| `pytest` (full) | **441 passed, 7 skipped, 0 failed** (87% cov) |
| `uv build` | PASS (wheel + sdist) |
| new milestone tests (chat/json/injection/cache/e2e) | 41 passed |
| core-only isolation (fresh interpreter) | PASS |
| real NLI / extractor / generator smokes | RAN (NLI 4/4; extractor 4/4 valid JSON; generator 6/6 valid) |
| final-matrix dry-run | RAN (no inference); `--execute` refused |

## Acceptance targets (§9)

Met: extractor valid JSON 4/4 (≥3/4), generator valid 6/6 (≥5/6), unknown-citation
0%, rejected-citation 0%, NLI 4/4, cold/warm equality 100%. **Not met by the 0.5B
model:** grounded-span extraction (0/4 — paraphrased spans) and insufficient-evidence
abstention (hallucinated). These are model-capability limits → larger model on GPU
(`model-comparison.md`, `gpu-readiness.md`). Adapters/pipeline themselves pass.
