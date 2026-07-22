# Fairness audit (§8)

The deterministic variant comparison is fair by construction:

- **Same examples:** all variants ran on the identical `hotpot-smoke-25` ids (and
  the identical FEVER smoke ids), in the same order.
- **Same shared settings:** one `RunSettings(top_k, evidence_token_budget,
  reserved_output_tokens, chunk_size, chunk_overlap)` is passed to every variant;
  no per-variant `generator_override`/`top_k_override`/`evidence_budget_override`
  was set (those fields exist only so `experiments/fairness.check_fairness` can
  *detect* an unfair difference).
- **Same components:** identical BM25 retriever, `SentenceClaimExtractor`,
  `LexicalPairClassifier`, and `FakeTextGenerator` across variants; only the
  intended toggle (family / selection / propagation / contradiction / temporal)
  differs.
- **Same seed:** 0 everywhere; deterministic decoding; runs are byte-reproducible.

**Unintended differences that would fail fairness:** none detected for the
deterministic pilot.

**Known fairness caveat for any future real-model run:** C1/C2/C3 intentionally
vary generator/budget/top-k, so a C1-vs-C2-vs-C3 comparison is **not** a clean
single-component ablation — it is a candidate-configuration comparison and must
be reported as such (not as a component ablation). The 8-variant matrix
(`final-matrix-plan.md`) is the clean single-component ablation and must share
one frozen generator + settings across variants.
