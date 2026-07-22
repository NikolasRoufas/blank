# Preserved failures — Qwen2.5-0.5B residual limits (after the adapter fix)

The chat-template + JSON-recovery fixes made the 0.5B model emit **valid**
structured output (extraction JSON 4/4, generation output 6/6). But two
model-capability failures remain and are preserved here honestly — they are
**not** adapter bugs, and were **not** masked.

## 1. Extraction: valid JSON but 0/4 grounded spans

All 4 extraction cases returned schema-valid JSON, but **every** claim was
rejected by the span-grounding check (`source_span_text` not found verbatim in
the passage) — the 0.5B model paraphrases the span instead of copying it. The
grounding rejection is **correct** (never fabricate provenance), so the honest
outcome is 0 grounded claims for this model. Example: passage "Marie Curie was a
Polish and naturalized-French physicist." → model span text drops/rewrites words
→ rejected. All 4 cases also required JSON recovery (prose wrapping).

## 2. Generation: hallucinates on insufficient evidence

Insufficient-evidence case — query "What year did Zorbix go bankrupt?" with
evidence only "Zorbix is a company mentioned in passing":

```
answer: "2018"   citations: ["c1"]   abstained: false
```

The model invented "2018" and cited c1 (a real, known claim that does **not**
support the year). Citation validation passes (c1 exists) because the check is
structural, not semantic — so a confident unsupported answer slips through for
this weak model. **Acceptance target "insufficient-evidence abstention correct in
all cases" is NOT met by the 0.5B model.**

Also observed: the yes/no case answered "Yes" but cited nothing (abstained) —
under-citation.

## Conclusion

Adapters are correct and verified; the 0.5B model is not faithful enough
(span-copying, abstention) for the benchmark matrix. Recommend a larger instruct
model on GPU (see `gpu-readiness.md`, `model-comparison.md`). No fake substituted;
no grounding/attribution rule weakened to manufacture success.
