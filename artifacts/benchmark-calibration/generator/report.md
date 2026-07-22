# Answer generator — real-model smoke (§5)

Adapter: `HuggingFaceGenerator` (existing) driving the full generation path via
`run_system` (retrieve → extract → [graph] → select → render → generate → parse →
validate). Model `Qwen/Qwen2.5-0.5B-Instruct` (smallest cached instruct model),
deterministic, seed 0, `max_new_tokens=64`, evidence budget 256, context limit
4096, CPU. Raw: `real-generator-smoke.json`.

| Model ID | Revision | Backend | Device | Precision | Context | Evidence budget | Output limit | Prompt | Decoding |
|---|---|---|---|---|---|---|---|---|---|
| Qwen2.5-0.5B-Instruct | cached snapshot | transformers text-generation pipeline (no chat template) | cpu | fp32 | 4096 | 256 tok | 64 tok | PlainTextEvidenceRenderer (instructs strict JSON) | greedy (deterministic) |

## Smoke suite (6 controlled cases)

| Case | variant | valid structured output | raw (truncated) | latency |
|------|---------|--------------------------|------------------|---------|
| direct | claim_only_rag | ✗ Extra data | `{"answer":"Polish","citations":[],"uncertainty":""}, where "Polish"…` | 5.6 s (incl. load) |
| two_hop | full_egrag | ✗ Extra data | `{"answer":"Paris",…}, {"answer":"Paris",…}` | 1.2 s |
| yes_no | full_egrag | ✗ Extra data | `{"answer":"yes",…}, where "yes" if we can conclude…` | 1.9 s |
| fever_supports | claim_only_rag | ✗ Extra data | `{"answer":"Yes",…}, {"answer":"No","citations":["clm-…"],…}` | 1.6 s |
| fever_refutes | full_egrag | ✗ Extra data | `{"answer":"Yes",…}, {"answer":"No","citations":["clm-…"]}` | 1.5 s |
| insufficient | claim_only_rag | ✗ expected object | `[{"answer":"2013","uncertainty":"Accepted evidence"},{}]` | 0.5 s |

## Metrics (§5)

- **valid structured-output rate: 0/6 (0.0).** All malformed (trailing prose
  after the JSON object, multiple objects, or a JSON array).
- **valid citation rate: 0** — citations are either empty or hallucinated claim
  IDs (`clm-3e51897c32e77f0d`) not present in the package.
- **invalid citation rate / cited-claim existence:** not reaching attribution
  validation because parsing fails first.
- **abstention correctness:** the insufficient-evidence case did **not** abstain;
  it **hallucinated** "2013" — a faithfulness failure.
- **deterministic repeatability:** ✅ identical raw output across two runs (seed 0).
- **latency:** ~1.6 s median / 64-token generation on CPU.

The content is often semantically plausible ("Polish", "Paris", "yes"), but the
output **contract** and **grounding** fail. The strict validators correctly
reject it (no silent acceptance).

**Stop condition honored:** generator structured-output validity is clearly
insufficient → no larger generation pilot, no fake-generator substitution. See
`live-smoke/summary.md` for the root cause (no chat template / no stop sequence)
and the unblock prerequisites.
