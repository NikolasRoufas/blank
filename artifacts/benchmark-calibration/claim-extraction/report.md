# Claim extraction — real-model smoke (§4)

Adapter: `StructuredClaimExtractor` (existing) + `HuggingFaceStructuredModel`
(`Qwen/Qwen2.5-0.5B-Instruct`), prompt `extraction_v1`, deterministic, seed 0,
`max_new_tokens=256`, CPU. Raw: `real-extractor-smoke.json`.

| Model ID | Revision | Backend | Device | Precision | Prompt | Decoding | Max claims | Truncation |
|---|---|---|---|---|---|---|---|---|
| Qwen2.5-0.5B-Instruct | cached snapshot | transformers text-generation pipeline | cpu | fp32 | extraction_v1 | greedy (deterministic) | `ExtractionConfig` default | passage delimited; span verified verbatim |

## Controlled smokes (direct / multi-fact / negation / temporal)

| Case | valid JSON | grounded claims | latency |
|------|-----------|-----------------|---------|
| direct | ✗ | 0 | 18.7 s |
| multi_fact | ✗ | 0 | 14.6 s |
| negation | ✗ | 0 | 23.3 s |
| temporal | ✗ | 0 | 43.3 s |

**Required-fact claim recall, source-span validity, entity/negation/temporal
preservation, dup/empty rates: not measurable** — the model returned **0/4**
valid JSON (`ExtractionError: model did not return valid JSON: Expecting value:
line 1 column 1`), i.e. it emitted prose, not the required
`{"claims":[…]}` object. The adapter does not apply Qwen's chat template and sets
no stop/JSON constraint (see `live-smoke/summary.md`).

**Stop condition honored:** structured-output validity is clearly insufficient,
so larger extraction pilots (30 HotpotQA / 30 FEVER) were **not** run on this
model. No fake extractor was substituted. The deterministic `SentenceClaimExtractor`
remains the offline pipeline's extractor and is what the deterministic pilots use.

## Recommendation

Unblock by applying the chat template + stop sequence (small adapter change with
tests) and/or a larger instruct model on the GPU PC; then re-run this smoke
before any extraction pilot. Latency (15–43 s/passage on CPU) also makes CPU
extraction pilots impractical regardless.
