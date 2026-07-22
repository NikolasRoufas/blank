# Real-model live smokes — summary (§4–6)

All three smokes ran on the relocated, healthy host (Apple M3, CPU, 16 GiB),
offline (`HF_HUB_OFFLINE=1`), against the existing adapters (no new framework).
Raw artifacts: `nli-calibration/real-nli-smoke.json`,
`generator/real-generator-smoke.json`, `claim-extraction/real-extractor-smoke.json`.

| Component | Model | Result | Verdict |
|-----------|-------|--------|---------|
| **NLI** | `roberta-large-mnli` @ `2a8f12d2…` | label mapping **valid**; 4/4 controlled relations correct (supports 0.98, contradicts 0.999, neutral mid, duplicate 0.994 both dirs) at E0.4/C0.7/D0.8 | ✅ **usable** |
| **Generator** | `Qwen/Qwen2.5-0.5B-Instruct` via `HuggingFaceGenerator` | **0/6** valid structured output: emits JSON-shaped text then trailing prose ("Extra data") or multiple objects; never valid citations; hallucinated "2013" instead of abstaining | ❌ **unusable as-is** |
| **Extractor** | `Qwen2.5-0.5B` via `StructuredClaimExtractor` (`extraction_v1`) | **0/4** valid JSON ("Expecting value: line 1 column 1" — emits prose, not `{"claims":…}`) | ❌ **unusable as-is** |

## Why generation fails (root cause, not the model's fault alone)

`HuggingFaceGenerator`/`HuggingFaceStructuredModel` feed the rendered prompt to
`transformers.pipeline("text-generation")` **without applying the model's chat
template**, despite `capabilities().chat_template=True`, and with **no stop
sequence / JSON constraint**. A 0.5B instruct model under a raw causal prompt
does not reliably produce a single strict-JSON object. The strict
`parse_generation`/`StructuredExtractionOutput` validators then (correctly)
reject the output — the integrity layer is working; the adapter+model pairing is
the problem.

**Determinism** held (identical raw output across repeated runs with seed 0), so
the failure is systematic, not sampling noise.

## Consequence (honest)

Real-model **end-to-end** benchmark calibration (answer EM/F1, FEVER label
accuracy, real-generation citation quality, real-NLI graph relations on benchmark
claims) is **blocked on a usable structured generator**. Per the milestone rules
the fake generator is **not** substituted to manufacture a "successful" pilot.

**Prerequisites to unblock (recommended, GPU PC):**
1. Apply the chat template in `HuggingFaceGenerator`/`HuggingFaceStructuredModel`
   (+ a stop sequence and/or JSON-constrained decoding) — a small, separately
   tested adapter change (touches `generation/adapters.py`,
   `adapters/extraction/huggingface.py`, capabilities, and tests).
2. And/or use a larger instruct model that reliably follows the JSON contract.
3. Re-run the §4–6 smokes; only then run real-model pilots/matrix.

The real **NLI** path is usable now and was already calibrated on gold claims in
the prior milestone (`scripts/run_real_nli_eval.py`, dev-only thresholds
E/C selected with a precision floor, duplicate fixed at 0.8) — see
`nli-calibration/report.md`.

## Measured latencies (CPU M3) — feed §13

- roberta-large-mnli: load+validate **24.4 s**; **~61 ms** / single-direction pair.
- Qwen2.5-0.5B generation: load ~few s; **~1.6 s** median / 64-token generation.
- Qwen2.5-0.5B extraction (256 new tokens): **15–43 s** / passage (much slower;
  longer output) — impractical for large CPU pilots.
