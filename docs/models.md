# Models

Model adapters are optional and isolated. Provider libraries (transformers, torch,
httpx) are imported lazily inside the adapter, never at module import, and never in
the domain layer.

## Interfaces

- Answer generation: `egrag.generation.interfaces.TextGenerator` (`complete`) and
  `ChatTextGenerator` (adds `complete_chat`).
- Claim extraction: `egrag.adapters.extraction.interfaces.StructuredModel` (a raw
  prompt in, raw text out), consumed by `StructuredClaimExtractor`.
- Relation classification: `egrag.graph.types.PairClassifier`.

## Fake models

`egrag.fakes` and `egrag.generation.FakeTextGenerator` give a deterministic path
with no dependencies. The fake generator reads claim IDs from the rendered evidence
and emits the JSON output contract, so the full pipeline and the experiment harness
run offline. Fakes are for tests and structural pilots; they are not used to
produce answer-quality results.

## Hugging Face adapters

`HuggingFaceGenerator` (generation) and `HuggingFaceStructuredModel` (extraction)
share `egrag.hf_runtime.HFTextPipeline`, which:

- loads the tokenizer and pipeline once, lazily;
- applies `tokenizer.apply_chat_template(...)` when the tokenizer defines one, and
  otherwise falls back to a plain role-concatenated prompt;
- uses `return_full_text=False` and sets `pad_token_id` from `eos_token_id` when it
  is missing;
- keeps decoding deterministic (`do_sample=False`) unless sampling is requested;
- passes stop sequences through when the pipeline supports them.

For extraction the prompt is sent as a system instruction plus a user payload; for
generation the `ChatEvidenceRenderer` places framework instructions in the system
role and untrusted evidence in the user role. `GenerationService` chooses the chat
path only for a generator that reports `chat_template` and implements
`complete_chat`; otherwise it uses the plain-text renderer.

## Structured output

Model output is untrusted. `egrag.structured_json.recover_json_object` first tries
a strict whole-string parse, and otherwise extracts exactly one balanced top-level
JSON object (tracking string state, so braces inside strings are ignored). It
rejects empty, truncated, non-object, and multiple-object output. Recovery is
recorded (`ParsedAnswer.recovered`, and an extraction warning) rather than hidden.
Citations are validated against the known claim IDs; a bracketed citation such as
`"[c1]"` is normalized to `c1` (the form shown in the evidence), and an unknown ID
is still rejected.

## Devices and precision

Adapters accept `device` (`cpu` / `mps` / `cuda` / `cuda:N` / integer / `auto`) and
an optional `dtype` (`float32` / `float16` / `bfloat16` / `auto`). `auto` resolves
CUDA → MPS → CPU. `egrag gpu-readiness` prints the resolved device, torch version,
CUDA/MPS availability, the Hugging Face cache directory, and free disk.

## Recommended models

For this project's runs, not as general claims:

- NLI: `roberta-large-mnli`, revision `2a8f12d27941090092df78e4ba6f0928eb5eac98`.
- Extraction and generation, final runs: `Qwen2.5-7B-Instruct` (or 3B under VRAM
  pressure), bfloat16 on CUDA.
- Extraction and generation, adapter smoke tests: `Qwen2.5-0.5B-Instruct` on CPU.

## The 0.5B limitation

With the chat template applied, `Qwen2.5-0.5B-Instruct` produces valid structured
output. It has two problems that make it unsuitable for evaluation: it paraphrases
source spans instead of copying them (so span-grounded extraction is rejected), and
it answers confidently when the evidence does not support an answer instead of
abstaining. These are model-capability issues, not adapter issues; the fix is a
larger instruct model, re-validated with the smoke checks before any benchmark run.
Details in `artifacts/real-adapter-repair/`.
