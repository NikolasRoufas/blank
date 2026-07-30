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
transformers version, CUDA/MPS availability (plus CUDA runtime version, GPU name,
VRAM), the Hugging Face cache directory, and free disk.

Setting `require_cuda: true` (config) / `--require-cuda` (CLI) on the generation
section makes CUDA mandatory: if CUDA is unavailable when the model would load,
construction raises immediately rather than silently continuing on CPU. This is
off by default so unit tests and CPU development are unaffected.

### Quantization

The `huggingface` generator adapter accepts `quantization: "none" | "4bit" | "8bit"`
(config) / `--generator-quantization` (CLI), backed by `bitsandbytes` (the
`quantization` extra). This is always explicit: the resolved quantization mode is
recorded in the run/experiment manifest (`generator_quantization`,
`generator_resolved`) alongside the resolved dtype and device, so a quantized run
is never reported as, or conflated with, an unquantized one. 4-bit and 8-bit loads
require `device_map="auto"`, which in turn requires `accelerate` (bundled with the
`local-models` extra).

### Hybrid-reasoning / "thinking" models

Some newer instruction-tuned models (e.g. the Qwen3.5 family) default to emitting
an interleaved reasoning trace (`<think>...</think>`) before the final answer. Left
on, this breaks the strict single-JSON-object output contract this project
enforces (`egrag.structured_json.recover_json_object`) unless the generation
budget is large enough to contain the full trace. `chat_template_kwargs` on the
adapter (`--generator-disable-thinking` on the CLI) passes `enable_thinking=False`
through to `tokenizer.apply_chat_template(...)` — a standard, model-family-agnostic
chat-template flag, not a Qwen-specific hack. Models without this concept simply
ignore an unset flag; do not enable it for models that do not support it, since an
unrecognized `apply_chat_template` kwarg raises rather than being silently ignored.

## Recommended models

For this project's runs, not as general claims:

- NLI: `roberta-large-mnli`, revision `2a8f12d27941090092df78e4ba6f0928eb5eac98`.
- Extraction and generation, adapter smoke tests: `Qwen2.5-0.5B-Instruct` on CPU.
- Extraction and generation, primary Qwen-scale matrix (all Qwen2.5, same
  generation/architecture, bfloat16 on CUDA): `Qwen2.5-3B-Instruct` and
  `Qwen2.5-7B-Instruct`. Both fit an RTX-3090-class 24 GB GPU comfortably
  alongside the NLI model, with no quantization required.
- Larger scale point (Experiment C): `Qwen/Qwen3.5-9B`, bf16, fits 24 GB VRAM
  unquantized. It is a **different Qwen generation** from A/B (different
  architecture — includes Gated Delta Networks and a vision-language head —
  different training data, and "thinking mode" decoding by default, requiring
  `--generator-disable-thinking` and a larger `max_new_tokens` budget than the
  3B/7B runs to reliably close its JSON output). Report it as a genuine
  cross-generation comparison, not a pure scale ablation — never presented as
  same-family scaling alongside the Qwen2.5 A/B runs. This substitution is not
  invented: `Qwen/Qwen3.5-9B` is a real, verified Hugging Face identifier; there
  is no Qwen release at 6B or at a plain 9B within the Qwen2.5 generation.
  `Qwen2.5-14B-Instruct` (14.7B params, ~29 GB in bf16, needs
  `--generator-quantization 4bit` to fit 24 GB) remains fully supported as an
  alternative or additional point but is not part of the current A/B/C matrix.

## The 0.5B limitation

With the chat template applied, `Qwen2.5-0.5B-Instruct` produces valid structured
output. It has two problems that make it unsuitable for evaluation: it paraphrases
source spans instead of copying them (so span-grounded extraction is rejected), and
it answers confidently when the evidence does not support an answer instead of
abstaining. These are model-capability issues, not adapter issues; the fix is a
larger instruct model, re-validated with the smoke checks before any benchmark run.
Details in `artifacts/real-adapter-repair/`.
