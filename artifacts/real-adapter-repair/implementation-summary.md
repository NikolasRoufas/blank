# Implementation summary (real-adapter-repair §2–§6, §10–§11)

Minimal, targeted changes — no architecture redesign. transformers stays lazily
imported; deterministic decoding preserved; scientific-integrity rules unchanged.

## New modules

- **`src/egrag/structured_json.py`** — brace/string-aware recovery of exactly one
  top-level JSON object; rejects empty/truncated/non-object/multiple-object output.
  No naive regex. Returns `(data, recovered)`.
- **`src/egrag/hf_runtime.py`** — shared lazy `HFTextPipeline` (one load site):
  `apply_chat_template` when the tokenizer has one else a role-concatenated
  fallback; device (`cpu`/`mps`/`cuda`/int/`auto`) + optional dtype; `pad_token_id`
  fallback; `return_full_text=False`; deterministic `do_sample=False`; stop
  sequences when supported. Plus `resolve_device` and `gpu_report`.
- **`src/egrag/adapters/extraction/caching.py`** — `CachedStructuredModel`.
- **`src/egrag/graph/caching.py`** — `CachedPairClassifier` (versioned prob JSON).

## Changed modules

- **`adapters/extraction/huggingface.py`** — `HuggingFaceStructuredModel` uses
  `HFTextPipeline`; sends a system + user chat conversation when a chat template
  exists; records device/dtype/revisions.
- **`adapters/extraction/structured.py`** — `_parse` uses the recovery util and
  returns a `recovered` flag; a recovery warning is surfaced (not hidden).
- **`generation/adapters.py`** — `HuggingFaceGenerator` uses `HFTextPipeline`, adds
  `complete_chat(messages, config)` (applies chat template), honest capabilities,
  device/dtype; new `CachedTextGenerator` wrapper.
- **`generation/interfaces.py`** — new narrow `ChatTextGenerator` protocol
  (`complete_chat`).
- **`generation/parsing.py`** — `parse_generation` uses the recovery util, adds a
  `recovered` diagnostic, and strips one pair of surrounding brackets from each
  citation string (`"[c1]"` → `"c1"`; unknown ids still rejected).
- **`generation/service.py`** — capability-aware renderer: chat-capable generators
  get `ChatEvidenceRenderer` messages (instructions in system, evidence in user);
  plain generators get `PlainTextEvidenceRenderer`.
- **`generation/rendering.py`** — clarified the citation instruction + output
  contract (bare-id list, no brackets/nesting).
- **`experiments/variants.py`** — `RunComponents` frozen dataclass + `run_system(…,
  components=None)`; retriever/chunker/extractor/classifier/reranker/token-counter/
  cache injectable; deterministic defaults unchanged; shared across families.
- **`cli/main.py`** — `egrag gpu-readiness` command.
- **`cli/experiment.py`** — `egrag experiment matrix --dry-run` (plans; no
  inference; `--execute` intentionally refused this milestone).

## Root cause of the original 0/4 and 0/6 failures

Both HF adapters fed the **raw prompt** to `pipeline("text-generation")` without
applying the tokenizer chat template, so an instruction-tuned model never saw its
expected chat format and emitted prose / prose-wrapped or trailing-text JSON;
the strict whole-string parser then rejected it. Fix = apply the chat template +
recover one JSON object + clarify the citation contract. Result: extraction
0/4→4/4 valid JSON, generation 0/6→6/6 valid output (see `final-report.md`).
