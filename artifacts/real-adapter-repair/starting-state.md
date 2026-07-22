# Real-adapter-repair milestone — starting state

Repo `~/dev/EGRAG` (non-iCloud), env healthy: torch 2.12.1 / transformers 5.12.1,
CPU, ~31 GiB free. HF cache has roberta-large-mnli, Qwen2.5-0.5B-Instruct,
cross-encoders, all-MiniLM-L6-v2. Source of truth = existing repo files.

## Confirmed blockers (with code locations)

1. **No chat template in structured extractor** —
   `adapters/extraction/huggingface.py::HuggingFaceStructuredModel.complete` does
   `transformers.pipeline("text-generation", model=name)(prompt, ...)` on the raw
   string. Qwen2.5-0.5B never gets its chat template → emits prose. **0/4 valid
   JSON** (`artifacts/benchmark-calibration/claim-extraction/real-extractor-smoke.json`,
   error `Expecting value: line 1 column 1`).
2. **No chat template in generator** — `generation/adapters.py::HuggingFaceGenerator`
   advertises `capabilities().chat_template=True` but `complete` feeds the raw
   prompt to the pipeline. **0/6 valid output** (`.../generator/real-generator-smoke.json`):
   JSON object then trailing prose, or multiple objects, or a JSON array.
3. **Whole-string-only JSON parsing** — `generation/parsing.py::parse_generation`
   does `json.loads(text)`; valid JSON + trailing prose → `GenerationError: Extra
   data`. Same strictness in `adapters/extraction/structured.py::_parse`.
4. **Renderer never chat** — `generation/service.py` always builds
   `PlainTextEvidenceRenderer`; `ChatEvidenceRenderer.render_messages` exists and
   keeps instructions in system / evidence in user but is unused.
5. **Components hardcoded** — `experiments/variants.py` `_run_claim`/`_run_graph`
   instantiate `SentenceClaimExtractor()` and `LexicalPairClassifier()` inline;
   `_retrieve` builds `BM25Retriever`/`SentenceAwareChunker` inline. No way to
   inject the real extractor, real NLI, caches, reranker, or token counter.
   `runner._build_generator` only allows `"fake"`.
6. **Caching not wired** — `caching/{disk,memory,keys}.py` are complete
   (`DiskCacheBackend` atomic+checksummed+quarantine; `build_cache_key`,
   `build_nli_cache_key`) but nothing in adapters/graph/generation/runner uses them.
7. **Final matrix stays blocked** until real extraction + generation pass bounded
   smokes.

## Affected files (planned minimal changes)

| File | Change |
|------|--------|
| `egrag/structured_json.py` (NEW) | shared brace/string-aware single-object JSON recovery |
| `adapters/extraction/huggingface.py` | lazy tokenizer; `apply_chat_template`; device/dtype; pad/eos; revisions; system+user chat |
| `adapters/extraction/structured.py` | use recovery util in `_parse`; record `recovered` warning |
| `adapters/extraction/caching.py` (NEW) | `CachedStructuredModel` wrapper |
| `generation/adapters.py` | `HuggingFaceGenerator`: chat template, `complete_chat`, device/dtype, honest capabilities; `CachedTextGenerator` |
| `generation/interfaces.py` | add narrow `ChatTextGenerator` protocol (`complete_chat(messages, config)`) |
| `generation/parsing.py` | use recovery util; add `recovered` diagnostic to `ParsedAnswer` |
| `generation/service.py` | capability-aware renderer: chat messages for chat generators |
| `graph/classification.py` | device/dtype/revisions on `HuggingFaceNLIClassifier`; `CachedPairClassifier` (graph/caching) |
| `experiments/variants.py` | `RunComponents` frozen dataclass; `run_system(..., components=None)` injection; defaults unchanged |
| `cli/main.py` / `cli/experiment.py` | `gpu-readiness` command; `experiment matrix --dry-run` |
| `tests/...` (NEW) | chat-template, JSON recovery, injection, caching, e2e regression |

## Current model IDs

- NLI: `roberta-large-mnli` @ `2a8f12d27941090092df78e4ba6f0928eb5eac98` (works).
- Generator/extractor candidate: `Qwen/Qwen2.5-0.5B-Instruct`.

## Current failure examples (verbatim, preserved)

- Extractor: `ExtractionError: model did not return valid JSON: Expecting value:
  line 1 column 1 (char 0)` (prose output).
- Generator direct: raw `' {"answer": "Polish", "citations": [], "uncertainty":
  ""}, where "Polish" is the answer.'` → `Extra data: line 1 column 57`.
- Generator insufficient: raw `' [{"answer":"2013",...},{}]'` (array + hallucinated
  year, no abstention).

## Constraints

Minimal changes, no architecture redesign; lazy transformers import (never at
module import); deterministic decoding preserved; no fabricated/rejected
citations; no fake generator in real eval; no JSON "repair" beyond one
unambiguous top-level object passing the full schema; no test-set tuning; frozen
configs unchanged; no commit/push; final matrix not executed.
