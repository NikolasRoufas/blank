# Real-adapter-repair milestone — Final report

Repo `/Users/nikolasroufas/dev/EGRAG` (non-iCloud). Nothing committed/pushed
(not a git repo). Final matrix **not** executed. No paper/LaTeX.

## Files changed

New: `src/egrag/structured_json.py`, `src/egrag/hf_runtime.py`,
`src/egrag/adapters/extraction/caching.py`, `src/egrag/graph/caching.py`;
tests `test_structured_json.py`, `test_hf_runtime.py`,
`test_chat_renderer_selection.py`, `test_adapter_caching.py`,
`test_component_injection.py`, `test_real_adapter_regression.py`; artifacts under
`artifacts/real-adapter-repair/`.
Changed: `adapters/extraction/huggingface.py`, `adapters/extraction/structured.py`,
`generation/adapters.py`, `generation/interfaces.py`, `generation/parsing.py`,
`generation/service.py`, `generation/rendering.py`, `experiments/variants.py`,
`cli/main.py`, `cli/experiment.py`, `generation/__init__.py`, plus the
`test_hotpotqa_blocked_cleanly` skip carried from the prior milestone.
(See `implementation-summary.md`.)

## Root cause of the original 0/4 and 0/6 failures

Both HF adapters passed the **raw prompt** straight to
`pipeline("text-generation")` — the tokenizer **chat template was never applied**,
so an instruction-tuned model (Qwen2.5-0.5B) got an unexpected format and produced
prose or prose-wrapped/trailing-text JSON, which the strict whole-string parser
rejected. Not a model-only problem: the adapter format was wrong.

## Chat-template implementation

A shared lazy `HFTextPipeline` applies `tokenizer.apply_chat_template(messages,
tokenize=False, add_generation_prompt=True)` when the tokenizer provides one, and
otherwise falls back to a role-concatenated prompt. Extraction sends a system+user
conversation; generation sends `ChatEvidenceRenderer` messages via
`complete_chat`. `return_full_text=False`; `do_sample=False` in deterministic
mode; `pad_token_id` falls back to `eos_token_id`; device (cpu/mps/cuda/int/auto)
and optional dtype; stop sequences when supported. Capabilities report chat
support honestly, and the service only takes the chat path for a generator that
reports it *and* implements `complete_chat`.

## JSON recovery rules

`recover_json_object`: strict whole-string parse first; else extract exactly one
balanced top-level `{…}` (brace- and string-aware, no naive regex) that parses to
an object. Rejects empty, truncated, non-object, and ≥2 competing objects.
Recovery is surfaced (`ParsedAnswer.recovered`; extraction warning), never
masked. Citations are additionally normalized by stripping one surrounding
bracket pair (`"[c1]"`→`"c1"`); unknown ids still rejected.

## Renderer selection

Capability-aware: chat generators → `ChatEvidenceRenderer.render_messages`
(instructions in system, untrusted evidence in user); plain generators →
`PlainTextEvidenceRenderer`. Provider message formatting stays inside the adapter.

## Injected components

Frozen `RunComponents` + `run_system(…, components=None)`: retriever/chunker/
extractor/classifier/reranker/token-counter/cache injectable; defaults unchanged;
same instances shared across families; gold data never a component.

## Cache wiring and key coverage

`CachedStructuredModel`, `CachedPairClassifier`, `CachedTextGenerator` over the
existing `DiskCacheBackend`. Keys cover content, model, model/tokenizer revision,
prompt version, schema version, decoding (seed/temperature/top-p/max tokens/stop),
truncation/length, and NLI thresholds/label-mapping. Cold==warm; corrupt→miss;
disabled→no writes; miss on revision/prompt/threshold change (all tested).
(`cache-report.md`.)

## Real-model smoke metrics (Qwen2.5-0.5B, roberta-large-mnli, CPU)

| Metric | Before | After |
|--------|--------|-------|
| Extractor valid JSON | 0/4 | **4/4** |
| Extractor grounded spans | — | 0/4 (paraphrased → grounding rejects) |
| Generator valid output | 0/6 | **6/6** |
| Unknown-citation rate | — | **0%** |
| Rejected-citation rate | — | **0%** |
| Insufficient-evidence abstention | — | 0/1 (hallucinated "2018") |
| NLI controlled relations | 4/4 | **4/4** |
| E2E (injected real ext+NLI+gen) | n/a | 3/3 complete; **cold/warm cache equal** |

## 0.5B vs larger model

Only 0.5B is cached; a larger model was not run (recorded, not invented).
The chat-template fix resolved all **format** failures; residual **faithfulness**
failures (span-copying, abstention) are 0.5B capability limits. Recommend
**Qwen2.5-7B-Instruct** (or 3B), bf16, CUDA, re-validated by the same smokes.
(`model-comparison.md`.)

## Remaining limitations

- 0.5B is not faithful enough for the matrix (grounded extraction 0/4;
  hallucinates on insufficient evidence).
- Citation validation is structural (id existence), not semantic support —
  grounding verifier remains heuristic.
- Estimates in the matrix dry-run are rough CPU projections.

## GPU-ready recommendation

`egrag gpu-readiness` (here: torch 2.12.1, cuda=false, mps=true → selected mps,
28.5 GiB free). For the matrix: CUDA + bf16 + Qwen2.5-7B-Instruct; WSL2 commands
in `gpu-readiness.md`.

## Dry-run command for the final matrix

```bash
uv run egrag experiment matrix --benchmark hotpotqa --dry-run \
  --sample artifacts/benchmark-calibration/samples/hotpot-dev-100.json \
  --output-dir artifacts/final-matrix/out --device cuda \
  --generator-model Qwen/Qwen2.5-7B-Instruct --extractor-model Qwen/Qwen2.5-7B-Instruct
```

## Quality-gate results

ruff format/check PASS · mypy PASS (104) · pytest **441 passed / 7 skipped / 0
failed** (87%) · uv build PASS · core-only isolation (fresh interpreter) PASS ·
smokes ran · dry-run ran (no inference).

## Readiness

**Adapters and infrastructure are READY** for the final experiment prompt: chat
template, JSON recovery, capability-aware rendering, component injection, and
persistent caching all work and are tested, and the real pipeline runs end-to-end
with caching. The **one blocker for producing trustworthy matrix numbers is the
model**: select a larger instruct generator/extractor on GPU and re-pass the
bounded smokes (especially grounded extraction + insufficient-evidence abstention)
before executing the matrix.
