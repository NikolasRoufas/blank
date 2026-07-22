# Chat-template tests (§8)

`tests/unit/test_hf_runtime.py` (6 tests, fake pipeline+tokenizer — no model
download, no transformers required):

- `test_chat_template_applied_when_available` — `apply_chat_template` is called
  with the exact system+user messages and the templated string is what reaches
  the pipeline.
- `test_fallback_when_no_chat_template` — no template → role-concatenated prompt
  with `SYSTEM:`/`USER:` sections ending in `ASSISTANT:`; template not used.
- `test_deterministic_mode_does_not_sample` — `do_sample=False`, no `temperature`,
  `return_full_text=False`.
- `test_sampling_mode_sets_temperature` — sampling path sets temperature/top_p.
- `test_raw_prompt_path_passes_prompt_through` — raw prompt path; `return_full_text=False`.
- `test_pad_token_falls_back_when_present` — `pad_token_id` passed through.

Capability-reporting matches behavior: `HuggingFaceGenerator.capabilities().chat_template`
reflects the `apply_chat_template` flag, and the service only takes the chat path
for a generator that both reports it and implements `complete_chat`
(`tests/unit/test_chat_renderer_selection.py`, 2 tests: system/user role
separation verified; plain generator uses the plain path).

Live confirmation (Qwen2.5-0.5B, CPU): with the chat template applied, structured
extraction produced **4/4** valid JSON (was 0/4) and generation **6/6** valid
output (was 0/6) — see `real-extractor-smoke.json` / `real-generator-smoke.json`.
