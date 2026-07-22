# Reproduction commands (real-adapter-repair)

Repo `~/dev/EGRAG`; venv synced with all extras (see benchmark-calibration
reproduction). Offline env vars: `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1`.

## Quality gates

```bash
cd ~/dev/EGRAG
uv run ruff format --check .
uv run ruff check .
uv run mypy src
uv run pytest
uv build
```

## Focused milestone tests

```bash
uv run pytest -q --no-cov \
  tests/unit/test_structured_json.py \
  tests/unit/test_hf_runtime.py \
  tests/unit/test_chat_renderer_selection.py \
  tests/unit/test_adapter_caching.py \
  tests/integration/test_component_injection.py \
  tests/integration/test_real_adapter_regression.py
```

## Core-only import isolation (fresh interpreter)

```bash
uv run pytest -q --no-cov \
  tests/unit/test_import_isolation.py::test_import_does_not_initialize_optional_models \
  tests/unit/test_retrieval_isolation.py::test_importing_retrieval_adapters_loads_no_optional_libs \
  tests/unit/test_packaging_hardening.py::test_core_only_runs_fake_pipeline_without_optional_libs
```

## Bounded real-model smokes (Qwen2.5-0.5B + roberta-large-mnli, CPU)

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false \
  .venv/bin/python artifacts/real-adapter-repair/_scripts/smoke.py
# writes real-extractor-smoke.json, real-generator-smoke.json,
#        real-nli-smoke.json, real-e2e-smoke.json
```

## GPU readiness

```bash
uv run egrag gpu-readiness --device auto           # or --device cuda --dtype bfloat16
```

## Final-matrix dry-run (no inference; --execute is refused)

```bash
uv run egrag experiment matrix --benchmark fever --dry-run \
  --sample artifacts/benchmark-calibration/samples/fever-dev-100.json \
  --output-dir artifacts/final-matrix/out --device auto

uv run egrag experiment matrix --benchmark hotpotqa --dry-run \
  --sample artifacts/benchmark-calibration/samples/hotpot-dev-100.json \
  --output-dir artifacts/final-matrix/out --device auto
```
