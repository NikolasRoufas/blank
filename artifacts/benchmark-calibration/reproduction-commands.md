# Reproduction commands

Repo: `~/dev/EGRAG` (relocated out of iCloud). Network available; HF model cache
at `~/.cache/huggingface/hub` (roberta-large-mnli, Qwen2.5-0.5B, MiniLM, cross-encoders).

## Environment

```bash
cd ~/dev/EGRAG
uv sync \
  --extra retrieval --extra dense --extra graph --extra local-models \
  --extra http-models --extra experiments --extra benchmarks --extra docs
.venv/bin/python -c "import egrag; print('ok')"          # ~0.03 s
```

## Quality gates

```bash
uv run ruff format --check .
uv run ruff check .
uv run mypy src
uv run pytest -q --no-cov \
  tests/integration/test_benchmark_calibration.py \
  tests/integration/test_benchmark_pipeline.py
uv run pytest -q --no-cov -m requires_benchmarks      # HotpotQA live parquet load
uv build
```

## HotpotQA live load + samples (§3)

```bash
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
.venv/bin/python scripts_scratch/hotpot_load.py     # see artifacts; 7405 rows, sha 78933c0a…
```

## Real-model smokes (§4–6) — offline

```bash
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
# NLI: label-mapping valid; 4/4 controlled relations correct
# Generator: 0/6 valid structured output (Qwen2.5-0.5B, no chat template)
# Extractor: 0/4 valid JSON
# (scripts in scratchpad; outputs saved under artifacts/benchmark-calibration/{nli-calibration,generator,claim-extraction})
```

The reusable real-NLI mechanism eval (dev-only thresholds, gold claims):

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH=src \
  .venv/bin/python scripts/run_real_nli_eval.py
```

## Deterministic structural pilots

```bash
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
# FEVER smoke-25 (uninformative for graph: 0 edges) and HotpotQA smoke-25
# (discriminating: greedy-connected selector halves recall under lexical edges)
# outputs: artifacts/benchmark-calibration/pilots/*.json
```

## Final matrix (GPU PC — do NOT run on this Mac)

After the prerequisites in `final-matrix-plan.md` (usable generator; wire real
adapters + cache into the runner; optionally HotpotQA distractor split):

```bash
CUDA_VISIBLE_DEVICES=0 uv run egrag experiment run <frozen-config>
```
