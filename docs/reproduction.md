# Reproduction

Commands assume a checkout with `uv` installed and Python 3.12. The core paths run
offline; the real-model paths read from the local Hugging Face cache.

## Quality gates

```bash
uv sync
uv run ruff format --check .
uv run ruff check .
uv run mypy src
uv run pytest
uv build
```

## Deterministic pipeline (CPU, no extras)

```bash
uv run egrag run -q "What does EG-RAG do?"
uv run egrag search -q "evidence graph"
uv run egrag reason
```

## Real-model smokes (CPU or MPS)

These need the `local-models` extra and the cached models (`roberta-large-mnli`,
`Qwen2.5-0.5B-Instruct`). They run offline:

```bash
uv sync --extra local-models --extra graph --extra experiments
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false \
  .venv/bin/python artifacts/real-adapter-repair/_scripts/smoke.py
```

This writes `real-extractor-smoke.json`, `real-generator-smoke.json`,
`real-nli-smoke.json`, and `real-e2e-smoke.json` under
`artifacts/real-adapter-repair/`.

## Device report

```bash
uv run egrag gpu-readiness --device auto            # or --device cuda --dtype bfloat16
```

## Benchmark matrix dry-run (no inference)

```bash
uv run egrag experiment matrix --benchmark fever --dry-run \
  --sample artifacts/benchmark-calibration/samples/fever-dev-100.json \
  --output-dir artifacts/final-matrix/out

uv run egrag experiment matrix --benchmark hotpotqa --dry-run \
  --sample artifacts/benchmark-calibration/samples/hotpot-dev-100.json \
  --output-dir artifacts/final-matrix/out
```

The dry-run reads the frozen config and sample manifest and prints the plan
(fingerprint, model revisions, variants, rough estimates, cache plan, output
paths). It runs no inference. `--execute` is rejected in this repository.

## GPU run (Windows + WSL2)

Run the benchmark matrix on a machine with an NVIDIA GPU. In WSL2 (Ubuntu) with a
CUDA-capable torch:

```bash
# one-time
curl -LsSf https://astral.sh/uv/install.sh | sh
cd <repo>
uv sync --extra retrieval --extra dense --extra graph --extra local-models \
        --extra http-models --extra experiments --extra benchmarks

# confirm the GPU is visible (expect cuda_available: true)
uv run egrag gpu-readiness --device auto --dtype bfloat16

# fetch the chosen models once (online)
uv run python -c "from huggingface_hub import snapshot_download; \
  snapshot_download('Qwen/Qwen2.5-7B-Instruct'); \
  snapshot_download('roberta-large-mnli', revision='2a8f12d27941090092df78e4ba6f0928eb5eac98')"

# re-run the smokes on GPU and require them to pass before the matrix
HF_HUB_OFFLINE=1 uv run python artifacts/real-adapter-repair/_scripts/smoke.py

# plan with the GPU device and the chosen model
uv run egrag experiment matrix --benchmark hotpotqa --dry-run \
  --sample artifacts/benchmark-calibration/samples/hotpot-dev-100.json \
  --output-dir artifacts/final-matrix/out --device cuda \
  --generator-model Qwen/Qwen2.5-7B-Instruct --extractor-model Qwen/Qwen2.5-7B-Instruct
```

Before executing a full matrix: the bounded smokes must pass (in particular
span-grounded extraction and abstention on insufficient evidence, which the 0.5B
model fails), the device report must show the expected GPU, and the frozen configs
must be unchanged.

## Artifact layout

- `artifacts/benchmark-calibration/` — dataset and sample manifests, frozen configs
  and checksums, deterministic pilots, and the calibration reports.
- `artifacts/real-adapter-repair/` — adapter-repair reports, the smoke outputs and
  their scripts (`_scripts/`), the dry-run record, and preserved failures.
