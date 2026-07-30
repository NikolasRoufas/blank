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

## Real-model smokes (CPU or MPS, small model)

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

## CUDA server setup

Target: Linux, an NVIDIA GPU (validated on an RTX 3090, 24 GB VRAM), a current
NVIDIA driver, `uv`. The server does not need models pre-cached.

```bash
# one-time environment setup
curl -LsSf https://astral.sh/uv/install.sh | sh
cd <repo>
uv sync --extra retrieval --extra dense --extra graph --extra local-models \
        --extra quantization --extra http-models --extra experiments --extra benchmarks

# point the Hugging Face cache at a disk with enough room (see "Disk sizing" below)
export HF_HOME=/path/to/large/disk/.hf_home   # or set it in the shell profile / .env

# confirm CUDA is visible (expect cuda_available: true, the GPU name, and its VRAM)
uv run egrag gpu-readiness --device cuda --dtype bfloat16
```

### Disk sizing

| Model | Download size (bf16) | VRAM (bf16) |
|---|---|---|
| `Qwen/Qwen2.5-3B-Instruct` | ~6 GB | ~6 GB |
| `Qwen/Qwen2.5-7B-Instruct` | ~15 GB | ~15 GB |
| `Qwen/Qwen3.5-9B` | ~18 GB | ~18 GB |
| `roberta-large-mnli` | ~1.4 GB | negligible |

A 24 GB GPU handles all three generators in VRAM with no quantization needed;
disk is the actual constraint on a small root volume — verify free space with
`df -h "$HF_HOME"` before downloading, and prefer a volume with 45+ GB free to
hold all three generator caches at once without deleting between models.

`Qwen/Qwen2.5-14B-Instruct` is also fully supported (`--generator-quantization
4bit`; ~29 GB download, ~8 GB VRAM quantized) if a fourth data point is wanted
later — see "Quantization" in `docs/models.md` — but it is not part of the
current A/B/C matrix.

### Download models once (online), then run offline

```bash
uv run python -c "from huggingface_hub import snapshot_download; \
  snapshot_download('Qwen/Qwen2.5-3B-Instruct'); \
  snapshot_download('Qwen/Qwen2.5-7B-Instruct'); \
  snapshot_download('Qwen/Qwen3.5-9B'); \
  snapshot_download('roberta-large-mnli', revision='2a8f12d27941090092df78e4ba6f0928eb5eac98')"

# verify a download landed and is usable
uv run python -c "from transformers import AutoTokenizer; \
  AutoTokenizer.from_pretrained('Qwen/Qwen2.5-3B-Instruct'); print('OK')"

# run everything else fully offline
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
```

### CUDA smoke test

Run immediately after provisioning, before any real experiment. Loads the
tokenizer, the generator, and the NLI model once each, runs one minimal
generation, one NLI classification, and one complete `full_egrag` example — it
does **not** run a benchmark or the experiment matrix.

```bash
uv run python scripts/cuda_smoke_test.py \
  --generator-model Qwen/Qwen2.5-3B-Instruct --device cuda --dtype bfloat16
```

Every step fails loudly with the exact exception on error (a missing dependency,
CUDA unavailable, a malformed model output, ...); it never silently skips a step
or falls back to CPU when `--require-cuda` (the default) is set. For the larger,
cross-generation third point:

```bash
uv run python scripts/cuda_smoke_test.py \
  --generator-model Qwen/Qwen3.5-9B --device cuda --dtype bfloat16 \
  --disable-thinking --max-new-tokens 256
```

## Qwen scale experiments (A/B/C)

Runs the same experimental design already used for the paper's fake-generator
pilots (`scripts/run_paper_experiments.py`'s variant list, seeds, top-k, and
evidence budget) through `egrag experiment run`, with a real Hugging Face
generator instead of the deterministic fake. Retrieval, claim extraction (the
deterministic sentence baseline), graph construction, and relation
classification (lexical) are held constant across A/B/C — **only the generator
model changes.**

- **A** — Qwen2.5-3B-Instruct, bf16.
- **B** — Qwen2.5-7B-Instruct, bf16. Same generation/architecture as A; a clean
  scale-only comparison.
- **C** — `Qwen/Qwen3.5-9B`, bf16, `enable_thinking=False`. A **different Qwen
  generation** from A/B (different architecture, training data, and default
  decoding behavior) — report it as a separate scale-and-generation comparison,
  not as a third same-family scale point.

All three have been run end to end on an RTX 3090; results and full artifacts
are under `artifacts/qwen-matrix/`.

```bash
VARIANTS="passage_rag,reranked_passage_rag,claim_only_rag,graph_no_propagation,graph_top_claim,graph_coherent_subgraph,graph_no_temporal,graph_no_contradiction,graph_with_propagation,full_egrag"

# Experiment A — Qwen2.5-3B-Instruct
for dataset in synthetic_graph temporal_conflict; do
  uv run egrag experiment run \
    --name "qwen2.5-3b_${dataset}" --dataset "$dataset" --variants "$VARIANTS" \
    --seeds 42,123,2026 --output-dir "artifacts/qwen-matrix/qwen2.5-3b-instruct/${dataset}" \
    --generator huggingface --generator-model Qwen/Qwen2.5-3B-Instruct \
    --generator-device cuda --generator-dtype bfloat16 --require-cuda \
    --top-k 3 --evidence-budget 256
done

# Experiment B — Qwen2.5-7B-Instruct
for dataset in synthetic_graph temporal_conflict; do
  uv run egrag experiment run \
    --name "qwen2.5-7b_${dataset}" --dataset "$dataset" --variants "$VARIANTS" \
    --seeds 42,123,2026 --output-dir "artifacts/qwen-matrix/qwen2.5-7b-instruct/${dataset}" \
    --generator huggingface --generator-model Qwen/Qwen2.5-7B-Instruct \
    --generator-device cuda --generator-dtype bfloat16 --require-cuda \
    --top-k 3 --evidence-budget 256
done

# Experiment C — Qwen3.5-9B, bf16 (different Qwen generation from A/B — label accordingly)
for dataset in synthetic_graph temporal_conflict; do
  uv run egrag experiment run \
    --name "qwen3.5-9b_${dataset}" --dataset "$dataset" --variants "$VARIANTS" \
    --seeds 42,123,2026 --output-dir "artifacts/qwen-matrix/qwen3.5-9b/${dataset}" \
    --generator huggingface --generator-model Qwen/Qwen3.5-9B \
    --generator-device cuda --generator-dtype bfloat16 --generator-disable-thinking --require-cuda \
    --top-k 3 --evidence-budget 256
done
```

Each run writes `manifest.json` (includes `generator_model`, `generator_revision`,
`generator_device`, `generator_dtype`, `generator_quantization`,
`generator_disable_thinking`, `generator_resolved` — the *actually*-resolved
runtime info read back from the loaded model/tokenizer — and an `environment`
block with the torch/transformers versions, CUDA runtime version, GPU name, and
VRAM, all populated only when actually used this run), `resolved_config.json`,
`results.jsonl` (per example/variant/seed), `aggregate.json`, `timing.json`,
`failures.log`, and per-example evidence packages/graphs, under its `--output-dir`.
Compare two variants with `egrag experiment compare`; summarize a run with
`egrag experiment summarize`.

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

Planning against the GPU device and a chosen model still only prints a plan — see
"CUDA server setup" and "Qwen scale experiments" above for the setup and commands
that actually execute inference:

```bash
uv run egrag experiment matrix --benchmark hotpotqa --dry-run \
  --sample artifacts/benchmark-calibration/samples/hotpot-dev-100.json \
  --output-dir artifacts/final-matrix/out --device cuda \
  --generator-model Qwen/Qwen2.5-7B-Instruct --extractor-model Qwen/Qwen2.5-7B-Instruct
```

Before executing a full matrix: the bounded smokes must pass (in particular
span-grounded extraction and abstention on insufficient evidence, which the 0.5B
model fails), the device report must show the expected GPU, and the frozen configs
must be unchanged. Note this is a **different, larger** effort than the Qwen
scale experiments above: it targets real HotpotQA/FEVER data (not yet available
offline in this repository — see `docs/benchmarks.md`) and both the real
structured extractor and real NLI classifier, not just the generator. `--execute`
remains rejected in this repository; unblocking it is out of scope for the
CUDA-server preparation covered by this document.

## Artifact layout

- `artifacts/benchmark-calibration/` — dataset and sample manifests, frozen configs
  and checksums, deterministic pilots, and the calibration reports.
- `artifacts/real-adapter-repair/` — adapter-repair reports, the smoke outputs and
  their scripts (`_scripts/`), the dry-run record, and preserved failures.
- `artifacts/cuda-smoke/` — JSON reports from `scripts/cuda_smoke_test.py` runs.
- `artifacts/qwen-matrix/` — the Qwen scale experiment (A/B/C) output directories.
