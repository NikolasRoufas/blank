# GPU readiness (real-adapter-repair §10)

All real adapters take an explicit device (`cpu` / `mps` / `cuda` / `cuda:N` /
integer / `auto`) and an optional dtype (`float32` / `float16` / `bfloat16` /
`auto`). `auto` resolves deterministically: **CUDA → MPS → CPU**. No CUDA-only
imports; torch is imported lazily. Command: `egrag gpu-readiness [--device …] [--dtype …]`.

## This host (Apple M3, dev machine)

```json
{
  "requested_device": "auto",
  "torch": "2.12.1",
  "cuda_available": false,
  "cuda_device_name": null,
  "mps_available": true,
  "selected_device": "mps",
  "model_dtype": "float32 (torch default)",
  "hf_cache_dir": "/Users/nikolasroufas/.cache/huggingface/hub",
  "hf_cache_exists": true,
  "free_disk_gib": 28.5
}
```

## Recommended GPU run (final matrix)

- **Device:** `cuda` (or `auto` on the GPU box).
- **dtype:** `bfloat16` (Ampere+); `float16` on older cards.
- **Generator model:** upgrade from Qwen2.5-0.5B — recommend **Qwen2.5-7B-Instruct**
  (or 3B-Instruct if VRAM-limited). The 0.5B model produces valid JSON but is not
  faithful enough (see `model-comparison.md`): it does not reliably copy source
  spans verbatim (extraction grounding) and hallucinates on insufficient evidence.
- **Extractor model:** same larger instruct model, or a dedicated extraction model.
- **NLI:** `roberta-large-mnli` @ `2a8f12d2…` (already validated; works on CPU/GPU).

## Exact commands for the future GPU PC (Windows / WSL2)

WSL2 (Ubuntu) with an NVIDIA GPU + recent driver (CUDA-enabled torch):

```bash
# In WSL2:
git clone <repo> ~/dev/EGRAG && cd ~/dev/EGRAG   # or copy the tree
curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync --extra retrieval --extra dense --extra graph --extra local-models \
        --extra http-models --extra experiments --extra benchmarks

# Verify CUDA is visible (expect cuda_available: true, a device name):
uv run egrag gpu-readiness --device auto --dtype bfloat16

# Pre-download the chosen models (once, online):
uv run python -c "from huggingface_hub import snapshot_download; \
  snapshot_download('Qwen/Qwen2.5-7B-Instruct'); \
  snapshot_download('roberta-large-mnli', revision='2a8f12d27941090092df78e4ba6f0928eb5eac98')"

# Re-run the bounded smokes on GPU BEFORE the matrix (must pass acceptance):
HF_HUB_OFFLINE=1 uv run python artifacts/real-adapter-repair/_scripts/smoke.py

# Plan the matrix (dry-run) with the GPU device + chosen model:
uv run egrag experiment matrix --benchmark hotpotqa --dry-run \
  --sample artifacts/benchmark-calibration/samples/hotpot-dev-100.json \
  --output-dir artifacts/final-matrix/out --device cuda \
  --generator-model Qwen/Qwen2.5-7B-Instruct --extractor-model Qwen/Qwen2.5-7B-Instruct
```

Native Windows (PowerShell) is possible but WSL2 is recommended for transformers
+ CUDA. On native Windows use `py -m` / `uv` equivalents and set
`$env:HF_HUB_OFFLINE=1`.

Note: the matrix `--execute` path is intentionally disabled in this milestone;
execution belongs to the dedicated final-experiment milestone.
