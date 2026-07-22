# Final-matrix environment check

Result: **BLOCKED — CUDA unavailable. The final matrix was not started.**

Step 1 of this milestone is a hard gate: if CUDA is unavailable, stop before any
model loading or benchmark execution. This machine is the development Mac, not the
GPU PC.

## Recorded state

| Field | Value |
|-------|-------|
| Date/time (UTC) | 2026-07-01T09:31:03Z |
| OS / kernel | macOS, Darwin 24.5.0, arm64 (Apple M3, `MAC-84AF0F`) |
| WSL/Linux | not applicable (native macOS, not WSL2/Linux) |
| Repository path | `/Users/nikolasroufas/dev/EGRAG` (outside iCloud/cloud-synced storage) |
| git state | working tree with one untracked file (`?? .env.example`) |
| Python | 3.14.3 (system); project venv uses 3.12 via uv |
| uv | 0.11.23 |
| torch | 2.12.1 |
| transformers | 5.12.1 |
| `nvidia-smi` | not found |
| CUDA available | **False** |
| CUDA device count | 0 |
| GPU name / VRAM / CUDA version / bf16 | not applicable (no CUDA device) |
| MPS available | True (Apple GPU; not the required CUDA target) |
| Free disk on repo volume | ~30 GiB (below the 50 GiB this milestone requires) |
| Model cache path | `~/.cache/huggingface/hub` (~5.0 GiB, writable) |
| Cached models | `Qwen2.5-0.5B-Instruct`, `roberta-large-mnli` only |
| Frozen configs / samples | present (`frozen-configs/{hotpotqa,fever}.yaml`, `checksums.json`, `samples/*.json`) |
| Prior `artifacts/final-matrix/` outputs | none (nothing to overwrite) |

## Blockers (all must clear before this milestone can run)

1. **No CUDA GPU.** `torch.cuda.is_available()` is False and `nvidia-smi` is
   absent. The milestone requires CUDA on the GPU PC (Windows + WSL2 with an
   NVIDIA GPU). MPS is present but is not the specified target, and the final
   matrix with a 7B model is not feasible on this machine.
2. **Recommended models not cached.** `Qwen2.5-7B-Instruct` (extractor + generator)
   is not in the local cache — only the 0.5B smoke model is. Using the 0.5B model
   (or any fake) for answer-quality metrics is explicitly disallowed.
3. **Insufficient free disk.** ~30 GiB free; the milestone requires at least
   50 GiB (7B weights plus the persistent cache and per-example artifacts).

## What was NOT done (deliberately)

No models were loaded, no smokes, dry-runs, pilots, or matrix runs were executed,
and no results were produced. Frozen configs were not modified. No fake or
undersized model was substituted to manufacture answer-quality numbers.

## Required to proceed

Run this milestone on the GPU PC (WSL2 + NVIDIA, CUDA-enabled torch), with
`Qwen2.5-7B-Instruct` and `roberta-large-mnli` cached and ≥50 GiB free. The exact
setup and dry-run commands are in `docs/reproduction.md` (GPU section) and
`artifacts/real-adapter-repair/gpu-readiness.md`. Re-run Step 1 there; when CUDA is
available and the smoke gates pass, continue from Step 2.
