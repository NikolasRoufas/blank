# NLI smoke evaluation — BLOCKED (runtime unavailable)

Date: 2026-06-25.

The real-NLI smoke evaluation (and the full end-to-end real-NLI run) **could not
be executed** because the `local-models` runtime is not installed and cannot be
installed in this environment.

## What is available vs. missing

| component | status |
|---|---|
| NLI model weights | **present** — `roberta-large-mnli` is fully cached at `~/.cache/huggingface/hub/models--roberta-large-mnli` (config.json, `model.safetensors`, tokenizer files); also cached: `cross-encoder/nli-deberta-v3-large`, `cross-encoder/nli-MiniLM2-L6-H768` |
| `transformers` | **not installed** |
| `torch` | **not installed** |
| network / PyPI / HF access | **unavailable** (project policy is offline/CPU/no-downloads; `uv pip install --offline transformers torch` cannot resolve a compatible stack) |
| disk free | 7.8 GiB (tight for a torch CPU stack) |

The model is cached, but without `transformers`+`torch` it cannot be loaded.
`uv pip install --offline transformers torch` fails to resolve (no network).

## Why we stopped here (per the task's section 2)

> "If downloading is not allowed or impossible, stop and provide the exact owner
> action required."

Installing `transformers`+`torch` is a network/PyPI operation that the offline
policy forbids and the environment blocks. We therefore did **not** fabricate any
NLI scores, accepted relations, or metrics. No smoke outputs (logits, labels,
accepted/rejected relations, metrics, runtime) exist because no inference ran.

## Exact owner action required to unblock

1. In an environment with network access (or a populated wheel cache), install the
   extra: `uv sync --extra local-models` (installs `transformers>=4.40`, `torch>=2.0`).
2. Confirm the cached model is usable (already present): `roberta-large-mnli`
   (revision `2a8f12d27941090092df78e4ba6f0928eb5eac98`; MNLI labels
   0=CONTRADICTION, 1=NEUTRAL, 2=ENTAILMENT).
3. Run the (already-implemented, ready) commands in `../final-report.md` →
   "Exact reproduction commands". The label-mapping validation and smoke harness
   will then run against the real model.

Approximate requirements: torch CPU wheel ≈ 190 MB, transformers ≈ 10 MB; the
model is already cached (~1.4 GB for roberta-large; ~90 MB for the MiniLM
cross-encoder if a smaller model is preferred). CPU inference is feasible on the
available Apple M3 (no GPU/CUDA).
