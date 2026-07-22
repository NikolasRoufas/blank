# Runtime & Resource Estimate

## Live measurement outcome: BLOCKED (disk / I/O)

The bounded timing smoke (load `roberta-large-mnli` + `Qwen2.5-0.5B-Instruct`,
time one NLI batch and one generation) **could not complete**. The Python process
wedged at ~0.1% CPU during **model load** (disk I/O wait), and free disk fell from
**8.0 GiB → 4.5 GiB** during the attempt before it was killed. This reproduces the
host's intermittent `TimeoutError [Errno 60]` under load.

**Conclusion:** the real-model pilot is **not safely runnable in this environment
right now.** Loading a 1.3 GB (NLI) + 0.95 GB (generator) model pair on CPU with
<5 GiB free disk thrashes/wedges. This is a section-12 stop condition
("runtime clearly infeasible; disk space insufficient"); no live pilot was run.

## Owner actions to unblock (then re-run the estimator)

1. Free disk to a comfortable margin (target ≥ 25–30 GiB free). Safe levers
   identified earlier: `uv cache prune`; remove unused cached models
   (`models--Qwen…` is needed for the generator — keep it; `nli-deberta-v3-large`
   1.6 GB and other unused models/datasets are removable); clear `~/Library/Caches`
   (11 GB, general macOS).
2. For HotpotQA: install a parquet reader (`pyarrow`/`datasets`) with network.
3. Re-run: `PYTHONPATH=src .venv/bin/python scripts/estimate_runtime.py` (added).

## Component estimates (from prior validated runs + model sizes)

These are **estimates**, clearly labelled, since live timing was blocked. Prior
offline real-NLI runs (controlled suite; hard-negative measurement) executed
successfully when the disk had headroom, giving the NLI figures below.

| stage | estimate (CPU, warm model) | notes |
|---|---|---|
| `roberta-large-mnli` load | ~20–60 s (healthy disk); **wedges < 5 GiB free** | dominant cold cost |
| NLI per pair (batch 16, len 256) | ~0.1–0.3 s | from prior controlled runs |
| `Qwen2.5-0.5B-Instruct` load | ~20–60 s (healthy disk) | 0.95 GB cached |
| generation (48 new tokens, deterministic) | ~3–10 s/example | small model, CPU |
| claim extraction per passage | ~3–10 s/passage | same model class (HF causal LM) |

## Projected FEVER pilot (gold-evidence setting)

Per FEVER example: ~1–3 evidence sentences → few claims → ~1–6 NLI candidate
pairs + 1 generation. Dominant per-example cost is generation.

- Per example (warm): ~5–15 s.
- 25 examples × 8 variants, **with shared extraction/NLI cache** (variants reuse
  cached intermediates; generation differs per variant): ≈ **15–60 min warm**,
  plus one-time model loads (~1–2 min) — **assuming the disk allows model load.**
- Cold cache / cold model adds the load costs above and is the part that currently
  wedges.

## Projected full test-matrix (NOT to be run now)

Full FEVER `valid` (15,935) × 8 variants at ~5–15 s/example is **~18–66 hours**
of generation on this CPU even with shared NLI/extraction caches — clearly a job
for a machine with a GPU and ample disk, not this host. HotpotQA is additionally
blocked on the parquet reader.

## Disk / cache growth

- Persistent caches (NLI/extraction/generation) grow with example count; FEVER
  pilot intermediate cache is small (text-keyed JSON). The binding constraint is
  **model weights + scratch during load**, not the result cache.
