# Model comparison (§9)

Only one instruct generator is cached offline: **Qwen2.5-0.5B-Instruct**. A larger
instruct model was **not** downloaded/run this milestone (network is available but
CPU inference for a comparison is slow and the milestone permits recording a GPU
recommendation instead of inventing results). So this compares the **measured**
0.5B behavior against the **recommended** larger model (clearly marked as
projected, not run).

## Qwen2.5-0.5B-Instruct — measured (CPU, chat template applied)

| Metric | Result | Target | Met? |
|--------|--------|--------|------|
| Extractor valid JSON | **4/4** (was 0/4) | ≥3/4 | ✅ |
| Extractor grounded spans | **0/4** | — | ❌ paraphrases spans → grounding rejects |
| Generator valid output | **6/6** (was 0/6) | ≥5/6 | ✅ |
| JSON recovery used (gen) | present on some cases | — | surfaced |
| Unknown citation rate | **0%** | 0% | ✅ |
| Rejected-evidence citation | **0%** | 0% | ✅ |
| Insufficient-evidence abstention | **0/1** (hallucinated "2018") | all | ❌ |
| NLI controlled relations | **4/4** | 4/4 | ✅ |
| Cold/warm cache equality | **100%** | 100% | ✅ |

The chat-template fix resolved the **format** failures (valid JSON, valid output).
The residual failures are **model faithfulness/capability**, not adapter bugs:
the 0.5B model does not copy source spans verbatim (so grounded extraction is 0)
and hallucinates a confident answer on insufficient evidence. See
`failures/qwen05b-residual-limits.md`.

## Recommended larger model — projected (run on GPU)

**Qwen2.5-7B-Instruct** (or 3B if VRAM-limited), bf16, CUDA. Expected to:
- copy source spans verbatim far more reliably → non-zero grounded extraction;
- abstain / express uncertainty on insufficient evidence (instruction-following);
- keep the already-passing format/citation behavior.

This is a projection based on general instruct-model capability scaling, **not**
a measured result — it must be verified by re-running
`artifacts/real-adapter-repair/_scripts/smoke.py` on the GPU box before the matrix.

## Verdict

Adapters/pipeline are **ready**. The **0.5B model is not sufficient** for faithful
benchmarking (fails grounded extraction and insufficient-evidence abstention).
The final matrix should use a larger instruct model on GPU, validated by the same
bounded smokes first.
