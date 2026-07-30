# EG-RAG Real-Benchmark Results (FEVER + HotpotQA)

Generated from real, completed experiment runs (12 full passes: 3 Qwen generators
× 2 benchmarks × 2 generation-token budgets, 25 examples × 10 system variants ×
1 seed each = 250 example-runs per pass, 3,000 total). Every number below is
read directly from `results.jsonl`/`aggregate.json`/`manifest.json` under
`artifacts/benchmark-matrix/`, computed by `scripts/analyze_benchmark_results.py`.
Nothing here is estimated. Where something could not be computed (e.g. a
p-value), that is stated explicitly rather than invented.

This document extends (does not replace) `artifacts/qwen-matrix/ACL_REPORT.md`,
which covers the earlier synthetic-fixture-only 3B/7B/Qwen3.5-9B comparison.

## Hardware

Identical to the earlier report — same session, same GPU, re-verified from
these runs' own manifests:

| Field | Value |
|---|---|
| GPU | NVIDIA GeForce RTX 3090 |
| VRAM | 23.6 GiB |
| CUDA runtime version | 13.0 |
| PyTorch | 2.12.1+cu130 |
| Transformers | 5.12.1 |

## Experiment configuration

Reproduces the pre-committed, dev-frozen calibration configs
(`artifacts/benchmark-calibration/frozen-configs/{fever,hotpotqa}.yaml`, written
before any real generator existed) as closely as possible, now that real
generators are available:

| Field | Value |
|---|---|
| Datasets | FEVER (`copenlu/fever_gold_evidence`, gold-evidence setting) and HotpotQA (`hotpotqa/hotpot_qa`, fullwiki) |
| Sample | the pre-selected, pre-committed `*-smoke-25.json` manifests (25 examples each, selected by label/type balance, fixed before any model ran) |
| Retrieval | BM25, top_k=5 |
| Chunking | sentence-aware, chunk_size=256, overlap=0 |
| Claim extraction | deterministic sentence baseline (unchanged from the frozen config — a real extractor failed structured-output validation in the prior calibration work) |
| Relation classification | **real NLI**, `roberta-large-mnli` (revision `2a8f12d27941090092df78e4ba6f0928eb5eac98`), thresholds entailment=0.4/contradiction=0.7/duplicate=0.8 (from the frozen configs, not the repo's defaults of 0.5/0.5/0.8) |
| System variants | all 10 (`passage_rag` … `full_egrag`), seed=0 |
| Evidence budget | 256 tokens (16 reserved for output at `max_new_tokens=64`; 64 reserved at `max_new_tokens=256`) |
| Decoding | deterministic (`do_sample=False`), `temperature=0.0`, `top_p=1.0` |
| `max_new_tokens` | **64** (exactly the frozen config) **and 256** (a labeled deviation — see below) — both run, both reported |
| Device / dtype | CUDA, bfloat16, unquantized, for all three generators |

**Why two token budgets:** a 3-example pilot at the frozen config's
`max_new_tokens=64` showed heavy truncation on real data (the frozen value was
set before any real generator existed and was never validated against one).
Rather than silently changing the frozen value, both are run and reported as
separate, labeled conditions.

Models: `Qwen/Qwen2.5-3B-Instruct`, `Qwen/Qwen2.5-7B-Instruct`,
`Qwen/Qwen3.5-9B` (`enable_thinking=False`) — same caveat as the synthetic
report: the 9B point is a different Qwen generation from the 3B/7B pair, not a
same-family scale point.

## Results

### Failure rate (citation-validation + structural rejections)

| Model | FEVER (mnt=64) | FEVER (mnt=256) | HotpotQA (mnt=64) | HotpotQA (mnt=256) |
|---|---|---|---|---|
| Qwen2.5-3B | 67.6% | 55.2% | 80.8% | 45.2% |
| Qwen2.5-7B | 48.0% | 36.4% | 63.2% | 34.8% |
| Qwen3.5-9B | 76.0% | **16.4%** | 87.6% | 28.4% |

`max_new_tokens=256` reduces failure rate substantially for every model on
every benchmark — most dramatically for Qwen3.5-9B (FEVER: 76.0%→16.4%),
consistent with it needing more headroom to close its JSON output even with
thinking disabled.

### Coverage-adjusted metrics (primary comparison — see note below)

**Note on methodology:** the natural "mean over successful generations only"
is **not comparable across conditions with different failure rates**, because
the population of "successes" is a different, self-selected subset each time
(this would be a real statistical mistake to report as-is). The tables below
instead report **coverage-adjusted scores**: `sum(metric over successful,
applicable examples) / n_total`, i.e. every failed or gold-inapplicable example
scores 0. This is the metric used for all cross-condition comparisons in this
report. Raw success-conditional means are in
`artifacts/benchmark-matrix/analysis/metrics_summary.json` for reference.

**FEVER** (`answer_accuracy` is 0.000 for every model at every setting — see
"FEVER answer-accuracy" below; `citation_recall` is the meaningful metric here):

| Model | citation_recall (mnt=64) | citation_recall (mnt=256) |
|---|---|---|
| Qwen2.5-3B | 0.288 | 0.412 |
| Qwen2.5-7B | **0.364** | **0.428** |
| Qwen3.5-9B | 0.152 | 0.364 |

**HotpotQA**:

| Model | token_f1 (64) | token_f1 (256) | citation_recall (64) | citation_recall (256) | answer_accuracy (64) | answer_accuracy (256) |
|---|---|---|---|---|---|---|
| Qwen2.5-3B | 0.061 | 0.095 | 0.032 | 0.130 | 0.048 | 0.064 |
| Qwen2.5-7B | **0.164** | **0.166** | 0.108 | 0.160 | **0.120** | **0.116** |
| Qwen3.5-9B | 0.007 | 0.042 | 0.004 | 0.138 | 0.000 | 0.000 |

**Qwen2.5-7B is the strongest model on every HotpotQA metric at every token
budget.** Qwen3.5-9B is the weakest by a wide margin on HotpotQA despite being
the largest model — consistent with the earlier finding (synthetic report)
that it produces verbose, hedging answers that diverge from gold answer
wording, and additionally suffers the highest failure rate at `mnt=64`.

### FEVER answer-accuracy: a real methodological limitation, not a bug

`answer_accuracy` (normalized exact match vs. gold) is **exactly 0.000 for
every model, every token budget, every FEVER run.** I inspected actual
successful answers rather than accept a suspicious zero at face value — e.g.:
*"Sebastian Stan did not have a role in a German miniseries."* FEVER's gold
answer is the bare label (`SUPPORTS`/`REFUTES`/`NOT ENOUGH INFO`), but this
pipeline's generation prompt answers the claim as a natural-language statement,
never the literal label word. `answer_accuracy`/`token_f1` are **structurally
near-meaningless for FEVER under this pipeline's current prompting design** —
this is a real limitation of the answer-generation setup for a
claim-verification task, not a defect in the metric computation, and not
something I patched. `citation_recall` (evidence-page overlap, independent of
answer wording) remains meaningful for FEVER and is reported above.

### Runtime and GPU memory

| Model | Mean latency/call, FEVER (mnt=64→256), ms | Mean latency/call, HotpotQA (mnt=64→256), ms | Peak GPU memory (bf16) |
|---|---|---|---|
| Qwen2.5-3B | 1708 → 1956 | 1724 → 2350 | 6.19 GB |
| Qwen2.5-7B | 1855 → 1864 | 1389 → 1684 | 15.00–15.95 GB |
| Qwen3.5-9B | 4321 → 4706 | 3141 → 5938 | 17.63–18.58 GB |

Qwen3.5-9B is consistently the slowest (roughly 2–3× the 3B/7B latency) and
uses the most VRAM, as expected for the largest model. Latency generally rises
modestly from `mnt=64` to `mnt=256` (more generation headroom used on average),
most visibly for Qwen3.5-9B on HotpotQA (3141→5938 ms). Peak GPU memory is
measured via `torch.cuda.max_memory_allocated()`, reset per example
(`egrag.experiments.runner._cuda_reset_peak_memory`) — a new instrumentation
this session added (the field existed in the schema but was never populated
before).

## Failure taxonomy (automatically categorized, all 2000 real-benchmark example-runs)

| Category | FEVER (64) | FEVER (256) | HotpotQA (64) | HotpotQA (256) |
|---|---|---|---|---|
| Hallucinated/malformed citation ID | 197 | 258 | 89 | 102 |
| Generation truncated / malformed JSON | 272 | 0 | 388 | 28 |
| Structural: duplicate `claim_id` (`ValidationError`) | 0 | 0 | 72 | 72 |
| Other generation error | 10 | 12 | 30 | 69 |

**New failure mode found and root-caused, not just labeled:** the
`ValidationError: duplicate claim_id values are not allowed` failures are
**identical in count (72) at both token budgets and across all three models** —
a strong signal it's unrelated to generation at all. I traced it: it occurs
only in `passage_rag`/`reranked_passage_rag` (12 of 25 HotpotQA examples — 48%
— produce this), and happens during `EvidencePackage` construction, **before**
the generator is ever called. Root cause: HotpotQA's adapter represents each
gold sentence as a separate `Document` sharing a `source_id` (the page title);
for these 12 examples, BM25's top-5 retrieved passages include two chunks that
resolve to the same `passage_id`, which the `passage_rag` variant then uses
directly as a claim ID, tripping `EvidencePackage`'s duplicate-ID guard. This
is a genuine, pre-existing structural defect in how the `passage_rag`-family
baseline handles HotpotQA's per-sentence document representation — not a model
issue, not a token-budget issue, and not something I patched (a fix would
change baseline behavior, which was out of scope for this evaluation pass).

### Citation-ID pattern breakdown (of the "hallucinated ID" category)

| Pattern | FEVER (64) | FEVER (256) | HotpotQA (64) | HotpotQA (256) |
|---|---|---|---|---|
| Short/compound ID mismatch (e.g. `old::p0`→`p0`, or an invented short ID) | 115 | 139 | 115 | 118 |
| Hash prefix dropped entirely | 72 | 95 | 14 | 16 |
| Evidence metadata leaked into the citation (e.g. `source=... belief=...`) | 15 | 30 | 0 | 0 |
| `clm-` truncated to `c` | 6 | 17 | 6 | 29 |

Consistent with the synthetic-fixture findings: models most often drop part of
a compound or hash-based ID rather than inventing an unrelated one — the same
mechanism, now confirmed on two independent real datasets.

## Statistical analysis

Paired bootstrap CIs (`egrag.experiments.stats.paired_comparison`, 2000
resamples, seed=0 — already implemented in the repository) for `full_egrag`
vs. the strongest available baseline, paired by example ID **restricted to
examples where both variants succeeded**. **No p-values or significance
claims** — this project's own `stats.py` deliberately reports only bootstrap
CIs, and this report follows that existing convention rather than introduce a
hypothesis test now.

**Critical caveat:** given the high failure rates above, the number of
examples where *both* compared variants succeeded (`n_paired`) is very small
in most cases — as low as 1–4 for several FEVER comparisons, where the
resulting "CI" is degenerate (a single-pair bootstrap trivially reproduces
that one pair) and must not be read as evidence of anything. Only the
HotpotQA/`mnt=256` comparisons reach `n_paired` in the 12–18 range, which is
still small by conventional standards but at least non-degenerate.

The one finding with `n_paired` large enough to take seriously, and a CI that
excludes zero:

| Model | Metric | full_egrag | claim_only_rag | Δ | 95% CI | n_paired |
|---|---|---|---|---|---|---|
| Qwen2.5-3B | token_f1 | 0.122 | 0.406 | **−0.284** | [−0.512, −0.078] | 12 |
| Qwen2.5-7B | token_f1 | 0.096 | 0.496 | **−0.400** | [−0.662, −0.167] | 14 |
| Qwen3.5-9B | citation_recall | 0.194 | 0.333 | **−0.139** | [−0.278, −0.028] | 18 |

**On HotpotQA at `mnt=256`, the full EG-RAG graph pipeline underperforms the
simpler claim-only baseline**, with a CI excluding zero for two of three
models on token F1 and one on citation recall. This is reported as found, not
suppressed: it does not favor the full pipeline, and the sample is small
(12–18 paired examples from a 25-example smoke sample) — this should be
read as a signal to investigate with a larger sample, not a settled result.
Full comparison data (including the low-`n` FEVER cases) is in
`artifacts/benchmark-matrix/analysis/bootstrap_comparisons.json`.

## Figures

Vector (PDF + SVG) in `artifacts/benchmark-matrix/analysis/figures/`:
`failure_rate_by_group`, `token_f1_scaling`, `runtime_by_group`,
`failure_taxonomy`, `citation_id_patterns`.

## ACL-ready Results section (draft)

> **Real-benchmark evaluation.** We evaluate EG-RAG on 25-example, pre-selected
> development samples from FEVER (gold-evidence setting) and HotpotQA
> (fullwiki), using the same three Qwen generators as the synthetic-fixture
> evaluation, with real NLI relation classification (`roberta-large-mnli`)
> substituted for the lexical baseline. We report two generation-token budgets:
> 64 (a pre-committed, dev-frozen value never previously validated against a
> real generator) and 256 (a labeled deviation motivated by high truncation
> observed at 64), since the choice materially affects results — e.g. Qwen3.5-9B's
> failure rate on FEVER falls from 76.0% to 16.4% purely from this change.
>
> Across both benchmarks, Qwen2.5-7B is the strongest generator on every
> coverage-adjusted metric, including on HotpotQA where the larger, different-
> generation Qwen3.5-9B underperforms both smaller Qwen2.5 models by a wide
> margin — reinforcing that scale alone does not predict pipeline performance
> once architecture/training data change alongside it. FEVER's answer-accuracy
> metric is structurally uninformative under this pipeline's natural-language
> answering design (the model never emits the bare gold label), a real
> limitation we surface rather than mask; citation recall is the meaningful
> FEVER metric and improves for every model at the larger token budget
> (e.g. Qwen2.5-7B: 0.364→0.428).
>
> All failures were automatically categorized into four types, the dominant
> being truncated/malformed generation output (reduced by the larger token
> budget) and hallucinated citation IDs (unaffected by token budget, consistent
> with the synthetic-fixture findings: models most often drop part of a
> compound or hash-based identifier rather than inventing an unrelated one). A
> fourth, token-budget-independent failure type was identified and root-caused
> to a structural defect in how the `passage_rag` baseline handles HotpotQA's
> per-sentence document representation, affecting 48% of examples under that
> baseline specifically — a methodological limitation of the baseline adapter,
> not the generators. Where paired bootstrap comparison was possible with a
> non-degenerate sample size (HotpotQA, `mnt=256`, n=12–18), the full EG-RAG
> pipeline underperformed the simpler claim-only baseline on two of three
> generators (95% CI excluding zero); we report this directly as a signal
> warranting a larger-sample follow-up, not as a settled result.

## Generated files

- `artifacts/benchmark-matrix/{fever,hotpotqa}/{model}[_mnt256]/` — full
  per-run artifacts (manifest, resolved config, results.jsonl, aggregate,
  timing, failures.log, evidence packages/graphs) for all 12 runs.
- `artifacts/benchmark-matrix/analysis/metrics_summary.{json,csv}` — every
  number in this report.
- `artifacts/benchmark-matrix/analysis/failure_taxonomy.json` — full
  per-model/per-group failure and citation-ID-pattern breakdowns.
- `artifacts/benchmark-matrix/analysis/bootstrap_comparisons.json` — every
  computed comparison, including the low-`n` degenerate ones.
- `artifacts/benchmark-matrix/analysis/figures/*.{pdf,svg}` — the 5 figures.
- `artifacts/benchmark-matrix/_full_matrix.log` — complete run log, including
  the argparse bug and its fix, for full reproducibility of what actually
  happened in this session.
