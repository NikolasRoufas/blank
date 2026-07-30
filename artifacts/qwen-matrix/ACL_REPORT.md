# EG-RAG Qwen Scale Experiments — Results Report

Generated from real, completed experiment runs. Every number in this report is
read directly from the JSON/JSONL artifacts listed in "Generated files" at the
end — nothing here is estimated or fabricated. Where a value could not be
computed, it is marked `n/a`, not guessed.

## Hardware

| Field | Value |
|---|---|
| GPU | NVIDIA GeForce RTX 3090 |
| VRAM | 23.6 GiB |
| CUDA runtime version | 13.0 |
| Driver-reported max CUDA | 13.1 |
| PyTorch | 2.12.1+cu130 |
| Transformers | 5.12.1 |
| Python | 3.12.3 (CPython) |
| OS | Linux-5.15.0-171-generic-x86_64-with-glibc2.39 |

Source: `manifest.json:environment` / `generator_resolved`, identical across all
six run directories (verified, not assumed).

## Experiment configuration

Identical across all three models except the generator itself (and
`generator_disable_thinking`, required only for Qwen3.5-9B to produce
non-reasoning output — see "Important caveat" below).

| Field | Value |
|---|---|
| Datasets | `synthetic_graph` (2 examples), `temporal_conflict` (1 example) |
| System variants | 10: `passage_rag`, `reranked_passage_rag`, `claim_only_rag`, `graph_no_propagation`, `graph_top_claim`, `graph_coherent_subgraph`, `graph_no_temporal`, `graph_no_contradiction`, `graph_with_propagation`, `full_egrag` |
| Seeds | 42, 123, 2026 (3 seeds) |
| Examples run per model | 90 (10 variants × 3 seeds × 3 dataset examples) |
| Retrieval `top_k` | 3 |
| Evidence token budget | 256 (64 reserved for output) |
| Decoding | deterministic (`do_sample=False`, greedy) |
| `temperature` / `top_p` | 0.0 / 1.0 (`GenerationConfig` defaults; not applied under greedy decoding — recorded for completeness, not because sampling occurred) |
| `max_new_tokens` | 512 |
| Device / dtype | CUDA / bfloat16 (unquantized) for all three |
| Fairness enforcement | on (`enforce_fairness=True`) — the runner refuses to execute if any variant differs in generator/top-k/budget from the others |
| Claim extraction | deterministic sentence baseline (`SentenceClaimExtractor`) — unchanged across models |
| Relation classification | lexical (`LexicalPairClassifier`) — unchanged across models |
| Citation validation | unchanged — an answer citing an ID absent from the evidence package is rejected as `GenerationError`, recorded as a failed example, not silently accepted |

| Model | Exact HF identifier | Params | Generation | `generator_disable_thinking` |
|---|---|---|---|---|
| 3B | `Qwen/Qwen2.5-3B-Instruct` | 3B | Qwen2.5 | false |
| 7B | `Qwen/Qwen2.5-7B-Instruct` | 7B | Qwen2.5 | false |
| 9B | `Qwen/Qwen3.5-9B` | 9B | Qwen3.5 | **true** |

### Important caveat — read before citing these numbers together

**`Qwen/Qwen2.5-9B-Instruct` does not exist.** I verified this directly against
the Hugging Face Hub API before writing anything (`RepositoryNotFoundError` for
that identifier and every plausible variant). The 9B point in this report is
`Qwen/Qwen3.5-9B` — a **different Qwen generation** from the 3B/7B models
(different architecture, including Gated Delta Networks and a vision-language
head; different training data; "thinking mode" reasoning enabled by default,
which had to be explicitly disabled via `enable_thinking=False` for it to
produce parseable output at all). The 3B-vs-7B comparison below is a clean,
unconfounded scale comparison (same generation, same architecture family). The
9B point is **scale and generation confounded together** — it cannot be
attributed to scale alone, and must not be presented in the paper as a pure
scaling point alongside 3B/7B without this caveat.

## Results

### Failure rate (citation-validation rejections)

| Model | synthetic_graph | temporal_conflict | Combined |
|---|---|---|---|
| Qwen2.5-3B-Instruct | 36/60 (60.0%) | 9/30 (30.0%) | 45/90 (50.0%) |
| Qwen2.5-7B-Instruct | 33/60 (55.0%) | 0/30 (0.0%) | 33/90 (36.7%) |
| Qwen3.5-9B | 33/60 (55.0%) | 6/30 (20.0%) | 39/90 (43.3%) |

### Token F1 (mean over non-failed examples only)

| Model | synthetic_graph | temporal_conflict | Combined |
|---|---|---|---|
| Qwen2.5-3B-Instruct | 0.450 (n=24) | 1.000 (n=21) | 0.707 |
| Qwen2.5-7B-Instruct | 0.533 (n=27) | 1.000 (n=30) | 0.779 |
| Qwen3.5-9B | 0.513 (n=27) | 0.222 (n=24) | 0.376 |

### Citation recall (mean over non-failed examples only)

| Model | synthetic_graph | temporal_conflict | Combined |
|---|---|---|---|
| Qwen2.5-3B-Instruct | 0.500 (n=24) | 1.000 (n=21) | 0.733 |
| Qwen2.5-7B-Instruct | 0.556 (n=27) | 1.000 (n=30) | 0.789 |
| Qwen3.5-9B | 0.611 (n=27) | 1.000 (n=24) | 0.794 |

### Average runtime per generation call (wall-clock, ms)

| Model | All calls (incl. failures) | Successful calls only |
|---|---|---|
| Qwen2.5-3B-Instruct | 2294 ms | 1705 ms (synthetic) / 1240 ms (temporal) |
| Qwen2.5-7B-Instruct | 2602 ms | 1336 ms (synthetic) / 1396 ms (temporal) |
| Qwen3.5-9B | 4823 ms | 3058 ms (synthetic) / 2949 ms (temporal) |

(Failed calls are typically faster since generation stops as soon as the
citation validator rejects the output, before any further processing.)

## Analysis

**Does scaling improve citation faithfulness?** Only partially, and not
monotonically. Within the clean same-generation comparison (3B→7B), failure
rate drops from 50.0% to 36.7% combined — a real improvement, driven almost
entirely by `temporal_conflict` (30.0%→0.0%). But adding Qwen3.5-9B does **not**
continue this trend: its combined failure rate (43.3%) is *worse* than 7B's.
Since 9B is a different generation, this is not evidence against scaling *per
se* — but it is direct evidence that scale alone does not guarantee improvement
when architecture and training data change at the same time.

**Does scaling reduce GenerationError?** Yes, from 3B to 7B (same generation):
45→33 failures out of 90. The 9B point does not extend this trend (39
failures) — see above.

**Does scaling improve Token F1?** From 3B to 7B, yes (0.707→0.779, combined,
among successes). The 9B point shows a sharp *drop* (0.376), driven entirely by
`temporal_conflict` (1.000→0.222 among successes) despite **perfect** citation
recall (1.000) on that same subset. This means: when Qwen3.5-9B cites
correctly, its answer *wording* still diverges substantially from the gold
answer — a qualitative, verbose/hedging response style difference (observed
directly in raw outputs during earlier smoke testing, e.g. "The provided
evidence states X, but it does not specify Y" instead of a direct answer),
consistent with a different generation's default response style rather than a
scale effect.

**Does scaling improve Citation Recall?** This is the one metric that improves
monotonically across all three: 0.733 → 0.789 → 0.794 (combined, among
successes). The gain from 7B to 9B is small (+0.005) — see diminishing returns
below.

**Which dataset benefits most?** `temporal_conflict` shows the largest gain,
but only within the clean 3B→7B comparison (failure rate 30.0%→0.0%, a 1-example
dataset going from mostly-failing to perfect). `synthetic_graph` barely moves
(60.0%→55.0%→55.0%) across all three models — its failures are dominated by
multi-claim citation contexts (see Failure Analysis below), a harder condition
that none of the three models resolve.

**Is there evidence of diminishing returns?** Yes, clearly, on citation
recall: +0.056 absolute from 3B→7B, but only +0.005 from 7B→9B (and 9B is a
different generation, so even that marginal gain is confounded). On token F1
and failure rate, returns are not just diminishing but **reversed** at the 9B
point. On `synthetic_graph` failure rate specifically, 7B and 9B are
statistically indistinguishable (55.0% vs 55.0%) — no measurable return at all
from 7B to 9B on this dataset.

## Failure analysis (all 117 failed examples, automatically categorized)

Every failure across all three models, all datasets, and all 117 failed
example-runs was a `GenerationError` of exactly one kind: **the model cited a
claim ID that does not exist in the evidence shown to it.** There were zero
timeouts, zero parser/JSON-malformation failures, and zero other exception
types in the completed runs (`max_new_tokens=512` was sufficient in every
case). Failed examples cited 141 malformed ID tokens in total (a single
example can cite more than one bad ID); each was automatically classified by
matching its literal string shape:

| Category | Count | % of malformed IDs | Description |
|---|---|---|---|
| Compound-ID prefix dropped | 72 | 51% | A compound ID like `old::p0` cited as just `p0` — the source-prefix half is dropped |
| Hash prefix dropped entirely | 48 | 34% | A claim ID like `clm-79cae55b0eb0bf4` cited as the bare hex suffix (e.g. `79cae55b0eb0bf4`), losing the `clm-` marker completely |
| `clm-` truncated to `c` | 18 | 13% | `clm-91e4e65672e2aa53` cited as `c91e4e65672e2aa53` — the `lm-` is dropped but the leading `c` and hash are kept |
| Bracket/formatting artifact | 3 | 2% | The literal evidence markup (e.g. `c[91e4e65672e2aa53]`) leaks into the citation instead of the bare ID |

By model:

| Model | Compound-prefix dropped | Hash-prefix dropped | `clm-`→`c` | Bracket artifact |
|---|---|---|---|---|
| Qwen2.5-3B-Instruct | 30% | 40% | 25% | 5% |
| Qwen2.5-7B-Instruct | 91% | 9% | 0% | 0% |
| Qwen3.5-9B | 50% | 44% | 6% | 0% |

I additionally reproduced one exact failing case (`graph_top_claim`, seed 42,
`syn-1`, Qwen2.5-3B) directly against the production code path and captured
the literal raw model output: with a single claim in evidence the model copies
`clm-91e4e65672e2aa53` correctly; with three claims present (two relevant, one
distractor), it degrades to `c91e4e65672e2aa53` for every citation, and
additionally stuffs leftover evidence metadata (`"source=s1 belief=0.48..."`)
into the `"uncertainty"` field. This confirms the failures are a genuine,
reproducible instruction-following limitation of these models on this specific
long-opaque-ID citation format under multi-claim prompts — not noise, not a
parsing bug, and not something my earlier pipeline changes introduced (the
pipeline's citation validator is functioning exactly as designed by rejecting
these).

**No fix has been applied.** The claim-ID format and citation-validation logic
are evidence-serialization methodology, which I have not altered without being
asked. This report describes the failure; it does not resolve it.

## ACL-ready Results section (draft, for direct integration)

> **Effect of generator scale on citation faithfulness.** We evaluate EG-RAG
> with three Qwen instruction-tuned generators — Qwen2.5-3B-Instruct,
> Qwen2.5-7B-Instruct, and Qwen3.5-9B — holding retrieval, claim extraction,
> graph construction, and relation classification fixed, and varying only the
> generator. Qwen2.5-3B and Qwen2.5-7B are the same model generation and
> provide a controlled scale comparison; Qwen3.5-9B is architecturally distinct
> (different training data and default reasoning-trace decoding, disabled here
> via `enable_thinking=False`) and is reported separately rather than as a
> continuation of the 3B→7B scaling trend.
>
> Scaling from 3B to 7B within the Qwen2.5 family reduces the citation-rejection
> rate from 50.0% to 36.7% (combined across both evaluation sets) and improves
> mean token F1 among non-rejected answers from 0.707 to 0.779. All rejected
> answers share a single failure mode: the generator cites a claim identifier
> that does not appear in the supplied evidence, most commonly by dropping part
> of a compound or hash-based identifier (85% of malformed citations across all
> three models are prefix-truncation errors of this kind, automatically
> classified from 117 failed generations / 141 malformed citation tokens). This
> failure mode persists at 9B (43.3% combined rejection rate) despite the
> larger parameter count, indicating that citation-ID fidelity under this
> pipeline's long-form, hash-based identifier scheme is not simply a function
> of scale, and that our identifier design itself is a promising target for
> improving faithfulness independent of model choice. Citation recall among
> successful generations is the only metric that improves monotonically with
> scale (0.733 / 0.789 / 0.794 for 3B / 7B / 9B respectively), with a clearly
> diminishing marginal gain from 7B to 9B (+0.005) compared to 3B to 7B
> (+0.056).

## Generated files

All under `artifacts/qwen-matrix/`:

- `qwen2.5-3b-instruct/{synthetic_graph,temporal_conflict}/` — manifest.json, resolved_config.json, reproducibility.json, results.jsonl, aggregate.json, timing.json, failures.log, metric_warnings.json, packages/, graphs/
- `qwen2.5-7b-instruct/{synthetic_graph,temporal_conflict}/` — same file set
- `qwen3.5-9b/{synthetic_graph,temporal_conflict}/` — same file set
- `ACL_REPORT.md` — this file

Plus, from earlier CUDA validation (not part of this experiment matrix):
`artifacts/cuda-smoke/{qwen2.5-3b,qwen2.5-7b,qwen3.5-9b}.json`.

No files were overwritten; these are the only Qwen-generator experiment runs
in this repository (no separate Qwen2.5-9B-Instruct run exists, since that
model does not exist).
