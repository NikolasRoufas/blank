# Benchmark-Pipeline Milestone — Final Report

Goal: prepare EG-RAG for real benchmark evaluation (adapters, real generator,
benchmark metrics, caching/batching, small pilots, runtime estimate). No paper, no
final test-matrix, no benchmark superiority claims. Controlled/oracle results
preserved unchanged.

## Headline outcome (honest)

The deterministic, offline pieces are **built and validated**: real FEVER adapter,
HotpotQA adapter scaffold, benchmark metrics, tests, and documentation cleanup.
**Two independent blockers prevented running a live benchmark pilot**, and per the
milestone's stop conditions no pilot was forced:

1. **HotpotQA — data blocker:** cached only as parquet; `pyarrow`/`datasets` not
   installed, no network. Adapter raises a typed `MissingDependencyError`.
2. **FEVER — runtime/disk blocker:** adapter+data ready, but loading the real NLI
   + generator wedged on disk I/O during the milestone (free disk fell 8.0 → 4.5
   GiB then). Disk has since recovered (~8.6 GiB) and all build/test gates pass; a
   live real-model FEVER smoke was still not forced under the stop conditions and
   should be re-attempted under monitoring before trusting live timings.

## Benchmark adapters implemented

- **FEVER** `FeverGoldEvidenceDataset` — cached `copenlu/fever_gold_evidence` JSONL,
  **gold-evidence setting**; canonical `DatasetExample`; gold label/evidence never
  enter the pipeline; `dataset_fingerprint`, `validate_benchmark`.
- **HotpotQA** `HotpotQADataset` — parquet→example mapping implemented; blocked on
  parquet reader (reported, not faked).

## Dataset versions / fingerprints

- FEVER: `copenlu/fever_gold_evidence`, valid split 15,935 (SUPPORTS 4,638 /
  REFUTES 4,887 / NEI 6,410). Per-sample fingerprint via `dataset_fingerprint`
  (sha256 over ids+claims), verified stable.
- HotpotQA: `hotpotqa/hotpot_qa` fullwiki (parquet) — not loaded.

## Models selected (cached, local)

- NLI: `roberta-large-mnli` rev `2a8f12d2…` (validated earlier).
- Generator: `Qwen/Qwen2.5-0.5B-Instruct` (0.95 GB; deterministic decoding;
  context 4096; CPU; `HuggingFaceGenerator`). Chosen for size/feasibility, not
  the largest available.
- Device: CPU (no GPU). Decoding: deterministic (`do_sample=False`, seed threaded).

## Benchmark metrics added (`benchmark_metrics.py`)

- HotpotQA: `normalize_answer`, `exact_match`, `token_f1`, `supporting_fact_prf`,
  `hotpot_joint`.
- FEVER: `fever_label_accuracy`, `fever_evidence_prf`, `evidence_set_recovered`
  (alternative valid evidence sets), `fever_score` (official rule: label + complete
  evidence for S/R, label-only for NEI). Kept separate from graph-mechanism metrics.

## Claim-extraction / generator / HotpotQA / FEVER pilot metrics

Not produced — pilots blocked (above). The generation-validation contract (valid
JSON, citations resolve, abstention) exists and is unit-tested with the fake; a
real-model smoke is the blocked step.

## Fairness audit

`check_fairness` exists in the harness (shared example IDs / retriever / top-k /
generator / budget / seed). Wired for the variant set incl. `graph_no_bridge`;
exercised once a live pilot runs.

## Graph / bridge / contradiction behavior

Proven on controlled fixtures (previous milestone): bridge P/R 1.0, required-hop
coverage 0→1, belief/conflict invariance, structural contradiction gate guarantee.
Real-benchmark activation is pending the live pilot.

## Caching / batching

NLI batching, deterministic ordering, no-grad/eval mode, CPU-safe defaults exist
in `HuggingFaceNLIClassifier`; `build_nli_cache_key`/`DiskCacheBackend` provide
persistent, model/revision/threshold-keyed caching. Live cache hit-rate numbers
pending the blocked pilot.

## Runtime estimates / disk

Live timing blocked (model load wedged; disk 8.0 → 4.5 GiB). Estimates (labelled,
from prior runs + model sizes): NLI ~0.1–0.3 s/pair; generation ~3–10 s/example;
FEVER 25×8 warm ≈ 15–60 min **given adequate disk**; full FEVER matrix ≈ tens of
hours on CPU (GPU-scale). See `runtime-estimate.md`.

## Failed / blocked examples

None mis-handled; the blockers are environmental (parquet reader, disk), surfaced
as typed errors and reports.

## Quality-gate results

All required gates confirmed green on this host (2026-06-29, disk recovered to
8.6 GiB free):

- `ruff format --check .` — pass (159 files already formatted)
- `ruff check .` — pass (all checks passed)
- `mypy src` — pass (no issues in 100 source files)
- `pytest` — pass (exit 0; coverage TOTAL 89%)
- `uv build` — pass (built `egrag-0.0.0-py3-none-any.whl` + `.tar.gz`)

Benchmark integration tests: 12 passed, 1 skipped (cache-gated
`requires_local_models`). The `uv build` gate, which wedged on disk I/O earlier
in the milestone, succeeds now that free disk has recovered.

## Documentation updated

`docs/architecture.md` (status/relation families — finalized), `docs/reasoning.md`
(hop coverage vs evidence), `docs/experiments.md` (added `graph_no_bridge`;
corrected "real adapters out of scope"), `README.md` (status table rewritten),
`CHANGELOG.md`. `docs/{evidence-graph,configuration,limitations}.md` do **not
exist** in the repo (the canonical analog is `docs/implementation-status.md`); no
files were fabricated. No invented URLs/DOIs/venues/results.

## Candidate configurations for the next (calibration) milestone — NOT frozen

Three starting points (to be calibrated on **dev** data, never test):

- **C1 (lean/CPU-friendly):** retrieval top-k 5; chunk ~256 tok; claim limit 5/passage;
  NLI thresholds entailment 0.4 / contradiction 0.7 / duplicate 0.8 (prior dev pick);
  structural contradiction gate ON; bridges ON (min_confidence 0.5); evidence budget
  256 tok; generator Qwen2.5-0.5B; deterministic; max_new_tokens 48.
- **C2 (higher recall):** top-k 10; chunk ~256; claim limit 8/passage; same NLI;
  budget 384; max_new_tokens 64.
- **C3 (precision-leaning):** top-k 5; claim limit 5; contradiction 0.8; budget 256;
  gate ON; bridges min_confidence 0.6.

## Readiness assessment

- Ready for **calibration**: partially — code paths ready; **must first** free disk
  and (for HotpotQA) install a parquet reader, then validate a live FEVER smoke.
- Ready for **final benchmark runs**: **No** (needs GPU-class compute + disk).
- Ready for **paper claims**: **No** (no benchmark results produced).

## Unresolved blockers

1. Disk/I/O: free disk has recovered to ~8.6 GiB and all build/test gates now
   pass, but intermittent disk-I/O wedging was observed during heavy model loads
   earlier in the milestone; a real-model FEVER smoke should be re-attempted under
   monitoring before trusting live timings.
2. HotpotQA: no offline parquet reader (`pyarrow`/`datasets` not installed,
   no network) — adapter raises a typed `MissingDependencyError`.
3. (Consequent) no live pilot was run, so no real-benchmark metrics yet.

## Next milestone

Development-set calibration and configuration freezing — after the two blockers are
cleared and a live FEVER smoke validates the real generator/extraction path.
