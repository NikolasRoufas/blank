# Benchmark-Calibration Milestone — Starting State

Captured 2026-06-29/30 before any calibration edits. Honest inventory of the
host, blockers, and assets. No results are fabricated; live pilots are gated by
the blockers below.

## Critical finding: corrupted `pyproject.toml` (repaired)

`pyproject.toml` was found **corrupted**: inode size 3526 B but **0 readable
bytes** (data extent lost — confirmed via `cat`, `dd`, `od`, and Python
`open().read()`). Root cause: dirty pages were never flushed during the earlier
low-disk / disk-I/O-wedge incident, so the file's data blocks were dropped while
the inode metadata survived.

Consequences discovered:

- The previous milestone's final `uv build` "✓" was **vacuous** — it built a
  degenerate `egrag-0.0.0` from an empty config (the sdist captured
  `pyproject.toml` already at 0 bytes).
- There is **no on-disk recovery source**: not a git repo, **no Time Machine
  local snapshots**, and the sdist copy is empty.

Recovery sources that *were* intact and used to reconstruct faithfully:

- Installed editable dist-info `egrag-0.1.0.dist-info/METADATA` → exact
  `[project]` metadata, dependencies, and all extras.
- `uv.lock` (302 KB, readable) → confirms deps, dev-deps, extras, version 0.1.0.
- `egrag-0.1.0.dist-info/WHEEL` → `Generator: hatchling 1.30.1` ⇒ original
  build backend = hatchling.
- Observed earlier-this-session gate output (coverage 89% w/ branch +
  term-missing; strict-clean mypy on 100 files; full ruff pass) ⇒ the code
  already satisfies a strict config; only `[tool.*]` had to be reconstructed and
  re-verified against the gates.

`pyproject.toml` has been reconstructed (hatchling backend; exact deps/extras;
strict mypy; ruff line-length 100 with `extend-exclude=["artifacts"]`; pytest
coverage addopts + registered markers) and the new `benchmarks` extra added. See
the gate results in the final report once re-verified.

## Host

| Resource | Value |
|----------|-------|
| Platform | macOS (Darwin 24.5.0), Apple M3 |
| CPU | 8 cores |
| Memory | 16 GiB |
| GPU | none (CPU only) |
| Python | 3.12.13 (uv-managed) |
| Free disk | ~7.0 GiB (tight; fluctuated 4.5–8.6 during prior milestone) |
| Network | disabled |

**Disk I/O instability:** recursive reads/greps over `src/`+`tests/`
intermittently wedge to a 2-minute timeout (the recurring `TimeoutError
[Errno 60]`), while single-file reads succeed. One file (`pyproject.toml`)
already lost its data. This is the dominant risk for live real-model runs.

## Installed dependencies (core + dev + optional present)

- Core: pydantic 2.13.4, pydantic-settings 2.14.2, pyyaml, typer (per lock).
- Present optional/runtime: `transformers` 5.12.1, `torch` 2.12.1,
  `tokenizers` 0.22.2, `huggingface-hub` 1.16.1, `networkx` 3.6.1, `numpy` 2.5.0.
- Dev: hypothesis, mypy, pytest, pytest-cov, ruff (0.15.18), types-pyyaml.

**Not installed (and not offline-installable — absent from uv/pip caches,
network disabled):** `pyarrow`, `datasets`, `pandas`, `fastparquet`, `duckdb`,
`polars`, `sentence-transformers`.

## Local model cache (`~/.cache/huggingface/hub`, ~5.0 GiB)

- `roberta-large-mnli` (NLI; validated in a prior milestone) — selected NLI.
- `Qwen/Qwen2.5-0.5B-Instruct` (~1 GB) — generator/extractor candidate.
- `cross-encoder/nli-deberta-v3-large`, `cross-encoder/nli-MiniLM2-L6-H768`
  (reranker / alt NLI).
- `sentence-transformers/all-MiniLM-L6-v2` (dense embeddings; note the
  `sentence-transformers` package itself is **not** installed).

## Local datasets (`~/.cache/huggingface/hub`)

- `copenlu/fever_gold_evidence` — JSONL, **gold-evidence setting**. Loadable
  fully offline (stdlib json). FEVER `valid` split ≈ 15,935 rows.
- `hotpotqa/hotpot_qa` — fullwiki, **parquet only** (`validation-00000-of-00001
  .parquet` ≈ 28 MB; no JSON form). Requires a parquet reader.
- Also present (not used this milestone): MuSiQue, 2WikiMultiHopQA, trivia_qa,
  wiki_qa, MedRAG/pubmed, medical_meadow_wikidoc, newsgroup, plain `fever`.

## Blockers

1. **HotpotQA live load — BLOCKED (owner action required).** `pyarrow` cannot be
   installed offline; HotpotQA is cached only as parquet. Decision this
   milestone: **defer live HotpotQA**; add the `benchmarks` extra (pyarrow) and
   wire/test everything offline-testable (mapping, typed error, fixtures). Live
   HotpotQA pilot remains blocked pending the owner running, with network:
   `uv pip install pyarrow` (or `uv sync --extra benchmarks`).
2. **Real-model FEVER run — AT RISK.** Models are cached, but real model loads
   previously wedged on disk I/O and free disk is ~7 GiB. A tiny FEVER smoke will
   be attempted under monitoring; per the stop conditions it will not be forced
   if it wedges or disk becomes unsafe.
3. **`sentence-transformers` absent** → dense retrieval calibration is limited to
   BM25/sparse unless installed (offline-blocked). Bridge/graph calibration can
   still proceed on FEVER gold-evidence + lexical retrieval.

## Candidate configurations carried in (from benchmark-integration milestone)

- **C1 (lean/CPU-friendly):** top-k 5; chunk ~256; claim limit 5/passage; NLI
  entailment 0.4 / contradiction 0.7 / duplicate 0.8; contradiction gate ON;
  bridges ON (min_confidence 0.5); evidence budget 256 tok; generator
  Qwen2.5-0.5B; deterministic; max_new_tokens 48.
- **C2 (higher recall):** top-k 10; claim limit 8; same NLI; budget 384;
  max_new_tokens 64.
- **C3 (precision-leaning):** top-k 5; claim limit 5; contradiction 0.8; budget
  256; gate ON; bridges min_confidence 0.6.

These are **not** calibrated or frozen yet; that is this milestone's goal where
the environment permits.

## Expected files to change

- `pyproject.toml` (repaired + `benchmarks` extra) — DONE.
- `src/egrag/experiments/benchmarks.py` (HotpotQA loader: extra-gated import,
  unchanged typed error; possibly small refactor for testable `_parse`).
- `tests/integration/test_benchmark_pipeline.py` and/or a new
  `tests/integration/test_benchmark_calibration.py` (the 22 required tests where
  offline-feasible).
- `artifacts/benchmark-calibration/**` (this milestone's outputs).
- Docs touch-ups (README/CHANGELOG/experiments) reflecting the `benchmarks` extra
  and calibration status.

**Not modified (preserved):** controlled-mechanism results, real-NLI controlled
results, bridge-milestone results, invalidated zero-edge results, and prior
benchmark-integration reports under `artifacts/`.
