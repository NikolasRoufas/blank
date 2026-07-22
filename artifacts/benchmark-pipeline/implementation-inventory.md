# Benchmark-Pipeline Implementation Inventory

Inspected the live repository and the local caches before editing.

## Environment / hard constraints (dominant findings)

- **Disk: ~8.0 GiB free** (57% capacity reported by `df`); the host filesystem
  intermittently throws `TimeoutError [Errno 60]` under load. No new large
  downloads are attempted.
- Installed: `transformers` 5.12.1, `torch` 2.12.1, `huggingface_hub`, `pyyaml`.
- **NOT installed: `datasets`, `pyarrow`, `pandas`** — and there is **no network**
  (offline policy). This gates how benchmark files can be read.

## Cached datasets (local, legally obtained earlier)

| dataset | cache format | usable offline now? |
|---|---|---|
| `copenlu/fever_gold_evidence` | **JSONL** (`train/valid/test.jsonl`) | **YES** — stdlib `json`; gold evidence sentences are inline |
| `hotpotqa/hotpot_qa` (fullwiki) | **parquet only** (`*.parquet`) | **NO** — needs `pyarrow`/`datasets`/`pandas` (none installed; no network) |
| `hotpot_qa` (older) | HF loading script (`.py`) | NO — needs `datasets` |
| also cached: MuSiQue, 2WikiMultiHopQA, trivia_qa, wiki_qa | parquet/script | NO (same blocker) |

**Consequence:** a real **FEVER** adapter is feasible now (gold-evidence
setting). A real **HotpotQA** adapter is **blocked** on a parquet reader; the
adapter is implemented but raises a typed `MissingDependencyError` until
`pyarrow`/`datasets` is installed.

## Cached models

- NLI: `roberta-large-mnli` (rev `2a8f12d2…`) — validated previously.
- Generator: **`Qwen/Qwen2.5-0.5B-Instruct`** (953 MB, cached) — small, instruction
  following, deterministic-capable, fits CPU/8 GiB disk. Selected for the pilot.

## Current public interfaces / adapters (reused, not duplicated)

- Datasets: `egrag.experiments.datasets` — `DatasetAdapter` protocol,
  `SyntheticGraphDataset`, `TemporalConflictDataset`, `JsonlDataset` (generic
  JSONL), `get_dataset`, integrity checks (`check_dataset_integrity`).
- Generators: `egrag.generation.adapters` — `FakeTextGenerator`,
  **`HuggingFaceGenerator`** (causal LM, lazy), `OpenAICompatibleGenerator`.
  `GenerationService` validates output (JSON contract, citations, abstention).
- Relations: `LexicalPairClassifier`, `HuggingFaceNLIClassifier` (validated),
  `StructuralContradictionGate`; `DeterministicBridgeDetector` + `detect_bridges`.
- Caching: `egrag.caching` — `DiskCacheBackend` (atomic, checksummed, metrics),
  `build_cache_key`, `build_nli_cache_key`.
- Metrics: `egrag.experiments.metrics` (EM/F1/citation/evidence, kinds);
  mechanism metrics in `mechanism_eval`.
- Experiment harness: `ExperimentRunner`, `run_system`, `VariantFlags`
  (incl. `bridges`), mechanism eval, fairness checks.
- CLI: `egrag run|search|extract|graph|reason|inspect-config|doctor|experiment …`.

## Missing / partial capabilities (this milestone)

- Real **FEVER** dataset adapter (gold-evidence) — to add.
- Real **HotpotQA** adapter — add scaffold; **blocked** on parquet reader.
- `egrag dataset prepare-*/validate/inspect` CLI — to add (thin).
- Benchmark-specific metrics (HotpotQA SP-F1/joint; FEVER label-acc/evidence-F1/
  FEVER-score) — to add.
- Persistent NLI/generation classification cache wired into the pipeline — partial
  (keys exist; disk-cache wrapper to add).
- Runtime estimator — to add.
- Real-generator benchmark pilot — **gated** by CPU runtime (see runtime estimate);
  a full 8-variant × 25-example pilot is not run this turn.

## Stale documentation to fix

- `docs/architecture.md` — header still "Proposal (planning phase). No production
  code exists yet." (partly fixed last milestone; finish).
- `docs/reasoning.md` — required-hop coverage language (fixed last milestone;
  verify).
- `docs/experiments.md` — variant list omits `graph_no_bridge`; "out of scope"
  real-model claims.
- `docs/evidence-graph.md`, `docs/configuration.md`, `docs/limitations.md`,
  `README.md`, `CHANGELOG.md` — relation families / real adapters / benchmark
  status.
- No invented URLs/DOIs/results anywhere.

## Files expected to change

- `src/egrag/experiments/benchmarks.py` (new) — FEVER + HotpotQA adapters +
  canonical benchmark example + validation.
- `src/egrag/experiments/benchmark_metrics.py` (new) — HotpotQA/FEVER metrics.
- `src/egrag/cli/dataset.py` (new) — `egrag dataset` subcommands; wired in `cli/main.py`.
- `src/egrag/experiments/__init__.py` — exports.
- docs (above) + `CHANGELOG.md`; tests under `tests/`.

## Go / no-go

- FEVER pilot: **feasible** (data ready; generator/NLI cached) but **CPU-slow** —
  validate runtime before scaling.
- HotpotQA pilot: **blocked** until a parquet reader is installed (owner action:
  `uv sync --extra <datasets/pyarrow>` with network) — reported, not faked.
