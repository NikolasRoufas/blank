# EG-RAG: Evidence Graph Retrieval-Augmented Generation

EG-RAG is a research framework for retrieval-augmented generation that puts an
explicit, inspectable evidence structure between retrieval and generation.
Instead of handing a generator a ranked list of passages, it breaks retrieved
text into atomic claims, links those claims with typed relations, scores and
propagates belief over the resulting graph, resolves conflicts, and selects a
compact set of claims to answer from. The generator is asked for an answer whose
citations point back to specific claims.

The core is pure Python and runs offline on CPU. Model-backed components
(dense retrieval, NLI relation classification, transformer extraction and
generation) are optional and live behind dependency extras.

This is research software. It contains no benchmark results: the behaviour
described here is exercised by the test suite and small controlled smoke runs,
not by a completed benchmark evaluation. See [Current limitations](#current-limitations).

## Motivation

Passage-level RAG returns a ranked list of text spans and leaves the rest to the
generator. That has some consequences EG-RAG is built to address:

- The retrieved evidence is unordered and unstructured; relationships between
  facts (agreement, contradiction, duplication, dependency) stay implicit.
- Retrieval rank reflects lexical or semantic similarity, not whether a passage
  is true or useful, and a ranker still returns a top result when nothing
  relevant was found.
- Multi-hop questions need two or more facts joined through a shared entity;
  similarity alone does not express that link.
- Contradictory or superseded statements are common and should remain visible to
  the reader instead of being averaged away.

EG-RAG moves the structuring step before generation: it builds the evidence
graph, decides what to keep, and only then generates. The design keeps five
quantities distinct throughout — `belief`, `extraction_confidence`,
`relation_confidence`, `source_reliability`, and `query_utility` — and preserves
provenance from every claim back to a source span.

## Pipeline

```text
query
  │
  ▼
retrieve ─▶ chunk ─▶ extract claims ─▶ classify relations ─▶ build graph
                                                               │
                                     initialize belief ◀───────┘
                                                               │
                            signed belief propagation ─▶ conflict resolution
                                                               │
                                       reasoning-subgraph selection
                                                               │
                                     serialize evidence ─▶ generate answer
                                                               │
                                     citation + grounding validation
```

The end-to-end wiring lives in `egrag.answering.answer_query` (library entry
point) and `egrag.composition.run_pipeline` (config-driven composition root).

## Architecture

The code follows a strict dependency direction:

```text
cli ─▶ application/composition ─▶ domain ◀─ adapters
                                    ▲
                       io / config ─┘
```

- **`egrag.domain`** — models, ports (typed `Protocol`/ABC interfaces), and
  errors. It imports only the standard library and Pydantic; never NetworkX,
  Transformers, sentence-transformers, HTTP clients, or storage libraries.
- **`egrag.adapters`** — concrete implementations that wrap external libraries
  (BM25, dense/hybrid retrieval, cross-encoder reranking, transformer and
  structured-JSON extraction, GraphML export) behind domain protocols.
- **`egrag.application`**, **`egrag.composition`**, **`egrag.answering`** —
  orchestration and the single composition root.
- **`egrag.cli`** — the Typer command-line interface.

Every model-facing capability is a typed protocol, and provider-specific imports
are kept lazy and inside their adapter. There is no import-time model loading,
network access, or hidden global state. Experimental algorithms (belief
propagation variants, conflict resolution, subgraph selection) are reachable only
through stable protocols. See `docs/architecture.md` for the full design.

## Installation

Requires Python 3.12 and [uv](https://docs.astral.sh/uv/).

```bash
uv sync                        # core + dev tooling (pure Python, offline)
uv sync --extra local-models   # + transformers, torch (real models)
uv sync --extra benchmarks     # + pyarrow (HotpotQA parquet)
uv sync --all-extras           # every runtime extra
```

## Dependencies

Core runtime dependencies are `pydantic`, `pydantic-settings`, `typer`, and
`pyyaml`. Everything model- or integration-specific is an optional extra
(declared in `pyproject.toml`):

| Extra | Adds | Enables |
|-------|------|---------|
| `retrieval` | `rank-bm25` | sparse BM25 helper |
| `dense` | `sentence-transformers` | dense / hybrid retrieval, cross-encoder reranking |
| `graph` | `networkx` | GraphML export |
| `local-models` | `transformers`, `torch` | local extraction, generation, NLI |
| `http-models` | `httpx` | OpenAI-compatible HTTP generator |
| `experiments` | `numpy` | experiment statistics |
| `benchmarks` | `pyarrow` | HotpotQA parquet loading |
| `docs` | `mkdocs`, `mkdocs-material` | documentation site |

A core-only install runs the full deterministic pipeline with none of these.

## Quick start

Command line (deterministic fake generator, built-in demo corpus, no extras):

```bash
uv run egrag run -q "What does EG-RAG do?"
```

Python:

```python
from egrag.answering import answer_query
from egrag.domain.models import Query
from egrag.fakes import build_demo_documents

result = answer_query(
    Query(query_id="q1", text="What does EG-RAG do?"),
    build_demo_documents(),
)
print(result.answer.text)
print(result.answer.cited_claim_ids)
```

Both paths use the deterministic pipeline; they need no extras and download
nothing.

## Command-line interface

`egrag --help` lists the commands. The pipeline commands use deterministic
components and the built-in demo corpus, so they run offline.

| Command | Purpose |
|---------|---------|
| `run` | Run the full pipeline for one query and print a grounded answer. |
| `search` | Retrieve and rank passages from the demo corpus. |
| `extract` | Retrieve, then extract claims (stops before the graph). |
| `graph` | Build and inspect the evidence graph (stops before belief). |
| `reason` | Run reasoning on a fixed synthetic example and print the trace. |
| `inspect-config` | Print the resolved configuration as JSON. |
| `doctor` | Report optional-dependency availability. |
| `gpu-readiness` | Report torch/CUDA/MPS, selected device, cache dir, free disk. |
| `experiment` | Run/analyze evaluation experiments; includes `matrix` (dry-run). |

Common options on `run` include `--generator {fake,openai,hf}`, `--model`,
`--base-url`, `--top-k`, `--evidence-budget`, `--deterministic/--sampled`,
`--config`, `--json`, `--package-out`, and `--graph-export`. Real generators are
optional: `openai` needs `--base-url` and `--model`; `hf` needs `--model` and the
`local-models` extra.

```bash
uv run egrag run -q "What does EG-RAG do?" --json
uv run egrag graph -q "What does EG-RAG do?" --graphml   # needs the graph extra
uv run egrag doctor
```

## Python API

- `egrag.answering.answer_query(query, documents, *, generator=None, config=None,
  top_k=4, evidence_token_budget=512, ...)` → `PipelineResult` — the full
  library pipeline over a sequence of `Document`s.
- `egrag.composition.run_pipeline(config, query, *, artifact_dir=None)` →
  `PipelineResult` — the config-driven composition root, with runtime validation
  of optional dependencies, model paths, budgets, and cache configuration.
- `egrag.fakes.build_demo_documents()` — the built-in demo corpus.

A `PipelineResult` exposes the `answer` (text, `cited_claim_ids`, `abstained`,
`uncertainty`, `unsupported_warnings`), the assembled `package`
(`EvidencePackage`), `metrics`, and a reproducibility `manifest`.

## Relations

Edge types are `egrag.domain.models.RelationType` (schema version 1.6.0):

| Enum | Role |
|------|------|
| `SUPPORT` | Directional entailment; contributes positive belief. |
| `CONTRADICTION` | Symmetric disagreement; forms conflict sets, negative belief. |
| `DUPLICATE` | Near-paraphrase; preserved but not counted as independent support. |
| `DEPENDENCY` | One claim depends on another; used for connectivity. |
| `SUPERSESSION` | A newer claim supersedes an older one under an explicit policy. |
| `BRIDGES` | Query-conditioned connectivity for multi-hop reasoning; does not affect belief, conflicts, or corroboration. |
| `NEUTRAL` | Classification outcome; not stored as an edge in normal construction. |

## Repository structure

```text
src/egrag/
  domain/          models, ports (protocols), errors, schema version
  adapters/        retrieval, reranking, extraction, graph-export adapters
  application/     pipeline orchestration
  answering.py     library entry point
  composition.py   config-driven composition root
  cli/             Typer command-line interface
  config/          settings (env) and hierarchical YAML config schema
  graph/           evidence-graph construction, classification, bridges, NLI
  reasoning/       belief scoring, propagation, conflict, selection
  generation/      evidence assembly, generators, grounding validation
  serialization/   versioned evidence-package serialization
  caching/         memory and disk cache backends
  experiments/     evaluation harness, variants, metrics, benchmarks
  experimental/    algorithms reachable only through stable protocols
  observability/   logging and metrics
configs/           baseline and ablation configurations
scripts/           experiment driver scripts
tests/             unit, integration, e2e, property, sanity
docs/              architecture and component documentation
artifacts/         calibration, pilot, and reproducibility records
```

## Configuration

Two layers of configuration:

- **Environment settings** (`egrag.config.EGRagSettings`, prefix `EGRAG_`): a
  small set of runtime knobs — `EGRAG_SEED`, `EGRAG_CHUNK_SIZE`,
  `EGRAG_CHUNK_OVERLAP`, `EGRAG_LOG_LEVEL` — optionally loaded from an
  `--env-file`. `egrag inspect-config` prints the resolved values.
- **Hierarchical YAML** (`egrag.config.schema.EGRagConfig`, passed with
  `--config`): sections for corpus, chunking, retrieval, reranking, extraction,
  graph construction, candidate generation, NLI, temporal reasoning, source
  reliability, initial scoring, propagation, conflict resolution, subgraph
  selection, serialization, generation, grounding verification, caching,
  security, and reproducibility. Baseline and ablation examples are under
  `configs/`.

Stochastic components take an explicit seed from configuration and default to
deterministic decoding, so the same input and seed produce the same output.

## Running experiments

The evaluation harness (`egrag.experiments`) defines a fixed variant set so a
comparison isolates one component. All variants share the same retriever,
chunker, extractor, classifier, and generator.

| Variant | Family | Isolates |
|---------|--------|----------|
| `passage_rag` | passage | retrieval only |
| `reranked_passage_rag` | passage | reranking (identity on BM25 order today) |
| `claim_only_rag` | claim | claim extraction, no graph |
| `graph_no_propagation` | graph | belief propagation off |
| `graph_with_propagation` | graph | belief propagation on |
| `graph_top_claim` | graph | top-claim selection |
| `graph_coherent_subgraph` | graph | greedy connected selection |
| `graph_no_temporal` | graph | temporal/supersession edges off |
| `graph_no_contradiction` | graph | contradiction edges off |
| `full_egrag` | graph | full pipeline |

`egrag experiment matrix` plans a real-data (HotpotQA/FEVER) benchmark run from a
frozen config and prints the plan without running inference; its `--execute` path
remains disabled here (that benchmark needs data not yet available offline in
this repository — see `docs/benchmarks.md`).

```bash
uv run egrag experiment matrix --benchmark fever --dry-run \
  --sample artifacts/benchmark-calibration/samples/fever-dev-100.json \
  --output-dir artifacts/final-matrix/out
```

Separately, `egrag experiment run --generator huggingface` runs the existing
synthetic-fixture experimental design (same variants/seeds/budgets as the
fake-generator pilots) with a real local model — CUDA-required, quantization, and
GPU/CUDA/dtype metadata are all supported; see "Qwen scale experiments (A/B/C)"
in `docs/reproduction.md`.

Driver scripts for the recorded milestones live under `scripts/`
(`run_paper_experiments.py`, `run_bridge_eval.py`, `run_mechanism_repair.py`,
`run_real_nli_eval.py`, `estimate_runtime.py`). See `docs/experiments.md` and
`docs/benchmarks.md`.

## Reproducing results

Every evidence package carries a run manifest (framework version, config hash,
seeds, adapter identities, input hash), and serialization is versioned.
Frozen benchmark settings and their checksums are under
`artifacts/benchmark-calibration/frozen-configs/`; fixed development samples are
under `artifacts/benchmark-calibration/samples/`. `docs/reproduction.md` gives
exact commands for CPU, MPS, and CUDA.

## Development workflow

```bash
uv run ruff format --check .
uv run ruff check .
uv run mypy src
uv run pytest
uv build
```

CI additionally verifies that the package builds and that a core-only install
runs an offline end-to-end pipeline with no optional model dependency. Contributor
conventions are in `CONTRIBUTING.md` and `docs/development.md`; architecture
decisions are in `docs/adr/`.

## Testing

```bash
uv run pytest              # whole suite
uv run pytest -m unit      # a single category
```

Tests are marked `unit`, `integration`, `e2e`, `property`, and `sanity`. They run
offline — a session fixture blocks network access — and never download models.
Tests needing an optional extra are marked `requires_<extra>` and skip cleanly
when it is absent. Import-isolation tests check that importing core packages pulls
in no optional model library, and `sanity` tests encode the scientific-integrity
invariants (distinct score quantities, contradictions never silently discarded,
provenance retained).

## Current limitations

- No final benchmark matrix (HotpotQA/FEVER, real data) has been run.
- The Qwen scale experiments (`docs/reproduction.md`) have been smoke-tested for
  real on an NVIDIA GPU (tokenizer, model, and NLI loading; one generation; one
  full pipeline example — see `artifacts/cuda-smoke/`), but a full multi-seed run
  producing reportable answer-quality numbers has not yet been executed.
- The small CPU smoke model (`Qwen2.5-0.5B-Instruct`) is only useful for
  exercising the adapters — it is not faithful enough for evaluation (it does not
  reliably copy source spans and answers even when the evidence is insufficient).
  A larger model on a GPU is needed for evaluation-quality numbers.
- `reranked_passage_rag` currently applies an identity reranker over BM25 order.
- FEVER uses the gold-evidence setting rather than open retrieval; the HotpotQA
  fullwiki split has limited gold-evidence coverage, so evidence-coverage metrics
  there are proxy measurements on the subset where the full chain is present.

## Citation

Publication details are pending; the venue, year, and identifiers are
intentionally omitted until publication.

```bibtex
@misc{roufas_egrag,
  author = {Nikolaos Roufas},
  title  = {EG-RAG: Evidence Graph Retrieval-Augmented Generation},
  note   = {Preprint in preparation}
}
```

## License

MIT. See [`LICENSE`](LICENSE).

## Documentation

- `docs/architecture.md` — layers, ports/adapters, pipeline internals.
- `docs/models.md` — model adapters, chat templates, JSON recovery, devices.
- `docs/caching.md` — cache backends, keys, invalidation.
- `docs/benchmarks.md` — dataset adapters, splits, metrics, caveats.
- `docs/experiments.md` — evaluation method, variants, fairness, runtime.
- `docs/reproduction.md` — exact commands (CPU, MPS, CUDA).
- `docs/development.md` — conventions for contributors.
- `docs/reasoning.md`, `docs/bridge-relations.md`, `docs/score-taxonomy.md` —
  reasoning subsystem, the bridge relation, and the score taxonomy.
