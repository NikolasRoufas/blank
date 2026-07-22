# EG-RAG: Evidence Graph Retrieval-Augmented Generation

EG-RAG retrieves passages, splits them into atomic claims, and connects the claims
with typed relations (support, contradiction, duplication, dependency,
supersession, and a query-conditioned bridge relation). It scores and propagates
belief over that graph, resolves conflicts, selects a compact set of evidence, and
asks a generator for an answer whose citations point back to specific claims. The
aim is to make the evidence a model reasons over inspectable, and to keep
contradictory or outdated evidence visible instead of discarding it.

This is research software. Nothing here is a benchmark result: the numbers below
come from unit tests and small controlled smoke runs, not from a completed
benchmark evaluation.

## Why claims instead of passages

Passage-level RAG hands the generator a ranked list of text spans. That has a few
consequences this project tries to address:

- The evidence is unordered and unstructured; relationships between facts are left
  implicit.
- Retrieval rank measures lexical/semantic similarity, not whether a passage is
  true or useful, and a ranker returns a top result even when nothing relevant was
  found.
- Multi-hop questions need two or more facts connected through a shared entity;
  similarity alone does not express that connection.
- Contradictory or superseded statements are common and should stay visible to the
  reader rather than being averaged away.

EG-RAG moves the structuring step before generation: it builds the evidence graph,
decides what to keep, and only then generates.

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

Stages, with their entry points:

1. Retrieval — BM25 (pure Python), optional dense and hybrid (`egrag.adapters.retrieval`).
2. Chunking — sentence-aware (`SentenceAwareChunker`).
3. Claim extraction — deterministic `SentenceClaimExtractor`, or a structured
   model via `StructuredClaimExtractor` (`egrag.adapters.extraction`).
4. Relation classification — lexical or NLI (`egrag.graph`).
5. Graph construction (`egrag.graph.GraphBuilder`).
6. Belief initialization (`egrag.reasoning`).
7. Signed belief propagation, or a no-propagation baseline.
8. Conflict-set resolution.
9. Subgraph selection — greedy-connected or top-claim.
10. Evidence serialization (plain / markdown / chat renderers).
11. Answer generation (`egrag.generation`).
12. Attribution and grounding validation.

## Status

| Area | State |
|------|-------|
| Domain models, ports, application pipeline | implemented, tested |
| Deterministic offline path (query → answer, no extras) | implemented, tested |
| Retrieval, graph, reasoning, generation | implemented; some paths need extras |
| Real NLI (`roberta-large-mnli`) | implemented; controlled cases verified |
| Local extraction/generation adapters (chat template, JSON recovery, caching) | implemented; structured-output smoke passes |
| Benchmark adapters (FEVER, HotpotQA) | implemented |
| Development calibration | recorded under `artifacts/benchmark-calibration/` |
| Final benchmark matrix (GPU) | not run — see limitations |

What has actually been checked: the test suite (441 passed, 7 skipped) and
`uv build`; and small smoke runs with `Qwen2.5-0.5B-Instruct` and
`roberta-large-mnli` on CPU. In those smokes the repaired adapters produced valid
structured output (extraction 4/4, generation 6/6, NLI 4/4 controlled cases, no
invalid citations, cold/warm cache outputs identical). The 0.5B model still writes
a confident answer when the evidence does not support one, and does not reliably
copy source spans verbatim, so it is only useful for exercising the adapters — not
for evaluation. See `artifacts/real-adapter-repair/final-report.md`.

## Installation

Python 3.12 and [uv](https://docs.astral.sh/uv/). A core install is pure Python and
runs offline.

```bash
uv sync                        # core + dev tools
uv sync --extra benchmarks     # + pyarrow (HotpotQA parquet)
uv sync --extra local-models   # + transformers, torch (real models)
uv sync --all-extras           # every runtime extra (excludes benchmarks/docs)
```

Extras are defined in `pyproject.toml`: `retrieval`, `dense`, `graph`,
`local-models`, `http-models`, `experiments`, `benchmarks`, `docs`.

## Quick start

Command line (deterministic fake generator, built-in demo corpus):

```bash
uv run egrag run -q "What does EG-RAG do?"
```

Python:

```python
from egrag.answering import answer_query
from egrag.domain.models import Query
from egrag.fakes import build_demo_documents

result = answer_query(Query(query_id="q1", text="What does EG-RAG do?"),
                      build_demo_documents())
print(result.answer.text)
print(result.answer.cited_claim_ids)
```

Both use the deterministic pipeline, so they need no extras and download nothing.

## Command line

`egrag --help` lists the commands:

| Command | Purpose |
|---------|---------|
| `run` | Full pipeline for one query. |
| `search` | Retrieve and rank passages from the demo corpus. |
| `extract` | Retrieve, then extract claims (stops before the graph). |
| `graph` | Build and inspect the evidence graph (stops before belief). |
| `reason` | Run reasoning on a fixed synthetic example and print the trace. |
| `inspect-config` | Print the resolved configuration as JSON. |
| `doctor` | Report optional-dependency availability. |
| `gpu-readiness` | Report torch/CUDA/MPS, selected device, cache dir, free disk. |
| `experiment` | Run/analyze evaluation experiments; includes `matrix` (dry-run). |

`egrag experiment matrix` plans a benchmark run from a frozen config and prints the
plan without running inference. Its `--execute` path is disabled here on purpose;
the matrix is intended for a GPU machine.

```bash
uv run egrag gpu-readiness --device auto
uv run egrag experiment matrix --benchmark fever --dry-run \
  --sample artifacts/benchmark-calibration/samples/fever-dev-100.json \
  --output-dir artifacts/final-matrix/out
```

## Relations

Edge types are `egrag.domain.models.RelationType`:

| Enum | Role |
|------|------|
| `SUPPORT` | Directional entailment; contributes positive belief. |
| `CONTRADICTION` | Symmetric disagreement; forms conflict sets, negative belief. |
| `DUPLICATE` | Near-paraphrase; preserved but not counted as independent support. |
| `DEPENDENCY` | One claim depends on another; used for connectivity. |
| `SUPERSESSION` | A newer claim supersedes an older one under an explicit policy. |
| `BRIDGES` | Query-conditioned connectivity for multi-hop reasoning; does not affect belief, conflicts, or corroboration. |
| `NEUTRAL` | Classification outcome; not stored as an edge in normal construction. |

## System variants

The experiment harness (`egrag.experiments.variants`) defines the ablation set. All
variants share the same retriever, chunker, extractor, classifier, and generator,
so a comparison isolates one component.

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

## Models

Real models are optional (`local-models` extra), loaded lazily, and never
downloaded at import time.

- **NLI:** `roberta-large-mnli` (revision `2a8f12d2…`). Its label mapping is
  validated at startup; controlled support/contradiction/neutral/duplicate cases
  are correct.
- **Extraction / generation (smoke):** `Qwen2.5-0.5B-Instruct`. Small enough to run
  on CPU for adapter checks. It follows the JSON output contract once the chat
  template is applied, but is not faithful enough for evaluation.
- **Extraction / generation (recommended, GPU):** `Qwen2.5-7B-Instruct` (or 3B if
  VRAM is limited), bfloat16 on CUDA. These are recommendations for this project's
  setup, not general claims about the best model.

Device is explicit (`cpu` / `mps` / `cuda` / `cuda:N` / integer / `auto`); `auto`
resolves CUDA → MPS → CPU. See `docs/models.md`.

## Benchmarks

- **FEVER** — `copenlu/fever_gold_evidence`, the gold-evidence setting (evidence
  sentences are supplied; there are no distractors). Loads offline.
- **HotpotQA** — `hotpotqa/hotpot_qa` fullwiki validation (parquet; needs the
  `benchmarks` extra). In the fullwiki split the gold supporting paragraphs are not
  guaranteed to be in the provided context, so evidence-coverage metrics are proxy
  measurements on the subset where the full chain is present.

Keep these apart: bounded smoke tests, deterministic structural pilots (fake
generator; they measure evidence selection and graph structure, not answer
quality), development calibration, and a final benchmark run with real models. Only
the first three have been done. See `docs/benchmarks.md` and
`docs/experiments.md`.

## Reproducibility

Stochastic components take an explicit seed and default to deterministic decoding;
the same input and seed give the same output. Evidence packages carry a run
manifest (version, config hash, seeds, adapter identities, input hash), and
serialization is versioned. Frozen benchmark settings and their checksums live in
`artifacts/benchmark-calibration/frozen-configs/`; fixed development samples live in
`artifacts/benchmark-calibration/samples/`. `docs/reproduction.md` gives exact
commands.

## Testing

```bash
uv run ruff format --check .
uv run ruff check .
uv run mypy src
uv run pytest
uv build
```

Tests run offline; a session fixture blocks network access. Tests that need an
optional extra are marked `requires_<extra>` and skip when it is absent. Core
import-isolation tests check that importing core packages pulls in no optional
model library.

## Limitations

- No final benchmark matrix has been run; there are no answer-quality benchmark
  numbers in this repository.
- The `Qwen2.5-0.5B-Instruct` model is not faithful enough for evaluation
  (span copying, abstention on insufficient evidence). A larger model on a GPU is
  needed, followed by re-running the smoke checks.
- The HotpotQA fullwiki split has limited gold-evidence coverage; FEVER uses the
  gold-evidence setting rather than open retrieval.
- Deterministic pilots use the fake generator and report structure, not answers.
- `reranked_passage_rag` currently applies an identity reranker over BM25 order.

## Citation

Placeholder until publication; venue, year, and identifiers are intentionally
omitted.

```bibtex
@misc{roufas_egrag,
  author = {Nikolaos Roufas},
  title  = {EG-RAG: Evidence Graph Retrieval-Augmented Generation},
  note   = {Preprint in preparation}
}
```

## License

`pyproject.toml` declares the MIT license, but no `LICENSE` file has been added
yet. Until that file exists, treat the licensing as incomplete.

## Documentation

- `docs/architecture.md` — layers, ports/adapters, pipeline internals.
- `docs/models.md` — model adapters, chat templates, JSON recovery, devices.
- `docs/caching.md` — cache backends, keys, invalidation.
- `docs/benchmarks.md` — dataset adapters, splits, metrics, caveats.
- `docs/experiments.md` — evaluation method, variants, fairness, runtime.
- `docs/reproduction.md` — exact commands (CPU, MPS, CUDA/WSL2).
- `docs/development.md` — conventions for contributors.
