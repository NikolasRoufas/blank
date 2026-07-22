
# EG-RAG Architecture

**Status:** Implemented (local, unreleased). The domain models, ports/adapters,
retrieval, atomic-claim extraction, evidence-graph construction, reasoning
(initial scoring, signed belief propagation, conflict resolution, subgraph
selection), model-agnostic evidence serialization + generation + grounding, the
experiment/evaluation harness, the real-NLI evidential path (offline
`roberta-large-mnli`), and the query-conditioned `BRIDGES` connectivity relation
are all implemented and tested. Earlier revisions described a planning-phase
empty repository; that is no longer accurate.

**Relation families (current):** the graph carries two distinct families.
*Evidential* relations (`SUPPORTS`, `CONTRADICTS`, `DUPLICATE`, `SUPERSEDES`)
affect belief, propagation, conflict detection, corroboration, and temporal
interpretation. The *reasoning-connectivity* relation (`BRIDGES`) is
query-conditioned, aids only subgraph connectivity / multi-hop coverage, and is
ignored by propagation, conflict detection, and corroboration
(see `docs/bridge-relations.md`).

**Scope:** This document describes the architecture: package boundaries, public
interfaces, the repository layout, the dependency policy, and the reproducibility
and caching strategies. Contributor conventions are in `docs/development.md`; the
evaluation method is in `docs/experiments.md`.

EG-RAG is generator-agnostic through standardized adapters for compatible
text-generation models; it is not claimed to work with every model.

---

## 1. Repository layout

The project uses a `src/` layout. The `egrag` package is organized into the layers
described below; source is under `src/egrag/`, tests under `tests/`, evaluation
records under `artifacts/`, and configuration examples under `configs/`.

---

## 2. Domain overview and pipeline

EG-RAG transforms a natural-language query into a grounded answer by routing
evidence through a typed, inspectable pipeline. The fourteen contractual stages
map onto pipeline steps as follows:

| # | Stage | Responsible component |
|---|-------|----------------------|
| 1 | Accept query | `pipeline` entrypoint / CLI |
| 2 | Retrieve passages | `Retriever` adapter |
| 3 | Rerank and filter | `Reranker` + filter policy |
| 4 | Decompose into atomic claims | `ClaimExtractor` |
| 5 | Preserve provenance + temporal metadata | domain models (`Provenance`, `Claim`) |
| 6 | Construct evidence graph | `GraphBuilder` over `GraphBackend` |
| 7 | Identify relations | `RelationClassifier` |
| 8 | Assign belief + query-utility | `scoring` (distinct estimators) |
| 9 | Propagate evidence | `propagation` (experimental) |
| 10 | Identify + resolve conflict sets | `conflict` (experimental) |
| 11 | Select reasoning subgraph | `selection` (experimental) |
| 12 | Serialize evidence package | `io.serialization` (versioned) |
| 13 | Pass to generation backend | `Generator` adapter |
| 14 | Return answer + citations + uncertainty + trace | domain `Answer` + `EvidenceTrace` |

The pipeline is a sequence of pure transformations over immutable-ish Pydantic
models. Each step takes a typed input artifact and produces a typed output
artifact. Adapters supply the side-effecting capabilities (retrieval,
embedding, generation) behind protocols.

---

## 3. Layered architecture and dependency direction

EG-RAG uses a strict, one-directional dependency graph. **Arrows point in the
direction a module is *allowed* to depend.**

```
        cli  ──────────────┐
                           ▼
        config ────────► pipeline ──────► domain  ◄────── adapters
                           │                ▲                 │
                           └────────────────┘                 │
                                  io ───────────────────────► domain
                                                              (adapters
                                                              implement
                                                              domain protocols)
```

Rules:

- **`domain`** depends on *nothing* in the project and only on the standard
  library + Pydantic. It must not import NetworkX, Transformers,
  sentence-transformers, HTTP clients, vector databases, numerical ML stacks,
  or storage libraries.
- **`adapters`** depend on `domain` (to implement its protocols and use its
  models) and on the third-party library they wrap. Adapters never depend on
  `pipeline` or `cli`.
- **`pipeline`** depends on `domain` only. It is wired with adapter *instances*
  injected at the composition root; it never imports adapter modules directly.
- **`io`** (serialization, caching) depends on `domain` only.
- **`config`** produces typed settings; it depends on `domain` for shared
  enums/value types but not on adapters.
- **`cli`** is the composition root: it reads `config`, instantiates concrete
  adapters, and injects them into `pipeline`. It is the only layer permitted to
  know which concrete adapter is in use.

This guarantees the contractual constraint: *provider-specific details never
leak into the domain layer.* A `mypy` import-graph check and a lightweight test
(see §11) enforce it.

---

## 4. Package boundaries

```
egrag
├── domain          # Pure core. No third-party deps except Pydantic.
│   ├── models      # Pydantic v2 data models (Query, Passage, Claim, ...)
│   ├── protocols   # Typed Protocol / ABC interfaces for all capabilities
│   ├── scoring     # Pure scoring math: belief, utility, reliability (distinct)
│   ├── graph       # Abstract graph interface + graph algorithms (backend-free)
│   └── errors      # Exception hierarchy
├── pipeline        # Orchestration of stages over domain protocols
│   ├── steps       # One module per pipeline stage
│   └── orchestrator.py
├── adapters        # Concrete implementations behind extras
│   ├── retrieval   # bm25 (pure-python baseline), dense (sentence-transformers)
│   ├── reranking
│   ├── extraction
│   ├── relations
│   ├── generation  # transformers (local), openai_compat (HTTP)
│   └── graph       # networkx backend
├── io              # Serialization (versioned) and caching
│   ├── serialization.py
│   └── cache
├── config          # Pydantic Settings
├── cli             # Typer app — composition root / dependency injection
└── experimental    # Research algorithms behind unstable, clearly-marked API
    ├── propagation
    ├── conflict
    └── selection
```

> **Stable vs. experimental.** `domain`, `pipeline` interfaces, `io`
> serialization format, `config`, and the adapter protocols form the **stable
> public surface**. `experimental` holds research algorithms whose methodology
> is not yet validated (see §7 and §8). Stable code may call experimental
> algorithms only through a stable protocol; experimental code may not be
> imported directly by the public API.

---

## 5. Core data model

All models are Pydantic v2, JSON-serializable, and carry a schema version where
they participate in the persisted evidence package. The five contractually
**distinct** quantities are modeled as separate fields and never collapsed into
one "score":

1. **`belief`** — estimated probability the claim is true given current evidence.
2. **`extraction_confidence`** — confidence that the claim was correctly
   extracted from its source span.
3. **`relation_confidence`** — confidence attached to each edge/relation.
4. **`source_reliability`** — trust assigned to the originating source
   (configurable input, *not* a scientific ground truth).
5. **`query_utility`** — estimated usefulness of the claim for answering *this*
   query (relevance), independent of its truth.

Key models (fields abbreviated):

```text
Query            { text, id, params }
SourceRef        { source_id, title?, uri?, reliability_prior?, timestamp? }
Span             { source_id, start, end, text }
Passage          { id, text, source: SourceRef, retrieval_score, rank }
Provenance       { spans: list[Span], source: SourceRef, observed_at? }
TemporalMeta     { asserted_at?, valid_from?, valid_to?, observed_at? }
Claim            { id, text, provenance: Provenance, temporal: TemporalMeta,
                   extraction_confidence: float,
                   belief: BeliefState, query_utility: float }
BeliefState      { value: float, support: float, contradiction: float,
                   method_id: str }            # how it was computed
Relation         { id, kind: RelationKind, source_claim, target_claim,
                   relation_confidence: float, rationale?, method_id }
RelationKind     = support | contradiction | duplicate | dependency | supersession
ConflictSet      { id, claims: list[ClaimId], relations, resolution? }
EvidenceGraphView (read model over the graph backend)
EvidencePackage  { schema_version, query, selected_claims, relations,
                   conflicts, uncertainty, run_manifest }
RunManifest      { egrag_version, config_hash, seeds, adapter_identities,
                   input_hash, created_at }
Answer           { text, citations: list[Citation], uncertainty: Uncertainty,
                   evidence_trace: EvidenceTrace }
Citation         { claim_id, spans }
Uncertainty      { answer_confidence?, unresolved_conflicts, abstained: bool }
EvidenceTrace    { steps: list[StepRecord] }   # inspectable per-stage record
```

Retrieved text is treated as **untrusted data**: claim/relation extraction
validates all model output (§ external-output validation) before it enters the
graph, and source text is never executed or interpolated into prompts without
escaping.

---

## 6. Public interfaces (protocols)

All model-facing capabilities are typed `Protocol`s (structural) in
`domain.protocols`. Adapters implement them. Each protocol method is fully type
annotated. Stochastic implementations accept an explicit seed via construction
config.

```python
class EmbeddingModel(Protocol):
    identity: AdapterIdentity
    def embed(self, texts: Sequence[str]) -> list[Vector]: ...

class Retriever(Protocol):
    identity: AdapterIdentity
    def retrieve(self, query: Query, k: int) -> list[Passage]: ...

class Reranker(Protocol):
    identity: AdapterIdentity
    def rerank(self, query: Query, passages: Sequence[Passage]) -> list[Passage]: ...

class ClaimExtractor(Protocol):
    identity: AdapterIdentity
    def extract(self, passage: Passage) -> list[Claim]: ...

class RelationClassifier(Protocol):
    identity: AdapterIdentity
    def classify(self, a: Claim, b: Claim) -> list[Relation]: ...

class Generator(Protocol):
    identity: AdapterIdentity
    def generate(self, package: EvidencePackage, decoding: DecodingParams) -> RawGeneration: ...

class GraphBackend(Protocol):           # abstracts NetworkX
    def add_node(self, claim_id: ClaimId, data: Mapping[str, object]) -> None: ...
    def add_edge(self, src: ClaimId, dst: ClaimId, data: Mapping[str, object]) -> None: ...
    def neighbors(self, claim_id: ClaimId) -> Iterable[ClaimId]: ...
    def subgraph(self, claim_ids: Collection[ClaimId]) -> "GraphBackend": ...
    # plus read-only traversal primitives used by domain.graph algorithms

class Cache(Protocol):
    def get(self, key: str) -> bytes | None: ...
    def set(self, key: str, value: bytes) -> None: ...
```

Experimental algorithms expose **stable** protocols so they remain swappable:

```python
class BeliefPropagator(Protocol):       # experimental implementations
    method_id: str
    def propagate(self, graph: GraphBackend, claims: Mapping[ClaimId, Claim]) -> Mapping[ClaimId, BeliefState]: ...

class ConflictResolver(Protocol):
    method_id: str
    def resolve(self, conflict: ConflictSet, claims: Mapping[ClaimId, Claim]) -> ConflictResolution: ...

class SubgraphSelector(Protocol):
    method_id: str
    def select(self, graph: GraphBackend, claims, budget: SelectionBudget) -> EvidenceGraphView: ...
```

`AdapterIdentity { name, version, model_id?, params_hash }` feeds the
`RunManifest` for reproducibility. `method_id` records which algorithm produced
a belief/relation/resolution so results stay auditable and never conflated
across methods.

---

## 7. Scientific uncertainties and algorithmic risks

These are explicitly **research** concerns. They are isolated in
`egrag.experimental` and documented as such. Configurable baselines are
**baselines**, not claimed universal truths.

1. **Claim atomicity is under-defined.** There is no ground-truth segmentation
   of text into "atomic" claims. The baseline extractor is a heuristic; its
   granularity is a tunable policy, not a scientific constant.
2. **Contradiction / relation detection is error-prone.** NLI-style
   classification produces false positives/negatives. `relation_confidence` is
   propagated, never assumed certain. Contradictions are **never silently
   discarded** (contractual).
3. **Belief propagation methodology is unsettled.** Bayesian, Dempster–Shafer,
   and iterative trust-propagation schemes make different independence and
   calibration assumptions. The default propagator is a documented, deterministic
   baseline labeled experimental; alternatives are pluggable via `BeliefPropagator`.
4. **Recency ≠ truth.** Supersession uses temporal metadata as *one signal*.
   Newer evidence must **not** automatically be treated as more truthful;
   supersession only applies under an explicit, configurable policy and records
   its rationale.
5. **Source independence.** Repetition of a claim from one source (or copies of
   one source) must **not** be counted as independent corroboration.
   Duplicate detection runs *before* support is aggregated; corroboration counts
   distinct sources, not distinct mentions.
6. **Subgraph selection is combinatorial.** Optimal compact-coherent-subgraph
   selection is intractable in general; the default selector is an approximation
   with a stated objective and budget. Quality is an open question.
7. **Score calibration.** `belief` and `query_utility` are estimates; absolute
   calibration is not guaranteed. They are reported with their `method_id` so
   downstream consumers can recalibrate.
8. **Generation faithfulness.** Even a well-grounded package can be ignored by a
   generator. The generator is *instructed not to invent unsupported evidence*,
   and outputs are validated/cited against the package; remaining
   unfaithfulness is surfaced in `Uncertainty`, not hidden.

---

## 8. Stable vs. experimental separation

| Concern | Classification | Location |
|--------|----------------|----------|
| Data models & schema versioning | Stable | `domain.models`, `io.serialization` |
| Capability protocols | Stable | `domain.protocols` |
| Pipeline orchestration contract | Stable | `pipeline` |
| Retrieval / rerank / extraction *interfaces* | Stable | `domain.protocols` |
| BM25 baseline, dense adapter | Stable adapters | `adapters.retrieval` |
| Graph backend interface | Stable | `domain.protocols`, `domain.graph` |
| Scoring *definitions* (the 5 distinct quantities) | Stable | `domain.scoring` |
| Belief propagation algorithm | **Experimental** | `experimental.propagation` |
| Conflict-set resolution policy | **Experimental** | `experimental.conflict` |
| Reasoning-subgraph selection | **Experimental** | `experimental.selection` |
| Claim-decomposition heuristics | **Experimental** | `experimental` / `adapters.extraction` baseline |

Experimental modules carry a module-level docstring warning, are excluded from
semantic-versioning guarantees, and are reachable from the pipeline only via the
stable `BeliefPropagator` / `ConflictResolver` / `SubgraphSelector` protocols.

---

## 9. Dependency policy: core vs. optional extras

**Core (always installed)** — pure-Python / lightweight, no model downloads,
no network, no GPU:

- `pydantic` (v2), `pydantic-settings`, `typer`, `networkx`.
- A **pure-Python BM25 baseline** so a core-only install is fully runnable
  end-to-end (with the deterministic fake generator).

> NetworkX is pure-Python and lightweight; it is core as the default
> `GraphBackend` implementation. The **domain layer still never imports it** —
> only `adapters.graph` does.

**Optional extras** (via `uv` / PEP 621 `[project.optional-dependencies]`):

| Extra | Pulls in | Enables |
|-------|----------|---------|
| `sparse` | `rank-bm25` | Optimized BM25 retriever |
| `dense` | `sentence-transformers` (+ torch) | Dense embedding retriever/reranker |
| `transformers` | `transformers` (+ torch) | Local HF text generation |
| `openai` | `httpx` | OpenAI-compatible HTTP generation |
| `all` | all of the above | Everything |
| `dev` | `pytest`, `pytest-cov`, `hypothesis`, `ruff`, `mypy`, type stubs | Development & CI |

CI verifies (a) the package builds and (b) a **core-only** install imports and
runs an end-to-end pipeline using deterministic fakes/baselines — without any
optional model dependency.

---

## 10. Repository tree

Main directories and files (abridged):

```text
pyproject.toml            # uv-managed; core deps + extras
uv.lock
README.md
docs/                     # architecture, models, caching, benchmarks, experiments, …
configs/                  # example run configs
scripts/                  # evaluation scripts
artifacts/                # evaluation records (manifests, samples, reports)
src/egrag/
├── domain/               # models, ports; stdlib + Pydantic only
├── application/          # pipeline orchestration over ports
├── adapters/
│   ├── retrieval/        # BM25, dense, hybrid, chunking
│   ├── reranking/
│   ├── extraction/       # sentence + structured claim extractors
│   └── graph/
├── graph/                # relation classification, graph construction
├── reasoning/            # scoring, propagation, conflict, selection, budgets
├── generation/           # renderers, adapters, service, parsing, validation
├── caching/              # in-memory, null, disk backends; key builders
├── experiments/          # datasets, variants, runner, metrics, benchmarks
├── config/  serialization/  observability/  fakes/  experimental/
├── answering.py  composition.py  security.py  reproducibility.py
├── hf_runtime.py  structured_json.py
└── cli/
tests/                    # unit, integration, e2e, property, sanity; conftest
```

The `src/` layout means tests run against the installed package and import
isolation is enforced (no `egrag` on the path by accident).

---

## 11. Dependency-rule checks

- The dependency direction (§3) is acyclic and enforced by a test that inspects
  `egrag.domain.*` imports and fails on any forbidden third-party or upward
  intra-project import.
- Domain imports only the standard library and Pydantic; verified by the same
  test and by `mypy`.
- Implemented and experimental algorithms are separated (§7, §8): experimental
  code is quarantined and labelled.

The remaining sections cover testing, reproducibility, caching, and assumptions.

---

## 12. Former implementation plan (removed)

The original document contained a staged build plan (M0–M13). The corresponding
functionality is implemented; the plan has been removed to avoid confusion with
the current state. Component status is summarized in
`docs/implementation-status.md`.

---

## 13. Testing strategy (summary)

Full detail in `docs/testing-strategy.md`. In brief: unit tests per module with
deterministic fakes; integration tests across adjacent stages; offline e2e tests
on tiny fixed corpora; property-based tests (Hypothesis) for serialization
round-trips, scoring monotonicity/bounds, and graph invariants; **scientific
sanity tests** encoding the contractual invariants (contradictions never
dropped, recency ≠ truth, single-source repetition ≠ corroboration, distinct
score concepts); strict **network isolation** (a session-wide autouse fixture
blocks sockets); optional-dependency tests skipped cleanly when an extra is
absent and exercised in a dedicated CI job; regression tests with golden
artifacts for serialized packages.

---

## 14. Reproducibility strategy

- **Explicit seeds** flow from `config` into every stochastic component; no
  hidden RNG. Default decoding is deterministic (greedy / temperature 0).
- **Locked dependencies** via `uv.lock`; Python pinned to 3.12.
- **Run manifest** records `egrag` version, config hash, seeds, per-adapter
  `AdapterIdentity` (incl. model id + params hash), and input hash; it is
  embedded in every `EvidencePackage` and `Answer.evidence_trace`.
- **Versioned, JSON-compatible serialization** lets a run be saved and
  **replayed**; a regression test asserts that replaying a saved manifest yields
  identical output for deterministic adapters.
- **No network / no download at import**; models are explicit, user-provided
  inputs. Tests never touch the network or large models.

---

## 15. Caching strategy (high level)

- A `Cache` protocol with an **in-memory** default (core) and optional disk/
  SQLite backends. Caching is transparent and **never affects correctness** —
  a cold cache and warm cache must produce identical results.
- **Content-addressed keys**: `hash(stage_id + input_payload + adapter_identity
  + relevant_config + seed + decoding_params)`. Changing any of these misses the
  cache, preventing stale cross-model/cross-version reuse.
- **Layered** caches for the expensive stages: embeddings, retrieval, claim
  extraction, relation classification, and generation.
- Cache values are the **validated** typed artifacts (serialized), so cached
  data re-enters the pipeline through the same validation path as fresh data.

---

## 16. Assumptions (require review)

1. **Language:** initial extraction/relation baselines target English; broader
   language support is future work.
2. **Modality:** passages are plain text; no images/tables/multimodal in scope.
3. **Source reliability** is a configurable prior supplied by the caller or
   defaulting to uniform; it is *not* a scientific ground truth and must not be
   presented as one.
4. **Temporal metadata** is used when available and treated as optional; absence
   must not break the pipeline, and presence must not by itself imply truth.
5. **NetworkX as core** default `GraphBackend` is acceptable given it is pure
   Python and lightweight; the domain layer remains backend-agnostic.
6. **numpy** may be adopted as a core dependency for scoring vector math; if a
   strictly minimal core is preferred, scoring falls back to pure Python. (Open
   decision — flag for review.)
7. **Offline operation:** users pre-download any model required by an ML extra;
   EG-RAG performs no automatic downloads and no network calls at import.
8. **No GPU required:** all defaults and tests run on CPU.
9. **Claim atomicity** has no ground truth; the baseline granularity is a
   policy, evaluated by sanity tests rather than against a gold standard.
10. **Generator compatibility** is limited to models exposing the adapter
    contract (local HF Transformers or OpenAI-compatible HTTP); "generator-
    agnostic" is scoped to these standardized adapters.
```
