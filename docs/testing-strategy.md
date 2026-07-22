# EG-RAG Testing Strategy

**Status:** Proposal (planning phase).
**Goal:** Make every subsystem testable offline, deterministically, without a
GPU, without network access, and without large model downloads — while encoding
the project's scientific-integrity invariants as executable tests.

Quality gates that every change must pass:

```
uv run ruff format --check .
uv run ruff check .
uv run mypy src
uv run pytest
```

CI additionally verifies that the package **builds** and that a **core-only**
installation imports and runs an end-to-end pipeline with no optional model
dependency.

---

## 1. Test taxonomy and layout

```
tests/
├── unit/          # one module/behavior under test, fakes for collaborators
├── integration/   # two or more adjacent stages wired together
├── e2e/           # full pipeline on a tiny fixed corpus, fakes only
├── property/      # Hypothesis-based invariants
├── fakes/         # deterministic fake adapters (shared)
└── data/          # tiny, fixed, committed corpora & golden artifacts
```

Markers (registered in `pyproject.toml`, `--strict-markers`):

- `unit`, `integration`, `e2e`, `property`
- `sanity` — scientific-integrity invariants (§9)
- `requires_<extra>` — needs an optional dependency (`requires_dense`,
  `requires_transformers`, `requires_openai`, `requires_sparse`); skipped if the
  import is unavailable.

Default `pytest` run executes everything that does not need a missing optional
dependency. Coverage via `pytest-cov`; the `domain` layer carries the highest
coverage expectation because it is pure and deterministic.

---

## 2. Unit tests

- Cover each `domain` module in isolation: models (validation, defaults,
  bounds), scoring estimators, graph algorithms (against a fake `GraphBackend`),
  serialization, errors.
- Adapters are unit-tested against their **own** logic using fakes for the
  wrapped library where the real library is an optional extra (e.g., the dense
  adapter's vector handling tested with a fake embedding function); behavior
  that genuinely needs the real library is marked `requires_<extra>`.
- Every public function/class is exercised with at least one nominal and one
  boundary/error case. All test functions and fakes are type-annotated.

---

## 3. Integration tests

- Exercise adjacent stages through their real domain interfaces with
  deterministic fakes for the model-facing parts:
  - retrieve → rerank → filter
  - extract → graph construction → relation classification
  - scoring → propagation → conflict detection → resolution
  - selection → evidence-package serialization
- Assert artifact contracts at each boundary: types, provenance preserved,
  temporal metadata carried through, `method_id`/`AdapterIdentity` recorded.

---

## 4. End-to-end tests

- Run the **entire** pipeline (`Query → Answer`) on a tiny committed corpus
  using only core/fake components:
  - pure-Python BM25 (or fake) retriever, rule-based extractor, baseline
    relation classifier, deterministic scoring/propagation/selection, and the
    **deterministic fake generator**.
- Assert: an `Answer` is produced; citations reference real claim spans;
  `Uncertainty` reflects unresolved conflicts; `EvidenceTrace` contains a record
  per stage; the run is reproducible from its `RunManifest`.
- Must complete in seconds on CPU, fully offline.

---

## 5. Property-based tests (Hypothesis)

Use where invariants are general and inputs are easily generated:

- **Serialization round-trip:** `deserialize(serialize(x)) == x` for all
  domain models and the `EvidencePackage`, across schema versions.
- **Scoring bounds & monotonicity:** all scores stay in `[0, 1]`; adding a
  supporting claim from a *new* source does not decrease belief; adding a
  contradicting claim does not increase it.
- **Graph invariants:** construction never drops a claim node; subgraph
  selection returns a subset of existing nodes/edges; no self-edges of an
  illegal kind.
- **Determinism:** the same input + seed yields byte-identical serialized
  output.

---

## 6. Deterministic fake components

Shared in `tests/fakes/`, implementing the domain protocols:

- `FakeRetriever` — returns a fixed, ordered passage set for a query.
- `FakeReranker` — deterministic permutation by a stored key.
- `FakeEmbeddingModel` — maps text to a fixed low-dimensional vector via a hash;
  no model download.
- `FakeClaimExtractor` — splits passages by a deterministic rule.
- `FakeRelationClassifier` — emits relations from a lookup table.
- `FakeGenerator` — composes a templated, **cited** answer purely from the
  evidence package (never invents content), enabling offline e2e.
- `FakeGraphBackend` — in-memory backend to prove `domain.graph` algorithms do
  not depend on NetworkX.
- `FakeCache` — in-memory, instrumented to assert cold/warm equivalence.

Fakes are deterministic, dependency-free, and type-annotated. They are the
default collaborators for unit/integration/e2e tests.

---

## 7. Network isolation

- A **session-wide autouse fixture** disables outbound sockets (patches
  `socket.socket` / connection creation) so any accidental network call fails
  loudly.
- No test performs a real HTTP request or model download. The `openai` adapter
  is tested against a fake transport/mock server object, never a live endpoint.
- CI runs with no network credentials configured for the test job.

---

## 8. Optional-dependency testing

- Tests needing an extra are marked `requires_<extra>` and **skip cleanly**
  (with a clear reason) when the import is unavailable — the default developer
  install must stay green.
- A dedicated CI matrix job installs each extra (`sparse`, `dense`,
  `transformers`, `openai`) and runs its `requires_*` tests, using the smallest
  feasible fixtures and any tiny/cached test artifacts — still no large
  downloads at test time where avoidable.
- A separate **core-only** CI job installs *no* extras and asserts: `import
  egrag` works, the import-isolation test passes, and the offline e2e pipeline
  runs with fakes/baselines.

---

## 9. Scientific sanity tests

Executable encodings of the contractual scientific invariants (marker
`sanity`). These guard methodology, not just code:

1. **Contradictions are never silently discarded** — a contradicting claim
   always appears in the graph, in the relevant `ConflictSet`, and in the
   `Answer.uncertainty`.
2. **Recency ≠ truth** — increasing only a claim's timestamp does not increase
   its belief; supersession applies only under an explicit policy and records a
   rationale.
3. **Single-source repetition ≠ corroboration** — duplicating a claim from the
   same source (or a copied source) does not raise corroboration/belief;
   corroboration counts distinct sources.
4. **Distinct score concepts** — belief, extraction_confidence,
   relation_confidence, source_reliability, and query_utility are independently
   settable and independently observable; no estimator collapses them.
5. **Provenance is total** — every claim retains ≥1 source span end-to-end;
   serialization preserves it.
6. **Untrusted text** — source text containing prompt-injection-like or
   markup-like content does not alter control flow or the generator's
   instruction to avoid unsupported evidence.
7. **Generator faithfulness signal** — when the package lacks support for a
   requested fact, the fake generator path yields an abstention/uncertainty
   rather than fabricated content.

---

## 10. Regression-test policy

- **Golden artifacts:** serialized `EvidencePackage` and `Answer` for fixed
  inputs are committed under `tests/data/`. A change to a golden file requires
  an explicit, reviewed update and a note in the PR/ADR explaining why.
- **Schema versioning:** any change to a serialized model bumps `schema_version`
  and adds a round-trip + migration/back-compat test; old golden artifacts must
  still deserialize.
- **Bug-fix discipline:** every fixed defect gets a regression test reproducing
  it before the fix.
- **Determinism guard:** a CI test re-runs a fixed e2e scenario twice and
  asserts byte-identical serialized output.

---

## 11. Import-isolation test

A dedicated test inspects the import graph of `egrag.domain.*` and **fails** if
any module imports NetworkX, Transformers, sentence-transformers, an HTTP
client, a vector DB, a storage library, or any sibling layer it is not allowed
to depend on (per `architecture.md` §3). This enforces "no provider leakage into
the domain layer" mechanically, alongside `mypy`.

---

## 12. Acceptance checks by capability

The checks below are grouped by capability. They run in the current suite, and
all must pass with the quality gates green. (The `M`-prefixed labels are historical
tags for the order in which capabilities were built.)

- **M0 Scaffold** — gates pass on empty package; `import egrag` works; core-only
  install CI job green; import-isolation test present (trivially passing).
- **M1 Domain models + serialization** — round-trip property tests pass; the
  five distinct scores exist as separate fields; import-isolation test passes;
  `schema_version` present.
- **M2 Protocols + config + CLI + logging** — `egrag --help` works; config loads
  from env/file; `--seed`/`--deterministic` accepted; protocols type-check.
- **M3 Retrieval + rerank** — BM25 baseline ranks the fixture corpus
  deterministically; returned `Passage`es carry provenance; fake retriever used
  in tests.
- **M4 Claim extraction** — every emitted claim has ≥1 source span and an
  `extraction_confidence`; temporal metadata preserved when present.
- **M5 Graph backend + construction** — graph built from claims; `domain.graph`
  algorithms pass against both NetworkX and `FakeGraphBackend`.
- **M6 Relations** — all five relation kinds producible; contradictions retained
  (sanity test #1); duplicates detected before corroboration counting.
- **M7 Scoring** — sanity tests #2, #3, #4 pass; scores bounded; corroboration
  counts distinct sources only.
- **M8 Belief propagation (experimental)** — deterministic given seed; `method_id`
  recorded; termination/convergence tested.
- **M9 Conflict sets + resolution (experimental)** — conflicts surfaced in
  output; supersession only under explicit policy with recorded rationale
  (sanity test #2).
- **M10 Subgraph selection (experimental)** — respects budget; returns a valid
  subset; deterministic; objective documented.
- **M11 Evidence package + generation** — package round-trips; fake generator
  produces a cited answer offline; `requires_transformers`/`requires_openai`
  tests pass in their CI jobs; generator prompt forbids unsupported evidence.
- **M12 End-to-end** — full offline, CPU-only, deterministic e2e produces an
  `Answer` with citations, uncertainty, and an inspectable `EvidenceTrace`.
- **M13 Docs + reproducibility** — replaying a saved `RunManifest` reproduces
  identical output for deterministic adapters; docstrings complete; ADRs added.
```
