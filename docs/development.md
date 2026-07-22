# Development

Conventions for working on EG-RAG. `CONTRIBUTING.md` covers the day-to-day
commands; this page records the design rules the code follows.

## Priorities

When these conflict, resolve them in this order: correctness, scientific validity,
clear interfaces, testability, reproducibility, maintainability, reasonable
performance, extensibility. Avoid premature optimization.

## Layering

Dependencies point inward: `cli → application → domain`, `adapters → domain`,
`io/config → domain`. The domain layer imports only the standard library and
Pydantic. It does not import NetworkX, Transformers, sentence-transformers, HTTP
clients, vector stores, or storage libraries. Adapters wrap those and implement the
domain protocols in `egrag.domain.ports` (and the narrower protocols in
`egrag.generation.interfaces`, `egrag.graph.types`, `egrag.adapters.extraction.interfaces`).

Every model-facing capability is a typed protocol; public functions and classes
are annotated. There is no hidden global state, and no module does expensive work,
loads a model, or touches the network at import time. Optional integrations sit
behind dependency extras so a core install stays pure Python and offline. Add an
abstraction only when there are at least two implementations or a real
architectural boundary.

Experimental algorithms (belief propagation, conflict resolution, subgraph
selection, claim-decomposition heuristics) live in `egrag.experimental` and are
reached only through stable protocols. Configurable baselines are not presented as
universally correct methods.

## Scientific-integrity rules

These are enforced by the design and by `sanity`-marked tests:

- Keep five quantities distinct; never collapse them: `belief`,
  `extraction_confidence`, `relation_confidence`, `source_reliability`,
  `query_utility`.
- Contradictory evidence is not silently dropped. It appears in the graph, in
  conflict sets, and in the answer's uncertainty.
- Newer evidence is not automatically more true. Supersession applies only under an
  explicit policy and records its rationale.
- Repetition from one source is not independent corroboration. Duplicates are
  detected before support is counted; corroboration counts distinct sources.
- Every claim keeps provenance to at least one source span, end to end.
- Retrieved and source text is untrusted: validate external model output before
  use, and never let source text change control flow.
- The generator is instructed not to invent unsupported evidence; unfaithfulness is
  surfaced as uncertainty, not hidden.

## Reproducibility

Thread explicit seeds from config into every stochastic component; decoding is
deterministic by default, so the same input and seed produce the same output. Each
evidence package records a run manifest (version, config hash, seeds, adapter
identities, input hash), and serialization is versioned and replayable. The core
pipeline runs on CPU, offline.

## Testing

Use the deterministic fakes in `egrag.fakes`; tests do not use the network, a GPU,
or large downloads, and a session fixture blocks sockets. Tests that need an extra
are marked `requires_<extra>` and skip when it is absent. The scientific invariants
above are encoded as `sanity` tests. Every bug fix gets a regression test first. A
schema change bumps `SCHEMA_VERSION` and keeps older serialized artifacts
readable.

## Keeping the tree green

If a quality gate fails, fix the cause rather than silencing it: no broad
`# type: ignore`, blanket `noqa`, or `skip`/`xfail` without a written reason (and,
for a skip, a removal condition). When you change an interface, update its
implementers, tests, and docstrings in the same change, and prefer small,
independently testable steps. See `CONTRIBUTING.md` for the gate commands and
`docs/extending.md` for worked examples of adding an adapter.
