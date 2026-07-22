# Contributing to EG-RAG

EG-RAG is a research framework; correctness, scientific validity, and
reproducibility come before features. `docs/development.md` records the
engineering and scientific-integrity conventions the code follows.

## Local development

```bash
uv sync                         # core + dev dependencies, Python 3.12
uv run ruff format --check .    # formatting
uv run ruff check .             # lint
uv run mypy src                 # static types
uv run pytest                   # tests (offline, no model downloads)
uv build                        # build sdist + wheel
```

> Note: if your environment's `.venv` `ruff` binary cannot run (a known local
> disk issue on some machines), use `uvx ruff@<version> ...` instead — it is the
> identical pinned ruff from uv's tool cache.

### Optional dependencies

Core is pure-Python and offline. Install extras only when you need them:

```bash
uv pip install 'egrag[retrieval]'      # sparse BM25 helper
uv pip install 'egrag[dense]'          # sentence-transformers
uv pip install 'egrag[graph]'          # NetworkX (GraphML export)
uv pip install 'egrag[local-models]'   # transformers (+ torch)
uv pip install 'egrag[http-models]'    # httpx (OpenAI-compatible server)
uv pip install 'egrag[experiments]'    # numpy
```

### Pre-commit hooks

```bash
uv run pre-commit install   # runs ruff format/check + mypy on commit
```

## Test categories

Tests are marked (see `pyproject.toml [tool.pytest.ini_options].markers`):

- `unit` — isolated component behavior with deterministic fakes.
- `integration` — adjacent stages / full fake pipeline.
- `e2e` — CLI end to end via Typer's `CliRunner`.
- `property` — Hypothesis-based invariants.
- `sanity` — scientific-integrity invariants.
- `requires_<extra>` — needs an optional extra; skipped cleanly when absent.

Run a subset, e.g. `uv run pytest -m unit`. All tests run offline and never
download models or open sockets (a session fixture blocks the network).

## Extending the framework

Every model-facing capability is a typed protocol. To add an implementation:

1. Implement the relevant protocol (e.g. `egrag.domain.ports.Retriever`,
   `egrag.generation.interfaces.TextGenerator`, `egrag.graph.types.PairClassifier`).
2. Keep provider-specific imports lazy and inside the adapter — never at import
   time, never in `egrag.domain`.
3. Wire it in the composition root (`egrag.composition`), not in scattered
   factories.
4. Add deterministic fakes and tests.

See `docs/extending.md` for worked examples.

## Conventions

- Domain layer imports only the standard library + Pydantic.
- No hidden global state; no import-time side effects, model loads, or network.
- Treat retrieved text as untrusted; never execute it.
- New serialized fields bump `SCHEMA_VERSION` and stay backward compatible.

## Before opening a PR

Run all five gates above; add tests for new behavior; update docs/ADRs.
Do not weaken security or grounding tests to make things pass.
