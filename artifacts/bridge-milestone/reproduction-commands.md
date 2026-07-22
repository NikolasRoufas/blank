# Reproduction Commands — bridge milestone

Offline, deterministic. CPU only.

## Controlled bridge evaluation (oracle mode; produces all artifacts here)

```bash
PYTHONPATH=src .venv/bin/python scripts/run_bridge_eval.py
```

## Regression tests

```bash
PYTHONPATH=src .venv/bin/pytest tests/integration/test_bridge_milestone.py
```

## Real-NLI evidential pass (optional; requires the local-models extra)

```bash
uv sync --extra local-models
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH=src \
  .venv/bin/python scripts/run_real_nli_eval.py        # evidential relations, real NLI
```

## Quality gates

```bash
uvx ruff@0.15.18 format --check .
uvx ruff@0.15.18 check .
PYTHONPATH=src .venv/bin/mypy src
PYTHONPATH=src .venv/bin/pytest
uv build
```

Notes: the bridge detector and the controlled evaluation are deterministic and
need no model. The real-NLI pass loads `roberta-large-mnli` from the local cache
(`HF_HUB_OFFLINE=1`); the local-model integration test runs only under
`EGRAG_RUN_LOCAL_MODELS=1`.
