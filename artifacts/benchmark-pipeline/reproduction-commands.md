# Reproduction Commands — benchmark pipeline

Offline / CPU. Set `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1` for model/data steps.

## Dataset adapters (offline)

```python
from egrag.experiments.benchmarks import FeverGoldEvidenceDataset, validate_benchmark
exs = FeverGoldEvidenceDataset(split="valid", limit=50).load()   # cached JSONL
assert validate_benchmark(exs) == []
```

HotpotQA (after installing a parquet reader with network):

```bash
uv pip install pyarrow      # or add a datasets/pyarrow extra and: uv sync --extra ...
python -c "from egrag.experiments.benchmarks import HotpotQADataset; print(len(HotpotQADataset(limit=25).load()))"
```

## Tests

```bash
PYTHONPATH=src .venv/bin/pytest tests/integration/test_benchmark_pipeline.py
# cache-gated real FEVER load:
EGRAG_RUN_LOCAL_MODELS=1 PYTHONPATH=src .venv/bin/pytest -k real_cached_fever
```

## Runtime estimate (needs comfortable free disk; may wedge otherwise)

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH=src \
  .venv/bin/python scripts/estimate_runtime.py
```

## Quality gates

```bash
uvx ruff@0.15.18 format --check .
uvx ruff@0.15.18 check .
PYTHONPATH=src .venv/bin/mypy src
PYTHONPATH=src .venv/bin/pytest
uv build
```

## Pilot sequence (run only after disk + parquet unblocked)

1. dataset smoke → 2. claim-extraction smoke → 3. generator smoke → 4. NLI smoke →
5. one HotpotQA example → 6. one FEVER example → 7. 5-example stratified →
8. 25-example → 9. runtime estimate → 10. optional 100-example.
Validate outputs before each scale-up; stop on the section-12 conditions.
