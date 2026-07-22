# Troubleshooting

## Installation & environment

- **`ModuleNotFoundError` for `torch`/`transformers`/`httpx`/`sentence_transformers`/`networkx`.**
  These are optional extras. Install the one the error names, e.g.
  `uv pip install 'egrag[local-models]'`. The framework surfaces a
  `MissingDependencyError` with the exact install command before doing work.
- **`egrag` import fails after editing.** Recreate the environment:
  `rm -rf .venv && uv sync`. Tests also run with `PYTHONPATH=src`.
- **`ruff` hangs or won't start (local disk issue).** Run the pinned ruff via
  `uvx ruff@<version> format --check .` / `uvx ruff@<version> check .`.

## Configuration

- **`ConfigurationError: invalid YAML`.** The file isn't valid YAML, or its root
  isn't a mapping. Only safe YAML is accepted.
- **`ConfigurationError: ... extra fields not permitted`.** A key is misspelled
  or unsupported — unknown fields are forbidden by design. Check the section
  names in `configs/baseline.yaml`.
- **`ConfigurationError: ... exceeds context_limit`.** Lower
  `generation.evidence_token_budget`/`max_new_tokens` or raise `context_limit`.
- **`ModelLoadError: model path does not exist`.** A local model path is wrong;
  fix `generation.model` or use a model identifier.

## Runtime

- **`GenerationError: empty model output` / `malformed structured output`.** The
  generator did not return the JSON contract; check the adapter/model.
- **`GenerationError: ... exceeds the adapter context limit`.** The rendered
  evidence is too large for the model; reduce the evidence budget.
- **Determinism warning in the manifest.** The chosen adapter cannot guarantee
  deterministic decoding; results may vary. Use a deterministic-capable adapter
  for evaluation.
- **`ConvergenceError`.** Belief propagation did not converge; raise
  `propagation.max_iterations`, increase `damping`, or set
  `propagation.on_nonconvergence: return` to get an explainable (non-converged)
  result instead of an error.

## Security

- **`SecurityError: path ... escapes the allowed base directory`.** Artifact and
  corpus paths must stay within the configured `security.artifact_base`.
- **`SecurityError: URL scheme ... is not allowed`.** Only `http`/`https` are
  permitted by default; adjust `security.allowed_url_schemes` deliberately.
- **`SecurityError: file ... exceeds the limit`.** Raise
  `corpus.max_file_bytes`/`security.max_file_bytes` only if you trust the input.
