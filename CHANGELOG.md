# Changelog

The repository is local and unreleased; the versions below are schema/internal,
not published releases.

## Unreleased

### Added — CUDA server support and real-generator experiments
- `egrag.hf_runtime`: `ensure_device_available` (fails loudly instead of
  silently falling back to CPU when CUDA is explicitly required); 4-bit/8-bit
  `bitsandbytes` quantization (new `quantization` extra); generic
  `chat_template_kwargs` passthrough (`--generator-disable-thinking` /
  `enable_thinking=False` for hybrid-reasoning models); `gpu_report` extended
  with CUDA runtime version, GPU name, VRAM, and transformers version.
- `egrag.reproducibility.environment_info`: records torch/transformers versions,
  CUDA availability/runtime version, GPU name, and VRAM — read from already-loaded
  modules only (never imports them itself), so the deterministic/offline harness
  still touches no optional model library.
- `egrag.experiments.runner`: the experiment harness now supports a real
  `huggingface` generator (previously `"fake"` only), with `generator_model`,
  `generator_revision`, `generator_device`, `generator_dtype`,
  `generator_quantization`, `generator_disable_thinking`, and `require_cuda` on
  `ExperimentConfig`, all recorded verbatim in `ExperimentManifest` alongside the
  actually-resolved runtime info.
- `egrag.config.schema.GenerationConfig`/`ExtractionConfig`/`NLIConfig`: added
  `device`/`dtype`/`model_revision`/`quantization`/`require_cuda` fields (the
  single-query `run --config` path now threads these through to the adapter).
  Added the `accelerate` dependency to the `local-models` extra (required by
  `device_map="auto"`, used by both `device="auto"` resolution and every
  quantized load — this was previously missing).
- New `scripts/cuda_smoke_test.py`: a bounded, configurable CUDA/GPU/model
  smoke test (device report, tokenizer, generator, NLI, one generation, one
  `full_egrag` example) — run once after provisioning a server, before any real
  experiment.
- Real Qwen2.5-3B-Instruct, Qwen2.5-7B-Instruct, and Qwen3.5-9B were validated
  end to end on an NVIDIA RTX 3090 (24 GB) via `scripts/cuda_smoke_test.py` and
  `egrag experiment run`; reports under `artifacts/cuda-smoke/`. See
  `docs/reproduction.md` for the full A/B/C command set and
  `docs/models.md` for the model-selection rationale (including why there is no
  Qwen 6B/plain-9B, and the cross-generation caveat for `Qwen/Qwen3.5-9B`).

### Added — benchmark integration
- `egrag.experiments.benchmarks`: real **FEVER** adapter
  (`FeverGoldEvidenceDataset`, gold-evidence setting, reads cached JSONL offline)
  and a **HotpotQA** adapter (`HotpotQADataset`, parquet mapping; raises a typed
  `MissingDependencyError` until a parquet reader is installed). Gold answers and
  gold evidence are never passed into the pipeline; `dataset_fingerprint` and
  `validate_benchmark` added.
- `egrag.experiments.benchmark_metrics`: HotpotQA (EM, normalized EM, token F1,
  supporting-fact P/R/F1, joint) and FEVER (label accuracy, evidence P/R/F1,
  alternative-evidence-set recovery, official FEVER score) metrics — pure,
  offline, separate from graph-mechanism metrics.
- Selected real models for the pilot (local, cached): NLI `roberta-large-mnli`,
  generator `Qwen/Qwen2.5-0.5B-Instruct`. Artifacts under
  `artifacts/benchmark-pipeline/`.
- **Known blocker:** HotpotQA is cached only as parquet; `pyarrow`/`datasets` are
  not installed and cannot be installed offline, so the HotpotQA pilot is not run.

### Added — query-conditioned reasoning connectivity (`BRIDGES`)
- New `RelationType.BRIDGES` and `BridgeMetadata` (`EvidenceRelation.bridge`);
  schema version → **1.6.0** (backward compatible).
- `egrag.graph.bridges`: `BridgeDetector` protocol + `DeterministicBridgeDetector`
  (query-conditioned, entity-structure baseline; NLI is not the bridge signal),
  `detect_bridges`, `extract_entities`, `query_subgoals`.
- `egrag.graph.nli.StructuralContradictionGate` — demotes structurally
  unjustified contradictions to neutral without changing NLI thresholds.
- Mechanism harness: bridge integration, `VariantFlags.bridges`, bridge metrics,
  belief capture; new `graph_no_bridge` ablation; new fixture categories
  `directional_support`, `hard_neutral`; multi-hop fixtures use bridges.
- Relation families separated: propagation/conflict/corroboration ignore
  `BRIDGES`; selector connectivity uses them. Documented in
  `docs/bridge-relations.md`; controlled results under `artifacts/bridge-milestone/`.
- Required-hop coverage improved 0.0 → 1.0 (no-bridge → full) on the controlled
  multi-hop suite, with proven belief and conflict invariance.

### Earlier (summary)
- Real-NLI evidential evaluation (offline `roberta-large-mnli`); zero-edge repair
  (mechanism fixtures + oracle/end-to-end modes); experiment/evaluation harness;
  research hardening (config, caching, reproducibility, security); model-agnostic
  evidence serialization + generation + grounding; reasoning subsystem; evidence
  graph; atomic-claim extraction; retrieval/chunking/reranking; domain foundation.
