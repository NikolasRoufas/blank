# Changelog

The repository is local and unreleased; the versions below are schema/internal,
not published releases.

## Unreleased

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
