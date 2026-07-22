"""Scientific experiment and evaluation harness (separate from production).

Reuses production components to run controlled system variants over datasets,
scores predictions with clearly-kind-labeled metrics, performs fair paired
comparisons with deterministic bootstrap intervals, and writes reproducible,
inspectable artifacts. Importing this package has no side effects.
"""

from __future__ import annotations

from egrag.experiments.benchmark_metrics import (
    exact_match,
    fever_evidence_prf,
    fever_label_accuracy,
    fever_score,
    normalize_answer,
    supporting_fact_prf,
    token_f1,
)
from egrag.experiments.benchmarks import (
    FeverGoldEvidenceDataset,
    HotpotQADataset,
    dataset_fingerprint,
    validate_benchmark,
)
from egrag.experiments.datasets import (
    BUILTIN_DATASETS,
    DatasetAdapter,
    JsonlDataset,
    SyntheticGraphDataset,
    TemporalConflictDataset,
    check_dataset_integrity,
    get_dataset,
    load_examples,
)
from egrag.experiments.mechanism_eval import (
    ActivationError,
    MechanismRun,
    VariantFlags,
    activation,
    activation_preflight,
    build_run,
    mechanism_metrics,
    run_example,
)
from egrag.experiments.mechanisms import (
    GoldBridge,
    GoldConflict,
    GoldPair,
    GoldRelationClassifier,
    MechanismExample,
    build_suite,
)
from egrag.experiments.models import (
    AggregateMetrics,
    ComparisonResult,
    DatasetExample,
    ExampleResult,
    ExperimentConfig,
    ExperimentManifest,
    GoldEvidence,
    SystemVariant,
)
from egrag.experiments.runner import (
    ExperimentRunner,
    compare,
    inspect_example,
    summarize,
)
from egrag.experiments.variants import VARIANTS, RunSettings, get_variant, run_system

__all__ = [
    "BUILTIN_DATASETS",
    "VARIANTS",
    "ActivationError",
    "AggregateMetrics",
    "ComparisonResult",
    "DatasetAdapter",
    "DatasetExample",
    "ExampleResult",
    "ExperimentConfig",
    "ExperimentManifest",
    "ExperimentRunner",
    "FeverGoldEvidenceDataset",
    "GoldBridge",
    "GoldConflict",
    "GoldEvidence",
    "GoldPair",
    "GoldRelationClassifier",
    "HotpotQADataset",
    "JsonlDataset",
    "MechanismExample",
    "MechanismRun",
    "RunSettings",
    "SyntheticGraphDataset",
    "SystemVariant",
    "TemporalConflictDataset",
    "VariantFlags",
    "activation",
    "activation_preflight",
    "build_run",
    "build_suite",
    "check_dataset_integrity",
    "compare",
    "dataset_fingerprint",
    "exact_match",
    "fever_evidence_prf",
    "fever_label_accuracy",
    "fever_score",
    "get_dataset",
    "get_variant",
    "inspect_example",
    "load_examples",
    "mechanism_metrics",
    "normalize_answer",
    "run_example",
    "run_system",
    "summarize",
    "supporting_fact_prf",
    "token_f1",
    "validate_benchmark",
]
