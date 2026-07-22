"""Fairness and leakage checks for fair, comparable experiments.

These detect hidden configuration differences between compared variants and
guard against gold-label leakage into generation. They report issues rather than
silently proceeding; the runner raises when ``enforce_fairness`` is set.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable, Mapping, Sequence

from egrag.experiments.models import ExampleResult, ExperimentConfig, SystemVariant

_GOLD_PARAM_HINTS = ("gold", "answer", "label", "target", "reference")


def check_fairness(config: ExperimentConfig, variants: Sequence[SystemVariant]) -> list[str]:
    """Return fairness issues from hidden per-variant setting differences."""

    issues: list[str] = []
    if not config.enforce_fairness:
        return issues
    generators = {v.generator_override for v in variants if v.generator_override is not None}
    if generators:
        issues.append(
            f"generator mismatch: variants override the generator ({sorted(generators)}) "
            "while fairness is enforced"
        )
    top_ks = {v.top_k_override for v in variants if v.top_k_override is not None}
    if top_ks:
        issues.append(f"retrieval-budget mismatch: top_k overrides {sorted(top_ks)} under fairness")
    budgets = {
        v.evidence_budget_override for v in variants if v.evidence_budget_override is not None
    }
    if budgets:
        issues.append(f"evidence-budget mismatch: overrides {sorted(budgets)} under fairness")
    return issues


def same_example_ids(results_by_variant: Mapping[str, Sequence[ExampleResult]]) -> list[str]:
    """Return issues if compared variants did not run identical example IDs."""

    issues: list[str] = []
    id_sets = {
        variant: {r.example_id for r in results} for variant, results in results_by_variant.items()
    }
    if not id_sets:
        return issues
    reference_variant, reference = next(iter(id_sets.items()))
    for variant, ids in id_sets.items():
        if ids != reference:
            missing = reference - ids
            extra = ids - reference
            issues.append(
                f"variant {variant!r} example IDs differ from {reference_variant!r}: "
                f"missing={sorted(missing)} extra={sorted(extra)}"
            )
    return issues


def align_by_example_id(
    a: Sequence[ExampleResult], b: Sequence[ExampleResult]
) -> tuple[list[tuple[ExampleResult, ExampleResult]], list[str]]:
    """Align two result lists by example ID; report unmatched IDs."""

    a_by_id = {r.example_id: r for r in a}
    b_by_id = {r.example_id: r for r in b}
    common = sorted(set(a_by_id) & set(b_by_id))
    issues: list[str] = []
    unmatched = (set(a_by_id) | set(b_by_id)) - set(common)
    if unmatched:
        issues.append(f"unaligned example IDs excluded from paired comparison: {sorted(unmatched)}")
    return [(a_by_id[i], b_by_id[i]) for i in common], issues


def assert_no_gold_parameters(func: Callable[..., object]) -> None:
    """Raise if a system-generation callable accepts gold-label-like parameters.

    A structural guard backing the rule that gold answers/evidence must not be
    available to generator components.
    """

    for param in inspect.signature(func).parameters:
        lowered = param.lower()
        if any(hint in lowered for hint in _GOLD_PARAM_HINTS):
            raise AssertionError(
                f"system callable {func.__name__!r} exposes a gold-like parameter {param!r}"
            )


def tuning_indicators(config: ExperimentConfig) -> list[str]:
    """Heuristic indicators that a config may have been tuned on the test set."""

    issues: list[str] = []
    name = config.name.lower()
    if "tuned" in name or "test_tuned" in name:
        issues.append(f"experiment name {config.name!r} suggests test-set tuning")
    return issues


__all__ = [
    "align_by_example_id",
    "assert_no_gold_parameters",
    "check_fairness",
    "same_example_ids",
    "tuning_indicators",
]
