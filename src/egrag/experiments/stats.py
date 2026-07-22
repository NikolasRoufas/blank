"""Statistical utilities with explicit seeds.

The bootstrap is a *percentile* bootstrap and is deterministic under a fixed
seed. We deliberately do **not** emit p-values or significance claims: a
percentile CI describes sampling variability of the estimate, not a hypothesis
test. Paired comparison resamples example-aligned deltas.
"""

from __future__ import annotations

import math
import random
import statistics
from collections.abc import Sequence

from egrag.experiments.models import ComparisonResult


def mean(values: Sequence[float]) -> float:
    return statistics.fmean(values) if values else 0.0


def median(values: Sequence[float]) -> float:
    return statistics.median(values) if values else 0.0


def stdev(values: Sequence[float]) -> float:
    """Sample standard deviation (0.0 for fewer than two values)."""

    return statistics.stdev(values) if len(values) >= 2 else 0.0


def percentile(values: Sequence[float], pct: float) -> float:
    """Linear-interpolation percentile (``pct`` in [0, 100])."""

    if not values:
        return 0.0
    if not 0.0 <= pct <= 100.0:
        raise ValueError("pct must be within [0, 100]")
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (pct / 100.0) * (len(ordered) - 1)
    low = math.floor(rank)
    high = math.ceil(rank)
    if low == high:
        return ordered[low]
    return ordered[low] + (ordered[high] - ordered[low]) * (rank - low)


def bootstrap_ci(
    values: Sequence[float],
    *,
    samples: int = 1000,
    seed: int = 0,
    confidence: float = 0.95,
) -> tuple[float, float]:
    """Deterministic percentile bootstrap CI for the mean."""

    if not values:
        return (0.0, 0.0)
    rng = random.Random(seed)
    n = len(values)
    means: list[float] = []
    for _ in range(samples):
        resample = [values[rng.randrange(n)] for _ in range(n)]
        means.append(statistics.fmean(resample))
    alpha = (1.0 - confidence) / 2.0
    return (percentile(means, alpha * 100), percentile(means, (1.0 - alpha) * 100))


def paired_comparison(
    metric: str,
    variant_a: str,
    variant_b: str,
    paired: Sequence[tuple[float, float]],
    *,
    samples: int = 1000,
    seed: int = 0,
    confidence: float = 0.95,
) -> ComparisonResult:
    """Paired percentile bootstrap of the per-example delta (a - b)."""

    a_values = [a for a, _ in paired]
    b_values = [b for _, b in paired]
    deltas = [a - b for a, b in paired]
    ci_low, ci_high = bootstrap_ci(deltas, samples=samples, seed=seed, confidence=confidence)
    return ComparisonResult(
        metric=metric,
        variant_a=variant_a,
        variant_b=variant_b,
        n_paired=len(paired),
        mean_a=mean(a_values),
        mean_b=mean(b_values),
        mean_delta=mean(deltas),
        ci_low=ci_low,
        ci_high=ci_high,
        bootstrap_samples=samples,
        seed=seed,
    )


def aggregate_seeds(per_seed: Sequence[dict[str, float]]) -> dict[str, dict[str, float]]:
    """Aggregate compatible per-seed metric dicts into mean/std/median per metric.

    Only metric keys present in *every* seed are aggregated (alignment guard).
    """

    if not per_seed:
        return {}
    common = set(per_seed[0])
    for entry in per_seed[1:]:
        common &= set(entry)
    out: dict[str, dict[str, float]] = {}
    for key in sorted(common):
        values = [entry[key] for entry in per_seed]
        out[key] = {"mean": mean(values), "std": stdev(values), "median": median(values)}
    return out


__all__ = [
    "aggregate_seeds",
    "bootstrap_ci",
    "mean",
    "median",
    "paired_comparison",
    "percentile",
    "stdev",
]
