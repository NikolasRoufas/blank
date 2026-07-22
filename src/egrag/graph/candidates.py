"""Candidate-pair generation with configurable pruning and a pair budget.

Unordered claim pairs are scored by configurable signals (shared entities,
lexical overlap, subject/predicate compatibility, temporal overlap, query
relevance, source diversity, claim type). Surviving pairs are expanded into both
ordered directions so directional entailment can be assessed. Ordering is fully
deterministic and the pair budget is enforced deterministically.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from egrag.domain.models import AtomicClaim, Query
from egrag.graph.text_utils import claim_entities, jaccard, tokens
from egrag.graph.types import (
    CandidateConfig,
    CandidateStats,
    CandidateStrategy,
    ClaimPair,
)


@dataclass(frozen=True)
class CandidateResult:
    """Generated ordered candidate pairs plus pruning statistics."""

    pairs: list[ClaimPair]
    stats: CandidateStats


def _subject_predicate_compatible(a: AtomicClaim, b: AtomicClaim) -> bool:
    if a.semantics is None or b.semantics is None:
        return False
    sa = (a.semantics.subject or "").casefold()
    sb = (b.semantics.subject or "").casefold()
    return bool(sa) and sa == sb


def _temporal_overlap(a: AtomicClaim, b: AtomicClaim) -> bool:
    return a.asserted_at is not None and b.asserted_at is not None


def _query_relevant(claim: AtomicClaim, query_tokens: set[str], threshold: float) -> bool:
    return jaccard(tokens(claim.text), query_tokens) >= threshold


def _pair_reasons(
    a: AtomicClaim,
    b: AtomicClaim,
    config: CandidateConfig,
    query_tokens: set[str],
) -> list[str]:
    """Return the signal names that make {a, b} a candidate pair."""

    reasons: list[str] = []
    if (
        config.use_shared_entities
        and len(claim_entities(a) & claim_entities(b)) >= config.shared_entities_min
    ):
        reasons.append("shared_entities")
    if (
        config.use_lexical_overlap
        and jaccard(tokens(a.text), tokens(b.text)) >= config.lexical_overlap_threshold
    ):
        reasons.append("lexical_overlap")
    if config.use_subject_predicate and _subject_predicate_compatible(a, b):
        reasons.append("subject_predicate")
    if config.use_temporal_overlap and _temporal_overlap(a, b):
        reasons.append("temporal_overlap")
    if (
        config.use_query_relevance
        and query_tokens
        and _query_relevant(a, query_tokens, config.query_relevance_threshold)
        and _query_relevant(b, query_tokens, config.query_relevance_threshold)
    ):
        reasons.append("query_relevance")
    if (
        config.use_source_diversity
        and a.provenance.source.source_id != b.provenance.source.source_id
    ):
        reasons.append("source_diversity")
    if config.use_claim_type:
        # Claim "type" is approximated by negation parity in the baseline.
        neg_a = a.semantics.negation if a.semantics else False
        neg_b = b.semantics.negation if b.semantics else False
        if neg_a == neg_b:
            reasons.append("claim_type")
    return reasons


def generate_candidates(
    claims: Sequence[AtomicClaim],
    config: CandidateConfig | None = None,
    *,
    query: Query | None = None,
) -> CandidateResult:
    """Generate ordered candidate pairs deterministically.

    Brute force keeps every unordered pair; pruned keeps only pairs with at
    least one passing signal. The budget caps the number of *unordered* pairs by
    a deterministic score (more signals first, then claim ids).
    """

    cfg = config or CandidateConfig()
    ordered_claims = sorted(claims, key=lambda c: c.claim_id)
    n = len(ordered_claims)
    possible = n * (n - 1) // 2
    query_tokens = tokens(query.text) if query is not None else set()

    scored: list[tuple[int, str, str, AtomicClaim, AtomicClaim, tuple[str, ...]]] = []
    for i in range(n):
        for j in range(i + 1, n):
            a, b = ordered_claims[i], ordered_claims[j]
            if cfg.strategy is CandidateStrategy.BRUTE_FORCE:
                reasons: list[str] = ["brute_force"]
            else:
                reasons = _pair_reasons(a, b, cfg, query_tokens)
                if not reasons:
                    continue
            scored.append((len(reasons), a.claim_id, b.claim_id, a, b, tuple(reasons)))

    # Deterministic ordering: most signals first, then by claim ids.
    scored.sort(key=lambda item: (-item[0], item[1], item[2]))

    budget_truncated = 0
    if cfg.max_pairs is not None and len(scored) > cfg.max_pairs:
        budget_truncated = len(scored) - cfg.max_pairs
        scored = scored[: cfg.max_pairs]

    pairs: list[ClaimPair] = []
    for _, _, _, a, b, reasons_tuple in scored:
        pairs.append(ClaimPair(source=a, target=b, reasons=reasons_tuple))
        pairs.append(ClaimPair(source=b, target=a, reasons=reasons_tuple))
    pairs.sort(key=lambda p: (p.source.claim_id, p.target.claim_id))

    stats = CandidateStats(
        num_claims=n,
        possible_pairs=possible,
        generated_pairs=len(scored),
        pruned_pairs=possible - len(scored) - budget_truncated,
        budget_truncated=budget_truncated,
    )
    return CandidateResult(pairs=pairs, stats=stats)


__all__ = ["CandidateResult", "generate_candidates"]
