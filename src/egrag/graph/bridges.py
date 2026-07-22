"""Query-conditioned bridge detection (reasoning connectivity, not evidence).

A ``BRIDGES`` edge connects two claims that contribute complementary information
to a query reasoning chain through a shared, non-generic entity, **without** either
claim entailing or contradicting the other. Bridges only help reasoning-subgraph
connectivity and multi-hop coverage; they never affect belief, conflicts, or
corroboration (enforced elsewhere and test-locked).

The deterministic baseline here uses query/claim entity structure (not NLI
entailment and not raw token overlap). It is pluggable behind
:class:`BridgeDetector` for a future learned replacement.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from egrag.domain.models import (
    AtomicClaim,
    BridgeMetadata,
    EvidenceRelation,
    Query,
    RelationDirection,
    RelationType,
)

_ENTITY_RE = re.compile(r"\b[A-Z][a-zA-Z0-9]*(?:\s+[A-Z][a-zA-Z0-9]*)*\b")
# Capitalized function words that are never named entities (sentence-initial).
_STOP = frozenset(
    {
        "the",
        "a",
        "an",
        "this",
        "that",
        "these",
        "those",
        "it",
        "its",
        "they",
        "their",
        "we",
        "he",
        "she",
        "his",
        "her",
        "who",
        "what",
        "when",
        "where",
        "which",
        "whom",
        "whose",
        "why",
        "how",
        "in",
        "on",
        "at",
        "by",
        "for",
        "from",
        "with",
        "and",
        "but",
        "or",
        "as",
        "if",
        "then",
        "according",
    }
)
# Generic nouns that may be capitalized but are not entities for bridging.
_GENERIC = frozenset(
    {
        "company",
        "companies",
        "project",
        "deadline",
        "report",
        "reports",
        "city",
        "country",
        "person",
        "people",
        "organization",
        "team",
        "group",
        "product",
        "event",
        "paper",
        "author",
        "winner",
        "founder",
        "founders",
        "capital",
        "population",
        "employer",
        "headquarters",
        "factory",
        "merger",
        "launch",
        "revenue",
        "profit",
        "inspection",
    }
)


def extract_entities(text: str) -> frozenset[str]:
    """Extract candidate named entities from text (stop/generic words removed)."""

    out: set[str] = set()
    for match in _ENTITY_RE.finditer(text):
        words = match.group().split()
        while words and words[0].lower() in _STOP:
            words.pop(0)
        phrase = " ".join(words)
        if phrase and phrase.lower() not in _GENERIC and phrase.lower() not in _STOP:
            out.add(phrase)
    return frozenset(out)


def claim_entities(claim: AtomicClaim) -> frozenset[str]:
    """Entities for a claim: its structured named entities, else extracted."""

    if claim.semantics and claim.semantics.named_entities:
        ents = {e for e in claim.semantics.named_entities if e.lower() not in _GENERIC | _STOP}
        if ents:
            return frozenset(ents)
    return extract_entities(claim.text)


def query_subgoals(query: Query) -> frozenset[str]:
    """A lightweight, inspectable query representation: its entity subgoals."""

    return extract_entities(query.text)


@dataclass(frozen=True)
class BridgeDecision:
    """The outcome of a bridge check (created or rejected, with full rationale)."""

    created: bool
    bridge_entity: str | None = None
    bridge_terms: tuple[str, ...] = ()
    confidence: float = 0.0
    method_id: str = "deterministic-bridge-v1"
    query_terms_a: tuple[str, ...] = ()
    query_terms_b: tuple[str, ...] = ()
    rationale: str | None = None
    rejection_reason: str | None = None


@runtime_checkable
class BridgeDetector(Protocol):
    """Detects a query-conditioned bridge between two claims."""

    method_id: str

    def detect(self, query: Query, a: AtomicClaim, b: AtomicClaim) -> BridgeDecision: ...


class DeterministicBridgeDetector:
    """Conservative entity-structure bridge baseline (no NLI, no raw overlap)."""

    method_id = "deterministic-bridge-v1"

    def __init__(self, *, min_confidence: float = 0.5) -> None:
        self._min_confidence = min_confidence

    def detect(self, query: Query, a: AtomicClaim, b: AtomicClaim) -> BridgeDecision:
        if a.text.strip() == b.text.strip():
            return BridgeDecision(created=False, rejection_reason="identical text (duplicate)")
        ents_a, ents_b = claim_entities(a), claim_entities(b)
        shared = ents_a & ents_b
        if not shared:
            return BridgeDecision(created=False, rejection_reason="no shared non-generic entity")
        q_ents = query_subgoals(query)
        a_query = ents_a & q_ents
        b_query = ents_b & q_ents
        if not (a_query or b_query):
            return BridgeDecision(
                created=False, rejection_reason="shared entity unrelated to the query"
            )
        # Complementary: each claim contributes distinct (non-shared) content.
        only_a, only_b = ents_a - shared, ents_b - shared
        if not (only_a and only_b):
            return BridgeDecision(
                created=False, rejection_reason="claims not complementary (no new coverage)"
            )
        # bridge entity: prefer one that is itself a query entity, else any shared.
        in_query = sorted(shared & q_ents)
        bridge_entity = in_query[0] if in_query else sorted(shared)[0]
        confidence = 0.6 + 0.2 * bool(shared & q_ents) + 0.2 * bool(a_query and b_query)
        if confidence < self._min_confidence:
            return BridgeDecision(created=False, rejection_reason="below min bridge confidence")
        return BridgeDecision(
            created=True,
            bridge_entity=bridge_entity,
            bridge_terms=tuple(sorted(shared)),
            confidence=round(min(confidence, 1.0), 4),
            method_id=self.method_id,
            query_terms_a=tuple(sorted(a_query)),
            query_terms_b=tuple(sorted(b_query)),
            rationale=(
                f"complementary claims linked by {bridge_entity!r}; "
                f"query entities A={sorted(a_query)} B={sorted(b_query)}"
            ),
        )


def _evidential_pairs(relations: Sequence[EvidenceRelation]) -> set[frozenset[str]]:
    evidential = {
        RelationType.SUPPORT,
        RelationType.CONTRADICTION,
        RelationType.DUPLICATE,
        RelationType.SUPERSESSION,
    }
    return {
        frozenset({r.source_claim_id, r.target_claim_id})
        for r in relations
        if r.relation_type in evidential
    }


def detect_bridges(
    query: Query,
    claims: Sequence[AtomicClaim],
    existing_relations: Sequence[EvidenceRelation],
    detector: BridgeDetector | None = None,
    *,
    max_bridge_degree: int = 4,
) -> list[EvidenceRelation]:
    """Build BRIDGES edges over claim pairs not already evidentially related.

    Skips duplicate-cluster internal pairs and pairs already linked by an
    evidential relation; caps per-node bridge degree to prevent bridge spam.
    """

    det = detector or DeterministicBridgeDetector()
    blocked = _evidential_pairs(existing_relations)
    degree: dict[str, int] = {}
    bridges: list[EvidenceRelation] = []
    ordered = sorted(claims, key=lambda c: c.claim_id)
    for i, a in enumerate(ordered):
        for b in ordered[i + 1 :]:
            key = frozenset({a.claim_id, b.claim_id})
            if key in blocked:
                continue
            if (
                degree.get(a.claim_id, 0) >= max_bridge_degree
                or degree.get(b.claim_id, 0) >= max_bridge_degree
            ):
                continue
            decision = det.detect(query, a, b)
            if not decision.created:
                continue
            src, tgt = sorted((a.claim_id, b.claim_id))
            bridges.append(
                EvidenceRelation(
                    relation_id=f"bridge-{src}-{tgt}",
                    source_claim_id=src,
                    target_claim_id=tgt,
                    relation_type=RelationType.BRIDGES,
                    relation_confidence=decision.confidence,
                    direction=RelationDirection.SYMMETRIC,
                    rationale=decision.rationale,
                    bridge=BridgeMetadata(
                        bridge_entity=decision.bridge_entity,
                        bridge_terms=decision.bridge_terms,
                        query_conditioned=True,
                        bridge_confidence=decision.confidence,
                        bridge_method_id=decision.method_id,
                    ),
                )
            )
            degree[a.claim_id] = degree.get(a.claim_id, 0) + 1
            degree[b.claim_id] = degree.get(b.claim_id, 0) + 1
    return bridges


__all__ = [
    "BridgeDecision",
    "BridgeDetector",
    "DeterministicBridgeDetector",
    "claim_entities",
    "detect_bridges",
    "extract_entities",
    "query_subgoals",
]
