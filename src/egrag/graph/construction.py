"""Evidence-graph construction: candidates → classification → edges → snapshot.

Builds an :class:`EvidenceGraphSnapshot` from atomic claims using a pluggable
pair classifier. Duplicate detection and temporal supersession are applied
before NLI edges, and each unordered pair yields at most one relation type
(precedence: lexical duplicate > supersession > semantic duplicate >
contradiction > support). Neutral classifications create no stored edge unless
``store_neutral`` is set. Every inferred edge records classifier provenance.
"""

from __future__ import annotations

import hashlib
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from time import perf_counter

from egrag.domain.models import (
    AtomicClaim,
    EvidenceGraphSnapshot,
    EvidenceRelation,
    Query,
    RelationDirection,
    RelationMetadata,
    RelationType,
)
from egrag.graph.api import EvidenceGraph
from egrag.graph.candidates import generate_candidates
from egrag.graph.duplicates import detect_lexical_duplicates
from egrag.graph.temporal import SupersessionEdge, SupersessionResolver
from egrag.graph.types import (
    CandidateConfig,
    CandidateStats,
    ClassificationConfig,
    DuplicateConfig,
    GraphConstructionMetrics,
    PairClassifier,
    RelationProbabilities,
    TemporalConfig,
)
from egrag.graph.validation import validate_components


@dataclass(frozen=True)
class GraphConstructionResult:
    """The built graph plus construction metrics and candidate statistics."""

    graph: EvidenceGraph
    metrics: GraphConstructionMetrics
    candidate_stats: CandidateStats


def _relation_id(relation_type: RelationType, source: str, target: str) -> str:
    digest = hashlib.sha256(f"{relation_type.value}\x00{source}\x00{target}".encode())
    return f"rel-{digest.hexdigest()[:16]}"


def _canonical(a: str, b: str) -> tuple[str, str]:
    return (a, b) if a <= b else (b, a)


class GraphBuilder:
    """Constructs evidence graphs from atomic claims."""

    def __init__(
        self,
        classifier: PairClassifier,
        *,
        candidate_config: CandidateConfig | None = None,
        classification_config: ClassificationConfig | None = None,
        duplicate_config: DuplicateConfig | None = None,
        temporal_config: TemporalConfig | None = None,
        snapshot_id: str = "graph",
        store_neutral: bool = False,
    ) -> None:
        self._classifier = classifier
        self._candidate_config = candidate_config or CandidateConfig()
        self._classification_config = classification_config or ClassificationConfig()
        self._duplicate_config = duplicate_config or DuplicateConfig()
        self._supersession = SupersessionResolver(temporal_config)
        self._snapshot_id = snapshot_id
        self._store_neutral = store_neutral

    def build(
        self, claims: Sequence[AtomicClaim], *, query: Query | None = None
    ) -> GraphConstructionResult:
        start = perf_counter()
        claim_list = list(claims)
        validate_components(claim_list, [])  # input claims: unique ids, valid provenance

        relations: list[EvidenceRelation] = []
        used: set[tuple[str, str]] = set()

        # 1. Lexical duplicates.
        for dup in detect_lexical_duplicates(claim_list, self._duplicate_config):
            relations.append(self._duplicate_relation(dup.a_id, dup.b_id, 1.0, dup.kind))
            used.add(_canonical(dup.a_id, dup.b_id))

        # 2. Temporal supersession.
        for edge in self._supersession.resolve(claim_list):
            key = _canonical(edge.newer_id, edge.older_id)
            if key in used:
                continue
            relations.append(self._supersession_relation(edge))
            used.add(key)

        # 3. Candidate generation + classification.
        candidates = generate_candidates(claim_list, self._candidate_config, query=query)
        probabilities = self._classifier.classify(candidates.pairs)
        nli_batch_count = 1 if candidates.pairs else 0
        prob_by_pair: dict[tuple[str, str], RelationProbabilities] = {
            (pair.source.claim_id, pair.target.claim_id): prob
            for pair, prob in zip(candidates.pairs, probabilities, strict=True)
        }

        unordered_keys = sorted(
            {_canonical(p.source.claim_id, p.target.claim_id) for p in candidates.pairs}
        )
        for x, y in unordered_keys:
            if (x, y) in used:
                continue
            relations.extend(self._edges_for_pair(x, y, prob_by_pair))

        snapshot = EvidenceGraphSnapshot(
            snapshot_id=self._snapshot_id,
            claims=tuple(claim_list),
            relations=tuple(relations),
        )
        validate_components(snapshot.claims, snapshot.relations, allow_neutral=self._store_neutral)

        edges_by_type = Counter(relation.relation_type.value for relation in relations)
        metrics = GraphConstructionMetrics(
            num_claims=len(claim_list),
            possible_pairs=candidates.stats.possible_pairs,
            generated_candidate_pairs=candidates.stats.generated_pairs,
            classified_pairs=len(candidates.pairs),
            pruned_pairs=candidates.stats.pruned_pairs,
            edges_by_type=dict(edges_by_type),
            nli_batch_count=nli_batch_count,
            construction_ms=(perf_counter() - start) * 1000.0,
        )
        return GraphConstructionResult(
            graph=EvidenceGraph(snapshot),
            metrics=metrics,
            candidate_stats=candidates.stats,
        )

    # --- edge builders -------------------------------------------------------

    def _edges_for_pair(
        self,
        x: str,
        y: str,
        prob_by_pair: dict[tuple[str, str], RelationProbabilities],
    ) -> list[EvidenceRelation]:
        cfg = self._classification_config
        forward = prob_by_pair.get((x, y))
        backward = prob_by_pair.get((y, x))
        entail_xy = forward.entailment if forward else 0.0
        entail_yx = backward.entailment if backward else 0.0
        contradiction = max(
            forward.contradiction if forward else 0.0,
            backward.contradiction if backward else 0.0,
        )

        if entail_xy >= cfg.duplicate_threshold and entail_yx >= cfg.duplicate_threshold:
            return [self._duplicate_relation(x, y, round(min(entail_xy, entail_yx), 6), "semantic")]
        if contradiction >= cfg.contradiction_threshold:
            return [self._contradiction_relation(x, y, round(contradiction, 6))]

        edges: list[EvidenceRelation] = []
        if entail_xy >= cfg.entailment_threshold:
            edges.append(self._support_relation(x, y, round(entail_xy, 6)))
        if entail_yx >= cfg.entailment_threshold:
            edges.append(self._support_relation(y, x, round(entail_yx, 6)))
        if not edges and self._store_neutral:
            neutral = max(
                forward.neutral if forward else 0.0, backward.neutral if backward else 0.0
            )
            edges.append(self._neutral_relation(x, y, round(neutral, 6)))
        return edges

    def _classifier_metadata(
        self, explanation: str, features: dict[str, float]
    ) -> RelationMetadata:
        return RelationMetadata(
            classifier_id=self._classifier.classifier_id,
            classifier_version=self._classifier.classifier_version,
            model_revision=self._classifier.model_revision,
            explanation=explanation,
            features=features,
        )

    def _support_relation(self, source: str, target: str, confidence: float) -> EvidenceRelation:
        return EvidenceRelation(
            relation_id=_relation_id(RelationType.SUPPORT, source, target),
            source_claim_id=source,
            target_claim_id=target,
            relation_type=RelationType.SUPPORT,
            relation_confidence=confidence,
            direction=RelationDirection.DIRECTED,
            rationale="entailment above threshold",
            metadata=self._classifier_metadata("entailment", {"entailment": confidence}),
        )

    def _contradiction_relation(self, x: str, y: str, confidence: float) -> EvidenceRelation:
        source, target = _canonical(x, y)
        return EvidenceRelation(
            relation_id=_relation_id(RelationType.CONTRADICTION, source, target),
            source_claim_id=source,
            target_claim_id=target,
            relation_type=RelationType.CONTRADICTION,
            relation_confidence=confidence,
            direction=RelationDirection.SYMMETRIC,
            rationale="contradiction above threshold (symmetric)",
            metadata=self._classifier_metadata("contradiction", {"contradiction": confidence}),
        )

    def _duplicate_relation(self, x: str, y: str, confidence: float, kind: str) -> EvidenceRelation:
        source, target = _canonical(x, y)
        classifier_id = (
            "lexical-duplicate"
            if kind in ("exact", "normalized")
            else (self._classifier.classifier_id)
        )
        return EvidenceRelation(
            relation_id=_relation_id(RelationType.DUPLICATE, source, target),
            source_claim_id=source,
            target_claim_id=target,
            relation_type=RelationType.DUPLICATE,
            relation_confidence=confidence,
            direction=RelationDirection.SYMMETRIC,
            rationale=f"{kind} duplicate (symmetric)",
            metadata=RelationMetadata(
                classifier_id=classifier_id,
                classifier_version="1.0.0"
                if kind in ("exact", "normalized")
                else self._classifier.classifier_version,
                explanation=f"{kind} duplicate",
                features={"duplicate": confidence},
            ),
        )

    def _supersession_relation(self, edge: SupersessionEdge) -> EvidenceRelation:
        return EvidenceRelation(
            relation_id=_relation_id(RelationType.SUPERSESSION, edge.newer_id, edge.older_id),
            source_claim_id=edge.newer_id,
            target_claim_id=edge.older_id,
            relation_type=RelationType.SUPERSESSION,
            relation_confidence=edge.confidence,
            direction=RelationDirection.DIRECTED,
            rationale=edge.explanation,
            metadata=RelationMetadata(
                classifier_id="temporal-resolver",
                classifier_version="1.0.0",
                explanation=edge.explanation,
                features={"update_confidence": edge.confidence},
            ),
        )

    def _neutral_relation(self, x: str, y: str, confidence: float) -> EvidenceRelation:
        source, target = _canonical(x, y)
        return EvidenceRelation(
            relation_id=_relation_id(RelationType.NEUTRAL, source, target),
            source_claim_id=source,
            target_claim_id=target,
            relation_type=RelationType.NEUTRAL,
            relation_confidence=confidence,
            direction=RelationDirection.SYMMETRIC,
            rationale="neutral (debug mode)",
            metadata=self._classifier_metadata("neutral", {"neutral": confidence}),
        )


__all__ = ["GraphBuilder", "GraphConstructionResult"]
