"""The EG-RAG application pipeline.

The pipeline orchestrates the end-to-end flow from a query to a grounded answer
using only the ports defined in :mod:`egrag.domain.ports`. Concrete behavior is
supplied by injected components, proving that implementations are interchangeable
and that the application layer is decoupled from any provider.

The pipeline is deterministic: given the same components, query, settings, seed,
and clock, it produces identical output. Per-stage timings are collected only
when explicitly requested, so the default result is fully reproducible.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TypeVar

from egrag import __version__
from egrag.caching.memory import content_key
from egrag.domain.errors import PipelineError
from egrag.domain.models import (
    AtomicClaim,
    ConflictSet,
    EvidenceGraphSnapshot,
    EvidencePackage,
    Passage,
    PipelineMetrics,
    PipelineResult,
    Query,
    ReasoningSubgraph,
    RunManifest,
    SelectedEvidence,
)
from egrag.domain.ports import (
    BeliefPropagator,
    ClaimExtractor,
    ConflictResolver,
    GenerationParams,
    Generator,
    GroundingVerifier,
    InitialClaimScorer,
    RelationClassifier,
    Reranker,
    Retriever,
    SourceReliabilityScorer,
    SubgraphSelector,
    TemporalResolver,
)
from egrag.observability.logging import get_logger

_T = TypeVar("_T")


@dataclass(frozen=True)
class PipelineComponents:
    """The set of port implementations a pipeline run depends on."""

    retriever: Retriever
    reranker: Reranker
    claim_extractor: ClaimExtractor
    temporal_resolver: TemporalResolver
    reliability_scorer: SourceReliabilityScorer
    claim_scorer: InitialClaimScorer
    relation_classifier: RelationClassifier
    belief_propagator: BeliefPropagator
    conflict_resolver: ConflictResolver
    subgraph_selector: SubgraphSelector
    generator: Generator
    grounding_verifier: GroundingVerifier

    def identities(self) -> dict[str, str]:
        """Map each role to the implementing class name (for the run manifest)."""

        return {
            "retriever": type(self.retriever).__name__,
            "reranker": type(self.reranker).__name__,
            "claim_extractor": type(self.claim_extractor).__name__,
            "temporal_resolver": type(self.temporal_resolver).__name__,
            "reliability_scorer": type(self.reliability_scorer).__name__,
            "claim_scorer": type(self.claim_scorer).__name__,
            "relation_classifier": type(self.relation_classifier).__name__,
            "belief_propagator": type(self.belief_propagator).__name__,
            "conflict_resolver": type(self.conflict_resolver).__name__,
            "subgraph_selector": type(self.subgraph_selector).__name__,
            "generator": type(self.generator).__name__,
            "grounding_verifier": type(self.grounding_verifier).__name__,
        }


def _utc_now() -> datetime:
    return datetime.now(UTC)


class EGRagPipeline:
    """Orchestrates retrieval, evidence construction, selection, and generation."""

    def __init__(
        self,
        components: PipelineComponents,
        *,
        top_k: int = 5,
        selection_budget: int = 5,
        seed: int = 0,
        deterministic: bool = True,
        max_answer_tokens: int | None = 256,
        collect_timings: bool = False,
        clock: Callable[[], datetime] = _utc_now,
    ) -> None:
        if top_k < 1:
            raise PipelineError(f"top_k must be >= 1, got {top_k}")
        if selection_budget < 1:
            raise PipelineError(f"selection_budget must be >= 1, got {selection_budget}")
        if seed < 0:
            raise PipelineError(f"seed must be >= 0, got {seed}")
        self._c = components
        self._top_k = top_k
        self._selection_budget = selection_budget
        self._seed = seed
        self._deterministic = deterministic
        self._max_answer_tokens = max_answer_tokens
        self._collect_timings = collect_timings
        self._clock = clock
        self._log = get_logger("pipeline")

    def run(self, query: Query) -> PipelineResult:
        """Execute the full pipeline for ``query`` and return a result."""

        timings: dict[str, float] = {}

        def timed(stage: str, fn: Callable[[], _T]) -> _T:
            start = time.perf_counter()
            result = fn()
            if self._collect_timings:
                timings[stage] = (time.perf_counter() - start) * 1000.0
            return result

        passages = timed("retrieve", lambda: self._c.retriever.retrieve(query, self._top_k))
        passages = timed("rerank", lambda: self._c.reranker.rerank(query, passages))
        claims = timed("extract", lambda: self._extract(passages))
        claims = timed("temporal", lambda: [self._c.temporal_resolver.resolve(c) for c in claims])
        claims = timed("reliability", lambda: [self._with_reliability(c) for c in claims])
        claims = timed("score", lambda: [self._c.claim_scorer.score(c, query) for c in claims])

        snapshot = timed("build_graph", lambda: self._build_snapshot(query, claims))
        snapshot = timed("propagate", lambda: self._c.belief_propagator.propagate(snapshot))
        conflicts = timed("conflicts", lambda: self._c.conflict_resolver.resolve(snapshot))
        subgraph = timed(
            "select",
            lambda: self._c.subgraph_selector.select(snapshot, query, self._selection_budget),
        )

        package = timed(
            "package",
            lambda: self._build_package(query, snapshot, subgraph, conflicts),
        )
        params = GenerationParams(
            seed=self._seed,
            deterministic=self._deterministic,
            max_tokens=self._max_answer_tokens,
        )
        answer = timed("generate", lambda: self._c.generator.generate(package, params))
        answer = timed("verify", lambda: self._c.grounding_verifier.verify(answer, package))

        metrics = PipelineMetrics(
            num_passages=len(passages),
            num_claims=len(snapshot.claims),
            num_relations=len(snapshot.relations),
            num_conflicts=len(package.conflicts),
            num_selected=len(package.selected),
            durations_ms=timings,
        )
        manifest = self._build_manifest(query)
        return PipelineResult(
            query=query,
            answer=answer,
            package=package,
            metrics=metrics,
            manifest=manifest,
        )

    # --- stage helpers -------------------------------------------------------

    def _extract(self, passages: list[Passage]) -> list[AtomicClaim]:
        claims: list[AtomicClaim] = []
        for passage in passages:
            claims.extend(self._c.claim_extractor.extract(passage))
        return claims

    def _with_reliability(self, claim: AtomicClaim) -> AtomicClaim:
        score = self._c.reliability_scorer.score(claim.provenance.source)
        if not 0.0 <= score <= 1.0:
            msg = f"source reliability score {score} is outside [0, 1]"
            raise PipelineError(msg)
        return claim.model_copy(update={"source_reliability": score})

    def _build_snapshot(self, query: Query, claims: list[AtomicClaim]) -> EvidenceGraphSnapshot:
        relations = self._c.relation_classifier.classify(claims)
        return EvidenceGraphSnapshot(
            snapshot_id=f"snapshot:{query.query_id}",
            claims=tuple(claims),
            relations=tuple(relations),
        )

    def _build_package(
        self,
        query: Query,
        snapshot: EvidenceGraphSnapshot,
        subgraph: ReasoningSubgraph,
        conflicts: list[ConflictSet],
    ) -> EvidencePackage:
        by_id = {claim.claim_id: claim for claim in snapshot.claims}
        selected: list[SelectedEvidence] = []
        for rank, claim_id in enumerate(subgraph.claim_ids):
            claim = by_id.get(claim_id)
            if claim is None:
                raise PipelineError(f"selected claim {claim_id!r} not present in snapshot")
            score = claim.query_utility if claim.query_utility is not None else 0.0
            selected.append(SelectedEvidence(claim_id=claim_id, selection_score=score, rank=rank))

        return EvidencePackage(
            package_id=f"package:{query.query_id}",
            query=query,
            claims=snapshot.claims,
            relations=snapshot.relations,
            conflicts=tuple(conflicts),
            selected=tuple(selected),
            subgraph=subgraph,
        )

    def _build_manifest(self, query: Query) -> RunManifest:
        config_hash = content_key(
            str(self._top_k),
            str(self._selection_budget),
            str(self._seed),
            str(self._deterministic),
            str(self._max_answer_tokens),
        )
        input_hash = content_key(query.query_id, query.text)
        return RunManifest(
            egrag_version=__version__,
            seed=self._seed,
            deterministic=self._deterministic,
            created_at=self._clock(),
            component_identities=self._c.identities(),
            config_hash=config_hash,
            input_hash=input_hash,
        )


__all__ = ["EGRagPipeline", "PipelineComponents"]
