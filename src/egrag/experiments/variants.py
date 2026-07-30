"""System variants and the evaluation runner that executes them.

Every variant reuses the **same** retriever (BM25), chunker, extractor, and
generator, so a comparison isolates only the intended component. The three
families differ at runtime:

* ``passage`` — retrieved passages are the evidence (no extraction/graph).
* ``claim``   — claims are extracted and ranked (no graph/propagation).
* ``graph``   — full evidence graph with per-variant toggles (propagation,
  selection strategy, temporal edges, contradiction edges).

Gold answers/evidence are never passed in here; only the question and documents.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from egrag.adapters.extraction import SentenceClaimExtractor
from egrag.adapters.retrieval import BM25Retriever, SentenceAwareChunker, prepare_passages
from egrag.domain.models import (
    AtomicClaim,
    ClaimProvenance,
    Document,
    EvidencePackage,
    GeneratedAnswer,
    PackageBudget,
    Passage,
    Query,
    SelectedEvidence,
    SourceMetadata,
)
from egrag.domain.ports import CacheBackend, Reranker, Retriever, TokenCounter
from egrag.experiments.models import SystemVariant
from egrag.generation import (
    GenerationConfig,
    GenerationService,
    TextGenerator,
    build_evidence_package,
)
from egrag.graph import ClassificationConfig, GraphBuilder, LexicalPairClassifier, TemporalConfig
from egrag.graph.types import PairClassifier
from egrag.reasoning import (
    BaselineInitialScorer,
    CharacterTokenCounter,
    ConflictSetResolver,
    GreedyConnectedSelector,
    MetadataReliability,
    NoPropagationBaseline,
    SignedBeliefPropagator,
    TokenBudget,
    TopClaimsSelector,
)


@runtime_checkable
class EvidenceExtractor(Protocol):
    """A claim extractor usable by the experiment runner.

    Matches both the deterministic ``SentenceClaimExtractor`` and the real
    ``StructuredClaimExtractor`` (passage in, claims out, with optional query and
    source). Gold data is never passed in.
    """

    def extract(
        self,
        passage: Passage,
        *,
        query: Query | None = ...,
        source: SourceMetadata | None = ...,
    ) -> list[AtomicClaim]: ...


@dataclass(frozen=True)
class SystemOutput:
    """Non-persisted output of running one variant on one example."""

    answer: str
    abstained: bool
    cited_claim_ids: tuple[str, ...]
    cited_source_ids: tuple[str, ...]
    selected_source_ids: tuple[str, ...]
    known_claim_ids: tuple[str, ...]
    evidence_texts: tuple[str, ...]
    package: EvidencePackage
    counts: dict[str, int] = field(default_factory=dict)
    warnings: tuple[str, ...] = ()


VARIANTS: dict[str, SystemVariant] = {
    "passage_rag": SystemVariant(
        name="passage_rag",
        description="Passage-level RAG.",
        family="passage",
        isolates=("retrieval_only",),
    ),
    "reranked_passage_rag": SystemVariant(
        name="reranked_passage_rag",
        description="Passage RAG with score reranking.",
        family="passage",
        rerank=True,
        isolates=("reranking",),
    ),
    "claim_only_rag": SystemVariant(
        name="claim_only_rag",
        description="Claim-level RAG without a graph.",
        family="claim",
        isolates=("claim_extraction",),
    ),
    "graph_no_propagation": SystemVariant(
        name="graph_no_propagation",
        description="Graph built, beliefs not propagated.",
        family="graph",
        propagation=False,
        isolates=("propagation",),
    ),
    "graph_with_propagation": SystemVariant(
        name="graph_with_propagation",
        description="Graph with belief propagation.",
        family="graph",
        propagation=True,
        isolates=("propagation",),
    ),
    "graph_top_claim": SystemVariant(
        name="graph_top_claim",
        description="Graph with top-claim selection.",
        family="graph",
        selection="top",
        isolates=("selection",),
    ),
    "graph_coherent_subgraph": SystemVariant(
        name="graph_coherent_subgraph",
        description="Graph with coherent-subgraph selection.",
        family="graph",
        selection="greedy",
        isolates=("selection",),
    ),
    "graph_no_temporal": SystemVariant(
        name="graph_no_temporal",
        description="Graph without temporal supersession edges.",
        family="graph",
        temporal=False,
        isolates=("temporal_edges",),
    ),
    "graph_no_contradiction": SystemVariant(
        name="graph_no_contradiction",
        description="Graph without contradiction edges.",
        family="graph",
        contradiction=False,
        isolates=("contradiction_edges",),
    ),
    "full_egrag": SystemVariant(
        name="full_egrag",
        description="Full EG-RAG pipeline.",
        family="graph",
        propagation=True,
        selection="greedy",
        temporal=True,
        contradiction=True,
        isolates=(),
    ),
}


def get_variant(name: str) -> SystemVariant:
    if name not in VARIANTS:
        raise KeyError(f"unknown variant {name!r}; known: {sorted(VARIANTS)}")
    return VARIANTS[name]


@dataclass(frozen=True)
class RunSettings:
    """Per-run, fairness-controlled settings shared across variants."""

    top_k: int = 4
    evidence_token_budget: int = 256
    reserved_output_tokens: int = 64
    chunk_size: int = 512
    chunk_overlap: int = 64


@dataclass(frozen=True)
class RunComponents:
    """Injectable, fairness-shared components for a run.

    Every field defaults to ``None`` meaning "use the deterministic default"
    (BM25 retriever, sentence-aware chunker, sentence claim extractor, lexical
    pair classifier, character token counter, no reranker, no cache). The SAME
    instances are used across passage/claim/graph families so a comparison stays
    fair. The final matrix injects the real extractor/NLI here. Gold data is never
    a component.
    """

    extractor: EvidenceExtractor | None = None
    classifier: PairClassifier | None = None
    reranker: Reranker | None = None
    token_counter: TokenCounter | None = None
    cache: CacheBackend | None = None
    retriever_factory: Callable[[Sequence[Passage]], Retriever] | None = None
    chunker: SentenceAwareChunker | None = None
    # Base NLI thresholds for graph variants; None uses ClassificationConfig()'s
    # defaults. The graph_no_contradiction ablation still overrides
    # contradiction_threshold=1.0 on top of this base, regardless.
    classification_config: ClassificationConfig | None = None


_DEFAULT_COMPONENTS = RunComponents()


def _chunker(settings: RunSettings, components: RunComponents) -> SentenceAwareChunker:
    return components.chunker or SentenceAwareChunker(
        chunk_size=settings.chunk_size, overlap=settings.chunk_overlap
    )


def _retriever(passages: Sequence[Passage], components: RunComponents) -> Retriever:
    factory = components.retriever_factory or BM25Retriever
    return factory(passages)


def _extractor(components: RunComponents) -> EvidenceExtractor:
    return components.extractor or SentenceClaimExtractor()


def _classifier(components: RunComponents) -> PairClassifier:
    return components.classifier or LexicalPairClassifier()


def _classification_config(
    components: RunComponents, variant: SystemVariant
) -> ClassificationConfig:
    base = components.classification_config or ClassificationConfig()
    if not variant.contradiction:
        return base.model_copy(update={"contradiction_threshold": 1.0})
    return base


def _token_counter(components: RunComponents) -> TokenCounter:
    return components.token_counter or CharacterTokenCounter()


def _generate(
    package: EvidencePackage, generator: TextGenerator, cfg: GenerationConfig
) -> GeneratedAnswer:
    if not package.selected:
        return GeneratedAnswer(
            text="No supported evidence was found to answer the query.",
            abstained=True,
            uncertainty="No evidence met the selection threshold.",
        )
    return GenerationService(token_counter=CharacterTokenCounter()).generate(
        package, generator, cfg
    )


def _source_ids_for(package: EvidencePackage, claim_ids: Sequence[str]) -> tuple[str, ...]:
    by_id = {c.claim_id: c.provenance.source.source_id for c in package.claims}
    seen: list[str] = []
    for cid in claim_ids:
        sid = by_id.get(cid)
        if sid is not None and sid not in seen:
            seen.append(sid)
    return tuple(seen)


def _finish(
    package: EvidencePackage,
    answer: GeneratedAnswer,
    *,
    extra_counts: dict[str, int],
) -> SystemOutput:
    selected_ids = tuple(s.claim_id for s in package.selected)
    text_by_id = {c.claim_id: c.text for c in package.claims}
    counts = {
        "num_selected": len(package.selected),
        "token_estimate": package.budget.used if package.budget else 0,
        **extra_counts,
    }
    return SystemOutput(
        answer=answer.text,
        abstained=answer.abstained,
        cited_claim_ids=tuple(answer.cited_claim_ids),
        cited_source_ids=_source_ids_for(package, answer.cited_claim_ids),
        selected_source_ids=_source_ids_for(package, selected_ids),
        known_claim_ids=tuple(c.claim_id for c in package.claims),
        evidence_texts=tuple(text_by_id[cid] for cid in selected_ids if cid in text_by_id),
        package=package,
        counts=counts,
        warnings=tuple(answer.unsupported_warnings),
    )


def _retrieve(
    query: Query,
    documents: Sequence[Document],
    settings: RunSettings,
    components: RunComponents,
    *,
    rerank: bool = False,
) -> tuple[list[Passage], dict[str, SourceMetadata]]:
    passages = prepare_passages(documents, _chunker(settings, components))
    hits = list(_retriever(passages, components).retrieve(query, settings.top_k))
    if rerank and components.reranker is not None:
        hits = list(components.reranker.rerank(query, hits))
    sources = {doc.source.source_id: doc.source for doc in documents}
    return hits, sources


def _passage_claims(
    hits: Sequence[Passage], sources: dict[str, SourceMetadata]
) -> list[AtomicClaim]:
    claims: list[AtomicClaim] = []
    for passage in hits:
        span = passage.span
        source = sources.get(span.source_id) or SourceMetadata(source_id=span.source_id)
        claims.append(
            AtomicClaim(
                claim_id=passage.passage_id,
                text=passage.text,
                provenance=ClaimProvenance(
                    source=source,
                    spans=(span,),
                    passage_id=passage.passage_id,
                ),
                extraction_confidence=1.0,
            )
        )
    return claims


def _budget_select(
    claims: Sequence[AtomicClaim], settings: RunSettings
) -> tuple[list[SelectedEvidence], int]:
    available = settings.evidence_token_budget - settings.reserved_output_tokens
    selected: list[SelectedEvidence] = []
    used = 0
    for rank, claim in enumerate(claims):
        estimate = max(1, len(claim.text) // 4)
        if used + estimate > available and selected:
            break
        used += estimate
        selected.append(
            SelectedEvidence(
                claim_id=claim.claim_id,
                selection_score=claim.extraction_confidence,
                rank=rank,
                belief=claim.belief,
                utility=claim.query_utility,
            )
        )
    return selected, used


def _run_passage(
    query: Query,
    documents: Sequence[Document],
    settings: RunSettings,
    variant: SystemVariant,
    generator: TextGenerator,
    cfg: GenerationConfig,
    components: RunComponents,
) -> SystemOutput:
    # An injected reranker is applied only for rerank variants; with no reranker
    # the BM25 order is unchanged (documented limitation of the default).
    hits, sources = _retrieve(query, documents, settings, components, rerank=variant.rerank)
    claims = _passage_claims(hits, sources)
    selected, used = _budget_select(claims, settings)
    package = EvidencePackage(
        package_id=f"pkg-{query.query_id}",
        query=query,
        claims=tuple(claims),
        selected=tuple(selected),
        budget=PackageBudget(
            total=settings.evidence_token_budget,
            reserved_output=settings.reserved_output_tokens,
            used=used,
        ),
    )
    return _finish(
        package,
        _generate(package, generator, cfg),
        extra_counts={
            "num_passages": len(hits),
            "num_claims": len(claims),
            "num_graph_nodes": 0,
            "num_graph_edges": 0,
        },
    )


def _run_claim(
    query: Query,
    documents: Sequence[Document],
    settings: RunSettings,
    variant: SystemVariant,
    generator: TextGenerator,
    cfg: GenerationConfig,
    components: RunComponents,
) -> SystemOutput:
    hits, sources = _retrieve(query, documents, settings, components)
    extractor = _extractor(components)
    claims: list[AtomicClaim] = []
    for passage in hits:
        claims.extend(
            extractor.extract(passage, query=query, source=sources.get(passage.span.source_id))
        )
    claims.sort(key=lambda c: (-c.extraction_confidence, c.claim_id))
    selected, used = _budget_select(claims, settings)
    package = EvidencePackage(
        package_id=f"pkg-{query.query_id}",
        query=query,
        claims=tuple(claims),
        selected=tuple(selected),
        budget=PackageBudget(
            total=settings.evidence_token_budget,
            reserved_output=settings.reserved_output_tokens,
            used=used,
        ),
    )
    return _finish(
        package,
        _generate(package, generator, cfg),
        extra_counts={
            "num_passages": len(hits),
            "num_claims": len(claims),
            "num_graph_nodes": 0,
            "num_graph_edges": 0,
        },
    )


def _run_graph(
    query: Query,
    documents: Sequence[Document],
    settings: RunSettings,
    variant: SystemVariant,
    generator: TextGenerator,
    cfg: GenerationConfig,
    components: RunComponents,
) -> SystemOutput:
    hits, sources = _retrieve(query, documents, settings, components)
    extractor = _extractor(components)
    claims: list[AtomicClaim] = []
    for passage in hits:
        claims.extend(
            extractor.extract(passage, query=query, source=sources.get(passage.span.source_id))
        )
    builder = GraphBuilder(
        _classifier(components),
        classification_config=_classification_config(components, variant),
        temporal_config=TemporalConfig(enabled=variant.temporal),
    )
    graph = builder.build(claims, query=query).graph
    board = BaselineInitialScorer(MetadataReliability()).score(graph, query)
    propagator = SignedBeliefPropagator() if variant.propagation else NoPropagationBaseline()
    board = propagator.apply(board, propagator.propagate(graph, board))
    conflicts = ConflictSetResolver().resolve(graph, board)
    budget = TokenBudget(
        total=settings.evidence_token_budget, reserved_output=settings.reserved_output_tokens
    )
    token_counter = _token_counter(components)
    selector = TopClaimsSelector() if variant.selection == "top" else GreedyConnectedSelector()
    selection = selector.select(
        graph, query, board, token_budget=budget, token_counter=token_counter, conflicts=conflicts
    )
    package = build_evidence_package(
        query, graph, board, conflicts, selection, budget=budget, token_counter=token_counter
    )
    return _finish(
        package,
        _generate(package, generator, cfg),
        extra_counts={
            "num_passages": len(hits),
            "num_claims": len(graph.nodes()),
            "num_graph_nodes": len(graph.nodes()),
            "num_graph_edges": len(graph.edges()),
        },
    )


_RUNNERS = {"passage": _run_passage, "claim": _run_claim, "graph": _run_graph}


def run_system(
    variant: SystemVariant,
    query: Query,
    documents: Sequence[Document],
    *,
    generator: TextGenerator,
    config: GenerationConfig,
    settings: RunSettings,
    components: RunComponents | None = None,
) -> SystemOutput:
    """Run one variant on one example. Gold data is intentionally not a parameter.

    ``components`` injects the retriever/chunker/extractor/classifier/reranker/
    token counter/cache shared across variants; when ``None`` the deterministic
    defaults are used (existing callers are unaffected).
    """

    return _RUNNERS[variant.family](
        query, documents, settings, variant, generator, config, components or _DEFAULT_COMPONENTS
    )


__all__ = [
    "VARIANTS",
    "EvidenceExtractor",
    "RunComponents",
    "RunSettings",
    "SystemOutput",
    "get_variant",
    "run_system",
]
