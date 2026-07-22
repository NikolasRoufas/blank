"""Full end-to-end answering composition (a composition root).

Wires retrieval → chunking → extraction → graph construction → reasoning →
evidence-package assembly → generation → grounding, returning a
:class:`PipelineResult`. Evidence processing and conflict handling all happen
before the (thin, interchangeable) generator is invoked.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime

from egrag import __version__
from egrag.adapters.extraction import SentenceClaimExtractor
from egrag.adapters.retrieval import BM25Retriever, SentenceAwareChunker, prepare_passages
from egrag.domain.models import (
    Document,
    GeneratedAnswer,
    PipelineMetrics,
    PipelineResult,
    Query,
    RunManifest,
)
from egrag.generation import (
    FakeTextGenerator,
    GenerationConfig,
    GenerationService,
    TextGenerator,
    build_evidence_package,
)
from egrag.graph import GraphBuilder, LexicalPairClassifier
from egrag.reasoning import (
    BaselineInitialScorer,
    CharacterTokenCounter,
    ConflictSetResolver,
    GreedyConnectedSelector,
    MetadataReliability,
    SignedBeliefPropagator,
    TokenBudget,
)


def answer_query(
    query: Query,
    documents: Sequence[Document],
    *,
    generator: TextGenerator | None = None,
    config: GenerationConfig | None = None,
    top_k: int = 4,
    evidence_token_budget: int = 512,
    reserved_output_tokens: int = 128,
    chunk_size: int = 512,
    chunk_overlap: int = 64,
) -> PipelineResult:
    """Answer a query over the given documents, returning a full PipelineResult."""

    generator = generator or FakeTextGenerator()
    cfg = config or GenerationConfig()
    token_counter = CharacterTokenCounter()

    chunker = SentenceAwareChunker(chunk_size=chunk_size, overlap=chunk_overlap)
    passages = prepare_passages(documents, chunker)
    sources = {doc.source.source_id: doc.source for doc in documents}
    hits = BM25Retriever(passages).retrieve(query, top_k)

    extractor = SentenceClaimExtractor()
    claims = []
    for passage in hits:
        claims.extend(
            extractor.extract(passage, query=query, source=sources.get(passage.span.source_id))
        )

    graph = GraphBuilder(LexicalPairClassifier()).build(claims, query=query).graph
    board = BaselineInitialScorer(MetadataReliability()).score(graph, query)
    propagator = SignedBeliefPropagator()
    board = propagator.apply(board, propagator.propagate(graph, board))
    conflicts = ConflictSetResolver().resolve(graph, board)
    budget = TokenBudget(total=evidence_token_budget, reserved_output=reserved_output_tokens)
    selection = GreedyConnectedSelector().select(
        graph, query, board, token_budget=budget, token_counter=token_counter, conflicts=conflicts
    )
    package = build_evidence_package(
        query, graph, board, conflicts, selection, budget=budget, token_counter=token_counter
    )

    if package.selected:
        answer = GenerationService(token_counter=token_counter).generate(package, generator, cfg)
    else:
        answer = GeneratedAnswer(
            text="No supported evidence was found to answer the query.",
            abstained=True,
            uncertainty="No evidence met the selection threshold.",
        )

    metrics = PipelineMetrics(
        num_passages=len(hits),
        num_claims=len(graph.nodes()),
        num_relations=len(graph.edges()),
        num_conflicts=len(conflicts),
        num_selected=len(package.selected),
    )
    manifest = RunManifest(
        egrag_version=__version__,
        seed=cfg.seed if cfg.seed is not None else 0,
        deterministic=cfg.deterministic,
        created_at=datetime.now(UTC),
    )
    return PipelineResult(
        query=query, answer=answer, package=package, metrics=metrics, manifest=manifest
    )


__all__ = ["answer_query"]
