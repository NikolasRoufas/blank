"""Unit tests for hybrid fusion retrieval (acceptance 4-9, 24)."""

from __future__ import annotations

import pytest

from egrag.adapters.retrieval.base import (
    BaseRetriever,
    FusionStrategy,
    RankStats,
    ScoredPassage,
)
from egrag.adapters.retrieval.hybrid import HybridRetriever
from egrag.domain.models import Passage, Query, SourceSpan


def _passage(passage_id: str, source_id: str = "src", text: str = "some text") -> Passage:
    return Passage(
        passage_id=passage_id,
        document_id=f"doc-{source_id}",
        text=text,
        span=SourceSpan(source_id=source_id, start=0, end=len(text), text=text),
    )


class StubRetriever(BaseRetriever):
    """Returns a fixed ranked list of (passage, score), highest first."""

    def __init__(self, name: str, scored: list[tuple[Passage, float]]) -> None:
        self.name = name
        self._scored = scored

    def _rank(self, query: Query, top_k: int) -> tuple[list[ScoredPassage], RankStats]:
        items = [
            ScoredPassage(
                passage=passage,
                score=score,
                components={self.name: score},
                retriever=self.name,
            )
            for passage, score in self._scored
        ]
        items.sort(key=lambda item: (-item.score, item.passage.passage_id))
        return items, RankStats()


QUERY = Query(query_id="q", text="query text")


@pytest.mark.unit
def test_rrf_handles_passage_missing_from_one_retriever() -> None:
    """Acceptance 4: RRF fuses a passage present in only one retriever."""

    p1, p2, p3 = _passage("p1"), _passage("p2"), _passage("p3")
    sparse = StubRetriever("sparse", [(p1, 5.0), (p2, 3.0)])
    dense = StubRetriever("dense", [(p2, 0.9), (p3, 0.8)])  # p1 missing here, p3 missing in sparse
    hybrid = HybridRetriever(
        {"sparse": sparse, "dense": dense}, strategy=FusionStrategy.RECIPROCAL_RANK
    )
    results = hybrid.search(QUERY, top_k=10)
    ids = {item.passage.passage_id for item in results}
    assert ids == {"p1", "p2", "p3"}  # union, nothing dropped
    # p2 appears in both lists, so it should rank first under RRF.
    assert results[0].passage.passage_id == "p2"


@pytest.mark.unit
def test_weighted_fusion_rejects_invalid_weights() -> None:
    """Acceptance 5: invalid weights are rejected."""

    p1 = _passage("p1")
    sparse = StubRetriever("sparse", [(p1, 1.0)])
    dense = StubRetriever("dense", [(p1, 1.0)])
    components = {"sparse": sparse, "dense": dense}

    with pytest.raises(ValueError):
        HybridRetriever(components, weights={"sparse": -1.0, "dense": 1.0})
    with pytest.raises(ValueError):
        HybridRetriever(components, weights={"sparse": 0.0, "dense": 0.0})
    with pytest.raises(ValueError):
        HybridRetriever(components, weights={"unknown": 1.0})


@pytest.mark.unit
def test_invalid_strategy_rejected() -> None:
    """Acceptance 5 (strategy): an invalid fusion strategy is rejected."""

    p1 = _passage("p1")
    with pytest.raises(ValueError):
        HybridRetriever({"a": StubRetriever("a", [(p1, 1.0)])}, strategy="bogus")


@pytest.mark.unit
def test_weight_normalization_is_scale_invariant() -> None:
    """Acceptance 6: weights are normalized, so uniform scaling changes nothing."""

    p1, p2 = _passage("p1"), _passage("p2")
    sparse = StubRetriever("sparse", [(p1, 10.0), (p2, 0.0)])
    dense = StubRetriever("dense", [(p2, 10.0), (p1, 0.0)])
    components = {"sparse": sparse, "dense": dense}

    base = HybridRetriever(components, weights={"sparse": 1.0, "dense": 1.0}).search(QUERY, 10)
    scaled = HybridRetriever(components, weights={"sparse": 5.0, "dense": 5.0}).search(QUERY, 10)
    assert [s.passage.passage_id for s in base] == [s.passage.passage_id for s in scaled]
    assert [round(s.score, 6) for s in base] == [round(s.score, 6) for s in scaled]
    # Equal weights, symmetric scores -> fused scores equal and in [0, 1].
    assert all(0.0 <= s.score <= 1.0 for s in base)


@pytest.mark.unit
def test_weighted_fusion_respects_weights() -> None:
    """Acceptance 6: normalized weights drive the fused score as documented."""

    p1, p2 = _passage("p1"), _passage("p2")
    # sparse favors p1, dense favors p2; weight sparse heavily.
    sparse = StubRetriever("sparse", [(p1, 10.0), (p2, 0.0)])
    dense = StubRetriever("dense", [(p2, 10.0), (p1, 0.0)])
    hybrid = HybridRetriever(
        {"sparse": sparse, "dense": dense}, weights={"sparse": 3.0, "dense": 1.0}
    )
    results = hybrid.search(QUERY, top_k=10)
    assert results[0].passage.passage_id == "p1"  # heavier sparse weight wins


@pytest.mark.unit
def test_identical_passages_same_source_are_deduplicated() -> None:
    """Acceptance 8: identical passages (same id) are deduplicated in fusion."""

    shared = _passage("p-shared", source_id="src-a")
    sparse = StubRetriever("sparse", [(shared, 5.0)])
    dense = StubRetriever("dense", [(shared, 0.9)])
    hybrid = HybridRetriever({"sparse": sparse, "dense": dense})
    results = hybrid.search(QUERY, top_k=10)
    assert len(results) == 1
    assert results[0].passage.passage_id == "p-shared"


@pytest.mark.unit
def test_similar_passages_different_sources_both_preserved() -> None:
    """Acceptance 9: passages from different sources keep both provenance records."""

    p_a = _passage("p-a", source_id="src-a")
    p_b = _passage("p-b", source_id="src-b")
    sparse = StubRetriever("sparse", [(p_a, 5.0)])
    dense = StubRetriever("dense", [(p_b, 0.9)])
    hybrid = HybridRetriever({"sparse": sparse, "dense": dense})
    results = hybrid.search(QUERY, top_k=10)
    sources = {item.passage.span.source_id for item in results}
    assert sources == {"src-a", "src-b"}


@pytest.mark.unit
def test_component_scores_inspectable_after_fusion() -> None:
    """Acceptance 24: raw component scores remain inspectable after fusion."""

    shared = _passage("p1")
    sparse = StubRetriever("sparse", [(shared, 5.0)])
    dense = StubRetriever("dense", [(shared, 0.9)])
    hybrid = HybridRetriever({"sparse": sparse, "dense": dense})
    result = hybrid.search(QUERY, top_k=10)[0]
    assert result.components == {"sparse": 5.0, "dense": 0.9}
    # raw component scores are kept distinct from the normalized fused score
    assert result.normalized_score is not None
    assert result.score != 5.0


@pytest.mark.unit
def test_fusion_ties_are_deterministic() -> None:
    """Acceptance 7: fused ties break deterministically by passage id."""

    p1, p2 = _passage("p2"), _passage("p1")  # note reversed insertion
    sparse = StubRetriever("sparse", [(p1, 1.0), (p2, 1.0)])
    dense = StubRetriever("dense", [(p1, 1.0), (p2, 1.0)])
    hybrid = HybridRetriever({"sparse": sparse, "dense": dense})
    ids = [item.passage.passage_id for item in hybrid.search(QUERY, top_k=10)]
    assert ids == ["p1", "p2"]  # equal fused scores -> sorted by id
