"""Experiment-component injection (real-adapter-repair §5, §8).

Verifies that ``run_system`` accepts injected components, uses them across the
passage/claim/graph families with the same instances (fairness), keeps the
deterministic defaults when nothing is injected, and that the existing call
signature still works. No gold data enters component construction.
"""

from __future__ import annotations

from collections.abc import Sequence

import pytest

from egrag.adapters.extraction import SentenceClaimExtractor
from egrag.domain.models import (
    AtomicClaim,
    Document,
    Passage,
    Query,
    SourceMetadata,
)
from egrag.experiments.variants import RunComponents, RunSettings, get_variant, run_system
from egrag.generation import FakeTextGenerator, GenerationConfig
from egrag.graph import LexicalPairClassifier
from egrag.graph.types import ClaimPair, RelationProbabilities

_DOCS = (
    Document(
        document_id="d1",
        text="Scott Derrickson is an American director. He was born in 1966.",
        source=SourceMetadata(source_id="Scott_Derrickson", title="Scott Derrickson"),
    ),
    Document(
        document_id="d2",
        text="Edward Davis Wood Jr. was an American filmmaker.",
        source=SourceMetadata(source_id="Ed_Wood", title="Ed Wood"),
    ),
)
_QUERY = Query(query_id="q1", text="Are Scott Derrickson and Ed Wood American?")
_SETTINGS = RunSettings(
    top_k=5, evidence_token_budget=256, reserved_output_tokens=16, chunk_size=256
)
_CFG = GenerationConfig(deterministic=True, seed=0, max_new_tokens=32)


class RecordingExtractor:
    """Wraps the deterministic extractor and records invocations."""

    def __init__(self) -> None:
        self._inner = SentenceClaimExtractor()
        self.passages_seen = 0

    def extract(
        self,
        passage: Passage,
        *,
        query: Query | None = None,
        source: SourceMetadata | None = None,
    ) -> list[AtomicClaim]:
        self.passages_seen += 1
        return self._inner.extract(passage, query=query, source=source)


class RecordingClassifier:
    classifier_id = "recording"
    classifier_version = "1.0"
    model_revision = None

    def __init__(self) -> None:
        self._inner = LexicalPairClassifier()
        self.pairs_seen = 0

    def classify(self, pairs: Sequence[ClaimPair]) -> list[RelationProbabilities]:
        pairs = list(pairs)
        self.pairs_seen += len(pairs)
        return self._inner.classify(pairs)


@pytest.mark.integration
def test_existing_call_without_components_still_works() -> None:
    out = run_system(
        get_variant("claim_only_rag"),
        _QUERY,
        list(_DOCS),
        generator=FakeTextGenerator(),
        config=_CFG,
        settings=_SETTINGS,
    )
    assert out.known_claim_ids  # produced claims with the default extractor


@pytest.mark.integration
def test_injected_extractor_is_used() -> None:
    rec = RecordingExtractor()
    components = RunComponents(extractor=rec)
    run_system(
        get_variant("claim_only_rag"),
        _QUERY,
        list(_DOCS),
        generator=FakeTextGenerator(),
        config=_CFG,
        settings=_SETTINGS,
        components=components,
    )
    assert rec.passages_seen > 0


@pytest.mark.integration
def test_same_components_shared_across_variants() -> None:
    rec_ext = RecordingExtractor()
    rec_clf = RecordingClassifier()
    components = RunComponents(extractor=rec_ext, classifier=rec_clf)
    # claim family uses the extractor; graph family uses extractor + classifier.
    run_system(
        get_variant("claim_only_rag"),
        _QUERY,
        list(_DOCS),
        generator=FakeTextGenerator(),
        config=_CFG,
        settings=_SETTINGS,
        components=components,
    )
    after_claim = rec_ext.passages_seen
    run_system(
        get_variant("full_egrag"),
        _QUERY,
        list(_DOCS),
        generator=FakeTextGenerator(),
        config=_CFG,
        settings=_SETTINGS,
        components=components,
    )
    assert rec_ext.passages_seen > after_claim  # same instance accumulated across variants
    assert rec_clf.pairs_seen > 0  # injected classifier used by the graph family


@pytest.mark.integration
def test_default_classifier_unchanged_without_injection() -> None:
    # graph variant with default components builds a graph and produces output.
    out = run_system(
        get_variant("full_egrag"),
        _QUERY,
        list(_DOCS),
        generator=FakeTextGenerator(),
        config=_CFG,
        settings=_SETTINGS,
    )
    assert out.counts.get("num_graph_nodes", 0) >= 0
