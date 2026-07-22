"""Deterministic fake implementations of the domain ports.

These fakes are dependency-free, deterministic, and infrastructure-free. They
exist for tests and demonstrations and let the full pipeline run from a query to
a grounded answer without any optional dependency, network access, or model
download. Each fake implements one port from :mod:`egrag.domain.ports`.
"""

from __future__ import annotations

from egrag.fakes.components import (
    FakeBeliefPropagator,
    FakeClaimExtractor,
    FakeConflictResolver,
    FakeEmbeddingProvider,
    FakeGenerator,
    FakeGroundingVerifier,
    FakeInitialClaimScorer,
    FakePairClassifier,
    FakeRelationClassifier,
    FakeReranker,
    FakeRetriever,
    FakeSourceReliabilityScorer,
    FakeStructuredModel,
    FakeTemporalResolver,
    FakeTokenCounter,
    build_demo_components,
    build_demo_corpus,
    build_demo_documents,
    build_fake_components,
)

__all__ = [
    "FakeBeliefPropagator",
    "FakeClaimExtractor",
    "FakeConflictResolver",
    "FakeEmbeddingProvider",
    "FakeGenerator",
    "FakeGroundingVerifier",
    "FakeInitialClaimScorer",
    "FakePairClassifier",
    "FakeRelationClassifier",
    "FakeReranker",
    "FakeRetriever",
    "FakeSourceReliabilityScorer",
    "FakeStructuredModel",
    "FakeTemporalResolver",
    "FakeTokenCounter",
    "build_demo_components",
    "build_demo_corpus",
    "build_demo_documents",
    "build_fake_components",
]
