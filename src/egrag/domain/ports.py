"""Typed ports (interfaces) for every model-facing capability.

These are structural :class:`typing.Protocol` types. Concrete implementations
live in adapters or in the deterministic ``fakes`` package; the application
pipeline depends only on these protocols, never on concrete implementations.

Provider-specific arguments must not appear in these signatures. Generic
controls that any implementation can honor (e.g. ``seed`` and ``deterministic``)
are passed via :class:`GenerationParams`.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field

from egrag.domain.models import (
    AtomicClaim,
    ConflictSet,
    Document,
    EvidenceGraphSnapshot,
    EvidencePackage,
    EvidenceRelation,
    GeneratedAnswer,
    Passage,
    Query,
    ReasoningSubgraph,
    SourceMetadata,
)

Embedding = tuple[float, ...]
"""A dense embedding vector represented as an immutable tuple of floats."""


class GenerationParams(BaseModel):
    """Generic, provider-independent generation controls."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    seed: int = Field(default=0, ge=0)
    deterministic: bool = True
    max_tokens: int | None = Field(default=None, ge=1)


@runtime_checkable
class DocumentLoader(Protocol):
    """Loads raw documents from a generic source reference (e.g. a path)."""

    def load(self, source: str) -> Iterable[Document]: ...


@runtime_checkable
class Chunker(Protocol):
    """Splits a document into retrievable passages."""

    def chunk(self, document: Document) -> list[Passage]: ...


@runtime_checkable
class Retriever(Protocol):
    """Retrieves candidate passages for a query."""

    def retrieve(self, query: Query, top_k: int) -> list[Passage]: ...


@runtime_checkable
class Reranker(Protocol):
    """Reorders candidate passages by relevance to a query."""

    def rerank(self, query: Query, passages: Sequence[Passage]) -> list[Passage]: ...


@runtime_checkable
class EmbeddingProvider(Protocol):
    """Produces dense embeddings for texts."""

    def embed(self, texts: Sequence[str]) -> list[Embedding]: ...


@runtime_checkable
class ClaimExtractor(Protocol):
    """Decomposes a passage into atomic claims with provenance."""

    def extract(self, passage: Passage) -> list[AtomicClaim]: ...


@runtime_checkable
class RelationClassifier(Protocol):
    """Classifies typed relations between atomic claims."""

    def classify(self, claims: Sequence[AtomicClaim]) -> list[EvidenceRelation]: ...


@runtime_checkable
class TemporalResolver(Protocol):
    """Resolves and normalizes temporal metadata on a claim."""

    def resolve(self, claim: AtomicClaim) -> AtomicClaim: ...


@runtime_checkable
class SourceReliabilityScorer(Protocol):
    """Scores the reliability of a source as a value in [0, 1].

    This is a configurable prior on trust, not a scientific ground truth.
    """

    def score(self, source: SourceMetadata) -> float: ...


@runtime_checkable
class InitialClaimScorer(Protocol):
    """Assigns initial belief and query utility to a claim.

    Returns a new claim; the distinct score fields must not be conflated.
    """

    def score(self, claim: AtomicClaim, query: Query) -> AtomicClaim: ...


@runtime_checkable
class BeliefPropagator(Protocol):
    """Propagates belief across the evidence graph, returning a new snapshot."""

    def propagate(self, snapshot: EvidenceGraphSnapshot) -> EvidenceGraphSnapshot: ...


@runtime_checkable
class ConflictResolver(Protocol):
    """Identifies and resolves conflict sets in the evidence graph.

    Contradictory evidence is surfaced, never silently discarded.
    """

    def resolve(self, snapshot: EvidenceGraphSnapshot) -> list[ConflictSet]: ...


@runtime_checkable
class SubgraphSelector(Protocol):
    """Selects a compact reasoning subgraph for a query within a budget."""

    def select(
        self, snapshot: EvidenceGraphSnapshot, query: Query, budget: int
    ) -> ReasoningSubgraph: ...


@runtime_checkable
class TokenCounter(Protocol):
    """Counts tokens in a piece of text."""

    def count(self, text: str) -> int: ...


@runtime_checkable
class EvidenceSerializer(Protocol):
    """Serializes and deserializes evidence packages."""

    def serialize(self, package: EvidencePackage) -> str: ...

    def deserialize(self, data: str) -> EvidencePackage: ...


@runtime_checkable
class Generator(Protocol):
    """Generates a grounded answer from an evidence package.

    Implementations must be instructed not to invent unsupported evidence.
    """

    def generate(self, package: EvidencePackage, params: GenerationParams) -> GeneratedAnswer: ...


@runtime_checkable
class GroundingVerifier(Protocol):
    """Verifies an answer against its evidence package and flags unsupported claims."""

    def verify(self, answer: GeneratedAnswer, package: EvidencePackage) -> GeneratedAnswer: ...


@runtime_checkable
class CacheBackend(Protocol):
    """A simple string-keyed cache for serialized artifacts."""

    def get(self, key: str) -> str | None: ...

    def set(self, key: str, value: str) -> None: ...


__all__ = [
    "BeliefPropagator",
    "CacheBackend",
    "Chunker",
    "ClaimExtractor",
    "ConflictResolver",
    "DocumentLoader",
    "Embedding",
    "EmbeddingProvider",
    "EvidenceSerializer",
    "GenerationParams",
    "Generator",
    "GroundingVerifier",
    "InitialClaimScorer",
    "RelationClassifier",
    "Reranker",
    "Retriever",
    "SourceReliabilityScorer",
    "SubgraphSelector",
    "TemporalResolver",
    "TokenCounter",
]
