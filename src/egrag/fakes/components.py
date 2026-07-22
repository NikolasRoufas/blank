"""Deterministic fake port implementations and a tiny demo corpus.

Every fake here is pure and deterministic: identical inputs yield identical
outputs, with no randomness, wall-clock dependence, or I/O. The fakes honor the
project's scientific-integrity rules — for example, the belief propagator counts
*distinct sources* (so repetition from one source is not corroboration) and the
conflict resolver retains contradictions rather than discarding them.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Sequence
from datetime import UTC, datetime

from egrag.application.pipeline import PipelineComponents
from egrag.domain.graph import (
    connected_components,
    distinct_supporting_sources,
    induced_relation_ids,
)
from egrag.domain.models import (
    AtomicClaim,
    ClaimProvenance,
    ConflictSet,
    Document,
    EvidenceGraphSnapshot,
    EvidencePackage,
    EvidenceRelation,
    GeneratedAnswer,
    Passage,
    Query,
    ReasoningSubgraph,
    RelationType,
    SourceMetadata,
    SourceSpan,
)
from egrag.domain.ports import Embedding, GenerationParams
from egrag.graph.types import ClaimPair, RelationProbabilities

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_SENTENCE_RE = re.compile(r"[^.!?]+")
_NEGATIONS = frozenset({"not", "no", "never", "cannot", "without"})


def _tokens(text: str) -> set[str]:
    return set(_TOKEN_RE.findall(text.lower()))


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 0.0
    union = a | b
    if not union:
        return 0.0
    return len(a & b) / len(union)


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


# --- Retrieval ---------------------------------------------------------------


class FakeRetriever:
    """Ranks a fixed passage corpus by lexical overlap with the query."""

    def __init__(self, corpus: Sequence[Passage]) -> None:
        self._corpus = list(corpus)

    def retrieve(self, query: Query, top_k: int) -> list[Passage]:
        q = _tokens(query.text)
        scored = sorted(
            self._corpus,
            key=lambda p: (-_jaccard(_tokens(p.text), q), p.passage_id),
        )
        results: list[Passage] = []
        for rank, passage in enumerate(scored[:top_k]):
            score = _jaccard(_tokens(passage.text), q)
            results.append(passage.model_copy(update={"retrieval_score": score, "rank": rank}))
        return results


class FakeReranker:
    """Re-orders passages by query overlap; deterministic and stable."""

    def rerank(self, query: Query, passages: Sequence[Passage]) -> list[Passage]:
        q = _tokens(query.text)
        ordered = sorted(
            passages,
            key=lambda p: (-_jaccard(_tokens(p.text), q), p.passage_id),
        )
        return [p.model_copy(update={"rank": rank}) for rank, p in enumerate(ordered)]


class FakeEmbeddingProvider:
    """Deterministic bag-of-tokens embedding provider — no model, no download.

    Determinism is process-stable: token-to-bucket mapping uses SHA-256 (the
    built-in ``hash`` is salted per process and must not be used here). When an
    explicit ``vocabulary`` is supplied, each token maps one-to-one to its own
    dimension, which makes cosine similarity exactly reflect token overlap and
    lets tests assert a known ranking order without bucket collisions.
    """

    def __init__(self, dim: int = 8, vocabulary: Sequence[str] | None = None) -> None:
        if vocabulary is not None:
            self._vocab: dict[str, int] | None = {token: i for i, token in enumerate(vocabulary)}
            self._dim = len(self._vocab)
            self.name = "fake-embedding-vocab"
        else:
            self._vocab = None
            self._dim = dim
            self.name = f"fake-embedding-{dim}"

    def _bucket(self, token: str) -> int | None:
        if self._vocab is not None:
            return self._vocab.get(token)
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        return int.from_bytes(digest[:8], "big") % self._dim

    def embed(self, texts: Sequence[str]) -> list[Embedding]:
        vectors: list[Embedding] = []
        for text in texts:
            buckets = [0.0] * self._dim
            for token in _TOKEN_RE.findall(text.lower()):
                index = self._bucket(token)
                if index is not None:
                    buckets[index] += 1.0
            norm = sum(v * v for v in buckets) ** 0.5 or 1.0
            vectors.append(tuple(round(v / norm, 6) for v in buckets))
        return vectors


# --- Extraction & enrichment -------------------------------------------------


class FakeClaimExtractor:
    """Splits a passage into one atomic claim per sentence, with provenance."""

    def __init__(self, sources: dict[str, SourceMetadata]) -> None:
        self._sources = sources

    def extract(self, passage: Passage) -> list[AtomicClaim]:
        source_id = passage.span.source_id
        source = self._sources.get(source_id)
        if source is None:
            source = SourceMetadata(source_id=source_id)
        claims: list[AtomicClaim] = []
        index = 0
        for match in _SENTENCE_RE.finditer(passage.text):
            raw = match.group()
            stripped = raw.strip()
            if not stripped:
                continue
            lead = len(raw) - len(raw.lstrip())
            start = passage.span.start + match.start() + lead
            span = SourceSpan(
                source_id=source_id,
                start=start,
                end=start + len(stripped),
                text=stripped,
            )
            provenance = ClaimProvenance(
                source=source,
                spans=(span,),
                passage_id=passage.passage_id,
                observed_at=source.published_at,
            )
            claims.append(
                AtomicClaim(
                    claim_id=f"{passage.passage_id}#c{index}",
                    text=stripped,
                    provenance=provenance,
                    extraction_confidence=0.9,
                    asserted_at=source.published_at,
                )
            )
            index += 1
        return claims


class FakeTemporalResolver:
    """Fills a claim's assertion time from its source when absent."""

    def resolve(self, claim: AtomicClaim) -> AtomicClaim:
        if claim.asserted_at is None and claim.provenance.source.published_at is not None:
            return claim.model_copy(update={"asserted_at": claim.provenance.source.published_at})
        return claim


class FakeSourceReliabilityScorer:
    """Returns the source's configured reliability prior, defaulting to 0.5."""

    def __init__(self, default: float = 0.5) -> None:
        self._default = default

    def score(self, source: SourceMetadata) -> float:
        if source.reliability_prior is not None:
            return source.reliability_prior
        return self._default


class FakeInitialClaimScorer:
    """Assigns distinct initial belief and query-utility values to a claim."""

    def score(self, claim: AtomicClaim, query: Query) -> AtomicClaim:
        utility = round(_jaccard(_tokens(claim.text), _tokens(query.text)), 6)
        reliability = claim.source_reliability if claim.source_reliability is not None else 0.5
        belief = round(claim.extraction_confidence * reliability, 6)
        return claim.model_copy(update={"belief": belief, "query_utility": utility})


# --- Relations ---------------------------------------------------------------


class FakeRelationClassifier:
    """Classifies relations between claims by lexical overlap and negation.

    * Identical normalized text -> DUPLICATE.
    * Shared content but differing negation -> CONTRADICTION.
    * High overlap, no negation conflict -> SUPPORT.
    """

    def __init__(self, support_threshold: float = 0.5) -> None:
        self._support_threshold = support_threshold

    def classify(self, claims: Sequence[AtomicClaim]) -> list[EvidenceRelation]:
        relations: list[EvidenceRelation] = []
        for i in range(len(claims)):
            for j in range(i + 1, len(claims)):
                relation = self._relation(claims[i], claims[j], i, j)
                if relation is not None:
                    relations.append(relation)
        return sorted(relations, key=lambda r: r.relation_id)

    def _relation(self, a: AtomicClaim, b: AtomicClaim, i: int, j: int) -> EvidenceRelation | None:
        ta, tb = _tokens(a.text), _tokens(b.text)
        norm_a = " ".join(sorted(ta))
        norm_b = " ".join(sorted(tb))
        content_a, content_b = ta - _NEGATIONS, tb - _NEGATIONS
        negated_a, negated_b = bool(ta & _NEGATIONS), bool(tb & _NEGATIONS)
        overlap = _jaccard(ta, tb)
        rid = f"r{i}-{j}"

        if norm_a == norm_b:
            return self._make(rid, a, b, RelationType.DUPLICATE, 0.99)
        if negated_a != negated_b and _jaccard(content_a, content_b) >= self._support_threshold:
            return self._make(rid, a, b, RelationType.CONTRADICTION, 0.7)
        if overlap >= self._support_threshold:
            return self._make(rid, a, b, RelationType.SUPPORT, round(overlap, 6))
        return None

    @staticmethod
    def _make(
        rid: str,
        a: AtomicClaim,
        b: AtomicClaim,
        kind: RelationType,
        confidence: float,
    ) -> EvidenceRelation:
        return EvidenceRelation(
            relation_id=rid,
            source_claim_id=a.claim_id,
            target_claim_id=b.claim_id,
            relation_type=kind,
            relation_confidence=confidence,
            rationale=f"fake classifier: {kind.value}",
        )


# --- Graph reasoning (deterministic baselines) -------------------------------


class FakeBeliefPropagator:
    """Adjusts belief using distinct supporting sources and contradictions.

    Corroboration counts distinct sources only, so repeated mentions from one
    source do not inflate belief.
    """

    def __init__(self, support_step: float = 0.05, contradiction_step: float = 0.1) -> None:
        self._support_step = support_step
        self._contradiction_step = contradiction_step

    def propagate(self, snapshot: EvidenceGraphSnapshot) -> EvidenceGraphSnapshot:
        source_of = {c.claim_id: c.provenance.source.source_id for c in snapshot.claims}
        contradiction_count: dict[str, int] = {c.claim_id: 0 for c in snapshot.claims}
        for rel in snapshot.relations:
            if rel.relation_type is RelationType.CONTRADICTION:
                contradiction_count[rel.source_claim_id] += 1
                contradiction_count[rel.target_claim_id] += 1

        updated: list[AtomicClaim] = []
        for claim in snapshot.claims:
            base = claim.belief if claim.belief is not None else 0.5
            support = distinct_supporting_sources(snapshot, claim.claim_id, source_of)
            adjusted = (
                base
                + self._support_step * support
                - self._contradiction_step * contradiction_count[claim.claim_id]
            )
            updated.append(claim.model_copy(update={"belief": round(_clamp01(adjusted), 6)}))
        return snapshot.model_copy(update={"claims": tuple(updated)})


class FakeConflictResolver:
    """Groups contradiction edges into conflict sets; never discards them."""

    def resolve(self, snapshot: EvidenceGraphSnapshot) -> list[ConflictSet]:
        contradiction_rels = [
            r for r in snapshot.relations if r.relation_type is RelationType.CONTRADICTION
        ]
        components = connected_components(snapshot, relation_types={RelationType.CONTRADICTION})
        conflicts: list[ConflictSet] = []
        for index, members in enumerate(components):
            if len(members) < 2:
                continue
            member_set = set(members)
            relation_ids = tuple(
                sorted(
                    r.relation_id
                    for r in contradiction_rels
                    if r.source_claim_id in member_set and r.target_claim_id in member_set
                )
            )
            conflicts.append(
                ConflictSet(
                    conflict_id=f"conflict:{index}",
                    claim_ids=members,
                    relation_ids=relation_ids,
                    resolved=False,
                    resolution_note="unresolved: contradictory evidence retained for transparency",
                )
            )
        return conflicts


class FakeSubgraphSelector:
    """Selects the top-utility claims within a budget, with induced relations."""

    def select(
        self, snapshot: EvidenceGraphSnapshot, query: Query, budget: int
    ) -> ReasoningSubgraph:
        ranked = sorted(
            snapshot.claims,
            key=lambda c: (
                -(c.query_utility or 0.0),
                -(c.belief or 0.0),
                c.claim_id,
            ),
        )
        chosen = [c.claim_id for c in ranked[:budget]]
        return ReasoningSubgraph(
            subgraph_id=f"subgraph:{query.query_id}",
            claim_ids=tuple(chosen),
            relation_ids=induced_relation_ids(snapshot, chosen),
        )


class FakeTokenCounter:
    """Counts whitespace-delimited tokens."""

    def count(self, text: str) -> int:
        return len(text.split())


# --- Generation --------------------------------------------------------------


class FakeGenerator:
    """Composes a cited answer strictly from selected claim texts.

    It never invents content: the answer text is built only from claims present
    in the evidence package. With no selected evidence it abstains.
    """

    def generate(self, package: EvidencePackage, params: GenerationParams) -> GeneratedAnswer:
        by_id = {claim.claim_id: claim for claim in package.claims}
        ordered = sorted(package.selected, key=lambda s: s.rank)
        cited = [item.claim_id for item in ordered if item.claim_id in by_id]
        if not cited:
            return GeneratedAnswer(
                text="No supported evidence was found to answer the query.",
                cited_claim_ids=(),
                confidence=0.0,
                abstained=True,
            )
        parts = [f"({n + 1}) {by_id[cid].text}" for n, cid in enumerate(cited)]
        text = "Based on the retrieved evidence: " + "; ".join(parts) + "."
        beliefs = [b for cid in cited if (b := by_id[cid].belief) is not None]
        confidence = round(sum(beliefs) / len(beliefs), 6) if beliefs else None
        return GeneratedAnswer(
            text=text,
            cited_claim_ids=tuple(cited),
            confidence=confidence,
            abstained=False,
        )


class FakeGroundingVerifier:
    """Flags any cited claim that is absent from the evidence package."""

    def verify(self, answer: GeneratedAnswer, package: EvidencePackage) -> GeneratedAnswer:
        known = {claim.claim_id for claim in package.claims}
        unsupported = tuple(
            f"cited claim {cid!r} is not present in the evidence package"
            for cid in answer.cited_claim_ids
            if cid not in known
        )
        if not unsupported:
            return answer
        return answer.model_copy(update={"unsupported_warnings": unsupported})


class FakeStructuredModel:
    """Deterministic structured-generation model returning canned JSON.

    Responses are keyed by a substring expected to appear in the prompt (usually
    the passage text). The first key contained in the prompt wins; otherwise the
    default response is returned. This lets tests drive exact extractor behavior
    — including malformed output — without any real model.
    """

    name = "fake-structured-model"

    def __init__(
        self,
        responses: dict[str, str] | None = None,
        *,
        default: str = '{"claims": []}',
    ) -> None:
        self._responses = dict(responses or {})
        self._default = default

    def complete(self, prompt: str, *, seed: int = 0, deterministic: bool = True) -> str:
        for key, response in self._responses.items():
            if key in prompt:
                return response
        return self._default


class FakePairClassifier:
    """Deterministic relation classifier returning canned probabilities.

    Responses are keyed by ``(source_claim_id, target_claim_id)`` so tests can
    control direction precisely. Unkeyed pairs return the default (neutral).
    Results preserve input order, proving batching does not reorder.
    """

    classifier_id = "fake-nli"
    classifier_version = "1.0.0"
    model_revision: str | None = None

    def __init__(
        self,
        responses: dict[tuple[str, str], RelationProbabilities] | None = None,
        *,
        default: RelationProbabilities | None = None,
    ) -> None:
        self._responses = dict(responses or {})
        self._default = default or RelationProbabilities(
            entailment=0.0, contradiction=0.0, neutral=1.0
        )

    def classify(self, pairs: list[ClaimPair]) -> list[RelationProbabilities]:
        return [
            self._responses.get((pair.source.claim_id, pair.target.claim_id), self._default)
            for pair in pairs
        ]


# --- Wiring helpers ----------------------------------------------------------


def build_fake_components(
    corpus: Sequence[Passage],
    sources: dict[str, SourceMetadata],
) -> PipelineComponents:
    """Assemble a full set of fake components over the given corpus."""

    return PipelineComponents(
        retriever=FakeRetriever(corpus),
        reranker=FakeReranker(),
        claim_extractor=FakeClaimExtractor(sources),
        temporal_resolver=FakeTemporalResolver(),
        reliability_scorer=FakeSourceReliabilityScorer(),
        claim_scorer=FakeInitialClaimScorer(),
        relation_classifier=FakeRelationClassifier(),
        belief_propagator=FakeBeliefPropagator(),
        conflict_resolver=FakeConflictResolver(),
        subgraph_selector=FakeSubgraphSelector(),
        generator=FakeGenerator(),
        grounding_verifier=FakeGroundingVerifier(),
    )


def build_demo_corpus() -> tuple[list[Document], list[Passage], dict[str, SourceMetadata]]:
    """Build a tiny, fixed demo corpus exercising support and contradiction."""

    source_a = SourceMetadata(
        source_id="src-a",
        title="EG-RAG Reference",
        reliability_prior=0.9,
        published_at=datetime(2024, 1, 1, tzinfo=UTC),
    )
    source_b = SourceMetadata(
        source_id="src-b",
        title="EG-RAG Blog",
        reliability_prior=0.6,
        published_at=datetime(2025, 1, 1, tzinfo=UTC),
    )
    text_a = (
        "EG-RAG constructs an evidence graph from retrieved passages. "
        "EG-RAG is generator agnostic through adapters."
    )
    text_b = "EG-RAG builds an evidence graph from passages. EG-RAG is not generator agnostic."
    documents = [
        Document(document_id="doc-a", text=text_a, source=source_a),
        Document(document_id="doc-b", text=text_b, source=source_b),
    ]
    passages = [
        Passage(
            passage_id="p-a",
            document_id="doc-a",
            text=text_a,
            span=SourceSpan(source_id="src-a", start=0, end=len(text_a), text=text_a),
        ),
        Passage(
            passage_id="p-b",
            document_id="doc-b",
            text=text_b,
            span=SourceSpan(source_id="src-b", start=0, end=len(text_b), text=text_b),
        ),
    ]
    sources = {"src-a": source_a, "src-b": source_b}
    return documents, passages, sources


def build_demo_components() -> PipelineComponents:
    """Build fake components wired to the built-in demo corpus."""

    _, passages, sources = build_demo_corpus()
    return build_fake_components(passages, sources)


def build_demo_documents() -> list[Document]:
    """Build a small, fixed set of documents for the retrieval-only demo."""

    entries = [
        (
            "doc-graph",
            "EG-RAG Overview",
            "EG-RAG builds an evidence graph from retrieved passages. "
            "The graph records support and contradiction between atomic claims.",
        ),
        (
            "doc-retrieval",
            "Retrieval Notes",
            "Sparse retrieval ranks passages with BM25 over query terms. "
            "Dense retrieval ranks passages by embedding cosine similarity.",
        ),
        (
            "doc-fusion",
            "Fusion Notes",
            "Hybrid retrieval fuses sparse and dense rankings. "
            "Reciprocal rank fusion combines ranked lists without score calibration.",
        ),
        (
            "doc-generation",
            "Generation Notes",
            "The generator is instructed to use only the provided evidence. "
            "Unsupported statements are flagged as uncertain.",
        ),
    ]
    return [
        Document(
            document_id=doc_id,
            text=text,
            source=SourceMetadata(source_id=f"src-{doc_id}", title=title),
        )
        for doc_id, title, text in entries
    ]


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
    "FakeSubgraphSelector",
    "FakeTemporalResolver",
    "FakeTokenCounter",
    "build_demo_components",
    "build_demo_corpus",
    "build_demo_documents",
    "build_fake_components",
]
