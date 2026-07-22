"""Validated, versionable, immutable domain models for EG-RAG.

All models are Pydantic v2, frozen (immutable where practical), forbid unknown
fields, and are JSON-serializable. The five distinct evidentiary quantities are
modeled as separate fields and never collapsed:

* ``extraction_confidence`` — confidence the claim was correctly extracted;
* ``source_reliability`` — trust in the originating source (a configurable
  prior, not a scientific ground truth);
* ``belief`` — estimated probability the claim is true given evidence;
* ``query_utility`` — usefulness of the claim for answering the query;
* ``relation_confidence`` — confidence attached to an evidence relation.

This module imports only the standard library and Pydantic.
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Annotated, Any

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    model_validator,
)

from egrag.domain.version import SCHEMA_VERSION

# --- Shared constrained types ------------------------------------------------

NonEmptyStr = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
"""A string that is non-empty after stripping surrounding whitespace."""

Probability = Annotated[float, Field(ge=0.0, le=1.0)]
"""A normalized value in the closed interval [0, 1]."""


class RelationType(StrEnum):
    """The kinds of relation an edge in the evidence graph may carry.

    The public/graph-facing names map to these values as follows: SUPPORTS →
    ``SUPPORT``, CONTRADICTS → ``CONTRADICTION``, DUPLICATE_OF → ``DUPLICATE``,
    SUPERSEDES → ``SUPERSESSION``, DEPENDS_ON → ``DEPENDENCY``. ``NEUTRAL`` is a
    classification outcome that is NOT stored as an edge during normal
    construction; it may be stored only when a debugging mode requests it.
    """

    SUPPORT = "support"
    CONTRADICTION = "contradiction"
    DUPLICATE = "duplicate"
    DEPENDENCY = "dependency"
    SUPERSESSION = "supersession"
    NEUTRAL = "neutral"
    # Query-conditioned reasoning-connectivity relation (NOT evidential): two
    # claims contribute complementary information to a query reasoning chain via a
    # shared entity/slot, without either entailing the other. BRIDGES never enters
    # belief propagation, conflict detection, or independent corroboration.
    BRIDGES = "bridges"


class RelationDirection(StrEnum):
    """Whether a relation is directed (source → target) or symmetric."""

    DIRECTED = "directed"
    SYMMETRIC = "symmetric"


class ConflictOutcome(StrEnum):
    """The resolution outcome for a conflict set."""

    PREFERRED = "preferred"
    SUPERSEDED = "superseded"
    UNRESOLVED = "unresolved"
    REJECTED_LOW_EVIDENCE = "rejected_low_evidence"
    EXCLUDED_IRRELEVANT = "excluded_irrelevant"


class EvidenceStatus(StrEnum):
    """How a selected claim should be presented to the generator."""

    ACCEPTED = "accepted"
    DISPUTED = "disputed"
    SUPERSEDED = "superseded"
    REJECTED = "rejected"


class ExtractionMethod(StrEnum):
    """How an atomic claim was produced."""

    SENTENCE_BASELINE = "sentence_baseline"
    STRUCTURED_GENERATION = "structured_generation"
    HUGGINGFACE = "huggingface"
    FAKE = "fake"


def _unique(values: tuple[str, ...]) -> bool:
    """Return True when all identifiers in ``values`` are distinct."""

    return len(set(values)) == len(values)


# --- Base --------------------------------------------------------------------


class _Frozen(BaseModel):
    """Base for immutable domain models that forbid unknown fields."""

    model_config = ConfigDict(frozen=True, extra="forbid")


# --- Corpus & retrieval ------------------------------------------------------


class Query(_Frozen):
    """A natural-language information need."""

    query_id: NonEmptyStr
    text: NonEmptyStr
    created_at: datetime | None = None


class SourceMetadata(_Frozen):
    """Provenance metadata describing a source of text."""

    source_id: NonEmptyStr
    title: str | None = None
    author: str | None = None
    uri: str | None = None
    published_at: datetime | None = None
    # A configurable prior on source trust. NOT a scientific ground truth.
    reliability_prior: Probability | None = None


class SourceSpan(_Frozen):
    """A contiguous character span within a source's text."""

    source_id: NonEmptyStr
    start: int = Field(ge=0)
    end: int = Field(ge=0)
    text: NonEmptyStr

    @model_validator(mode="after")
    def _check_offsets(self) -> SourceSpan:
        if self.end <= self.start:
            msg = f"span end ({self.end}) must be greater than start ({self.start})"
            raise ValueError(msg)
        return self


class Document(_Frozen):
    """A whole source document prior to chunking."""

    document_id: NonEmptyStr
    text: NonEmptyStr
    source: SourceMetadata


class Passage(_Frozen):
    """A retrievable chunk of a document, with offsets back into the source."""

    passage_id: NonEmptyStr
    document_id: NonEmptyStr
    text: NonEmptyStr
    span: SourceSpan
    retrieval_score: float | None = None
    rank: int | None = Field(default=None, ge=0)


# --- Claims ------------------------------------------------------------------


class ClaimProvenance(_Frozen):
    """Links a claim back to one or more concrete source spans."""

    source: SourceMetadata
    spans: tuple[SourceSpan, ...] = Field(min_length=1)
    passage_id: NonEmptyStr | None = None
    observed_at: datetime | None = None


class ClaimSemantics(_Frozen):
    """Structured semantic decomposition of a claim.

    Fields are best-effort and may be ``None``/empty when an extractor cannot
    determine them reliably; extractors must not invent missing values.
    """

    subject: str | None = None
    predicate: str | None = None
    object: str | None = None
    # The attributed speaker for a reported statement (e.g. "Bob" in
    # "According to Bob, ..."). Distinguishes reported statements from facts.
    attribution: str | None = None
    named_entities: tuple[str, ...] = ()
    temporal_expressions: tuple[str, ...] = ()
    quantities: tuple[str, ...] = ()
    # Explicit negation present in the claim (e.g. "did not").
    negation: bool = False
    # Uncertainty/modality markers preserved verbatim (e.g. "may", "reportedly").
    modality: tuple[str, ...] = ()


class ExtractionMetadata(_Frozen):
    """Records how and by what an atomic claim was extracted."""

    method: ExtractionMethod
    extractor_id: NonEmptyStr
    extractor_version: NonEmptyStr
    prompt_version: str | None = None
    warnings: tuple[str, ...] = ()


class AtomicClaim(_Frozen):
    """An atomic factual claim with provenance and distinct evidentiary scores.

    ``belief``, ``query_utility``, and ``source_reliability`` are optional
    because they are assigned by downstream pipeline stages; updates produce new
    instances via :meth:`pydantic.BaseModel.model_copy`. ``extraction_confidence``
    is the only score an extractor may set — it never assigns truth belief.
    """

    claim_id: NonEmptyStr
    text: NonEmptyStr
    provenance: ClaimProvenance

    extraction_confidence: Probability
    source_reliability: Probability | None = None
    belief: Probability | None = None
    query_utility: Probability | None = None

    semantics: ClaimSemantics | None = None
    extraction: ExtractionMetadata | None = None

    asserted_at: datetime | None = None
    valid_from: datetime | None = None
    valid_to: datetime | None = None

    @model_validator(mode="after")
    def _check_validity_window(self) -> AtomicClaim:
        if (
            self.valid_from is not None
            and self.valid_to is not None
            and self.valid_to < self.valid_from
        ):
            msg = (
                f"valid_to ({self.valid_to.isoformat()}) must not precede "
                f"valid_from ({self.valid_from.isoformat()})"
            )
            raise ValueError(msg)
        return self


# --- Graph -------------------------------------------------------------------


class RelationMetadata(_Frozen):
    """Provenance for an inferred relation: which classifier produced it, how."""

    classifier_id: NonEmptyStr
    classifier_version: NonEmptyStr
    model_revision: str | None = None
    created_at: datetime | None = None
    explanation: str | None = None
    # Optional feature trace (e.g. classifier probabilities or signals).
    features: dict[str, float] = Field(default_factory=dict)


class BridgeMetadata(_Frozen):
    """Query-conditioned connectivity metadata for a ``BRIDGES`` relation.

    Kept separate from evidential ``relation_confidence`` (score taxonomy): the
    authoritative score for a bridge is ``bridge_confidence`` (query-conditioned
    connectivity), which must never be read as evidential support.
    """

    bridge_entity: str | None = None
    bridge_terms: tuple[str, ...] = ()
    query_conditioned: bool = True
    bridge_confidence: Probability = 0.0
    bridge_method_id: NonEmptyStr = "unknown"


class EvidenceRelation(_Frozen):
    """A typed relation between two atomic claims.

    ``direction`` records whether the relation is directed (``source → target``)
    or symmetric. For symmetric relations (e.g. CONTRADICTION, DUPLICATE) a
    single canonical edge is stored with ``source_claim_id < target_claim_id``;
    consumers must treat it as bidirectional. ``bridge`` is populated only for
    ``BRIDGES`` relations and carries query-conditioned connectivity metadata.
    """

    relation_id: NonEmptyStr
    source_claim_id: NonEmptyStr
    target_claim_id: NonEmptyStr
    relation_type: RelationType
    relation_confidence: Probability
    direction: RelationDirection = RelationDirection.DIRECTED
    rationale: str | None = None
    metadata: RelationMetadata | None = None
    bridge: BridgeMetadata | None = None

    @model_validator(mode="after")
    def _no_self_relation(self) -> EvidenceRelation:
        if self.source_claim_id == self.target_claim_id:
            msg = f"relation {self.relation_id!r} cannot link a claim to itself"
            raise ValueError(msg)
        return self


class EvidenceGraphSnapshot(_Frozen):
    """An immutable snapshot of the evidence graph (claims + relations)."""

    snapshot_id: NonEmptyStr
    schema_version: str = SCHEMA_VERSION
    claims: tuple[AtomicClaim, ...] = ()
    relations: tuple[EvidenceRelation, ...] = ()

    @model_validator(mode="after")
    def _check_integrity(self) -> EvidenceGraphSnapshot:
        claim_ids = tuple(claim.claim_id for claim in self.claims)
        if not _unique(claim_ids):
            raise ValueError("duplicate claim_id values are not allowed")
        relation_ids = tuple(rel.relation_id for rel in self.relations)
        if not _unique(relation_ids):
            raise ValueError("duplicate relation_id values are not allowed")
        known = set(claim_ids)
        for rel in self.relations:
            missing = {rel.source_claim_id, rel.target_claim_id} - known
            if missing:
                msg = (
                    f"relation {rel.relation_id!r} references unknown "
                    f"claim id(s): {sorted(missing)}"
                )
                raise ValueError(msg)
        return self


class ConflictMember(_Frozen):
    """A competing claim within a conflict set, with the signals used to judge it.

    Each evidentiary quantity is kept distinct; ``timestamp`` may be ``None`` when
    unknown.
    """

    claim_id: NonEmptyStr
    propagated_belief: Probability
    source_reliability: Probability
    independent_support: int = Field(ge=0)
    relation_confidence: Probability | None = None
    timestamp: datetime | None = None


class ConflictSet(_Frozen):
    """A set of mutually conflicting claims, retained for transparency.

    Contradictory evidence is never silently discarded; it is surfaced here. A
    resolution is recorded as an explicit ``outcome`` with a ``rationale``; close
    competitors are left ``UNRESOLVED`` rather than forced to a winner.
    """

    conflict_id: NonEmptyStr
    claim_ids: tuple[NonEmptyStr, ...] = Field(min_length=2)
    relation_ids: tuple[NonEmptyStr, ...] = ()
    members: tuple[ConflictMember, ...] = ()
    outcome: ConflictOutcome | None = None
    preferred_claim_id: NonEmptyStr | None = None
    resolved: bool = False
    resolution_note: str | None = None

    @model_validator(mode="after")
    def _unique_members(self) -> ConflictSet:
        if not _unique(tuple(self.claim_ids)):
            raise ValueError("a conflict set may not list a claim more than once")
        if self.preferred_claim_id is not None and self.preferred_claim_id not in self.claim_ids:
            raise ValueError("preferred_claim_id must be one of the conflict's claim_ids")
        return self


class ReasoningPath(_Frozen):
    """A truly ordered sequence of claims (use only for linear reasoning).

    For branching evidence structures use :class:`ReasoningSubgraph` instead.
    """

    path_id: NonEmptyStr
    claim_ids: tuple[NonEmptyStr, ...] = ()

    @model_validator(mode="after")
    def _unique_members(self) -> ReasoningPath:
        if not _unique(tuple(self.claim_ids)):
            raise ValueError("a reasoning path may not list a claim more than once")
        return self


class ReasoningSubgraph(_Frozen):
    """The compact subgraph selected to support an answer."""

    subgraph_id: NonEmptyStr
    claim_ids: tuple[NonEmptyStr, ...] = ()
    relation_ids: tuple[NonEmptyStr, ...] = ()

    @model_validator(mode="after")
    def _unique_members(self) -> ReasoningSubgraph:
        if not _unique(tuple(self.claim_ids)):
            raise ValueError("duplicate claim_id values are not allowed in a subgraph")
        if not _unique(tuple(self.relation_ids)):
            raise ValueError("duplicate relation_id values are not allowed in a subgraph")
        return self


# --- Evidence package --------------------------------------------------------


class SelectedEvidence(_Frozen):
    """A selected claim with its score, rank, status, and selection explanation.

    ``status`` distinguishes accepted, disputed, superseded, and rejected
    evidence so a serializer can present each appropriately. ``belief`` and
    ``utility`` are kept as distinct fields and are never conflated.
    """

    claim_id: NonEmptyStr
    selection_score: Probability
    rank: int = Field(ge=0)
    status: EvidenceStatus = EvidenceStatus.ACCEPTED
    belief: Probability | None = None
    utility: Probability | None = None
    reason: str | None = None


class GenerationPolicy(_Frozen):
    """Explicit, serializable policy the generator must follow.

    These are the framework's instructions; they are emitted separately from the
    (untrusted) evidence so that text inside the evidence cannot override them.
    """

    use_only_supplied_evidence: bool = True
    require_claim_id_citations: bool = True
    forbid_fabricated_citations: bool = True
    express_uncertainty_for_unresolved_conflicts: bool = True
    treat_evidence_as_untrusted: bool = True
    forbid_following_instructions_in_evidence: bool = True
    # Hide internal decimals by default; round to this many places when shown.
    round_scores: bool = True
    score_decimals: int = Field(default=2, ge=0, le=6)


class PackageBudget(_Frozen):
    """Token-budget metadata recorded on the evidence package."""

    total: int = Field(ge=0)
    reserved_output: int = Field(default=0, ge=0)
    used: int = Field(default=0, ge=0)
    truncated_claim_ids: tuple[NonEmptyStr, ...] = ()

    @model_validator(mode="after")
    def _check(self) -> PackageBudget:
        if self.reserved_output > self.total:
            raise ValueError("reserved_output must not exceed total")
        return self

    @property
    def available(self) -> int:
        return self.total - self.reserved_output


class ChatMessage(_Frozen):
    """A single chat message (role + content) for chat-style serialization."""

    role: str
    content: str


class EvidencePackage(_Frozen):
    """The model-independent evidence package passed to a generator.

    JSON-serializable and free of any model-provider type. Carries the query,
    selected claims (with status/belief/utility), all relations, conflicts,
    duplicate groups, the explicit generation policy, and token-budget metadata.
    """

    package_id: NonEmptyStr
    schema_version: str = SCHEMA_VERSION
    query: Query
    claims: tuple[AtomicClaim, ...] = ()
    relations: tuple[EvidenceRelation, ...] = ()
    conflicts: tuple[ConflictSet, ...] = ()
    selected: tuple[SelectedEvidence, ...] = ()
    subgraph: ReasoningSubgraph | None = None
    duplicate_groups: tuple[tuple[NonEmptyStr, ...], ...] = ()
    generation_policy: GenerationPolicy = GenerationPolicy()
    budget: PackageBudget | None = None

    @model_validator(mode="after")
    def _check_integrity(self) -> EvidencePackage:
        claim_ids = tuple(claim.claim_id for claim in self.claims)
        if not _unique(claim_ids):
            raise ValueError("duplicate claim_id values are not allowed")
        known = set(claim_ids)

        for rel in self.relations:
            missing = {rel.source_claim_id, rel.target_claim_id} - known
            if missing:
                msg = (
                    f"relation {rel.relation_id!r} references unknown "
                    f"claim id(s): {sorted(missing)}"
                )
                raise ValueError(msg)

        selected_ids = tuple(item.claim_id for item in self.selected)
        if not _unique(selected_ids):
            raise ValueError("a claim may not be selected more than once")
        missing_selected = set(selected_ids) - known
        if missing_selected:
            msg = f"selected evidence references unknown claim id(s): {sorted(missing_selected)}"
            raise ValueError(msg)

        for conflict in self.conflicts:
            missing_conflict = set(conflict.claim_ids) - known
            if missing_conflict:
                msg = (
                    f"conflict set {conflict.conflict_id!r} references unknown "
                    f"claim id(s): {sorted(missing_conflict)}"
                )
                raise ValueError(msg)

        if self.subgraph is not None:
            missing_sub = set(self.subgraph.claim_ids) - known
            if missing_sub:
                msg = (
                    f"subgraph {self.subgraph.subgraph_id!r} references unknown "
                    f"claim id(s): {sorted(missing_sub)}"
                )
                raise ValueError(msg)
        return self


# --- Answer & run results ----------------------------------------------------


class GeneratedAnswer(_Frozen):
    """A grounded answer with citations and uncertainty information."""

    text: NonEmptyStr
    cited_claim_ids: tuple[NonEmptyStr, ...] = ()
    confidence: Probability | None = None
    abstained: bool = False
    uncertainty: str = ""
    unsupported_warnings: tuple[str, ...] = ()

    @model_validator(mode="after")
    def _unique_citations(self) -> GeneratedAnswer:
        if not _unique(tuple(self.cited_claim_ids)):
            raise ValueError("duplicate cited_claim_id values are not allowed")
        return self


class PipelineMetrics(_Frozen):
    """Counts and timings describing a pipeline run."""

    num_passages: int = Field(default=0, ge=0)
    num_claims: int = Field(default=0, ge=0)
    num_relations: int = Field(default=0, ge=0)
    num_conflicts: int = Field(default=0, ge=0)
    num_selected: int = Field(default=0, ge=0)
    durations_ms: dict[str, float] = Field(default_factory=dict)


class RunManifest(_Frozen):
    """Reproducibility metadata for a single pipeline run."""

    egrag_version: NonEmptyStr
    schema_version: str = SCHEMA_VERSION
    seed: int = Field(ge=0)
    deterministic: bool = True
    created_at: datetime
    component_identities: dict[str, str] = Field(default_factory=dict)
    config_hash: str | None = None
    input_hash: str | None = None
    # Extended reproducibility fields.
    git_commit: str | None = None
    started_at: datetime | None = None
    ended_at: datetime | None = None
    environment: dict[str, str] = Field(default_factory=dict)
    model_identifiers: dict[str, str] = Field(default_factory=dict)
    model_revisions: dict[str, str] = Field(default_factory=dict)
    prompt_versions: dict[str, str] = Field(default_factory=dict)
    corpus_fingerprint: str | None = None
    artifact_paths: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    deterministic_capability: bool = True
    resolved_config: dict[str, Any] = Field(default_factory=dict)


class PipelineResult(_Frozen):
    """The complete result of a pipeline run."""

    schema_version: str = SCHEMA_VERSION
    query: Query
    answer: GeneratedAnswer
    package: EvidencePackage
    metrics: PipelineMetrics
    manifest: RunManifest


__all__ = [
    "AtomicClaim",
    "BridgeMetadata",
    "ChatMessage",
    "ClaimProvenance",
    "ClaimSemantics",
    "ConflictMember",
    "ConflictOutcome",
    "ConflictSet",
    "Document",
    "EvidenceGraphSnapshot",
    "EvidencePackage",
    "EvidenceRelation",
    "EvidenceStatus",
    "ExtractionMetadata",
    "ExtractionMethod",
    "GeneratedAnswer",
    "GenerationPolicy",
    "NonEmptyStr",
    "PackageBudget",
    "Passage",
    "PipelineMetrics",
    "PipelineResult",
    "Probability",
    "Query",
    "ReasoningPath",
    "ReasoningSubgraph",
    "RelationDirection",
    "RelationMetadata",
    "RelationType",
    "RunManifest",
    "SelectedEvidence",
    "SourceMetadata",
    "SourceSpan",
]
