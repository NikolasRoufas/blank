"""Generic structured-generation claim extractor.

Builds a versioned prompt, calls a narrow :class:`StructuredModel`, then parses
and strictly validates the JSON output. Malformed JSON or schema violations
raise a typed, recoverable :class:`ExtractionError`. Claims whose source-span
text is not found verbatim in the passage are rejected (never given fabricated
provenance). The extractor never assigns truth belief.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field
from pydantic import ValidationError as PydanticValidationError

from egrag.adapters.extraction.interfaces import (
    ExtractionConfig,
    ExtractionResult,
    StructuredModel,
)
from egrag.adapters.extraction.postprocess import (
    claim_id,
    deduplicate_same_source,
    find_span,
    normalize_text,
    verify_span,
)
from egrag.adapters.extraction.prompts_loader import load_prompt
from egrag.domain.errors import ExtractionError
from egrag.domain.models import (
    AtomicClaim,
    ClaimProvenance,
    ClaimSemantics,
    ExtractionMetadata,
    ExtractionMethod,
    NonEmptyStr,
    Passage,
    Query,
    SourceMetadata,
)
from egrag.structured_json import StructuredOutputError, recover_json_object


class RawExtractedClaim(BaseModel):
    """Strict schema for a single claim emitted by a structured model."""

    model_config = ConfigDict(extra="forbid")

    claim_text: NonEmptyStr
    source_span_text: NonEmptyStr
    subject: str | None = None
    predicate: str | None = None
    object: str | None = None
    attribution: str | None = None
    named_entities: list[str] = Field(default_factory=list)
    temporal_expressions: list[str] = Field(default_factory=list)
    quantities: list[str] = Field(default_factory=list)
    negation: bool = False
    modality: list[str] = Field(default_factory=list)
    confidence: float = Field(default=0.7, ge=0.0, le=1.0)


class StructuredExtractionOutput(BaseModel):
    """Strict schema for the full model response."""

    model_config = ConfigDict(extra="forbid")

    claims: list[RawExtractedClaim] = Field(default_factory=list)


class StructuredClaimExtractor:
    """Extractor driven by a structured-generation model and a versioned prompt."""

    extractor_id = "structured-generation"
    version = "1.0.0"

    def __init__(
        self,
        model: StructuredModel,
        *,
        prompt_version: str = "extraction_v1",
        seed: int = 0,
        deterministic: bool = True,
    ) -> None:
        self._model = model
        self._prompt_version = prompt_version
        self._template = load_prompt(prompt_version)
        self._seed = seed
        self._deterministic = deterministic

    def build_prompt(self, passage: Passage, query: Query | None = None) -> str:
        """Render the extraction prompt with the passage as delimited untrusted data."""

        query_section = (
            f"## Query context\nThe user query is: {query.text}\n" if query is not None else ""
        )
        return self._template.replace("{query_section}", query_section).replace(
            "{passage}", passage.text
        )

    def extract(
        self,
        passage: Passage,
        *,
        query: Query | None = None,
        config: ExtractionConfig | None = None,
        source: SourceMetadata | None = None,
    ) -> list[AtomicClaim]:
        return list(self.extract_result(passage, query=query, config=config, source=source).claims)

    def extract_result(
        self,
        passage: Passage,
        *,
        query: Query | None = None,
        config: ExtractionConfig | None = None,
        source: SourceMetadata | None = None,
    ) -> ExtractionResult:
        cfg = config or ExtractionConfig()
        resolved_source = source or SourceMetadata(source_id=passage.span.source_id)
        prompt = self.build_prompt(passage, query)
        raw = self._model.complete(prompt, seed=self._seed, deterministic=self._deterministic)
        output, recovered = self._parse(raw)

        claims: list[AtomicClaim] = []
        warnings: list[str] = []
        if recovered:
            warnings.append(
                "structured output required JSON recovery (one object extracted from "
                "surrounding prose); model formatting was unfaithful to the contract"
            )
        for raw_claim in output.claims:
            span = find_span(passage, raw_claim.source_span_text)
            if span is None or not verify_span(passage, span):
                warnings.append(
                    f"rejected ungrounded claim (span not found in passage): "
                    f"{raw_claim.claim_text!r}"
                )
                continue
            normalized = normalize_text(raw_claim.claim_text)
            claim = AtomicClaim(
                claim_id=claim_id(passage.passage_id, normalized, span.start),
                text=normalized,
                provenance=ClaimProvenance(
                    source=resolved_source,
                    spans=(span,),
                    passage_id=passage.passage_id,
                ),
                extraction_confidence=raw_claim.confidence,
                semantics=ClaimSemantics(
                    subject=raw_claim.subject,
                    predicate=raw_claim.predicate,
                    object=raw_claim.object,
                    attribution=raw_claim.attribution,
                    named_entities=tuple(raw_claim.named_entities),
                    temporal_expressions=tuple(raw_claim.temporal_expressions),
                    quantities=tuple(raw_claim.quantities),
                    negation=raw_claim.negation,
                    modality=tuple(raw_claim.modality),
                ),
                extraction=ExtractionMetadata(
                    method=ExtractionMethod.STRUCTURED_GENERATION,
                    extractor_id=self.extractor_id,
                    extractor_version=self.version,
                    prompt_version=self._prompt_version,
                ),
            )
            if claim.extraction_confidence >= cfg.min_confidence:
                claims.append(claim)

        if cfg.deduplicate:
            claims, dedupe_warnings = deduplicate_same_source(claims)
            warnings.extend(dedupe_warnings)
        if cfg.max_claims is not None:
            claims = claims[: cfg.max_claims]
        return ExtractionResult(claims=tuple(claims), warnings=tuple(warnings))

    def _parse(self, raw: str) -> tuple[StructuredExtractionOutput, bool]:
        try:
            recovery = recover_json_object(raw)
        except StructuredOutputError as exc:
            raise ExtractionError(f"model did not return valid JSON: {exc}") from exc
        try:
            output = StructuredExtractionOutput.model_validate(recovery.data)
        except PydanticValidationError as exc:
            raise ExtractionError(
                f"model output failed schema validation: {exc.error_count()} error(s)"
            ) from exc
        return output, recovery.recovered


__all__ = ["RawExtractedClaim", "StructuredClaimExtractor", "StructuredExtractionOutput"]
