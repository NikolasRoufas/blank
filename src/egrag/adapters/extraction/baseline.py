"""Deterministic, model-free sentence-based claim extractor.

Splits a passage into sentences (and, optionally, conjunction clauses), and
derives structured fields with explicit, deterministic rules: negation,
modality/uncertainty markers, attribution, dates, quantities, and named
entities are detected by regex and copied verbatim. The extractor never assigns
truth belief and never invents semantic fields it cannot determine.

Pronoun resolution rule: a leading pronoun is resolved only when
``resolve_clear_pronouns`` is enabled AND exactly one distinct named entity
precedes the clause in the passage. Otherwise the pronoun is left as written and
a warning is recorded.

Source-span offsets are expressed in source coordinates: a passage-local offset
``i`` maps to source offset ``passage.span.start + i``.
"""

from __future__ import annotations

import re

from egrag.adapters.extraction.interfaces import ExtractionConfig, ExtractionResult
from egrag.adapters.extraction.postprocess import (
    claim_id,
    deduplicate_same_source,
    normalize_text,
    sentence_spans,
)
from egrag.domain.models import (
    AtomicClaim,
    ClaimProvenance,
    ClaimSemantics,
    ExtractionMetadata,
    ExtractionMethod,
    Passage,
    Query,
    SourceMetadata,
    SourceSpan,
)

_EXTRACTOR_ID = "sentence-baseline"
_VERSION = "1.0.0"

_NEGATION_RE = re.compile(
    r"\b(?:not|no|never|cannot|without|neither|nor|none)\b|n't", re.IGNORECASE
)
_MODALITY_WORDS = (
    "may",
    "might",
    "could",
    "would",
    "should",
    "must",
    "possibly",
    "perhaps",
    "reportedly",
    "allegedly",
    "apparently",
    "likely",
    "probably",
    "presumably",
    "seemingly",
    "supposedly",
)
_MODALITY_RE = re.compile(r"\b(" + "|".join(_MODALITY_WORDS) + r")\b", re.IGNORECASE)
_ATTRIBUTION_ACCORDING_RE = re.compile(r"according to ([A-Z][\w.\-]*)", re.IGNORECASE)
_ATTRIBUTION_VERB_RE = re.compile(
    r"\b([A-Z][\w.\-]*)\s+(?:said|reported|claimed|stated|announced|argued)\b"
)
_DATE_RE = re.compile(
    r"\b(?:January|February|March|April|May|June|July|August|September|October|"
    r"November|December)\s+\d{1,2}(?:,\s*\d{4})?\b"
    r"|\b\d{4}-\d{2}-\d{2}\b"
    r"|\b(?:19|20)\d{2}\b",
    re.IGNORECASE,
)
_QUANTITY_RE = re.compile(
    r"\$\d[\d,]*(?:\.\d+)?(?:\s?(?:million|billion|thousand))?"
    r"|\b\d[\d,]*\.\d+\b"
    r"|\b\d[\d,]*\s?(?:%|percent|million|billion|thousand|kg|km|miles|dollars)\b"
    r"|\b\d[\d,]*%",
    re.IGNORECASE,
)
_ENTITY_RE = re.compile(r"\b[A-Z][a-zA-Z0-9]*(?:\s+[A-Z][a-zA-Z0-9]*)*\b")
_MONTHS = frozenset(
    {
        "January",
        "February",
        "March",
        "April",
        "May",
        "June",
        "July",
        "August",
        "September",
        "October",
        "November",
        "December",
    }
)
_PRONOUNS = frozenset(
    {"it", "he", "she", "they", "them", "this", "that", "these", "those", "him", "her", "its"}
)
# Capitalized function words that the proper-noun heuristic would otherwise
# misread as named entities when they start a sentence (e.g. "The", "An").
# Filtered case-insensitively so a real entity like "IT" is unaffected only when
# it matches one of these lowercased forms.
_NON_ENTITY_WORDS = frozenset(
    {
        "the",
        "a",
        "an",
        "this",
        "that",
        "these",
        "those",
        "it",
        "its",
        "they",
        "their",
        "them",
        "we",
        "our",
        "he",
        "she",
        "his",
        "her",
        "there",
        "here",
        "in",
        "on",
        "at",
        "by",
        "for",
        "from",
        "with",
        "and",
        "but",
        "or",
        "as",
        "if",
        "when",
        "while",
        "then",
        "also",
        "however",
        "according",
        "reports",
        "reportedly",
        "both",
        "no",
        "not",
    }
)


def _ordered_unique(values: list[str]) -> tuple[str, ...]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            out.append(value)
    return tuple(out)


def _strip_leading_non_entities(phrase: str) -> str:
    """Drop leading capitalized function words (e.g. "The Acme" -> "Acme")."""

    words = phrase.split()
    while words and words[0].lower() in _NON_ENTITY_WORDS:
        words.pop(0)
    return " ".join(words)


def _entities(text: str) -> tuple[str, ...]:
    found = [_strip_leading_non_entities(m.group().strip()) for m in _ENTITY_RE.finditer(text)]
    return _ordered_unique([e for e in found if e and e not in _MONTHS])


def _attribution(text: str) -> str | None:
    match = _ATTRIBUTION_ACCORDING_RE.search(text) or _ATTRIBUTION_VERB_RE.search(text)
    return match.group(1) if match else None


class SentenceClaimExtractor:
    """Rule-based extractor that turns sentences into atomic claims.

    Satisfies the :class:`egrag.domain.ports.ClaimExtractor` protocol via
    :meth:`extract`; :meth:`extract_result` additionally returns warnings.
    """

    extractor_id = _EXTRACTOR_ID
    version = _VERSION

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
        resolved_source = self._resolve_source(passage, source)
        claims: list[AtomicClaim] = []
        warnings: list[str] = []

        for sent_start, sent_end in sentence_spans(passage.text):
            sentence = passage.text[sent_start:sent_end]
            if "?" in sentence:  # questions carry no factual assertion
                continue
            for local_start, local_end in self._clause_spans(
                sentence, sent_start, cfg.split_conjunctions
            ):
                clause = passage.text[local_start:local_end]
                if len([t for t in clause.split() if any(ch.isalpha() for ch in t)]) < 2:
                    continue
                claim = self._build_claim(passage, resolved_source, clause, local_start, cfg)
                if claim.extraction_confidence >= cfg.min_confidence:
                    claims.append(claim)

        if cfg.deduplicate:
            claims, dedupe_warnings = deduplicate_same_source(claims)
            warnings.extend(dedupe_warnings)
        if cfg.max_claims is not None:
            claims = claims[: cfg.max_claims]
        return ExtractionResult(claims=tuple(claims), warnings=tuple(warnings))

    @staticmethod
    def _resolve_source(passage: Passage, source: SourceMetadata | None) -> SourceMetadata:
        if source is not None:
            return source
        return SourceMetadata(source_id=passage.span.source_id)

    def _clause_spans(self, sentence: str, sent_start: int, split: bool) -> list[tuple[int, int]]:
        if not split:
            return [(sent_start, sent_start + len(sentence))]
        delimiters = list(re.finditer(r"\s+and\s+", sentence))
        if not delimiters:
            return [(sent_start, sent_start + len(sentence))]
        parts: list[tuple[int, int]] = []
        prev = 0
        for delimiter in delimiters:
            parts.append((prev, delimiter.start()))
            prev = delimiter.end()
        parts.append((prev, len(sentence)))
        # Only split when every resulting proposition stays meaningful.
        if not all(len(sentence[s:e].split()) >= 3 for s, e in parts):
            return [(sent_start, sent_start + len(sentence))]
        spans: list[tuple[int, int]] = []
        for s, e in parts:
            piece = sentence[s:e]
            lead = len(piece) - len(piece.lstrip())
            trail = len(piece) - len(piece.rstrip())
            start = sent_start + s + lead
            end = sent_start + e - trail
            if end > start:
                spans.append((start, end))
        return spans

    def _build_claim(
        self,
        passage: Passage,
        source: SourceMetadata,
        clause: str,
        local_start: int,
        cfg: ExtractionConfig,
    ) -> AtomicClaim:
        abs_start = passage.span.start + local_start
        span = SourceSpan(
            source_id=passage.span.source_id,
            start=abs_start,
            end=abs_start + len(clause),
            text=clause,
        )

        entities = _entities(clause)
        normalized = normalize_text(clause)
        claim_warnings: list[str] = []
        subject = entities[0] if entities else None

        tokens = normalized.split()
        leading = tokens[0].strip(",.;:").lower() if tokens else ""
        if leading in _PRONOUNS:
            antecedents = _entities(passage.text[:local_start])
            if cfg.resolve_clear_pronouns and len(antecedents) == 1:
                resolved = antecedents[0]
                tokens[0] = resolved
                normalized = " ".join(tokens)
                subject = resolved
                claim_warnings.append(f"resolved pronoun {leading!r} -> {resolved!r}")
            else:
                subject = tokens[0]
                claim_warnings.append(f"ambiguous pronoun {leading!r} left unresolved")

        semantics = ClaimSemantics(
            subject=subject,
            predicate=None,
            object=None,
            attribution=_attribution(clause),
            named_entities=entities,
            temporal_expressions=_ordered_unique([m.group() for m in _DATE_RE.finditer(clause)]),
            quantities=_ordered_unique([m.group() for m in _QUANTITY_RE.finditer(clause)]),
            negation=bool(_NEGATION_RE.search(clause)),
            modality=_ordered_unique([m.group() for m in _MODALITY_RE.finditer(clause)]),
        )
        metadata = ExtractionMetadata(
            method=ExtractionMethod.SENTENCE_BASELINE,
            extractor_id=self.extractor_id,
            extractor_version=self.version,
            prompt_version=None,
            warnings=tuple(claim_warnings),
        )
        return AtomicClaim(
            claim_id=claim_id(passage.passage_id, normalized, span.start),
            text=normalized,
            provenance=ClaimProvenance(
                source=source,
                spans=(span,),
                passage_id=passage.passage_id,
            ),
            extraction_confidence=cfg.default_confidence,
            semantics=semantics,
            extraction=metadata,
        )


__all__ = ["SentenceClaimExtractor"]
