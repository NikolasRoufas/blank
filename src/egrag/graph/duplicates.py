"""Lexical duplicate detection (exact and normalized).

Duplicate detection is kept separate from support. It identifies that two
claims say the same thing; it never merges or erases nodes, so every claim's
provenance is preserved. Semantic (paraphrase) duplicates are handled separately
during classification via mutual entailment.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from egrag.domain.models import AtomicClaim
from egrag.graph.text_utils import normalize
from egrag.graph.types import DuplicateConfig


@dataclass(frozen=True)
class DuplicatePair:
    """An unordered duplicate pair (``a_id`` < ``b_id``) and its kind."""

    a_id: str
    b_id: str
    kind: str  # "exact" or "normalized"


def detect_lexical_duplicates(
    claims: Sequence[AtomicClaim],
    config: DuplicateConfig | None = None,
) -> list[DuplicatePair]:
    """Return unordered duplicate pairs detected by exact/normalized text.

    Deterministic: claims are processed in id order and pairs are ordered by id.
    """

    cfg = config or DuplicateConfig()
    if not (cfg.detect_exact or cfg.detect_normalized):
        return []

    ordered = sorted(claims, key=lambda c: c.claim_id)
    buckets: dict[str, list[AtomicClaim]] = {}
    for claim in ordered:
        key = normalize(claim.text) if cfg.detect_normalized else claim.text
        buckets.setdefault(key, []).append(claim)

    pairs: list[DuplicatePair] = []
    for group in buckets.values():
        for i in range(len(group)):
            for j in range(i + 1, len(group)):
                a, b = group[i], group[j]
                kind = "exact" if a.text == b.text else "normalized"
                if kind == "exact" and not cfg.detect_exact:
                    # Only normalized detection requested; still a duplicate.
                    kind = "normalized"
                pairs.append(DuplicatePair(a_id=a.claim_id, b_id=b.claim_id, kind=kind))
    pairs.sort(key=lambda p: (p.a_id, p.b_id))
    return pairs


__all__ = ["DuplicatePair", "detect_lexical_duplicates"]
