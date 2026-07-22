"""Attribution validation for parsed generation output.

Hard failures (raising :class:`GenerationError`): citing a claim ID that does
not exist, or citing rejected evidence as support. These are never silently
accepted. Softer grounding heuristics live in the grounding verifier.
"""

from __future__ import annotations

from egrag.domain.errors import GenerationError
from egrag.domain.models import EvidencePackage, EvidenceStatus
from egrag.generation.parsing import ParsedAnswer


def validate_attribution(parsed: ParsedAnswer, package: EvidencePackage) -> None:
    """Reject invalid attribution; raise :class:`GenerationError` on violations."""

    known = {claim.claim_id for claim in package.claims}
    unknown = [cid for cid in parsed.cited_claim_ids if cid not in known]
    if unknown:
        raise GenerationError(f"answer cites unknown claim id(s): {sorted(unknown)}")

    status_by_id = {item.claim_id: item.status for item in package.selected}
    rejected = [
        cid for cid in parsed.cited_claim_ids if status_by_id.get(cid) is EvidenceStatus.REJECTED
    ]
    if rejected:
        raise GenerationError(f"answer cites rejected evidence as support: {sorted(rejected)}")


__all__ = ["validate_attribution"]
