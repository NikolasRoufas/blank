"""Explicit schema versioning for serializable domain models.

The schema version is independent of the package version. It changes only when
the serialized shape of domain models changes in a way that affects
compatibility, so that persisted evidence packages remain interpretable.
"""

from __future__ import annotations

SCHEMA_VERSION: str = "1.6.0"
"""Current version of the serializable domain-model schema.

History:
* 1.0.0 — initial schema.
* 1.1.0 — added structured-claim fields (``ClaimSemantics``,
  ``ExtractionMetadata``) and ``SourceMetadata.author``.
* 1.2.0 — added relation provenance/direction (``RelationMetadata``,
  ``RelationDirection``, ``EvidenceRelation.direction``/``metadata``) and the
  ``RelationType.NEUTRAL`` outcome.
* 1.3.0 — added conflict resolution detail (``ConflictMember``,
  ``ConflictOutcome``, ``ConflictSet.members``/``outcome``/``preferred_claim_id``)
  and ``ReasoningPath``.
* 1.4.0 — added evidence-package generation fields (``EvidenceStatus``,
  ``SelectedEvidence`` status/belief/utility/reason, ``GenerationPolicy``,
  ``PackageBudget``, ``EvidencePackage`` duplicate_groups/generation_policy/
  budget) and ``ChatMessage``.
* 1.5.0 — added extended reproducibility fields to ``RunManifest`` (git commit,
  start/end timestamps, environment, model identifiers/revisions, prompt
  versions, corpus fingerprint, artifact paths, warnings, deterministic
  capability, resolved configuration).
* 1.6.0 — added the query-conditioned reasoning-connectivity relation
  ``RelationType.BRIDGES`` and ``BridgeMetadata`` (``EvidenceRelation.bridge``).
  All additions are optional/defaulted, so earlier payloads still deserialize.
"""
