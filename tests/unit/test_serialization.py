"""Unit tests for evidence serialization (acceptance cases 17-18)."""

from __future__ import annotations

import pytest

from egrag.application.pipeline import EGRagPipeline
from egrag.domain.errors import SerializationError
from egrag.domain.models import EvidencePackage, Query
from egrag.fakes import build_demo_components
from egrag.serialization import JsonEvidenceSerializer


def _package() -> EvidencePackage:
    pipeline = EGRagPipeline(build_demo_components())
    result = pipeline.run(Query(query_id="q1", text="Is EG-RAG generator agnostic?"))
    return result.package


@pytest.mark.unit
def test_serialization_round_trip_preserves_data() -> None:
    """Acceptance 17: a serialize/deserialize round trip preserves the package."""

    serializer = JsonEvidenceSerializer()
    package = _package()

    restored = serializer.deserialize(serializer.serialize(package))

    assert restored == package


@pytest.mark.unit
def test_round_trip_is_stable_across_two_passes() -> None:
    """Serializing the restored package yields identical bytes."""

    serializer = JsonEvidenceSerializer()
    package = _package()

    first = serializer.serialize(package)
    second = serializer.serialize(serializer.deserialize(first))

    assert first == second


@pytest.mark.unit
@pytest.mark.parametrize(
    "payload",
    [
        "",  # empty
        "{ not json",  # invalid JSON syntax
        "{}",  # valid JSON, missing required fields
        '{"package_id": "", "query": {"query_id": "q", "text": "t"}}',  # invalid field
        "[1, 2, 3]",  # wrong shape
    ],
)
def test_malformed_data_rejected_safely(payload: str) -> None:
    """Acceptance 18: malformed serialized data raises a domain SerializationError."""

    serializer = JsonEvidenceSerializer()
    with pytest.raises(SerializationError):
        serializer.deserialize(payload)


@pytest.mark.unit
def test_schema_version_present_in_serialized_output() -> None:
    """The schema version travels inside the serialized package."""

    serializer = JsonEvidenceSerializer()
    text = serializer.serialize(_package())
    assert '"schema_version"' in text
