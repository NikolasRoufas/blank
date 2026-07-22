"""Full-pipeline generation tests (29, 30) and the fake-pipeline demonstration."""

from __future__ import annotations

import json

import pytest

from egrag.answering import answer_query
from egrag.domain.models import PipelineResult, Query
from egrag.fakes import build_demo_documents
from egrag.generation import FakeTextGenerator, GeneratorCapabilities

QUERY = Query(query_id="q", text="how does dense retrieval rank passages")


@pytest.mark.integration
def test_fake_components_execute_full_pipeline() -> None:
    """Acceptance 30: fake components can execute the full pipeline."""

    result = answer_query(QUERY, build_demo_documents(), generator=FakeTextGenerator())
    assert isinstance(result, PipelineResult)
    assert result.answer.text
    assert result.package.selected  # evidence was assembled and selected
    assert result.package.schema_version  # versioned package
    # citations (if any) reference real claims
    known = {c.claim_id for c in result.package.claims}
    assert set(result.answer.cited_claim_ids) <= known


@pytest.mark.integration
def test_full_pipeline_is_deterministic() -> None:
    first = answer_query(QUERY, build_demo_documents(), generator=FakeTextGenerator())
    second = answer_query(QUERY, build_demo_documents(), generator=FakeTextGenerator())
    assert first.answer.text == second.answer.text
    assert first.answer.cited_claim_ids == second.answer.cited_claim_ids


class _UncitedGenerator:
    """Returns a confident but uncited answer to exercise grounding warnings."""

    def capabilities(self) -> GeneratorCapabilities:
        return GeneratorCapabilities(context_limit=100_000, deterministic_decoding=True)

    def complete(self, prompt: str, config: object) -> str:
        return json.dumps(
            {
                "answer": "Acme revenue grew strongly and the deal absolutely closed on time.",
                "citations": [],
                "uncertainty": "",
            }
        )


@pytest.mark.integration
def test_grounding_warnings_preserved_in_pipeline_result() -> None:
    """Acceptance 29: grounding validation warnings are preserved in PipelineResult."""

    result = answer_query(QUERY, build_demo_documents(), generator=_UncitedGenerator())
    assert result.answer.unsupported_warnings  # grounding flagged the uncited answer
    assert any("uncited" in w or "no citations" in w for w in result.answer.unsupported_warnings)
