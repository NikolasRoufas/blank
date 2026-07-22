"""Integration tests for the application pipeline (acceptance cases 7, 8, 9)."""

from __future__ import annotations

import dataclasses
from datetime import UTC, datetime

import pytest

from egrag.application.pipeline import EGRagPipeline, PipelineComponents
from egrag.domain.models import EvidencePackage, GeneratedAnswer, Query
from egrag.domain.ports import GenerationParams
from egrag.fakes import build_demo_components

QUERY = Query(query_id="q1", text="Is EG-RAG generator agnostic?")


def _fixed_clock() -> datetime:
    return datetime(2024, 6, 1, 12, 0, 0, tzinfo=UTC)


@pytest.mark.integration
def test_pipeline_completes_query_to_answer() -> None:
    """Acceptance 7: the fake pipeline runs from query to a grounded answer."""

    pipeline = EGRagPipeline(build_demo_components())
    result = pipeline.run(QUERY)

    assert result.answer.text
    assert result.package.claims  # claims were extracted
    assert result.answer.cited_claim_ids  # the answer cites evidence
    assert result.metrics.num_claims == len(result.package.claims)
    # every cited claim exists in the package (grounding)
    known = {claim.claim_id for claim in result.package.claims}
    assert set(result.answer.cited_claim_ids) <= known


class _UpperCaseGenerator:
    """An alternative Generator implementation used to prove interchangeability."""

    def generate(self, package: EvidencePackage, params: GenerationParams) -> GeneratedAnswer:
        cited = tuple(item.claim_id for item in sorted(package.selected, key=lambda s: s.rank))
        return GeneratedAnswer(text="ALTERNATIVE ANSWER", cited_claim_ids=cited, abstained=False)


@pytest.mark.integration
def test_component_is_interchangeable_without_pipeline_change() -> None:
    """Acceptance 8: swapping one fake component requires no pipeline change."""

    base = build_demo_components()
    swapped = dataclasses.replace(base, generator=_UpperCaseGenerator())

    # The exact same pipeline class and call site work with the swapped component.
    result = EGRagPipeline(swapped).run(QUERY)

    assert result.answer.text == "ALTERNATIVE ANSWER"
    assert result.answer.cited_claim_ids  # selection still flows through


@pytest.mark.integration
def test_pipeline_output_is_deterministic() -> None:
    """Acceptance 9: identical inputs and a fixed clock yield identical results."""

    def make() -> EGRagPipeline:
        return EGRagPipeline(build_demo_components(), seed=7, clock=_fixed_clock)

    first = make().run(QUERY)
    second = make().run(QUERY)

    assert first == second
    assert first.model_dump_json() == second.model_dump_json()


@pytest.mark.integration
def test_pipeline_depends_only_on_protocols() -> None:
    """The pipeline accepts any objects satisfying the port protocols."""

    components = build_demo_components()
    assert isinstance(components, PipelineComponents)
    # Construction does not require any concrete adapter import.
    EGRagPipeline(components).run(QUERY)
