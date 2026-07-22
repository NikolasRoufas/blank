"""Bridge-milestone regression tests (evidential vs reasoning-connectivity).

All deterministic and offline (oracle classifier + fakes; no model download).
"""

from __future__ import annotations

from collections.abc import Sequence

import pytest

from egrag.domain.models import (
    AtomicClaim,
    ClaimProvenance,
    ClaimSemantics,
    EvidenceGraphSnapshot,
    Query,
    RelationType,
    SourceMetadata,
    SourceSpan,
)
from egrag.experiments.mechanism_eval import (
    VariantFlags,
    activation,
    build_run,
    mechanism_metrics,
    run_example,
)
from egrag.experiments.mechanisms import build_suite
from egrag.graph import (
    DeterministicBridgeDetector,
    GraphSerializer,
    StructuralContradictionGate,
    detect_bridges,
    extract_entities,
    structural_contradiction_ok,
)
from egrag.graph.types import ClaimPair, RelationProbabilities

_SUITE = build_suite()
_BY_CAT: dict[str, list] = {}
for _ex in _SUITE:
    _BY_CAT.setdefault(_ex.category, []).append(_ex)


def _claim(
    cid: str, text: str, ents: tuple[str, ...] = (), *, subject: str = "", negation: bool = False
) -> AtomicClaim:
    return AtomicClaim(
        claim_id=cid,
        text=text,
        provenance=ClaimProvenance(
            source=SourceMetadata(source_id=cid),
            spans=(SourceSpan(source_id=cid, start=0, end=len(text), text=text),),
        ),
        extraction_confidence=0.9,
        semantics=ClaimSemantics(named_entities=ents, subject=subject or None, negation=negation),
    )


class _AlwaysContradict:
    def classify(self, pairs: Sequence[ClaimPair]) -> list[RelationProbabilities]:
        return [
            RelationProbabilities(entailment=0.0, contradiction=0.95, neutral=0.05) for _ in pairs
        ]


@pytest.mark.integration
def test_bridges_serialize_roundtrip() -> None:
    """Regression 1."""

    q = Query(query_id="q", text="Who founded the company that acquired DeepMind?")
    a = _claim("a", "Google acquired DeepMind.", ("Google", "DeepMind"))
    b = _claim("b", "Google was founded by Larry Page.", ("Google", "Larry Page"))
    bridges = detect_bridges(q, [a, b], [])
    assert bridges and bridges[0].relation_type is RelationType.BRIDGES
    snap = EvidenceGraphSnapshot(snapshot_id="g", claims=(a, b), relations=tuple(bridges))
    restored = GraphSerializer().deserialize(GraphSerializer().serialize(snap))
    assert restored.relations[0].relation_type is RelationType.BRIDGES
    assert restored.relations[0].bridge.bridge_entity == "Google"


@pytest.mark.integration
def test_deepmind_bridges_on_google_and_selects_both() -> None:
    """Regression 11 & 12: bridge on Google; both required claims selected."""

    q = Query(query_id="q", text="Who founded the company that acquired DeepMind?")
    a = _claim("a", "Google acquired DeepMind.", ("Google", "DeepMind"))
    b = _claim("b", "Google was founded by Larry Page.", ("Google", "Larry Page"))
    decision = DeterministicBridgeDetector().detect(q, a, b)
    assert decision.created and decision.bridge_entity == "Google"
    run = build_run([a, b], q, _AlwaysNeutral(), VariantFlags())
    assert set(run.selected_ids) == {"a", "b"}


class _AlwaysNeutral:
    def classify(self, pairs: Sequence[ClaimPair]) -> list[RelationProbabilities]:
        return [
            RelationProbabilities(entailment=0.0, contradiction=0.0, neutral=1.0) for _ in pairs
        ]


@pytest.mark.integration
def test_bridges_do_not_change_belief() -> None:
    """Regression 4."""

    for ex in _BY_CAT["multi_hop"][:5]:
        with_b = run_example(ex, "oracle", VariantFlags(bridges=True))
        no_b = run_example(ex, "oracle", VariantFlags(bridges=False))
        assert with_b.beliefs == no_b.beliefs


@pytest.mark.integration
def test_bridges_create_no_conflicts() -> None:
    """Regression 5."""

    for ex in _BY_CAT["multi_hop"][:5]:
        run = run_example(ex, "oracle", VariantFlags(bridges=True))
        assert activation(run)["bridge_edges"] >= 1
        assert len(run.conflict_sets) == 0


@pytest.mark.integration
def test_bridges_are_not_support_corroboration() -> None:
    """Regression 6: bridged multi-hop claims carry no support edge."""

    run = run_example(_BY_CAT["multi_hop"][0], "oracle", VariantFlags(bridges=True))
    assert activation(run)["support_edges"] == 0
    assert activation(run)["bridge_edges"] >= 1


@pytest.mark.integration
def test_bridges_connect_selected_subgraph_and_hop_coverage() -> None:
    """Regression 7 & 13: required-hop coverage reaches 1.0 with bridges."""

    covered = [
        mechanism_metrics(ex, run_example(ex, "oracle", VariantFlags(bridges=True)))[
            "required_hop_coverage"
        ]
        for ex in _BY_CAT["multi_hop"]
    ]
    assert all(c == 1.0 for c in covered)


@pytest.mark.integration
def test_no_bridge_ablation_zero_bridges_and_lower_coverage() -> None:
    """Regression 21 & acceptance 4: no-bridge → 0 bridges and coverage falls."""

    ex = _BY_CAT["multi_hop"][0]
    no_b = run_example(ex, "oracle", VariantFlags(bridges=False))
    assert activation(no_b)["bridge_edges"] == 0
    assert mechanism_metrics(ex, no_b)["required_hop_coverage"] == 0.0


@pytest.mark.integration
def test_duplicate_clusters_do_not_bridge_internally() -> None:
    """Regression 8."""

    q = Query(query_id="q", text="What profit?")
    a = _claim("a", "Acme posted profit of 5 million.", ("Acme",))
    b = _claim("b", "Acme posted profit of 5 million.", ("Acme",))  # identical
    assert detect_bridges(q, [a, b], []) == []


@pytest.mark.integration
def test_generic_words_and_articles_not_bridges_or_entities() -> None:
    """Regression 9 & 10."""

    assert "The" not in extract_entities("The company grew.")
    q = Query(query_id="q", text="What happened?")
    a = _claim("a", "The company grew.", ("company",))
    b = _claim("b", "The company shrank.", ("company",))
    assert detect_bridges(q, [a, b], []) == []


@pytest.mark.integration
def test_directional_support_not_duplicate() -> None:
    """Regression 14 & acceptance 8."""

    for ex in _BY_CAT["directional_support"][:5]:
        run = run_example(ex, "oracle", VariantFlags())
        assert mechanism_metrics(ex, run)["support_edge_recall"] == 1.0
        assert activation(run)["duplicate_edges"] == 0


@pytest.mark.integration
def test_paraphrases_remain_duplicates() -> None:
    """Regression 15."""

    for ex in _BY_CAT["duplicate"][:5]:
        assert (
            mechanism_metrics(ex, run_example(ex, "oracle", VariantFlags()))[
                "duplicate_cluster_accuracy"
            ]
            == 1.0
        )


@pytest.mark.integration
def test_structural_gate_rejects_unrelated_contradiction() -> None:
    """Regression 16."""

    a = _claim("a", "The deadline is June 30.", (), subject="deadline")
    b = _claim("b", "The manager works in London.", ("London",), subject="manager")
    assert not structural_contradiction_ok(a, b).accepted
    gate = StructuralContradictionGate(_AlwaysContradict())
    [prob] = gate.classify([ClaimPair(source=a, target=b)])
    assert prob.contradiction == 0.0  # demoted to neutral
    assert prob.neutral == pytest.approx(1.0)


@pytest.mark.integration
def test_structural_gate_keeps_true_contradiction() -> None:
    """Regression 17."""

    a = _claim("a", "The company was profitable.", ("Acme",), subject="acme")
    b = _claim("b", "The company was not profitable.", ("Acme",), subject="acme", negation=True)
    assert structural_contradiction_ok(a, b).accepted
    gate = StructuralContradictionGate(_AlwaysContradict())
    [prob] = gate.classify([ClaimPair(source=a, target=b)])
    assert prob.contradiction == 0.95  # kept


@pytest.mark.integration
def test_structural_gate_improves_hard_negative_precision() -> None:
    """Regression 18: gate drops contradictions on hard-negative pairs."""

    base = _AlwaysContradict()
    gate = StructuralContradictionGate(base)
    kept_base = kept_gated = 0
    for ex in _BY_CAT["hard_neutral"]:
        pairs = [ClaimPair(source=ex.claims[0], target=ex.claims[1])]
        kept_base += sum(1 for p in base.classify(pairs) if p.contradiction >= 0.5)
        kept_gated += sum(1 for p in gate.classify(pairs) if p.contradiction >= 0.5)
    assert kept_base > 0 and kept_gated == 0  # all spurious contradictions removed


@pytest.mark.integration
def test_no_propagation_and_no_contradiction_ablations() -> None:
    """Regression 19 & 20."""

    con = _BY_CAT["contradiction"][0]
    assert run_example(con, "oracle", VariantFlags(propagation=False)).propagation_iterations == 0
    assert (
        activation(run_example(con, "oracle", VariantFlags(contradiction=False)))[
            "contradiction_edges"
        ]
        == 0
    )


@pytest.mark.integration
def test_full_egrag_uses_bridge_connectivity() -> None:
    """Regression 23."""

    run = run_example(_BY_CAT["multi_hop"][0], "oracle", VariantFlags())
    assert activation(run)["bridge_edges"] >= 1


@pytest.mark.integration
def test_repeated_runs_deterministic() -> None:
    """Regression 24."""

    ex = _BY_CAT["multi_hop"][0]
    assert run_example(ex, "oracle", VariantFlags()) == run_example(ex, "oracle", VariantFlags())
