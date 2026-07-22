"""Regression tests for the zero-edge repair (mechanism-level, oracle mode).

These prove each graph mechanism is exercised and that ablations, serialization,
aggregation, preflight, and determinism behave correctly. All deterministic,
offline, synthetic.
"""

from __future__ import annotations

import sys

import pytest

from egrag.experiments.mechanism_eval import (
    ActivationError,
    MechanismRun,
    VariantFlags,
    activation,
    activation_preflight,
    mechanism_metrics,
    run_example,
)
from egrag.experiments.mechanisms import build_suite

_SUITE = build_suite()
_BY_CAT: dict[str, list] = {}
for _ex in _SUITE:
    _BY_CAT.setdefault(_ex.category, []).append(_ex)
_FLAGS = VariantFlags()


def _first(cat: str):
    return _BY_CAT[cat][0]


def _ids(ex):
    return [c.claim_id for c in ex.claims]


@pytest.mark.integration
def test_support_fixtures_produce_support_edges() -> None:
    """Regression 1."""

    for ex in _BY_CAT["support"]:
        act = activation(run_example(ex, "oracle", _FLAGS))
        assert act["support_edges"] >= 1, ex.example_id


@pytest.mark.integration
def test_contradiction_fixtures_produce_contradiction_edges() -> None:
    """Regression 2."""

    for ex in _BY_CAT["contradiction"]:
        assert activation(run_example(ex, "oracle", _FLAGS))["contradiction_edges"] >= 1


@pytest.mark.integration
def test_temporal_fixtures_produce_supersession_edges() -> None:
    """Regression 3."""

    for ex in _BY_CAT["temporal"]:
        assert activation(run_example(ex, "oracle", _FLAGS))["supersession_edges"] >= 1


@pytest.mark.integration
def test_unrelated_newer_claim_does_not_supersede() -> None:
    """Regression 4: the distractor newer claim must not supersede anything."""

    for ex in _BY_CAT["temporal"]:
        run = run_example(ex, "oracle", _FLAGS)
        other_id = _ids(ex)[2]  # the "hired CFO" distractor
        assert all(other_id not in (s, t) for (rel, s, t) in run.edges if rel == "supersedes")


@pytest.mark.integration
def test_multi_hop_retains_required_claims_and_hops() -> None:
    """Regression 5 & 6."""

    for ex in _BY_CAT["multi_hop"]:
        m = mechanism_metrics(ex, run_example(ex, "oracle", _FLAGS))
        assert m["required_claim_recall"] == 1.0
        assert m["required_hop_coverage"] == 1.0


@pytest.mark.integration
def test_same_source_duplicates_clustered_and_independent_preserved() -> None:
    """Regression 7 & 8."""

    for ex in _BY_CAT["duplicate"]:
        run = run_example(ex, "oracle", _FLAGS)
        assert mechanism_metrics(ex, run)["duplicate_cluster_accuracy"] == 1.0
        # the independent source (claim index 2) is preserved as a node
        assert (
            _ids(ex)[2] in {s for _r, s, _t in run.edges} | {t for _r, _s, t in run.edges}
            or run.num_nodes >= 3
        )


@pytest.mark.integration
def test_unresolved_conflicts_remain_unresolved() -> None:
    """Regression 9."""

    for ex in _BY_CAT["unresolved_conflict"]:
        run = run_example(ex, "oracle", _FLAGS)
        assert run.conflict_sets and all(cs[1] == "unresolved" for cs in run.conflict_sets)


@pytest.mark.integration
def test_preferred_conflicts_select_expected_claim() -> None:
    """Regression 10."""

    for ex in _BY_CAT["preferred_conflict"]:
        run = run_example(ex, "oracle", _FLAGS)
        assert mechanism_metrics(ex, run)["conflict_resolution_accuracy"] == 1.0


@pytest.mark.integration
def test_neutral_predictions_create_no_edge() -> None:
    """Regression 11: a gold-"none" pair yields no canonical edge.

    The temporal distractor (officer-hired) shares no relation with the
    headquarters claims, so no support/contradiction edge connects them.
    """

    ex = _first("temporal")
    run = run_example(ex, "oracle", _FLAGS)
    other = _ids(ex)[2]
    assert all(
        other not in (s, t) for (rel, s, t) in run.edges if rel in {"supports", "contradicts"}
    )


@pytest.mark.integration
def test_candidate_pruning_retains_gold_positive_pairs() -> None:
    """Regression 12."""

    for ex in _SUITE:
        m = mechanism_metrics(ex, run_example(ex, "oracle", _FLAGS))
        if m["candidate_pair_recall"] is not None:
            assert m["candidate_pair_recall"] == 1.0, ex.example_id


@pytest.mark.integration
def test_serialization_preserves_edges_and_provenance() -> None:
    """Regression 13: building + serializing the package keeps relations/provenance."""

    from egrag.domain.models import Query
    from egrag.experiments.mechanism_eval import build_run
    from egrag.experiments.mechanisms import GoldRelationClassifier
    from egrag.generation import build_evidence_package as build_pkg
    from egrag.graph import GraphBuilder
    from egrag.reasoning import (
        BaselineInitialScorer,
        CharacterTokenCounter,
        ConflictSetResolver,
        GreedyConnectedSelector,
        MetadataReliability,
        SignedBeliefPropagator,
        TokenBudget,
    )
    from egrag.serialization import JsonEvidenceSerializer

    ex = _first("contradiction")
    query = Query(query_id=ex.example_id, text=ex.query)
    graph = GraphBuilder(GoldRelationClassifier(ex)).build(list(ex.claims), query=query).graph
    board = BaselineInitialScorer(MetadataReliability()).score(graph, query)
    prop = SignedBeliefPropagator()
    board = prop.apply(board, prop.propagate(graph, board))
    conflicts = ConflictSetResolver().resolve(graph, board)
    tc = CharacterTokenCounter()
    budget = TokenBudget(total=256, reserved_output=64)
    sel = GreedyConnectedSelector().select(
        graph, query, board, token_budget=budget, token_counter=tc, conflicts=conflicts
    )
    pkg = build_pkg(query, graph, board, conflicts, sel, budget=budget, token_counter=tc)
    assert pkg.relations  # at least the contradiction relation
    serializer = JsonEvidenceSerializer()
    restored = serializer.deserialize(serializer.serialize(pkg))
    assert restored.relations == pkg.relations
    assert all(c.provenance.spans for c in restored.claims)

    # Regression 14: aggregate edge count matches the serialized graph edge count.
    run = build_run(list(ex.claims), query, GoldRelationClassifier(ex), _FLAGS)
    assert len(run.edges) == len(graph.edges())


@pytest.mark.integration
def test_ablations_disable_intended_component() -> None:
    """Regression 15."""

    con = _first("contradiction")
    tmp = _first("temporal")
    sup = _first("support")
    # no-propagation -> zero iterations
    assert run_example(sup, "oracle", VariantFlags(propagation=False)).propagation_iterations == 0
    # propagation on -> non-empty graph runs at least one iteration
    assert run_example(sup, "oracle", VariantFlags(propagation=True)).propagation_iterations >= 1
    # no-contradiction -> zero contradiction edges
    assert (
        activation(run_example(con, "oracle", VariantFlags(contradiction=False)))[
            "contradiction_edges"
        ]
        == 0
    )
    assert (
        activation(run_example(con, "oracle", VariantFlags(contradiction=True)))[
            "contradiction_edges"
        ]
        >= 1
    )
    # no-temporal -> zero supersession edges
    assert (
        activation(run_example(tmp, "oracle", VariantFlags(temporal=False)))["supersession_edges"]
        == 0
    )
    assert (
        activation(run_example(tmp, "oracle", VariantFlags(temporal=True)))["supersession_edges"]
        >= 1
    )


@pytest.mark.integration
def test_preflight_rejects_zero_edge_mechanism_runs() -> None:
    """Regression 16."""

    ex = _first("support")  # requires edges
    empty = MechanismRun(
        edges=[],
        selected_ids=(),
        propagation_iterations=0,
        conflict_sets=[],
        num_nodes=2,
        duplicate_groups=(),
    )
    with pytest.raises(ActivationError):
        activation_preflight([(ex, empty)])
    # a real run passes preflight
    activation_preflight([(ex, run_example(ex, "oracle", _FLAGS))])


@pytest.mark.integration
def test_oracle_and_end_to_end_modes_separate() -> None:
    """Regression 17: the two modes use different classifiers and are not mixed."""

    from egrag.experiments.mechanisms import GoldRelationClassifier, make_classifier

    ex = _first("contradiction")
    assert isinstance(make_classifier("oracle", ex), GoldRelationClassifier)
    assert not isinstance(make_classifier("end_to_end", ex), GoldRelationClassifier)
    # end-to-end recovers the contradiction edge from raw text
    assert activation(run_example(ex, "end_to_end", _FLAGS))["contradiction_edges"] >= 1


@pytest.mark.integration
def test_repeated_runs_are_deterministic() -> None:
    """Regression 18."""

    ex = _first("preferred_conflict")
    a = run_example(ex, "oracle", _FLAGS)
    b = run_example(ex, "oracle", _FLAGS)
    assert a == b


@pytest.mark.integration
def test_no_optional_libs_imported() -> None:
    """Regression 19: no network / model libraries are pulled in.

    Optional libs are removed from ``sys.modules`` first (and restored after) so
    the check reflects what *this* run imports, not what other tests in the
    session already loaded (e.g. ``requires_dense``/``requires_local_models``).
    """

    optional = {"torch", "transformers", "httpx", "sentence_transformers", "networkx", "numpy"}
    saved = {name: sys.modules[name] for name in optional if name in sys.modules}
    for name in optional:
        sys.modules.pop(name, None)
    try:
        run_example(_first("support"), "oracle", _FLAGS)
        leaked = optional & set(sys.modules)
        assert not leaked, f"mechanism run pulled in optional libraries: {sorted(leaked)}"
    finally:
        sys.modules.update(saved)


@pytest.mark.integration
def test_entity_normalization_excludes_leading_function_words() -> None:
    """Regression (secondary defect): named entities are not sentence-initial words."""

    from egrag.adapters.extraction.baseline import _entities

    ents = _entities("The company announced record revenue growth in 2023.")
    assert "The" not in ents
    assert _entities("The Acme Corporation grew.")[0] == "Acme Corporation"
