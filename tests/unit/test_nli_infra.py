"""Offline NLI-infrastructure tests (no real model required).

Covers label-mapping validation, directionality, precedence, NLI cache keys,
lazy loading, and mocked adapter failures. Real-model behavior is exercised only
under ``requires_local_models`` and skips cleanly when the extra is absent.
"""

from __future__ import annotations

import importlib
import sys
from collections.abc import Sequence

import pytest

from egrag.caching import build_cache_key, build_nli_cache_key
from egrag.domain.models import (
    AtomicClaim,
    ClaimProvenance,
    Query,
    SourceMetadata,
    SourceSpan,
)
from egrag.graph import (
    GraphBuilder,
    NLILabelMappingError,
    classify_directional,
    decide_relation,
    validate_label_mapping,
)
from egrag.graph.nli import LABEL_VALIDATION_CASES
from egrag.graph.types import ClaimPair, RelationProbabilities


def _claim(cid: str, text: str) -> AtomicClaim:
    return AtomicClaim(
        claim_id=cid,
        text=text,
        provenance=ClaimProvenance(
            source=SourceMetadata(source_id=cid),
            spans=(SourceSpan(source_id=cid, start=0, end=len(text), text=text),),
        ),
        extraction_confidence=1.0,
    )


class CorrectLabelNLI:
    """Fake NLI that maps the controlled cases to their correct labels."""

    def classify(self, pairs: Sequence[ClaimPair]) -> list[RelationProbabilities]:
        out = []
        by_hyp = {c.hypothesis: c.expected for c in LABEL_VALIDATION_CASES}
        for p in pairs:
            expected = by_hyp.get(p.target.text, "neutral")
            out.append(
                RelationProbabilities(
                    entailment=0.9 if expected == "entailment" else 0.05,
                    contradiction=0.9 if expected == "contradiction" else 0.05,
                    neutral=0.9 if expected == "neutral" else 0.05,
                )
            )
        return out


class SwappedLabelNLI:
    """Fake NLI with a broken mapping (always predicts neutral)."""

    def classify(self, pairs: Sequence[ClaimPair]) -> list[RelationProbabilities]:
        return [
            RelationProbabilities(entailment=0.1, contradiction=0.1, neutral=0.8) for _ in pairs
        ]


class ScriptedNLI:
    """Returns preset probabilities keyed by (source_id, target_id)."""

    def __init__(self, table: dict[tuple[str, str], RelationProbabilities]) -> None:
        self.table = table
        self.calls: list[tuple[str, str]] = []

    def classify(self, pairs: Sequence[ClaimPair]) -> list[RelationProbabilities]:
        out = []
        for p in pairs:
            self.calls.append((p.source.claim_id, p.target.claim_id))
            out.append(
                self.table.get(
                    (p.source.claim_id, p.target.claim_id),
                    RelationProbabilities(entailment=0.0, contradiction=0.0, neutral=1.0),
                )
            )
        return out


@pytest.mark.unit
def test_label_mapping_validates_correct_classifier() -> None:
    """Tests 1-4: controlled entailment/contradiction/neutral map correctly."""

    assert validate_label_mapping(CorrectLabelNLI()) == []


@pytest.mark.unit
def test_incorrect_label_mapping_fails_validation() -> None:
    """Test 17: a wrong label mapping raises a hard error."""

    with pytest.raises(NLILabelMappingError):
        validate_label_mapping(SwappedLabelNLI())
    issues = validate_label_mapping(SwappedLabelNLI(), raise_on_error=False)
    assert len(issues) >= 2  # entailment and contradiction cases mismapped


@pytest.mark.unit
def test_directionality_preserved_and_batch_order() -> None:
    """Tests 5 & 6: direction chosen from the entailing side; pair order preserved."""

    a, b = _claim("a", "A specific claim."), _claim("b", "A general claim.")
    scripted = ScriptedNLI(
        {
            ("a", "b"): RelationProbabilities(entailment=0.9, contradiction=0.0, neutral=0.1),
            ("b", "a"): RelationProbabilities(entailment=0.2, contradiction=0.0, neutral=0.8),
        }
    )
    decision = classify_directional(scripted, a, b)
    assert decision.relation == "supports"
    assert decision.source_first is True  # a entails b more strongly
    assert decision.entailment_ab == 0.9 and decision.entailment_ba == 0.2
    assert scripted.calls == [("a", "b"), ("b", "a")]  # order preserved


@pytest.mark.unit
def test_precedence_policy() -> None:
    """Test 15: DUPLICATE > SUPPORTS > CONTRADICTS > NEUTRAL."""

    assert decide_relation(0.9, 0.9, 0.0).relation == "duplicate"  # mutual high entailment
    assert decide_relation(0.9, 0.2, 0.0).relation == "supports"  # one-directional entailment
    assert decide_relation(0.1, 0.1, 0.9).relation == "contradicts"
    assert decide_relation(0.1, 0.1, 0.1).relation == "neutral"


@pytest.mark.unit
def test_neutral_creates_no_edge() -> None:
    """Test 18: neutral predictions build no canonical edge."""

    a, b = _claim("a", "Alpha companies grew last year."), _claim("b", "Beta rivers flow north.")
    neutral = ScriptedNLI({})  # everything neutral by default
    graph = GraphBuilder(neutral).build([a, b], query=Query(query_id="q", text="x")).graph
    assert len(graph.edges()) == 0


@pytest.mark.unit
def test_nli_cache_keys() -> None:
    """Tests 13 & 14: keys differ by revision/threshold/truncation; never reuse lexical keys."""

    base: dict[str, object] = {
        "model": "roberta-large-mnli",
        "model_revision": "r1",
        "tokenizer_revision": "r1",
        "max_length": 256,
        "truncation": True,
        "entailment_threshold": 0.5,
        "contradiction_threshold": 0.5,
        "duplicate_threshold": 0.8,
    }
    k = build_nli_cache_key("p", "h", **base)
    assert k == build_nli_cache_key("p", "h", **base)  # stable
    assert k != build_nli_cache_key("p", "h", **{**base, "model_revision": "r2"})
    assert k != build_nli_cache_key("p", "h", **{**base, "entailment_threshold": 0.7})
    assert k != build_nli_cache_key("p", "h", **{**base, "truncation": False})
    assert k != build_nli_cache_key("p", "h", **{**base, "max_length": 128})
    assert k != build_nli_cache_key("h", "p", **base)  # directional
    # never collides with a lexical-classifier cache key
    lexical = build_cache_key(namespace="relation", content="p||h", algorithm="lexical-nli")
    assert k != lexical and k.startswith("nli-")


@pytest.mark.unit
def test_metadata_and_lazy_loading() -> None:
    """Tests 7, 8, 9, 11: metadata present; model not loaded at construction/import."""

    from egrag.graph.classification import HuggingFaceNLIClassifier

    clf = HuggingFaceNLIClassifier("roberta-large-mnli", model_revision="abc123")
    assert clf.classifier_version == "roberta-large-mnli"
    assert clf.model_revision == "abc123"
    assert clf._pipeline is None  # lazy: not loaded on construction
    assert "transformers" not in sys.modules  # importing egrag.graph did not load it


@pytest.mark.unit
def test_mocked_adapter_reports_missing_dependency() -> None:
    """Test 12: a model-load failure surfaces an actionable, typed error."""

    if importlib.util.find_spec("transformers") is not None:
        pytest.skip("transformers is installed; the missing-dependency path is not exercised")
    from egrag.graph.classification import HuggingFaceNLIClassifier

    clf = HuggingFaceNLIClassifier("roberta-large-mnli")
    with pytest.raises(RuntimeError, match="local-models"):
        clf.classify([ClaimPair(source=_claim("a", "x y"), target=_claim("b", "x y"))])


@pytest.mark.unit
def test_deterministic_repeated_decisions() -> None:
    """Test 20: repeated inference over the same fake input matches."""

    a, b = _claim("a", "one"), _claim("b", "two")
    scripted = ScriptedNLI(
        {
            ("a", "b"): RelationProbabilities(entailment=0.6, contradiction=0.0, neutral=0.4),
            ("b", "a"): RelationProbabilities(entailment=0.6, contradiction=0.0, neutral=0.4),
        }
    )
    assert classify_directional(scripted, a, b) == classify_directional(scripted, a, b)


@pytest.mark.requires_local_models
def test_real_nli_label_mapping(monkeypatch: pytest.MonkeyPatch) -> None:
    """Tests 1-4/17 with the real model (gated by EGRAG_RUN_LOCAL_MODELS=1).

    Forces offline mode so it loads from the local cache under the session socket
    block; uses the recorded revision.
    """

    pytest.importorskip("transformers")
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    from egrag.graph.classification import HuggingFaceNLIClassifier

    clf = HuggingFaceNLIClassifier(
        "roberta-large-mnli", model_revision="2a8f12d27941090092df78e4ba6f0928eb5eac98"
    )
    assert validate_label_mapping(clf) == []
