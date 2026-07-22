"""Mechanism-level evaluation for the evidence graph (repairs the zero-edge run).

This module evaluates the graph mechanisms *directly* against gold annotations,
instead of relying on answer-text metrics from a fake generator. It provides:

* gold-annotated synthetic fixtures that explicitly exercise each mechanism
  (support, contradiction, temporal supersession, duplicate-source, multi-hop,
  unresolved conflict, preferred conflict), with enough lexical overlap that the
  baseline lexical classifier can form edges in **end-to-end** mode;
* a deterministic :class:`GoldRelationClassifier` for **oracle** mode, isolating
  downstream reasoning from the lexical classifier's limits;
* mechanism metrics with explicit empty-set/denominator policies;
* an activation preflight that fails edge-requiring runs that produce no edges.

Fixtures carry gold annotations only; final graph outputs are never hard-coded.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from egrag.domain.models import (
    AtomicClaim,
    ClaimProvenance,
    ClaimSemantics,
    Document,
    SourceMetadata,
    SourceSpan,
)
from egrag.graph import LexicalPairClassifier
from egrag.graph.types import ClaimPair, PairClassifier, RelationProbabilities

Category = Literal[
    "support",
    "directional_support",
    "contradiction",
    "temporal",
    "duplicate",
    "multi_hop",
    "unresolved_conflict",
    "preferred_conflict",
    "hard_neutral",
]
GoldRelation = Literal["supports", "contradicts", "supersedes", "none"]


# --- gold annotation models -------------------------------------------------


class _Model(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class GoldPair(_Model):
    """A gold relation between two claims, by index into the example's claims."""

    source: int = Field(ge=0)
    target: int = Field(ge=0)
    relation: GoldRelation


class GoldConflict(_Model):
    """A gold conflict set; ``preferred`` is the index that should win (or None)."""

    claims: tuple[int, ...] = Field(min_length=2)
    outcome: Literal["resolved", "unresolved"]
    preferred: int | None = None


class GoldBridge(_Model):
    """A gold query-conditioned bridge between two claims (shared entity)."""

    source: int = Field(ge=0)
    target: int = Field(ge=0)
    entity: str | None = None


class MechanismExample(_Model):
    """A gold-annotated example. ``claims`` is for oracle mode; ``documents`` for
    end-to-end mode (passages re-extracted by the real extractor)."""

    example_id: str = Field(min_length=1)
    category: Category
    query: str = Field(min_length=1)
    claims: tuple[AtomicClaim, ...] = Field(min_length=1)
    documents: tuple[Document, ...] = ()
    gold_pairs: tuple[GoldPair, ...] = ()
    gold_conflicts: tuple[GoldConflict, ...] = ()
    gold_bridges: tuple[GoldBridge, ...] = ()
    required_claims: tuple[int, ...] = ()
    provenance_groups: tuple[tuple[int, ...], ...] = ()
    expect_connected_subgraph: bool = True

    @model_validator(mode="after")
    def _check_indices(self) -> MechanismExample:
        n = len(self.claims)
        for gp in self.gold_pairs:
            if gp.source >= n or gp.target >= n:
                raise ValueError(f"{self.example_id}: gold pair index out of range")
        for gb in self.gold_bridges:
            if gb.source >= n or gb.target >= n:
                raise ValueError(f"{self.example_id}: gold bridge index out of range")
        for gc in self.gold_conflicts:
            if any(i >= n for i in gc.claims) or (gc.preferred is not None and gc.preferred >= n):
                raise ValueError(f"{self.example_id}: gold conflict index out of range")
        if any(i >= n for i in self.required_claims):
            raise ValueError(f"{self.example_id}: required-claim index out of range")
        return self

    @property
    def requires_edges(self) -> bool:
        return any(gp.relation != "none" for gp in self.gold_pairs)


# --- claim / document construction helpers ----------------------------------


def _claim(
    cid: str,
    text: str,
    source_id: str,
    *,
    subject: str | None = None,
    predicate: str | None = None,
    obj: str | None = None,
    entities: tuple[str, ...] = (),
    negation: bool = False,
    valid_from: datetime | None = None,
    observed_at: datetime | None = None,
) -> AtomicClaim:
    return AtomicClaim(
        claim_id=cid,
        text=text,
        provenance=ClaimProvenance(
            source=SourceMetadata(source_id=source_id),
            spans=(SourceSpan(source_id=source_id, start=0, end=len(text), text=text),),
            observed_at=observed_at,
        ),
        extraction_confidence=0.9,
        valid_from=valid_from,
        semantics=ClaimSemantics(
            subject=subject,
            predicate=predicate,
            object=obj,
            named_entities=entities,
            negation=negation,
        ),
    )


def _doc(source_id: str, text: str) -> Document:
    return Document(document_id=source_id, text=text, source=SourceMetadata(source_id=source_id))


def _y(year: int) -> datetime:
    return datetime(year, 1, 1, tzinfo=UTC)


# --- fixture generators (varied, >=20 each; no trivial copies) ---------------

_ENTITIES = [
    "Acme",
    "Globex",
    "Initech",
    "Umbrella",
    "Soylent",
    "Hooli",
    "Stark Industries",
    "Wayne Enterprises",
    "Wonka",
    "Cyberdyne",
    "Tyrell",
    "Nakatomi",
    "Massive Dynamic",
    "Aperture",
    "Black Mesa",
    "Oscorp",
    "Pied Piper",
    "Vandelay",
    "Gekko",
    "Bluth",
    "Prestige",
    "Duff",
    "Monarch",
    "Vault Tec",
    "Weyland",
]
_CITIES = [
    "Springfield",
    "Rivertown",
    "Lakeview",
    "Fairhaven",
    "Brightport",
    "Ashford",
    "Crestwood",
    "Hollowmere",
    "Stonebridge",
    "Greenfield",
    "Westbury",
    "Eastvale",
    "Northgate",
    "Southport",
    "Kingsford",
    "Queenstown",
    "Oakdale",
    "Maplewood",
    "Cedar Falls",
    "Pinehurst",
    "Elmwood",
    "Bayview",
    "Hilldale",
    "Foxborough",
    "Riverside",
]


def _support_examples(n: int = 22) -> list[MechanismExample]:
    out = []
    for i in range(n):
        ent = _ENTITIES[i % len(_ENTITIES)]
        amt = 3 + i  # vary the value
        # high lexical overlap, two independent sources, optional voice change
        t1 = f"{ent} reported revenue of {amt} billion dollars in 2023."
        t2 = (
            f"Revenue of {amt} billion dollars in 2023 was reported by {ent}."
            if i % 2
            else f"{ent} reported revenue of {amt} billion dollars in 2023 as well."
        )
        c0 = _claim(
            f"sup{i}-a",
            t1,
            f"sup{i}-s1",
            subject=ent,
            predicate="reported revenue",
            obj=f"{amt} billion",
            entities=(ent,),
        )
        c1 = _claim(
            f"sup{i}-b",
            t2,
            f"sup{i}-s2",
            subject=ent,
            predicate="reported revenue",
            obj=f"{amt} billion",
            entities=(ent,),
        )
        out.append(
            MechanismExample(
                example_id=f"support-{i:02d}",
                category="support",
                query=f"What revenue did {ent} report?",
                claims=(c0, c1),
                documents=(_doc(f"sup{i}-s1", t1), _doc(f"sup{i}-s2", t2)),
                gold_pairs=(GoldPair(source=0, target=1, relation="supports"),),
                required_claims=(0, 1),
                expect_connected_subgraph=True,
            )
        )
    return out


def _contradiction_examples(n: int = 22) -> list[MechanismExample]:
    out = []
    for i in range(n):
        ent = _ENTITIES[i % len(_ENTITIES)]
        # same proposition, opposite polarity (shared content + negation)
        t1 = f"The {ent} merger closed on schedule in June 2023."
        t2 = f"The {ent} merger did not close on schedule in June 2023."
        c0 = _claim(
            f"con{i}-a",
            t1,
            f"con{i}-s1",
            subject=f"{ent} merger",
            predicate="closed on schedule",
            obj="June 2023",
            entities=(ent,),
        )
        c1 = _claim(
            f"con{i}-b",
            t2,
            f"con{i}-s2",
            subject=f"{ent} merger",
            predicate="closed on schedule",
            obj="June 2023",
            entities=(ent,),
            negation=True,
        )
        out.append(
            MechanismExample(
                example_id=f"contradiction-{i:02d}",
                category="contradiction",
                query=f"Did the {ent} merger close on schedule?",
                claims=(c0, c1),
                documents=(_doc(f"con{i}-s1", t1), _doc(f"con{i}-s2", t2)),
                gold_pairs=(GoldPair(source=0, target=1, relation="contradicts"),),
                gold_conflicts=(GoldConflict(claims=(0, 1), outcome="unresolved"),),
            )
        )
    return out


def _temporal_examples(n: int = 22) -> list[MechanismExample]:
    out = []
    for i in range(n):
        ent = _ENTITIES[i % len(_ENTITIES)]
        old_city = _CITIES[i % len(_CITIES)]
        new_city = _CITIES[(i + 7) % len(_CITIES)]
        old_year, new_year = 2010 + (i % 5), 2020 + (i % 5)
        t_old = f"In {old_year} {ent} headquarters was located in {old_city}."
        t_new = f"In {new_year} {ent} headquarters was located in {new_city}."
        # an unrelated newer claim that must NOT supersede the others
        t_other = f"In {new_year} {ent} hired a new chief financial officer."
        c_old = _claim(
            f"tmp{i}-old",
            t_old,
            f"tmp{i}-s1",
            subject=f"{ent} headquarters",
            predicate="located in",
            obj=old_city,
            entities=(ent,),
            valid_from=_y(old_year),
        )
        c_new = _claim(
            f"tmp{i}-new",
            t_new,
            f"tmp{i}-s2",
            subject=f"{ent} headquarters",
            predicate="located in",
            obj=new_city,
            entities=(ent,),
            valid_from=_y(new_year),
        )
        c_other = _claim(
            f"tmp{i}-oth",
            t_other,
            f"tmp{i}-s3",
            subject=f"{ent} officer",
            predicate="hired",
            obj="cfo",
            entities=(ent,),
            valid_from=_y(new_year),
        )
        out.append(
            MechanismExample(
                example_id=f"temporal-{i:02d}",
                category="temporal",
                query=f"Where is {ent} headquarters now?",
                claims=(c_old, c_new, c_other),
                gold_pairs=(GoldPair(source=1, target=0, relation="supersedes"),),
            )
        )
    return out


def _duplicate_examples(n: int = 22) -> list[MechanismExample]:
    out = []
    for i in range(n):
        ent = _ENTITIES[i % len(_ENTITIES)]
        amt = 4 + i
        exact = f"{ent} posted profit of {amt} million dollars in 2022."
        # an independent source making the same claim in different words
        indep = f"In 2022 {ent} reported a profit totaling {amt} million dollars."
        # two same-source EXACT repetitions (shared lineage -> lexical duplicate)
        c0 = _claim(
            f"dup{i}-a",
            exact,
            f"dup{i}-srcA",
            subject=ent,
            predicate="posted profit",
            obj=f"{amt} million",
            entities=(ent,),
        )
        c1 = _claim(
            f"dup{i}-b",
            exact,
            f"dup{i}-srcA",
            subject=ent,
            predicate="posted profit",
            obj=f"{amt} million",
            entities=(ent,),
        )
        c2 = _claim(
            f"dup{i}-c",
            indep,
            f"dup{i}-srcB",
            subject=ent,
            predicate="posted profit",
            obj=f"{amt} million",
            entities=(ent,),
        )
        out.append(
            MechanismExample(
                example_id=f"duplicate-{i:02d}",
                category="duplicate",
                query=f"What profit did {ent} post in 2022?",
                claims=(c0, c1, c2),
                gold_pairs=(GoldPair(source=0, target=2, relation="supports"),),
                provenance_groups=((0, 1),),  # same-source lineage cluster
            )
        )
    return out


_FOUNDERS = [
    "Ada Lovelace",
    "Alan Turing",
    "Grace Hopper",
    "John Mccarthy",
    "Marvin Minsky",
    "Barbara Liskov",
    "Donald Knuth",
    "Edsger Dijkstra",
    "Ken Thompson",
    "Dennis Ritchie",
]


def _multi_hop_examples(n: int = 30) -> list[MechanismExample]:
    """Complementary multi-hop claims linked by a shared entity (a BRIDGE, not
    entailment). Varies the bridge position, voice, and answer type."""

    out = []
    for i in range(n):
        ent = _ENTITIES[i % len(_ENTITIES)]
        person = _FOUNDERS[i % len(_FOUNDERS)]
        city = _CITIES[i % len(_CITIES)]
        # vary voice / answer type across examples
        if i % 3 == 0:
            t1 = f"{ent} chief executive is {person}."
            t2 = f"{person} was born in {city}."
        elif i % 3 == 1:
            t1 = f"The chief executive of {ent} is {person}."
            t2 = f"{city} is the birthplace of {person}."
        else:
            t1 = f"{person} leads {ent} as chief executive."
            t2 = f"{person} grew up in {city}."
        distract = f"{ent} opened a small cafeteria in 2023."
        c0 = _claim(
            f"mh{i}-a",
            t1,
            f"mh{i}-s1",
            subject=ent,
            predicate="chief executive",
            obj=person,
            entities=(ent, person),
        )
        c1 = _claim(
            f"mh{i}-b",
            t2,
            f"mh{i}-s2",
            subject=person,
            predicate="born in",
            obj=city,
            entities=(person, city),
        )
        c2 = _claim(
            f"mh{i}-d",
            distract,
            f"mh{i}-s3",
            subject=ent,
            predicate="opened",
            obj="cafeteria",
            entities=(ent,),
        )
        out.append(
            MechanismExample(
                example_id=f"multihop-{i:02d}",
                category="multi_hop",
                query=f"Where was the chief executive of {ent} born?",
                claims=(c0, c1, c2),
                # the two required claims do NOT entail each other; they BRIDGE on
                # the shared person entity.
                gold_bridges=(GoldBridge(source=0, target=1, entity=person),),
                required_claims=(0, 1),
                expect_connected_subgraph=True,
            )
        )
    return out


def _directional_support_examples(n: int = 20) -> list[MechanismExample]:
    """Asymmetric entailment (A entails B) — SUPPORT, not duplicate."""

    out = []
    specifics = [
        (
            "reported revenue of {v} million euros in 2025",
            "reported revenue above 10 million euros in 2025",
        ),
        ("vote passed by 70 votes to 30", "vote passed with a majority"),
        ("hired {v} new engineers in 2024", "hired new staff in 2024"),
        ("grew by {v} percent in 2023", "grew in 2023"),
    ]
    for i in range(n):
        ent = _ENTITIES[i % len(_ENTITIES)]
        spec, gen = specifics[i % len(specifics)]
        v = 12 + i
        t1 = f"The {ent} {spec.format(v=v)}."
        t2 = f"The {ent} {gen}."
        c0 = _claim(
            f"dsup{i}-a",
            t1,
            f"dsup{i}-s1",
            subject=ent,
            predicate="reported",
            obj=f"specific-{v}",
            entities=(ent,),
        )
        c1 = _claim(
            f"dsup{i}-b",
            t2,
            f"dsup{i}-s2",
            subject=ent,
            predicate="reported",
            obj="general",
            entities=(ent,),
        )
        out.append(
            MechanismExample(
                example_id=f"directional-{i:02d}",
                category="directional_support",
                query=f"What did {ent} report?",
                claims=(c0, c1),
                gold_pairs=(GoldPair(source=0, target=1, relation="supports"),),
            )
        )
    return out


def _hard_neutral_examples(n: int = 20) -> list[MechanismExample]:
    """Unrelated claim pairs that should NOT be contradictions (hard negatives)."""

    out = []
    pairs = [
        ("The deadline is June 30.", "The project manager works in {city}."),
        ("{ent} acquired a startup.", "{ent2} was founded in {city}."),
        ("The report was published on Monday.", "The office is located in {city}."),
        ("{ent} hired a new chief financial officer.", "The cafeteria menu changed in {city}."),
    ]
    for i in range(n):
        ent = _ENTITIES[i % len(_ENTITIES)]
        ent2 = _ENTITIES[(i + 5) % len(_ENTITIES)]
        city = _CITIES[i % len(_CITIES)]
        ta, tb = pairs[i % len(pairs)]
        t1 = ta.format(ent=ent, ent2=ent2, city=city)
        t2 = tb.format(ent=ent, ent2=ent2, city=city)
        c0 = _claim(f"hn{i}-a", t1, f"hn{i}-s1", subject=f"subjA{i}", predicate="predA", obj="x")
        c1 = _claim(f"hn{i}-b", t2, f"hn{i}-s2", subject=f"subjB{i}", predicate="predB", obj="y")
        out.append(
            MechanismExample(
                example_id=f"hardneutral-{i:02d}",
                category="hard_neutral",
                query=f"What happened at {ent}?",
                claims=(c0, c1),
                # no gold edges: these must NOT produce a contradiction
            )
        )
    return out


def _unresolved_conflict_examples(n: int = 22) -> list[MechanismExample]:
    # symmetric evidence -> conflict should remain UNRESOLVED
    out = []
    for i in range(n):
        ent = _ENTITIES[i % len(_ENTITIES)]
        t1 = f"The {ent} product launch succeeded in 2023."
        t2 = f"The {ent} product launch did not succeed in 2023."
        c0 = _claim(
            f"unr{i}-a",
            t1,
            f"unr{i}-s1",
            subject=f"{ent} launch",
            predicate="succeeded",
            obj="2023",
            entities=(ent,),
        )
        c1 = _claim(
            f"unr{i}-b",
            t2,
            f"unr{i}-s2",
            subject=f"{ent} launch",
            predicate="succeeded",
            obj="2023",
            entities=(ent,),
            negation=True,
        )
        out.append(
            MechanismExample(
                example_id=f"unresolved-{i:02d}",
                category="unresolved_conflict",
                query=f"Did the {ent} product launch succeed?",
                claims=(c0, c1),
                gold_pairs=(GoldPair(source=0, target=1, relation="contradicts"),),
                gold_conflicts=(GoldConflict(claims=(0, 1), outcome="unresolved"),),
            )
        )
    return out


def _preferred_conflict_examples(n: int = 22) -> list[MechanismExample]:
    # asymmetric: the supported side has an extra corroborating source
    out = []
    for i in range(n):
        ent = _ENTITIES[i % len(_ENTITIES)]
        t1 = f"The {ent} factory passed inspection in 2023."
        t2 = f"The {ent} factory did not pass inspection in 2023."
        # independent corroboration of side 0, paraphrased so it is a SUPPORT
        # edge (not a lexical duplicate of c0)
        t3 = f"Inspectors confirmed the {ent} factory passed inspection during 2023."
        c0 = _claim(
            f"pre{i}-a",
            t1,
            f"pre{i}-s1",
            subject=f"{ent} factory",
            predicate="passed inspection",
            obj="2023",
            entities=(ent,),
        )
        c1 = _claim(
            f"pre{i}-b",
            t2,
            f"pre{i}-s2",
            subject=f"{ent} factory",
            predicate="passed inspection",
            obj="2023",
            entities=(ent,),
            negation=True,
        )
        c2 = _claim(
            f"pre{i}-c",
            t3,
            f"pre{i}-s3",
            subject=f"{ent} factory",
            predicate="passed inspection",
            obj="2023",
            entities=(ent,),
        )
        out.append(
            MechanismExample(
                example_id=f"preferred-{i:02d}",
                category="preferred_conflict",
                query=f"Did the {ent} factory pass inspection?",
                claims=(c0, c1, c2),
                gold_pairs=(
                    GoldPair(source=0, target=1, relation="contradicts"),
                    GoldPair(source=0, target=2, relation="supports"),
                ),
                gold_conflicts=(GoldConflict(claims=(0, 1), outcome="resolved", preferred=0),),
            )
        )
    return out


def build_suite() -> list[MechanismExample]:
    """Build the full gold-annotated mechanism suite (validated on construction)."""

    suite = (
        _support_examples()
        + _directional_support_examples()
        + _contradiction_examples()
        + _temporal_examples()
        + _duplicate_examples()
        + _multi_hop_examples()
        + _unresolved_conflict_examples()
        + _preferred_conflict_examples()
        + _hard_neutral_examples()
    )
    ids = [ex.example_id for ex in suite]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate mechanism example ids")
    return suite


# --- oracle relation classifier ---------------------------------------------


class GoldRelationClassifier:
    """Deterministic oracle: returns gold relations for annotated pairs.

    Supersession is left to the temporal resolver (driven by timestamps), so this
    classifier reports support/contradiction/neutral only. Implements the
    ``PairClassifier`` protocol.
    """

    classifier_id = "gold-oracle"
    classifier_version = "1.0.0"
    model_revision: str | None = None

    def __init__(self, example: MechanismExample) -> None:
        ids = [c.claim_id for c in example.claims]
        self._rel: dict[frozenset[str], str] = {}
        for gp in example.gold_pairs:
            self._rel[frozenset({ids[gp.source], ids[gp.target]})] = gp.relation

    def classify(self, pairs: Sequence[ClaimPair]) -> list[RelationProbabilities]:
        out = []
        for pair in pairs:
            relation = self._rel.get(
                frozenset({pair.source.claim_id, pair.target.claim_id}), "none"
            )
            if relation == "supports":
                # 0.7 is above the support (entailment) threshold (0.5) but below
                # the duplicate threshold (0.8), so it stores a SUPPORT edge rather
                # than being treated as a semantic duplicate.
                out.append(RelationProbabilities(entailment=0.7, contradiction=0.0, neutral=0.3))
            elif relation == "contradicts":
                # 0.95 is above the contradiction threshold (0.5) but below 1.0,
                # so the no-contradiction ablation (threshold 1.0) suppresses it,
                # matching the lexical classifier's sub-1.0 behavior.
                out.append(RelationProbabilities(entailment=0.0, contradiction=0.95, neutral=0.05))
            else:
                out.append(RelationProbabilities(entailment=0.0, contradiction=0.0, neutral=1.0))
        return out


def make_classifier(
    mode: Literal["oracle", "end_to_end"], example: MechanismExample
) -> PairClassifier:
    return GoldRelationClassifier(example) if mode == "oracle" else LexicalPairClassifier()


__all__ = [
    "Category",
    "GoldBridge",
    "GoldConflict",
    "GoldPair",
    "GoldRelation",
    "GoldRelationClassifier",
    "MechanismExample",
    "_claim",
    "_contradiction_examples",
    "_doc",
    "_duplicate_examples",
    "_multi_hop_examples",
    "_preferred_conflict_examples",
    "_support_examples",
    "_temporal_examples",
    "_unresolved_conflict_examples",
    "_y",
    "build_suite",
    "make_classifier",
]
