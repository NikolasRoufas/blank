"""Build evidence graphs for mechanism examples and score them against gold.

Two modes are kept strictly separate:

* **oracle** — uses the hand-built gold claims and the :class:`GoldRelationClassifier`,
  isolating downstream reasoning (propagation, conflict, temporal, selection)
  from upstream extraction/classification error;
* **end_to_end** — re-extracts claims from the example's passages and uses the
  real lexical classifier, testing whether the pipeline recovers structure.

Mechanism metrics use explicit empty-set policies: precision is 1.0 when nothing
is predicted, recall is 1.0 when there is no gold of that type; a metric that does
not apply to an example is ``None`` and is excluded from aggregation.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

from egrag.domain.models import (
    AtomicClaim,
    EvidenceGraphSnapshot,
    Query,
    RelationType,
)
from egrag.experiments.mechanisms import GoldRelationClassifier, MechanismExample
from egrag.generation import build_evidence_package
from egrag.graph import (
    ClassificationConfig,
    EvidenceGraph,
    GraphBuilder,
    LexicalPairClassifier,
    TemporalConfig,
    detect_bridges,
)
from egrag.graph.candidates import generate_candidates
from egrag.graph.types import CandidateConfig, PairClassifier
from egrag.reasoning import (
    BaselineInitialScorer,
    CharacterTokenCounter,
    ConflictSetResolver,
    GreedyConnectedSelector,
    MetadataReliability,
    NoPropagationBaseline,
    SignedBeliefPropagator,
    TokenBudget,
    TopClaimsSelector,
)

_NO_CONTRADICTION = ClassificationConfig(contradiction_threshold=1.0)
_REL_NAME = {
    RelationType.SUPPORT: "supports",
    RelationType.CONTRADICTION: "contradicts",
    RelationType.SUPERSESSION: "supersedes",
    RelationType.DUPLICATE: "duplicate",
    RelationType.DEPENDENCY: "depends",
    RelationType.BRIDGES: "bridges",
}


@dataclass(frozen=True)
class MechanismRun:
    edges: list[tuple[str, str, str]]  # (relation_name, source_id, target_id)
    selected_ids: tuple[str, ...]
    propagation_iterations: int
    conflict_sets: list[tuple[tuple[str, ...], str, str | None]]  # (claim_ids, outcome, preferred)
    num_nodes: int
    duplicate_groups: tuple[tuple[str, ...], ...]
    candidate_pairs: set[frozenset[str]] = field(default_factory=set)
    bridge_pairs: dict[frozenset[str], str] = field(default_factory=dict)  # pair -> bridge_entity
    beliefs: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class VariantFlags:
    propagation: bool = True
    temporal: bool = True
    contradiction: bool = True
    selection: str = "greedy"  # "greedy" | "top"
    bridges: bool = True


def build_run(
    claims: list[AtomicClaim],
    query: Query,
    classifier: PairClassifier,
    flags: VariantFlags,
    *,
    evidence_budget: int = 256,
    reserved: int = 64,
    classification_config: ClassificationConfig | None = None,
) -> MechanismRun:
    # Threshold precedence: the no-contradiction ablation always wins (it must
    # suppress contradiction edges); otherwise an explicit (e.g. dev-frozen)
    # config is used, falling back to the builder default.
    if not flags.contradiction:
        cls_cfg: ClassificationConfig | None = (
            _NO_CONTRADICTION
            if classification_config is None
            else classification_config.model_copy(update={"contradiction_threshold": 1.0})
        )
    else:
        cls_cfg = classification_config
    builder = GraphBuilder(
        classifier,
        classification_config=cls_cfg,
        temporal_config=TemporalConfig(enabled=flags.temporal),
    )
    base = builder.build(claims, query=query).graph
    # Query-conditioned bridge edges are added AFTER evidential construction. They
    # provide reasoning connectivity only: propagation and conflict detection
    # ignore BRIDGES, while the selector's graph.neighbors() uses them.
    bridge_edges = detect_bridges(query, claims, base.edges()) if flags.bridges else []
    if bridge_edges:
        snap = base.snapshot()
        graph = EvidenceGraph(
            EvidenceGraphSnapshot(
                snapshot_id=snap.snapshot_id,
                claims=snap.claims,
                relations=tuple(snap.relations) + tuple(bridge_edges),
            )
        )
    else:
        graph = base
    board = BaselineInitialScorer(MetadataReliability()).score(graph, query)
    propagator = SignedBeliefPropagator() if flags.propagation else NoPropagationBaseline()
    result = propagator.propagate(graph, board)
    board = propagator.apply(board, result)
    conflicts = ConflictSetResolver().resolve(graph, board)
    token_counter = CharacterTokenCounter()
    budget = TokenBudget(total=evidence_budget, reserved_output=reserved)
    selector = TopClaimsSelector() if flags.selection == "top" else GreedyConnectedSelector()
    selection = selector.select(
        graph, query, board, token_budget=budget, token_counter=token_counter, conflicts=conflicts
    )
    package = build_evidence_package(
        query, graph, board, conflicts, selection, budget=budget, token_counter=token_counter
    )
    edges = [
        (_REL_NAME.get(r.relation_type, str(r.relation_type)), r.source_claim_id, r.target_claim_id)
        for r in graph.edges()
    ]
    cands = {
        frozenset({p.source.claim_id, p.target.claim_id})
        for p in generate_candidates(claims, CandidateConfig()).pairs
    }
    return MechanismRun(
        edges=edges,
        selected_ids=tuple(s.claim_id for s in package.selected),
        propagation_iterations=result.iterations,
        conflict_sets=[
            (tuple(c.claim_ids), str(c.outcome) if c.outcome else "none", c.preferred_claim_id)
            for c in conflicts
        ],
        num_nodes=len(graph.nodes()),
        duplicate_groups=package.duplicate_groups,
        candidate_pairs=cands,
        bridge_pairs={
            frozenset({r.source_claim_id, r.target_claim_id}): (
                (r.bridge.bridge_entity or "") if r.bridge else ""
            )
            for r in bridge_edges
        },
        beliefs={cid: round(v, 6) for cid, v in result.beliefs.items()},
    )


# --- gold-vs-prediction metrics ---------------------------------------------


def _prf(predicted: set[object], gold: set[object]) -> tuple[float, float]:
    precision = 1.0 if not predicted else len(predicted & gold) / len(predicted)
    recall = 1.0 if not gold else len(predicted & gold) / len(gold)
    return precision, recall


def _gold_edges(example: MechanismExample, relation: str) -> set[object]:
    ids = [c.claim_id for c in example.claims]
    if relation == "supersedes":
        return {
            (ids[gp.source], ids[gp.target]) for gp in example.gold_pairs if gp.relation == relation
        }
    return {
        frozenset({ids[gp.source], ids[gp.target]})
        for gp in example.gold_pairs
        if gp.relation == relation
    }


def _pred_edges(run: MechanismRun, relation: str) -> set[object]:
    if relation == "supersedes":
        return {(s, t) for (rel, s, t) in run.edges if rel == relation}
    return {frozenset({s, t}) for (rel, s, t) in run.edges if rel == relation}


def _connected(selected: tuple[str, ...], run: MechanismRun) -> bool:
    if len(selected) <= 1:
        return True
    adj: dict[str, set[str]] = {s: set() for s in selected}
    sel = set(selected)
    for _rel, s, t in run.edges:
        if s in sel and t in sel:
            adj[s].add(t)
            adj[t].add(s)
    seen = {selected[0]}
    queue = deque([selected[0]])
    while queue:
        node = queue.popleft()
        for nb in adj[node]:
            if nb not in seen:
                seen.add(nb)
                queue.append(nb)
    return seen == sel


def mechanism_metrics(example: MechanismExample, run: MechanismRun) -> dict[str, float | None]:
    ids = [c.claim_id for c in example.claims]
    m: dict[str, float | None] = {}

    for relation in ("supports", "contradicts", "supersedes"):
        gold = _gold_edges(example, relation)
        pred = _pred_edges(run, relation)
        prefix = {
            "supports": "support_edge",
            "contradicts": "contradiction_edge",
            "supersedes": "supersession",
        }[relation]
        if not gold and not pred:
            m[f"{prefix}_precision"] = None
            m[f"{prefix}_recall"] = None
        else:
            p, r = _prf(pred, gold)
            m[f"{prefix}_precision"] = p
            m[f"{prefix}_recall"] = r

    # candidate-pair recall over gold positive pairs
    gold_pos = {
        frozenset({ids[gp.source], ids[gp.target]})
        for gp in example.gold_pairs
        if gp.relation != "none"
    }
    m["candidate_pair_recall"] = (
        None if not gold_pos else len(gold_pos & run.candidate_pairs) / len(gold_pos)
    )

    # conflict metrics
    if example.gold_conflicts:
        recalled = 0
        res_ok = res_total = 0
        unr_ok = unr_total = 0
        for gc in example.gold_conflicts:
            want = {ids[i] for i in gc.claims}
            match = next((cs for cs in run.conflict_sets if want <= set(cs[0])), None)
            if match is not None:
                recalled += 1
            if gc.outcome == "resolved":
                res_total += 1
                if match is not None and gc.preferred is not None and match[2] == ids[gc.preferred]:
                    res_ok += 1
            else:
                unr_total += 1
                if match is not None and match[1] == "unresolved":
                    unr_ok += 1
        m["conflict_set_recall"] = recalled / len(example.gold_conflicts)
        m["conflict_resolution_accuracy"] = (res_ok / res_total) if res_total else None
        m["unresolved_conflict_accuracy"] = (unr_ok / unr_total) if unr_total else None
    else:
        m["conflict_set_recall"] = None
        m["conflict_resolution_accuracy"] = None
        m["unresolved_conflict_accuracy"] = None

    # required-claim / hop coverage / connectivity
    if example.required_claims:
        req = {ids[i] for i in example.required_claims}
        sel = set(run.selected_ids)
        m["required_claim_recall"] = len(req & sel) / len(req)
        m["required_hop_coverage"] = 1.0 if req <= sel else 0.0
    else:
        m["required_claim_recall"] = None
        m["required_hop_coverage"] = None
    m["selected_subgraph_connectivity"] = (
        (1.0 if _connected(run.selected_ids, run) else 0.0)
        if example.expect_connected_subgraph
        else None
    )

    # duplicate handling
    if example.provenance_groups:
        ok = 0
        for group in example.provenance_groups:
            want = {ids[i] for i in group}
            ok += any(want <= set(g) for g in run.duplicate_groups)
        m["duplicate_cluster_accuracy"] = ok / len(example.provenance_groups)
    else:
        m["duplicate_cluster_accuracy"] = None

    # bridge (reasoning-connectivity) metrics
    gold_bridges = {
        frozenset({ids[gb.source], ids[gb.target]}): gb.entity for gb in example.gold_bridges
    }
    pred_bridges = run.bridge_pairs
    if gold_bridges or pred_bridges:
        p, r = _prf(set(pred_bridges), set(gold_bridges))
        m["bridge_precision"] = p
        m["bridge_recall"] = r
        if gold_bridges:
            correct_entity = sum(
                1
                for k, ent in gold_bridges.items()
                if k in pred_bridges and (not ent or pred_bridges[k] == ent)
            )
            m["bridge_entity_accuracy"] = correct_entity / len(gold_bridges)
        else:
            m["bridge_entity_accuracy"] = None
    else:
        m["bridge_precision"] = None
        m["bridge_recall"] = None
        m["bridge_entity_accuracy"] = None
    return m


def activation(run: MechanismRun) -> dict[str, int]:
    by_type: dict[str, int] = {}
    for rel, _s, _t in run.edges:
        by_type[rel] = by_type.get(rel, 0) + 1
    return {
        "support_edges": by_type.get("supports", 0),
        "contradiction_edges": by_type.get("contradicts", 0),
        "supersession_edges": by_type.get("supersedes", 0),
        "duplicate_edges": by_type.get("duplicate", 0),
        "bridge_edges": by_type.get("bridges", 0),
        "propagation_iterations": run.propagation_iterations,
        "conflict_sets": len(run.conflict_sets),
        "connected_selected": int(_connected(run.selected_ids, run)),
        "selected_claims": len(run.selected_ids),
    }


def run_example(example: MechanismExample, mode: str, flags: VariantFlags) -> MechanismRun:
    query = Query(query_id=example.example_id, text=example.query)
    if mode == "oracle":
        return build_run(list(example.claims), query, GoldRelationClassifier(example), flags)
    # end-to-end: re-extract from passages with the real lexical classifier
    from egrag.adapters.extraction import SentenceClaimExtractor
    from egrag.adapters.retrieval import BM25Retriever, SentenceAwareChunker, prepare_passages

    passages = prepare_passages(example.documents, SentenceAwareChunker(chunk_size=512, overlap=64))
    hits = BM25Retriever(passages).retrieve(query, max(3, len(passages)))
    sources = {d.source.source_id: d.source for d in example.documents}
    extractor = SentenceClaimExtractor()
    claims: list[AtomicClaim] = []
    for p in hits:
        claims.extend(extractor.extract(p, query=query, source=sources.get(p.span.source_id)))
    return build_run(claims, query, LexicalPairClassifier(), flags)


class ActivationError(RuntimeError):
    """Raised when an edge-requiring mechanism run produced zero edges."""


def activation_preflight(examples_runs: list[tuple[MechanismExample, MechanismRun]]) -> None:
    """Fail before accepting aggregates if edge-requiring examples have no edges."""

    offenders = [ex.example_id for ex, run in examples_runs if ex.requires_edges and not run.edges]
    if offenders:
        raise ActivationError(
            f"{len(offenders)} edge-requiring examples produced zero edges: {offenders[:5]}"
        )


__all__ = [
    "ActivationError",
    "MechanismRun",
    "VariantFlags",
    "activation",
    "activation_preflight",
    "build_run",
    "mechanism_metrics",
    "run_example",
]
