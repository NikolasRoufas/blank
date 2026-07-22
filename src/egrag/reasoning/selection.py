"""Reasoning-subgraph selection (configurable baselines).

Three selectors are provided:

* :class:`TopClaimsSelector` — rank by base value, fill the budget (ablation;
  may be disconnected).
* :class:`GreedyConnectedSelector` — grow a connected subgraph by marginal
  objective gain, then retain contradiction counterparts for unresolved
  conflicts so uncertainty stays explainable.
* :class:`BeamSearchSelector` — bounded beam search over connected extensions.

The objective rewards query utility, propagated belief, query-entity coverage,
coherent support, independent provenance, and necessary uncertainty evidence;
it penalizes redundancy, repeated source lineage, disconnection, and avoidable
unresolved contradiction. Token budgeting is deterministic and tokenizer-
independent (a :class:`TokenCounter`), with a reserved output allocation. Every
candidate — selected or rejected — gets an explanation entry.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from egrag.domain.models import (
    ConflictOutcome,
    ConflictSet,
    EvidenceGraphSnapshot,
    Query,
    ReasoningSubgraph,
)
from egrag.domain.ports import TokenCounter
from egrag.graph.api import EvidenceGraph
from egrag.graph.text_utils import tokens
from egrag.reasoning.models import (
    ScoreBoard,
    ScoreComponents,
    SelectionConfig,
    SelectionEntry,
    SelectionResult,
    SelectionStrategy,
    TokenBudget,
)

_NEUTRAL_COMPONENTS = ScoreComponents(
    retrieval=0.0,
    query_relevance=0.0,
    extraction=0.0,
    source_reliability=0.0,
    temporal_validity=0.0,
    independent_support=0.0,
)


@dataclass
class _Context:
    snapshot: EvidenceGraphSnapshot
    belief: dict[str, float]
    utility: dict[str, float]
    token_cost: dict[str, int]
    support_nb: dict[str, set[str]]
    contra_nb: dict[str, set[str]]
    dup_nb: dict[str, set[str]]
    any_nb: dict[str, set[str]]
    source: dict[str, str]
    covered_terms: dict[str, set[str]]
    query_terms: set[str]
    unresolved: set[str]
    config: SelectionConfig
    components: dict[str, str] = field(default_factory=dict)


def _build_context(
    graph: EvidenceGraph,
    query: Query,
    board: ScoreBoard,
    token_counter: TokenCounter,
    conflicts: list[ConflictSet],
    config: SelectionConfig,
) -> _Context:
    snapshot = graph.snapshot()
    scores = board.by_id()
    query_terms = tokens(query.text)
    belief: dict[str, float] = {}
    utility: dict[str, float] = {}
    token_cost: dict[str, int] = {}
    support_nb: dict[str, set[str]] = {}
    contra_nb: dict[str, set[str]] = {}
    dup_nb: dict[str, set[str]] = {}
    any_nb: dict[str, set[str]] = {}
    source: dict[str, str] = {}
    covered: dict[str, set[str]] = {}
    for claim in snapshot.claims:
        cid = claim.claim_id
        score = scores.get(cid)
        belief[cid] = (
            score.propagated_belief
            if score is not None and score.propagated_belief is not None
            else (score.initial_belief if score is not None else 0.5)
        )
        utility[cid] = score.query_utility if score is not None else 0.0
        token_cost[cid] = token_counter.count(claim.text)
        support_nb[cid] = {c.claim_id for c in graph.supporting_evidence(cid)}
        contra_nb[cid] = {c.claim_id for c in graph.contradicting_evidence(cid)}
        dup_nb[cid] = {c.claim_id for c in graph.duplicates(cid)}
        any_nb[cid] = set(graph.neighbors(cid))
        source[cid] = claim.provenance.source.source_id
        covered[cid] = tokens(claim.text) & query_terms

    unresolved: set[str] = set()
    for conflict in conflicts:
        if conflict.outcome is ConflictOutcome.UNRESOLVED:
            unresolved.update(conflict.claim_ids)

    return _Context(
        snapshot=snapshot,
        belief=belief,
        utility=utility,
        token_cost=token_cost,
        support_nb=support_nb,
        contra_nb=contra_nb,
        dup_nb=dup_nb,
        any_nb=any_nb,
        source=source,
        covered_terms=covered,
        query_terms=query_terms,
        unresolved=unresolved,
        config=config,
    )


def _base_value(ctx: _Context, cid: str) -> float:
    cfg = ctx.config
    return cfg.utility_weight * ctx.utility[cid] + cfg.belief_weight * ctx.belief[cid]


def _marginal_gain(
    ctx: _Context, cid: str, selected: set[str], covered: set[str], used_sources: set[str]
) -> float:
    cfg = ctx.config
    gain = _base_value(ctx, cid)
    new_terms = ctx.covered_terms[cid] - covered
    if ctx.query_terms:
        gain += cfg.entity_coverage_weight * (len(new_terms) / len(ctx.query_terms))
    if ctx.support_nb[cid] & selected:
        gain += cfg.support_coherence_weight
    if ctx.source[cid] not in used_sources:
        gain += cfg.independence_weight
    else:
        gain -= cfg.repeated_lineage_penalty
    if ctx.contra_nb[cid] & selected:
        gain += cfg.uncertainty_weight  # retaining both sides explains uncertainty
    if ctx.dup_nb[cid] & selected:
        gain -= cfg.redundancy_penalty
    if cid in ctx.unresolved and not (ctx.contra_nb[cid] & selected):
        gain -= cfg.unresolved_conflict_penalty
    return gain


def _connected(ctx: _Context, cid: str, selected: set[str]) -> bool:
    return bool(ctx.any_nb[cid] & selected)


def _induced_relations(snapshot: EvidenceGraphSnapshot, selected: set[str]) -> tuple[str, ...]:
    return tuple(
        r.relation_id
        for r in snapshot.relations
        if r.source_claim_id in selected and r.target_claim_id in selected
    )


def _build_entries(
    ctx: _Context,
    board: ScoreBoard,
    selected: list[str],
    contributions: dict[str, float],
    reasons: dict[str, str],
) -> tuple[SelectionEntry, ...]:
    scores = board.by_id()
    entries: list[SelectionEntry] = []
    selected_set = set(selected)
    for claim in ctx.snapshot.claims:
        cid = claim.claim_id
        score = scores.get(cid)
        entries.append(
            SelectionEntry(
                claim_id=cid,
                initial_belief=score.initial_belief if score else ctx.belief[cid],
                propagated_belief=ctx.belief[cid],
                query_utility=ctx.utility[cid],
                components=score.components if score else _NEUTRAL_COMPONENTS,
                selection_contribution=round(contributions.get(cid, 0.0), 6),
                final_selection_score=round(_base_value(ctx, cid), 6),
                selected=cid in selected_set,
                reason=reasons.get(cid, "not selected"),
                supporting_neighbors=tuple(sorted(ctx.support_nb[cid])),
                contradicting_neighbors=tuple(sorted(ctx.contra_nb[cid])),
                duplicate_cluster=tuple(sorted(ctx.dup_nb[cid])),
                source_id=ctx.source[cid],
                tokens=ctx.token_cost[cid],
            )
        )
    return tuple(entries)


class _BaseSelector:
    def __init__(self, config: SelectionConfig | None = None) -> None:
        self._config = config or SelectionConfig()

    def _finish(
        self,
        ctx: _Context,
        board: ScoreBoard,
        selected: list[str],
        contributions: dict[str, float],
        reasons: dict[str, str],
        budget: TokenBudget,
        connected: bool,
    ) -> SelectionResult:
        entries = _build_entries(ctx, board, selected, contributions, reasons)
        total_tokens = sum(ctx.token_cost[c] for c in selected)
        return SelectionResult(
            subgraph_id="subgraph",
            strategy=self._config.strategy,
            selected_claim_ids=tuple(selected),
            relation_ids=_induced_relations(ctx.snapshot, set(selected)),
            entries=entries,
            total_tokens=total_tokens,
            budget=budget,
            connected=connected,
        )


class TopClaimsSelector(_BaseSelector):
    """Ablation baseline: highest base value first, fill the budget."""

    def __init__(self, config: SelectionConfig | None = None) -> None:
        super().__init__(config or SelectionConfig(strategy=SelectionStrategy.TOP_CLAIMS))

    def select(
        self,
        graph: EvidenceGraph,
        query: Query,
        board: ScoreBoard,
        *,
        token_budget: TokenBudget,
        token_counter: TokenCounter,
        conflicts: list[ConflictSet] | None = None,
    ) -> SelectionResult:
        ctx = _build_context(graph, query, board, token_counter, conflicts or [], self._config)
        order = sorted(ctx.belief, key=lambda c: (-_base_value(ctx, c), c))
        selected: list[str] = []
        contributions: dict[str, float] = {}
        reasons: dict[str, str] = {}
        used = 0
        for cid in order:
            cost = ctx.token_cost[cid]
            if used + cost <= token_budget.available:
                selected.append(cid)
                contributions[cid] = _base_value(ctx, cid)
                reasons[cid] = "top base value within budget"
                used += cost
            else:
                reasons[cid] = "rejected: exceeds token budget"
        connected = self._is_connected(ctx, selected)
        return self._finish(ctx, board, selected, contributions, reasons, token_budget, connected)

    @staticmethod
    def _is_connected(ctx: _Context, selected: list[str]) -> bool:
        if len(selected) <= 1:
            return True
        chosen = set(selected)
        return all(ctx.any_nb[c] & (chosen - {c}) for c in selected)


class GreedyConnectedSelector(_BaseSelector):
    """Grow a connected subgraph by marginal gain; retain uncertainty evidence."""

    def __init__(self, config: SelectionConfig | None = None) -> None:
        super().__init__(config or SelectionConfig(strategy=SelectionStrategy.GREEDY_CONNECTED))

    def select(
        self,
        graph: EvidenceGraph,
        query: Query,
        board: ScoreBoard,
        *,
        token_budget: TokenBudget,
        token_counter: TokenCounter,
        conflicts: list[ConflictSet] | None = None,
    ) -> SelectionResult:
        ctx = _build_context(graph, query, board, token_counter, conflicts or [], self._config)
        selected: list[str] = []
        contributions: dict[str, float] = {}
        reasons: dict[str, str] = {}
        covered: set[str] = set()
        used_sources: set[str] = set()
        used = 0
        connected = True

        candidates = sorted(ctx.belief, key=lambda c: (-_base_value(ctx, c), c))
        if not candidates:
            return self._finish(ctx, board, selected, contributions, reasons, token_budget, True)

        # Seed with the best claim that fits the budget.
        seed = next((c for c in candidates if ctx.token_cost[c] <= token_budget.available), None)
        if seed is None:
            for cid in candidates:
                reasons[cid] = "rejected: exceeds token budget"
            return self._finish(ctx, board, selected, contributions, reasons, token_budget, True)
        selected.append(seed)
        contributions[seed] = _base_value(ctx, seed)
        reasons[seed] = "seed: highest base value"
        covered |= ctx.covered_terms[seed]
        used_sources.add(ctx.source[seed])
        used += ctx.token_cost[seed]

        while True:
            best: tuple[float, str] | None = None
            for cid in candidates:
                if cid in reasons and reasons[cid].startswith(("seed", "selected")):
                    continue
                if cid in selected:
                    continue
                if used + ctx.token_cost[cid] > token_budget.available:
                    continue
                if not _connected(ctx, cid, set(selected)):
                    continue
                gain = _marginal_gain(ctx, cid, set(selected), covered, used_sources)
                if gain <= 0.0:
                    continue
                key = (gain, cid)
                if best is None or (-key[0], key[1]) < (-best[0], best[1]):
                    best = key
            if best is None:
                break
            gain, cid = best
            selected.append(cid)
            contributions[cid] = gain
            reasons[cid] = "selected: positive marginal gain"
            covered |= ctx.covered_terms[cid]
            used_sources.add(ctx.source[cid])
            used += ctx.token_cost[cid]

        # Retain contradiction counterparts for unresolved conflicts (uncertainty).
        for cid in list(selected):
            if cid not in ctx.unresolved:
                continue
            for counterpart in sorted(ctx.contra_nb[cid]):
                if counterpart in selected:
                    continue
                if used + ctx.token_cost[counterpart] <= token_budget.available:
                    selected.append(counterpart)
                    contributions[counterpart] = ctx.config.uncertainty_weight
                    reasons[counterpart] = "retained: explains unresolved contradiction"
                    used += ctx.token_cost[counterpart]

        for cid in candidates:
            reasons.setdefault(cid, "rejected: no positive connected marginal gain")
        return self._finish(ctx, board, selected, contributions, reasons, token_budget, connected)


class BeamSearchSelector(_BaseSelector):
    """Bounded beam search over connected extensions, scored by total objective."""

    def __init__(self, config: SelectionConfig | None = None) -> None:
        super().__init__(config or SelectionConfig(strategy=SelectionStrategy.BEAM))

    def select(
        self,
        graph: EvidenceGraph,
        query: Query,
        board: ScoreBoard,
        *,
        token_budget: TokenBudget,
        token_counter: TokenCounter,
        conflicts: list[ConflictSet] | None = None,
    ) -> SelectionResult:
        ctx = _build_context(graph, query, board, token_counter, conflicts or [], self._config)
        candidates = sorted(ctx.belief, key=lambda c: (-_base_value(ctx, c), c))
        width = self._config.beam_width

        # Each beam state: (selected tuple, used tokens). Scored by total objective.
        seeds = [c for c in candidates if ctx.token_cost[c] <= token_budget.available]
        beam: list[tuple[str, ...]] = [(c,) for c in seeds[:width]]
        if not beam:
            reasons = dict.fromkeys(candidates, "rejected: exceeds token budget")
            return self._finish(ctx, board, [], {}, reasons, token_budget, True)

        best_state = max(beam, key=lambda s: (self._score(ctx, s), tuple(s)))
        improved = True
        while improved:
            improved = False
            next_beam: list[tuple[str, ...]] = []
            for state in beam:
                state_set = set(state)
                used = sum(ctx.token_cost[c] for c in state)
                for cid in candidates:
                    if cid in state_set:
                        continue
                    if used + ctx.token_cost[cid] > token_budget.available:
                        continue
                    if not _connected(ctx, cid, state_set):
                        continue
                    extended = (*state, cid)
                    next_beam.append(extended)
            if not next_beam:
                break
            next_beam.sort(key=lambda s: (-self._score(ctx, s), tuple(sorted(s))))
            beam = next_beam[:width]
            top = beam[0]
            if self._score(ctx, top) > self._score(ctx, best_state):
                best_state = top
                improved = True

        selected = list(best_state)
        contributions = {cid: _base_value(ctx, cid) for cid in selected}
        reasons = dict.fromkeys(selected, "selected by beam search")
        for cid in candidates:
            reasons.setdefault(cid, "rejected: not in best beam state")
        return self._finish(ctx, board, selected, contributions, reasons, token_budget, True)

    def _score(self, ctx: _Context, state: tuple[str, ...]) -> float:
        selected: set[str] = set()
        covered: set[str] = set()
        used_sources: set[str] = set()
        total = 0.0
        for cid in state:
            total += _marginal_gain(ctx, cid, selected, covered, used_sources)
            selected.add(cid)
            covered |= ctx.covered_terms[cid]
            used_sources.add(ctx.source[cid])
        return total


def to_reasoning_subgraph(result: SelectionResult) -> ReasoningSubgraph:
    """Build a domain :class:`ReasoningSubgraph` from a selection result."""

    return ReasoningSubgraph(
        subgraph_id=result.subgraph_id,
        claim_ids=result.selected_claim_ids,
        relation_ids=result.relation_ids,
    )


def query_entity_coverage(claim_texts: list[str], query: Query) -> float:
    """Return the fraction of query terms covered by the given claim texts."""

    query_terms = tokens(query.text)
    if not query_terms:
        return 0.0
    covered: set[str] = set()
    for text in claim_texts:
        covered |= tokens(text) & query_terms
    return len(covered) / len(query_terms)


__all__ = [
    "BeamSearchSelector",
    "GreedyConnectedSelector",
    "TopClaimsSelector",
    "query_entity_coverage",
    "to_reasoning_subgraph",
]
