"""Typer CLI: run, search, extract, graph, reason, inspect-config, doctor.

Commands return a non-zero exit code on invalid configuration or execution
failure, with an actionable message on standard error. ``run``, ``search``,
``extract``, ``graph``, and ``reason`` use deterministic components and built-in
demo data, so they work offline without optional dependencies.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import NoReturn

import typer
from pydantic import ValidationError as PydanticValidationError

from egrag import __version__
from egrag.adapters.extraction import SentenceClaimExtractor
from egrag.adapters.retrieval import BM25Retriever, SentenceAwareChunker, prepare_passages
from egrag.answering import answer_query
from egrag.cli.experiment import experiment_app
from egrag.composition import run_pipeline
from egrag.config import EGRagSettings, load_config, load_settings
from egrag.domain.errors import (
    ConfigurationError,
    GenerationError,
    MissingDependencyError,
    ModelLoadError,
    PipelineError,
    SecurityError,
)
from egrag.domain.models import AtomicClaim, EvidenceGraphSnapshot, Query
from egrag.fakes import build_demo_documents
from egrag.generation import (
    FakeTextGenerator,
    GenerationConfig,
    HuggingFaceGenerator,
    OpenAICompatibleGenerator,
    TextGenerator,
)
from egrag.graph import GraphBuilder, GraphSerializer, LexicalPairClassifier
from egrag.observability.logging import configure_logging
from egrag.reasoning import (
    BaselineInitialScorer,
    CharacterTokenCounter,
    ConflictSetResolver,
    GreedyConnectedSelector,
    MetadataReliability,
    SignedBeliefPropagator,
    TokenBudget,
)
from egrag.reasoning.demo import build_demo_reasoning_graph, demo_query

app = typer.Typer(
    name="egrag",
    help="Evidence Graph Retrieval-Augmented Generation.",
    no_args_is_help=True,
    add_completion=False,
)
app.add_typer(experiment_app)

# Exit codes
_EXIT_CONFIG = 2
_EXIT_FAILURE = 1

# Core and optional dependencies surfaced by `doctor`.
_CORE_DEPS = ("pydantic", "pydantic_settings", "typer")
_OPTIONAL_DEPS = {
    "rank_bm25": "retrieval",
    "sentence_transformers": "dense",
    "networkx": "graph",
    "transformers": "local-models",
    "httpx": "http-models",
    "numpy": "experiments",
}


def _fail(message: str, code: int) -> NoReturn:
    """Print an actionable error to stderr and exit with ``code``."""

    typer.echo(f"error: {message}", err=True)
    raise typer.Exit(code)


def _load(env_file: str | None) -> EGRagSettings:
    try:
        return load_settings(env_file)
    except ConfigurationError as exc:
        _fail(f"{exc}. Check EGRAG_* variables or the --env-file you supplied.", _EXIT_CONFIG)


@app.command()
def run(
    query: str = typer.Option(..., "--query", "-q", help="The natural-language query."),
    generator: str = typer.Option("fake", "--generator", help="Generator: fake|openai|hf."),
    base_url: str | None = typer.Option(None, "--base-url", help="OpenAI-compatible base URL."),
    model: str | None = typer.Option(None, "--model", help="Model name for openai/hf generators."),
    top_k: int = typer.Option(4, "--top-k", help="Passages to retrieve."),
    evidence_budget: int = typer.Option(512, "--evidence-budget", help="Evidence token budget."),
    deterministic: bool = typer.Option(True, "--deterministic/--sampled", help="Decoding mode."),
    config_path: str | None = typer.Option(None, "--config", help="Hierarchical YAML config."),
    artifact_dir: str | None = typer.Option(None, "--artifact-dir", help="Write run artifacts."),
    json_output: bool = typer.Option(False, "--json", help="Emit the full result as JSON."),
    debug: bool = typer.Option(False, "--debug", help="Print a debug trace to stderr."),
    package_out: str | None = typer.Option(
        None, "--package-out", help="Write EvidencePackage JSON."
    ),
    graph_export: str | None = typer.Option(None, "--graph-export", help="Write graph JSON."),
    env_file: str | None = typer.Option(None, "--env-file", help="Optional .env config file."),
) -> None:
    """Run the full evidence-graph pipeline for a query and produce a grounded answer."""

    settings = _load(env_file)
    configure_logging(settings.log_level)

    try:
        parsed_query = Query(query_id="cli-query", text=query)
    except PydanticValidationError:
        _fail("the query text must be a non-empty string.", _EXIT_CONFIG)

    if config_path is not None:
        try:
            config = load_config(config_path)
        except ConfigurationError as exc:
            _fail(f"invalid configuration: {exc}", _EXIT_CONFIG)
        try:
            result = run_pipeline(config, parsed_query, artifact_dir=artifact_dir)
        except (ConfigurationError, ModelLoadError, MissingDependencyError, SecurityError) as exc:
            _fail(f"configuration/runtime error: {exc}", _EXIT_CONFIG)
        except (PipelineError, GenerationError, ValueError) as exc:
            _fail(f"pipeline failed: {exc}", _EXIT_FAILURE)
    else:
        text_generator = _build_generator(generator, base_url, model)
        documents = build_demo_documents()
        gen_config = GenerationConfig(deterministic=deterministic, seed=settings.seed)
        try:
            result = answer_query(
                parsed_query,
                documents,
                generator=text_generator,
                config=gen_config,
                top_k=top_k,
                evidence_token_budget=evidence_budget,
                chunk_size=settings.chunk_size,
                chunk_overlap=settings.chunk_overlap,
            )
        except (PipelineError, GenerationError, ValueError) as exc:
            _fail(f"pipeline failed: {exc}", _EXIT_FAILURE)

    if package_out is not None:
        Path(package_out).write_text(result.package.model_dump_json(indent=2), encoding="utf-8")
    if graph_export is not None:
        Path(graph_export).write_text(
            GraphSerializer(indent=2).serialize(
                EvidenceGraphSnapshot(
                    snapshot_id="export",
                    claims=result.package.claims,
                    relations=result.package.relations,
                )
            ),
            encoding="utf-8",
        )

    if json_output:
        typer.echo(result.model_dump_json(indent=2))
        return

    answer = result.answer
    typer.echo(answer.text)
    typer.echo("")
    typer.echo(f"citations: {', '.join(answer.cited_claim_ids) or '(none)'}")
    typer.echo(f"uncertainty: {answer.uncertainty or '(none)'}")
    typer.echo(f"abstained: {answer.abstained}")
    typer.echo(f"conflicts: {result.metrics.num_conflicts}")
    typer.echo(f"validation warnings: {len(answer.unsupported_warnings)}")
    for warning in answer.unsupported_warnings:
        typer.echo(f"  - {warning}")
    if debug:
        typer.echo(
            f"[debug] passages={result.metrics.num_passages} claims={result.metrics.num_claims} "
            f"relations={result.metrics.num_relations} selected={result.metrics.num_selected} "
            f"budget_used={result.package.budget.used if result.package.budget else 0}",
            err=True,
        )


def _build_generator(name: str, base_url: str | None, model: str | None) -> TextGenerator:
    if name == "fake":
        return FakeTextGenerator()
    if name == "openai":
        if not base_url or not model:
            _fail("the openai generator requires --base-url and --model.", _EXIT_CONFIG)
        return OpenAICompatibleGenerator(base_url, model)
    if name == "hf":
        if not model:
            _fail("the hf generator requires --model.", _EXIT_CONFIG)
        return HuggingFaceGenerator(model)
    _fail(f"unknown generator {name!r}; choose fake, openai, or hf.", _EXIT_CONFIG)


@app.command()
def search(
    query: str = typer.Option(..., "--query", "-q", help="The natural-language query."),
    top_k: int = typer.Option(5, "--top-k", help="Maximum number of passages to return."),
    show_components: bool = typer.Option(
        False, "--show-components", help="Print raw per-retriever component scores."
    ),
    json_output: bool = typer.Option(False, "--json", help="Emit ranked passages as JSON."),
    env_file: str | None = typer.Option(None, "--env-file", help="Optional .env config file."),
) -> None:
    """Retrieve and rank passages from a small demo corpus (no claim extraction)."""

    settings = _load(env_file)
    configure_logging(settings.log_level)

    try:
        parsed_query = Query(query_id="cli-search", text=query)
    except PydanticValidationError:
        _fail("the query text must be a non-empty string.", _EXIT_CONFIG)

    chunker = SentenceAwareChunker(chunk_size=settings.chunk_size, overlap=settings.chunk_overlap)
    passages = prepare_passages(build_demo_documents(), chunker)
    retriever = BM25Retriever(passages)

    try:
        report = retriever.search_report(parsed_query, top_k)
    except ValueError as exc:
        _fail(f"invalid retrieval request: {exc}", _EXIT_CONFIG)

    if json_output:
        typer.echo(report.model_dump_json(indent=2))
        return

    if not report.results:
        typer.echo("(no matching passages)")
    for item in report.results:
        typer.echo(f"[{item.rank}] {item.passage.passage_id}  score={item.score:.4f}")
        typer.echo(f"    {item.passage.text}")
        if show_components:
            rendered = ", ".join(f"{name}={value:.4f}" for name, value in item.components.items())
            typer.echo(f"    components: {rendered or '(none)'}")
    typer.echo(
        f"candidates={report.stats.num_candidates} results={report.stats.num_results} "
        f"retrieval_ms={report.stats.retrieval_ms:.2f}"
    )


@app.command()
def extract(
    query: str = typer.Option(..., "--query", "-q", help="The natural-language query."),
    top_k: int = typer.Option(3, "--top-k", help="Number of passages to extract from."),
    json_output: bool = typer.Option(False, "--json", help="Emit claims as JSON."),
    env_file: str | None = typer.Option(None, "--env-file", help="Optional .env config file."),
) -> None:
    """Retrieve passages and extract atomic claims (stops before graph construction)."""

    settings = _load(env_file)
    configure_logging(settings.log_level)

    try:
        parsed_query = Query(query_id="cli-extract", text=query)
    except PydanticValidationError:
        _fail("the query text must be a non-empty string.", _EXIT_CONFIG)

    documents = build_demo_documents()
    sources = {doc.source.source_id: doc.source for doc in documents}
    chunker = SentenceAwareChunker(chunk_size=settings.chunk_size, overlap=settings.chunk_overlap)
    passages = prepare_passages(documents, chunker)
    retriever = BM25Retriever(passages)
    extractor = SentenceClaimExtractor()

    try:
        hits = retriever.retrieve(parsed_query, top_k)
    except ValueError as exc:
        _fail(f"invalid retrieval request: {exc}", _EXIT_CONFIG)

    claims: list[AtomicClaim] = []
    for passage in hits:
        source = sources.get(passage.span.source_id)
        claims.extend(extractor.extract(passage, query=parsed_query, source=source))

    if json_output:
        payload = {"claims": [claim.model_dump(mode="json") for claim in claims]}
        typer.echo(json.dumps(payload, indent=2))
        return

    if not claims:
        typer.echo("(no claims extracted)")
    for claim in claims:
        span = claim.provenance.spans[0]
        typer.echo(f"- {claim.text}")
        typer.echo(
            f"    source={claim.provenance.source.source_id} "
            f"passage={claim.provenance.passage_id} span=[{span.start}:{span.end}] "
            f"confidence={claim.extraction_confidence:.2f}"
        )
        if claim.semantics is not None:
            sem = claim.semantics
            details = []
            if sem.negation:
                details.append("negated")
            if sem.modality:
                details.append(f"modality={list(sem.modality)}")
            if sem.attribution:
                details.append(f"attributed_to={sem.attribution}")
            if details:
                typer.echo(f"    {', '.join(details)}")
    typer.echo(f"claims={len(claims)} passages={len(hits)}")


@app.command()
def graph(
    query: str = typer.Option(..., "--query", "-q", help="The natural-language query."),
    top_k: int = typer.Option(4, "--top-k", help="Number of passages to extract from."),
    json_output: bool = typer.Option(False, "--json", help="Emit the graph snapshot as JSON."),
    graphml: bool = typer.Option(False, "--graphml", help="Emit GraphML (needs the graph extra)."),
    env_file: str | None = typer.Option(None, "--env-file", help="Optional .env config file."),
) -> None:
    """Build and inspect the evidence graph for a query (stops before belief)."""

    settings = _load(env_file)
    configure_logging(settings.log_level)

    try:
        parsed_query = Query(query_id="cli-graph", text=query)
    except PydanticValidationError:
        _fail("the query text must be a non-empty string.", _EXIT_CONFIG)

    documents = build_demo_documents()
    sources = {doc.source.source_id: doc.source for doc in documents}
    chunker = SentenceAwareChunker(chunk_size=settings.chunk_size, overlap=settings.chunk_overlap)
    passages = prepare_passages(documents, chunker)
    retriever = BM25Retriever(passages)
    extractor = SentenceClaimExtractor()

    try:
        hits = retriever.retrieve(parsed_query, top_k)
    except ValueError as exc:
        _fail(f"invalid retrieval request: {exc}", _EXIT_CONFIG)

    claims: list[AtomicClaim] = []
    for passage in hits:
        claims.extend(
            extractor.extract(
                passage, query=parsed_query, source=sources.get(passage.span.source_id)
            )
        )

    result = GraphBuilder(LexicalPairClassifier()).build(claims, query=parsed_query)
    snapshot = result.graph.snapshot()

    if graphml:
        from egrag.adapters.graph import to_graphml

        try:
            typer.echo(to_graphml(snapshot))
        except RuntimeError as exc:
            _fail(str(exc), _EXIT_FAILURE)
        return

    if json_output:
        payload = {
            "snapshot": snapshot.model_dump(mode="json"),
            "metrics": result.metrics.model_dump(mode="json"),
        }
        typer.echo(json.dumps(payload, indent=2))
        return

    summary = result.graph.summary()
    typer.echo(f"nodes={summary.num_nodes} edges={summary.num_edges} sources={summary.num_sources}")
    typer.echo(f"relations: {summary.edges_by_type or '{}'}")
    typer.echo(
        f"candidate pruning: possible={result.metrics.possible_pairs} "
        f"generated={result.metrics.generated_candidate_pairs} "
        f"classified={result.metrics.classified_pairs} pruned={result.metrics.pruned_pairs}"
    )
    typer.echo(f"components={summary.num_components}")
    for relation in snapshot.relations:
        typer.echo(
            f"- {relation.source_claim_id} --{relation.relation_type.value}"
            f"[{relation.direction.value}]--> {relation.target_claim_id} "
            f"(conf={relation.relation_confidence:.2f})"
        )


@app.command()
def reason(
    json_output: bool = typer.Option(
        False, "--json", help="Emit the full reasoning trace as JSON."
    ),
) -> None:
    """Run the reasoning pipeline on a fixed synthetic example and print the trace."""

    graph = build_demo_reasoning_graph()
    query = demo_query()
    scorer = BaselineInitialScorer(MetadataReliability())
    board = scorer.score(graph, query)
    propagator = SignedBeliefPropagator()
    propagation = propagator.propagate(graph, board)
    board = propagator.apply(board, propagation)
    conflicts = ConflictSetResolver().resolve(graph, board)
    selection = GreedyConnectedSelector().select(
        graph,
        query,
        board,
        token_budget=TokenBudget(total=256, reserved_output=64),
        token_counter=CharacterTokenCounter(),
        conflicts=conflicts,
    )

    if json_output:
        payload = {
            "scores": [s.model_dump(mode="json") for s in board.scores],
            "diagnostics": [d.model_dump(mode="json") for d in propagation.diagnostics],
            "converged": propagation.converged,
            "conflicts": [c.model_dump(mode="json") for c in conflicts],
            "selection": selection.model_dump(mode="json"),
        }
        typer.echo(json.dumps(payload, indent=2))
        return

    typer.echo("initial scores:")
    for score in board.scores:
        typer.echo(
            f"  {score.claim_id}: initial={score.initial_belief:.4f} "
            f"propagated={score.propagated_belief:.4f} utility={score.query_utility:.4f}"
        )
    typer.echo("relations:")
    for relation in graph.edges():
        typer.echo(
            f"  {relation.source_claim_id} --{relation.relation_type.value}--> "
            f"{relation.target_claim_id} (conf={relation.relation_confidence:.2f})"
        )
    typer.echo(
        f"propagation: converged={propagation.converged} iterations={propagation.iterations}"
    )
    for diag in propagation.diagnostics:
        typer.echo(f"  iter {diag.iteration}: max_delta={diag.max_delta:.3e}")
    typer.echo("conflicts:")
    for conflict in conflicts:
        typer.echo(
            f"  {conflict.conflict_id}: {sorted(conflict.claim_ids)} -> "
            f"{conflict.outcome.value if conflict.outcome else 'none'} "
            f"(preferred={conflict.preferred_claim_id})"
        )
    typer.echo(f"selected subgraph: {list(selection.selected_claim_ids)}")
    for entry in selection.entries:
        state = "SELECTED" if entry.selected else "rejected"
        typer.echo(f"  [{state}] {entry.claim_id}: {entry.reason}")


@app.command("inspect-config")
def inspect_config(
    env_file: str | None = typer.Option(None, "--env-file", help="Optional .env config file."),
) -> None:
    """Print the resolved configuration as JSON."""

    settings = _load(env_file)
    typer.echo(settings.model_dump_json(indent=2))


@app.command()
def doctor() -> None:
    """Check the environment and report optional-dependency availability."""

    typer.echo(f"egrag version: {__version__}")
    typer.echo(f"python: {sys.version.split()[0]}")

    missing_core = [name for name in _CORE_DEPS if importlib.util.find_spec(name) is None]
    typer.echo("core dependencies:")
    for name in _CORE_DEPS:
        present = importlib.util.find_spec(name) is not None
        typer.echo(f"  - {name}: {'ok' if present else 'MISSING'}")

    typer.echo("optional dependencies:")
    for name, extra in _OPTIONAL_DEPS.items():
        present = importlib.util.find_spec(name) is not None
        state = "installed" if present else f"not installed (extra: {extra})"
        typer.echo(f"  - {name}: {state}")

    if missing_core:
        _fail(
            f"missing core dependencies: {', '.join(missing_core)}. Run `uv sync`.",
            _EXIT_FAILURE,
        )


@app.command("gpu-readiness")
def gpu_readiness(
    device: str = typer.Option("auto", "--device", help="cpu/mps/cuda/cuda:N/int/auto."),
    dtype: str | None = typer.Option(None, "--dtype", help="float32/float16/bfloat16/auto."),
) -> None:
    """Print a device/GPU readiness report (torch, CUDA/MPS, selected device, disk)."""

    from egrag.hf_runtime import gpu_report

    typer.echo(json.dumps(gpu_report(device=device, dtype=dtype), indent=2))


def main() -> None:
    """Console-script / ``python -m`` entry point."""

    app()


if __name__ == "__main__":
    main()


__all__ = ["app", "main"]
