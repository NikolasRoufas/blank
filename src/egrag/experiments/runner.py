"""The experiment runner: execute variants, score, persist artifacts, resume.

Evaluation is kept separate from the production pipeline. Gold labels are used
only to score predictions *after* generation; they are never passed to a system
variant. Runs are resumable (completed example/variant/seed triples are skipped
and preserved) and corruption-tolerant (a partial trailing JSONL line is skipped
and recomputed). Failed examples remain visible: aggregates preserve failure
counts and never drop failures.
"""

from __future__ import annotations

import time
from pathlib import Path

from egrag import __version__
from egrag.domain.errors import ConfigurationError
from egrag.experiments import artifacts as art
from egrag.experiments.datasets import (
    check_dataset_integrity,
    dataset_fingerprint,
    get_dataset,
    load_examples,
)
from egrag.experiments.fairness import check_fairness, tuning_indicators
from egrag.experiments.metrics import (
    METRIC_KINDS,
    answer_accuracy,
    citation_completeness,
    citation_precision,
    citation_recall,
    evidence_precision,
    evidence_recall,
    exact_match,
    heuristic_contradiction_rate,
    invalid_citation_count,
    is_empty_prediction,
    lexical_entailment,
    metric_kind,
    normalized_exact_match,
    token_f1,
    unsupported_claim_rate,
)
from egrag.experiments.models import (
    AggregateMetrics,
    ComparisonResult,
    DatasetExample,
    ExampleResult,
    ExperimentConfig,
    ExperimentManifest,
    SystemVariant,
)
from egrag.experiments.stats import aggregate_seeds, mean, paired_comparison
from egrag.experiments.variants import RunSettings, SystemOutput, get_variant, run_system
from egrag.generation import FakeTextGenerator, GenerationConfig, TextGenerator
from egrag.reproducibility import environment_info, git_commit, now
from egrag.security import safe_artifact_path


def _build_generator(name: str) -> TextGenerator:
    if name == "fake":
        return FakeTextGenerator()
    raise ConfigurationError(
        f"experiment generator {name!r} is not supported offline; use 'fake' "
        "(real adapters require extras and are out of scope for the deterministic harness)"
    )


def _compute_metrics(
    example: DatasetExample, output: SystemOutput
) -> tuple[dict[str, float], list[str]]:
    warnings: list[str] = []
    m: dict[str, float] = {}

    if example.gold_answers:
        m["exact_match"] = exact_match(output.answer, example.gold_answers)
        m["normalized_exact_match"] = normalized_exact_match(output.answer, example.gold_answers)
        m["token_f1"] = token_f1(output.answer, example.gold_answers)
        m["answer_accuracy"] = answer_accuracy(output.answer, example.gold_answers)
    else:
        warnings.append(f"no gold answers for {example.example_id!r}; answer metrics skipped")

    m["empty_prediction"] = float(is_empty_prediction(output.answer))
    m["invalid_citations"] = float(
        invalid_citation_count(output.cited_claim_ids, output.known_claim_ids)
    )

    if example.gold_evidence.has_evidence:
        gold = example.gold_evidence.source_ids
        m["citation_precision"] = citation_precision(output.cited_source_ids, gold)
        m["citation_recall"] = citation_recall(output.cited_source_ids, gold)
        m["citation_completeness"] = citation_completeness(output.cited_source_ids, gold)
        m["evidence_precision"] = evidence_precision(output.selected_source_ids, gold)
        m["evidence_recall"] = evidence_recall(output.selected_source_ids, gold)
    else:
        warnings.append(
            f"missing gold evidence for {example.example_id!r}; citation/evidence metrics skipped"
        )

    # Heuristic (clearly labeled) metrics.
    m["answer_evidence_entailment"] = lexical_entailment(output.answer, output.evidence_texts)
    m["contradiction_rate"] = heuristic_contradiction_rate(output.answer, output.evidence_texts)
    sentences = [s for s in output.answer.replace("!", ".").split(".") if s.strip()]
    m["unsupported_claim_rate"] = unsupported_claim_rate(sentences, output.evidence_texts)

    for key, value in output.counts.items():
        m[key] = float(value)
    return m, warnings


def _aggregate(results: list[ExampleResult], variant: str, seed: int) -> AggregateMetrics:
    subset = [r for r in results if r.variant == variant and r.seed == seed]
    successful = [r for r in subset if not r.failed]
    metric_names: set[str] = set()
    for r in successful:
        metric_names |= set(r.metrics)
    metrics = {
        name: mean([r.metrics[name] for r in successful if name in r.metrics])
        for name in sorted(metric_names)
    }
    warnings: list[str] = []
    for r in subset:
        warnings.extend(r.warnings)
    return AggregateMetrics(
        variant=variant,
        seed=seed,
        num_examples=len(subset),
        num_failed=sum(1 for r in subset if r.failed),
        metrics=metrics,
        metric_kinds={name: str(metric_kind(name)) for name in metrics},
        warnings=tuple(dict.fromkeys(warnings)),
    )


class ExperimentRunner:
    """Runs experiments and writes inspectable, reproducible artifacts."""

    def __init__(self, config: ExperimentConfig) -> None:
        self._config = config
        # Resolve the output directory first, then validate the *resolved* path.
        # Passing the raw (possibly relative) output_dir to safe_artifact_path
        # would re-join it onto the base and produce a doubly-nested directory.
        self._base = Path(config.output_dir).resolve()
        self._dir = safe_artifact_path(self._base, self._base.parent)
        self._settings = RunSettings(
            top_k=config.top_k,
            evidence_token_budget=config.evidence_token_budget,
            reserved_output_tokens=config.reserved_output_tokens,
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
        )

    def run(self, *, resume: bool = False) -> ExperimentManifest:
        config = self._config
        started = now()
        self._dir.mkdir(parents=True, exist_ok=True)

        variants = [get_variant(name) for name in config.variants]
        fairness_issues = check_fairness(config, variants) + tuning_indicators(config)
        if config.enforce_fairness and fairness_issues:
            raise ConfigurationError("fairness violations: " + "; ".join(fairness_issues))

        adapter = get_dataset(config.dataset, path=config.dataset_path)
        examples = load_examples(adapter, limit=config.limit)
        integrity_issues = check_dataset_integrity(examples)
        fingerprint = dataset_fingerprint(examples)

        results_path = self._dir / art.RESULTS_FILE
        existing, skipped = art.read_results(results_path)
        if not resume:
            existing, skipped = [], 0
            results_path.unlink(missing_ok=True)
        done = art.completed_keys(existing)
        results: list[ExampleResult] = list(existing)

        generator = _build_generator(config.generator)
        timings: dict[str, float] = {}

        for variant in variants:
            for seed in config.seeds:
                gen_cfg = GenerationConfig(deterministic=True, seed=seed)
                for example in examples:
                    key = (variant.name, seed, example.example_id)
                    if key in done:
                        continue
                    result = self._run_one(example, variant, seed, generator, gen_cfg)
                    results.append(result)
                    art.append_jsonl(results_path, result.model_dump_json())
                    timings[f"{variant.name}|{seed}|{example.example_id}"] = result.latency_ms

        manifest = ExperimentManifest(
            experiment_name=config.name,
            egrag_version=__version__,
            dataset=config.dataset,
            dataset_fingerprint=fingerprint,
            num_examples=len(examples),
            variants=tuple(variants),
            seeds=config.seeds,
            generator=config.generator,
            top_k=config.top_k,
            evidence_token_budget=config.evidence_token_budget,
            reserved_output_tokens=config.reserved_output_tokens,
            fairness_enforced=config.enforce_fairness,
            started_at=started,
            ended_at=now(),
            git_commit=git_commit(),
            environment=environment_info(),
            resolved_config=config.model_dump(mode="json"),
            warnings=tuple(
                integrity_issues
                + ([f"skipped {skipped} corrupted result lines"] if skipped else [])
            ),
        )
        self._write_artifacts(manifest, results, examples, variants, timings)
        return manifest

    def _run_one(
        self,
        example: DatasetExample,
        variant: SystemVariant,
        seed: int,
        generator: TextGenerator,
        gen_cfg: GenerationConfig,
    ) -> ExampleResult:
        from egrag.domain.models import Query

        query = Query(query_id=f"{example.example_id}", text=example.question)
        start = time.perf_counter()
        try:
            output = run_system(
                variant,
                query,
                example.documents,
                generator=generator,
                config=gen_cfg,
                settings=self._settings,
            )
        # Broad by design: any failure must be recorded as a *visible* result
        # (with its error) rather than crashing the run or being dropped.
        except Exception as exc:
            latency = (time.perf_counter() - start) * 1000.0
            return ExampleResult(
                example_id=example.example_id,
                variant=variant.name,
                seed=seed,
                failed=True,
                error=f"{type(exc).__name__}: {exc}",
                latency_ms=latency,
            )
        latency = (time.perf_counter() - start) * 1000.0
        metrics, warnings = _compute_metrics(example, output)
        metrics["latency_ms"] = latency
        return ExampleResult(
            example_id=example.example_id,
            variant=variant.name,
            seed=seed,
            answer=output.answer,
            citations=output.cited_claim_ids,
            evidence_source_ids=output.selected_source_ids,
            metrics=metrics,
            counts=output.counts,
            latency_ms=latency,
            warnings=tuple(warnings) + output.warnings,
        )

    def _write_artifacts(
        self,
        manifest: ExperimentManifest,
        results: list[ExampleResult],
        examples: list[DatasetExample],
        variants: list[SystemVariant],
        timings: dict[str, float],
    ) -> None:
        art.write_json(self._dir / art.CONFIG_FILE, self._config.model_dump(mode="json"))
        art.write_json(self._dir / art.MANIFEST_FILE, manifest.model_dump(mode="json"))
        art.write_json(
            self._dir / art.REPRO_FILE,
            {
                "egrag_version": manifest.egrag_version,
                "schema_version": manifest.schema_version,
                "git_commit": manifest.git_commit,
                "environment": manifest.environment,
                "dataset_fingerprint": manifest.dataset_fingerprint,
                "seeds": list(manifest.seeds),
            },
        )

        per_seed: list[AggregateMetrics] = []
        seed_groups: dict[str, list[dict[str, float]]] = {}
        for variant in variants:
            for seed in self._config.seeds:
                agg = _aggregate(results, variant.name, seed)
                per_seed.append(agg)
                seed_groups.setdefault(variant.name, []).append(agg.metrics)
        seed_aggregated = {name: aggregate_seeds(group) for name, group in seed_groups.items()}
        art.write_json(
            self._dir / art.AGGREGATE_FILE,
            {
                "per_seed": [a.model_dump(mode="json") for a in per_seed],
                "seed_aggregated": seed_aggregated,
            },
        )

        warnings = {a.variant + "|" + str(a.seed): list(a.warnings) for a in per_seed}
        art.write_json(self._dir / art.WARNINGS_FILE, warnings)
        art.write_json(self._dir / art.TIMING_FILE, timings)

        failures = [r for r in results if r.failed]
        art.atomic_write_text(
            self._dir / art.FAILURES_FILE,
            "\n".join(f"{r.variant}|{r.seed}|{r.example_id}\t{r.error}" for r in failures) + "\n"
            if failures
            else "no failures\n",
        )

        # Evidence packages and graph artifacts for selected (first seed) examples.
        self._write_example_artifacts(examples, variants)

    def _write_example_artifacts(
        self, examples: list[DatasetExample], variants: list[SystemVariant]
    ) -> None:
        from egrag.domain.models import Query

        generator = _build_generator(self._config.generator)
        gen_cfg = GenerationConfig(deterministic=True, seed=self._config.seeds[0])
        for variant in variants:
            for example in examples:
                query = Query(query_id=example.example_id, text=example.question)
                try:
                    output = run_system(
                        variant,
                        query,
                        example.documents,
                        generator=generator,
                        config=gen_cfg,
                        settings=self._settings,
                    )
                # Artifacts are best-effort; any per-example failure is already
                # recorded as a visible result/failures.log entry by the main loop.
                except Exception:
                    continue
                stem = f"{variant.name}__{example.example_id}"
                art.write_json(
                    self._dir / art.PACKAGES_DIR / f"{stem}.json",
                    output.package.model_dump(mode="json"),
                )
                if variant.family == "graph":
                    art.write_json(
                        self._dir / art.GRAPHS_DIR / f"{stem}.json",
                        {
                            "claims": [c.claim_id for c in output.package.claims],
                            "relations": [
                                {
                                    "source": r.source_claim_id,
                                    "target": r.target_claim_id,
                                    "type": str(r.relation_type),
                                }
                                for r in output.package.relations
                            ],
                        },
                    )


def summarize(output_dir: str | Path) -> dict[str, AggregateMetrics]:
    """Aggregate per-variant/seed metrics from a completed run's results."""

    results, _ = art.read_results(Path(output_dir) / art.RESULTS_FILE)
    keys = sorted({(r.variant, r.seed) for r in results})
    return {f"{v}|{s}": _aggregate(results, v, s) for v, s in keys}


def compare(
    output_dir: str | Path,
    variant_a: str,
    variant_b: str,
    metric: str,
    *,
    seed: int = 0,
    samples: int = 1000,
    bootstrap_seed: int = 0,
) -> ComparisonResult:
    """Paired bootstrap comparison aligned by example ID for one seed."""

    from egrag.experiments.fairness import align_by_example_id

    results, _ = art.read_results(Path(output_dir) / art.RESULTS_FILE)
    a = [r for r in results if r.variant == variant_a and r.seed == seed and not r.failed]
    b = [r for r in results if r.variant == variant_b and r.seed == seed and not r.failed]
    paired_results, _issues = align_by_example_id(a, b)
    paired = [
        (ra.metrics.get(metric, 0.0), rb.metrics.get(metric, 0.0)) for ra, rb in paired_results
    ]
    return paired_comparison(
        metric, variant_a, variant_b, paired, samples=samples, seed=bootstrap_seed
    )


def inspect_example(output_dir: str | Path, example_id: str) -> list[ExampleResult]:
    results, _ = art.read_results(Path(output_dir) / art.RESULTS_FILE)
    return [r for r in results if r.example_id == example_id]


__all__ = ["METRIC_KINDS", "ExperimentRunner", "compare", "inspect_example", "summarize"]
