"""Artifact I/O for experiments: safe writes and corruption-tolerant reads."""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from egrag.experiments.models import ExampleResult

RESULTS_FILE = "results.jsonl"
MANIFEST_FILE = "manifest.json"
CONFIG_FILE = "resolved_config.json"
AGGREGATE_FILE = "aggregate.json"
FAILURES_FILE = "failures.log"
TIMING_FILE = "timing.json"
WARNINGS_FILE = "metric_warnings.json"
REPRO_FILE = "reproducibility.json"
PACKAGES_DIR = "packages"
GRAPHS_DIR = "graphs"


def atomic_write_text(path: Path, text: str) -> None:
    """Write text atomically (temp file in the same dir, then replace)."""

    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=".tmp-", suffix=path.suffix)
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    except OSError:
        tmp.unlink(missing_ok=True)
        raise


def write_json(path: Path, obj: Any) -> None:
    atomic_write_text(path, json.dumps(obj, indent=2, sort_keys=True, default=str))


def append_jsonl(path: Path, line_obj: str) -> None:
    """Append one already-serialized JSON line, flushing to disk."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(line_obj + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def read_results(path: Path) -> tuple[list[ExampleResult], int]:
    """Read existing per-example results, tolerating a corrupted trailing line.

    Returns the parsed results and the count of skipped (corrupted) lines, so
    interrupted/partial writes never silently corrupt a resumed run.
    """

    if not path.is_file():
        return [], 0
    results: list[ExampleResult] = []
    skipped = 0
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            results.append(ExampleResult.model_validate_json(line))
        except ValueError:
            skipped += 1
    return results, skipped


def completed_keys(results: Iterable[ExampleResult]) -> set[tuple[str, int, str]]:
    return {(r.variant, r.seed, r.example_id) for r in results}


__all__ = [
    "AGGREGATE_FILE",
    "CONFIG_FILE",
    "FAILURES_FILE",
    "GRAPHS_DIR",
    "MANIFEST_FILE",
    "PACKAGES_DIR",
    "REPRO_FILE",
    "RESULTS_FILE",
    "TIMING_FILE",
    "WARNINGS_FILE",
    "append_jsonl",
    "atomic_write_text",
    "completed_keys",
    "read_results",
    "write_json",
]
