"""Dataset adapters and integrity checks.

Large datasets are **not** committed. Synthetic fixtures are built in-memory;
HotpotQA-/FEVER-style adapters read a local JSONL the user downloads following
``docs/experiments.md``. Benchmark support is only claimed once the adapter and
metrics are tested (see the tests). All adapters are offline.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from egrag.domain.errors import ConfigurationError, InvalidInputError
from egrag.domain.models import Document, SourceMetadata
from egrag.experiments.models import DatasetExample, GoldEvidence
from egrag.security import check_file_size, validate_within_base


@runtime_checkable
class DatasetAdapter(Protocol):
    """Loads a dataset into :class:`DatasetExample` objects."""

    name: str

    def load(self) -> list[DatasetExample]: ...


def _doc(source_id: str, text: str, title: str | None = None) -> Document:
    return Document(
        document_id=source_id,
        text=text,
        source=SourceMetadata(source_id=source_id, title=title),
    )


def dataset_fingerprint(examples: Sequence[DatasetExample]) -> str:
    """Stable fingerprint over example IDs and questions (order-sensitive)."""

    digest = hashlib.sha256()
    for ex in examples:
        digest.update(ex.example_id.encode("utf-8"))
        digest.update(b"\x00")
        digest.update(ex.question.encode("utf-8"))
        digest.update(b"\x01")
    return digest.hexdigest()


class SyntheticGraphDataset:
    """A tiny, fully-specified graph-reasoning dataset for deterministic tests."""

    name = "synthetic_graph"

    def load(self) -> list[DatasetExample]:
        return [
            DatasetExample(
                example_id="syn-1",
                question="What did the company announce about revenue?",
                documents=(
                    _doc("s1", "The company announced record revenue growth in 2023.", "Report A"),
                    _doc(
                        "s2", "Analysts confirmed the company's revenue grew sharply.", "Report B"
                    ),
                    _doc("s3", "An unrelated note about office furniture procurement.", "Memo C"),
                ),
                gold_answers=("record revenue growth",),
                gold_evidence=GoldEvidence(source_ids=("s1", "s2"), stance="supports"),
            ),
            DatasetExample(
                example_id="syn-2",
                question="Did the merger close on time?",
                documents=(
                    _doc("s4", "The merger closed on schedule in June.", "Filing D"),
                    _doc("s5", "Reports indicate the merger was delayed past June.", "News E"),
                ),
                gold_answers=("it is disputed",),
                gold_evidence=GoldEvidence(source_ids=("s4", "s5"), stance="not_enough_info"),
            ),
        ]


class TemporalConflictDataset:
    """Conflicting evidence with a temporal supersession signal (EG-RAG focus)."""

    name = "temporal_conflict"

    def load(self) -> list[DatasetExample]:
        return [
            DatasetExample(
                example_id="tmp-1",
                question="What is the current headquarters city?",
                documents=(
                    _doc("old", "In 2010 the headquarters was located in Springfield.", "Archive"),
                    _doc("new", "As of 2024 the headquarters is located in Rivertown.", "Update"),
                ),
                gold_answers=("Rivertown",),
                gold_evidence=GoldEvidence(source_ids=("new",), stance="supports"),
                metadata={"focus": "temporal_supersession"},
            ),
        ]


class JsonlDataset:
    """Adapter for HotpotQA-/FEVER-style examples stored as local JSONL.

    Each line is an object with ``id``, ``question``/``claim``, ``documents``
    (list of ``{source_id, text, title?}``), ``answers``/``label``, and optional
    ``gold_source_ids``. The file is read with path-traversal and size guards.
    """

    def __init__(
        self, path: str | Path, *, name: str = "jsonl", base_dir: str | Path = "."
    ) -> None:
        self.name = name
        self._path = Path(path)
        self._base = base_dir

    def load(self) -> list[DatasetExample]:
        resolved = validate_within_base(self._path, self._base)
        if not resolved.is_file():
            raise ConfigurationError(f"dataset file not found: {resolved}")
        check_file_size(resolved, max_bytes=50 * 1024 * 1024)
        examples: list[DatasetExample] = []
        for line_no, line in enumerate(resolved.read_text(encoding="utf-8").splitlines(), start=1):
            line = line.strip()
            if not line:
                continue
            try:
                examples.append(self._parse(json.loads(line)))
            except (ValueError, KeyError, TypeError) as exc:
                raise InvalidInputError(f"invalid dataset line {line_no}: {exc}") from exc
        return examples

    @staticmethod
    def _parse(obj: dict[str, Any]) -> DatasetExample:
        raw_docs = obj.get("documents", [])
        if not isinstance(raw_docs, list) or not raw_docs:
            raise ValueError("each example needs a non-empty 'documents' list")
        documents = tuple(
            _doc(str(d["source_id"]), str(d["text"]), title=d.get("title")) for d in raw_docs
        )
        question = obj.get("question") or obj.get("claim")
        if not question:
            raise ValueError("each example needs 'question' or 'claim'")
        answers = obj.get("answers") or ([obj["answer"]] if "answer" in obj else [])
        label = obj.get("label")
        stance_map = {
            "SUPPORTS": "supports",
            "REFUTES": "refutes",
            "NOT ENOUGH INFO": "not_enough_info",
        }
        stance = stance_map.get(str(label).upper()) if label is not None else None
        gold_sources = tuple(str(s) for s in obj.get("gold_source_ids", []))
        return DatasetExample(
            example_id=str(obj["id"]),
            question=str(question),
            documents=documents,
            gold_answers=tuple(str(a) for a in answers),
            gold_evidence=GoldEvidence(
                available=bool(gold_sources) or stance is not None,
                source_ids=gold_sources,
                # stance comes from a fixed map; Pydantic re-validates the Literal.
                stance=stance,  # type: ignore[arg-type]
            ),
            split=str(obj.get("split", "test")),
        )


# --- Integrity / leakage checks ----------------------------------------------


def _example_signature(ex: DatasetExample) -> str:
    payload = ex.question + "||" + "||".join(d.text for d in ex.documents)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def check_dataset_integrity(examples: Sequence[DatasetExample]) -> list[str]:
    """Return human-readable integrity issues (does not raise)."""

    issues: list[str] = []

    ids = [ex.example_id for ex in examples]
    seen: set[str] = set()
    for example_id in ids:
        if example_id in seen:
            issues.append(f"duplicate dataset id: {example_id!r}")
        seen.add(example_id)

    signatures: dict[str, str] = {}
    for ex in examples:
        sig = _example_signature(ex)
        if sig in signatures:
            issues.append(f"duplicate example content: {ex.example_id!r} ~ {signatures[sig]!r}")
        else:
            signatures[sig] = ex.example_id

    for ex in examples:
        if not ex.gold_evidence.has_evidence:
            issues.append(f"missing gold evidence: {ex.example_id!r}")

    # train/test overlap by question across differing splits
    by_question: dict[str, set[str]] = {}
    for ex in examples:
        by_question.setdefault(ex.question, set()).add(ex.split)
    for question, splits in by_question.items():
        if len(splits) > 1:
            issues.append(
                f"train/test overlap for question {question[:40]!r}: splits {sorted(splits)}"
            )

    return issues


BUILTIN_DATASETS: dict[str, DatasetAdapter] = {
    "synthetic_graph": SyntheticGraphDataset(),
    "temporal_conflict": TemporalConflictDataset(),
}


def get_dataset(name: str, *, path: str | None = None) -> DatasetAdapter:
    """Resolve a dataset adapter by name (or a JSONL file when ``path`` given)."""

    if path is not None:
        return JsonlDataset(path, name=name)
    if name not in BUILTIN_DATASETS:
        raise ConfigurationError(
            f"unknown dataset {name!r}; built-ins: {sorted(BUILTIN_DATASETS)} "
            "(or pass a JSONL path)"
        )
    return BUILTIN_DATASETS[name]


def load_examples(adapter: DatasetAdapter, *, limit: int | None = None) -> list[DatasetExample]:
    examples = list(adapter.load())
    return examples[:limit] if limit is not None else examples


__all__ = [
    "BUILTIN_DATASETS",
    "DatasetAdapter",
    "JsonlDataset",
    "SyntheticGraphDataset",
    "TemporalConflictDataset",
    "check_dataset_integrity",
    "dataset_fingerprint",
    "get_dataset",
    "load_examples",
]
