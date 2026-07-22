"""Real benchmark dataset adapters (FEVER gold-evidence; HotpotQA scaffold).

Reads **locally cached** dataset files only; never downloads or redistributes a
corpus. Produces the harness's canonical :class:`DatasetExample`. Gold answers and
gold evidence live only on the example and are never passed into the pipeline.

* **FEVER** — `copenlu/fever_gold_evidence` (cached JSONL). The gold evidence
  sentences are bundled inline, so this is the **gold-evidence** verification
  setting (not full-Wikipedia retrieval). Feasible fully offline.
* **HotpotQA** — only cached as parquet; loading requires a parquet reader
  (``pyarrow``), provided by the ``benchmarks`` optional extra. ``pyarrow`` is
  imported **lazily** inside :meth:`HotpotQADataset.load` so importing core
  EG-RAG never imports it. When the extra is absent the adapter raises a typed
  :class:`MissingDependencyError` naming the ``benchmarks`` extra (reported, not
  faked).
"""

from __future__ import annotations

import glob
import hashlib
import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from egrag.domain.errors import ConfigurationError, InvalidInputError, MissingDependencyError
from egrag.domain.models import Document, SourceMetadata
from egrag.experiments.models import DatasetExample, GoldEvidence
from egrag.security import check_file_size

_HF_HUB = Path("~/.cache/huggingface/hub").expanduser()
_FEVER_LABELS = {
    "SUPPORTS": "supports",
    "REFUTES": "refutes",
    "NOT ENOUGH INFO": "not_enough_info",
}
ADAPTER_VERSION = "1.0.0"


def _default_fever_path(split: str) -> Path | None:
    hits = sorted(
        glob.glob(
            str(_HF_HUB / f"datasets--copenlu--fever_gold_evidence/snapshots/*/{split}.jsonl")
        )
    )
    return Path(hits[0]) if hits else None


class FeverGoldEvidenceDataset:
    """FEVER (gold-evidence setting) from the cached `copenlu/fever_gold_evidence` JSONL."""

    name = "fever"

    def __init__(
        self, *, split: str = "valid", path: str | Path | None = None, limit: int | None = None
    ) -> None:
        self._split = split
        self._path = Path(path) if path is not None else _default_fever_path(split)
        self._limit = limit

    def load(self) -> list[DatasetExample]:
        if self._path is None or not self._path.is_file():
            raise ConfigurationError(
                f"FEVER {self._split} JSONL not found; expected the cached "
                "copenlu/fever_gold_evidence dataset or an explicit path"
            )
        check_file_size(self._path, max_bytes=500 * 1024 * 1024)
        out: list[DatasetExample] = []
        for line_no, line in enumerate(self._path.read_text(encoding="utf-8").splitlines(), 1):
            line = line.strip()
            if not line:
                continue
            try:
                out.append(self._parse(json.loads(line)))
            except (ValueError, KeyError, TypeError) as exc:
                raise InvalidInputError(f"invalid FEVER line {line_no}: {exc}") from exc
            if self._limit is not None and len(out) >= self._limit:
                break
        return out

    def _parse(self, obj: dict[str, Any]) -> DatasetExample:
        claim = str(obj["claim"])
        label = str(obj["label"])
        stance = _FEVER_LABELS.get(label)
        if stance is None:
            raise ValueError(f"unknown FEVER label {label!r}")
        evidence = obj.get("evidence") or []
        documents: list[Document] = []
        pages: list[str] = []
        facts: list[str] = []
        skipped_empty = 0
        for j, item in enumerate(evidence):
            # item: [page_id, sentence_index, sentence_text]
            page, sent_idx, sentence = str(item[0]), str(item[1]), str(item[2])
            if not sentence.strip():
                # FEVER occasionally ships an evidence row whose sentence text is
                # empty; skip the empty span rather than emit an invalid empty
                # Document. The count is recorded in metadata for auditability.
                skipped_empty += 1
                continue
            documents.append(
                Document(
                    document_id=f"{page}#{sent_idx}#{j}",
                    text=sentence,
                    source=SourceMetadata(source_id=page, title=page.replace("_", " ")),
                )
            )
            if page not in pages:
                pages.append(page)
            facts.append(sentence)
        return DatasetExample(
            example_id=str(obj["id"]),
            question=claim,
            documents=tuple(documents),
            gold_answers=(label,),
            gold_evidence=GoldEvidence(
                available=stance != "not_enough_info",
                source_ids=tuple(pages),
                facts=tuple(facts),
                stance=stance,  # type: ignore[arg-type]
            ),
            split=self._split,
            metadata={
                "benchmark": "fever",
                "label": label,
                "verifiable": str(obj.get("verifiable", "")),
                "original_id": str(obj.get("original_id", "")),
                "evidence_pages": json.dumps(pages),
                "num_evidence": str(len(evidence)),
                "num_evidence_skipped_empty": str(skipped_empty),
                "adapter_version": ADAPTER_VERSION,
            },
        )


class HotpotQADataset:
    """HotpotQA (cached parquet). Requires a parquet reader (pyarrow/datasets).

    The mapping is implemented; loading raises :class:`MissingDependencyError`
    until `pyarrow` is available, since the dataset is cached only as parquet and
    cannot be installed offline.
    """

    name = "hotpotqa"

    def __init__(
        self, *, split: str = "validation", path: str | Path | None = None, limit: int | None = None
    ) -> None:
        self._split = split
        self._path = path
        self._limit = limit

    def _parquet_path(self) -> Path:
        if self._path is not None:
            return Path(self._path)
        pat = str(
            _HF_HUB / "datasets--hotpotqa--hotpot_qa/snapshots/*/fullwiki/validation-*.parquet"
        )
        hits = sorted(glob.glob(pat))
        if not hits:
            raise ConfigurationError("cached HotpotQA fullwiki validation parquet not found")
        return Path(hits[0])

    def load(self) -> list[DatasetExample]:
        import importlib

        try:
            pq = importlib.import_module("pyarrow.parquet")
        except ModuleNotFoundError as exc:
            raise MissingDependencyError("pyarrow", "benchmarks") from exc
        table = pq.read_table(self._parquet_path())
        rows = table.to_pylist()
        out: list[DatasetExample] = []
        for row in rows:
            out.append(self._parse(row))
            if self._limit is not None and len(out) >= self._limit:
                break
        return out

    @staticmethod
    def _parse(row: dict[str, Any]) -> DatasetExample:
        context = row["context"]  # {"title": [...], "sentences": [[...], ...]}
        titles = context["title"]
        sent_lists = context["sentences"]
        documents: list[Document] = []
        for ti, title in enumerate(titles):
            for si, sentence in enumerate(sent_lists[ti]):
                if sentence.strip():
                    documents.append(
                        Document(
                            document_id=f"{title}#{si}",
                            text=sentence.strip(),
                            source=SourceMetadata(source_id=title, title=title),
                        )
                    )
        sp = row["supporting_facts"]  # {"title": [...], "sent_id": [...]}
        gold_pages = list(dict.fromkeys(sp["title"]))
        return DatasetExample(
            example_id=str(row["id"]),
            question=str(row["question"]),
            documents=tuple(documents),
            gold_answers=(str(row["answer"]),),
            gold_evidence=GoldEvidence(source_ids=tuple(gold_pages), stance=None),
            split="validation",
            metadata={
                "benchmark": "hotpotqa",
                "type": str(row.get("type", "")),
                "level": str(row.get("level", "")),
                "supporting_facts": json.dumps(
                    [[t, int(s)] for t, s in zip(sp["title"], sp["sent_id"], strict=False)]
                ),
                "adapter_version": ADAPTER_VERSION,
            },
        )


def dataset_fingerprint(examples: Iterable[DatasetExample]) -> str:
    digest = hashlib.sha256()
    for ex in examples:
        digest.update(ex.example_id.encode("utf-8"))
        digest.update(b"\x00")
        digest.update(ex.question.encode("utf-8"))
        digest.update(b"\x01")
    return digest.hexdigest()


def validate_benchmark(examples: list[DatasetExample]) -> list[str]:
    """Benchmark-specific integrity checks (returns human-readable issues)."""

    issues: list[str] = []
    seen: set[str] = set()
    for ex in examples:
        if ex.example_id in seen:
            issues.append(f"duplicate id: {ex.example_id!r}")
        seen.add(ex.example_id)
        if not ex.gold_answers:
            issues.append(f"missing gold label/answer: {ex.example_id!r}")
        stance = ex.gold_evidence.stance
        if stance in {"supports", "refutes"} and not ex.gold_evidence.source_ids:
            issues.append(f"{ex.example_id!r}: {stance} without evidence pages")
        for d in ex.documents:
            if not d.source.source_id:
                issues.append(f"{ex.example_id!r}: document with empty source id")
    return issues


__all__ = [
    "ADAPTER_VERSION",
    "FeverGoldEvidenceDataset",
    "HotpotQADataset",
    "dataset_fingerprint",
    "validate_benchmark",
]
