"""Isolation tests for retrieval adapters (acceptance 21, 22, 23)."""

from __future__ import annotations

import importlib
import sys

import pytest

OPTIONAL_LIBS = {"sentence_transformers", "torch", "numpy", "rank_bm25"}

RETRIEVAL_MODULES = [
    "egrag.adapters.retrieval",
    "egrag.adapters.retrieval.base",
    "egrag.adapters.retrieval.bm25",
    "egrag.adapters.retrieval.dense",
    "egrag.adapters.retrieval.hybrid",
    "egrag.adapters.retrieval.chunking",
    "egrag.adapters.retrieval.tokenization",
    "egrag.adapters.reranking",
    "egrag.adapters.reranking.score",
    "egrag.adapters.reranking.cross_encoder",
]


def _leaked_optional_in_subprocess(setup_code: str, optional: set[str]) -> list[str]:
    """Run ``setup_code`` in a fresh interpreter; return which ``optional`` libs
    landed in ``sys.modules``. A subprocess makes the check independent of what
    the surrounding pytest session already imported (e.g. ``requires_dense`` tests
    loading sentence-transformers in-process)."""

    import json
    import subprocess

    code = (
        "import sys, json\n"
        f"{setup_code}\n"
        f"_opt = {sorted(optional)!r}\n"
        "print(json.dumps(sorted(set(_opt) & set(sys.modules))))\n"
    )
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert proc.returncode == 0, f"isolation subprocess failed:\n{proc.stderr}"
    return list(json.loads(proc.stdout.strip().splitlines()[-1]))


@pytest.mark.unit
def test_importing_retrieval_adapters_loads_no_optional_libs() -> None:
    """Acceptance 22 & 23: importing retrieval adapters pulls in no model libs."""

    setup = f"import importlib\nfor _m in {RETRIEVAL_MODULES!r}:\n    importlib.import_module(_m)"
    leaked = _leaked_optional_in_subprocess(setup, OPTIONAL_LIBS)
    assert not leaked, f"retrieval import leaked optional libraries: {leaked}"


@pytest.mark.unit
def test_core_only_retrieval_stack_runs_without_extras() -> None:
    """Acceptance 23: the full core retrieval stack runs with no optional deps.

    Uses BM25 (pure Python), dense retrieval with the deterministic fake
    embedder, and hybrid fusion — none of which require an extra.
    """

    setup = "\n".join(
        [
            "from egrag.adapters.retrieval import (BM25Retriever, DenseRetriever, "
            "HybridRetriever, SentenceAwareChunker, prepare_passages)",
            "from egrag.domain.models import Query",
            "from egrag.fakes import FakeEmbeddingProvider, build_demo_documents",
            "passages = prepare_passages(build_demo_documents(), SentenceAwareChunker())",
            "bm25 = BM25Retriever(passages)",
            "dense = DenseRetriever(passages, FakeEmbeddingProvider(dim=32))",
            "hybrid = HybridRetriever({'bm25': bm25, 'dense': dense})",
            "q = Query(query_id='q', text='evidence graph retrieval')",
            "results = hybrid.search(q, top_k=3)",
            "assert results, 'core retrieval stack returned no results'",
        ]
    )
    leaked = _leaked_optional_in_subprocess(setup, OPTIONAL_LIBS)
    assert not leaked, f"core retrieval stack leaked optional libraries: {leaked}"


@pytest.mark.unit
def test_reimport_under_network_block_succeeds() -> None:
    """Acceptance 21: retrieval modules import with the network blocked.

    The session-wide fixture blocks sockets; a fresh re-import proves the import
    path makes no network call.
    """

    for module in RETRIEVAL_MODULES:
        sys.modules.pop(module, None)
    try:
        for module in RETRIEVAL_MODULES:
            importlib.import_module(module)
    finally:
        for module in RETRIEVAL_MODULES:
            importlib.import_module(module)
