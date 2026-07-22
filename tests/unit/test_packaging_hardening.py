"""Packaging / isolation acceptance tests (22, 23, 24, 25, 30)."""

from __future__ import annotations

import importlib
import sys

import pytest

from egrag.domain.errors import ConfigurationError, MissingDependencyError

_OPTIONAL_LIBS = {"torch", "transformers", "httpx", "sentence_transformers", "networkx", "numpy"}
_CORE_MODULES = [
    "egrag",
    "egrag.config",
    "egrag.composition",
    "egrag.answering",
    "egrag.generation",
    "egrag.graph",
    "egrag.reasoning",
    "egrag.adapters.retrieval",
    "egrag.adapters.extraction",
    "egrag.adapters.reranking",
    "egrag.security",
    "egrag.caching",
    "egrag.reproducibility",
]


@pytest.mark.unit
def test_typed_exceptions_preserve_context() -> None:
    """Acceptance 22: typed exceptions preserve useful context."""

    error = MissingDependencyError("httpx", "http-models")
    assert error.dependency == "httpx"
    assert "egrag[http-models]" in str(error)

    from egrag.config import config_from_yaml

    with pytest.raises(ConfigurationError) as excinfo:
        config_from_yaml("a: [unclosed")
    assert excinfo.value.__cause__ is not None  # original cause preserved via `raise ... from`


@pytest.mark.unit
def test_core_only_import() -> None:
    """Acceptance 23 & 25: core modules import without optional dependencies.

    Restores ``sys.modules`` afterward so re-importing egrag does not create
    duplicate class objects that would break other tests in the session.
    """

    saved = dict(sys.modules)
    try:
        for name in [m for m in sys.modules if m == "egrag" or m.startswith("egrag.")]:
            del sys.modules[name]
        for optional in _OPTIONAL_LIBS:
            sys.modules.pop(optional, None)
        for module in _CORE_MODULES:
            importlib.import_module(module)
        leaked = _OPTIONAL_LIBS & set(sys.modules)
        assert not leaked, f"core import pulled in optional libraries: {sorted(leaked)}"
    finally:
        sys.modules.clear()
        sys.modules.update(saved)


@pytest.mark.integration
def test_core_only_runs_fake_pipeline_without_optional_libs() -> None:
    """Acceptance 24 & 30: the fake pipeline runs offline with no optional libs.

    Starts from a clean ``sys.modules`` (egrag and optional libs purged, then
    restored) so the leak check reflects what *this* pipeline run imports, not
    what other tests in the session already loaded (e.g. ``requires_dense``).
    """

    saved = dict(sys.modules)
    try:
        for name in [m for m in sys.modules if m == "egrag" or m.startswith("egrag.")]:
            del sys.modules[name]
        for optional in _OPTIONAL_LIBS:
            sys.modules.pop(optional, None)

        from egrag.answering import answer_query
        from egrag.domain.models import Query
        from egrag.fakes import build_demo_documents

        result = answer_query(Query(query_id="q", text="evidence graph"), build_demo_documents())
        assert result.answer.text
        leaked = _OPTIONAL_LIBS & set(sys.modules)
        assert not leaked, f"fake pipeline pulled in optional libraries: {sorted(leaked)}"
    finally:
        sys.modules.clear()
        sys.modules.update(saved)
