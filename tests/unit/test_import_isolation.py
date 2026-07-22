"""Import-isolation and layering tests (acceptance cases 10, 11, 12, 15, 16)."""

from __future__ import annotations

import ast
import builtins
import importlib
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import pytest

import egrag.domain

# Third-party infrastructure packages the domain layer must never import.
FORBIDDEN_INFRA = {
    "networkx",
    "transformers",
    "sentence_transformers",
    "torch",
    "httpx",
    "requests",
    "openai",
    "sqlalchemy",
    "faiss",
    "chromadb",
    "numpy",
    "pandas",
    "rank_bm25",
}

# Sibling layers the domain must not depend on (dependency direction).
FORBIDDEN_INTERNAL_PREFIXES = (
    "egrag.adapters",
    "egrag.application",
    "egrag.cli",
    "egrag.fakes",
    "egrag.config",
    "egrag.caching",
    "egrag.serialization",
    "egrag.observability",
)

# Optional model/infra libraries that core code must not pull in.
OPTIONAL_LIBS = {
    "torch",
    "transformers",
    "sentence_transformers",
    "networkx",
    "rank_bm25",
    "httpx",
    "numpy",
}

CORE_MODULES = [
    "egrag",
    "egrag.domain.models",
    "egrag.domain.ports",
    "egrag.domain.graph",
    "egrag.domain.errors",
    "egrag.application.pipeline",
    "egrag.serialization",
    "egrag.config",
    "egrag.caching",
    "egrag.observability",
    "egrag.fakes",
    "egrag.cli.main",
]


def _domain_files() -> list[Path]:
    root = Path(egrag.domain.__file__).parent
    return sorted(root.rglob("*.py"))


def _imported_roots(tree: ast.AST) -> set[str]:
    roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                roots.add(alias.name)
        elif isinstance(node, ast.ImportFrom) and node.module is not None and node.level == 0:
            roots.add(node.module)
    return roots


@pytest.mark.unit
def test_domain_does_not_import_infrastructure() -> None:
    """Acceptance 16: domain modules import no infrastructure packages."""

    for path in _domain_files():
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for module in _imported_roots(tree):
            top = module.split(".")[0]
            assert top not in FORBIDDEN_INFRA, f"{path.name} imports infrastructure {module!r}"
            assert not module.startswith(FORBIDDEN_INTERNAL_PREFIXES), (
                f"{path.name} imports sibling layer {module!r}"
            )


def _purge_egrag_modules() -> None:
    for name in [m for m in sys.modules if m == "egrag" or m.startswith("egrag.")]:
        del sys.modules[name]


@contextmanager
def _fresh_import_then_restore() -> Iterator[None]:
    """Purge egrag so the body re-executes module code, then restore originals.

    Restoring the exact ``sys.modules`` snapshot returns the original module
    objects (and their class identities) so that other tests' collection-time
    imports remain valid — re-importing fresh would create duplicate classes.
    """

    saved = dict(sys.modules)
    _purge_egrag_modules()
    try:
        yield
    finally:
        sys.modules.clear()
        sys.modules.update(saved)


@pytest.mark.unit
def test_import_does_not_access_network() -> None:
    """Acceptance 11: re-importing egrag under the network block succeeds.

    The session-wide fixture blocks sockets; a clean re-import that re-executes
    module code proves nothing reaches the network at import time.
    """

    with _fresh_import_then_restore():
        for module in CORE_MODULES:
            importlib.import_module(module)


@pytest.mark.unit
def test_import_does_not_read_files() -> None:
    """Acceptance 12: importing egrag reads no corpus or file from disk."""

    real_open = builtins.open

    def _blocked_open(*args: object, **kwargs: object) -> Iterator[None]:
        raise AssertionError("import-time file access is not allowed")

    with _fresh_import_then_restore():
        builtins.open = _blocked_open  # type: ignore[assignment]
        try:
            for module in CORE_MODULES:
                importlib.import_module(module)
        finally:
            builtins.open = real_open


def _leaked_optional_in_subprocess(setup_code: str, optional: set[str]) -> list[str]:
    """Run ``setup_code`` in a fresh interpreter and return which ``optional``
    libraries ended up in ``sys.modules``.

    A subprocess is used so the check is independent of what the surrounding
    pytest session already imported (e.g. ``requires_dense`` tests importing
    sentence-transformers in-process). This measures the real property — a clean
    import of core code pulls in no optional library — regardless of session state.
    """

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
def test_import_does_not_initialize_optional_models() -> None:
    """Acceptance 10 & 15: importing core code pulls in no optional model libs."""

    setup = f"import importlib\nfor _m in {CORE_MODULES!r}:\n    importlib.import_module(_m)"
    leaked = _leaked_optional_in_subprocess(setup, OPTIONAL_LIBS)
    assert not leaked, f"core import leaked optional libraries: {leaked}"
