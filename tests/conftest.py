"""Shared test fixtures and global network isolation.

A session-wide autouse fixture disables outbound network access so that any
accidental network call during a test fails loudly. Tests must run offline.
"""

from __future__ import annotations

import os
import socket
from collections.abc import Iterator

import pytest


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Keep the ordinary suite lightweight: skip ``requires_local_models`` tests
    (which load a real model) unless ``EGRAG_RUN_LOCAL_MODELS=1`` is set.

    They run as a separate explicit gate (``EGRAG_RUN_LOCAL_MODELS=1 pytest -m
    requires_local_models``), so the default ``pytest`` never loads transformers/
    torch — which also keeps the import-isolation / no-optional-libs assertions
    valid now that the extra is installed.
    """

    if os.environ.get("EGRAG_RUN_LOCAL_MODELS") == "1":
        return
    skip = pytest.mark.skip(reason="needs EGRAG_RUN_LOCAL_MODELS=1 (loads a real model)")
    for item in items:
        if "requires_local_models" in item.keywords:
            item.add_marker(skip)


class _NetworkBlockedError(RuntimeError):
    """Raised when test code attempts to open a network connection."""


@pytest.fixture(autouse=True, scope="session")
def _block_network() -> Iterator[None]:
    """Block real socket creation for the duration of the test session."""

    original_socket = socket.socket
    original_create_connection = socket.create_connection

    def _blocked_socket(*args: object, **kwargs: object) -> object:
        raise _NetworkBlockedError("network access is disabled during tests")

    def _blocked_create_connection(*args: object, **kwargs: object) -> object:
        raise _NetworkBlockedError("network access is disabled during tests")

    socket.socket = _blocked_socket  # type: ignore[assignment, misc]
    socket.create_connection = _blocked_create_connection  # type: ignore[assignment]
    try:
        yield
    finally:
        socket.socket = original_socket  # type: ignore[misc]
        socket.create_connection = original_create_connection
