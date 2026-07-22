"""Security acceptance tests (15, 16, 17, 18, 21)."""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from egrag.domain.errors import SecurityError
from egrag.observability import document_log_fields, log_event
from egrag.security import (
    check_file_size,
    redact_secrets,
    safe_artifact_path,
    validate_url,
    validate_within_base,
)


@pytest.mark.unit
def test_secrets_redacted() -> None:
    """Acceptance 15: secrets are redacted from logs/structured fields."""

    payload = {"api_key": "sk-secret123456789", "model": "gpt", "nested": {"token": "abc"}}
    redacted = redact_secrets(payload)
    assert redacted["api_key"] == "***REDACTED***"
    assert redacted["nested"]["token"] == "***REDACTED***"
    assert redacted["model"] == "gpt"
    # also redacted inside a free-text value
    assert "sk-secret123456789" not in redact_secrets("authorization: sk-secret123456789")


@pytest.mark.unit
def test_secrets_redacted_in_log_event(caplog: pytest.LogCaptureFixture) -> None:
    """Acceptance 15: log_event does not emit secret values."""

    logger = logging.getLogger("egrag.test")
    with caplog.at_level(logging.INFO, logger="egrag.test"):
        message = log_event(logger, "call", api_key="sk-supersecretvalue", endpoint="http://x")
    assert "sk-supersecretvalue" not in message
    assert "sk-supersecretvalue" not in caplog.text


@pytest.mark.unit
def test_path_traversal_rejected(tmp_path: Path) -> None:
    """Acceptance 16: path traversal attempts are rejected."""

    with pytest.raises(SecurityError):
        validate_within_base("../../etc/passwd", tmp_path)
    with pytest.raises(SecurityError):
        validate_within_base("/etc/passwd", tmp_path)
    # a legitimate relative path inside the base is allowed
    assert validate_within_base("sub/file.txt", tmp_path).is_relative_to(tmp_path.resolve())


@pytest.mark.unit
def test_oversized_files_rejected(tmp_path: Path) -> None:
    """Acceptance 17: oversized files are rejected per configuration."""

    big = tmp_path / "big.txt"
    big.write_text("x" * 2048, encoding="utf-8")
    with pytest.raises(SecurityError):
        check_file_size(big, max_bytes=1024)
    assert check_file_size(big, max_bytes=4096) == 2048


@pytest.mark.unit
def test_unsafe_urls_rejected() -> None:
    """Acceptance 18: unsafe URLs follow the documented allow-list policy."""

    assert validate_url("https://api.example.com/v1") == "https://api.example.com/v1"
    for bad in ("file:///etc/passwd", "ftp://host/x", "data:text/plain,abc"):
        with pytest.raises(SecurityError):
            validate_url(bad)
    with pytest.raises(SecurityError):
        validate_url("https://")  # no host


@pytest.mark.unit
def test_artifact_path_must_be_within_base(tmp_path: Path) -> None:
    with pytest.raises(SecurityError):
        safe_artifact_path("../escape.json", tmp_path)
    assert safe_artifact_path("out/run.json", tmp_path).is_relative_to(tmp_path.resolve())


@pytest.mark.unit
def test_private_documents_not_logged_by_default() -> None:
    """Acceptance 21: private document content is not logged by default."""

    text = "Confidential: the secret merger closes Friday."
    default_fields = document_log_fields(text)
    assert default_fields == {"chars": len(text)}
    assert "secret merger" not in str(default_fields)
    # only when explicitly enabled is a preview included
    opted_in = document_log_fields(text, log_documents=True)
    assert "preview" in opted_in
