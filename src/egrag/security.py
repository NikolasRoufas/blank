"""Security utilities: safe YAML, path/URL validation, size limits, redaction.

These are *baseline* protections, documented in ``docs/implementation-status.md``.
They reduce — but do not eliminate — risk. Retrieved content is always treated
as untrusted data and is never executed; third-party model code is never loaded
by default (see the generation adapters).
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import yaml

from egrag.domain.errors import ConfigurationError, SecurityError

# Default maximum size (bytes) for a corpus/document file we are willing to read.
DEFAULT_MAX_FILE_BYTES = 10 * 1024 * 1024  # 10 MiB

_ALLOWED_URL_SCHEMES = frozenset({"http", "https"})
_SECRET_KEY_RE = re.compile(r"(?i)(api[_-]?key|token|secret|password|authorization|bearer)")
_SECRET_VALUE_RE = re.compile(r"(?i)\b(sk-[A-Za-z0-9]{8,}|bearer\s+[A-Za-z0-9._-]{8,})")
_REDACTED = "***REDACTED***"


def safe_load_yaml(text: str) -> dict[str, Any]:
    """Parse YAML using the safe loader (no arbitrary object construction).

    Raises:
        ConfigurationError: if the YAML is invalid or is not a mapping.
    """

    try:
        data = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise ConfigurationError(f"invalid YAML: {exc}") from exc
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ConfigurationError("configuration root must be a mapping")
    return data


def validate_within_base(path: str | Path, base: str | Path) -> Path:
    """Resolve ``path`` and ensure it stays within ``base`` (no traversal).

    Raises:
        SecurityError: if the resolved path escapes ``base``.
    """

    base_resolved = Path(base).resolve()
    candidate = (
        (base_resolved / Path(path)).resolve()
        if not Path(path).is_absolute()
        else Path(path).resolve()
    )
    try:
        candidate.relative_to(base_resolved)
    except ValueError as exc:
        raise SecurityError(
            f"path {str(path)!r} escapes the allowed base directory {str(base)!r}"
        ) from exc
    return candidate


def check_file_size(path: str | Path, *, max_bytes: int = DEFAULT_MAX_FILE_BYTES) -> int:
    """Return the file size, raising if it exceeds ``max_bytes``."""

    size = Path(path).stat().st_size
    if size > max_bytes:
        raise SecurityError(
            f"file {str(path)!r} is {size} bytes, exceeding the limit of {max_bytes} bytes"
        )
    return size


def validate_url(url: str, *, allowed_schemes: frozenset[str] = _ALLOWED_URL_SCHEMES) -> str:
    """Validate a URL scheme against the allow-list (default http/https only).

    Raises:
        SecurityError: for disallowed schemes (e.g. ``file://``, ``ftp://``).
    """

    parsed = urlparse(url)
    if parsed.scheme not in allowed_schemes:
        raise SecurityError(
            f"URL scheme {parsed.scheme!r} is not allowed; permitted: {sorted(allowed_schemes)}"
        )
    if not parsed.netloc:
        raise SecurityError(f"URL {url!r} has no host")
    return url


def redact_secrets(value: Any) -> Any:
    """Recursively redact secret-looking keys/values for safe logging."""

    if isinstance(value, dict):
        return {
            key: (_REDACTED if _SECRET_KEY_RE.search(str(key)) else redact_secrets(val))
            for key, val in value.items()
        }
    if isinstance(value, (list, tuple)):
        return type(value)(redact_secrets(item) for item in value)
    if isinstance(value, str):
        return _SECRET_VALUE_RE.sub(_REDACTED, value)
    return value


def safe_artifact_path(path: str | Path, base: str | Path) -> Path:
    """Validate an artifact output path is within ``base`` and return it.

    Raises:
        SecurityError: if the path escapes ``base``.
    """

    return validate_within_base(path, base)


__all__ = [
    "DEFAULT_MAX_FILE_BYTES",
    "check_file_size",
    "redact_secrets",
    "safe_artifact_path",
    "safe_load_yaml",
    "validate_url",
    "validate_within_base",
]
