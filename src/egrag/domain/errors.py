"""Domain exception hierarchy.

These exceptions belong to the domain layer and carry no infrastructure detail.
Adapters and application code should translate third-party failures into these
where a domain-level meaning exists.
"""

from __future__ import annotations


class EGRagError(Exception):
    """Base class for all EG-RAG domain errors."""


class ValidationError(EGRagError):
    """Raised when domain data fails validation outside of Pydantic construction."""


class SerializationError(EGRagError):
    """Raised when serialized evidence data cannot be produced or parsed safely."""


class ExtractionError(EGRagError):
    """Raised when model output for claim extraction cannot be parsed or validated.

    This is a recoverable error: callers may catch it, log it, and continue with
    the remaining passages rather than aborting the whole run.
    """


class GraphValidationError(EGRagError):
    """Raised when an evidence graph fails structural validation.

    Causes include dangling edges, duplicate node IDs, invalid relation types,
    invalid confidence values, unsupported self-edges, and malformed provenance
    references.
    """


class ConvergenceError(EGRagError):
    """Raised when belief propagation does not converge within the iteration limit."""


class GenerationError(EGRagError):
    """Raised when generation output is empty, malformed, or fails validation.

    This is a recoverable, typed error: callers can surface it rather than
    silently accepting invalid attribution or empty answers.
    """


class PipelineError(EGRagError):
    """Raised when the application pipeline cannot complete a run."""


class ConfigurationError(EGRagError):
    """Raised when configuration is invalid or inconsistent."""


class InvalidInputError(EGRagError):
    """Raised when caller-supplied input is invalid."""


class MissingDependencyError(EGRagError):
    """Raised when an optional dependency is required but not installed.

    The message includes the install command for the relevant extra so the
    failure is actionable.
    """

    def __init__(self, dependency: str, extra: str) -> None:
        self.dependency = dependency
        self.extra = extra
        super().__init__(
            f"optional dependency {dependency!r} is not installed; "
            f"install it with: uv pip install 'egrag[{extra}]'"
        )


class ModelLoadError(EGRagError):
    """Raised when a model or its path/revision cannot be loaded."""


class RetrievalError(EGRagError):
    """Raised when retrieval cannot complete."""


class EndpointError(EGRagError):
    """Raised when a remote endpoint request fails (non-timeout)."""


class GenerationTimeoutError(EGRagError):
    """Raised when generation exceeds its configured timeout."""


class CacheCorruptionError(EGRagError):
    """Raised when a cache entry is detected to be corrupted."""


class ArtifactWriteError(EGRagError):
    """Raised when an artifact cannot be written to a validated path."""


class SecurityError(EGRagError):
    """Raised when a security policy is violated (path, URL, size, secret)."""


__all__ = [
    "ArtifactWriteError",
    "CacheCorruptionError",
    "ConfigurationError",
    "ConvergenceError",
    "EGRagError",
    "EndpointError",
    "ExtractionError",
    "GenerationError",
    "GenerationTimeoutError",
    "GraphValidationError",
    "InvalidInputError",
    "MissingDependencyError",
    "ModelLoadError",
    "PipelineError",
    "RetrievalError",
    "SecurityError",
    "SerializationError",
    "ValidationError",
]
