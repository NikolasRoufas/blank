# ADR 0001: Record architecture decisions

- Status: accepted
- Date: 2026-06-22

## Context

EG-RAG is a modular, generator-agnostic RAG framework with a strict layering
discipline. We want a lightweight, durable record of significant architectural
choices.

## Decision

We keep Architecture Decision Records as short Markdown files under `docs/adr/`.
Each ADR has a status, date, context, decision, and consequences.

Foundational decisions captured elsewhere and summarized here:

- **Pure domain layer.** `egrag.domain` imports only the standard library and
  Pydantic. Infrastructure (NetworkX, Transformers, sentence-transformers, HTTP,
  storage) lives behind adapters and optional extras.
- **Ports & adapters.** Every model-facing capability is a typed protocol;
  concrete adapters implement it and load optional dependencies lazily.
- **Generator-agnostic via standardized adapters** — not "works with every
  model".
- **Versioned, JSON-serializable domain models** with an explicit
  `SCHEMA_VERSION`.
- **Deterministic, offline, CPU-only defaults**; tests never use the network or
  download models.

## Consequences

A core-only install is fully runnable and testable; optional capabilities are
additive; provider details never leak into domain or application code.
