# ADR 0002: Research-and-real-use hardening

- Status: accepted
- Date: 2026-06-23

## Context

The pipeline was complete end to end on fakes, but lacked a coherent
configuration system, a single composition root, durable caching,
reproducibility manifests, and documented security posture.

## Decision

- **Hierarchical, strict config.** A `pydantic`-validated `EGRagConfig` with one
  section per subsystem, loaded from **safe** YAML (`yaml.safe_load`). Unknown
  fields are **forbidden** (fail fast); cross-section consistency is validated up
  front. Provided as `configs/*.yaml` (baseline, cpu_demo, and five ablations).
- **One composition root** (`egrag.composition`) builds the pipeline and runs
  `validate_runtime` *before* any expensive processing (optional deps, model
  paths, endpoint config, context limits, evidence budgets, deterministic
  capability, adapter compatibility, cache config).
- **Content-addressed caching** (`egrag.caching`): keys include content hash,
  algorithm, model id/revision, prompt version, schema version, and relevant
  config. Disk backend uses atomic writes + checksum corruption detection +
  metrics; secrets are never cached.
- **Reproducibility manifest** (`egrag.reproducibility`) captures versions, git
  commit, resolved (secret-redacted) config, seeds, components, model
  ids/revisions, prompt versions, corpus fingerprint, timestamps, environment,
  artifact paths, warnings, and deterministic-capability status.
- **Security baseline** (`egrag.security`): safe YAML, path-traversal prevention,
  file-size limits, URL allow-list, secret redaction, validated artifact paths.
  Documented as a baseline, not a guarantee.

## Consequences

Runs are reproducible and configurable without code changes; misconfiguration
fails fast with typed errors; the determinism caveat is surfaced as a warning
rather than hidden. Deterministic decoding remains a *warning* (not a hard
failure) when an adapter cannot guarantee it, since the run can still proceed.
