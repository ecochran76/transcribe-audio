# Policy | Policy Management

## Adopted Profile

This repo uses an operations-platform policy composition, with repo-specific overrides for a Python CLI/service codebase. The deterministic selector initially classified the current repo as `standalone-library`, but the active target state is a tenant-aware meeting-intelligence operations platform with runtime state, provider routing, graph-backed context, and deposition workflows.

## Policy

- Keep durable repo-local policy under `docs/dev/policies/`.
- Keep `AGENTS.md` as the policy-loading entrypoint and the home for repo-specific commands and constraints.
- Re-read the relevant policy files at the start of non-trivial work and when scope changes.
- Treat policy adoption and upgrades as deliberate maintenance, not copy/paste drift.
- When shared policy changes are reviewed, record what was adopted, deferred, or overridden in `RUNBOOK.md` or a dated note under `docs/dev/notes/`.
- Preserve repo-local nuance when shared profile guidance is too generic.

## Source

Adopted from `repo-policy-selector` bundle `v0.1.13`, using the `operations-platform` profile as the closest target-state fit.
