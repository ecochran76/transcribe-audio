# Policy | Memory And Context Routing

## Policy

- Treat graph-backed memory as durable retrievable context, not a scratchpad.
- Use graph memory for compact, stable facts: people, organizations, matters, recurring routing decisions, durable user preferences, and cross-turn project decisions.
- Do not store secrets, raw credentials, raw transcripts, raw logs, transient command output, raw reasoning traces, or unreviewed private content in graph memory.
- Treat memory-derived claims as advisory until verified against transcript artifacts, calendar records, repository files, or cited episodes.
- Prefer one compact, source-backed memory over repeated near-duplicate entries; update durable state when a fact changes rather than narrating every intermediate step.
- Treat destructive Graphiti or memory-maintenance tools as explicit cleanup or repair operations, not routine discovery commands.
- Keep route decisions auditable: record candidate repositories, confidence, evidence, rejected alternatives, and fallback behavior.
- Low-confidence routing must go to a review queue rather than depositing into a guessed repository.
- Contextual rereads should cite which supporting context was used.
- Harvest Graphiti/OpenClaw memories only from structured readout fields that were designed for memory candidates.

## Local Routing Targets

Supported targets may include local folders, Google Drive resources, Odoo records, Graphiti/OpenClaw entities, or a review queue. New target types require an active plan and a depositor contract before unattended writes are enabled.

## Repo Memory Group

- Repo-scoped Graphiti memory group: `transcribe_audio_main`.
- Use the `graphiti-discovery` skill before non-trivial planning, debugging, architecture, routing, memory, or handoff work.
- Use `~/.local/bin/graphiti-runtime doctor` when Graphiti availability or MCP health matters.
- Query `transcribe_audio_main` first for repo-scoped memory. If the right group is unclear or the task crosses repos, tenants, or domains, use a reviewed atlas/routing layer before descending into source groups.
- Refresh repo memory when a roadmap lane materially changes, a bounded plan closes, a runtime contract changes, or a live smoke proves or disproves an operational assumption.
- Seed only curated source-backed facts from `ROADMAP.md`, `RUNBOOK.md`, `docs/dev/plans/`, `docs/dev/policies/`, and validated artifacts.

## Source

Adopts the shared `graph-backed-memory-usage` module with repo-local Graphiti
group `transcribe_audio_main`.
