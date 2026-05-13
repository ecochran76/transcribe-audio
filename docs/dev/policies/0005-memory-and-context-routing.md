# Policy | Memory And Context Routing

## Policy

- Treat graph-backed memory as durable retrievable context, not a scratchpad.
- Use graph memory for compact, stable facts: people, organizations, matters, recurring routing decisions, durable user preferences, and cross-turn project decisions.
- Do not store secrets, raw credentials, raw transcripts, transient command output, or unreviewed private content in graph memory.
- Treat memory-derived claims as advisory until verified against transcript artifacts, calendar records, repository files, or cited episodes.
- Keep route decisions auditable: record candidate repositories, confidence, evidence, rejected alternatives, and fallback behavior.
- Low-confidence routing must go to a review queue rather than depositing into a guessed repository.
- Contextual rereads should cite which supporting context was used.
- Harvest Graphiti/OpenClaw memories only from structured readout fields that were designed for memory candidates.

## Local Routing Targets

Supported targets may include local folders, Google Drive resources, Odoo records, Graphiti/OpenClaw entities, or a review queue. New target types require an active plan and a depositor contract before unattended writes are enabled.

## Repo Memory Group

- Repo-scoped Graphiti memory group: `transcribe_audio_main`.
- Use the `graphiti-discovery` skill before non-trivial planning, debugging, architecture, routing, memory, or handoff work.
- Refresh repo memory when a roadmap lane materially changes, a bounded plan closes, a runtime contract changes, or a live smoke proves or disproves an operational assumption.
- Seed only curated source-backed facts from `ROADMAP.md`, `RUNBOOK.md`, `docs/dev/plans/`, `docs/dev/policies/`, and validated artifacts.
