# Policy | Planning, Roadmap, And Runbook

## Policy

- Use `ROADMAP.md` as the master plan and priority map.
- Use `RUNBOOK.md` as the dated turn-by-turn execution log.
- Use `docs/dev/plans/` for bounded implementation slices.
- Plan filenames must use `NNNN-YYYY-MM-DD-plan-slug.md`.
- Each active plan must include `State`, `Scope`, `Non-Goals`, `Current State`, `Acceptance Criteria`, and `Validation`.
- Use a small plan-state vocabulary: `PLANNED`, `OPEN`, `CLOSED`, `CANCELLED`.
- Wire every active plan into `ROADMAP.md`; record material plan creation, closure, or reprioritization in `RUNBOOK.md`.
- Do not materially rename, reorder, or reprioritize roadmap lanes without making the priority decision explicit.
- Close or split plans before they become endless catch-all documents.

## Local Planning Contract

- Roadmap lanes use `P## | <Lane Title>`.
- Immediate implementation work should live in a numbered plan file, not only in `docs/platform-expansion-plan.md`.
- `docs/platform-expansion-plan.md` is retained as background architecture notes; it is not the planning authority after this adoption.
