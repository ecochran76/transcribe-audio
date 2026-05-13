# Policy | Git, Release, And Validation

## Policy

- Start branch-sensitive work by checking `git status`.
- Treat pre-existing dirty state as a constraint and do not revert user work unless explicitly requested.
- Keep commits focused on one coherent slice.
- Use truthful present-tense commit subjects and include rationale or operator impact in the body when the diff does not make it obvious.
- Prefer short-lived branches for implementation slices; use explicit integration steps for overlapping roadmap lanes.
- Version user-visible CLI flags, config schema changes, output formats, and runtime state schema changes deliberately.
- Update README, sample config, and API key samples in the same slice when behavior or configuration changes.
- Run targeted verification before handoff and record concrete evidence.
- Manual smoke tests remain mandatory for transcription/output changes; use mocks for API-heavy automated tests where practical.

## Closeout

Closeouts should state what changed, what was verified, and any residual risk. Prefer a concrete recommended next slice over vague follow-up language.
