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
- Run the relevant validation for the touched surface before commit, handoff, or merge preparation.
- Prefer targeted verification that matches the changed area, and widen to broader suites when the impact is user-visible or cross-cutting.
- Include concrete pass/fail evidence in the handoff or closeout note.
- Keep handoff notes concise, explicit about remaining risk, and clear about the next recommended action.
- When live or manual smoke matters for the changed surface, record whether it was run and what it proved.
- Prefer validation receipts that bind the result to a durable commit, artifact,
  installed version, endpoint response, or other current-state identifier.
  Temporary paths alone are not durable handoff evidence; preserve or publish
  the necessary artifact in a repo-approved location, or record why the proof
  is intentionally ephemeral and how it can be reproduced.
- Distinguish validation run by the primary agent from validation reported by a subagent or delegated worker.
- If validation was delegated, record whether the primary agent independently verified the result or accepted the delegated evidence as-is.
- For failed, timed-out, incomplete, or unknown subagent statuses, state what was trusted, what was ignored, and what remains unverified.
- For long-running, high-risk, or consequential work, separate implementation
  from final judgment by using an independent evaluator with fresh context and
  explicit acceptance criteria.
- Treat evaluator output as candidate evidence, not an automatic veto. The
  primary agent owns adjudication and records each candidate as `blocking`,
  `nonblocking_backlog`, `rejected`, or `needs_evidence` against the frozen
  objective, acceptance criteria, non-goals, and applicable safety controls.
- Require each candidate finding to state the criterion, evidence, consequence,
  reproducer, confidence, and suggested disposition. A useful independent
  review may return no findings; novelty and finding count are not quality
  metrics.
- Separate review modes. Use one broad fresh-context `drift_discovery` pass to
  find material divergence. After adjudication, use `closed_world` remediation
  verification limited to accepted blocking findings and critical regressions
  introduced by their fixes. Do not reopen broad discovery merely because a
  new evaluator performs final verification.
- Bound review and rework at the goal level, not only per plan version. Prefer
  one consolidated candidate set and one bounded remediation pass; if accepted
  blocking findings still fail verification, split, reframe, or block the unit
  instead of continuing an open-ended evaluator/optimizer loop. Record
  nonblocking concerns in backlog without silently expanding the active plan.
- Validate the resulting outcome and current external state, not only the
  transcript, diff shape, test count, or agent's narrative of progress.
- Treat fail-closed gates as successful policy execution when they prevent an
  unsafe or disproven change from integrating. Report the blocked outcome and
  evidence instead of grading effectiveness only by shipped changes.

## Closeout

Closeouts should state what changed, what was verified, and any residual risk. Prefer a concrete recommended next slice over vague follow-up language.
