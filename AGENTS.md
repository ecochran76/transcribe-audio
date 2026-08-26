# Repository Guidelines

## Product vision

- [VISION.md](VISION.md) is the canonical product north star for this repo.
- Read it before non-trivial planning, architecture, prioritization, or goal
  execution.
- Every substantive plan must state which vision outcomes it advances, the
  current and target maturity levels, and the evidence that will measure
  progress.
- If a bounded milestone or implementation detail conflicts with the vision,
  reconcile the conflict explicitly instead of silently narrowing the product
  objective.

## Project Structure & Module Organization
- The transcription entry points are `assembly_transcribe.py`, `faster_whisper_transcribe.py`, and `watch_transcriptions.py`; shared export, calendar, and formatting behavior belongs in `transcribe_common.py`.
- Keep new modules in the repo root until a larger `src/` layout is justified in `ROADMAP.md` and a bounded plan under `docs/dev/plans/`.
- Config artifacts sit alongside the script: `requirements.txt` lists runtime deps and `api_keys.json.sample` describes required secrets. Real keys belong in the ignored `api_keys.json`.
- Add sample assets only when essential for testing; prefer short clips under `tests/data/` and link to heavier media externally.

## Build, Test, and Development Commands
- Create a virtual environment: `python -m venv .venv && source .venv/bin/activate` (Windows: `.venv\Scripts\activate`).
- Install deps with `pip install -r requirements.txt`; this pulls in `requests`, `python-docx`, and Google Calendar client libraries needed for optional calendar metadata.
- Exercise the CLI via `python assembly_transcribe.py demo.wav --text-output` (add `--use-calendar` for calendar tests) and note that patterns such as `python assembly_transcribe.py "~/Downloads/*.m4a"` or `python assembly_transcribe.py "C:\\Calls\\*.mp3"` are expanded by the script itself.

## Coding Style & Naming Conventions
- Follow PEP 8 with 4-space indentation and snake_case identifiers; continue using postponed annotations and pathlib for filesystem work.
- Limit user feedback to succinct `print()` calls unless structured logging provides clear value to CLI users.
- Encapsulate AssemblyAI calls in small helpers (`upload_audio`, `poll_transcript`) and accept dependencies as parameters to simplify testing.

## Testing Guidelines
- Manual smoke tests are mandatory: run the CLI on short audio clips to verify DOCX (and optional TXT) output after every change.
- For automated coverage, create a `tests/` package powered by `pytest`; mock AssemblyAI endpoints using `responses`, `httpx_mock`, or similar tools.
- Summarize manual test commands and scenarios in PRs, noting any API settings touched (chunk size, polling interval, diarization flags).

## Commit & Pull Request Guidelines
- Use present-tense subjects ≤72 characters (e.g., `Add text transcript flag`) and add rationale, sample commands, and API notes in the body.
- PR descriptions should explain the user impact, enumerate test evidence, and highlight adjustments that influence API usage or quotas.
- When formatting changes, drop before/after snippets or DOCX screenshots so reviewers can spot regressions.

## Configuration & Secrets
- Store AssemblyAI keys in env vars or `api_keys.json` (ignored). Mirror any new fields in `api_keys.json.sample` and document them in the README to prevent drift.
- Google Calendar access uses `credentials.json` (OAuth client) and a generated `token.json`; both are ignored by Git. Document any new scopes or calendar-related flags when they change.
- Prefer CLI flags for behavior tweaks; add config files only when options multiply, and describe resolution order clearly in documentation.

## Policy Loading Contract

- `AGENTS.md` is a routing surface, not a one-time pointer.
- Re-read the relevant policy files under `docs/dev/policies/` at the start of any non-trivial turn.
- Re-read the relevant policy files when task scope changes mid-session.
- When behavior is ambiguous, prefer re-reading policy over improvising from stale assumptions.

## Policy Entry

This repo keeps its durable repo-local policy under `docs/dev/policies/`.

Read and follow:
- `docs/dev/policies/0001-policy-management.md`
- `docs/dev/policies/0002-planning-roadmap-runbook.md`
- `docs/dev/policies/0003-runtime-tenant-state.md`
- `docs/dev/policies/0004-architecture-productization.md`
- `docs/dev/policies/0005-memory-and-context-routing.md`
- `docs/dev/policies/0006-git-release-validation.md`
- `docs/dev/policies/0007-codegraph-usage.md`
- `docs/dev/policies/0008-policy-upgrade-management.md`
- `docs/dev/policies/0009-policy-adoption-feedback-loop.md`
- `docs/dev/policies/0010-notes-and-memories.md`
- `docs/dev/policies/0011-planning-discipline.md`
- `docs/dev/policies/0012-parallel-plan-design.md`
- `docs/dev/policies/0013-git-worktree-hygiene.md`
- `docs/dev/policies/0014-commit-history-discipline.md`
- `docs/dev/policies/0015-commit-and-push-cadence.md`
- `docs/dev/policies/0016-multi-agent-reconciliation.md`
- `docs/dev/policies/0017-subagent-workflow-optimization.md`
- `docs/dev/policies/0018-turn-closeout.md`
- `docs/dev/policies/0019-policy-harvest-loop.md`
- `docs/dev/policies/0020-subagent-runtime-governance.md`
- `docs/dev/policies/0021-preview-artifact-review.md`
- `docs/dev/policies/0022-code-testing-discipline.md`
- `docs/dev/policies/0023-policy-management.md`
- `docs/dev/policies/0024-policy-upgrade-management.md`
- `docs/dev/policies/0025-policy-adoption-feedback-loop.md`
- `docs/dev/policies/0026-notes-and-memories.md`
- `docs/dev/policies/0027-graph-backed-memory-usage.md`
- `docs/dev/policies/0028-codegraph-usage.md`
- `docs/dev/policies/0029-code-testing-discipline.md`
- `docs/dev/policies/0030-planning-discipline.md`
- `docs/dev/policies/0031-goal-execution-governance.md`
- `docs/dev/policies/0032-parallel-plan-design.md`
- `docs/dev/policies/0033-git-worktree-hygiene.md`
- `docs/dev/policies/0034-commit-history-discipline.md`
- `docs/dev/policies/0035-branch-and-integration-strategy.md`
- `docs/dev/policies/0036-commit-and-push-cadence.md`
- `docs/dev/policies/0037-multi-agent-reconciliation.md`
- `docs/dev/policies/0038-subagent-workflow-optimization.md`
- `docs/dev/policies/0039-versioning-and-release.md`
- `docs/dev/policies/0040-turn-closeout.md`
- `docs/dev/policies/0041-policy-harvest-loop.md`
- `docs/dev/policies/0042-validation-and-handoff.md`
- `docs/dev/policies/0043-subagent-runtime-governance.md`
- `docs/dev/policies/0044-preview-artifact-review.md`
- `docs/dev/policies/0045-active-lane-coordination.md`

## Graphiti Memory Discovery
- Use the `graphiti-discovery` skill at the start of non-trivial planning, debugging, architecture, routing, memory, or handoff work.
- Query repo group `transcribe_audio_main` before assuming prior context exists only in chat history.
- Treat Graphiti as advisory; verify cited facts against repo files, artifacts, commits, tests, or source episodes before changing code or live systems.
- When bootstrapping or refreshing repo memory, harvest from `ROADMAP.md`, `RUNBOOK.md`, `docs/dev/plans/`, `docs/dev/policies/`, and validated artifacts only.
- Do not seed secrets, raw private data, raw transcripts, full logs, or unreviewed speculation.

## Policy Re-read Triggers

- re-read planning-related policy before opening, revising, or closing a substantive plan
- re-read documentation-related policy before changing docs, contracts, or canonical authorities
- re-read validation and closeout policy before claiming work complete
- re-read branch, commit, and integration policy before starting a multi-file or multi-step implementation slice

## Scope

- `AGENTS.md` includes repo-local guidance plus the policy entry section.
- The durable policy body lives under `docs/dev/policies/`.
- Keep repo-specific commands, environment details, and operational caveats in this file or adjacent local docs.
