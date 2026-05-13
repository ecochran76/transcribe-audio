# Repository Guidelines

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
- `AGENTS.md` is the entrypoint; durable policy lives under `docs/dev/policies/`.
- Re-read relevant policy files at the start of non-trivial planning, architecture, runtime, tenant, memory, routing, or release work.
- Re-read relevant policy files whenever the task scope changes mid-session.
- Use `ROADMAP.md` as the master priority map, `RUNBOOK.md` as the dated execution log, and `docs/dev/plans/` for bounded implementation slices.

## Policy Entry
Read and follow these repo-local policies as applicable:
- `docs/dev/policies/0001-policy-management.md`
- `docs/dev/policies/0002-planning-roadmap-runbook.md`
- `docs/dev/policies/0003-runtime-tenant-state.md`
- `docs/dev/policies/0004-architecture-productization.md`
- `docs/dev/policies/0005-memory-and-context-routing.md`
- `docs/dev/policies/0006-git-release-validation.md`

## Graphiti Memory Discovery
- Use the `graphiti-discovery` skill at the start of non-trivial planning, debugging, architecture, routing, memory, or handoff work.
- Query repo group `transcribe_audio_main` before assuming prior context exists only in chat history.
- Treat Graphiti as advisory; verify cited facts against repo files, artifacts, commits, tests, or source episodes before changing code or live systems.
- When bootstrapping or refreshing repo memory, harvest from `ROADMAP.md`, `RUNBOOK.md`, `docs/dev/plans/`, `docs/dev/policies/`, and validated artifacts only.
- Do not seed secrets, raw private data, raw transcripts, full logs, or unreviewed speculation.
