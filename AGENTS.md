# Repository Guidelines

## Project Structure & Module Organization
- The sole entry point is `assembly_transcribe.py`, which handles uploads, polling, and DOCX output. Keep new modules in the repo root until a larger `src/` layout is justified and documented.
- Config artifacts sit alongside the script: `requirements.txt` lists runtime deps and `api_keys.json.sample` describes required secrets. Real keys belong in the ignored `api_keys.json`.
- Add sample assets only when essential for testing; prefer short clips under `tests/data/` and link to heavier media externally.

## Build, Test, and Development Commands
- Create a virtual environment: `python -m venv .venv && source .venv/bin/activate` (Windows: `.venv\Scripts\activate`).
- Install deps with `pip install -r requirements.txt`; keep the list minimal unless an AssemblyAI feature mandates more.
- Exercise the CLI locally via `python assembly_transcribe.py demo.wav --text-output` and inspect both terminal output and artifacts before altering command signatures.

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
- Store secrets in env vars or `api_keys.json` (ignored). Mirror any new fields in `api_keys.json.sample` and document them in the README to prevent drift.
- Prefer CLI flags for behavior tweaks; add config files only when options multiply, and describe resolution order clearly in documentation.
