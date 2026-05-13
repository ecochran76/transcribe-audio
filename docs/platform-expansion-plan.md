# Transcription Platform Expansion Plan

Status: Background architecture note.

Planning authority now lives in `ROADMAP.md`, `RUNBOOK.md`, and bounded plans under `docs/dev/plans/`. Keep this file as conceptual context; do not use it as the source of truth for lane priority or plan state.

## Goal

Evolve the current transcription scripts into a meeting-intelligence pipeline:

1. watch for finished recordings
2. transcribe with the selected speech backend
3. enrich with calendar and identity context
4. summarize and contextualize with a selectable intelligence backend
5. identify the related matter or repository
6. re-read the transcript with supporting context
7. store the upgraded readout and harvest durable memories

The existing CLI behavior should remain stable. New behavior should be additive and config driven.

## Current Shape

- `assembly_transcribe.py` owns AssemblyAI upload, polling, and handoff to shared output processing.
- `faster_whisper_transcribe.py` owns local transcription and reuses the shared output path.
- `transcribe_common.py` owns calendar lookup, file/event matching, output naming, DOCX/TXT/SRT writing, translation, and subtitle embedding.
- `watch_transcriptions.py` owns directory scanning, file stability checks, backend fallback, state, retries, and notifications.
- `watch_transcriptions.json` is the right place for unattended service policy.

The key design pressure is to avoid adding summarization, routing, and deposition logic directly inside the two transcription backends. Those backends should produce a normalized transcript artifact; downstream stages should consume that artifact.

## Proposed Pipeline

### Stage 1: Capture

Input:

- media path
- watcher job name
- backend selection
- file fingerprint

Output:

- stable `TranscriptArtifact` metadata record
- raw transcript text
- structured utterances
- event metadata when available
- generated file paths

Implementation direction:

- introduce `transcript_artifacts.py` with dataclasses and JSON serialization
- have both transcription CLIs emit a sidecar JSON artifact next to DOCX/TXT outputs
- store the sidecar path in `watch_transcriptions_state.json`

### Stage 2: Calendar And Identity Context

Calendar provider selection should be explicit and ordered:

```json
{
  "calendar": {
    "enabled": true,
    "calendar_id": "primary",
    "providers": [
      {
        "type": "gog",
        "account": "eric@example.com",
        "client": "default"
      },
      {
        "type": "gws",
        "config_dir": "~/.config/gws-soylei"
      },
      {
        "type": "google_api",
        "credentials": "credentials.json",
        "token": "token.json"
      }
    ]
  }
}
```

Observed local command support:

- `gog` supports `--account` and `--client`, which maps well to tenant/profile selection.
- `gws` appears to select credentials through environment variables, especially `GOOGLE_WORKSPACE_CLI_CONFIG_DIR` and token/credential-file variables.
- the built-in Google Calendar API should remain the fallback for machines without either CLI.

Implementation direction:

- replace the current presence-based `build_calendar_service()` ordering with a `CalendarConfig` loader
- preserve the old `--use-calendar` flags as defaults
- add `--calendar-provider`, `--calendar-account`, `--calendar-client`, and `--calendar-config-dir` for CLI override
- let watcher jobs pass calendar settings without shell quoting complex provider lists

### Stage 3: Intelligence Backend

Summarization and contextualization should use a provider interface:

```python
class IntelligenceProvider:
    def summarize(self, artifact: TranscriptArtifact, context: ContextBundle) -> Readout:
        ...
```

Provider types:

- `codex_exec`: invokes `codex exec` with a deterministic prompt and local file inputs
- `auracall`: calls a local service or CLI under `../auracall`
- `openclaw`: invokes OpenClaw for analysis, notification, and possible memory deposition
- `openai_compatible`: calls an OpenAI-compatible HTTP API using `OPENAI_API_KEY`, `OPENAI_BASE_URL`, and model config

Config sketch:

```json
{
  "intelligence": {
    "enabled": true,
    "provider": "openai_compatible",
    "model": "gpt-5.4",
    "base_url": "https://api.openai.com/v1",
    "api_key_env": "OPENAI_API_KEY",
    "outputs": ["summary", "action_items", "participants", "topics", "risks", "memory_candidates"]
  }
}
```

The first readout should be deliberately simple:

- executive summary
- participant and organization guesses with confidence
- topics and decisions
- action items
- unresolved questions
- likely related matters
- candidate memories

### Stage 4: Matter Routing

Routing should be a scoring problem, not a single brittle rule.

Inputs:

- event title, attendees, organizer, location, calendar IDs
- transcript summary and salient entities
- historical transcript artifacts
- Graphiti/OpenClaw memory graph
- configured repository indexes

Candidate repository types:

- local folder
- Google Drive folder or document
- Odoo record or project/task/note
- Graphiti entity or episode
- OpenClaw resource
- no confident match, requiring review

Config sketch:

```json
{
  "routing": {
    "enabled": true,
    "min_confidence": 0.75,
    "candidate_sources": [
      {"type": "local_index", "path": "~/.openclaw/matters.json"},
      {"type": "graphiti", "endpoint": "http://127.0.0.1:8817"},
      {"type": "google_drive", "provider": "gog"},
      {"type": "odoo", "profile": "soylei-prod"}
    ],
    "on_low_confidence": "queue_review"
  }
}
```

Routing output:

- chosen repository
- confidence
- evidence
- rejected alternatives
- next action plan

### Stage 5: Contextual Reanalysis

After routing finds a likely matter, gather supporting context and run a second readout.

Context examples:

- recent notes for the matter
- prior transcripts with the same people/topic
- Drive docs in the target folder
- Odoo project notes/tasks
- Graphiti neighborhoods for detected people, companies, matters, and technologies

The second readout should be stored separately from the raw transcript:

- `*.readout.md`
- `*.readout.json`
- optional DOCX for human sharing

The JSON form should be the durable automation contract.

### Stage 6: Deposition And Memory Harvest

Depositors should be pluggable:

```python
class RepositoryDepositor:
    def deposit(self, artifact: TranscriptArtifact, readout: Readout, route: RouteDecision) -> DepositResult:
        ...
```

Initial depositors:

- local filesystem
- Google Drive through `gog` or `gws`
- OpenClaw notification/log
- Graphiti episode/memory export
- Odoo note/task placeholder once the target model is defined

Do not let deposition delete or move source recordings until the readout and route record are written.

## Suggested Milestones

### Milestone 1: Normalize Artifacts

- add `TranscriptArtifact` sidecar JSON
- return output paths from `process_transcription_outputs()`
- update watcher state to record artifact paths
- add tests around artifact generation and processed-state migration

### Milestone 2: Calendar Provider Config

- add structured calendar config
- support ordered provider selection
- add tenant/profile selection for `gog` and `gws`
- preserve built-in Google API fallback
- add verbose logs for provider choice and provider failures

### Milestone 3: First Readout

- add `summarize_transcript.py`
- implement `openai_compatible` provider first
- add `codex_exec` provider second for local/manual high-quality passes
- write `*.readout.md` and `*.readout.json`
- wire watcher post-processing behind `intelligence.enabled`

### Milestone 4: Routing Queue

- add route decision schema
- implement local review queue for low-confidence routing
- implement Graphiti/OpenClaw candidate lookup adapters
- do not auto-deposit outside local filesystem until route evidence is auditable

### Milestone 5: Repository Deposition

- implement local and Google Drive deposition
- add Odoo once record model and tenant/profile handling are clear
- implement Graphiti memory harvesting from readout JSON

## Open Decisions

- What is the canonical identity for a "matter": folder ID, Graphiti entity ID, Odoo record, OpenClaw resource, or a local registry entry?
- Should every transcript be deposited somewhere, or should low-confidence transcripts wait in a review queue?
- Which intelligence backend is the unattended default?
- Should `codex exec` be allowed in the systemd service path, or reserved for manual enrichment because of runtime and permissions?
- What is the minimum useful Odoo target: note, project, task, CRM lead, contact, or custom matter model?
- Should Graphiti memories be harvested only after human review, or can high-confidence readouts write memories automatically?

## Near-Term Recommendation

Start with artifacts and calendar config before routing. Routing and contextual reanalysis need reliable, structured transcript records and event metadata. Once those exist, summarization and route scoring can evolve without rewriting the transcription backends.
