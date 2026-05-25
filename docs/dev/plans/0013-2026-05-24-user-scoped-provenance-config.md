# Plan 0013 | User-Scoped Provenance Configuration

State: COMPLETE

Lane: P09

## Scope

Define and implement a shared provenance configuration system that lives under
user-scoped runtime state, not the repository directory. The command-line tools,
watcher, transcript API, and React console should all resolve and mutate the
same settings through one configuration contract.

The first source families are:

- `gog` profiles for Google account/client scoped access.
- `gws` profiles for Google Workspace Calendar, Drive, People/Contacts, and
  Gmail surfaces.
- `msgcli` profiles for Outlook/mailbox message, contact, and attachment
  provenance.
- `odollo` profiles for specific Odoo/Odollo tenant profiles and enabled
  models such as `res.partner` and `mail.message`.
- `ical_calendar` profiles for private or shared iCalendar feeds such as Zoho
  calendar exports.

The system must be provider-extensible without letting unimplemented provider
kinds silently run. Unknown source kinds are valid to store only when disabled
or marked as planned; active source kinds require a registered adapter.

## Non-Goals

- No credentials, private iCalendar URLs, raw mailbox data, raw contact exports,
  or tenant-private records in tracked repo files.
- No external writes to Google, Outlook, Odoo/Odollo, Graphiti, Drive, or
  filesystem deposition targets.
- No removal of existing watcher or contact-provenance config paths until the
  shared resolver has compatibility tests and a migration path.
- No broad CRM enrichment or dedupe engine beyond the configuration contract
  and source selection needed by calendar metadata, participant identity, and
  context workbench flows.

## Current State

The repo has several useful but fragmented configuration surfaces:

- `watch_transcriptions.json` holds watcher behavior and, currently, calendar
  provider/provenance settings for watcher-driven runs.
- CLI flags such as `--calendar-providers`,
  `--calendar-provenance-calendar-id`, and
  `--calendar-provenance-ical-url` can configure one-off calendar provenance
  but do not share a default source registry with the web service.
- `contact-provenance.config.json` under
  `~/.local/state/transcribe-audio/` configures `gws` People/Contacts and
  Odollo contact provenance for participant identity.
- `intelligence_config.py` already demonstrates the desired preview/apply
  pattern for user-scoped mutable runtime config.

Plan 0013 creates the provenance equivalent: one resolver and one user-scoped
config file that both CLI and API code can consume.

## Configuration Location

Default runtime path:

```text
~/.local/state/transcribe-audio/provenance.config.json
```

Resolution order:

1. Explicit CLI/API override: `--provenance-config`.
2. Environment override: `TRANSCRIPTS_PROVENANCE_CONFIG`.
3. Default user-scoped path above.
4. Legacy compatibility reads from `contact-provenance.config.json` and watcher
   `calendar` blocks until migration is complete.
5. Repo sample `provenance.config.json.sample` is used only for initialization
   and documentation.

The active profile defaults to `active_profile` in the config, and can be
overridden per command or API request with `--provenance-profile` or an API
field named `provenance_profile`.

## Configuration Contract

The tracked sample is `provenance.config.json.sample`. The schema version is:

```text
transcribe-audio.provenance-config.v1
```

Top-level sections:

- `active_profile`: default operator/runtime profile.
- `profiles`: named workflow profiles. Each profile selects source ids and
  workflow-specific source subsets.
- `sources`: source registry keyed by stable local source id.
- `mutation_policy`: local config mutation constraints and audit location.

Common source fields:

- `kind`: one of `gog`, `gws`, `msgcli`, `odollo`, `ical_calendar`.
- `enabled`: boolean.
- `label`: human-readable display label.
- `capabilities`: provider surfaces such as `calendar`, `people`, `drive`,
  `gmail`, `mail`, `contacts`, `attachments`.
- `read_only`: defaults to true for every provenance adapter.
- `limits`: source-specific caps for events, contacts, messages, or notes.
- `secret_refs`: references such as `env:NAME` or future keyring/file refs.
- `sensitive_fields`: fields that must be redacted in API responses and run
  manifests.

Workflow sections:

- `calendar_metadata`: primary event-selection source plus additional calendar
  provenance sources, including iCalendar feeds.
- `participant_identity`: contact-provenance sources used to deanonymize
  speakers from attendees, readout participants, and operator input.
- `context_workbench`: read-only source pool for context preview and App
  Intelligence handoff bundles.
- `message_provenance`: optional mailbox/message provenance through `msgcli`.

## Provider Requirements

`gog` source:

- Supports account and client selectors.
- Can supply primary calendar lookup and additional Google Calendar provenance.
- May later expose Drive/Gmail/People capabilities through the same source id.

`gws` source:

- Supports `GOOGLE_WORKSPACE_CLI_CONFIG_DIR`.
- Supports Calendar, Drive metadata, People grouped contacts, Other Contacts,
  optional directory people, and Gmail surfaces as enabled capabilities.
- Must keep stderr out of JSON parsing paths.

`msgcli` source:

- Supports account/profile selection, mailbox folders, search defaults, result
  limits, and attachment policy.
- Defaults to read-only message/contact metadata; attachment materialization
  requires an explicit future workflow policy.

`odollo` source:

- Supports one config entry per tenant profile.
- Stores tenant profile, command path, repo root, config path, enabled models,
  query limits, and secret refs.
- Contact identity may use `res.partner`; context provenance may also use
  log-note/message models when enabled.

`ical_calendar` source:

- Supports private iCalendar feed URLs by `url_ref` or user-scoped literal
  `url`.
- Writes only stable hashed calendar ids such as `ical:<hash>` to artifacts.
- Parses event time windows, attendees, organizers, cancellations, EXDATE, and
  common recurrence rules.

## CLI And API Integration

Shared module:

```text
provenance_config.py
```

Responsibilities:

- Load, validate, redact, and write the user-scoped config.
- Resolve active profile and workflow-specific source sets.
- Convert source definitions into existing adapter config objects, including
  `CalendarProviderConfig`, iCalendar feeds, `GwsProvenanceConfig`, and
  `OdolloProvenanceConfig`.
- Provide a small command surface:
  `show`, `init-sample`, `doctor`, `preview-update`, and `apply-update`.

CLI changes:

- Add `--provenance-config` and `--provenance-profile` to transcription,
  repair, route/context, and relevant store/readout commands.
- Make `--use-calendar` resolve configured `calendar_metadata` provenance by
  default.
- Keep existing provider/calendar flags as explicit one-off overrides or
  additions.

Watcher changes:

- Keep `watch_transcriptions.json` focused on watch jobs, backend selection,
  readout/store behavior, and optional `provenance_profile`.
- Stop storing private calendar feeds in watcher config after migration.

API changes:

- Add `GET /api/provenance/config` for redacted config and resolved profile
  status.
- Add `POST /api/provenance/config/preview` for validation and diff preview.
- Add `POST /api/provenance/config/apply` for local config writes with the
  approval token from `mutation_policy`.
- Add read-only readiness/smoke endpoints that never print secret values.

## Migration Plan

1. Add `provenance_config.py` with validation, redaction, atomic write, and
   profile/workflow resolution.
2. Generate the live user-scoped config from existing local watcher calendar
   settings and `contact-provenance.config.json`.
3. Update direct CLI calendar lookup so `--use-calendar` loads configured
   calendar provenance by default.
4. Update participant identity to resolve `gws` and Odollo contact profiles
   from the shared config while retaining legacy contact config fallback.
5. Update context workbench and App Intelligence prompt-packet preparation to
   use the shared workflow source selection.
6. Expose redacted config read/preview/apply endpoints in the web service and
   UI.
7. Move live private iCalendar feeds out of watcher config and into
   `~/.local/state/transcribe-audio/provenance.config.json`.
8. Keep legacy flags/config for compatibility, with warnings once the shared
   config is active.

## Acceptance Criteria

- Direct CLI `--use-calendar` picks up configured shared Google and iCalendar
  provenance without requiring watcher-only settings.
- Watcher, CLI, transcript API, context workbench, and participant identity all
  resolve source definitions through the same config loader.
- The web API can show and mutate redacted config through preview/apply flows
  without exposing secret values.
- `gog`, `gws`, `msgcli`, multiple `odollo` tenant profiles, and arbitrary
  iCalendar feeds are represented in the schema.
- Runtime config and mutation audit records live under user-scoped state.
- Transcript artifacts, readout bundles, run manifests, and API responses do
  not contain private feed URLs or secret values.
- Existing tests for calendar matching, watcher expansion, participant
  identity, and context workbench continue to pass.

## Validation

- `.venv/bin/python -m py_compile provenance_config.py transcript_api.py participant_identity.py assembly_transcribe.py faster_whisper_transcribe.py repair_calendar_metadata.py route_transcript.py watch_transcriptions.py tests/test_provenance_config.py tests/test_transcript_api.py`
  passed.
- `.venv/bin/python -m pytest tests/test_provenance_config.py tests/test_participant_identity.py tests/test_transcript_api.py::test_provenance_config_endpoint_redacts_and_applies_updates tests/test_transcript_artifacts.py::test_watcher_calendar_config_expands_to_cli_args -q`
  passed: 14 tests.
- `.venv/bin/python -m pytest -q` passed: 227 tests.
- `npm --prefix frontend run build` passed.
- `.venv/bin/python -m json.tool provenance.config.json.sample` parsed the
  sample configuration successfully.
- `.venv/bin/python provenance_config.py doctor` reported `status: ok`.
- `watch_transcriptions.py --check --check-json` reported `status: ok` after
  watcher jobs were migrated to `calendar.provenance_profile`.
- Live F&B/SABER repair smoke used only `--provenance-profile default` and found
  one iCalendar provenance match plus four Google calendar matches; the
  refreshed source and stored artifacts contain no private Zoho URL and retain
  two attendee emails under hashed iCalendar provenance.
- Live API smoke showed `/api/provenance/config` redacts the SABER feed, reports
  one `gws` profile and two Odollo profiles, and `/api/provenance/config/doctor`
  reports `status: ok`.
- Live context-workbench smoke for conversation `6e8eee4f19a1d5a9b23f`
  reported three calendar attendees, 20 proposed contacts, and GWS/Odollo
  contact sources through the shared config.
- Browser smoke opened `/?view=Provenance`, rendered six source controls,
  displayed `SABER Zoho=[redacted]`, previewed a no-write provenance config
  update, and did not expose the private Zoho URL.
- Restarted `transcripts.service` and `transcribe-watch.service`; both are
  active and the API health endpoint returns `status: ok`.
