# Plan 0012 | Speaker Deanonymization And Context Workbench

State: CLOSED

Lane: P09

## Scope

Build the next dogfooding milestone around speaker deanonymization and a
participant-aware context workbench before deposition or external apply work.

The software should endeavor to identify anonymous transcript speakers by using
deterministic attendee/contact evidence first, then operator review, then a
high-powered contextual readout provider with an explicit participant/context
bundle. Contact matching should come from configured user-scoped provenance
sources. Calendar invite attendees are deterministic match evidence, not the
contact system of record.

The first configurable contact provenance sources are:

- `gws` People API surfaces, exposed by the local `gws people` service:
  grouped Google Contacts, Other Contacts, and optionally directory people when
  the active profile has scope.
- `odollo` Odoo tenant profiles, using read-only partner/contact lookups against
  configured tenants such as SoyLei or Saber.

## Non-Goals

- No Drive, Odoo, Graphiti, or filesystem deposition apply work in this slice.
- No raw private transcripts, calendar exports, contact lists, or attendee
  fixtures in the repo.
- No automatic claim that `Speaker A` is a person without recorded evidence and
  confidence.
- No broad CRM enrichment system beyond the contact identity fields needed for
  speaker review and readout context.
- No unattended external writes from identity, context-workbench, or readout
  actions.

## Current State

Plan 0010 closed the first dogfoodable conversation review loop. The
conversation workspace can show transcript turns, selected first-pass summary
actions, SQLite-backed speaker/contact review, context provenance inspection,
contextual readout inspection, and no-write deposition/memory preview queueing.

Plan 0012 closes the next identity/context milestone. Speaker/contact review
now builds a deliberate participant-identity bundle from configured
`gws`/Odollo contact provenance, calendar attendee evidence, readout
participants, local reviewed contacts, and operator input. Context-workbench
previews and first-pass/contextual readout preparation carry that bundle before
any high-powered provider is asked to reason over anonymous speaker labels.
Deposition/memory preview remains gated when identity or context warnings are
unresolved.

Implemented seams:

- `participant_identity.py` defines
  `transcribe-audio.participant-identity-bundle.v1`.
- `context_sources.py` has read-only `gws` People/Contacts provenance for
  grouped contacts, Other Contacts, and optional directory people.
- Odollo `res.partner` contact provenance is promoted into the participant
  identity candidate pool while log-note provenance stays out of identity
  matching.
- `transcript_api.py` exposes the bundle through conversation detail,
  identity-review, context-workbench, first-pass preparation, and final-preview
  gate surfaces.
- The React workflow shows calendar evidence, configured source profiles,
  candidate provenance, manual contact entry, and blocked final previews.

## Contact Source Configuration

Contact provenance configuration should live under user-scoped runtime state,
not tracked repo files. The repo may provide a sample schema, but real profile
names, account paths, tenant labels, credentials, and contact outputs remain
under `~/.local/state/transcribe-audio/`, `~/.config/gws*`, `~/.odollo/`, or
the relevant source runtime.

The configuration shape is:

- `gws` contact profiles: profile label, `GOOGLE_WORKSPACE_CLI_CONFIG_DIR`,
  enabled People API surfaces (`contacts`, `other_contacts`, `directory`),
  query limits, readiness status, and read-only scope notes.
- `odollo` contact profiles: tenant profile label, config path, enabled Odoo
  models, query limits, readiness status, and read-only scope notes.
- Per-workflow policy: which contact provenance profiles may be used for
  speaker deanonymization, context workbench, and readout prompt bundles.
- No write scopes are required for this milestone.

The repo ships `contact-provenance.config.json.sample`; the live workstation
configuration for this milestone is user-scoped at
`~/.local/state/transcribe-audio/contact-provenance.config.json`.

## Work Items

- Define a participant identity bundle schema for readout/context runs. It must
  include transcript speaker labels, calendar attendees, matching-calendar
  participants, provenance contact candidates, operator decisions, confidence,
  evidence, unresolved ambiguities, and source profile metadata.
- Add deterministic attendee extraction and normalization from transcript
  sidecar `event.participants` and `event.matching_calendars` fields, including
  name/email handling where providers expose both.
- Add a read-only `gws` People/Contacts provenance adapter for grouped
  contacts, Other Contacts, and optional directory people, using configured
  user-scoped `gws` profiles.
- Promote Odollo tenant contact provenance from routing-only evidence into the
  participant identity candidate pool, preserving tenant/profile metadata.
- Generate contact candidates by matching calendar attendees, transcript
  participant labels, and speaker labels against configured `gws`, Odollo, and
  local reviewed contact sources before invoking an LLM.
- Add an operator input path for naming or correcting contacts that are not
  found in deterministic sources.
- Extend the conversation identity-review API payload so the UI can distinguish
  attendee evidence, `gws` contact matches, Odollo contact matches,
  existing-local-contact matches, operator-created candidates, and unresolved
  anonymous speakers.
- Extend context-workbench preview manifests to include the participant identity
  bundle and to make missing/ambiguous identities visible before reread.
- Pass the participant/context bundle into the high-powered readout phase,
  including providers such as AuraCall/Extended Pro ChatGPT, without making the
  provider the sole authority for identity.
- Keep low-confidence or conflicting identity decisions in Review Queue instead
  of carrying them into deposition-ready artifacts.
- Add tests for deterministic attendee lookup, identity-bundle serialization,
  conversation API exposure, and readout prompt inclusion.
- Add a browser smoke that opens a conversation, reviews speaker candidates,
  previews context with the participant bundle, and verifies that final readout
  preparation surfaces the bundle.

## Acceptance Criteria

- A selected conversation exposes a participant identity bundle through the API
  without leaking secrets or raw private contact exports.
- Configured `gws` People/Contacts and Odollo contact provenance can produce
  speaker/contact candidates before any LLM call.
- Calendar attendee data is used as deterministic matching evidence against
  configured contact provenance, not as the only contact source.
- The operator can confirm, defer, or provide a contact label for anonymous
  speakers, with decisions stored in user-scoped runtime state.
- Context-workbench preview includes participant identity state, provenance
  sources, unresolved ambiguities, and warning status.
- First-pass or contextual readout preparation can include the reviewed
  participant/context bundle for a high-powered provider.
- Deposition/memory preview remains gated and is not treated as ready when
  identity or context warnings remain unresolved.

## Validation

- Focused `pytest` coverage for attendee extraction, `gws`/Odollo contact
  provenance normalization, contact candidate scoring, identity-bundle
  persistence, conversation API payloads, and readout prompt payloads.
- `python -m py_compile` for touched backend modules and scripts.
- `npm --prefix frontend run build` for UI changes.
- Browser smoke evidence under
  `~/.local/state/transcribe-audio/browser-smokes/` showing the identity and
  context-workbench path on a local runtime conversation.
- Manual dogfood pass on one calendar-backed recording where invite attendees
  are available and one recording where operator input is required.

## Closeout Evidence

- `.venv/bin/python -m pytest -q` passed: 213 tests.
- `npm --prefix frontend run build` passed.
- `.venv/bin/python -m py_compile` passed for touched backend modules, scripts,
  and focused tests.
- Count-only live contact-provenance validation loaded 3 configured source
  profiles and found 6 compact sources for the Tempo query with zero adapter
  warnings, without printing contact records.
- Browser smoke passed:
  `~/.local/state/transcribe-audio/browser-smokes/20260523T181351Z-conversation-review-loop-smoke.json`.
  The selected live Tempo conversation had 4 speaker labels, 14 contact
  candidates, 6 calendar attendees, 3 source profiles, 3 pending speaker
  assignments, manual contact controls, context identity profile chips, and
  final-preview gating because 5 identity/context warnings remained.
