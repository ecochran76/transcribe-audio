# Plan 0012 | Speaker Deanonymization And Context Workbench

State: OPEN

Lane: P09

## Scope

Build the next dogfooding milestone around speaker deanonymization and a
participant-aware context workbench before deposition or external apply work.

The software should endeavor to identify anonymous transcript speakers by using
deterministic attendee/contact evidence first, then operator review, then a
high-powered contextual readout provider with an explicit participant/context
bundle. Calendar invite attendees are the first deterministic source because
they are already attached to transcript sidecars through `event` and
`event.matching_calendars`.

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

That loop is not yet strong enough for deposition. Speaker/contact review is
mostly operator-driven, context-workbench actions are local manifest previews,
and first-pass/contextual readouts do not yet receive a deliberate
participant-identity bundle built from deterministic calendar attendee lookup
and reviewed operator input. P05 external deposition and richer memory harvest
should wait until this identity/context bundle exists.

## Work Items

- Define a participant identity bundle schema for readout/context runs. It must
  include transcript speaker labels, calendar attendees, matching-calendar
  participants, candidate contacts, operator decisions, confidence, evidence,
  unresolved ambiguities, and source profile metadata.
- Add deterministic attendee extraction and normalization from transcript
  sidecar `event.participants` and `event.matching_calendars` fields, including
  name/email handling where providers expose both.
- Generate contact candidates from calendar attendees and existing local
  contact/identity tables before invoking an LLM.
- Add an operator input path for naming or correcting contacts that are not
  found in deterministic sources.
- Extend the conversation identity-review API payload so the UI can distinguish
  deterministic attendee matches, existing-contact matches, operator-created
  candidates, and unresolved anonymous speakers.
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
- Calendar attendee data can deterministically produce speaker/contact
  candidates before any LLM call.
- The operator can confirm, defer, or provide a contact label for anonymous
  speakers, with decisions stored in user-scoped runtime state.
- Context-workbench preview includes participant identity state, provenance
  sources, unresolved ambiguities, and warning status.
- First-pass or contextual readout preparation can include the reviewed
  participant/context bundle for a high-powered provider.
- Deposition/memory preview remains gated and is not treated as ready when
  identity or context warnings remain unresolved.

## Validation

- Focused `pytest` coverage for attendee extraction, contact candidate scoring,
  identity-bundle persistence, conversation API payloads, and readout prompt
  payloads.
- `python -m py_compile` for touched backend modules and scripts.
- `npm --prefix frontend run build` for UI changes.
- Browser smoke evidence under
  `~/.local/state/transcribe-audio/browser-smokes/` showing the identity and
  context-workbench path on a local runtime conversation.
- Manual dogfood pass on one calendar-backed recording where invite attendees
  are available and one recording where operator input is required.
