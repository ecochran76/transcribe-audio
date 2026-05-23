# Plan 0010 | Dogfoodable Conversation Review Loop

State: OPEN

Lane: P09, with dependencies on P04, P05, and P06

## Scope

Create the first end-to-end operator milestone for the transcript console: a
human can start from an ingested conversation, inspect source media and
transcript content, review or generate the first-pass summary, gather and
inspect provenance context, resolve speaker/contact identity, produce a
contextual readout, and queue deposition or memory-harvest preview decisions
without unattended external writes.

This plan turns the broad P09 review console into a dogfoodable workflow rather
than another sequence of disconnected UI increments. It does not replace the
P04 routing/contextual reread lane, the P05 deposition/memory lane, or the P06
service reliability lane; it binds their current contracts into one operator
path.

## Non-Goals

- Do not enable unattended Google Drive, Odoo, Graphiti, or other external
  writes from the UI.
- Do not add new provenance or intelligence providers unless an existing M1
  workflow step is blocked without them.
- Do not expand App Intelligence branch, fork, rollback, or autonomous apply
  behavior beyond what is needed for reviewed summary, context, or readout
  work.
- Do not add raw private transcripts, audio blobs, credentials, share tokens,
  or runtime state to tracked repo files.
- Do not spend this milestone on visual polish that is not tied to the
  operator review loop.

## Current State

The transcript console already has a React + Vite shell, server-backed
conversation rows, URL-addressable library state, copyable workspace links,
resizable panes and columns, a full-viewport conversation workspace, stored
blob playback routes, transcript/source-resolution behavior for readouts, a
first-pass batch queue/status surface, App Intelligence readiness and ledger
inspection surfaces, and repeatable `agent-browser` smokes for library
deep-link/share checks.

The user-scoped store under `~/.transcripts` contains imported historical
documents, blobs, artifact copies, chunk indexes, semantic vectors, and
workflow metadata. Runtime job records, review queues, smokes, ledgers, share
state, and provider configuration remain under user-scoped runtime locations.

The remaining gap is workflow continuity. Several important panels exist as
inspection or planned affordances, but the dogfooding path is not yet a single
operator loop from conversation selection through final reviewed output.

## Milestone Outcome

M1 is complete when an operator can use the UI to complete this sequence for a
representative stored conversation:

1. Search or filter the Library and open one conversation object.
2. Verify source audio availability and play it from the stored blob.
3. Inspect a scrollable, speaker-delineated transcript.
4. Review an existing first-pass summary or prepare/check/materialize one from
   the selected conversation.
5. Inspect and resolve speaker/contact identity review state.
6. Run or inspect context gathering between the first-pass summary and final
   readout.
7. Inspect included and excluded provenance sources, warnings, and confidence.
8. Generate or inspect a contextual readout tied to the same conversation.
9. Queue deposition and memory-harvest preview decisions for human review.
10. Reopen the same workspace by URL and reproduce the same workflow state.

## Workflow Contract

The conversation workspace is the milestone's primary surface. It should own
the whole review loop instead of scattering one conversation across unrelated
navbar pages.

Workspace views:

- `Transcript`: source media, playback speed, transcript turns, timestamps,
  source artifact metadata, and re-transcription review actions.
- `First-pass summary`: existing summary readout, summary generation status,
  prepare/submit/status/materialize actions, and failure diagnostics.
- `Speakers`: speaker labels, known assignments, candidate contacts,
  confidence/evidence, and defer/confirm review actions.
- `Context workbench`: provenance-source readiness, deterministic context
  recipe status, included/excluded source packets, warnings, and rerun actions.
- `Final readout`: contextual readout content, provenance delta, unresolved
  warnings, deposition preview status, and memory-harvest preview status.

The navbar can still expose Library, Review Queue, Context Runs, Contacts,
Provenance, Intelligence, Depositions, and Settings, but M1 work should wire
the selected conversation path first. Other sections may aggregate or inspect
state, not become the primary route for this milestone.

## Data And API Requirements

The milestone should use existing user-scoped runtime boundaries:

- `~/.transcripts/transcripts.sqlite3` for conversation, document, chunk,
  blob, contact, speaker, and workflow metadata.
- `~/.transcripts/blobs/` for source recordings and derived binary artifacts.
- `~/.transcripts/artifacts/` and `~/.transcripts/legacy-artifacts/` for
  copied JSON/Markdown artifacts.
- `~/.local/state/transcribe-audio/` for manifests, review queues, smokes,
  App Intelligence ledgers, provider readiness caches, and share state.

Required backend contracts:

- Conversation detail returns transcript, first-pass summary, contextual
  readout, media, artifact membership, participants, speaker assignments,
  context status, and review state in one payload or documented follow-up
  endpoints.
- Blob playback/download stays range-capable and never streams arbitrary
  original filesystem paths.
- Contact and speaker endpoints support read, candidate listing, confirm,
  defer, and audit records before automatic disambiguation is attempted.
- Context endpoints expose preview/run/status/materialize behavior over the
  existing route/contextual reread contracts.
- Deposition and memory-harvest endpoints expose preview and review state only
  until external apply contracts are explicitly implemented.

## Implementation Slices

1. Conversation contract audit: compare `/api/conversations/<id>` with the M1
   workflow needs and document or add the missing fields without changing the
   UI first.
2. Speaker/contact review foundation: add contact, identity, speaker
   assignment, and audit surfaces needed for the workspace to show real review
   state.
3. First-pass summary workspace wiring: run prepare/status/materialize actions
   from the selected conversation and show stored results in place.
4. Context workbench wiring: expose route/context acquisition status,
   provenance included/excluded decisions, warnings, and contextual reread
   generation for the selected conversation.
5. Final readout and preview queue wiring: show contextual readout content and
   create deposition/memory-harvest preview review items without external
   writes.
6. Review Queue integration: link queue items back to the selected
   conversation and allow local review decisions for route, contact, context,
   deposition, and memory candidates.
7. Browser smoke coverage: add a repeatable `agent-browser` happy-path smoke
   that opens a conversation deep link, verifies media/transcript/summary/
   context/final-readout states, and records JSON plus screenshot evidence.

## Acceptance Criteria

- A stored conversation is represented as one workflow object across transcript,
  summary, contextual readout, media, participants, and review state.
- The conversation workspace can be opened by URL and restored to the selected
  workflow view.
- Source audio playback is available through stored blob routes when media is
  linked, with a clear disabled state when it is not.
- Transcript content renders in a scrollable speaker-delineated frame and does
  not overlap other panels.
- First-pass summary actions are scoped to the selected conversation and can
  materialize completed readouts into the store.
- Speaker/contact review is backed by durable contact and assignment records,
  not only placeholder UI.
- Context gathering happens between first-pass summary and final readout, with
  included/excluded provenance, warnings, and confidence visible to the
  operator.
- Contextual readout generation or inspection is available from the same
  conversation workspace.
- Deposition and memory-harvest actions remain preview/review gated and do not
  perform unattended external writes.
- Review Queue items link back to the relevant conversation and workflow stage.
- At least one browser smoke validates the M1 happy path against local runtime
  state without embedding private fixture data in the repo.

## Validation

- `python -m py_compile` for touched Python backend modules.
- Focused `pytest` coverage for conversation, blob, contact/speaker, context,
  and review endpoints touched by the milestone.
- `npm --prefix frontend run build` for frontend changes.
- `agent-browser` smoke for the conversation review loop with JSON and
  screenshot evidence under `~/.local/state/transcribe-audio/browser-smokes/`.
- Manual dogfood pass on at least one representative historical conversation
  and one recent watcher-ingested conversation, documenting gaps in `RUNBOOK.md`.

## Checkpoint Policy

Use small commits at meaningful milestone checkpoints:

- Plan/roadmap/runbook creation.
- Backend contract additions with tests.
- Frontend workflow wiring with build validation.
- Browser-smoke evidence.
- Manual dogfood findings and follow-up plan split if M1 becomes too broad.

If a slice starts growing beyond the M1 review loop, split it into a new plan
instead of expanding this document into another catch-all.
