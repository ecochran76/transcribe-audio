# Plan 0029 | Conversation knowledge storage and retrieval

State: OPEN

Lane: P09

## Scope

Turn the existing user-scoped transcript store into the durable conversation
knowledge and evidence-retrieval layer defined by
[`docs/conversation-knowledge-storage-and-retrieval.md`](../../conversation-knowledge-storage-and-retrieval.md).

The campaign adds versioned domain storage, shadow projection from processing
sidecars, immutable evidence snapshots, temporal and tenant-aware retrieval,
reviewed person and affinity observations, and one host-owned evidence-bundle
interface. It then evaluates the combined retrieval path on chronological
speaker-identity cases before any storage-authority or automation cutover.

## Non-Goals

- No immediate replacement or deletion of `.processing.json` sidecars.
- No raw mailbox, Drive, contact-export, or Odollo log-note corpus copied into
  the transcript store.
- No model-controlled provider access.
- No automatic contact merge, speaker assignment, CRM mutation, deposition,
  Graphiti write, or external model training.
- No new vector or graph database without measured SQLite limits.
- No automatic confirmation until a future unseen holdout validates the
  complete retrieval and confidence path.
- No full-conversation interpretation pass in this campaign.

## Current State

`transcript_store.py` already owns the user-scoped SQLite database, copied
artifacts and media blobs, FTS5 document search, chunk embeddings, local
contacts, reviewed speaker assignments, and assignment audits.
`conversation_processing.py` owns durable conversation and recording IDs plus
append-only evaluation and review history in a conversation-owned processing
sidecar.

`speaker_identity_preprocess.py` already separates transcript Clue Discovery,
host-owned provenance retrieval, person grouping, Identity Evaluation,
prepared-reference validation, evidence scoring, and confidence calibration.
Configured GWS and Odollo sources retain semantic Source Context. These
primitives don't yet share a normalized cross-conversation model for external
identities, source records, relationships, concepts, observations, evidence
snapshots, retrieval requests, or immutable evidence bundles.

Plan 0026 remains the identity-quality campaign. Its next chronological batch
must not resume until this plan reaches the explicit evaluation gate or is
rejected earlier.

C1 is complete in source. `conversation_knowledge_store.py` provides a deep
storage interface over versioned transactional migrations, private
integrity-checked backups, tested rollback, conversation/recording/utterance
snapshots, cross-source person records, and immutable evaluation/review
history. The v1 schema includes the remaining relationship, concept,
observation, claim, and projection-state records required by later milestones.
The implementation is additive, leaves authority in sidecars, and has not
migrated the live user store.

C2 is complete in source. `conversation_knowledge_projection.py` provides a
read-only, hash-bound preview and approval-token-gated apply interface for
normalized transcripts, processing histories, linked legacy contacts, and
speaker assignments. It records deterministic watermarks and immutable
private reconciliation receipts, projects legacy assignments as immutable
source observations, and exports semantically equivalent processing sidecars
without changing their source. Fixture reconciliation and a private isolated
preview over the three current Voice Recordings processing sidecars passed.
The live user store was not migrated and `sidecar` remains the authority mode.

C3 is complete in source. Schema version 2 adds bounded evidence snapshots,
evidence-independence groups, lexical and embedding-profile indexes, immutable
retrieval requests, content-hashed evidence bundles, and reason-coded bundle
items. `conversation_knowledge_evidence.py` provides exact scoped identity
lookup, tenant/account/capability/time-isolated lexical and semantic evidence
search, typed concepts and mentions, and replayable request/bundle interfaces.
Raw provider-body fields and over-cap snippets or metadata are rejected.
Migration, rollback, isolation, immutability, hash-integrity, and private
live-database-copy rehearsals pass. The live user store remains unmigrated and
sidecars remain authoritative. C4 reviewed observations and deterministic
affinity projections are complete in source.

C4 adds schema version 3 and `conversation_knowledge_profiles.py`. Confirmed,
rejected, deferred, superseded, split-speaker, mixed-speaker, and
reviewer-asserted outcomes become typed immutable observations. Versioned
source-record and concept-mention observations preserve every source affinity.
Deterministic current person and interaction, organization, project, topic,
terminology, and source-relationship profiles cite their complete supporting
observation IDs and build watermark. Same-name ambiguous people remain
separate. Deleting and rebuilding all materialized profiles produces the same
records without changing the observation ledger. A private isolated preview
over the three current sidecars produced three diarization observations and an
unchanged second rebuild; no reviewed person profiles were expected because
those sidecars contain no review decisions or linked contacts. C5 host-owned
evidence retrieval is the current critical path.

## Stable architectural decisions

- Follow
  [`ADR 0002`](../../adr/0002-use-a-user-scoped-conversation-knowledge-store.md).
- Keep SQLite and content-addressed artifacts under the user-scoped transcript
  home.
- Treat sidecars as authoritative during shadow projection.
- Store observations immutably and derive rebuildable current-state profiles.
- Bind all provider records and queries to explicit source profiles, accounts,
  tenants, capabilities, and temporal policy.
- Keep retrieval host-owned and give App Intelligence only immutable prepared
  evidence IDs.
- Keep Graphiti limited to reviewed compact projections.

## Milestones

### C1 | Versioned domain schema and migration harness

Status: COMPLETE

Outcome:

- Define normalized tables for conversations, recordings, utterances, people,
  external identities, source records, relationships, concepts, observations,
  claims, evaluations, review decisions, and projection state.
- Add transactional schema versions, forward migration, tested rollback, and
  backup preflight.
- Define repository interfaces so callers don't depend on raw SQL or table
  layout.

Write surface:

- New focused storage modules and migrations.
- `transcript_store.py` compatibility wiring.
- Redacted schema fixtures and tests.

Stop condition:

- Stop before live migration if schema rollback or existing-store compatibility
  fails.

### C2 | Sidecar shadow projection

Status: COMPLETE

Evidence:

- Two end-to-end behavior tests cover read-only preview, explicit apply
  approval, source-change rejection, idempotent re-apply, private immutable
  receipts, legacy contact and assignment projection, reconciliation, and
  sidecar round trip.
- The private isolated live preview reconciled 3 conversations, 3 recordings,
  245 utterances, 3 evaluations, 11 proposals, and 0 current decisions without
  migrating the live store.
- All 3 projected live sidecars round-tripped semantically, and authority
  remained `sidecar`.

Outcome:

- Idempotently project normalized transcript identities, processing
  evaluations, review decisions, contacts, and speaker assignments into the
  new tables.
- Record source artifact hashes, projection watermarks, and reconciliation
  receipts.
- Export projected records back to a semantically equivalent sidecar without
  changing the source sidecar.

Gate:

- Conversation, recording, evaluation, decision, proposal, and current-state
  counts and identities must reconcile on fixtures and a private live preview.
- Sidecars remain authoritative.

### C3 | Evidence, concept, and retrieval records

Status: COMPLETE

Evidence:

- Six C3 tests cover bounded-content rejection, exact source-scoped identity
  lookup, tenant/account/capability/temporal isolation, lexical and semantic
  evidence retrieval, concepts and mentions, immutable request/bundle replay,
  reason-coded inclusion/exclusion, provider failures, migration failure, and
  v2 rollback.
- The complete 342-test inventory passes in isolated partitions while the host
  filesystem journal is degraded: 330 non-participant tests and all 12
  participant-identity tests.
- A consistent private copy of the live version-0 store migrated through
  versions 1 and 2, preserved legacy document counts, rolled version 2 back to
  version 1, and reapplied version 2 with `sidecar` authority unchanged.

Outcome:

- Add source records, Source Context, evidence snapshots, evidence-independence
  groups, concepts and mentions, retrieval requests, evidence bundles, and
  inclusion/exclusion reason codes.
- Add exact-identifier, FTS5, relationship, timestamp, source-scope, and
  embedding indexes.
- Preserve provider failures, freshness, redaction, truncation, and content
  hashes.

Gate:

- No provider body may be persisted outside the bounded snapshot contract.
- Tenant, account, capability, and temporal isolation tests must pass.

### C4 | Reviewed person and affinity observations

Status: COMPLETE

Evidence:

- Three C4 behavior tests cover every required review/diarization outcome,
  immutable re-append, same-name ambiguity, every source affinity, complete
  supporting-observation IDs and watermark, deterministic delete/rebuild, and
  version-3 rollback that preserves version-2 evidence records.
- The complete 345-test inventory passes in host-safe partitions: 333
  non-participant tests and all 12 participant-identity tests.
- The private isolated live-source preview appended three split/mixed
  diarization observations from three sidecars and produced an unchanged
  second rebuild without migrating the live store.

Outcome:

- Convert confirmed, rejected, deferred, superseded, split-speaker,
  mixed-speaker, and reviewer-asserted outcomes into immutable observations.
- Build replaceable current person, interaction, organization, project, topic,
  and terminology affinity projections.
- Preserve ambiguous same-name records and every source affinity after person
  grouping.

Gate:

- Derived profiles must identify their supporting observation IDs and build
  watermark.
- Rebuilding a projection must produce the same result without mutating
  observations.

### C5 | Host-owned evidence retrieval

Outcome:

- Implement
  `prepare_identity_evidence(conversation_id, *, speaker_labels, clue_ids, as_of, policy)`.
- Apply exact attendee-email and authoritative-ID lookup first, followed by
  bounded lexical, semantic, and relationship retrieval.
- Rank supporting and contradicting evidence, enforce independence groups and
  packet budgets, and persist an immutable bundle receipt before model use.

Gate:

- Existing calendar-only and prepared-candidate behavior must remain available
  as a fallback.
- Provider partial failure must return a labeled partial bundle rather than
  treating unavailable evidence as negative evidence.

### C6 | Speaker workflow integration

Outcome:

- Replace ad hoc evidence assembly one caller at a time with the retrieval
  interface.
- Preserve Clue Discovery, Identity Evaluation, exact prepared-reference
  validation, factor records, confidence calibration, and human-review gates.
- Expose retrieval provenance, freshness, temporal class, included/excluded
  reasons, and warnings in the existing review workflow.

Gate:

- No automatic assignment or external write.
- Existing regression and holdout artifacts remain immutable.

### C7 | Chronological evaluation and authority decision

Outcome:

- Freeze a new chronological evaluation set before inspecting predictions.
- Compare calendar-only, transcript-only, provenance-only,
  accumulated-history, and combined retrieval.
- Measure candidate recall, top identity correctness, correct-person presence,
  High/Very High correct and wrong proposals, diarization findings, validation
  yield, provider yield, latency, and packet size independently.
- Record an explicit accept, refine, reject, or stop decision.

Authority gate:

- SQLite cannot become the processing authority until sidecar round-trip,
  shadow-read agreement, backup/restore, and rollback evidence all pass.
- Automatic confirmation remains disabled until an unseen holdout validates
  the complete accepted path.

## Critical path and parallel work

Critical path:

1. C1 schema and interfaces.
2. C2 shadow projection and reconciliation.
3. C3 evidence records.
4. C5 retrieval interface.
5. C6 workflow integration.
6. C7 chronological evaluation and authority decision.

C4 projection algorithms may proceed in parallel with the latter part of C3
after observation and source-record schemas are stable. UI presentation may
proceed in parallel with C5 tests only after the evidence-bundle schema is
versioned. The critical-path owner reconciles both before C6.

## Acceptance Criteria

- One documented authority mode distinguishes sidecar authority, shadow
  projection, and database authority.
- Stable opaque IDs survive artifact moves and projection rebuilds.
- Every stored provider record retains source profile, account or tenant,
  Source Context, authoritative identifiers, timestamps, and hashes.
- Exact email and provider-ID lookup precedes broad retrieval.
- Retrieval applies explicit `as_of`, source-scope, capability, freshness, and
  hindsight policies.
- Every evidence bundle is immutable, content-hashed, budgeted, and replayable.
- Every claim cites exact transcript clues, evidence snapshots, and
  independence groups.
- Person grouping preserves source records and relationship context.
- Derived person, relationship, topic, and terminology profiles are
  reproducible from immutable observations.
- Graphiti receives no raw or unreviewed conversation content.
- Provider failures remain visible and don't become negative evidence.
- The existing speaker confidence calibration and review gates remain active.
- A chronological comparison isolates the value of each retrieval family
  before any automation or authority cutover.

## Validation

- Migration, rollback, backup, restore, and old-database compatibility tests.
- Sidecar/database round-trip and idempotent projection tests.
- Tenant, account, capability, temporal, and hindsight-policy tests.
- Exact, FTS5, semantic, and bounded relationship retrieval tests.
- Evidence-independence, packet-budget, source-failure, and bundle-hash tests.
- Observation and deterministic projection rebuild tests.
- Focused speaker workflow, API, store, and provider-adapter tests.
- Full suite and `git diff --check` after every integrated milestone.
- Private dry-run migration and reconciliation receipts before live apply.
- Frozen chronological comparison before any accepted behavior change.

## Rollback

- C1-C6 remain additive while sidecars are authoritative.
- Schema migrations run transactionally and preserve a preflight backup.
- Failed projections leave the source artifact authoritative and record a
  bounded error.
- Projection tables can be rebuilt from artifacts and observations.
- Retrieval integration retains the existing prepared-evidence fallback until
  C7 acceptance.
- No milestone deletes source artifacts, sidecars, evidence receipts, or
  review history.

## Stop Condition

Stop this campaign after C7 records an accepted, rejected, or explicitly
bounded refinement decision. If an authority-cutover gate fails, keep sidecars
authoritative and open a successor plan rather than extending this plan
indefinitely.
