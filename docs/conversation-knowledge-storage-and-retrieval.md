# Conversation knowledge storage and retrieval

This document defines how the system stores and retrieves the accumulated
conversation, person, relationship, topic, terminology, and provenance
evidence used for speaker identity inference. It is the evergreen architecture
reference for the domain model and retrieval contract. Implementation progress
belongs in [Plan 0029](dev/plans/0029-2026-07-26-conversation-knowledge-storage-retrieval.md)
and `RUNBOOK.md`.

## Goals

The architecture must:

- improve identity inference as reviewed conversations accumulate;
- preserve the exact evidence available to every inference;
- distinguish source records from people and observations from conclusions;
- retain account, tenant, relationship, and temporal context;
- prevent duplicate records from inflating corroboration;
- support exact, relational, lexical, and semantic retrieval;
- keep provider access host-owned and bounded;
- preserve private runtime data outside Git; and
- migrate from sidecars without creating two silent authorities.

It must not turn model output into an automatic contact merge, speaker
assignment, CRM update, Graphiti write, or training record.

## Authority model

The user-scoped transcript home is the target local authority:

```text
~/.transcripts/
├── transcripts.sqlite3
├── artifacts/
├── blobs/
├── legacy-artifacts/
└── exports/
```

SQLite stores normalized records and indexes. Content-addressed files store
audio and immutable JSON payloads. Provider systems remain authoritative for
complete external records.

During migration, the conversation-owned `.processing.json` sidecar remains
the processing authority and SQLite is a verified projection. The authority
may switch only after round-trip, count, identity, hash, and current-state
comparisons pass. After cutover, SQLite becomes authoritative and sidecars
become versioned, history-preserving exports generated from committed database
state.

The implementation must expose the current authority mode explicitly. It must
never resolve a conflict by silently preferring whichever copy was read first.

```text
Provider systems ──bounded retrieval──┐
                                      v
Source transcript/sidecar ──> SQLite knowledge store ──> Evidence bundle
          │                    │                │                 │
          │                    ├─ exact/FTS     └─ projections    v
          │                    └─ embeddings                App Intelligence
          │                                                   │
          └──── authority during migration <──── review/evaluation history
```

## Storage layers

### Immutable payload layer

Store large or source-shaped payloads as content-addressed files:

- normalized transcript artifacts;
- source recordings and derived audio;
- processing sidecar exports;
- bounded evidence packet payloads when their size makes row storage
  impractical; and
- redacted replay or migration receipts.

Each database reference records the content hash, media type, byte count,
storage path, role, creation time, and source reference when applicable.

### Normalized domain layer

Store queryable domain records in SQLite. Every durable record uses an opaque
ID. Names, emails, paths, hashes, speaker labels, and provider IDs are
attributes or aliases, not cross-domain identity keys.

### Derived projection layer

Build replaceable projections for:

- current speaker identity;
- current person profile;
- person-to-person interaction affinity;
- person-to-organization and person-to-project affinity;
- topic and terminology affinity;
- source freshness and availability; and
- retrieval ranking features.

Each projection records its schema version, input watermark, and build time.
Deleting and rebuilding a projection must not delete its source observations.

### Search index layer

Use:

- ordinary SQLite indexes for exact IDs, emails, timestamps, and source scope;
- FTS5 for names, aliases, organizations, projects, topics, terms, and bounded
  text;
- existing document and utterance-chunk embeddings for semantic retrieval; and
- indexed relationship edges for bounded graph expansion.

Add another database implementation only after measurements show that SQLite
cannot meet the required corpus size, latency, or concurrency.

## Core records

### Conversation and recording

`conversation` owns the durable interaction identity. `recording` owns one
captured media file or segment. A conversation may have multiple recordings.

Conversation records include:

- conversation ID;
- title and time range;
- current processing state;
- calendar-association state;
- creation and update timestamps; and
- explicit links to recordings and artifacts.

Recording records include:

- recording ID and conversation ID;
- source and stored blob references;
- media metadata and hashes;
- transcription backend and model;
- capture and processing timestamps; and
- the transcript artifact ID.

### Utterance and diarized speaker

Store each utterance with:

- utterance ID, conversation ID, and recording ID;
- original diarized speaker label;
- start and end times;
- normalized text;
- source artifact and ordinal;
- optional audio-segment reference; and
- lexical and embedding index state.

Preserve original diarization. Split-speaker groups, mixed-speaker findings,
and utterance-specific identities are claims layered over utterances, not
destructive relabeling.

### Person and external identity

`person` is the internal canonical identity. It may begin as provisional and
must retain its resolution status.

`external_identity` connects a person or unresolved source record to:

- an email address;
- a Google Workspace contact or directory identifier;
- an Odollo contact, lead, company, or tenant identifier;
- a calendar attendee identity;
- a reviewed local contact; or
- another provider-specific identifier.

Every external identity records its source profile, tenant or account,
identifier authority, validity interval, verification state, and source
record. Matching external identifiers may support person grouping, but names
alone don't merge people.

### Source record and source context

A source record is the representation returned by one account, tenant, or
database. It retains:

- source record ID and provider identifier;
- source profile, account, tenant, and declared Source Context;
- owning person or organization;
- relationship scope and evidence capabilities;
- retrieval and source-event timestamps;
- source URI or opaque reference;
- bounded normalized fields;
- content hash; and
- freshness or tombstone state.

Several source records may link to one person. They remain independently
addressable after grouping because the database in which a person appears
conveys relationship context.

### Relationship

Relationships connect typed entities:

- person to person;
- person to organization;
- person to project or matter;
- person to conversation;
- organization to project; and
- conversation to calendar event.

A relationship is an evidence-backed observation or derived projection, not an
unqualified fact. It records type, direction, source, temporal interval,
review state, confidence assessment when applicable, and supporting evidence
IDs.

### Topic and term

Store normalized topics, projects, matters, products, places, organizations,
technical terms, and characteristic phrases as typed concepts.

Concept mentions link to exact utterances or evidence snapshots. Derived
affinities connect concepts to people, organizations, and conversations with
support counts, independent-interaction counts, first and last observed times,
and review state.

Repeated mention frequency is a ranking feature. It isn't identity proof.

### Observation, claim, evaluation, and review decision

An `observation` records what a source or reviewer supplied. It doesn't
overwrite earlier observations.

A `claim` connects a subject, predicate, and object or value. It records:

- claim type and status;
- supporting and contradicting evidence IDs;
- evidence-independence groups;
- temporal applicability;
- model and rubric versions when inferred;
- confidence assessment;
- alternatives and warnings; and
- the evaluation that produced it.

An `evaluation` is an immutable App Intelligence result over one prepared
evidence bundle. A `review_decision` confirms, rejects, defers, or supersedes a
specific claim or proposal with reviewer, method, timestamp, and optional
correction.

Current state is a projection over immutable evaluations and decisions.

## Evidence snapshots

An evidence snapshot is the exact bounded material made available to one
evaluation. Store:

- evidence ID;
- source record and source profile;
- source type and capability;
- bounded snippet or structured metadata;
- source-event, observed, retrieved, and expiry timestamps;
- source URI or opaque provider reference;
- content hash;
- redaction and truncation metadata;
- evidence-independence group; and
- the retrieval request and bundle that selected it.

Don't store complete mailbox, Drive, contact-export, or log-note bodies solely
to make later prompting convenient. Retrieve from the authoritative provider
again when a new evaluation requires different evidence.

Evidence used historically remains immutable even if the live provider record
changes or disappears.

## Temporal model

Every retrieval request includes an `as_of` time, normally the conversation
time. Evidence records distinguish:

- `source_event_at`: when the underlying interaction or record applies;
- `observed_at`: when the local workflow first observed it;
- `retrieved_at`: when the provider returned it;
- `valid_from` and `valid_to`: when an identity or relationship applies; and
- `reviewed_at`: when a reviewer supplied or accepted a conclusion.

Retrieval classifies later evidence explicitly:

- contemporaneous evidence available by `as_of`;
- later-retrieved evidence describing the earlier period; or
- hindsight evidence learned only after the conversation.

Evaluation and benchmarking policies decide which class is allowed. Blind
historical evaluation must not treat later corrections as contemporaneous
evidence.

## Tenant and account isolation

Every provider-backed record carries a source profile and account or tenant
scope. Retrieval starts from an explicit user/runtime profile and permitted
source list.

Cross-source person grouping may connect records from several databases, but
it doesn't erase their source scopes. One tenant's relationship evidence
cannot silently satisfy a request scoped to another tenant.

Exact identity evidence and relationship context remain separate. The same
email may support identity linkage, while the source database explains whose
relationship with that person the record represents.

## Retrieval contract

The external seam should remain small:

```python
prepare_identity_evidence(
    conversation_id,
    *,
    speaker_labels=(),
    clue_ids=(),
    as_of=None,
    policy=None,
) -> EvidenceBundle
```

The interface hides provider commands, caches, indexes, ranking features,
deduplication, graph traversal, temporal classification, and packet budgeting.
Callers and tests receive the same immutable `EvidenceBundle`.

### Retrieval request

A retrieval request records:

- conversation, recording, speaker-label, and clue IDs;
- conversation time and `as_of` time;
- prepared candidate person IDs;
- permitted source profiles and evidence capabilities;
- maximum records, characters, provider calls, and relationship hops;
- freshness and hindsight policy;
- retrieval and ranking versions; and
- requesting workflow and run IDs.

### Retrieval stages

1. Extract typed clues from cited transcript utterances: names, aliases, roles,
   employers, organizations, projects, matters, places, relationships, forms
   of address, topics, and characteristic terms.
2. Generate deterministic candidates from exact calendar attendee emails,
   authoritative provider IDs, reviewed identities, and prepared contacts.
3. Retrieve exact identifier matches before broader lexical or semantic
   searches.
4. Search permitted source records, evidence metadata, reviewed conversations,
   and concept mentions with FTS5 and chunk embeddings.
5. Expand at most the policy-approved number of typed relationship hops from
   candidate people, organizations, projects, and events.
6. Apply tenant, account, temporal, capability, freshness, and access filters.
7. Group compatible source records into person candidates while preserving
   ambiguous records and evidence-independence groups.
8. Rank supporting and contradicting evidence with source affinity, temporal
   fit, topic fit, relationship fit, identifier authority, and independent
   interaction count.
9. Enforce per-source and total packet budgets.
10. Persist the request, selected evidence IDs, rejected candidate reasons,
    ranking version, and bundle hash before model reasoning.

### Evidence bundle

The bundle contains:

- prepared person candidates and their source records;
- exact transcript clues;
- calendar-association candidates;
- bounded provider evidence snapshots;
- prior reviewed identity observations allowed by temporal policy;
- relationship and concept summaries with their supporting IDs;
- included and excluded evidence with reason codes;
- warnings, freshness state, and source failures; and
- exact allowlists for the App Intelligence output schema.

App Intelligence may cite only bundle IDs. It cannot initiate retrieval,
expand source scope, or convert a missing record into evidence.

## Ranking and duplicate control

Ranking features include:

- exact authoritative identifier match;
- calendar attendee status;
- direct self-identification;
- name and form-of-address agreement;
- organization, role, project, topic, and terminology fit;
- prior reviewed interaction frequency and recency;
- source affinity and relationship scope;
- temporal compatibility;
- speaker-group and mixing evidence; and
- explicit contradictions.

Store the raw feature values and ranking version. Don't persist only a final
rank.

Evidence records that derive from one interaction share an independence group.
For example, an email-derived GWS contact, calendar attendee, and copied Odoo
note may all repeat one fact. They may improve retrieval recall, but they
cannot count as three independent confirmations.

## Accumulated profiles

Confirmed reviews append observations such as:

- a person spoke under one or more diarized labels;
- two diarized labels represented the same person;
- a label contained mixed speakers;
- a person participated in a conversation;
- a person was associated with an organization, project, or topic at a
  particular time; or
- a reviewer asserted an identity without provider evidence.

Materialized profiles summarize these observations for retrieval. Each profile
records the observation IDs and build watermark that produced it.

Profiles may improve candidate generation and ranking. They don't become
independent evidence merely because they summarize several source records, and
they don't silently train or update an external model.

## Freshness and provider access

Provider adapters own source-specific query syntax, rate limits, pagination,
redaction, and error handling. The retrieval planner works through a common
read-only adapter interface.

Cache records include fetch time, expiry policy, query fingerprint, source
profile, result hashes, and provider warnings. Expired cache data can support
historical replay when policy allows it, but it must be labeled stale.

Provider failure is partial and visible. A Gmail timeout must not erase
Calendar, Drive, local transcript, or Odollo evidence. The bundle records the
failed capability and prevents confidence logic from treating absence as
negative evidence.

## Graphiti boundary

Graphiti is an optional reviewed projection for compact durable facts and
associative discovery. Eligible projections include reviewed people,
organizations, relationships, matters, and stable routing decisions with
source citations.

Never project raw transcripts, complete provider bodies, raw evaluation
output, transient rankings, secrets, or unreviewed private claims. Graphiti
facts remain advisory until the workflow resolves their cited source records.

## Migration and authority cutover

### Phase 1: Schema and shadow projection

Add versioned tables and project existing transcripts, sidecars, assignments,
contacts, evaluations, and decisions into SQLite. Sidecars remain
authoritative. Re-running projection must be idempotent.

### Phase 2: Retrieval records and indexes

Add observations, external identities, source records, relationships,
concepts, evidence snapshots, retrieval requests, bundles, FTS indexes, and
embedding metadata. No model behavior changes in this phase.

### Phase 3: Person and affinity projections

Build reviewed person-resolution observations and rebuildable person,
relationship, topic, and terminology profiles. Preserve ambiguous records.

### Phase 4: Retrieval planner

Implement the bounded retrieval interface and replace ad hoc candidate
assembly one caller at a time. Record bundle receipts and compare them with
the existing prepared packets.

### Phase 5: Identity integration

Feed immutable bundles into Clue Discovery and Identity Evaluation without
weakening prepared-reference validation, confidence calibration, or review
gates.

### Phase 6: Chronological evaluation

Compare calendar-only, transcript-only, provenance-only, accumulated-history,
and combined retrieval on frozen chronological cases. Measure candidate
recall, top identity correctness, calibration, diarization findings, provider
yield, latency, and packet size separately.

### Phase 7: Authority cutover

Make SQLite authoritative only after:

- every eligible sidecar projects without loss;
- round-trip exports reproduce domain meaning;
- conversation, recording, evaluation, decision, and current-state counts
  reconcile;
- hashes and opaque IDs remain stable;
- shadow reads agree with sidecar reads;
- rollback to sidecar authority is tested; and
- backup and restore of the user-scoped store are proven.

## Failure and rollback behavior

- Schema migrations run transactionally and record their version.
- Projection failures identify the source artifact and leave it authoritative.
- Retrieval source failures produce warnings and partial bundles.
- Invalid model references reject the output; they don't trigger fuzzy
  remapping.
- Projection rebuilds can be discarded and regenerated from observations.
- Authority mode can return to sidecar reads until the cutover is declared
  irreversible in a later ADR.
- No migration deletes source artifacts or processing sidecars.

## Validation

Validation must cover:

- schema migrations and rollback;
- sidecar-to-database-to-sidecar round trips;
- idempotent projection;
- source-profile and tenant isolation;
- temporal and hindsight policies;
- exact identifier precedence;
- person grouping without same-name merges;
- evidence-independence grouping;
- FTS and semantic retrieval;
- deterministic packet budgets and bundle hashes;
- provider partial failure;
- historical evidence replay;
- current-state projection from immutable decisions; and
- chronological blind comparison against the existing identity workflow.

## Related documents

- [ADR 0002: Use a user-scoped conversation knowledge store](adr/0002-use-a-user-scoped-conversation-knowledge-store.md)
- [ADR 0001: Use durable conversation identities](adr/0001-use-durable-conversation-identities.md)
- [Plan 0025: App Intelligence speaker preprocessing](dev/plans/0025-2026-07-21-app-intelligence-speaker-preprocessing.md)
- [Plan 0029: Conversation knowledge storage and retrieval](dev/plans/0029-2026-07-26-conversation-knowledge-storage-retrieval.md)
