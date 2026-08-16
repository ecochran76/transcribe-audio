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

Provider contacts are source affinities, not canonical people. A provider
record is uniquely addressed by provider, account or tenant, record type, and
provider record ID. Google Workspace, Odollo, receipts repositories, calendar
attendees, reviewed local contacts, and future sources may therefore retain
different but compatible records for one canonical person without copying or
collapsing their source-specific relationship context.

Baseline ingest is reconciliation, not source mutation. Auto-deduplicate the
same provider/account/type/record ID. Auto-link distinct source records to one
provisional person only through a non-conflicting exact person-specific email
or verified phone. Shared/role addresses never auto-link, and name,
organization, role, address, or fuzzy similarity creates a reviewed merge
proposal. Missing records in a later provider read are not tombstoned without
source-specific evidence. Reviewed local preferred fields remain overlays and
never overwrite provider observations.

Display names, titles, email addresses, acoustic profile labels, and
conversation-specific aliases are attributes or evidence. They must never
become the durable cross-source person key. Historical aliases remain
addressable after a reviewed merge, and merge/split decisions retain an
auditable redirect and reversal path.

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

### Roles and relationship graph

A role is a contextual, temporal relationship, not a field on `person` and not
a reason to create another person. Examples include employee of an
organization, attorney for a client or matter, physician for a patient,
project investigator, vendor representative, meeting host, or caller. The same
person may hold several concurrent or historical roles, and one provider may
expose only one of them.

The ontology is hierarchical and versioned. Relationship families such as
`FAMILY`, `PROFESSIONAL`, `MEDICAL`, `LEGAL`, `COMMERCIAL`, `EDUCATIONAL`,
`PROJECT`, and `SOCIAL` organize leaf types such as `PARENT_OF`, `SPOUSE_OF`,
`PHYSICIAN_FOR`, and `EMPLOYEE_OF`. Each type defines direction, inverse, and
symmetry behavior and may carry a more specific detail such as father, mother,
guardian, or department. Reviewer-proposed additions require near-duplicate
checking, parent placement, and a new ontology version.

Keep entity relationship edges distinct from contextual role assertions.
People may hold many simultaneous roles and relationships. Conflicting
assertions coexist with evidence, effective intervals, and conflict links
until review. Organizations support hierarchical, multi-affiliation edges such
as `PART_OF`, `SUBSIDIARY_OF`, and `DEPARTMENT_OF`.

Store people, organizations, projects, matters, conversations, recordings,
events, and source records as graph-addressable nodes. Store typed,
directional relationships such as `WORKS_FOR`, `REPRESENTS`,
`COLLABORATES_WITH`, `PARTICIPATED_IN`, `ADVISES`, `SUPPLIES`, `OWNS`, and
`DISCUSSED` as edges with evidence and temporal metadata. SQLite node and edge
tables may implement the authoritative graph initially; the domain contract
must not depend on one graph engine.

Role-only labels such as "meeting host", "company manager", or "doctor's
nurse" remain unresolved role observations until evidence supports a person
link. They do not silently create canonical people or contacts.

Relationship retrieval is bounded graph traversal. A request begins with
candidate people or other exact anchors, expands only the policy-approved
number of typed hops, preserves direction and source scope, and returns the
path plus supporting evidence IDs. Graphiti may receive compact accepted
projections for broader discovery, but the private conversation knowledge
store remains authoritative for identities, edges, evidence, and review
history.

### Topic and term

Store normalized topics, projects, matters, products, places, organizations,
technical terms, and characteristic phrases as typed concepts.

Concept mentions link to exact utterances or evidence snapshots. Derived
affinities connect concepts to people, organizations, and conversations with
support counts, independent-interaction counts, first and last observed times,
and review state.

Repeated mention frequency is a ranking feature. It isn't identity proof.

A `terminology_entry` records canonical spelling, expansion, definition,
aliases, ASR-confusion forms, pronunciation hints, supporting conversations,
validity, and scope. Scope precedence is conversation, project or matter,
organization, domain, then global; equal-scope conflicts require review. An
ASR confusion is not a synonym. For example, SoyLei `CISO` to `SESO` for
semi-epoxidized soybean oil is chemistry-scoped and must not become a global
replacement.

A `transcript_correction_proposal` identifies an exact raw-ASR span,
replacement, utterance/time range, context, evidence, confidence/version, and
review state. Accepted proposals produce a versioned normalized transcript.
Raw ASR and diarization remain immutable, searchable, and citable.

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

### Conversation association and participant hypotheses

A `conversation_association_candidate` links a recording or conversation to a
possible calendar event without asserting that the event is correct. It stores
the event's source identity, time fit, title/topic/entity/attendee factors,
credible alternatives, contradicting evidence, the calendar-association
rubric version, and an Evidence Strength Score. Calendar association is scored
independently of person linking and speaker identity.

A `participant_hypothesis` records that a person or unresolved source record
may have participated. Calendar attendance, organizer status, or a CRM link
may create a hypothesis, but none proves presence or speech. Suspected calendar
participants may therefore enter the normalized source-record inventory while
their conversation participation and speaker mappings remain reviewable
claims.

Every recording-backed queue or processing record retains the actual original
recording filename separately from stored blob names, transcript artifact
names, enriched display titles, and filesystem paths.

### Voice samples and profile versions

A `voice_sample` is an immutable reference to an exact source-audio interval.
It records conversation, recording, utterance or diarized-speaker lineage;
start/end times; source and derivative hashes; preparation recipe; quality and
overlap findings; identity-review authority; biometric-use authority; and
active, excluded, invalidated, retention, and deletion state.

A `voice_profile_version` is derived from an exact allowlist of eligible,
reviewed voice samples. It records the canonical person or still-unbound
acoustic subject, model and recipe versions, predecessor and successor,
training/calibration/evaluation split lineage, measurement results, activation
interval, and rollback state. Profile payloads and embeddings remain in the
private content-addressed layer rather than relational columns or Graphiti.

Unreviewed identity proposals may be scored against governed active profiles,
but they never enroll, extend, or retrain a profile. Speaker reassignment,
person split/merge reversal, source-audio invalidation, or withdrawn biometric
authority invalidates dependent samples and queues a deterministic new profile
version; it does not overwrite the prior profile.

Unreviewed samples and embeddings are retained indefinitely for now in
private, person-unbound storage until explicit deletion or policy change.
Recurring-voice clustering may run automatically, but membership is soft and
reversible and may preserve alternatives or no cluster. One person may have
multiple profile families for distinct acoustic conditions. Confirming one
cluster member may re-score and materially requeue others; it never assigns
them or enrolls a named profile.

Per-person and per-recording biometric exclusions invalidate dependent
samples, embeddings, profiles, and benchmark material. Initial deletion is
addressable by sample, cluster, person profile, recording, and person, with
previewed downstream effects and a minimal non-biometric audit tombstone.
Active storage is deleted immediately and future backups exclude the material;
historical encrypted backups expire according to their established schedule.
Routine biometric processing is local. Any external challenger is opt-in,
bounded, pseudonymous, reviewed, and evaluation-only until measurable lift is
accepted.

### Correction events and identity review queue

A `correction_event` is an append-only merge, split, redirect, retraction,
speaker reassignment, source-record correction, relationship correction, or
profile invalidation. It identifies the superseded record or decision, the
replacement, reviewer and method, recorded and effective times, affected
derivatives, and the projection/rebuild receipt. Corrections preserve both the
original historical result and the current accepted view.

An `identity_review_queue_item` is a rebuildable read model over current
reviewable claims. It points to, but does not duplicate authority from, the
conversation association, participant, person-link, speaker, relationship,
and acoustic records. Its UI projection includes the actual original filename,
calendar candidates and association strength, suspected attendees, source
records, per-speaker samples, independent contextual/acoustic evidence,
alternatives, warnings, prior decisions, and proposed downstream effects.

Review decisions support confirmation, correction, rejection, not-listed,
unresolved, mixed-speaker, split-label, same-person grouping, merge, split,
reversal, and defer states. A deterministic projector builds current speaker
assignments, people, roles, relationships, affinities, and eligible profile
rebuild requests from those decisions. Stale review submissions must fail
rather than overwrite a newer decision.

Freeform reviewer comments are immutable semantic-correction observations.
They may generate proposed structured corrections in a secondary queue, but
only reviewed structured derivatives become learning labels. A queue item is
complete when every speaker has an explicit disposition; unresolved is a
valid completion state. Requeue requires a material evidence, correction,
cluster, score, rubric, or model change.

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

### App Intelligence relationship inference

App Intelligence should endeavor to infer useful relationship and role
candidates from the immutable bounded evidence bundle. Its structured output
may propose relationship type and direction, the entities involved, temporal
scope, supporting and conflicting evidence IDs, confidence, alternatives, and
unresolved questions. Transcript language, introductions, calendar context,
provider contacts, CRM records, prior conversations, messages, documents, and
accepted graph neighborhoods may contribute when the host has prepared them.

These outputs are proposals, not graph mutations. Host validation checks every
entity and evidence reference, prevents circular identity/relationship
corroboration, applies review or acceptance policy, and only then appends an
observation and rebuildable relationship projection. Uncertain proposals stay
reviewable and may be strengthened, contradicted, or superseded later.

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

## Processing supervisor and learning loop

A host-owned supervisor composes deterministic stages and source-specific
adapters. One `processing_run` binds the conversation and recording IDs,
actual original filename, source and media hashes, `as_of` time, configured
capabilities, budgets, provider and model versions, evidence bundle, stage
outcomes, failures, proposed effects, and replay hashes. The supervisor, not a
model, owns retries, idempotency, partial failure, leases, checkpoints, and
effect application.

The ordinary identity-learning sequence is asynchronous after transcript
artifacts stabilize:

1. baseline contact/source-record reconciliation before historical work;
2. immutable raw transcript/diarization ingest and transcript-only semantic map;
3. scoped terminology normalization and pre-identity correction proposals;
4. calendar candidate generation and semantic association assessment;
5. exact-first attendee, external-identity, contact, message, document, CRM,
   prior-conversation, and bounded relationship retrieval;
6. enriched provisional readout and immutable evidence-bundle construction;
7. acoustic sample qualification, soft clustering, and governed-profile scoring;
8. contextual and acoustic proposal generation with separate pillar scores;
9. one bounded post-identity correction pass and material-change requeue;
10. Identity Review and People projections;
11. append-only review/correction decision and deterministic local projection;
12. accepted readout plus affected profile invalidation or candidate version; and
13. periodic calibration/evaluation and separately governed promotion.

One correction-to-identity cascade may run per processing version. A second
material cascade stops as `manual_resolution_required`. Unreviewed transcript
corrections may generate retrieval candidates but cannot corroborate their own
identity proposal.

Reviewed decisions create calibration outcomes. They do not immediately train
a model. Training, calibration, and source-disjoint evaluation partitions are
frozen explicitly; unreviewed predictions never become ground truth or feed
their own training. A new rubric, model, or profile version is promoted only
after measured acceptance, rollback proof, and an explicit release decision.

The host stores 0-100 Evidence Strength independently for calendar
association, person link, contextual speaker, acoustic speaker, and combined
ranking. It is not a probability; scores may rise or fall, and original values
remain immutable beside re-scores. Material contradictions cap combined
strength. Empirical Calibrated Likelihood requires at least 30 reviewed,
source-disjoint outcomes in the relevant band and exposes sample size,
interval, and evaluation version.

Evaluate reviewed outcomes weekly and propose a candidate version only after
25 new reviewed speaker decisions or a material correction. Automatic named
acceptance requires at least 100 varied source-disjoint outcomes and at least
99% precision in the proposed band, safe abstention, and no systematic
high-strength failure. Person merges and splits always remain reviewed.

Historical and new conversations use the same schemas and idempotent stages.
Historical backfill is oldest-forward, bounded, checkpointed, and resumable.
Blind historical evaluation excludes later facts and corrections through its
`as_of` policy; present-day operational reprocessing may use later accepted
knowledge only when it is labeled as hindsight evidence.

Reserve provider/model budget for new conversations and keep a separate
oldest-first historical queue. After 500 actionable items, continue cheap
normalization, metadata, sample extraction, and clustering while throttling
expensive enrichment. Retry one transient idempotent provider read, then
continue with visible partial evidence. Provider recovery appends evidence and
requeues only material changes.

## Authelia-protected identity review surface

The identity-learning queue is a private application surface behind the
dashboard's existing Authelia gate, not a public review page. Initial launch
adds no Google OAuth, local login, step-up authentication, or second security
layer. Its API exposes bounded range-playback handles,
optimistic-concurrency tokens, effect previews, and append-only decisions. It
never exposes raw filesystem paths, profile payloads, embeddings, unrestricted
audio URLs, or complete provider bodies.

`Identity Review` is conversation-first and `People` is the authoritative
people/role/relationship/profile editing view. Both show actual original
recording filenames wherever recording evidence appears. Preserve the
dashboard's existing request protections plus stale-write rejection, bounded
media access, audit history, backup/restore, rollback, and
privacy/retention/deletion validation. Anonymous access and public biometric
review links are prohibited.

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

- [Correction-first evidence supervisor](correction-first-evidence-supervisor.md)
- [Correction-first biometric custody](correction-first-biometric-custody.md)
- [Correction-first transcript learning](correction-first-transcript-learning.md)
- [Correction-first identity ledger](correction-first-identity-ledger.md)
- [Correction-first identity-learning contracts](correction-first-identity-learning-contracts.md)
- [ADR 0003: Freeze correction-first identity-learning contracts](adr/0003-freeze-correction-first-identity-learning-contracts.md)
- [ADR 0002: Use a user-scoped conversation knowledge store](adr/0002-use-a-user-scoped-conversation-knowledge-store.md)
- [ADR 0001: Use durable conversation identities](adr/0001-use-durable-conversation-identities.md)
- [Plan 0025: App Intelligence speaker preprocessing](dev/plans/0025-2026-07-21-app-intelligence-speaker-preprocessing.md)
- [Plan 0029: Conversation knowledge storage and retrieval](dev/plans/0029-2026-07-26-conversation-knowledge-storage-retrieval.md)
- [Plan 0072: Correction-first speaker, contact, and acoustic learning](dev/plans/0072-2026-08-16-correction-first-speaker-contact-learning.md)
- [Note 0058: Plan 0072 grilled architecture decisions](dev/notes/0058-2026-08-16-plan0072-grilled-architecture.md)
