# Speaker identity pillar integration architecture

Date: 2026-08-08

## Decision

Build speaker identity as three independently testable evidence pillars joined
by one host-owned decision workflow:

1. acoustic speaker analysis;
2. bounded context collection; and
3. canonical-person, contact, role, and relationship knowledge.

Each pillar must emit immutable, source-bound evidence. A later identity-case
orchestrator may combine those bundles into a proposal, alternatives,
contradictions, calibrated confidence, or an abstention. Only a separate
acceptance path may create an authoritative speaker assignment or accepted
conversation observation.

The first integration milestone must reach Level 2 shadow behavior across the
joined flow. It must not enable automatic assignment, live knowledge-store
authority, profile learning, provider write-back, relationship inference
writes, or historical reprocessing.

This decision extends the canonical-person and relationship sequencing in
[Note 0052](0052-2026-08-05-contact-role-relationship-sequencing.md). The
evergreen storage and retrieval authority remains
[Conversation knowledge storage and retrieval](../../conversation-knowledge-storage-and-retrieval.md).

## Why this boundary exists

Speaker identity is a conversation-level inference. Acoustic similarity can
support a known enrolled subject, but it cannot establish which provider
contact, role, organization, or relationship applies. Context can identify
strong candidates, but a calendar attendee or contact match is not proof that
the person spoke. The knowledge store can preserve accepted history, but it
must not treat an unreviewed inference as new evidence for the same inference.

Separating the pillars prevents four unsafe shortcuts:

- a diarization label or display name becoming a global identity key;
- raw audio or a voice embedding becoming a contact record;
- duplicate provider records being counted as independent corroboration; and
- an inferred identity creating a relationship that is immediately reused to
  support that identity.

## Current maturity and target

Current evidence on 2026-08-08 establishes the following baseline.

| Capability | Current maturity | First integration target |
| --- | --- | --- |
| Acoustic speaker evidence | Level 2 integrated shadow for two enrolled subject IDs | Level 2 reusable, source-bound acoustic bundle for a selected conversation |
| Context collection | Mixed Level 1 and Level 2 components with bounded retrieval, participant candidates, and provider adapters | Level 2 replayable context bundle on real selected conversations |
| Canonical-person knowledge | Level 1 implementation in source; the live database remains schema version 0 with sidecar authority | Level 2 private-copy shadow projection and reviewed reconciliation preview |
| Role and relationship knowledge | Level 1 affinity and relationship summaries; no general live authoritative edge graph | Level 1 typed temporal contract plus Level 2 read-only use of accepted evidence |
| Human review surface | Level 2 synthetic and integrated-shadow proof | Level 2 joined evidence review with explicit unresolved outcomes |
| Live automatic orchestration | Level 0 | Level 0 in the first integration plan |

The current user-scoped transcript store contains 466 documents, 2 contacts,
and 3 speaker assignments. These counts are a planning readback, not a frozen
execution cohort. An activated plan must freeze fresh counts and hashes before
projection or evaluation.

## End-to-end flow

```text
normalized transcript and diarized utterances ready
                    |
          freeze one identity case
                    |
        +-----------+------------+
        |                        |
  acoustic branch           context branch
  local speaker refs        calendar and transcript clues
  enrolled subject IDs      contacts and provider snapshots
  model and source hashes   accepted historical knowledge
        |                        |
  AcousticEvidenceBundle    ContextEvidenceBundle
        |                        |
        +-----------+------------+
                    |
          host-owned evidence join
                    |
      proposal, alternatives, contradictions,
         confidence cap, or explicit abstention
                    |
             human review gate
                    |
       accepted decision writer in a later plan
                    |
      speaker assignment and accepted observations
                    |
        context for future conversations
```

The watcher remains a transport and backend-orchestration surface. After a
transcript reaches its durable ready state, it may eventually enqueue an
identity case. It must not perform provider retrieval, acoustic scoring,
person reconciliation, or assignment writes inline.

## Identifier contract

| Identifier | Scope and authority |
| --- | --- |
| `conversation_id` | Stable conversation identity in the user-scoped knowledge domain |
| `recording_id` | Stable recording identity within the conversation model |
| `document_id` | Stored transcript or readout artifact identity |
| `speaker_ref` | Recording-local diarization label; never a person identifier |
| `acoustic_subject_id` | Opaque enrolled voice identity; never a contact or person by itself |
| `person_id` | Canonical internal human identity |
| `source_record_id` | One independently addressable provider, account, tenant, calendar, or local record |
| `evidence_id` | Immutable observation or snapshot identity |
| `bundle_id` | Immutable, content-hashed evidence collection identity |
| `evaluation_id` | One versioned identity evaluation over exact bundles |
| `decision_id` | Human or policy-qualified authority for an accepted, rejected, or unresolved outcome |

Names, aliases, email addresses, phone numbers, provider IDs, acoustic display
labels, diarization labels, and role labels are attributes or external
identities. None is a global person key.

## Pillar contracts

### Acoustic speaker analysis

The acoustic branch accepts an exact recording, diarized utterances, and an
allowlisted reference set. It returns one immutable acoustic bundle containing:

- recording and document bindings;
- recording-local speaker references;
- an opaque enrolled subject candidate or abstention;
- supporting and opposing unit counts;
- calibrated score and confidence band;
- source-media, execution, model, and identity-state hashes; and
- explicit negative-action flags.

Raw audio, review clips, voice embeddings, and model-private features remain in
private runtime storage. The conversation knowledge database may retain opaque
artifact references and evidence hashes, but it must not store these values as
contact fields or canonical-person attributes.

### Bounded context collection

The context branch accepts the frozen conversation, recording-local speaker
references, explicit source scopes, an as-of time, capability allowlists, and
retrieval budgets. It returns one immutable context bundle containing:

- calendar candidates and overlap evidence;
- transcript clue references and prepared query terms;
- canonical-person candidates;
- independently scoped provider and local source records;
- accepted historical role and relationship evidence;
- support, contradiction, and evidence-independence features;
- inclusion and exclusion reasons;
- partial-provider failures and warnings; and
- retrieval, ranking, source, and policy versions.

Provider failure is partial and visible. A failed provider must not erase
evidence from another source, and missing evidence must not become negative
evidence unless the policy explicitly supports that interpretation.

### Canonical-person and relationship knowledge

The user-scoped SQLite conversation knowledge store is authoritative for
accepted conversation-derived knowledge. Provider systems remain authoritative
for their full external records. The store must preserve:

- canonical people;
- external identities and source records with account and tenant affinity;
- immutable observations and evidence snapshots;
- review decisions and supersession links;
- contextual, time-bounded roles; and
- typed, directional, temporal, evidence-backed relationships.

The first foundation plan does not need a complete multi-hop relationship
graph. It must freeze the record and authority contracts, preserve reviewed
existing relationship evidence, and leave uncertain role-only labels
unresolved. Relationship inference, authoritative edge persistence, and
multi-hop ranking require later bounded milestones.

Graphiti may receive compact, reviewed, source-backed projections after a
separate apply gate. It is not the raw contact, transcript, acoustic, or
unreviewed relationship store.

### Identity-case orchestration

The orchestrator accepts references to one frozen transcript/diarization
snapshot, one acoustic bundle, one context bundle, and one canonical-person
candidate snapshot. It returns an immutable evaluation record. It does not
own provider adapters, acoustic models, database migrations, review UI state,
or assignment persistence.

The evaluation must preserve separate acoustic, context, relationship, and
contradiction factors. It must not collapse provenance into an unexplained
single score. Host-owned policy applies reason-coded confidence caps when
evidence is incomplete, dependent, stale, scope-conflicting, or materially
contradictory.

## State and authority model

```text
pending
  -> evidence_collecting
  -> evidence_ready | evidence_partial | evidence_failed
  -> proposed | abstained
  -> review_required
  -> accepted | rejected | unresolved
  -> projected
```

The first integration plan stops at `accepted`, `rejected`, or `unresolved` in
a shadow decision ledger. `projected` belongs to a later apply plan.

Every transition must record the actor, timestamp, input hashes, policy
version, prior state, next state, and failure or decision reason. Re-running an
exact completed transition must return the existing receipt or fail on drift.

## Integration and failure rules

- Freeze the conversation time and source hashes before either evidence branch
  begins.
- Run acoustic and context collection independently after the freeze.
- Join only bundles that match the exact conversation, recording, document,
  speaker-reference set, and input hashes.
- Continue to review with explicit warnings when one non-required provider
  fails and the remaining evidence stays valid.
- Abstain when a required binding is missing, the evidence conflicts, or a
  confidence cap prevents a supported proposal.
- Preserve unresolved identities; don't synthesize a person to satisfy a
  denominator.
- Keep every source, account, tenant, capability, and as-of-time boundary in
  the joined evaluation.
- Prevent circular evidence by assigning lineage and independence groups to
  observations and derived projections.
- Never use a relationship proposed by the current evaluation as evidence in
  that evaluation.
- Keep transcript completion successful when identity enrichment stops or
  fails.

## Module boundaries

Keep this as a modular Python application in the current repository layout.
Do not create a new service or top-level package solely for this milestone.

Existing modules retain these responsibilities:

- `watch_transcriptions.py`: watched-file and backend orchestration only;
- `conversation_knowledge_store.py`: schema, migrations, and durable domain
  records;
- `conversation_knowledge_projection.py`: hash-bound sidecar projection;
- `conversation_knowledge_evidence.py`: scoped evidence snapshots, retrieval
  requests, and bundles;
- `conversation_identity_retrieval.py`: bounded identity evidence preparation;
- `participant_identity.py`: compatibility participant/contact candidate
  bundle;
- `acoustic_shadow_evidence.py`: source-bound non-authoritative acoustic
  evidence;
- `speaker_identity_preprocess.py`: clue and proposal validation plus
  confidence calibration; and
- `transcript_api.py`: transport and review-surface endpoints, not domain
  orchestration.

An implementation plan may add one focused orchestration module and one
decision-application module when those responsibilities become active. It
must not grow `watch_transcriptions.py` or `transcript_api.py` into the domain
service.

## Staged maturity campaign

1. Freeze identifiers, evidence schemas, state transitions, and negative
   actions.
2. Rehearse the current knowledge schema, projection, reconciliation, backup,
   and rollback on a private live-database copy.
3. Validate acoustic and context branches independently against exact real
   artifacts.
4. Join the branches in a selected-conversation shadow workflow and expose the
   evidence in the existing review console.
5. Compare context-only, acoustic-only, and combined results on the same frozen
   chronological cohort.
6. Plan live shadow enqueueing only after the selected-conversation join is
   replayable and safe under partial failure.
7. Plan accepted assignment and relationship projection only after live shadow
   quality, review burden, rollback, and tenant-isolation gates pass.
8. Consider policy-qualified automatic assignment only after a later unseen
   holdout demonstrates calibrated quality with zero unacceptable
   high-confidence errors.

## Measurement contract

The joined evaluation must report:

- eligible, entered, completed, partial, stopped, and failed conversations;
- eligible and covered speaker references;
- canonical candidate recall;
- enrolled acoustic recall and proposal precision;
- context-only, acoustic-only, and combined top-person correctness;
- wrong and high-confidence-wrong proposal counts;
- appropriate abstention and unresolved rates;
- duplicate-person fork count;
- tenant, account, provenance, and as-of-time completeness;
- deterministic replay and rollback results;
- provider and acoustic failure behavior; and
- review decisions and manual touches per conversation.

Provider availability, schema migration, service health, or passing tests do
not alone prove speaker-identity progress.

## Rollout and rollback

The first implementation must run from an explicit selected-conversation
action and remain disabled by default. It must use private runtime artifacts,
content-addressed receipts, and a private copy of the live database. Rollback
discards the shadow database and derived bundles; sidecars, legacy contacts,
speaker assignments, and the live schema remain authoritative and unchanged.

A later plan may propose live additive schema migration while retaining
sidecar authority. That plan must include a current backup, integrity check,
count reconciliation, round-trip projection, and proven rollback before any
authority change.

## Supersession and next authority

This note does not rewrite the historical constraints or results of Plans
0056 through 0058. It supersedes only the stale next-step pointer in Note 0052
by defining the cross-lane integration architecture after those plans closed.

[Plan 0059](../plans/0059-2026-08-08-speaker-identity-foundation-shadow-orchestration.md)
is the first bounded implementation plan for this architecture. It is
`PLANNED`, not activated, and grants no implementation or runtime authority.
