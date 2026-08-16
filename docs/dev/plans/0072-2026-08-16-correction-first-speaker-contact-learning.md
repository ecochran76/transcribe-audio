# Plan 0072 | Correction-first speaker, contact, and acoustic learning

State: PLANNED

Planning boundary: this artifact defines architecture and future bounded
implementation packets only. It does not authorize historical processing,
background workers, live-store migration, provider calls, contact or identity
mutation, biometric enrollment, dashboard publication, or deployment.

Lane: P09/P10

Cross-lane dependency: closed Plans 0025, 0029, 0030, 0063, 0064, and 0071;
open P08 user-scoped storage and P09 authenticated-console productization

Critical-Path Owner: primary agent

## Scope

Design the correction-first product path that processes eligible historical
and new conversations into reviewable conversation associations, participant
hypotheses, speaker identity proposals, contact/source-record candidates,
roles, relationships, voice samples, and versioned acoustic profiles. Present
the result in a dedicated authenticated dashboard tab where the operator can
confirm, correct, reject, split, merge, or defer each proposal.

The queue must display the actual original recording filename, not an enriched
artifact name; candidate calendar events and their independently scored
association strength; event attendees and other prepared participant
candidates; each diarized speaker and playable source-bound sample; proposed
identity, alternatives, supporting and contradicting evidence; and the exact
effect a review decision would have.

The architecture extends
[`docs/conversation-knowledge-storage-and-retrieval.md`](../../conversation-knowledge-storage-and-retrieval.md)
and [ADR 0002](../../adr/0002-use-a-user-scoped-conversation-knowledge-store.md).
It does not create a second contact database or make Graphiti the authority.

## Vision outcomes and maturity movement

| Capability | Current maturity | Target before live launch | Evidence |
| --- | --- | --- | --- |
| Conversation and calendar association | Level 2: candidate evidence exists in selected shadow workflows | Level 3: every eligible conversation receives replayable candidates, calibrated association strength, and safe fallback | Chronological historical/new shadow corpus with candidate recall, top-event correctness, calibration, abstention, and explicit stop reasons |
| Contact, role, and relationship gathering | Level 1/2: normalized source records and rebuildable profiles exist, but collection is not one continuous product loop | Level 2 before launch, then Level 3 after a separately accepted local projection policy | Deterministic source observation ingestion, person-resolution review, merge/split replay, temporal edge reconstruction, and duplicate-control measurements |
| Speaker identification | Level 2: acoustic and contextual branches run on reviewed cohorts but combined automatic acceptance remains withheld | Level 2 correction-driven shadow with representative coverage; Level 3 only for a separately validated acceptance band | Per-speaker correctness, candidate recall, calibration, high-strength error, review rate, appropriate abstention, and complete lineage |
| Acoustic learning | Level 2: governed references and versioned profiles exist for a small reviewed population | Level 2 scalable reviewed sample/profile lifecycle; Level 3 only after promotion and rollback gates pass | Sample provenance, quality and consent/authority checks, profile rebuild equality, source-disjoint evaluation, drift monitoring, rollback |
| Review queue and corrections | Level 1/2: React queue and several strict private review surfaces exist, but no unified identity-learning queue | Level 3 authenticated private workflow with idempotent decisions and rebuildable projections | Desktop/mobile browser proof, authorization tests, original-filename proof, decision round trips, supersession/split/merge/retraction tests |
| Reusable conversation knowledge | Level 2: reviewed observations and retrieval bundles are implemented in shadow form | Level 3 only after accepted decisions update local projections and improve later retrieval measurably | Before/after retrieval utility, deterministic projection replay, provenance completeness, tenant isolation, backup/restore |

This advances VISION outcomes 2, 3, 4, 6, 7, and 8. It improves the automatic
knowledge loop by turning review corrections into attributable local evidence
for later conversations without treating unreviewed predictions as truth.

## Current State

- Durable conversation and recording IDs, append-only processing evaluations,
  review decisions, source artifacts, and current-evaluation pointers already
  exist. Sidecars remain processing authority during migration.
- Schema version 3 already models canonical people, external identities,
  independently addressable source records, observations, claims,
  relationships, evidence snapshots, and rebuildable person/affinity
  projections in the user-scoped SQLite store.
- Host-owned retrieval already supports exact-first identity lookup, bounded
  lexical/semantic/relationship search, tenant/account/capability/as-of
  filtering, evidence-independence groups, partial-provider failures, and
  immutable evidence bundles.
- Reviewed canonicalization has installed six people, reviewed speaker
  bindings, governed voice references, and versioned profiles. Those records
  prove the lifecycle on a small reviewed population; they do not establish
  reliable general automatic identification.
- The React console already has a Review Queue and speaker/contact workspace,
  but P09 still lacks the complete share/auth boundary and a unified
  identity-learning queue. Existing acoustic reviews are campaign-specific.
- Plans 0064-0071 found useful acoustic candidates but inadequate contextual
  availability for the tested residual path. The new architecture must improve
  evidence gathering and review throughput without weakening abstention or
  relabeling those development results as validation.

## Stable architecture decisions

### 1. Identification is not authentication

This capability performs speaker identification: it estimates which known or
new person spoke in a conversation. It is not suitable for account access,
legal attestation, transaction approval, or any other authentication decision.
The UI, APIs, schemas, and metrics must not call a voice match an authenticated
identity.

### 2. One authority, several evidence classes

The private user-scoped conversation knowledge store remains the normalized
authority. Provider systems remain authoritative for their full records.
Content-addressed private storage retains media, exact voice samples, and
derived biometric payloads. Graphiti may receive only compact reviewed
projections and never raw audio, embeddings, transcripts, unreviewed
hypotheses, or provider bodies.

The host owns retrieval, source authorization, temporal policy, person
grouping, duplicate-evidence control, scoring, schema validation, effect
application, and audit receipts. App Intelligence receives only immutable
prepared evidence IDs and returns schema-constrained factor assessments and
proposals. It cannot browse providers, mutate records, choose its own people,
or emit an authoritative confidence probability.

### 3. Normalized correction-first domain

The following records are distinct and independently addressable:

| Record | Purpose and authority |
| --- | --- |
| `source_observation` | Immutable provider, transcript, calendar, reviewer, or system observation with source scope, times, hash, and independence group |
| `person` | Durable canonical internal identity with `provisional`, `reviewed`, `merged`, `split`, or `retired` resolution state |
| `external_identity` / `source_record` | Provider/account/tenant-scoped identity and contact view; never silently collapsed into a person |
| `person_alias` | Versioned alias evidence with validity, source, review state, and redirect lineage |
| `role_assertion` | Contextual, directional, time-bounded role claim; a role label alone does not create a person |
| `relationship_assertion` | Typed, directional, temporal, evidence-backed edge among people, organizations, projects, matters, events, recordings, and conversations |
| `conversation_association_candidate` | A recording/conversation-to-event hypothesis with independent evidence assessment and alternatives |
| `participant_hypothesis` | A claim that a person/source record likely participated; calendar attendance is candidate evidence, not proof of speech |
| `speaker_identity_proposal` | Versioned, non-authoritative speaker or utterance identity claim with alternatives, factor record, acoustic/context lineage, and abstention state |
| `speaker_review_decision` | Append-only confirm, correct, reject, not-listed, unresolved, mixed-speaker, split-label, or same-person decision that may supersede an earlier decision |
| `current_speaker_assignment` | Deterministic projection over applicable review decisions and later policy-qualified acceptances; never directly overwritten |
| `voice_sample` | Immutable audio segment reference with recording, utterance/speaker, timing, hash, quality, preparation lineage, review/consent authority, and exclusion state |
| `voice_profile_version` | Derived profile over an exact allowlist of reviewed samples, model/recipe version, evaluation status, active interval, and predecessor/successor lineage |
| `correction_event` | Append-only merge, split, redirect, retraction, reassignment, profile invalidation, or evidence correction with affected-record lineage |
| `processing_run` | Host supervisor ledger binding inputs, capabilities, policies, model/profile versions, budgets, outputs, failures, effects, and replay hashes |
| `identity_review_queue_item` | Rebuildable read model pointing to the current reviewable claims without duplicating their authority |

Current state is always a deterministic projection over immutable observations,
evaluations, decisions, and correction events. No correction deletes history.
Person merges preserve redirects and can be reversed; person splits rebind only
explicitly adjudicated records. Speaker reassignment invalidates downstream
voice samples and profiles whose authority depended on the superseded identity,
then queues deterministic rebuilds.

### 4. Source gathering and conversation association

A host-owned supervisor processes one durable conversation at a time through
capability-scoped adapters. The initial adapter registry may include local
transcript/knowledge data, Calendar, Google Workspace People/Contacts, Gmail,
Drive, configured Odollo contacts/leads/messages, and other already authorized
user-scoped evidence providers. “All available tools” means all configured,
healthy, authorized capabilities within declared tenant, temporal, query,
record, character, call, and latency budgets; it never means unbounded search.

The supervisor must:

1. bind the conversation, recording, actual original filename, artifact and
   media hashes, processing policy, `as_of` time, and provider capabilities;
2. generate calendar candidates from time proximity without selecting one as
   truth;
3. compare transcript topic/entity/action clues with event title,
   description, location, organizer, attendees, and bounded adjacent evidence;
4. persist factor assessments, credible alternatives, failures, and an
   Evidence Strength Score under a versioned calendar rubric;
5. collect exact attendee identities before broader contact, message,
   document, CRM, lexical, semantic, or relationship retrieval;
6. retain suspected attendees as source records and participant hypotheses,
   not automatically accepted people, relationships, or speakers; and
7. persist the immutable evidence bundle before any model inference.

Calendar association strength, person-link strength, and speaker-identity
strength remain separate. Scores are rubric values, not claimed probabilities.
Provider absence or failure is unknown evidence, not contrary evidence.

### 5. Acoustic processing and profile custody

Eligible speech segments may be extracted automatically into immutable
`voice_sample` candidates after diarization and quality checks. Samples are
linked to the original recording and preparation lineage and can be replayed
without changing the source artifact.

An unreviewed speaker proposal may be scored against active governed profiles,
but it cannot enroll, extend, or retrain a profile. A sample becomes eligible
for a person-bound profile only after the relevant person and speaker decision
is reviewed and its consent/authority and retention status allow biometric
use. Profile construction is append-only: build a new version from an exact
sample allowlist, evaluate it against frozen genuine/impostor/open-set data,
then promote or reject it. Retain the predecessor for rollback. Never train on
unreviewed predictions or on the evaluation holdout.

Raw audio, excerpts, embeddings, scores, and profile payloads remain private
and encrypted or filesystem-protected at rest. API responses expose only
authorized range-playback handles and bounded metadata. Retention, export,
deletion, biometric consent/authority, and jurisdiction-specific policy must
be decided before any live background collection begins.

### 6. Identity-learning review tab

Add one dedicated `Identity review` tab backed by an API read model rather
than campaign-specific static pages. Each conversation row must show:

- actual original recording filename, capture time, processing state, and
  source-artifact lineage;
- top calendar event plus alternatives, Evidence Strength Score/band, named
  positive and negative factors, and provider warnings;
- suspected calendar attendees and other participant hypotheses, each with
  source chips and person-link state;
- every diarized label, sample playback, proposed person, alternatives,
  acoustic and contextual evidence shown independently, and mixed/split
  warnings; and
- current review state, prior decision/supersession history, downstream effects
  that would be proposed, and any profile-rebuild consequence.

Required controls are confirm, choose another existing person, create a
reviewed provisional person, mark not listed, leave unresolved, reject event,
choose another event, mark no matching event, mark mixed speaker, group labels
as one person, split a label by utterance, correct source records/roles/
relationships, undo through a superseding decision, and defer. Search must
cover canonical people and still-separate provider records without merging by
name. Batch actions may defer or confirm only exact homogeneous cases and must
preview every effect.

The tab is externally reachable only through an authenticated, authorized,
private deployment with tenant-scoped sessions, CSRF protection, expiring
media access, audit logging, rate limits, and no raw-path disclosure. Public
share links and anonymous biometric review are prohibited. Local dogfood,
authenticated preview, and production deployment are separate gates.

### 7. Review, correction, and learning loop

Review submission writes a decision/correction event first. A deterministic
projector then proposes or applies only the authorized local effects, rebuilds
current assignments and person/relationship profiles, invalidates affected
voice/profile derivatives, and emits a receipt. Provider write-back remains a
separate field-owned proposal workflow and is not part of this plan's live
launch target.

Reviewed decisions produce calibration outcomes, not immediate training data.
A periodic evaluator builds frozen train, calibration, and source-disjoint
evaluation splits; reports correctness, error and abstention by condition;
checks person-merge/split and profile drift; and proposes a new rubric, model,
or profile version. Promotion requires explicit measured thresholds, rollback,
and a separately authorized release. Unreviewed model outputs never feed their
own training or count as ground truth.

### 8. Historical and new-conversation operation

Historical backfill and new-conversation processing use the same idempotent
pipeline and schemas. Backfill runs oldest-forward in bounded checkpointed
batches with resumable leases, per-provider budgets, failure isolation, and
an immutable run ledger. It preserves original filenames and source hashes.

For historical quality evaluation, `as_of` excludes later reviews and provider
facts. For present-day operational reprocessing, later accepted knowledge may
be used but is labeled hindsight evidence. Historical evidence, original
predictions, and original review decisions remain intact when a newer pipeline
version re-evaluates the conversation.

## Proposed execution graph

Every packet below requires a later implementation turn. Finishing one packet
does not authorize the next packet's live effects.

| Packet | Depends on | Bounded outcome | Write surface | Terminal gate |
| --- | --- | --- | --- | --- |
| A0 architecture freeze | this plan | Version the domain, correction, privacy/threat, API, adapter, and supervisor contracts | Docs, ADRs, schemas, redacted fixtures/tests | Architecture audit passes; unresolved privacy decisions block A1 |
| A1 normalized identity ledger | A0 | Add append-only assertions, corrections, projections, merge/split/reversal, and schema migration on disposable/private copies | Product schema/modules/tests; no live migration | Migration/rollback/rebuild/reconciliation pass |
| A2 evidence supervisor | A1 | Compose capability-scoped provider adapters, processing ledger, calendar association, contact gathering, and partial-failure behavior | Product modules/tests and private shadow artifacts | Exact replay, budget/isolation tests, zero provider writes |
| A3 biometric custody | A1 | Add voice-sample inventory, eligibility, profile version/invalidation/rebuild, retention hooks, and governed storage | Product modules/tests and private derived artifacts | Source/sample/profile lineage, rollback, access and deletion tests pass |
| A4 review API/read model | A1-A3 | Build identity queue projection and reviewed decision/effect-preview endpoints | Local API/schema/tests only | Idempotency, authorization, concurrency and stale-decision rejection pass |
| A5 dashboard tab | A4 | Implement original-filename-bearing Identity review workflow | Frontend/API tests and authenticated local preview | Desktop/mobile/accessibility/audio/decision browser proof passes |
| A6 bounded shadow | A2-A5 | Process a small oldest-forward historical batch and new arrivals without applying identity/profile conclusions | Private run/evidence/queue records | Coverage, failure and privacy receipts complete; no accepted effects |
| A7 reviewed local projection | A6 | Apply reviewed decisions to local people, assignments, roles, relationships, and eligible profile rebuilds | User-scoped local store after backup/rehearsal gate | Exact apply/replay/rollback and deterministic rebuild pass |
| A8 calibration and promotion | A7 | Measure learning value on frozen source-disjoint data and define any policy-qualified automatic band | Private evaluation and tracked aggregate decision | Accept/refine/withhold; no automatic promotion by test success |
| A9 authenticated live launch | A4-A8 | Deploy scheduled historical/new processing and the private externally reachable tab | Installed services and authenticated deployment | Security/privacy/recovery/load/product acceptance all pass |

A1, A2, and A3 may be designed in parallel after A0, but schema integration is
serialized through A1. A4 joins the data paths. A5 may proceed against redacted
fixtures while A2/A3 finish, then must integrate with A4. A6-A9 are serialized.

## Bounds for future execution

- `max_work_unit_attempts`: 2 per packet before local reframe.
- `max_review_rework_cycles`: 1 closed-world cycle per packet.
- `max_broad_review_discovery_passes`: 1 for the whole campaign.
- `max_historical_shadow_batch`: 25 conversations for the first A6 run.
- `max_new_conversation_shadow_window`: 7 days for the first A6 run.
- `max_provider_retries`: 1 retry only for transient, idempotent reads; no
  retry for authorization, tenant, schema, or privacy failures.
- `max_model_reference_repairs`: 1 reference-only repair per model phase.
- `max_profile_rebuilds_per_correction`: 1 deterministic rebuild attempt,
  followed by fail-closed review if equality or evaluation fails.
- `max_hardening_checkpoints_without_outcome_progress`: 2.
- `checkpoint_interval`: after every packet and before private corpus access,
  live-store migration, background scheduling, biometric activation,
  authenticated publication, or automatic-policy promotion.
- `delegation`: `not_spawned` for this planning turn; future packets follow
  then-current system and repo governance.

## Non-Goals

- No claim of speaker authentication, dependable Level 4 identity, or legal
  identity verification.
- No automatic person merge from a shared name, calendar attendance, role,
  organization, acoustic similarity, or model assertion.
- No automatic profile enrollment from predicted identities.
- No provider contact mutation or write-back in this plan's launch target.
- No public/anonymous dashboard, raw filesystem path, unrestricted audio URL,
  or Graphiti storage of private or unreviewed content.
- No silent rewrite of transcripts, diarization, evaluations, decisions,
  source records, relationships, or historical artifacts.
- No self-training on unreviewed predictions and no evaluation-set leakage.
- No implementation, live processing, provider retrieval, deployment, or
  runtime mutation in this planning turn.

## Acceptance criteria

- The architecture has one normalized person/source-record/role/relationship
  model and one append-only correction ledger; it does not fork the existing
  conversation knowledge authority.
- Every queue item and downstream record retains conversation ID, recording ID,
  actual original recording filename, source artifact/media hashes,
  processing-run ID, model/rubric/profile versions, and evidence lineage.
- Calendar association, person linking, and speaker identity are separately
  scored under versioned rubrics with factors and alternatives.
- Suspected calendar attendees become source observations and participant
  hypotheses without becoming proof of presence or speech.
- All configured evidence adapters are host-owned, capability/tenant/time/budget
  scoped, partial-failure tolerant, and unable to write providers.
- Confirm, correct, reject, not-listed, unresolved, mixed, group, split,
  merge, and reversal decisions replay idempotently and preserve history.
- Person split/merge and speaker reassignment deterministically invalidate and
  rebuild only affected projections, samples, and profiles.
- Voice profile versions cite only reviewed eligible samples and have exact
  predecessor, evaluation, activation, rollback, retention, and deletion state.
- The authenticated dashboard proves original-filename display, safe playback,
  complete alternatives/evidence, stale-decision rejection, keyboard and
  mobile usability, and no anonymous access.
- The first shadow campaign reports pipeline yield, candidate recall,
  correctness, calibration, high-strength errors, abstention, review load,
  provider yield/failure, latency, duplicate control, and knowledge integrity.
- No automatic acceptance or live background processing begins until the
  relevant source-disjoint quality, privacy, security, backup/restore,
  rollback, load, and user-workflow gates pass.

## Validation for future packets

- Schema migration, rollback, backup/restore, legacy compatibility, immutable
  ledger, supersession, merge/split/reversal, and deterministic projection
  rebuild tests.
- Tenant/account/capability/as-of/hindsight, exact-first retrieval,
  evidence-independence, partial-provider failure, budget, redaction, and
  adapter-contract tests.
- Audio/sample hash, range access, quality, eligibility, retention/deletion,
  profile versioning, invalidation, rebuild, evaluation-split, promotion, and
  rollback tests.
- API authorization, CSRF, tenant isolation, optimistic concurrency,
  idempotency, effect preview, audit, pagination, filtering, and stale queue
  projection tests.
- Redacted component tests plus authenticated desktop/mobile browser proof for
  queue states, original filenames, media playback, evidence, every decision
  control, undo/supersession, and inaccessible anonymous routes.
- Oldest-forward chronological shadow evaluation with frozen exclusions and
  aggregate tracked metrics; private exact receipts for any live rehearsal.
- Focused tests, full suite, planning audit, CodeGraph status/readback,
  `git diff --check`, clean commit, push, and upstream equality at each
  integrated packet.

## Planning exit and next authority

This planning slice is complete when Plan 0072, the evergreen architecture,
the canonical vocabulary, ROADMAP, and RUNBOOK agree and pass documentation
validation. The recommended next slice is A0 only: version the schemas, ADRs,
privacy/threat model, API contracts, redacted fixtures, and deterministic
tests. A0 must remain non-live and must not start historical processing or
collect biometric data.
