# Plan 0072 | Correction-first speaker, contact, and acoustic learning

State: OPEN

Planning boundary: this artifact defines architecture and bounded execution
packets. The 2026-08-16 A6 checkpoint separately authorizes private reads and
processing for the frozen 25-conversation historical cohort and one seven-day
arrival window, plus private A6 run/evidence/queue writes. It does not
authorize background workers, live-store migration, provider calls, contact
or identity mutation, biometric enrollment, dashboard publication, or
deployment.

Execution authority: the user goal `execute plan 72` opens ordinary bounded
implementation under repo policy. Packet-specific private, live, provider,
publication, biometric, migration, and deployment gates remain in force.

Lane: P09/P10

Cross-lane dependency: closed Plans 0025, 0029, 0030, 0063, 0064, and 0071;
open P08 user-scoped storage and P09 authenticated-console productization

Critical-Path Owner: primary agent

## Scope

Design the correction-first product path that processes eligible historical
and new conversations into reviewable conversation associations, participant
hypotheses, speaker identity proposals, contact/source-record candidates,
roles, relationships, voice samples, and versioned acoustic profiles. Present
the result in dedicated Authelia-protected Identity Review and People tabs
where the operator can confirm, correct, reject, split, merge, or defer each
proposal.

The queue must display the actual original recording filename, not an enriched
artifact name; candidate calendar events and their independently scored
association strength; event attendees and other prepared participant
candidates; each diarized speaker and playable source-bound sample; proposed
identity, alternatives, supporting and contradicting evidence; and the exact
effect a review decision would have.

The architecture extends
[`docs/conversation-knowledge-storage-and-retrieval.md`](../../conversation-knowledge-storage-and-retrieval.md)
and [ADR 0002](../../adr/0002-use-a-user-scoped-conversation-knowledge-store.md).
The implemented A5 review surface is documented in
[`docs/correction-first-identity-review.md`](../../correction-first-identity-review.md).
It does not create a second contact database or make Graphiti the authority.

The accepted answers from the bounded design interview are frozen in
[Note 0058](../notes/0058-2026-08-16-plan0072-grilled-architecture.md). That
note is the detailed decision authority for reconciliation, ontology,
retention, confidence, transcript correction, review behavior, and staged
launch. If this campaign plan summarizes a decision more narrowly, Note 0058
controls until a later explicit plan revision supersedes it.

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
- Schema version 8 extends the A1-A3 identity, transcript, and biometric
  ledgers and A4 supervisor history with the replaceable identity-review queue
  plus append-only review submissions and effect previews.
- Host-owned retrieval already supports exact-first identity lookup, bounded
  lexical/semantic/relationship search, tenant/account/capability/as-of
  filtering, evidence-independence groups, partial-provider failures, and
  immutable evidence bundles.
- Reviewed canonicalization has installed six people, reviewed speaker
  bindings, governed voice references, and versioned profiles. Those records
  prove the lifecycle on a small reviewed population; they do not establish
  reliable general automatic identification.
- The React console now has separate Identity Review and People views over
  local rebuildable projections. They preserve the dashboard's existing
  Authelia route and add no second authentication layer. The authorized A6
  campaign has populated its isolated private queue; accepted decisions and
  the authenticated live-route proof remain gated.
- Plans 0064-0071 found useful acoustic candidates but inadequate contextual
  availability for the tested residual path. The new architecture must improve
  evidence gathering and review throughput without weakening abstention or
  relabeling those development results as validation.

## Current execution state

Packets A0-A5 are closed. A6 is active under the explicit 2026-08-16
checkpoint for private reads and processing of the 25 oldest eligible
historical conversations plus one seven-day stabilized-arrival window. That
checkpoint authorizes only private A6 run, evidence, and queue records.
Provider access, live migration, accepted identity/profile effects, deletion,
deployment, and background scheduling remain unauthorized.

The 2026-08-29 operator feedback opens one bounded A6-R1 read-side Contacts
refinement on the existing authenticated dashboard. It may deploy API and UI
code that projects already-local canonical people, local contacts, and exact
operator-reviewed speaker labels into one compact directory. Each authority
class must remain visibly separate. This refinement performs no provider call
or write, schema migration, contact/person merge, speaker-assignment apply,
biometric effect, or accepted identity mutation.

| Field | A6-R1 control |
| --- | --- |
| Bounded outcome | Replace the empty People pane with a compact Contacts directory that connects existing person/contact records to exact prior speaker-review evidence without claiming they are already reconciled. |
| Vision movement | Advances outcomes 3, 7, and 8; contact/identity visibility moves from Level 1 built-but-empty UI to Level 2 usable read-side projection on live local evidence. It does not establish reviewed-learning effects or Level 3 contact management. |
| Write surface | Existing `/api/people` read model, React Contacts view, focused tests, plan/roadmap/runbook records, and the already-authorized dashboard deployment. |
| Inputs | Existing schema-v8 local store, current-person profiles, local `contacts` rows, A6 queue artifacts, and exact validated operator gold already used by the Review Queue. |
| Validation | Focused API/workflow tests, frontend build, read-only live API counts, desktop/mobile Agent Browser proof, no decision/contact POST during QA, service/process readback, planning audit, and git checks. |
| Terminal condition | Contacts is non-empty, source classes and unresolved links are explicit, prior speaker-review occurrences are inspectable, mobile remains usable, and all prohibited effect counts remain zero. |

A6-R1 delivered on 2026-08-29:

- `/api/people` now projects 6 canonical people, 2 unlinked local contacts,
  and 35 exact operator-reviewed speaker names with 71 review appearances;
- the compact Contacts view supports search, source-class filters, client-side
  sorting, a dense master/detail layout, and a mobile contact picker;
- five records with exact display-name overlap expose review-required related
  records, but no merge or contact/person link was inferred or written; and
- focused tests, the frontend build, live desktop/mobile Agent Browser proof,
  GET-only network evidence, service/process readback, the active planning
  audit, and git checks passed. This completes only the Level 2 read-side
  refinement; Plan 0072 A6 and all Level 3 contact-management effects remain
  open or gated as stated above.

The 2026-08-29 operator request also opens one bounded A6-R2 attendee-contact
ingestion slice. It may read calendar-attendee evidence from the complete
current transcript corpus, perform read-only exact-email lookups against
already-configured GWS and Odollo contact sources, and write replayable local
contact projections plus private run receipts. It may also extend the existing
Contacts read model and authenticated dashboard to show those projections.

| Field | A6-R2 control |
| --- | --- |
| Bounded outcome | Account for every unique calendar-attendee email in the current corpus as an inserted, enriched, unchanged, conflicted, or explicitly excluded local contact candidate, with compact recording provenance in Contacts. |
| Vision movement | Advances outcomes 3, 7, and 8. Calendar-attendee contact gathering moves from Level 0/1 ad hoc evidence to Level 2 manual/shadow ingestion on the real corpus. It does not establish automatic recurring ingestion or Level 3 identity resolution. |
| Write surface | Existing local `contacts` rows, a private user-scoped ingest receipt, the `/api/people` read model, React Contacts view, focused tests, and plan/roadmap/runbook records. |
| Inputs | Exact normalized attendee emails and names from current transcript calendar metadata; configured GWS Contacts/Other Contacts and Odollo profiles through read-only calls only. |
| Identity boundary | Exact email is a source-observation join key, not proof that a contact is a canonical person or a diarized speaker. Shared/role addresses, duplicate existing rows, conflicting names, and non-exact matches remain review-required. No name-only/fuzzy link is applied. |
| Validation | Corpus accounting equality, exact-match provenance, idempotent replay, before/after receipt evidence, focused tests, frontend build, live API counts, Agent Browser desktop/mobile proof, service/process readback, and zero provider writes/person merges/speaker applies. |
| Terminal condition | Every unique attendee email is accounted for, Contacts exposes human-friendly occurrence evidence, a second identical run is a no-op, and all prohibited-effect counters remain zero. |

A6-R2 explicitly excludes provider writes, canonical-person creation or merge,
speaker assignment, biometric/profile effects, fuzzy identity reconciliation,
background scheduling, and automatic acceptance. Those remain separately
gated even when an attendee email exactly matches a configured contact source.

A6-R2 delivered on 2026-08-29:

- all 186 unique attendee emails across 1,063 calendar appearances are now
  represented in local Contacts; 185 contacts were inserted and one existing
  contact was enriched without changing its operator-entered label or external
  reference;
- 159 attendee emails received one or more configured exact-email matches from
  9,337 GWS/Odollo rows read in 12 bounded provider calls, with zero provider
  writes, person merges, or speaker-assignment effects;
- every applied mutation has a private before/after receipt and guarded undo;
  a final provider-backed replay classified all 186 contacts as unchanged;
- `/api/people?limit=500` returns all 228 directory records, including all 186
  attendee contacts and all 1,063 appearances, and the compact Contacts detail
  exposes contact methods, organizations, recording files, dates, calendar
  events, and exact-email source observations; and
- 76 focused backend tests, the frontend production build, live service/API
  readback, and desktop/mobile Agent Browser review pass. This completes the
  bounded Level 2 manual/shadow ingestion slice only; recurring automation and
  identity resolution remain gated.

The 2026-08-29 operator request opens one bounded A6-R3 relationship-and-role
discovery slice. It may refresh the already-authorized exact-email contact
observations through bounded read-only provider calls, derive replayable graph
hypotheses from provider-declared roles/affiliations and calendar co-invitation,
and expose those hypotheses in the authenticated Contacts view. Discovery must
remain distinct from reviewed graph facts.

| Field | A6-R3 control |
| --- | --- |
| Bounded outcome | Give each eligible contact a compact, evidence-bearing view of proposed contextual roles, organization affiliations, and recurring calendar co-invitation relationships that can later support speaker deduction and conversation contextualization. |
| Vision movement | Advances outcomes 3, 4, 6, 7, and 8. Relationship/role discovery moves from Level 0/1 empty projections and design contracts to Level 2 deterministic shadow hypotheses over the real corpus. It does not establish reviewed graph authority or Level 3 automatic acceptance. |
| Write surface | Calendar-contact enrichment metadata, a deterministic read-only discovery projection, the `/api/people` read model, React Contacts detail, focused tests, and plan/roadmap/runbook records. No knowledge-schema migration is included. |
| Inputs | Current local attendee contacts and calendar appearances; exact-email GWS/Odollo source observations; provider-declared organization/title fields; existing accepted ontology and evidence boundaries. |
| Semantic boundary | A title is a contextual role hypothesis, provider-declared organization membership is an affiliation hypothesis, and repeated calendar co-invitation is a symmetric interaction observation. None proves employment, personal relationship, meeting presence, speech, person identity, or speaker identity. |
| Validation | Deterministic IDs and replay, temporal bounds, symmetric-pair deduplication, source/evidence provenance, shared-address exclusion, focused tests, frontend build, corpus accounting, live API/browser proof, service/process readback, planning audit, and zero provider writes/person merges/speaker applies/accepted relationship effects. |
| Terminal condition | Contacts exposes non-empty role and relationship hypotheses with human-friendly counterpart/evidence detail; identical inputs replay identically; every hypothesis states why it is not yet accepted; and prohibited-effect counters remain zero. |

A6-R3 explicitly excludes canonical-person creation or merge, speaker assignment,
accepted role/relationship ledger events, ontology mutation, biometric/profile
effects, Graphiti writes, provider writes, background scheduling, and automatic
acceptance. Review/apply controls for graph hypotheses are a later bounded slice.

A6-R3 delivered on 2026-08-29:

- the refreshed local contact projection retains provider-declared title,
  organization, department, current-state, exact-email source, and observation
  provenance without changing an external provider;
- deterministic discovery over 186 attendee contacts exposes 3 contextual-role,
  59 affiliation, and 279 recurring co-invitation hypotheses across 103 contacts,
  with a two-invitation minimum and stable symmetric relationship IDs;
- `/api/people` and the compact Contacts detail expose human-readable graph leads
  as `Needs review`, while accepted `roles` and `relationships` remain empty;
- replay retains the same input watermark and candidate IDs, and all accepted
  effect, provider-write, person-merge, and speaker-assignment counters remain
  zero; and
- 79 focused backend tests, Python compilation, the frontend production build,
  live service/API readback, and an Agent Browser desktop Contacts review pass.
  This completes only Level 2 shadow discovery; reviewed graph decisions and
  speaker/context consumers remain later slices.

| Field | A0 control |
| --- | --- |
| Bounded outcome | Freeze versioned domain, correction, privacy, threat, review, adapter, and supervisor contracts. |
| Write surface | Contract module, ADR/reference docs, redacted fixtures, tests, plan/roadmap/runbook state. |
| Inputs | This plan, Note 0058, VISION, ADR 0002, evergreen storage architecture, existing contract seams. |
| Validation | Focused contract tests, full suite, planning audit, link checks, CodeGraph status, and git checks. |
| Terminal condition | Contract versions and fixtures agree, no privacy decision remains unresolved, and A0 records zero live/provider/biometric effects. |

A0 delivered:

- `identity_learning_contracts.py`, which exposes one small catalog and
  validation seam;
- [ADR 0003](../../adr/0003-freeze-correction-first-identity-learning-contracts.md)
  and the [contract reference](../../correction-first-identity-learning-contracts.md);
- redacted contract, artifact, and threat-control fixtures under
  `docs/dev/fixtures/plan-0072-a0/`; and
- deterministic tests for original filenames, private-field rejection,
  adapter scope and budgets, stale-safe review, zero-effect previews,
  biometric custody, correction bounds, and supervisor limits.

Validation passed: 9 focused contract tests, the 1,132-test full suite, the
active planning audit, link-target checks, `git diff --check`, and a fresh
CodeGraph readback with 349 files, 10,164 nodes, 34,499 edges, zero pending
changes, and no worktree mismatch. Commit `d5bee6a` records the implementation.
All A0 provider, historical, identity, biometric, migration, publication,
deployment, and external effect counts are zero.

| Field | A1 control |
| --- | --- |
| Bounded outcome | Add the append-only identity/contact/ontology/correction ledger and rebuildable projections on synthetic disposable stores. |
| Write surface | Knowledge schema v4, ledger module, tests, reference docs, plan/roadmap/runbook state. |
| Inputs | A0 contracts, this plan, Note 0058, schema v3, and the verified contact/role sequencing note. |
| Validation | Migration/rollback, deterministic rebuild, corrections, merge/split/reversal, exact dedup/linking, privacy, conflict, full-suite, planning, CodeGraph, and git checks. |
| Terminal condition | All A1 behavioral gates pass with zero live migration, private-directory read, provider write, biometric, publication, deployment, or external effects. |

A1 delivered knowledge schema v4 and
[`identity_learning_ledger.py`](../../../identity_learning_ledger.py), with a
focused [ledger reference](../../correction-first-identity-ledger.md). The v4
migration adds immutable event and ontology tables plus replaceable people,
source-record, hashed-external-identity, role, relationship, and reconciliation
projections. It preserves v1-v3 records and rolls back additively to v3. The
frozen Plan 0063 rehearsal remains explicitly pinned to its original v3
authority.

Synthetic disposable-store tests prove content-hashed idempotent append,
hierarchical and inverse ontology terms, simultaneous roles, directional and
conflicting relationships, source/role/relationship corrections, exact-scope
deduplication, verified person-specific exact linking, shared-address
exclusion, conflict-preserving proposals, deterministic rebuild, explicit
merge/split, reversal, atomic replay failure, and v4 rollback. Raw email and
phone values are rejected from source-record ledger events; persisted external
identity values are hashes. Provider-write count remains zero.

A1 did not read a private baseline directory, migrate a live store, call a
provider, collect biometric material, publish a dashboard, deploy a service,
or write Graphiti memory.

| Field | A2 control |
| --- | --- |
| Bounded outcome | Add scoped terminology, immutable raw/normalized transcript generations, reviewed span corrections, transcript-only semantic maps, dual-layer search, and bounded identity cascades. |
| Write surface | Knowledge schema v5, transcript correction ledger, tests, redacted fixtures, reference docs, plan/roadmap/runbook state. |
| Inputs | A0 correction contracts, A1 schema/identity lineage, this plan, Note 0058, and synthetic fixture text. |
| Validation | Migration/rollback, scope precedence/conflicts, hint eligibility, correction supersession, raw preservation, normalization, search/reindex provenance, semantic lineage, cascade bounds, full-suite, planning, CodeGraph, and git checks. |
| Terminal condition | All A2 behavioral gates pass with zero historical/private processing, live migration, provider, biometric, publication, deployment, or external effects. |

A2 delivered knowledge schema v5 and
[`transcript_correction_ledger.py`](../../../transcript_correction_ledger.py),
with a focused [transcript-learning reference](../../correction-first-transcript-learning.md)
and redacted fixtures under `docs/dev/fixtures/plan-0072-a2/`. The v5 migration
is additive over A1 and rolls back to v4 without changing A1 identity history.
Authoritative v5 artifacts are append-only; the selected-normalized projection
and raw/normalized FTS index are derived and replaceable.

Synthetic disposable-store tests prove reviewed-only terminology hints with
version/content-hash pinning, conversation-to-global scope precedence,
equal-scope review conflicts, explicit decision supersession, non-destructive
raw/normalized replay, exact span and semantic lineage, dual-layer search and
reindex receipts, two correction passes, one identity requeue, and a manual
stop on the second cascade. The redacted SESO fixture remains chemistry-scoped.

A2 validation passed 19 focused correction/store/rollback tests, the
1,149-test full suite, Python compilation, active and goal-only planning
audits, internal-link checks, `git diff --check`, and a fresh CodeGraph
readback with 353 indexed files, 10,316 nodes, and 35,384 edges.

A2 did not process a private conversation, migrate a live store, activate a
provider hint, call or write a provider, collect biometric material, publish a
dashboard, deploy a service, schedule a worker, or write Graphiti memory. A3
is the next safe packet but retains its separate biometric custody and private
artifact gates.

| Field | A3 control |
| --- | --- |
| Bounded outcome | Add governed voice-sample custody, soft anonymous clustering, exclusions, previewed deletion, profile families/versions, invalidation, rollback, and deterministic rebuild on synthetic temporary stores. |
| Write surface | Knowledge schema v6, custody ledger, synthetic private test objects, redacted fixtures, tests, reference docs, and plan/roadmap/runbook state. |
| Inputs | A0 biometric contracts, A1 identity lineage, this plan, Note 0058, and generated non-voice byte strings. |
| Validation | Migration/rollback, private access, sample/cluster/profile lineage, material rescore, deterministic rebuild, all deletion scopes, stale preview, full suite, planning, CodeGraph, and git checks. |
| Terminal condition | A3 behavioral gates pass with zero real voice, private-corpus, historical, live-migration, provider, enrollment, publication, deployment, or external effects. |

A3 delivered knowledge schema v6 and
[`biometric_custody_ledger.py`](../../../biometric_custody_ledger.py), with a
focused [biometric-custody reference](../../correction-first-biometric-custody.md)
and redacted fixtures under `docs/dev/fixtures/plan-0072-a3/`. The additive v6
migration rolls back to v5 without changing A1 identity or A2 transcript
history. Immutable tables record samples, sample events, cluster versions,
soft memberships, cluster events, material-rescore receipts, profile families
and versions, profile events, rebuild receipts, and deletion tombstones.

Synthetic temporary-store tests prove restrictive private-root and object
permissions, source/range/hash/preparation lineage, person-unbound unreviewed
samples, reviewed and consented profile allowlists, multiple profile families,
pending activation, explicit supersession and rollback, exact and drifted
rebuild receipts, reversible cluster membership, and material-only requeue
after a confirmed anchor. They also prove previewed exclusion/deletion for
sample, cluster, profile, recording, and person scopes; stale-preview refusal;
transactional byte quarantine; backup dispositions; minimal tombstones; and
idempotent replay.

A3 validation passed 36 focused custody/store/profile/evidence tests, the
1,166-test full suite, Python compilation, the active planning audit, internal
link and fixture checks, `git diff --check`, and a fresh CodeGraph readback
with 355 indexed files, 10,402 nodes, and 35,862 edges. No real voice, private
corpus, historical conversation, live store, provider, external benchmark,
named enrollment, dashboard, deployment, worker, or Graphiti write was
accessed or changed.

| Field | A4 control |
| --- | --- |
| Bounded outcome | Add a deterministic zero-effect supervisor with exact run/stage history, capability-scoped adapter receipts, calendar/purpose/participant hypotheses, evidence-pillar scores, calibration history, budgets, and partial-failure isolation. |
| Write surface | Knowledge schema v7, supervisor module, tests, redacted fixtures, reference docs, and plan/roadmap/runbook state. |
| Inputs | A0 supervisor/adapter contracts, A1-A3 ledgers, this plan, Note 0058, and synthetic fixture evidence. |
| Validation | Migration/rollback, exact replay, source-scope and budget enforcement, transient retry bounds, partial-failure isolation, score lineage, contradiction/duplicate caps, calibration threshold, full-suite, planning, CodeGraph, and git checks. |
| Terminal condition | A4 gates pass with zero provider calls/writes, private corpus reads, historical processing, accepted identity/profile effects, publication, deployment, or external effects. |

A4 delivered knowledge schema v7 and
[`identity_evidence_supervisor.py`](../../../identity_evidence_supervisor.py),
with an [evidence-supervisor reference](../../correction-first-evidence-supervisor.md)
and redacted fixtures under `docs/dev/fixtures/plan-0072-a4/`. The additive v7
migration rolls back to v6 without changing A1-A3 history. Immutable tables
record runs, sequential stage events, adapter exchanges, calendar candidates,
purpose and participant hypotheses, pillar assessments, re-score chains,
calibration outcomes, and calibration snapshots.

Synthetic tests prove exact run and stage replay, original-filename and source-
hash retention, tenant/account/profile/capability scope binding, cumulative
record/character/call/latency budgets, one transient retry, visible partial
failure without loss of successful observations, and zero provider writes.
They also prove four separately visible Evidence Strength pillars, conservative
caps for material contradiction or duplicated independence groups, immutable
re-score lineage, nonbinding hypotheses, and withheld empirical likelihood
until 30 source-disjoint reviewed outcomes exist in the relevant band.

A4 validation passed 43 focused supervisor/store/profile/evidence/biometric
tests, the 1,174-test full suite, Python compilation, the active planning
audit, internal-link and fixture checks, `git diff --check`, and a fresh
CodeGraph readback with 357 indexed files, 10,477 nodes, and 36,210 edges. No
provider was called, no private corpus or conversation was read, and no
historical, live-migration, accepted identity/profile, publication, deployment,
worker, Graphiti, or external effect occurred.

A1 validation passed 33 focused ledger/store/profile/evidence/private-
rehearsal tests, the 1,139-test full suite, Python compilation, active and
goal-only planning audits, internal-link checks, `git diff --check`, and a
fresh CodeGraph readback with 351 files, 10,236 nodes, and 34,943 edges.

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

Before historical speaker processing, ingest a read-only baseline from every
configured authorized contact directory. Preserve source records, auto-dedup
only exact provider/account/type/record IDs, auto-link to one provisional
person only through a non-conflicting exact person-specific email or verified
phone, and route every fuzzy or conflicting reconciliation to review. Shared
and role addresses never auto-link to people. Clean source records do not each
require review, and reviewed local overrides never overwrite provider fields.

Roles and relationships use a shared versioned hierarchy but remain distinct
assertions: entity-to-entity relationship edges and contextual roles in a
conversation, event, organization, project, matter, or time. People may hold
many simultaneous roles and relationships; conflicting assertions coexist
until reviewed. Organization hierarchy, inverse/directional edge rules, and
reviewed ontology extensions are first-class rather than flattened strings.

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

Unreviewed samples and embeddings remain private, person-unbound, and retained
indefinitely for now until explicit deletion or a later policy revision.
Anonymous recurring-voice clustering is automatic, but memberships remain
soft and reversible. Confirming one member may re-score and materially requeue
related samples; it never assigns the cluster in bulk. A person may have
multiple acoustic profile families for different conditions.

Raw audio, excerpts, embeddings, scores, and profile payloads remain private
and encrypted or filesystem-protected at rest. API responses expose only
authorized range-playback handles and bounded metadata. Per-person and
per-recording biometric exclusions plus deletion by sample, cluster, profile,
recording, or person must invalidate dependent derivatives and preserve a
minimal audit tombstone. Local processing is the default; only an opt-in,
bounded pseudonymous external challenger benchmark may be designed before
measurable lift justifies any broader cloud path.

### 6. Identity Review and People tabs

Add dedicated `Identity Review` and `People` tabs backed by API read models
rather than campaign-specific static pages. Identity Review is
conversation-first, with cross-conversation acoustic-cluster context. People
is the authoritative editing view for canonical/provisional people, source
records, aliases, organizations, roles, relationships, clusters, profiles, and
correction/merge/score history. Each conversation row must show:

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

Freeform comments are immutable input for richer semantic corrections. App
Intelligence may propose structured derivatives into a separate correction
queue, but only reviewed structured derivatives become learning labels.

The tabs use the dashboard's existing Authelia-protected route as the sole
initial authentication gate; this plan adds no Google OAuth, local login, or
step-up authentication or second security layer. Preserve existing request
protections, stale-write rejection, bounded media access, audit history, and
no raw-path disclosure. Public share links and anonymous biometric review are
prohibited.

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

Host-computed 0-100 Evidence Strength may increase or decrease and is not a
probability. Preserve original scores and append re-scores with evidence,
rubric, and model lineage. Show calendar, person-link, contextual, acoustic,
and combined pillars separately; contradictions cap the combined result.
Empirical Calibrated Likelihood appears only with at least 30 source-disjoint
reviewed outcomes in the relevant band and must show sample size, interval,
and evaluation version.

Evaluate accumulated reviews weekly. A candidate rubric or model requires at
least 25 new reviewed speaker decisions or a material correction. Automatic
named acceptance remains disabled until at least 100 varied source-disjoint
reviewed speaker outcomes demonstrate at least 99% precision in the proposed
band, safe abstention, and no systematic high-strength failure. Person merges
and splits always remain reviewed.

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

Processing begins asynchronously only after transcript artifacts stabilize;
the watcher does not perform model/provider enrichment inline. Hard daily
budgets reserve capacity for new conversations. Above 500 actionable queue
items, cheap normalization, metadata, sample extraction, and clustering may
continue while expensive enrichment throttles. One transient idempotent
provider retry is allowed; subsequent partial failure remains visible and does
not discard other evidence.

Transcript correction is a first-class layer before identity review. Preserve
raw ASR and diarization, store versioned span-level proposals, and use accepted
normalized transcripts downstream. A scoped terminology registry handles
canonical terms, definitions, aliases, pronunciations, and ASR confusions; for
example, SoyLei `CISO` to `SESO` is a chemistry-scoped ASR confusion rather
than a global replacement. Run at most one pre-identity and one post-identity
correction pass; one material correction/identity cascade may requeue, after
which the same processing version stops for manual resolution.

Each conversation has a transcript-only semantic map, an enriched provisional
draft, and an accepted reviewed readout. Historical evaluation retains both a
contemporaneous as-of assessment and a separately labeled current assessment
that may use accepted hindsight.

## Proposed execution graph

Every packet below requires a later implementation turn. Finishing one packet
does not authorize the next packet's live effects.

| Packet | Depends on | Bounded outcome | Write surface | Terminal gate |
| --- | --- | --- | --- | --- |
| A0 architecture freeze | this plan | Version the domain, correction, privacy/threat, API, adapter, and supervisor contracts | Docs, ADRs, schemas, redacted fixtures/tests | Architecture audit passes; unresolved privacy decisions block A1 |
| A1 identity/contact/ontology ledger | A0 | Add append-only identities, source records, roles, hierarchical relationships, corrections, projections, merge/split/reversal, and baseline directory reconciliation on disposable/private copies | Product schema/modules/tests; no live migration or provider write | Migration/rollback/rebuild/reconciliation and dedup pass |
| A2 terminology and transcript correction | A1 | Add terminology registry, raw/normalized transcript generations, span corrections, semantic map, and bounded correction cascades | Product schema/modules/tests and redacted fixtures | Scope precedence, non-destructive replay, reindex, and cascade bounds pass |
| A3 biometric custody and clustering | A1 | Add voice-sample inventory, soft anonymous clustering, exclusions, deletion, profile family/version/invalidation/rebuild, and governed storage | Product modules/tests and private derived artifacts | Source/sample/cluster/profile lineage, rollback, access and deletion tests pass |
| A4 evidence supervisor and confidence | A1-A3 | Compose capability-scoped adapters, run ledger, calendar/purpose/participant hypotheses, pillar scores, calibration history, budgets, and partial failures | Product modules/tests and private shadow artifacts | Exact replay, score lineage, budget/isolation tests, zero provider writes |
| A5 review APIs and views | A4 | Build queue/People projections and decision/effect-preview APIs, then implement original-filename-bearing Identity Review and People tabs | Local API/schema/frontend tests and Authelia-protected preview | Idempotency, existing-route regression, stale rejection, desktop/mobile/audio/decision proof pass |
| A6 live shadow | A4-A5 | Process 25 oldest-forward historical conversations and seven days of new arrivals without applying identity/profile conclusions | Private run/evidence/queue records | Replayable queue, usability, failure/privacy receipts, and zero accepted effects |
| A7 reviewed learning | A6 | Apply explicit reviews to local people, assignments, roles, relationships, transcript layers, and candidate profiles | User-scoped local store after backup/rehearsal gate | Exact apply/replay/rollback and deterministic rebuild pass |
| A8 calibration and promotion | A7 | Measure learning value on frozen source-disjoint data and define any policy-qualified automatic band | Private evaluation and tracked aggregate decision | Threshold met or explicit refine/withhold; no automatic promotion by test success |
| A9 policy-qualified automation | A8 | Enable only an accepted automatic band while retaining review fallback | Installed services and Authelia-protected deployment | Existing-route/privacy/recovery/load/product acceptance and 100-case/99%-precision gate pass |

A1 is the schema critical path. A2 and A3 follow its identity and lineage
contracts and may then proceed independently. A4 joins them; A5 may build
against redacted fixtures before A4 finishes but must integrate before A6.
A6-A9 are serialized launch stages.

## Bounds for future execution

- `max_work_unit_attempts`: 2 per packet before local reframe.
- `max_review_rework_cycles`: 1 closed-world cycle per packet.
- `max_broad_review_discovery_passes`: 1 for the whole campaign.
- `max_historical_shadow_batch`: 25 conversations for the first A6 run.
- `max_new_conversation_shadow_window`: 7 days for the first A6 run.
- `expensive_enrichment_backlog_threshold`: 500 actionable conversations;
  preserve new-work capacity and continue cheap preprocessing.
- `max_provider_retries`: 1 retry only for transient, idempotent reads; no
  retry for authorization, tenant, schema, or privacy failures.
- `max_model_reference_repairs`: 1 reference-only repair per model phase.
- `max_transcript_identity_cascades`: 1 per processing version, followed by
  `manual_resolution_required`.
- `max_profile_rebuilds_per_correction`: 1 deterministic rebuild attempt,
  followed by fail-closed review if equality or evaluation fails.
- `calibrated_likelihood_min_source_disjoint_outcomes`: 30 per relevant band.
- `candidate_version_min_new_reviews`: 25, unless a material correction
  requires evaluation sooner; evaluation cadence is weekly.
- `automatic_acceptance_min_source_disjoint_outcomes`: 100 across varied
  conditions, with at least 99% precision in the proposed band.
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
- No automatic person merge or split at any launch stage.
- No automatic profile enrollment from predicted identities.
- No provider contact mutation or write-back in this plan's launch target.
- No public/anonymous dashboard, raw filesystem path, unrestricted audio URL,
  or Graphiti storage of private or unreviewed content.
- No silent rewrite of transcripts, diarization, evaluations, decisions,
  source records, relationships, or historical artifacts.
- No claim that Evidence Strength is a probability or that later scores only
  increase; re-scores may rise or fall and retain their versioned history.
- No self-training on unreviewed predictions and no evaluation-set leakage.
- No implementation, live processing, provider retrieval, deployment, or
  runtime mutation in this planning turn.

## Acceptance criteria

- The architecture has one normalized person/source-record/role/relationship
  model and one append-only correction ledger; it does not fork the existing
  conversation knowledge authority.
- Baseline contact ingest proves the three-level exact-dedup/exact-person-link/
  review-proposal policy, preserves provider history, excludes shared-role
  addresses from person auto-linking, and performs no provider write-back.
- Roles, relationships, organizations, and ontology additions prove
  hierarchical, temporal, multi-role, inverse/directional, conflict-preserving
  behavior rather than flat last-write-wins fields.
- Every queue item and downstream record retains conversation ID, recording ID,
  actual original recording filename, source artifact/media hashes,
  processing-run ID, model/rubric/profile versions, and evidence lineage.
- Calendar association, person linking, contextual speaker, acoustic speaker,
  and combined ranking are separately visible under versioned rubrics with
  factors and alternatives. Empirical likelihood remains unavailable until
  the 30-outcome source-disjoint calibration minimum is met.
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
- Anonymous clustering is soft and reversible; cluster confirmation never
  silently assigns identities. Biometric exclusions and every initial deletion
  scope invalidate dependent material with previewed effects and an audit
  tombstone.
- Raw and normalized transcripts, scoped terminology, span corrections,
  correction/readout generations, and the one-cascade limit replay without
  destroying raw ASR or creating circular evidence.
- The Authelia-protected dashboard proves original-filename display, safe
  playback, complete alternatives/evidence, stale-decision rejection, keyboard
  and mobile usability, Identity Review and People separation, and no
  anonymous access or second login requirement.
- The first shadow campaign reports pipeline yield, candidate recall,
  correctness, calibration, high-strength errors, abstention, review load,
  provider yield/failure, latency, duplicate control, and knowledge integrity.
- No automatic acceptance or live background processing begins until the
  relevant source-disjoint quality, privacy, existing-route regression,
  backup/restore, rollback, load, and user-workflow gates pass.
- Stage 2 begins only after 25 historical conversations and seven days of new
  shadow work are replayable and usable. Stage 3 requires the explicit
  100-outcome, 99%-precision gate and safe abstention.

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
- Existing Authelia-route/request-protection regression, optimistic
  concurrency, idempotency, effect preview, audit, pagination, filtering, and
  stale queue projection tests; no second authentication layer.
- Redacted component tests plus authenticated desktop/mobile browser proof for
  queue states, original filenames, media playback, evidence, every decision
  control, undo/supersession, and inaccessible anonymous routes.
- Oldest-forward chronological shadow evaluation with frozen exclusions and
  aggregate tracked metrics; private exact receipts for any live rehearsal.
- Focused tests, full suite, planning audit, CodeGraph status/readback,
  `git diff --check`, clean commit, push, and upstream equality at each
  integrated packet.

## Packet closeout and next authority

A0-A5 are closed at their non-live terminal gates. A5 adds knowledge schema
v8, rebuildable queue and People projections, stale-safe/idempotent decision
and exact-effect-preview APIs, and original-filename-bearing Identity Review
and People tabs. Redacted browser proof covers desktop/mobile layouts, bounded
audio playback, every frozen decision action, zero-effect preview, projection
v2 recording, and the Identity Review to People transition. Focused validation
passes 86 tests; the full-suite and packet audit evidence are recorded in
RUNBOOK Turn 396.

The A6 Level 1 readiness slice is also built and tested without private data.
`identity_shadow_campaign.py` freezes the oldest 25 eligible historical cases
by conversation time, separates artifact-stabilization time for the half-open
seven-day arrival window, creates content-addressed private activation and
arrival ledgers, verifies terminal A4 supervisor runs from the schema-v8
store, projects optional A5 queue items, and emits a hash-replayable aggregate
scorecard. Every stage repeats exact zero identity/profile/provider/deletion
effects. The redacted CLI and fixtures are documented in the
[A6 shadow campaign guide](../../correction-first-shadow-campaign.md);
RUNBOOK Turn 397 records validation.
This proves the execution bounds and replay mechanics only. It does not meet
the A6 Level 2 real-artifact, seven-day, or authenticated-usability terminal
gate.

The explicit A6 checkpoint was granted and campaign
`identity-shadow-36ce4c7eb1467aea97e071b4` was activated from a reviewed
content-addressed preview. The live schema-v3 store remained read-only; a
SQLite-consistent private snapshot was migrated to schema v8 for the campaign.
All 25 historical cases reached terminal A4 state and projected 25 private A5
queue records. The aggregate readback contains 108 speaker labels, 25 calendar
candidates, and 55 participant hypotheses. Every case is intentionally
`partial` because the provider baseline was unavailable under the closed
provider-access gate: there were zero provider calls and retries, zero review
submissions, and zero accepted identity, profile, provider-write, or deletion
effects.

Local loopback browser proof loaded all 25 real queue records on desktop and
mobile, exposed calendar, participant, speaker, and bounded audio evidence,
and returned HTTP 206 for an audio byte range with no browser errors or console
entries. This proves local real-queue usability only; authenticated route
usability remains unproven because deployment and authenticated live access
were outside the checkpoint.

The half-open new-arrival window is active from `2026-08-16T23:33:20Z` through
`2026-08-23T23:33:20Z`. The first read-only observation, through
`2026-08-16T23:46:21Z`, found zero new documents. A6 remains open until the
full window is observed, every eligible arrival is recorded, and the terminal
scorecard/replay gates pass. A7 remains blocked. This advances the Vision's
correction-first speaker identification, conversation association, and
reusable-knowledge outcomes from redacted Level 1 readiness into a bounded
Level 2 private shadow, without claiming the terminal Level 2 evaluation or
any Level 3 accepted-effect maturity.
