---
last_updated: 2026-08-16
applies_to: transcribe-audio.identity-learning-contract-catalog.v1
---

# Correction-first identity-learning contracts

This reference freezes the non-live interfaces for Plan 0072. Later packets
must use these versions when they add storage, adapters, processing, review,
or projection behavior.

> **Note:** These interfaces are experimental and aren't a live feature. A0
> prohibits provider calls, historical processing, biometric collection,
> live-store migration, dashboard publication, and deployment.

Plan 0072 and its accepted design note define product intent. The executable
catalog in `identity_learning_contracts.py` defines artifact fields and
validation at the host seam.

## Contract authority

Use these authorities in order when implementing a later packet:

1. [Plan 0072](dev/plans/0072-2026-08-16-correction-first-speaker-contact-learning.md)
   defines scope, execution order, and acceptance gates.
2. [Note 0058](dev/notes/0058-2026-08-16-plan0072-grilled-architecture.md)
   freezes the accepted product and privacy decisions.
3. [ADR 0003](adr/0003-freeze-correction-first-identity-learning-contracts.md)
   freezes the host-owned seams and authority split.
4. This reference defines the interfaces that later packets must implement.
5. `identity_learning_contracts.py` and the
   [A0 contract freeze](dev/fixtures/plan-0072-a0/contract-freeze.json) provide
   the executable schema catalog.

The user-scoped conversation knowledge store remains the normalized authority.
Provider systems remain authoritative for their complete source records.
Content-addressed private storage retains media and biometric payloads.
Graphiti may receive only compact, reviewed projections.

## Version set

The A0 catalog freezes seven coordinated contract versions.

| Contract | Version | Later owner |
| --- | --- | --- |
| Domain | `transcribe-audio.identity-learning-domain.v1` | A1 and A3 |
| Correction | `transcribe-audio.identity-learning-correction.v1` | A1 and A2 |
| Privacy | `transcribe-audio.identity-learning-privacy.v1` | Every packet |
| Threat model | `transcribe-audio.identity-learning-threat-model.v1` | Every packet |
| Review interface | `transcribe-audio.identity-learning-api.v1` | A5 |
| Provider adapter | `transcribe-audio.identity-learning-adapter.v1` | A4 |
| Supervisor | `transcribe-audio.identity-learning-supervisor.v1` | A4 and A6 |

Changing a required field, authority meaning, privacy class, or safety
invariant requires a new contract version. Adding optional bounded metadata
doesn't require a new version when old readers can ignore it safely.

## Domain record schemas

All records are append-only inputs or deterministic projections. No current
projection may erase its source observations, evaluations, decisions, or
superseded versions.

| Record group | Versioned artifact schemas | Authority rule |
| --- | --- | --- |
| People and sources | `source-observation.v1`, `person.v1`, `external-identity.v1`, `source-record.v1`, `person-alias.v1` | Exact provider records stay separate from canonical people. |
| Roles and relationships | `role-assertion.v1`, `relationship-assertion.v1` | Assertions retain direction, effective time, ontology version, conflicts, and evidence. |
| Conversation association | `conversation-association-candidate.v1`, `participant-hypothesis.v1` | Calendar candidates and attendees remain hypotheses until reviewed or policy-qualified. |
| Speaker review | `speaker-identity-proposal.v1`, `speaker-review-decision.v1` | Proposals aren't assignments. Decisions append and may supersede earlier decisions. |
| Acoustic custody | `voice-sample.v1`, `voice-profile-version.v1` | Named samples require reviewed authority. Profiles cite an exact reviewed allowlist. |
| Corrections | `correction-event.v1`, `transcript-correction-proposal.v1`, `normalized-transcript-generation.v1`, `terminology-entry.v1` | Corrections preserve raw ASR and create new versions. |
| Processing and review | `processing-run.v1`, `identity-review-queue-item.v1`, `identity-review-submission.v1`, `effect-preview.v1` | The run ledger is immutable. Queue state is rebuildable. Review writes reject stale versions. |
| Provider evidence | `provider-adapter-request.v1`, `provider-adapter-result.v1` | Adapters accept bounded read requests and return bounded observations or visible partial failures. |

Every queue item and downstream identity record must retain its conversation,
recording, processing run, model, rubric, profile, source-artifact, media-hash,
and evidence lineage. The queue must expose the original recording filename.
It must reject filesystem paths and enriched transcript filenames.

## Person reconciliation contract

A1 must implement three reconciliation levels without provider write-back.

1. Deduplicate only the exact provider, account, record type, and record ID.
2. Link one source record to one provisional person only through a
   non-conflicting person-specific email or verified phone.
3. Route every fuzzy, name, organization, role, address, or conflicting match
   to a ranked review proposal.

Shared and role addresses must never auto-link to a person. Reviewed local
overrides must not overwrite provider fields. Person merges preserve redirects
and support reversal. Person splits rebind only explicitly reviewed records.

## Transcript correction contract

A2 must preserve the raw transcript and diarization. A correction proposal
binds a raw transcript hash, exact span hash, replacement, evidence, scope,
review state, pass, processing version, and cascade count.

Scope precedence is conversation, project or matter, organization, domain,
then global. Equal-scope conflicts require review. One pre-identity pass and
one post-identity pass are allowed. One material correction-to-identity
cascade may requeue the conversation. A second cascade must stop with
`manual_resolution_required`.

Accepted corrections create a new `normalized-transcript-generation.v1`.
Search indexes raw and normalized text. Citations retain raw lineage.

## Provider adapter interface

A4 adapters implement one read-only seam. The host owns authorization,
candidate selection, tenant routing, temporal policy, budgets, scoring, and
durable writes.

`provider-adapter-request.v1` contains these required controls:

- one conversation and processing run;
- one provider, profile, account, tenant, and capability scope;
- an `as_of` time;
- a bounded query;
- record, character, call, and latency ceilings; and
- a read-only mode and idempotency key.

`provider-adapter-result.v1` must bind the request, processing run, and exact
source scope. It returns bounded observations, consumed budgets, warnings, and
a visible failure when status is `partial` or `unavailable`. It must report
zero provider writes.

One transient idempotent read retry is allowed. Authorization, tenant, schema,
and privacy failures must not retry. A provider failure doesn't discard valid
results from other adapters.

## Supervisor interface

The host supervisor processes one durable conversation through these stages:

1. `bind_conversation`
2. `pre_identity_correction`
3. `calendar_candidate_generation`
4. `participant_and_evidence_collection`
5. `speaker_and_relationship_proposals`
6. `post_identity_correction`
7. `queue_projection`
8. `complete`

Processing starts asynchronously after transcript artifacts stabilize. The
watcher must not run provider or model enrichment inline.

The supervisor permits one provider retry, two transcript-correction passes,
one correction-to-identity cascade, and one reference-only model repair per
phase. At 500 actionable queue items, it throttles expensive enrichment while
cheap preprocessing continues.

`processing-run.v1` binds original filenames, source hashes, capabilities,
budgets, versions, inputs, outputs, failures, effects, and replay state.
`contract_fixture` and `shadow` modes require zero accepted effects.

## Review interface

A5 reserves these authenticated interfaces. A0 doesn't add routes.

| Method and path | Request | Response and rule |
| --- | --- | --- |
| `GET /api/identity-review/queue` | Bounded filters and cursor | `identity-review-queue-item.v1` projections |
| `GET /api/identity-review/items/{queue_item_id}` | Queue item ID | One current item plus alternatives and history |
| `POST /api/identity-review/items/{queue_item_id}/preview` | `identity-review-submission.v1` | `effect-preview.v1`; no mutation |
| `POST /api/identity-review/items/{queue_item_id}/decisions` | Submission plus exact preview authority | Append-only decision and receipt; reject stale projection versions |
| `GET /api/people` | Bounded search and cursor | Canonical people and separate source records |
| `GET /api/people/{person_id}` | Person ID | Person, sources, assertions, corrections, clusters, and profile history |
| `GET /api/identity-media/{handle}` | Authorized opaque handle and range | Bounded media bytes; never a raw path or unrestricted URL |

The existing Authelia-protected dashboard route remains the sole initial
authentication gate. A5 must preserve current request protections. It must not
add a second login, public share link, or anonymous media access.

Review submissions require an idempotency key and expected queue projection
version. Controls cover confirmation, correction, rejection, not-listed,
unresolved, mixed-speaker, label grouping, utterance splitting, person merge
or split, supersession, and defer. Every mutation must preview its downstream
effects first.

## Privacy classes

The catalog assigns each artifact one storage and transport class.

| Class | Allowed location | Prohibited exposure |
| --- | --- | --- |
| `private_user_scoped` | User-scoped knowledge store | Public routes and shared memory |
| `private_provider_request` | Ephemeral host request ledger | App Intelligence and unrelated adapters |
| `private_bounded_evidence` | User-scoped evidence store | Full provider bodies and cross-tenant reuse |
| `private_review_metadata` | Authenticated review interface | Raw paths, unrestricted media, and inline biometrics |
| `private_review_decision` | Append-only decision ledger | Anonymous or stale mutation |
| `restricted_biometric` | Filesystem-protected or encrypted private storage | Portable payloads, Graphiti, and inline vectors |

App Intelligence receives immutable prepared evidence IDs and bounded
snapshots. It cannot browse providers, select its own people, mutate records,
or emit authoritative probabilities.

## Threat controls

The [A0 threat-control matrix](dev/fixtures/plan-0072-a0/threat-control-matrix.json)
binds each threat to one named catalog control and a later verification gate.
The controls cover authentication, tenant isolation, read budgets, provider
write prohibition, media access, biometric custody, stale writes,
supersession, evaluation leakage, self-training, correction loops, deletion,
duplicate evidence, and hindsight.

The architecture has no unresolved privacy decision at A0. It freezes these
operator decisions for later implementation:

- Retain unreviewed samples and embeddings in private, person-unbound storage
  until explicit deletion or a versioned policy change.
- Delete active biometric derivatives immediately and exclude them from new
  backups. Existing encrypted backups expire on their retention schedule.
- Keep raw biometric processing local by default. An external challenger
  requires a separate opt-in benchmark authority.
- Use the existing Authelia route without a second authentication layer.
- Keep raw transcripts, provider bodies, audio, embeddings, and unreviewed
  hypotheses out of Graphiti.

## A0 verification and A1 entry gate

A0 passes when the executable catalog, redacted fixtures, ADR, this reference,
and deterministic tests agree. A1 may start only when no unresolved privacy
decision remains and the planning audit recognizes Plan 0072 as open.

A0 doesn't prove schema migration, live privacy enforcement, provider access,
historical processing, biometric collection, UI behavior, or product quality.
Those claims remain gated by A1 through A9.
