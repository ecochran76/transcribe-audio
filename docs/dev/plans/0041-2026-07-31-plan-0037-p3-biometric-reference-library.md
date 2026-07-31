# Plan 0041 | Plan 0037 P3 biometric reference library

State: OPEN

Lane: P10

Plan Version: 2

Parent: Plan 0037 P3

Owner: primary agent

Expected Write Surface: one focused biometric-reference module and tests,
Plan 0037, `ROADMAP.md`, and `RUNBOOK.md`; synthetic private lifecycle
artifacts only under
`~/.local/state/transcribe-audio/plan-0037/biometric-references/`.

## Vision alignment

P3 advances trustworthy speaker identification by creating the private,
reviewed source authority that later acoustic scoring may materialize. It
supports the north-star outcome while preserving ambiguity: a reference is
never a scoring profile, withdrawn/deleted material cannot seed later
evidence, and calendar membership, transcript inference, contact assignment,
or model proposals are not biometric-enrollment approval.

The P0 scoring-profile contract is maturity `1 - Built`; no distinct reference
authority or lifecycle store exists. This packet targets maturity `2 - Shadow`
for the reference mechanism using synthetic metadata only. P4 exclusively owns
model-specific embedding materialization, dispersion, calibration, and
scoring-eligible `biometric_profile.v1` artifacts. Evidence is exact lifecycle
replay, concurrency/adversarial tests, privacy proof, and independent review.

## Scope

- Add a distinct private
  `transcribe-audio.biometric-reference-profile.v1` contract. Do not weaken or
  reinterpret the frozen `transcribe-audio.biometric-profile.v1` contract.
- Store opaque `person_ref_id`, immutable profile generations, exact source,
  recording, conversation, speaker/session, source-hash, and original-time
  segment references, quality-evidence references, and optional validated
  P1/P2 receipt lineage.
- Require explicit biometric-enrollment approval with opaque approval ID,
  reviewer reference, reviewed timestamp, purpose/scope, and exact approved
  generation/source hashes. Ordinary speaker/contact confirmation is invalid.
- Preserve multiple sessions plus device/acoustic-condition metadata without
  computing embeddings, centroids, dispersion, or model output.
- Provide deterministic dry-run, explicit create/apply, replay, supersede,
  withdraw, and delete operations. Mutations are canonical-dry-run-hash bound,
  append-only audited, idempotent, and protected by lock plus compare-and-swap
  against an exact active head.
- Expose only `eligible_for_materialization`; reference objects never contain
  `eligible_for_scoring` and never authorize named-person scoring.
- Maintain an immutable generation/event ledger and a private derived head per
  opaque person reference. Replay reconstructs state from validated history.
- Publish descendant invalidation requirements that P4 must bind to every
  derived profile. Withdrawal/deletion immediately deny materialization and
  mark all registered descendants ineligible.
- Prove the complete lifecycle with deterministic synthetic metadata and fake
  reference identities; do not read or copy audio.

## Non-goals

- No P0 corpus audio read, real source registration, real enrollment, model or
  embedding execution, model acquisition, centroid/dispersion computation,
  named-person scoring, verification, calibration, or threshold selection.
- No calendar/contact/transcript/model inference as approval, and no reuse of
  an ordinary `status=confirmed` identity review as biometric consent.
- No raw audio, transcript text, names/emails, embedding/vector values,
  unrestricted enrollment media, portable biometric sidecar, prompt payload,
  App Intelligence call, or external provider write.
- No API/UI product surface, unattended batch enrollment, historical
  reprocessing, default pipeline change, or P4 scoring-profile store.
- No status reversal that resurrects a superseded, withdrawn, or deleted
  generation. Restoration/re-enrollment requires a new profile identity,
  approval, and generation.

## Current State

P0 froze `biometric_profile.v1` as a model-materialized restricted object: all
non-deleted profiles require model/preprocessing revisions plus a hashed
private embedding reference, and `active` means scoring-eligible. That schema
cannot truthfully represent P3 reference-only registration. No separate
biometric approval/source-reference schema, immutable generation/head store,
CAS, resolver, or P3-to-P4 invalidation contract exists.

Existing reviewed speaker/contact assignments establish semantic identity
evidence only. They lack biometric-specific purpose/consent, complete source
hashes, exact segment bounds, and acoustic lineage and therefore cannot be
silently promoted into this library.

Graphiti discovery was healthy and found only advisory Plan 0025 contextual
speaker-preprocessing facts; those confirm model identity output is
human-review-only, not enrollment authority. Plan 0037, P0 contracts, current
code/tests, and synthetic receipts control.

Read-only reviewer `/root/p1_review_final` returned the P3 design packet that
established this P3/P4 boundary and the required lifecycle, concurrency,
approval, privacy, and descendant-invalidation gates. No private source,
audio, embeddings, or model asset was inspected.

## Authorization and fail-closed gates

- Dry run writes only its immutable private plan receipt and never opens media.
  Create/apply requires a token bound to the canonical plan hash:
  `CREATE_BIOMETRIC_REFERENCE:<run-id>:<dry-run-sha256>`.
- Supersede requires
  `SUPERSEDE_BIOMETRIC_REFERENCE:<old-generation-id>:<new-run-id>:<dry-run-sha256>`;
  the exact old head must still be active when the new generation commits.
- Withdraw requires
  `WITHDRAW_BIOMETRIC_REFERENCE:<generation-id>:<dry-run-sha256>` and is a
  permanent non-destructive revocation.
- Delete requires
  `DELETE_BIOMETRIC_REFERENCE:<profile-id>:<dry-run-sha256>`. It irreversibly
  minimizes reference content, retains a non-biometric tombstone/audit, and
  requires invalidation receipts for every registered P4 descendant.
- Identical token and plan replay returns the original receipt. Reusing a token
  with changed action, payload, target, predecessor, or plan hash conflicts.
- Every segment must be finite, monotonic, positive-duration, bounded by the
  declared source duration, and bound to exact source/recording/conversation
  IDs and hashes. Duplicate or cross-person source use fails closed.
- P1/P2 lineage, when present, is consumed only as an atomically returned
  validated host object/receipt. P3 does not reopen media or provider objects.
- Dedicated private root containment, `0700` directories, `0600` files,
  non-symlink/non-hard-linked files, canonical hashes, lock ownership, and CAS
  are mandatory. Drift or ambiguity changes no active state.
- Person merge/split/rebinding is never implicit. It requires a new explicitly
  approved generation and preserves predecessor/tombstone history.
- Rollback applies only to incomplete pre-commit staging; it cannot resurrect
  inactive generations or erase committed audit history.

## Execution graph

| Unit | Owner | Dependency | Terminal condition |
| --- | --- | --- | --- |
| P3A reference/storage design | primary plus read-only reviewer | P0, P1 | P3/P4 boundary, schema, approval, lifecycle, CAS, and invalidation contract reconciled |
| P3B private reference lifecycle | primary | P3A | dry/create/replay/supersede/withdraw/delete synthetic tests pass |
| P3C concurrency/adversarial validation | primary | P3B | races, crashes, tamper, lineage, privacy, permissions, and idempotence fail closed |
| P3D terminal audit | read-only reviewer | P3B-P3C | synthetic receipts replay with no unresolved privacy or lifecycle blocker |

Intended concurrency is two active agents. The primary owns the only write
surface and all synthetic private artifacts. Reused reviewer
`/root/p1_review_final` owns the completed design report and terminal read-only
audit, with no private corpus/audio/embedding access. One bounded
repair-and-rerun cycle follows each review; a remaining blocker keeps P3 open.

## Acceptance criteria

- The reference schema and store require opaque person/profile/generation IDs,
  exact source IDs/hashes, finite segment bounds, distinct session evidence,
  quality references, biometric-purpose approval, canonical hashes, lifecycle
  head/generation, and audit bindings.
- Portable and broad receipts never contain audio, transcript text, names,
  email/contact records, embeddings, vectors, model output, or provider-native
  objects. Reference manifests remain restricted and user-scoped.
- P3 has no model/preprocessing revision, embedding reference, dispersion, or
  scoring-eligibility field. P4 materialization must consume an immutable P3
  generation hash and register its descendant profile ID/hash.
- Create is no-clobber and idempotent. Lock plus CAS allows only one successor
  from an exact active head; concurrent/stale supersession has one winner and
  one truthful conflict without split-brain.
- Supersede installs the fully validated successor before making the prior
  generation permanently ineligible. Withdraw and delete deny resolution
  immediately. Restoration requires a new profile ID and approval.
- Replay validates schema, full canonical history, source/approval bindings,
  event ordering, permissions, exact head, status, and descendant
  invalidations. Inactive replay may succeed; eligible resolution rejects it.
- Delete minimizes usable reference detail, retains only a restricted
  non-biometric tombstone/audit, and proves all registered P4 descendants are
  invalidated. Partial erasure is a blocked state, never successful deletion.
- Original audio, P1/P2 artifacts, transcript store, P0 corpus, Plan 0036
  predictions, and external systems remain unread and unchanged.
- Synthetic lifecycle proves create, replay, supersede, concurrent stale-head
  rejection, withdraw, delete/tombstone, forbidden resurrection, and exact head
  behavior without claiming real enrollment.

## Validation

- Focused biometric-reference, acoustic-contract, and P1/P2 interface tests.
- Joined speaker-evaluation, identity-preprocessing, transcript-artifact/store,
  and workflow regressions.
- Synthetic private lifecycle smoke: create -> eligible reference -> supersede
  -> old inactive/new eligible -> withdraw -> resolution denied ->
  delete/tombstone -> recreation denied.
- Adversarial missing/substituted approval, invalid/NaN/Inf bounds, duplicate or
  cross-person source, stale lineage/head, two-concurrent-successor race,
  partial crash, tampered hash/receipt, conflicting token, traversal,
  symlink/hard-link, permission, event-order, resurrection, and descendant
  invalidation tests.
- Explicit readback that no model, embedding, private audio, private corpus, or
  external write was used.
- Active planning-contract audit, `python -m py_compile`, `git diff --check`,
  and full repository suite.
- Reconcile `/root/p1_review_final` terminal report.

## Terminal condition

Close when the synthetic reference-only lifecycle passes approval, create,
supersession, withdrawal, deletion, replay, CAS/concurrency, descendant
invalidation, privacy, permission, tamper, and independent-review evidence
with no unresolved blocker. Closure proves the P3 reference-authority
mechanism only. Real source registration remains prohibited until a later
explicit biometric-enrollment manifest/apply authorization. P4 remains the
sole owner of real embedding materialization and scoring-eligible
`biometric_profile.v1` artifacts.
