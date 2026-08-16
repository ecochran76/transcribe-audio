---
last_updated: 2026-08-16
applies_to: Plan 0072 A3 and knowledge schema v6
---

# Correction-first biometric custody

Plan 0072 A3 adds a non-live custody layer for synthetic voice samples,
anonymous clusters, and reviewed voice profiles. The layer preserves exact
lineage and keeps private bytes outside portable metadata.

> **Note:** This component is experimental and non-live. It doesn't authorize
> access to a private corpus, historical processing, live migration, biometric
> enrollment, provider calls, publication, or deployment.

This packet advances VISION outcomes 2, 3, 4, 6, 7, and 8. It moves acoustic
custody from a Level 1 contract to a Level 2 replayable component. Private
shadow evidence, review-product acceptance, and automatic learning remain
behind the A4-A9 gates in Plan 0072.

## Public interface

`BiometricCustodyLedger` in `biometric_custody_ledger.py` owns the A3 boundary:

- `store_private_object(...)` writes hash-bound bytes with `0600` permissions
  under an explicit `0700` private root.
- `register_sample(...)` appends source, range, quality, review, consent, and
  preparation lineage. An unreviewed sample must remain person-unbound.
- `record_cluster_version(...)` appends ranked, soft memberships with evidence.
- `record_cluster_rescore(...)` records score changes after a confirmed anchor.
  It requeues only materially changed, related, unreviewed samples.
- `register_profile_family(...)` separates profile conditions for one person.
- `build_profile_version(...)` accepts an exact allowlist of reviewed,
  consented, eligible samples.
- `record_profile_event(...)` appends activation, rejection, supersession,
  invalidation, rollback, or deletion.
- `verify_profile_rebuild(...)` records whether a deterministic rebuild is
  byte-identical to its source profile.
- `preview_effect(...)` resolves exclusion or deletion effects before a write.
- `apply_effect(...)` rejects stale previews and appends bounded events and a
  minimal deletion tombstone.

## Knowledge schema v6

`ConversationKnowledgeStore.migrate()` adds immutable ledgers for samples,
sample events, anonymous cluster versions, soft memberships, cluster events,
cluster-rescore receipts, profile families, profile versions, profile events,
rebuild receipts, and deletion tombstones.

SQLite triggers reject `UPDATE` and `DELETE` on authoritative v6 rows. A
rollback from v6 removes only v6 objects and restores schema v5. It preserves
the identity/contact and transcript-correction histories from A1 and A2.

## Sample and cluster rules

Every sample cites the source-media SHA-256, time range, sample SHA-256,
quality decision, preparation recipe, and private-object SHA-256. Portable
reads omit the private object identifier and filesystem path.

Unreviewed samples remain private and person-unbound until explicit deletion
or a later retention policy. A person binding requires a reviewed identity,
review authority, and consent authority.

Cluster membership remains soft and reversible. Each version preserves rank,
score, evidence identifiers, and membership state. Excluding or deleting a
sample changes its effective membership to `excluded` without rewriting the
historical cluster version. A confirmed cluster anchor never assigns an
identity to another member.

## Profile rules

A person may have multiple profile families for different acoustic
conditions. Each profile version cites one family, exact sample allowlist,
evaluation identifier, model revision, recipe revision, predecessor, and
private-object hash.

Only reviewed, included, consented, and quality-eligible samples for the same
person can enter the allowlist. A new profile remains `pending` until an
explicit activation event. Supersession retains the predecessor for rollback,
and rebuild receipts record both matching and drifted outputs.

## Exclusion and deletion rules

The ledger previews sample, cluster, profile, recording, and person scopes.
Per-recording and per-person exclusions invalidate dependent samples and
profiles without deleting their bytes. Restoring a sample doesn't reactivate
an invalidated profile.

Deletion moves affected private objects into a test-local quarantine before
the database transaction. It commits append-only delete or invalidation events
and then removes the quarantined bytes. A failed transaction restores the
bytes, and a stale preview causes no partial effect.

The tombstone contains target scope, preview hash, deleted-object hashes,
invalidated identifiers, authority, and time. It contains no private object
identifier or path. Deleted data is excluded from future backups. Existing
encrypted backups retain their scheduled expiry unless a later design adds
cryptographic shredding.

## Redacted replay evidence

The synthetic [custody replay](dev/fixtures/plan-0072-a3/custody-replay.json)
and [deletion preview](dev/fixtures/plan-0072-a3/deletion-preview.json) exercise
reviewed and unreviewed samples, a soft cluster, an exact profile allowlist,
and a previewed sample deletion. The fixtures contain no real voice, person,
recording, provider data, private path, or reusable biometric payload.

## Current authority boundary

A3 tests use generated byte strings and pytest temporary directories only.
The packet doesn't access raw media, derive a real embedding, enroll a named
profile, call an external benchmark, migrate a live store, or change a running
service. A4 may consume these metadata contracts in a zero-provider-write
evidence supervisor after A3 closes.

## Related documents

- [Correction-first identity-learning contracts](correction-first-identity-learning-contracts.md)
- [Correction-first identity ledger](correction-first-identity-ledger.md)
- [Correction-first transcript learning](correction-first-transcript-learning.md)
- [Plan 0072](dev/plans/0072-2026-08-16-correction-first-speaker-contact-learning.md)
- [Plan 0072 grilled architecture decisions](dev/notes/0058-2026-08-16-plan0072-grilled-architecture.md)
