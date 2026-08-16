# Correction-first identity ledger

Plan 0072 A1 adds a non-live, user-scoped identity/contact ledger to knowledge
schema v4. The ledger is append-only; current people, source records, external
identities, roles, relationships, and reconciliation decisions are projections
that can be discarded and rebuilt from immutable events.

This module performs identification and contact reconciliation. It is not an
authentication system, provider writer, live-directory synchronizer, or
biometric authority.

## Public interface

`IdentityLearningLedger` in `identity_learning_ledger.py` exposes four bounded
operations:

- `append_event(...)` appends one content-hashed, idempotent event;
- `register_ontology(...)` freezes a hierarchical role/relationship ontology;
- `rebuild()` replaces current projections deterministically from ledger order;
- `reconcile_baseline(...)` deduplicates and evaluates synthetic or explicitly
  authorized private source records without writing a provider.

`projection_snapshot()` is a deterministic test/readback seam. It is not a
provider export contract.

## Schema v4

`ConversationKnowledgeStore.migrate()` now advances ordinary stores through
v4. The migration adds immutable ontology and event tables plus rebuildable
projection tables. It does not change the processing authority, mutate legacy
knowledge rows, query a provider, or read a directory.

The v4 event and ontology tables reject `UPDATE` and `DELETE` with SQLite
triggers. Corrections append `source_record_corrected`, `role_corrected`, or
`relationship_corrected` events. Merge, split, and reversal are also events;
no historical row is rewritten.

Rollback from v4 drops only v4 ledger/projection objects and restores schema
version 3. Existing transcript and v1-v3 knowledge rows survive the migration
and rollback. The frozen Plan 0063 private rehearsal explicitly requests v3 so
that its historical receipt and table-count authority do not change.

## Identity and privacy rules

- A person is canonical. Provider/profile/account/tenant-scoped source records
  and external identities remain evidence about that person, not global keys.
- Raw email and phone values are rejected from source-record ledger events.
  Persisted external-identity projections accept only lowercase SHA-256 value
  hashes. Baseline reconciliation may compare raw values in memory on an
  explicitly supplied disposable/private input, but does not persist or send
  them.
- A source record links automatically only through one unambiguous, verified,
  person-specific email or phone exact match.
- Shared or role identifiers never auto-link. Missing and conflicting exact
  identifiers remain explicit reconciliation proposals.
- Exact duplicates use the full provider/profile/account/tenant/record-type/
  external-reference scope. Name, organization, address, and fuzzy similarity
  do not deduplicate or bind a person.
- Roles are contextual and time-bounded; multiple roles may coexist.
  Relationships retain subject/object direction, inverse references, evidence,
  temporal bounds, and conflict metadata.

## Rebuild and reversal

Events replay in `(occurred_at, event_id)` order. A merge redirects source
records and external identities to its target in the projection. A split moves
only the explicit source-record allowlist. A reversal excludes its referenced
event on the next rebuild, recovering the prior merge, split, or correction
state without deleting history.

Projection `built_at` and input watermarks derive from the event stream rather
than wall-clock rebuild time. Rebuilding the same event set therefore produces
the same projection hash and byte-equivalent logical snapshot. A failed replay
does not replace the last valid projection.

## Current authority boundary

A1 was proved only with pytest temporary directories and synthetic records.
No live store was migrated; no private baseline directory was read; no
provider was called or written; no Graphiti memory was written; and no
dashboard, worker, deployment, or biometric workflow was activated.
