# Use a user-scoped conversation knowledge store

Status: Accepted

## Context

Conversation processing currently spans normalized transcript artifacts,
conversation-owned processing sidecars, SQLite document and chunk indexes,
reviewed speaker assignments, configured provenance sources, and private App
Intelligence run artifacts. This is sufficient for bounded preprocessing, but
it doesn't yet provide one durable model for accumulating reviewed identities,
relationships, topics, terms, evidence snapshots, and historical evaluations.

Speaker identity inference needs both exact lookup and contextual retrieval.
Calendar attendee emails, provider identifiers, reviewed speaker decisions,
prior conversations, topic affinities, and relationship histories must remain
traceable to the account, tenant, source record, and time from which they came.
Repeated copies of one underlying fact must not count as independent evidence.

## Decision

Use the existing user-scoped transcript home as the target canonical storage
location:

- SQLite under `~/.transcripts/` stores normalized domain records, indexes,
  observations, claims, evaluations, and review decisions.
- Content-addressed files under `~/.transcripts/` retain audio, normalized
  artifacts, and other immutable payloads that don't belong in relational
  columns.
- Conversation-owned `.processing.json` sidecars remain the authoritative
  processing history during migration. After a verified authority cutover,
  they become portable, history-preserving exports of database state.
- Provider systems remain authoritative for full Gmail, Drive, Calendar, and
  Odollo bodies and records. The local store keeps bounded immutable evidence
  snapshots, source references, timestamps, and hashes used by a particular
  evaluation.
- Host code owns provider retrieval, temporal and tenant filtering, person
  grouping, duplicate-evidence control, packet budgets, and reference
  validation. App Intelligence reasons only over prepared evidence IDs.
- Graphiti receives only compact, reviewed, source-backed projections that are
  useful across conversations. It doesn't own raw transcripts, provider
  snapshots, evaluations, or review history.

The storage model separates observations from derived profiles. Confirmed
reviews append attributable observations. Rebuildable projections summarize
person, relationship, topic, and terminology affinities without overwriting
their supporting history.

## Consequences

- Exact identifiers, relational traversal, FTS5, and existing chunk embeddings
  can serve one retrieval planner without adding a second database system.
- Every inference can be replayed against the exact bounded evidence available
  to that run.
- Historical processing can apply an explicit `as_of` time and distinguish
  contemporaneous evidence from later knowledge.
- Account and tenant relationship context remains visible after cross-source
  person grouping.
- Schema migration and projection rebuilding require explicit versioning,
  compatibility tests, and rollback paths.
- SQLite remains the default until measured scale or concurrency requires a
  different implementation behind the same storage interface.

## Alternatives considered

### Keep JSON sidecars as the only store

Rejected as the long-term authority. Sidecars remain useful portable audit
records, but cross-conversation identity, relationship, temporal, lexical, and
semantic retrieval would require repeated full-file scans and fragile
cross-file joins.

### Use Graphiti as the primary conversation database

Rejected. Graphiti is appropriate for compact reviewed facts and associative
memory, but not for raw private artifacts, transactional review history,
provider snapshots, deterministic migrations, or exact replay.

### Add a separate vector or graph database now

Deferred. The existing SQLite, FTS5, relational indexes, and chunk embeddings
cover the current scale. A new implementation is justified only by measured
limits and must preserve the same storage and retrieval interfaces.

## References

- [Conversation knowledge storage and retrieval](../conversation-knowledge-storage-and-retrieval.md)
- [Use durable conversation identities](0001-use-durable-conversation-identities.md)
- [Plan 0029](../dev/plans/0029-2026-07-26-conversation-knowledge-storage-retrieval.md)
