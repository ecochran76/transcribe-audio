# ADR 0003: Freeze correction-first identity-learning contracts

Status: Accepted

Version: `transcribe-audio.identity-learning-contract-catalog.v1`

## Decision summary

Plan 0072 uses one host-owned, correction-first contract family. The
user-scoped conversation knowledge store remains the normalized authority.
Providers, private biometric storage, App Intelligence, Graphiti, and review
views keep distinct responsibilities.

This decision freezes interfaces only. It doesn't authorize provider access,
historical processing, live migration, biometric collection, dashboard
publication, background scheduling, or deployment.

## Context

The repository already stores durable conversation IDs, source records,
review decisions, evidence snapshots, and rebuildable person projections.
Acoustic campaigns also use governed samples and versioned profiles.

Plan 0072 joins those capabilities into one correction loop. Without a frozen
contract, later packets could create competing identity authorities, leak
private payloads across seams, or turn proposals into unreviewed truth.

## Decision

The host exposes a small contract interface through
`identity_learning_contracts.contract_catalog()`,
`identity_learning_contracts.validate_artifact()`, and
`identity_learning_contracts.validate_adapter_exchange()`.

The interface freezes seven coordinated versions:

- domain;
- correction;
- privacy;
- threat model;
- review interface;
- provider adapter; and
- supervisor.

All current state is a deterministic projection over immutable observations,
evaluations, decisions, and correction events. A proposal isn't an assignment.
A score isn't a probability. Speaker identification isn't authentication.

Provider adapters accept only bounded read-only requests. The host owns scope,
authorization, budgets, temporal policy, scoring, durable writes, and retries.
App Intelligence receives prepared evidence and returns constrained proposals.

Review writes require an idempotency key and an expected projection version.
They append decisions before applying authorized local effects. Provider
write-back remains prohibited.

Raw audio, samples, embeddings, and profile payloads remain in private storage.
Portable review payloads expose opaque playback handles and bounded metadata.
Graphiti receives only compact reviewed projections.

## Consequences

A1 can extend the existing store without inventing a second person database.
A2 through A5 can implement against stable record, adapter, supervisor, and
review seams. Later schema changes must preserve lineage or introduce a new
version.

The initial contract is intentionally strict. It requires explicit source
scope, original filenames, content hashes, review authority, effect previews,
and zero provider writes. These fields add storage and test work, but they make
replay, rollback, privacy review, and stale-write rejection deterministic.

A0 does not demonstrate runtime enforcement. Each later packet must prove its
own migration, access, replay, browser, or live-shadow gate.

## Alternatives considered

### Extend campaign-specific review artifacts

Rejected. Campaign artifacts bind one evaluation denominator and don't provide
a durable person, correction, adapter, or supervisor seam.

### Make provider contacts the canonical person authority

Rejected. One person may have compatible identities across several accounts
and tenants. Provider records must retain their independent authority and
history.

### Make Graphiti the primary identity store

Rejected. Graphiti isn't the authority for raw transcripts, private provider
content, biometric material, transactional reviews, or deterministic
migrations.

### Let App Intelligence browse and apply effects

Rejected. That design would weaken tenant scope, evidence replay, budget
control, and provider-write guarantees.

## References

- [Correction-first identity-learning contract reference](../correction-first-identity-learning-contracts.md)
- [Conversation knowledge storage and retrieval](../conversation-knowledge-storage-and-retrieval.md)
- [ADR 0002: Use a user-scoped conversation knowledge store](0002-use-a-user-scoped-conversation-knowledge-store.md)
- [Plan 0072](../dev/plans/0072-2026-08-16-correction-first-speaker-contact-learning.md)
- [Note 0058](../dev/notes/0058-2026-08-16-plan0072-grilled-architecture.md)
