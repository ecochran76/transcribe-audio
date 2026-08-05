# Contact, role, and relationship sequencing decision

Date: 2026-08-05

## Context

Plan 0055 proved that acoustic evidence can improve conservative identification
of the two enrolled speakers and authorized planning a limited pilot. The
operator then clarified that speaker matches must not create a trail of
duplicate identities. Contacts can represent different source-specific views
of one person, people may have several roles, and relationship context is
necessary to understand a conversation. App Intelligence should endeavor to
infer relationships from available bounded context, and accepted relationships
should be stored as graph edges that support bounded multi-hop discovery.

The Plan 0055 gold artifact contains 22 selected speaker rows and 11
evaluation-only person IDs. It contains no contact IDs. Nine rows for the two
enrolled people also carry their existing acoustic subject IDs. The
evaluation-only IDs are deterministic hashes of normalized labels; they are
not canonical cross-provider people and must not be promoted as such.

## Durable decisions

1. `person` is the canonical internal identity. Names, aliases, emails,
   provider contacts, acoustic labels, and conversation roles are not global
   identity keys.
2. Google Workspace, Odollo, receipts repositories, calendar, local contacts,
   and future sources retain independently addressable external identities and
   source records. Several may link to one canonical person without erasing
   their account, tenant, provenance, or relationship meaning.
3. Roles are contextual and time-bounded relationships. A role-only speaker
   label remains unresolved until evidence supports a person link; it does not
   create a person automatically.
4. Relationships are typed, directional, evidence-backed graph edges among
   people, organizations, projects, matters, conversations, recordings,
   events, and source records.
5. App Intelligence proposes identities, roles, and relationships from
   host-prepared bounded evidence. Host validation and acceptance policy own
   durable graph writes.
6. Relationship retrieval returns evidence-bearing paths through a bounded
   number of policy-approved hops. Graphiti is a reviewed discovery projection,
   not the primary store for raw contacts, transcripts, or unreviewed claims.
7. Historical evaluation artifacts remain immutable. Later canonicalization
   adds reviewed mappings and redirects rather than rewriting old evidence.

## Must happen before the limited acoustic pilot

- Limit the pilot to the two already enrolled acoustic subject IDs.
- Make the stable subject ID, not the display name, the machine identity in
  every proposal and review record.
- Require human confirmation before accepting a proposed speaker assignment.
- Prohibit creation or mutation of people, contacts, aliases, roles,
  relationships, acoustic profiles, or provider records.
- Treat every non-enrolled identity and every role-only label as unresolved or
  out of scope.
- Prove by replay that one person cannot be forked by spelling, punctuation,
  title, or provider-record variation within the pilot.

This narrow guard prevents new duplicate identities while allowing the
acoustic milestone to continue. It does not require a live cross-provider
contact migration or relationship graph rollout.

## Deferred to the natural conversation-knowledge milestone

- Read-only reconciliation of GWS, Odollo, receipts-repository, calendar, and
  local contact records into canonical-person candidates.
- Reviewed merge, split, alias, redirect, and reversal workflows.
- Full role and temporal relationship modeling and authoritative edge
  persistence.
- App Intelligence relationship-inference evaluation and acceptance policy.
- Bounded multi-hop graph retrieval and relationship-path ranking.
- Live conversation-knowledge authority cutover, provider write-back,
  historical backfill, and autonomous profile learning.

These items belong to the P09 conversation-knowledge/productization path. They
should be implemented before general multi-person automatic speaker identity,
but they are not a prerequisite for a two-enrolled-person, human-confirmed,
non-mutating acoustic pilot.

## Next authority

Plan 0056 defines the bounded enrolled-only pilot and its identity guard. The
evergreen domain contract remains
`docs/conversation-knowledge-storage-and-retrieval.md`.
