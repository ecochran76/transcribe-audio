# Correction-first identity review

Plan 0072 A5 adds a local, rebuildable review surface over the frozen A0
contracts. It does not turn a proposal into identity truth. The review queue,
effect preview, immutable submission, and People read model remain distinct
objects with explicit lineage and optimistic concurrency.

## Surfaces

`GET /api/identity-review` returns the conversation-first queue. It supports
bounded pagination plus review-state and text filters. Every item retains the
actual original recording filename, conversation and recording identifiers,
source artifact and media hashes, processing-run identifier, model/rubric/
profile versions, calendar alternatives, participant hypotheses, speakers,
evidence, audio bounds, and decision history.

`GET /api/identity-review/items/{queue_item_id}` returns one queue item.

`POST /api/identity-review/items/{queue_item_id}/preview` validates the frozen
submission contract and returns an exact effect preview. A5 previews only a
local review-projection change. Provider writes, raw deletions, accepted
identity changes, profile changes, and relationship changes remain zero.

`POST /api/identity-review/items/{queue_item_id}/decisions` records one
append-only submission and its effect preview. The route rejects a stale
`expected_projection_version` with HTTP 409. Replaying the same idempotency key
and content returns the existing result; reusing the key for different content
also returns HTTP 409. Recording advances only the replaceable queue
projection and preserves the immutable history.

`GET /api/people` returns the local People projection with bounded pagination,
status filtering, and text search. Source records, roles, and relationships
remain separate authoritative tables. Relationship display is capped at two
hops; names and aliases are not merge authority.

## Operator workflow

Identity Review is conversation-first. The list shows review state, priority,
original filename, speaker count, and calendar-candidate count. The detail
view exposes lineage, the top three calendar candidates plus no-match,
participant hypotheses, every diarized label, bounded source audio, independent
evidence pillars, alternatives, and every frozen A0 decision action.

The operator selects an action and may add an immutable comment. Preview must
succeed before Record is enabled. After recording, the UI reloads the queue so
the new projection version and state are visible. A stale browser cannot
silently overwrite a newer projection.

People is a separate tab. It shows canonical/provisional status, aliases,
source records, roles, relationships, the input watermark, build time, and the
two-hop display boundary. A5 does not provide accepted People mutation forms;
those effects remain gated for a later packet.

## Storage and migration boundary

Conversation-knowledge schema v8 adds a replaceable queue projection and two
append-only tables for submissions and effect previews. Migration and rollback
are additive and preserve the earlier identity, correction, biometric, and
supervisor histories. Ordinary API startup does not migrate a store. The A5
fixture preview migrates only an operator-supplied disposable root.

## Security and effect boundary

The existing Authelia-protected dashboard route remains the only intended
authentication boundary. A5 adds no login, OAuth flow, anonymous share route,
or second authentication layer. Blob playback uses the existing bounded range
route and the UI discloses hashes and opaque identifiers rather than raw
filesystem paths.

A5 authorizes redacted fixtures and local review projections only. It does not
authorize historical or new-conversation processing, private corpus access,
provider retrieval or write-back, live migration, biometric enrollment,
accepted identity/profile effects, public publication, deployment, or
background scheduling.

## Validation

The packet is accepted only when schema migration/rollback, queue filtering
and pagination, People aggregation, idempotent replay, idempotency conflict,
stale rejection, existing API regression, frontend build, and full repository
tests pass. Browser evidence must cover desktop and mobile layouts, original
filename display, bounded audio playback, every decision option, exact-effect
preview, decision recording, the Identity Review to People transition, and an
empty console/error readback.

The redacted browser evidence is stored under
`docs/dev/fixtures/plan-0072-a5/browser-qa/`.
