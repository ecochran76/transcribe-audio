# Plan 0056 | Enrolled-only acoustic pilot identity guard

State: OPEN

Lane: P10

## Scope

Run one bounded, human-confirmed acoustic speaker-identification pilot for the
two existing enrolled subjects without creating or mutating people, contacts,
aliases, roles, relationships, references, or provider records. Every machine
proposal must use an already enrolled stable subject ID; display names remain
review attributes only.

The pilot measures whether the Plan 0055 conservative voice-augmented behavior
survives a representative shadow workflow while enforcing the identity guard
recorded in
`docs/dev/notes/0052-2026-08-05-contact-role-relationship-sequencing.md`.

## Vision Outcomes And Maturity Movement

| Capability | Current | Target | Evidence |
| --- | --- | --- | --- |
| Acoustic speaker identity | Level 2 measured shadow | Level 2 bounded pilot evidence | Human-confirmed enrolled-only proposals on fresh recordings |
| Identity integrity | Evaluation labels are not canonical contacts | Stable enrolled-subject references with zero identity creation | Replayable proposal and review receipts |
| Automatic assignment/profile learning | Level 0 | Level 0 | Every mutation and unattended acceptance action remains false |
| Relationship context | Contract documented; not required by enrolled-only pilot | Unchanged | Explicit deferral to P09 conversation-knowledge work |

This advances speaker identity and knowledge integrity without claiming that
cross-provider person resolution, relationship inference, or the automatic
conversation knowledge loop is operational.

## Non-Goals

- No GWS, Odollo, receipts-repository, calendar, or local-contact merge.
- No creation of canonical people from names or role-only labels.
- No relationship inference, graph-edge write, or multi-hop retrieval.
- No acoustic profile learning, replacement, or new enrollment.
- No automatic speaker assignment, default integration, provider write-back,
  historical reprocessing, or production enablement.
- No rewrite of Plan 0055 gold or historical evaluation IDs.

## Current State

Plan 0055 closed with independent PASS and terminal decision
`advance_to_limited_pilot_plan`. Voice augmentation produced 6/22 correct
assignments, including 6/9 enrolled appearances, with zero wrong assignments.
The frozen artifact contains no contact IDs and did not mutate profiles or
references. The two enrolled people have stable acoustic subject IDs; all
other Plan 0055 identities are evaluation-only labels.

The durable person/external-identity/relationship model already exists in
source under the P09 conversation-knowledge architecture, but live authority
cutover and cross-provider reconciliation are intentionally deferred.

Execution is activated on 2026-08-05 from clean, upstream-even commit
`d139ba7`. The first packet must freeze the exact two-subject allowlist,
pre-execution cardinality/generation snapshot, prior-evidence exclusion union,
fresh source set, scoring policy, private paths, and all negative actions
before any pilot audio decode or model execution.

## Critical Path

1. Freeze a pilot contract that allowlists exactly the two existing enrolled
   subject IDs and rejects names, aliases, contact IDs, or newly generated IDs
   as machine identity.
2. Prepare a fresh, bounded pilot recording set without reading outcome gold
   or changing enrolled profiles.
3. Produce acoustic proposals that assign only an allowlisted subject ID or
   abstain/review.
4. Present every proposal for human confirmation and record immutable,
   replayable review evidence.
5. Independently audit denominators, correctness, wrong assignments,
   abstentions/reviews, identity-creation count, mutation flags, and replay.
6. Freeze a terminal decision: stop, refine, or plan the next bounded
   integration milestone.

## Acceptance Criteria

- Every non-abstaining proposal references one of the two pre-existing
  enrolled subject IDs.
- Zero new person, contact, alias, role, relationship, or profile identifiers
  are created.
- Zero profile/reference mutations, provider writes, automatic assignments,
  default integration actions, or historical reprocessing actions occur.
- Every proposed assignment receives explicit human confirmation or rejection.
- Name/title/punctuation/provider variants cannot fork an enrolled subject in
  deterministic replay.
- Complete pilot correctness, enrolled recall, wrong-assignment,
  high-confidence-wrong, review, and abstention denominators are reported.
- An independent reviewer recomputes the identity guard and terminal decision.

## Validation

- Focused unit tests for allowlist enforcement, variant-name rejection,
  role-only abstention, no-creation invariants, and immutable replay.
- Full repository test suite.
- Hash-bound private pilot authority and receipts with `0600` files under a
  `0700` user-scoped runtime tree.
- Read-only verification that contact/person/profile/relationship cardinality
  and generations are unchanged before and after execution.
- Independent terminal review before any successor integration plan opens.

## Stop Conditions

- Stop on any non-allowlisted or name-derived machine identity.
- Stop on any person/contact/profile/relationship creation or mutation.
- Stop on missing human review, incomplete denominators, gold leakage,
  non-replayable output, or a high-confidence wrong assignment.
- Stop rather than widening to additional people or building the deferred
  relationship/contact system inside this plan.
