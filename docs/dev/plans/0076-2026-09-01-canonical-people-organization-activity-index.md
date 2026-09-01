# Plan 0076 | Canonical People, Organization, And Activity Index

State: CLOSED

Lane: P09

Date: 2026-09-01

Related authority: Plan 0072 A1/A5/A6, Plans 0074-0075,
`docs/conversation-knowledge-storage-and-retrieval.md`, and `VISION.md`

## Scope

Turn the current source-oriented Contacts projection into a compact
person-and-organization intelligence index. The main directory will represent
one canonical person per resolved identity, while still-separate source
records appear as nested evidence or as explicit unresolved reconciliation
groups rather than peer contacts that look like duplicate people.

Add first-class organization records and temporal person-organization
affiliations. Build one source-neutral activity projection over transcript,
calendar, and Mail Receipts evidence so every person and organization can show
compact channel history, coverage, recency, and an expandable cited timeline.
The same accepted people, affiliations, and history must remain eligible for
the shared evidence fabric used by speaker deduction and conversation
understanding.

This planning checkpoint does not execute the plan. Each implementation packet
must preserve the authority and effect boundaries below.

## Vision outcomes and maturity movement

This plan advances the `VISION.md` outcomes for speaker grounding,
relationship and history context, source provenance, uncertainty, accepted
conversation knowledge, and grounded retrieval for later conversations.

| Capability | Current maturity | Target | Outcome evidence |
| --- | --- | --- | --- |
| Person directory and reconciliation | Level 2 source-oriented shadow: canonical people, provider contacts, and reviewed speaker labels are peer rows | Level 2 canonical/reconciliation projection over the real local corpus | One resolved-person row with nested source identities; unresolved clusters remain explicit; merge/split/reversal replay passes |
| Organizations and affiliations | Level 1 strings and proposed affiliation hypotheses | Level 2 first-class organization and temporal-affiliation projection | Organization index, alias/source lineage, hierarchy and affiliation evidence, conflict-safe replay |
| Cross-channel history | Level 1 fragmented calendar appearances, speaker reviews, and mail relationship hypotheses | Level 2 unified transcript/calendar/email activity summaries and cited timelines | Per-channel counts, first/last interaction, coverage state, provenance, and duplicate-control equality on the current corpus |
| Conversation contextualization | Level 1 shared accepted-knowledge retrieval seam | Level 2 only if a bounded replay proves accepted person/organization/history context is useful and correctly cited | Frozen before/after retrieval comparison with as-of filtering and no self-corroboration |

The plan does not claim Level 3 automatic contact management, automatic person
merging, general speaker-identification lift, or an unattended learning loop.

## Current State

The 2026-09-01 live Contacts projection returns 228 peer rows: 6
`canonical_person`, 187 `local_contact`, and 35 `reviewed_speaker` records.
Exact case-folded display names form 15 repeated-name groups containing 39
records. The current graph-discovery projection reports zero person merges.

`Baker Kuehl` demonstrates the problem without proving the answer: two local
contact candidates carry three and four source records respectively, with 7
and 17 calendar-linked recordings, while one reviewed-speaker record carries
one reviewed appearance. All three records are marked possibly related by
exact display name. They must not be silently merged, but they also should not
look like three confirmed people.

Calendar-linked recording occurrences and reviewed speaker occurrences are
separate sections today. A calendar association does not prove attendance or
speech, and the current `recording_count` can therefore be mistaken for
transcript participation. Mail Receipts contributes 120 proposed relationship
hypotheses attached to 20 contacts, not a general per-contact email history.
Fifty-seven current directory rows carry one or more organization strings, but
organizations are not independently addressable entities in the Contacts
projection.

The schema-v8 identity ledger already provides append-only people, source
records, external identities, temporal roles, relationships, reconciliation,
and deterministic projections. The missing layer is a coherent live domain
projection and review workflow over those primitives, not another contact
database or a new source-specific branch.

## Product and domain contract

### Canonical people and source identities

- `person` is the durable internal identity. A provider contact, attendee,
  email address, reviewed speaker label, acoustic subject, alias, or display
  name is evidence about a person, not the person key.
- Exact duplicate provider/account/type/record IDs collapse deterministically.
- A distinct source record may auto-link only through the existing
  non-conflicting exact person-specific email or verified-phone policy.
- Name, organization, title, domain, address, fuzzy similarity, calendar
  co-invitation, and model output create reviewable proposals only.
- Shared and role addresses never auto-link to a person.
- Merge, split, redirect, rejection, and reversal are append-only decisions.
  No grouping deletes source records or rewrites historical evidence.
- Before review, likely matches may render as one reconciliation group, but
  the UI must call it unresolved and must not aggregate its evidence as one
  accepted person.

### Organizations and affiliations

- `organization` is a first-class graph-addressable entity with canonical
  name, aliases, domains, websites, organization type, locations, hierarchy,
  source records, review state, validity, and provenance.
- Provider organization strings and email domains are source observations.
  They do not create an accepted organization or affiliation by themselves.
- Person-to-organization membership is a temporal, evidence-backed role or
  relationship such as `WORKS_FOR`, `ADVISES`, `REPRESENTS`, or `OWNS`.
- Parent, subsidiary, department, customer, vendor, and partner relationships
  remain typed, directional, temporal, and independently reviewable.
- Conflicting affiliations coexist until review; history is never flattened
  into one permanent employer field on `person`.

### Activity and history

Define one immutable `interaction_observation` contract that can project:

- accepted or proposed transcript participation and reviewed speaker
  appearances;
- calendar invitation, organizer, attendance/response when known, event, and
  recording association;
- direct sent/received mail, thread coparticipation, and correspondence
  evidence from bounded Mail Receipts results; and
- later Drive, SysRAG, messages, CRM, or other adapter evidence without
  changing the person or organization API contract.

Every observation retains source profile, account/tenant scope, source record,
source-event and retrieval times, `as_of`, direction, participant status,
evidence strength/state, independence group, content hash, and a bounded
source locator. Counts operate on distinct observation/independence IDs so
duplicate provider copies do not inflate history.

Each channel summary must distinguish:

- confirmed, proposed, conflicted, and rejected participation;
- observed zero from unavailable, unauthorized, stale, partial, or not-yet-
  queried coverage; and
- event/message occurrence from evidence that the person attended, spoke, or
  interacted.

Raw transcript text, message bodies, provider payloads, and private file
contents do not enter the directory response. Expansion returns bounded,
authorized citations or source links only.

## Operator workflow and UX contract

The primary People table remains dense and resizable. It defaults to most
recent accepted-or-observed interaction descending and exposes visible SVG
sort controls for:

| Column | Compact content |
| --- | --- |
| Person | Preferred name plus unresolved/verified state |
| Organization and role | Current accepted affiliation, or an explicit proposed/conflicted state |
| Transcripts | Confirmed and proposed counts plus most recent date |
| Calendar | Event count plus most recent date and coverage state |
| Email | Direct/thread count plus most recent date and coverage state |
| Last interaction | Latest dated observation across enabled channels |
| Identity health | Source-record count, conflicts, and reconciliation work |

Expanding one row reveals a single reverse-chronological timeline with channel
filters, followed by compact source-identity, affiliation, relationship,
reconciliation, and provenance tables. It must not create a dashboard of
large cards or nested panels. Controls remain small, clearly labelled SVG
icons; state is conveyed with text and table structure rather than large pills.

Add a sibling Organizations table using the same interaction summaries and
expansion pattern. People, Organizations, and Unresolved Sources are views of
one authority, not separate data stores.

## Execution graph

| Packet | Depends on | Bounded outcome | Expected write surface | Terminal condition |
| --- | --- | --- | --- | --- |
| P0 contract and fixtures | This plan | Freeze person, organization, affiliation, activity, coverage, API, and decision schemas with adversarial same-name/shared-address fixtures | Domain docs, schemas, redacted fixtures, behavior tests | Contracts reject silent name/domain merges, ambiguous coverage, provenance loss, and duplicate-count inflation |
| P1 canonical directory projection | P0 | Project canonical people, nested source identities, and unresolved reconciliation groups from the existing ledger | Identity projection/workflow modules, API tests, disposable stores | Exact source duplicates collapse; distinct same-name people remain separate; merge/split/reversal and replay equality pass |
| P2 organization authority | P0-P1 | Add first-class organizations, source identities, hierarchy, and temporal affiliation projections | Ledger/store migrations, projectors, review workflow, tests | Organization merge/split/alias and affiliation history rebuild deterministically without auto-accepting domain/name matches |
| P3 unified activity index | P0-P1 | Normalize local transcript, calendar, reviewed-speaker, and Mail Receipts evidence into source-neutral activity observations and summaries | Activity module, adapter seams, projections, API tests | Per-channel and total counts reconcile to distinct observations; coverage and participation states remain explicit; replay hash is stable |
| P4 compact People/Organizations UX | P1-P3 | Replace peer source rows with compact canonical/unresolved tables and one cited activity timeline | `/api/people`, organization/activity APIs, React Contacts workspace, focused tests | Sortable/resizable dense UI passes desktop/mobile Agent Browser QA with no large cards, panels, pills, or text icons |
| P5 real-corpus rehearsal and reviewed cutover | P1-P4 | Rehearse migration/rollback, then publish the canonical projection and bounded reconciliation queue over the current local corpus | User-scoped store after backup/rehearsal, private receipts, installed dashboard | Baker's three records appear as one unresolved group until reviewed; accepted synthetic and operator decisions preserve all history; rollback/rebuild and live readback pass |
| P6 contextual utility measurement | P5 | Compare frozen before/after retrieval for a bounded chronological corpus using accepted person/organization/history context | Private evaluation artifacts and aggregate tracked decision | Correct cited-history coverage, abstention, temporal integrity, duplicate control, and reviewer usefulness are measured; result is advance/refine/withhold, not automatic promotion |

P0-P1 form the critical path. After P1 freezes stable identifiers, P2 and P3
may proceed independently and join at P4. P5 and P6 are serialized because
live projection and outcome measurement depend on the integrated read model.
Each packet gets at most two implementation attempts and one closed-world
remediation cycle before local reframe. No subagent is authorized or required
by this planning turn.

## Acceptance Criteria

- The default People list contains one row per resolved canonical person and
  explicit unresolved groups, never one apparent person per provider/source
  record.
- The Baker Kuehl evidence appears together without an inferred merge. An
  explicit same-person review produces one canonical row with every original
  source record and non-inflated history; rejection or reversal restores the
  prior separation deterministically.
- Two different people with the same name remain separate in the adversarial
  fixture and in every name-only path.
- Organizations are independently searchable and sortable, retain source
  records and aliases, and expose temporal affiliations and organization
  relationships with evidence and review state.
- Every person and organization exposes transcript, calendar, and email
  summary counts, first/last dates, and coverage status. Expanded history is
  chronological, cited, privacy-bounded, and free of duplicate corroboration.
- Confirmed speaker participation is never conflated with a calendar-linked
  recording, and unavailable source coverage is never rendered as zero.
- Merge, split, alias, affiliation, and activity projections rebuild from the
  append-only ledger with identical semantic hashes; stale decisions fail
  without partial effects.
- Accepted person/organization/history context is retrievable through the
  shared evidence fabric under existing `as_of`, tenant, scope, hop, budget,
  and anti-circularity rules.
- The compact People and Organizations tables visibly sort and resize; row
  expansion provides one activity timeline and evidence tables; desktop and
  mobile inspection confirms the density and control contract.
- Current-corpus accounting records source coverage, unresolved records,
  proposed/accepted/rejected decisions, duplicate groups, and zero prohibited
  effects.

## Non-Goals and effect boundaries

- No automatic fuzzy, name-only, organization-only, domain-only, calendar-
  only, relationship-only, acoustic-only, or model-proposed person merge.
- No automatic acceptance of organizations, affiliations, relationships,
  transcript participation, speaker identity, or contact-field changes.
- No provider/contact/mailbox/CRM write-back, message send, calendar mutation,
  Graphiti write, biometric enrollment, voice-profile change, or public share.
- No unbounded mailbox, Drive, SysRAG, message, or CRM crawl. Reads use the
  shared capability/budget/temporal contract and record partial coverage.
- No message bodies, raw transcripts, raw provider payloads, audio, or secrets
  in list responses, tracked fixtures, logs, or broadly shared memory.
- No claim that directory consolidation alone improves speaker recognition.
  P6 must measure contextual utility before a successor changes automatic
  speaker policy.
- No redesign of the Review Queue, Settings, Library, or unrelated console
  surfaces.

## Validation

- Red/green contract tests for exact-source deduplication, name-only refusal,
  shared-address refusal, organization ambiguity, temporal affiliations,
  interaction independence grouping, coverage states, stale decisions, and
  merge/split/reversal replay.
- Disposable schema migration, rollback, and deterministic full-projector
  rebuild before any user-scoped migration.
- Focused identity-ledger, reconciliation, evidence-fabric, relationship,
  transcript API, and Contacts regressions at the cheapest stable seams.
- Provider-free comprehensive suite, Python compilation for touched modules,
  frontend unit/build checks, and `git diff --check`.
- Read-only current-corpus accounting before mutation; backup and disposable-
  copy rehearsal before a reviewed local cutover.
- Installed API readback plus named-session Agent Browser inspection at desktop
  and mobile widths. Browser QA performs no merge, split, affiliation, or mail
  decision against live evidence.
- Active-only planning audit, CodeGraph sync/status, meaningful committed
  checkpoints, push, and upstream equality for implementation packets.
- P6 freezes its cohort, baseline, metrics, and `as_of` rules before comparison
  and reports advance, refine, or withhold without moving an acceptance band.

## Rollback

The canonical directory, organization index, and activity index are
deterministic projections over append-only evidence and decisions. A live
cutover requires a user-scoped backup plus a proven downgrade/rebuild path.
Removing the new projection selector restores the current source-oriented
Contacts read model without deleting source records. Accepted local decisions
are reversed only through superseding ledger events, never row deletion.

## Definition of Done

Plan 0076 is complete when P0-P6 satisfy their terminal conditions and the
current corpus is usable as a compact canonical people-and-organization index
with explicit unresolved identity groups and truthful cross-channel history.
Completion establishes measured Level 2 product capability. It does not close
Plan 0072's broader reviewed-learning/automation campaign or authorize Level 3
automatic identity, relationship, speaker, or provider effects.

## Execution outcome

Plan 0076 completed P0-P6 on 2026-09-01.

| Packet | Outcome evidence |
| --- | --- |
| P0 | The v1 directory contract, adversarial fixtures, and behavior tests freeze canonical people, unresolved same-name groups, organizations, affiliations, source-neutral activity, coverage, provenance, and zero-effect decision rules. |
| P1 | The directory now projects 6 accepted people plus 202 explicit unresolved groups from 548 retained source records. Exact source duplicates collapse, while name-only records do not become one accepted person. |
| P2 | Schema v9 and the append-only identity ledger support first-class organizations, sources, temporal roles, merge/split/alias/correction, reversal, and deterministic rebuild. All 40 current organization entities remain proposed. |
| P3 | Transcript, calendar, and email evidence project into one activity contract with explicit participation, evidence, coverage, independence, time, and bounded citation fields. The installed corpus exposes 1,245 rows: 71 confirmed and 1,174 proposed. |
| P4 | The live Contacts workspace is one compact sortable/resizable table with People, Organizations, and Unresolved views, SVG controls, inline evidence expansion, and desktop/mobile Agent Browser acceptance. Pointer and keyboard resizing both pass. |
| P5 | Disposable v8-to-v9 migration and rollback preserved SQLite integrity. The backed-up live store migrated cleanly to v9; installed API and UI readback pass, and Baker Kuehl renders as one unresolved group containing three separate records, seven source records, and 25 activities. |
| P6 | The frozen current-corpus measurement reports complete bounded citations, zero within-entity duplicate IDs, zero non-descending timelines, and correct abstention. The live authority contains zero accepted organization or activity rows, so accepted-history retrieval correctly returns zero. Directory shadow use advances; automatic speaker/context policy promotion is withheld. |

The installed checkpoint is schema v9 with directory semantic hash
`e9a194045bd9ee47ff23f0b3513754ea4af0fc57bddfad19642dd5fb83f3a55a`.
The private aggregate P6 receipt is
`~/.local/state/transcribe-audio/plan0076/p6-contextual-utility.json`.
No provider write, mailbox or calendar mutation, person merge, organization or
affiliation acceptance, speaker/biometric effect, public share, or Graphiti
write occurred.

The measured maturity outcome is Level 2 for the canonical directory,
organization proposal index, and cross-channel history presentation. Level 2
accepted-history contextual utility is not claimed because the frozen live
cohort has no accepted organization or activity rows; a successor must first
accumulate reviewed authority and then run a chronological before/after
comparison without moving its acceptance band.
