# People, Organization, And Activity Index Contract

Plans 0076 and 0077 define a source-neutral directory read model over the
existing append-only identity authority. The current public schema is
`transcribe-audio.people-organization-activity-index.v5`.

## Identity boundaries

- A canonical person or organization exists only after an explicit authority
  event. Provider contacts, reviewed speaker labels, attendee records,
  organization strings, and domains remain source evidence.
- Exact source-scope duplicates collapse by source record or observation ID.
  Same-name records render together only as an `unresolved_group`; each member
  and source record remains independently addressable.
- Name, domain, organization, calendar overlap, or a shared/role address never
  creates an accepted person, organization, or affiliation.
- Merge, split, correction, alias, and reversal decisions are append-only.
  Rebuilding from active events must yield the same semantic hash.
- `display_name` is presentation only. `person_name_candidates` contains
  complete human-name candidates retained from source evidence; exact
  organization labels are excluded from both fields but remain visible in
  organization/source evidence.

## Affiliation and role boundary

- Every role appointment remains an independently correctable temporal
  assertion with its own durable `role_id`, status, evidence IDs, and validity
  interval.
- `organizations[]` groups those appointments by person and organization for
  presentation. Its stable `affiliation_id` is derived; an affiliation is not
  a second mutable authority.
- `primary_affiliation` is a deterministic compact display projection, and
  `additional_organization_count` reports the other affiliation groups. It
  does not imply that one organization or role is permanently primary.
- Provider organization strings can create proposed affiliations with an
  empty `roles[]`. They never create accepted employment, membership, or
  ownership.
- Accepted contextual retrieval uses role appointments only when they match
  an evidence anchor, are effective at the request time, were accepted by that
  time, and did not originate in the conversation being interpreted.

## Activity boundary

Every activity observation has a stable observation ID, subject type and ID,
channel, occurrence time, participation and evidence states, source scope,
independence group, content hash, and bounded locator. Transcript, calendar,
and email summaries count distinct independence groups. Calendar association
is proposed participation unless separate accepted evidence says otherwise.

Coverage is independent of counts. `not_queried`, `partial`, `unavailable`,
`unauthorized`, and `stale` must never render as an observed zero. Directory
responses exclude raw transcripts, message bodies, provider payloads, audio,
and secrets.

## API and operator workflow

`GET /api/people` accepts `view=people|organizations|unresolved`, `q`, `limit`,
and `offset`. It defaults to people ordered by latest accepted-or-observed
interaction descending. The response includes compact channel summaries and
bounded expansion data. `GET /api/person-repairs` derives current accepted-name,
identity-ambiguity, and possible-duplicate findings without mutation. Its POST
peer accepts only exact stale-safe operator decisions. The UI uses sortable,
resizable compact tables and inline detail rather than cards or nested panels;
the Repairs mode contains only repair findings and accepted decisions that can
be corrected.

The fixtures in `docs/dev/fixtures/plan-0076-p0/` are redacted adversarial
examples. They freeze the refusal to merge same-name people or shared
addresses, the proposed-only status of organization strings, and duplicate
activity control.
