# People, Organization, And Activity Index Contract

Plan 0076 defines a source-neutral directory read model over the existing
append-only identity authority. The public schema is
`transcribe-audio.people-organization-activity-index.v1`.

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
bounded expansion data. The UI must use one sortable, resizable table and one
inline timeline rather than cards or nested panels.

The fixtures in `docs/dev/fixtures/plan-0076-p0/` are redacted adversarial
examples. They freeze the refusal to merge same-name people or shared
addresses, the proposed-only status of organization strings, and duplicate
activity control.
