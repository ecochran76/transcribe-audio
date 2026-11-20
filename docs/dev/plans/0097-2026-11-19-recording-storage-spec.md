# Recording storage lifecycle specification

State: CLOSED
Triage: complete
Owner: storage lifecycle integration coordinator
Source: accepted Plan0097 Q1-Q17; Q16 explicitly removes canary and timed hold

## Problem Statement

Recording history, expanded audio and qualification caches consume NVMe without an
intentional lifecycle. Operators need durable historical access and bounded active
storage without losing provenance, recovery or pending work.

## Solution

Application-owned placement separates stable media identity from physical storage.
Keep active metadata/work on NVMe, historical media on the native HDD and retire
only qualified copies/caches through a journaled, observable policy. Initial and
recurring migration share the same verification rules; there is no native canary
or artificial hold after a file satisfies its requirements.

## User Stories

1. As an operator, I want to retain original recordings including screen video on HDD, so that storage remains recoverable, useful and bounded.
2. As an operator, I want to search transcripts when HDD media is unavailable, so that storage remains recoverable, useful and bounded.
3. As an operator, I want to play historical recordings through normal application paths, so that storage remains recoverable, useful and bounded.
4. As an operator, I want to keep stable recording identities when bytes move, so that storage remains recoverable, useful and bounded.
5. As an operator, I want to resolve legacy media paths without rewriting provenance, so that storage remains recoverable, useful and bounded.
6. As an operator, I want to resume interrupted copies from a durable journal, so that storage remains recoverable, useful and bounded.
7. As an operator, I want to verify every destination before publishing its location, so that storage remains recoverable, useful and bounded.
8. As an operator, I want to retain independent original-source recovery before retiring NVMe copies, so that storage remains recoverable, useful and bounded.
9. As an operator, I want to protect active readers and workers from retirement, so that storage remains recoverable, useful and bounded.
10. As an operator, I want to respect source-erasure fences during migration and restore, so that storage remains recoverable, useful and bounded.
11. As an operator, I want to retain exact decoded WAV copies for the initial migration, so that storage remains recoverable, useful and bounded.
12. As an operator, I want to qualify exact reconstruction before evicting the last derivative payload, so that storage remains recoverable, useful and bounded.
13. As an operator, I want to retire eligible completed scratch after 48 hours, so that storage remains recoverable, useful and bounded.
14. As an operator, I want to cold-move inactive unresolved payloads after seven days without retrying inference, so that storage remains recoverable, useful and bounded.
15. As an operator, I want to enforce a 20 GiB transient budget, so that storage remains recoverable, useful and bounded.
16. As an operator, I want to keep 100 GiB free for background admission, so that storage remains recoverable, useful and bounded.
17. As an operator, I want to quarantine unclassified recordings without age-based deletion, so that storage remains recoverable, useful and bounded.
18. As an operator, I want to detect absent or changed HDDs before writes, so that storage remains recoverable, useful and bounded.
19. As an operator, I want to avoid silent cold-store fallback to NVMe, so that storage remains recoverable, useful and bounded.
20. As an operator, I want to see explicit held reasons and per-item progress, so that storage remains recoverable, useful and bounded.
21. As an operator, I want to take and rotate verified compressed database backups, so that storage remains recoverable, useful and bounded.
22. As an operator, I want to restore isolated snapshots and verify integrity monthly, so that storage remains recoverable, useful and bounded.
23. As an operator, I want to run one hourly maintenance worker within 16 GiB and 30-minute bounds, so that storage remains recoverable, useful and bounded.
24. As an operator, I want to stop safely on crashes, clock outages and conflicts, so that storage remains recoverable, useful and bounded.
25. As an operator, I want to migrate directly after validation without a canary or 24-hour hold, so that storage remains recoverable, useful and bounded.
26. As an operator, I want to preserve existing formats and exact bytes during first rollout, so that storage remains recoverable, useful and bounded.
27. As an operator, I want to keep Speech Receipts and shared assets outside this spec, so that storage remains recoverable, useful and bounded.
28. As an operator, I want to measure actual exclusive-block recovery rather than assume logical savings, so that storage remains recoverable, useful and bounded.
29. As an operator, I want to reconcile every initial object as migrated or held with an explicit cause, so that storage remains recoverable, useful and bounded.
30. As an operator, I want to preserve failed-job checkpoints and operator decisions, so that storage remains recoverable, useful and bounded.

## Implementation Decisions

- The accepted lifecycle contract and ADRs0006/0007 govern identity, placement,
  independent source replicas, classification, retention and ownership.
- A transactional SQLite catalog/journal owns physical locations, leases/pins and
  copy/verify/publish/retire states; existing work-progress authority is unchanged.
- Configured native mount identity, full hashes, source stability, owned roots,
  private permissions and erasure fences guard mutation. Missing proof holds data.
- Consumer integration resolves catalog locations while preserving legacy IDs,
  checkpoint hashes, original paths and containment semantics.
- Policy, catalog/migration, backup and operator-maintenance responsibilities have
  separate public interfaces and one owner each. Shared services remain external.
- StableTime is required for age/deadline decisions; no guest-time substitution.
- One explicit local integration branch owns the ticket DAG. Implementation tests
  use existing pytest with real temporary files and SQLite, not a custom harness.
- Source implementation, installation and native migration have separate evidence.

## Testing Decisions

Use operator storage commands and public media/catalog APIs as stable seams. Verify
observable retention, recovery, access and failure behavior; do not mock internal
implementation or assert private database layouts. Real temporary files/SQLite and
boundary-injected time/mount observations are sufficient. Reuse existing media,
managed-recording, source-erasure and conversation-store tests. No GPU, provider
calls, private-corpus fixtures, new test framework or elaborate fixture project.
One final two-axis code review and one bounded post-review TDD remediation pass.

## Out of Scope

Speech Receipts spec/implementation, shared model/environment ownership, paid CI,
source erasure authority, transcoding, inference retry/fallback and OpenClaw work.

## Further Notes

All policy choices were accepted by the operator. Implementation details and the
frozen policy thresholds are in the consolidated lifecycle contract. Native actions
require actual file verification and scope checks; no sample/canary gate exists.


## Completion evidence

Tickets0098-0103 are CLOSED. One Standards/Spec review and one bounded post-review
TDD remediation pass completed; all nine accepted blockers resolved. Final runtime
source fe71dc0ff95685b6af0e6536ef91d64da7229fee passed113 focused tests; affected
SourceOwner checks passed7 with one opt-in Speech skip outside scope. Native
installation, cold access, exact WAV recovery, backup and isolated restore passed.
405 WAV placements retired recovered38,512,484,352 exclusive NVMe bytes; all896
frozen identities unchanged. Current833 canonical NVMe paths and48 unqualified WAV
placements remain held by policy (one canonical arrived during live operation).
Hourly maintenance is enabled/active; its first scheduled catch-up completed,
publishing a verified backup and reporting the monthly drill not due. Long-term unattended footprint is not yet measured. Future
managed-profile activation remains unqualified/disabled at the shared-owner boundary.
See execution ledger and review report; private evidence remains outside Git.
