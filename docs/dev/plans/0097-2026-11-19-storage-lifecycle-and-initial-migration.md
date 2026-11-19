# Storage lifecycle and initial migration

State: PLANNED
Owner: primary design interviewer
Scope: resolve and document initial storage migration and future automated tiering and retention for Transcribe Audio
Lane: P11
Design version: 4
Interview state: Q1-Q17 accepted with Q16 override; consolidated confirmation pending
Execution authority: documentation and read-only discovery only until shared understanding is confirmed

## Objective and vision

Make durable recording evidence, useful transcript retrieval, reproducible processing,
and autonomous operation sustainable as the recording corpus grows. Advance VISION
outcomes for evidence integrity, grounded retrieval, replay and autonomy. Existing
stores and resumable processing are built; explicit storage-tier lifecycle and
unattended retention are not qualified. Initial target: level 2, a verified bounded
migration and recovery demonstration. Subsequent target: level 3, measured unattended
policy enforcement with observable recovery and bounded NVMe use. This does not
claim end-to-end transcription or contextual-readout acceptance.

## Current State

- Transcript SQLite, artifacts and original-media blobs share the user-scoped store.
  Blob paths are also embedded in durable metadata. Source ingestion copies media.
- The operator's read-only inventory found original video dominating retained media,
  substantial decoded WAV staging from completed historical processing, repeated
  per-worker CUDA caches, identical extracted model payloads and historical SQLite
  snapshots. Private inventories and recording identities are not copied into Git.
- Managed CLI replay currently requires its decoded WAV and rejects a symlink at
  that file. Corpus readers require resolved blob paths inside the store's blob
  directory. Storage migration cannot assume arbitrary path substitution works.
- ADR0005 preserves raw and intermediate evidence for resumability. Accepted
  ADR0006 clarifies terminal derivative retirement while protecting active and
  uncertain work; implementation and recovery qualification remain pending.
- Plan0089 owns source erasure, copy inventory, original roots, durable fences and
  partial-cleanup receipts. Migration, retained aliases and archives must remain
  inside that authority.
- Plan0095 remains OPEN; completed historical batch scheduling does not close its
  wider native qualification acceptance. This lane does not alter those outcomes.
- Existing CONTEXT.md is the domain glossary. No competing GLOSSARY.md is needed.
- Source baseline: 30c5bd184cb4ec31fd9005fd11b669de1482fab7. Documentation custody:
  design/transcribe-storage-lifecycle-20261119. Existing frontend/research changes
  remain in the original checkout.

## Scope

Resolve policy and implementation boundaries for canonical sources, derived media,
checkpoints, exports, indexes, caches, Transcribe-owned qualification payloads
and snapshots; distinguish initial migration from recurring automation; define ownership, classification, lifecycle,
recoverability, accounting, bounded execution and rollout acceptance.

## Non-Goals

No live migration, deletion, service restart, model/GPU work, provider call, retry of
failed recordings or installation change during the interview. No changes to the
parked OpenClaw lane. Speech Receipts storage lifecycle is explicitly reserved
for a separate spec.
No direct migration, pruning or ownership of shared Speech Receipts/LocalAI assets.
No raw private media, transcript content, credentials or private host inventories
in repository documentation. No reprioritization of other roadmap lanes.

## Design tree and interview ledger

Q1-Q5 were accepted by the operator on 2026-11-19 ("yes all"), with Speech
Receipts explicitly assigned to a separate spec. Later recommendations remain
proposals until answered.

| Node | Prerequisites | Frontier decision | Decision or proposal | State |
| --- | --- | --- | --- | --- |
| Q1 | none | Canonical source retention | Preserve original recordings, including screen video, on HDD | accepted |
| Q2 | none | Storage ownership | Own Transcribe objects; Speech Receipts is a separate spec; shared assets retain owner authority | accepted |
| Q3 | none | Intermediate evidence retirement | Permit terminal derivative eviction only with verified recovery and retained provenance; preserve unresolved work | accepted |
| Q4 | none | HDD access contract | Ordinary application-readable historical media; text/index operations remain available independently | accepted |
| Q5 | none | Automation authority | Qualified versioned policy runs unattended within its agreed scope, with per-item receipts | accepted |

## Accepted foundation contract

- Canonical original bytes, including screen video, remain retained on HDD.
  Migration alone never authorizes source erasure or lossy source replacement.
- Transcribe owns its media, processing state, outputs and private qualification
  artifacts. Speech Receipts lifecycle belongs to a separate spec. Shared model
  assets remain with their existing service owner.
- Terminal derivatives may retire only after verified publication and verified
  recovery, with source identity, configuration, hashes, checkpoints and results
  retained. Incomplete, failed or uncertain work is protected from disposal.
- HDD historical media remains directly usable by ordinary application consumers.
  Text/index operations remain available when HDD-dependent operations cannot run.
- A qualified versioned policy may run unattended within agreed rules and bounds,
  with per-item receipts. Foundation acceptance is not final confirmation of the
  unresolved design and does not start runtime execution.

## Accepted recovery and operating decisions

| Node | Prerequisites | Decision | Accepted decision | State |
| --- | --- | --- | --- | --- |
| Q6 | Q2/Q4 | Object identity and placement | Stable media identity/hash plus an application-owned location catalog and configurable roots; explicit legacy compatibility | accepted |
| Q7 | Q1/Q3 | Derivative recovery | Initial cold-copy recovery; later reconstruction only after byte-exact hash qualification with retained decoder provenance | accepted |
| Q8 | Q3/Q5 | Scratch retention | Completed eligible scratch: 48-hour grace; inactive failed/uncertain payloads may cold-move after 7 days while state remains protected | accepted |
| Q9 | Q2/Q5 | Resource budget | 20 GiB aggregate transient media cache/staging budget; 100 GiB root-free reserve for background admission; no eviction of pinned work | accepted |
| Q10 | Q1/Q2/Q3 | Unclassified media | HDD quarantine with inventory and references retained; no age-based automatic source disposal | accepted |
| Q11 | Q1/Q4 | Independent source recovery | Require another verified durable copy in a separate failure domain before the last NVMe canonical-source replica retires | accepted |

The operator accepted Q6-Q11 on 2026-11-19 ("yes all"). These are design
decisions, not defaults already installed or authority to begin runtime mutation. Grace periods require the existing StableTime authority; unreliable
or unavailable trusted time cannot establish age-based eligibility.

## Bounded discovery for the next decisions

- Existing conversation_knowledge_store._create_backup backs up SQLite, not
  recording media. Broader architectural backup/restore acceptance does not yet
  define media replica count, separate-device placement, RPO/RTO or rotation.
- acoustic_audio_derivatives replay verifies retained source/output bytes, recipe,
  quality, manifest and PCM metrics; its rollback retains both files. Managed CLI
  also requires retained decoded WAV bytes. Neither proves reconstruction after
  eviction.
- Existing source-owner validation rejects symlink components, hardlinks and
  files over 32,000,000 bytes. The new design must preserve current fences and
  explicitly qualify relevant large-media behavior; it cannot claim broad erasure
  acceptance from the existing bounded small-file protocol. Original source
  erasure remains a separate authorized effect from replica retirement.

## Accepted operating contract

- Media identity and expected hashes remain stable while an application-owned
  catalog resolves physical placements under configured owned roots. Legacy path
  compatibility must be explicit; historical provenance is not rewritten to hide
  earlier locations.
- The initial migration preserves exact decoded WAV bytes on HDD. Reconstruction
  may replace cold-copy recovery only after byte-exact qualification with retained
  canonical sources and decoder provenance; this qualification does not yet exist.
- Completed scratch becomes eligible after a 48-hour grace period and all recovery,
  publication and ownership predicates pass. Inactive failed/uncertain payloads
  may cold-move after seven days only without active reader/worker ownership;
  checkpoints/results remain protected and no retry is triggered.
- Transient media staging/caches share a 20 GiB budget; new background admission
  requires a 100 GiB filesystem-free reserve. Capacity pressure does not revoke
  pins, cancel unrelated work or authorize unclassified disposal.
- Unclassified media belongs in inventoried HDD quarantine until ownership and
  references are reconciled. Age alone never authorizes its source disposal.
- Before the last NVMe replica of a canonical source retires, require a verified
  HDD destination and another verified durable replica in a separate failure
  domain. An existing source can qualify only with verified identity and durable
  custody. Missing proof holds that object's NVMe replica; migration of independent
  eligible objects may continue. This rule does not require duplicate permanent
  replicas of every disposable cache or reconstructed derivative.

## Accepted migration and rollout decisions

| Node | Prerequisites | Decision | Accepted decision | State |
| --- | --- | --- | --- | --- |
| Q12 | Q6/Q7/Q11 | Migration commit/recovery | Per-object durable journal: own/pin, copy, verify, publish catalog, then retire eligible old replica; resume interrupted work without guessing | accepted |
| Q13 | Q4/Q5/Q6/Q9 | Missing/changed HDD and corruption | Block dependent writes/retirement, preserve text operations, revalidate identity/content before unattended resume; quarantine disagreements | accepted |
| Q14 | Q1/Q4/Q11 | Backup history and restore cadence | Hourly DB snapshots retained 24h, seven daily and four weekly restore points; compressed HDD copies; monthly restore drill; original media replica proofs remain separate | accepted |
| Q15 | Q5/Q8/Q9 | Scheduler and work bounds | Hourly reconciliation, one background migration worker, 16 GiB or 30 min per pass with safe checkpoints and independent time availability handling | accepted |
| Q16 | Q7/Q9/Q11 | Initial rollout | Local/synthetic validation then direct bounded migration; no canary or 24h hold; verify each object before old-copy retirement | accepted |
| Q17 | Q1/Q3/Q7 | Format scope | Preserve current bytes/formats in the first rollout; qualify lossless derivative format optimization separately | accepted |

Q12-Q17 were accepted on 2026-11-19 with an explicit Q16 override: no native
canary and no 24-hour NVMe rollback hold. This changes rollout staging only;
per-file copy/hash verification, catalog publication, ownership checks and Q11
independent canonical-source replica protection remain accepted requirements.

Protocol invariants (engineering consequences of accepted decisions):

- An active reader/writer/processing pin, uncertain owner release, erasure fence,
  changed payload identity or unresolved policy classification excludes retirement.
- No mutation falls back to an empty mountpoint directory on NVMe. Text operations
  remain independent of missing media; invalid reliable time pauses age-dependent
  actions without inventing guest-clock eligibility.
- Placement-copy retirement is distinct from erasing a recording. Migration and
  recovery respect existing source-erasure fences/tombstones and cannot resurrect
  erased content. Large-file compatibility is separately proven within Transcribe
  ownership; work requiring Speech changes remains in its separate spec.
- Private inventory, replica proofs, policy versions and per-item receipts stay
  user-scoped. Repository fixtures remain synthetic or explicitly redacted.
- Transcribe-owned qualification payloads may consolidate equal bytes while
  retaining distinct experiment configurations and outcomes; no shared model
  owner boundary is crossed.

The complete contract is consolidated in
[recording storage lifecycle](../../recording-storage-lifecycle.md). Routine
engineering defaults are explicit there; no accepted decision requires a new
per-item approval. A material departure discovered during implementation must be
surfaced rather than silently assumed.

## Documentation decisions

CONTEXT.md records accepted recording-source, processing-derivative, placement
migration and retirement terms. ADR0006 records the accepted foundations and
partially qualifies ADR0005 only for terminal intermediate retention. ADR0007
records accepted placement/identity and independent-replica tradeoffs. Q6-Q17
recovery, grace periods, budgets, protocol, backup history and rollout are accepted,
including removal of the proposed canary/hold. Extend
this plan, ROADMAP and RUNBOOK as answers settle.

## Acceptance Criteria

- Every design-tree branch has an explicit accepted decision or documented exclusion;
  the operator confirms shared understanding before runtime execution.
- Each owned storage class has a placement, retention and recovery contract, with
  source deletion, migration, derivative eviction and cache pruning distinguished.
- Initial migration and recurring automation have separate bounded acceptance gates.
- Existing transcript identity, media access, replay, erasure and privacy contracts
  are preserved or explicitly superseded with migration compatibility specified.
- Tests/receipts required to prove integrity, recovery, failure isolation and actual
  exclusive NVMe block recovery are defined before implementation begins.
- Shared services, deployment, native mounts and data owners have explicit boundaries.

## Validation

During the interview: documentation consistency, source cross-checks, git diff
--check, and scripts/check_policy_pointers.py. No inference or live transcription
smoke is warranted for documentation changes.

Before runtime rollout: synthetic migration and lifecycle tests, fault injection at
copy/publication/retirement boundaries, reader/writer/erasure concurrency, mount
outage and return, path containment, hashes/counts/reference reconciliation,
recovery drills and exclusive-block measurements. Native migration verifies each
actual file directly; the operator removed the separate native canary and timed
hold. Accepted thresholds and limits are in the consolidated lifecycle contract.

## Current checkpoint

State: awaiting consolidated shared-understanding confirmation.
Progress classification: outcome_progress for design acceptance, not runtime maturity.
Q1-Q17 are accepted, with Q16 explicitly removing the proposed canary and 24-hour
hold. The design tree has no remaining policy frontier; deployment paths/replica
availability are facts to verify within implementation, not choices to guess.
[The consolidated contract](../../recording-storage-lifecycle.md) is the review
surface. No runtime migration or deletion has occurred. Next action: obtain the
skill-required shared-understanding confirmation, then implement, validate and
execute direct bounded migration within this contract.
