# Storage lifecycle and initial migration

State: PLANNED
Owner: primary design interviewer
Scope: resolve and document initial storage migration and future automated tiering and retention for Transcribe Audio
Lane: P11
Design version: 2
Interview state: foundations accepted; recovery and operating frontier pending
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

## Recovery and operating frontier

| Node | Prerequisites | Decision | Proposed recommendation | State |
| --- | --- | --- | --- | --- |
| Q6 | Q2/Q4 | Object identity and placement | Stable media identity/hash plus an application-owned location catalog and configurable roots; explicit legacy compatibility | pending |
| Q7 | Q1/Q3 | Derivative recovery | Initial cold-copy recovery; later reconstruction only after byte-exact hash qualification with retained decoder provenance | pending |
| Q8 | Q3/Q5 | Scratch retention | Completed eligible scratch: 48-hour grace; inactive failed/uncertain payloads may cold-move after 7 days while state remains protected | pending |
| Q9 | Q2/Q5 | Resource budget | 20 GiB aggregate transient media cache/staging budget; 100 GiB root-free reserve for background admission; no eviction of pinned work | pending |
| Q10 | Q1/Q2/Q3 | Unclassified media | HDD quarantine with inventory and references retained; no age-based automatic source disposal | pending |
| Q11 | Q1/Q4 | Independent source recovery | Require another verified durable copy in a separate failure domain before the last NVMe canonical-source replica retires | pending |

These recommendations are not defaults already installed or authority to mutate
runtime state. Grace periods require the existing StableTime authority; unreliable
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

Remaining dependent branches:

- Q1/Q3: canonical formats, derivative identity, exact replay requirements,
  reconstruction versus cold-copy recovery, snapshots, orphan reachability and
  independent backup policy.
- Q2/Q4: logical object location versus physical placement, compatibility paths,
  migration ownership/fencing, erasure coverage, mount identity and availability,
  cross-device copy/verify/publish/retire and crash recovery.
- Q3/Q5: active-reader/writer pins, pending/failed/uncertain jobs, terminal outcomes,
  eligibility proofs, grace periods, reliable time, budgets, concurrency and quotas.
- Q4/Q5: scheduling, inventory/reconciliation, pause/resume, receipts, metrics,
  dry-run/canary activation, rollback, mount-outage behavior and recovery drills.
- All foundations: bounded first migration batches, policy exceptions, acceptance
  evidence, integration/deployment ownership and final shared-understanding check.

## Documentation decisions

CONTEXT.md records accepted recording-source, processing-derivative, placement
migration and retirement terms. ADR0006 records the accepted foundations and
partially qualifies ADR0005 only for terminal intermediate retention. Remaining
recovery, location, budget and rollout choices remain explicitly pending. Extend
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
recovery drills, bounded native canary and exclusive-block measurements. Exact
thresholds and permitted effects remain dependent interview decisions.

## Current checkpoint

State: awaiting recovery and operating answers (Q6-Q11).
Progress classification: outcome_progress for design acceptance, not runtime maturity.
Q1-Q5 are accepted; ADR0006 and domain terms record the tradeoff. Speech Receipts
is excluded into a separate spec. Next action: settle the complete Q6-Q11 frontier,
then resolve dependent protocol, backup, scheduling and rollout choices. No media,
service, model or installed policy has changed.
