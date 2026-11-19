# Recording storage lifecycle

Status: accepted interview decisions Q1-Q17; consolidated shared-understanding
confirmation pending. Implementation and native migration have not begun.

Authority: [Plan0097](dev/plans/0097-2026-11-19-storage-lifecycle-and-initial-migration.md),
[ADR0006](adr/0006-own-tiered-recording-storage-and-terminal-derivative-retirement.md),
[ADR0007](adr/0007-separate-media-identity-placement-and-source-replica-protection.md)
and the recording terms in [CONTEXT.md](../CONTEXT.md).

## Ownership and objective

Keep the NVMe footprint bounded by active work while historical recording storage
grows on the WSL-native HDD. Preserve recording identity, source bytes, provenance,
ordinary historical access and recoverability. Initial migration and recurring
maintenance are intentional application capabilities with the same integrity rules.

Transcribe Audio owns its recording media, derived processing payloads, checkpoints,
exports, indexes, private qualification artifacts and private caches. Speech Receipts
storage lifecycle is a separate spec. Shared LocalAI/Speech assets, environments,
GPU admission and model residency remain under their existing owners.

This contract authorizes no source erasure, inference retry, GPU/model operation,
provider fallback, lossy source replacement or unrelated service cleanup. Future
runtime activation follows final shared understanding and implementation validation.

## Placement and retention classes

| Class | Normal placement | Lifetime |
| --- | --- | --- |
| Live database, indexes, current small artifacts, watcher/configuration state | NVMe | Durable; metadata backup required |
| Current code and required runtime environment | NVMe | Retain active generations; ordinary deployment owns retirement |
| Active transient media and working derivatives | NVMe | Reserved and pinned while in use; aggregate transient budget |
| Canonical original recordings, including original screen video | HDD ordinary files | Retained; source erasure is a separate authority |
| Historical exact processing derivatives | HDD ordinary files initially | Cold-copy recovery until reconstruction/retirement is qualified |
| Completed eligible scratch | Evict after 48-hour grace | Verified publication/recovery and no ownership/pin conflict required |
| Inactive failed/uncertain work payloads | May cold-move after seven days | Preserve exact payloads and checkpoint/result state; never infer retry or disposal authority |
| Unclassified media | Inventoried HDD quarantine | No age-based disposal; reconcile ownership and references |
| Closed evaluation bundles and historical evidence snapshots | HDD; archive packaging permitted | Preserve required lineage, configurations and outcome receipts |
| Worker-private regenerable compiler/playback caches | Bounded working cache | Evict only when inactive and unpinned; do not prune another owner's package store |
| Equal Transcribe-owned qualification payloads | One canonical byte payload with explicit references | Keep distinct experiment conditions and outcomes |
| Operational database restore points | Compressed HDD backups | Rotation below; not the same class as frozen evaluation snapshots |

Media placement retirement removes one copy; derivative retirement ends retention
of the derived payload. The completed-scratch disposal grace does not add a rollout
hold to a verified migration that retains exact cold bytes. Class-specific
eligibility, including the accepted inactive unresolved-work rule, still applies.
A migration journal's retired state refers to its old placement, never source
erasure or deletion of the recording identity.

Moving a private payload cold does not erase its identity or make its checkpoint
terminal. Equal bytes do not make two experiments or recording identities identical.

## Identity, location and compatibility

The application-owned catalog records object identity, full content hash, role,
retention classification, policy version and placements under configured owned
roots. Existing recording/blob IDs remain stable. A pathname is a location or
legacy alias, not the media object's identity.

Catalog publication and its migration journal use durable, transactional state.
The live database remains on NVMe. Filesystem manifests/receipts are recovery and
audit exports tied to catalog transactions, not an independent competing authority.

Resolve media through a shared storage interface for ingestion, playback, processing,
evaluation and erasure-aware readers. Compatibility must preserve old checkpoint
keys, expected hashes and original source provenance. Do not rewrite immutable
historical evidence to pretend it originated at a new location. Existing direct
path, symlink-rejection and containment behavior must have explicit compatible tests.
A legacy consumer not yet qualified against the resolver holds its dependent media.

HDD historical media remains directly application-readable. Archive containers are
for closed bundles, not an extraction prerequisite for ordinary recording playback.
Configured HDD identity, native filesystem, ownership, roots and capacity are
validated before writing. An absent/mismatched mount never becomes an ordinary
NVMe directory fallback. Private configuration contains host paths/device identity;
repository fixtures use synthetic roots and identifiers.

## Recoverability and replica protection

Initial migration retains exact decoded WAV bytes on HDD. Existing retained-byte
validation is not proof of reconstruction. Later reconstruction requires retained
canonical source, decoder/recipe provenance and byte-exact expected-hash validation.
Different reconstructed bytes are a recovery failure, not a silent new generation.
No inference is replayed to validate storage migration.

Before retiring the last NVMe replica of a canonical source, verify both the HDD
destination and another durable replica in a separate failure domain. Existing
recording sources can qualify only with verified content, identity and durable
custody. Distinct pathnames alone do not prove independent recovery. No new paid
backup service or third-party storage is implicitly provisioned by this spec.
Missing proof holds that object's NVMe copy and reports the unmet requirement;
independent eligible objects may progress. Disposable caches do not inherit the
canonical-source replica requirement.

Replica proofs record hash, size, storage identity, custody and verification state.
A changed identity/version/custody invalidates its proof. Recovery and migration
respect source-erasure fences and tombstones; restoring a backup cannot resurrect
an erased source. Existing large-file erasure compatibility must be qualified
honestly within Transcribe ownership without claiming Speech acceptance.

## Per-object migration protocol

One named controller owns each migration operation and its bounded retries. Durable
states distinguish inventoried, held, copying, verified, published, retirement-pending,
retired, quarantined and failed. Retained placement identity and expected source
identity are explicit throughout.

1. Classify the exact object, establish policy eligibility, reserve work capacity,
   and acquire ownership/pins coordinated with readers, writers and source erasure.
2. Copy stable source bytes into an owned temporary HDD object. Record source
   identity before/after capture. Never replace a conflicting existing destination.
3. Read back and verify full content hash, size, required metadata and destination
   identity. Recheck source stability and required independent-replica proofs.
4. Publish the verified destination and journal transition transactionally in the
   catalog while rechecking fences/pins. Retain logical identity and compatibility.
5. Retire only the eligible old replica after durable publication, recovery and
   ownership checks pass. Record outcome and exclusive NVMe block measurements.

There is no whole-batch all-or-nothing transaction. A crash can leave extra copies;
restart reconciles catalog/journal and actual identities before resuming the exact
storage operation. Unknown publication/retirement outcomes are reconciled rather
than guessed. No source is deleted merely because a destination pathname exists.

Avoid a store-wide lock across bulk copying. Short coordinated transitions and
per-object ownership must preserve foreground access. An active/unknown reader,
worker, publication or erasure ownership excludes retirement. Lease expiration or
old mtime alone is not proof that an owner stopped using the payload.

Storage retry default: at most three object-local transient attempts before holding
and reporting that object. Definitive identity/hash conflicts quarantine immediately.
A systemic mount outage pauses dependent work rather than consuming every object's
retry allowance. Stuck collectors/workers are not replaced with unbounded replicas.
This never retries failed transcription or resets its checkpoint.

## Time, capacity and scheduling

- Use StableTime for trustworthy grace/deadline decisions with its documented
  measurement instant and uncertainty. Unavailable trusted time cannot establish
  age eligibility. Continue independent inventory/verification where its required
  bounds can still be honored; do not substitute guest time.
- Transient staging/working-media caches share an initial 20 GiB budget. Reserve
  capacity before admitting new work; active/pinned data is not evicted to fit.
  Installed code and separately owned shared models are outside this transient cap.
- Background admission requires a 100 GiB free-space reserve on the NVMe filesystem.
  Preserve this reserve with projected writes accounted for. Existing over-budget
  legacy data is visible migration debt, not permission for forced eviction.
- Reconcile hourly with one background migration worker. Each pass is bounded by
  16 GiB of payload writes (including retries) or 30 minutes, stopping at a safe
  checkpoint. Track read/hash I/O separately. Reserve enough capacity/time for the
  admitted object; an object exceeding a hard pass bound is held with a reason.
- A missing required timing authority pauses work that needs the timed bound;
  unrelated operations may continue only with their own explicit enforceable bounds.
- Explicit initial migration may run bounded passes consecutively; the hourly timer
  is the steady-state cadence. Do not cancel unrelated work or bypass shared-service
  admission to accelerate migration.

Grace and inactivity for legacy objects must come from qualified observations or
explicit reviewed inventory adoption. Untrusted historical guest timestamps cannot
silently satisfy a destructive age predicate. Exact cold placement can preserve
legacy bytes while retirement eligibility remains unresolved.

## Backup and restore operation

Take hourly consistent SQLite snapshots, retaining 24 hourly, seven daily and four
weekly restore points. Verify snapshots, compress them on HDD, record checksums and
restore metadata, and expire only superseded eligible operational restore points.
Media content is not recopied in full for every database snapshot. Frozen evidence
snapshots remain a separate retention class until their lineage obligations resolve.

Perform a monthly restore drill in a private isolated destination: database integrity,
reference resolution and bounded representative media recovery with expected hashes.
Record coverage and failures; a sample drill does not claim a full-corpus scrub.
Never restore over the live store as a maintenance test. Canonical-media independent
replica protection remains distinct from metadata snapshot rotation.

## Outage and conflict behavior

If HDD identity, availability or capacity is invalid, pause dependent writes and old-
copy retirement. Keep text/index operations available. Never quietly switch the cold
store to NVMe, start provider fallback or retry inference. Resume storage operations
only after device and journal/content revalidation. Conflicting bytes or uncertain
custody are quarantined/held with a concrete reason rather than overwritten.

## Initial rollout: no canary or timed hold

The operator explicitly removed the proposed native canary and 24-hour NVMe hold.
After synthetic/local validation and compatible consumer installation, start direct
bounded migration: inactive staging/private caches first, then canonical media whose
independent replica proofs qualify. Each object advances as soon as its own copy,
integrity, catalog, recovery and ownership checks pass. No artificial observation
period or sample count delays an eligible object.

Enable hourly recurring enforcement after the same protocol is locally validated
and its installed identity/policy roots are verified. The first real files are
ordinary journaled migration work, not a separate approval or canary stage.
Original formats/bytes are preserved in this rollout. Lossless derivative codec
optimization requires its own qualification against hashes/reconstruction contracts.
Speech Receipts implementation remains outside this spec.

## Validation and visibility

Before runtime execution, pass synthetic tests for state-machine restart at every
copy/publication/retirement boundary; reader/writer/erasure races; changed source;
wrong/full/missing HDD and return; conflicting destination; failed/unknown jobs;
legacy resolver and checkpoint identity; exact reconstruction failure; quarantine;
backup/restore; time unavailability; reservations/limits and hardlink accounting.

Local validation satisfies the repository CI policy; no hosted CI purchase/run is
required. Native execution verifies each actual file and reports outcomes as it
proceeds; there is no separate canary requirement. Do not start inference or provider
operations merely to validate storage placement.

Expose versioned policy and installed identity, logical placement/classification,
active pins, held reasons, migration progress, verification/replica proof, scratch
reservations and measured exclusive block recovery. Private receipts distinguish
copy completion, content verification, catalog publication and old-replica retirement.
Logical byte totals and external hardlinks are not guaranteed reclaimable blocks.

Implementation slices: catalog/policy/ownership contracts and synthetic tests;
consumer compatibility and replay/erasure-aware resolution; copy/reconcile/retire
engine and dry-run CLI; snapshots/maintenance/status; then validated installation
and direct initial migration with the recurring worker. Keep each slice bounded
and truthful about installed versus source-only behavior.
