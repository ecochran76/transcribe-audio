# Recording storage operations

The lifecycle contract in `recording-storage-lifecycle.md` governs these commands.
Installation, qualification, migration and native acceptance are separate steps.
All paths, device identities, filenames and native receipts belong in private
operator configuration outside Git. Never point this controller at Speech or
shared models. No operation starts transcription or retries inference.

Use one integrated, locally validated commit. `install_product(source,
destination, expected_commit)` in `scripts/initial_storage_migration.py` installs
all regular tracked files from that commit into a private immutable generation.
Dirty/untracked work, environments and private workspace files are excluded.
Repeated installation verifies existing bytes and refuses conflicts. Preserve the
literal virtualenv Python path. Install the API using a unit drop-in that preserves
its other environment drop-ins, arguments and static frontend directory; inspect
active work before restarting any service. `qualify_api(config, installation,
pid, health_url)` verifies the actual running API command, full installed byte set
and live health before writing its private consumer receipt. It does not qualify
uninstalled watchers, frozen batch launchers or unknown dependent consumers.

Prepare the controller using `scripts/install_storage_maintenance.py` with the
same private configuration. Its digest generation is independent of the product
Git commit. Bind `source_generation` to the controller generation, then qualify
against the final configuration bytes. The exact same configuration file must
exist at `store_root/storage-maintenance.json` for producer admission. Do not
change frozen processing configuration. Installed executable search paths need
`prlimit`, `ffmpeg` and `ffprobe`. Activate only after validating installed paths,
consumer hashes and current owned roots. Reinstallation/requalification after a
code fix does not reset the catalog, checkpoints, manifests or placement journal.

For managed watcher production, set `TRANSCRIBE_STORAGE_ROOT` explicitly in the
installed watcher unit environment. Both managed command entrypoints accept this
selector; an explicit `--storage-root` takes precedence. The watcher adapter
passes the selected root into the same managed public run path. With no selector
legacy callers retain their existing behavior; installing new code alone does
not activate bounded admission. Verify the running process environment and
installed backend path. Keep this selector outside immutable processing config.

The configuration adds `initial_inventory` to the existing maintenance schema;
include this filename in `inventories`. The private adoption document has version
1, exact `scope_roots`, optional `objects` and optional `processors`. Each object
binds `id`, absolute logical `path`, expected `sha256`, classification, explicit
boolean `active` and a nonempty `custody_evidence` source locator. Each processor
binds its owned `path` and reviewed `custody_evidence`. This is reviewed evidence,
not directory-wide inactivity permission. Hold media used by an unqualified
consumer. Preserve active leases/pins and unknown ownership. Unknown caches,
snapshots, unresolved checkpoints and missing frozen identities remain held;
never infer disposal eligibility from historical guest mtime.

The refresh adapter reads current SQLite blob registration every invocation. New
registrations without reviewed custody are reported held. An optional private
`completed_transcript_registration` adoption entry binds `root` to `store_root`,
`source_generation` to the qualified installed generation and `custody_evidence`
to reviewed producer provenance. It qualifies new canonical custody only from
an authoritative completed transcript link with matching source paths, recording
and conversation IDs. Missing links, changed generation or absent qualification
hold that registration. It supplies no independent-backup proof.
Known canonical holds are reported before reading their payloads, so discovery
can reach eligible derivatives without hashing originals that cannot retire. It also discovers new
CLI manifests in reviewed processor roots and joins them to completed processor
publication by logical WAV path, recording ID and conversation ID. Exact decoded
hash and original source hash remain separate. Only that qualified completion
establishes historical derivative placement; missing lineage is reported held.
Retired logical paths remain in the inventory without recreating their old bytes.

Run these commands with the installed Python and installed script:

```text
python scripts/initial_storage_migration.py --config PRIVATE --adoption REVIEWED refresh
python scripts/initial_storage_migration.py --config PRIVATE --adoption REVIEWED pass
python scripts/initial_storage_migration.py --config PRIVATE account
```

Each pass refreshes inventory and runs the existing single-worker maintenance
protocol with full hashes, native mount guard, transactional publication, reader
pins and source-erasure fences. Historical derivatives run first. Canonical
sources without a current independent replica proof are held before copying.
An explicit reviewed `copy_only_ids` list permits publication while retaining NVMe
for application access validation. Its existing published copy is not recopied
on subsequent passes. Independent replicas must pass physical disk identity,
content and durable custody checks; another pathname on the HDD backing disk is
insufficient. Never provision recovery copies on another volume without authority.

An operator-confirmed peer-replicated network share can supply the independent
replica without provisioning another backup. A private per-object proof uses
`kind: replicated_share`, `path`, `identity` (the complete file_identity tuple),
`sha256`, nonempty reviewed `custody`, and `share_mount` containing `mount_point`,
`filesystem` and `source`. The controller checks the actual longest covering
mount against that exact identity and permits CIFS, NFS/NFS4 or SSHFS; a local
fallback directory or changed mount does not qualify. Topology attestation does
not establish file coverage. The catalog still checks each exact replica's full
content and stable identity and journals/rechecks the proof before unlinking NVMe.
Missing or conflicting matches remain held; the replicated share is read-only
for this migration and must not receive propagated source deletions.

Initial passes may run consecutively. Each remains bounded to 16 GiB payload
writes including retries and 30 minutes using fresh StableTime observations.
Byte accounting applies to every chunk; time observations occur at safe I/O
checkpoints of at most64MiB and explicit admission/publication/retirement phase
boundaries. Observations remain service measurement instants; no guest-clock
projection or extra counter helper is used.
Stop when no eligible work progresses; holds are outcomes, not infinite retries.
No canary or timed rollout hold applies. Source replica obligations and scratch
retention rules still apply. Never simulate an outage by unmounting the real HDD.

Configure the hourly service to invoke the same refresh-plus-maintenance path,
not a frozen one-off inventory. Preserve backup and restore scheduling as well as
migration. The regular maintenance command takes verified consistent compressed
SQLite snapshots, rotates only operational restore points, and runs isolated
restore drills with current erasure tombstones and bounded media checks. An
explicit isolated restore uses `storage_maintenance.py restore --snapshot ID
--destination NEW_DIRECTORY`; it must never replace the live database.

Native acceptance records mount UUID and actual host backing disk, installed API
and controller identities, timer/service/process readback, API health and actual
cold-media range playback, exact resolved WAV hash/header and unchanged checkpoint
identity, a real backup and isolated restore, and every selected object's outcome.
`account` reconciles journal phases and held adoption entries and sums each
object's maximum durable exclusive-block retirement receipt once, alongside fresh
filesystem free space. Logical bytes and multiply linked bytes are not exclusive
recovery. Preserve all receipts and report policy holds separately from bounded
pass deferrals. Source-only checks do not establish native acceptance.

Active producer admission covers managed decode/staging, Assembly's temporary
lossless audio, browser playback conversion/cache and new source-blob copies.
Assembly preserves source sample rate/channels and its existing lossless format;
storage limits never authorize truncation or a different inference backend.
`TRANSCRIBE_STORAGE_ROOT` is an explicit profile selector; managed and Assembly
CLI `--storage-root` overrides it. Include all actual owned staging/cache roots,
including `store_root/upload-staging` and `store_root/playback-cache`, in private
`transient_roots`. Shared LocalAI spools are not Transcribe migration roots. New
managed profiles use an owned spool; historical frozen configurations stay intact.


The catalog defaults new registrations to unclassified. Direct retirement requires
an explicit eligible retained-media role; unknown/live roles cannot confer disposal
authority. Disposable private caches and qualified completed scratch use a separate
journaled `evicted` transition after policy eligibility, consumer, pin, ownership and
full-hash checks; no cold copy is created. Evicted disposable references are omitted
from restore media checks. Retained media continue through exact cold migration.
Canonical retirement also records the verified independent replica's identity,
content, failure domain and custody in its durable journal before unlink.

Runtime-bound consumer receipts revalidate the recorded API PID, process start
identity and installed command. API qualification checks the healthy store/database
against private configuration. Requalify after replacing a process or generation.
A systemic storage/time/budget pause preserves object-local retry allowance.
Accounting reports ID plus logical path so held and retired placements sharing one
logical ID remain separate outcomes.

Hourly maintenance publishes its consistent metadata backup before media migration,
so migration debt cannot starve backups. Rotation verifies the manifest, compressed
size and full compressed checksum of already fully verified published snapshots
without rewriting every historical database. Explicit verify/restore and monthly
drills retain full SQLite integrity checks. Read I/O remains timed and accounted.
