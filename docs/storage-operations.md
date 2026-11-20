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
registrations without reviewed custody are reported held. It also discovers new
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

Initial passes may run consecutively. Each remains bounded to 16 GiB payload
writes including retries and 30 minutes using fresh StableTime observations.
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
