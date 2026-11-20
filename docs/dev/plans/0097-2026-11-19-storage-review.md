# Plan0097 single final review

Mode: one broad two-axis review; target 876fb12d5b1d9fee17a1cc45c323e0e752ae656f.
Fixed point: d325e60abde054d2b0e201288422e1b83d8efa36, assumed accepted-design
baseline after the unanswered optional question; not represented as user approval.
Diff: `git diff d325e60abde054d2b0e201288422e1b83d8efa36...876fb12d5b1d9fee17a1cc45c323e0e752ae656f`.
Commits: `git log d325e60abde054d2b0e201288422e1b83d8efa36..876fb12d5b1d9fee17a1cc45c323e0e752ae656f --oneline`.
All tickets0098-0103 closed in 876fb12 before either axis began. No second broad
review is authorized by this packet. Verification after fixes is closed-world.

## Standards

Independent reviewer /root/storage_finish/standards_review_once inspected the frozen
diff and AGENTS plus relevant policies (0003/0004/0006/0007/0011/0013/0014/0016/0022/0024).
No CODING_STANDARDS.md or CONTRIBUTING.md exists. Tooling-enforced lint was excluded.
The Fowler smell baseline was supplied in full as judgment calls subordinate to
repo rules. Primary adjudication:

- S1 nonblocking_backlog: README lacks the new storage selector/policy pointer
  required by policy0006. Operations documentation already covers it; add a concise
  discoverability pointer during closeout. High confidence, no runtime failure.
- S2 nonblocking_backlog: file identity is a nine-position list used by catalog,
  CLI and derivative readers. A named decoder could reduce coupling, but no current
  behavioral defect was demonstrated. Preserve serialized compatibility; defer
  structural refactoring beyond the bounded behavior remediation.

Standards: zero blocking findings, two accepted nonblocking findings.

## Spec

Primary separately reviewed catalog, policy, maintenance, backup, admission, initial
adapter and consumer diffs against the accepted specification and lifecycle contract.
Each finding below is blocking for its affected capability, high confidence, and
accepted for this one remediation pass:

- R1 classification: contract says “Classify the exact object” and unclassified media
  has “No age-based disposal.” Catalog register defaults to processing_derivative
  (storage_catalog.py:165); direct migrate retires any noncanonical class. A default,
  unknown or unclassified registration can lose its old placement without explicit
  eligible classification. Reproducer: register without a class, then migrate.
- R2 disposal: “Completed eligible scratch | Evict after 48-hour grace” and cache
  policy require disposal. Maintenance sends evict_cache/evict_scratch through
  migrate (storage_maintenance.py:508-537), retaining disposable payloads forever on
  HDD and requiring cold capacity. Reproducer: qualified inactive cache inventory.
- R3 replica evidence: “Replica proofs record hash, size, storage identity, custody
  and verification state.” Catalog has an unused replica journal field and only an
  ephemeral tuple (storage_catalog.py:387-423). A successful canonical retirement
  leaves no durable proof. Reproducer: qualified independent replica migration.
- R4 erasure compatibility: “source-erasure fences and tombstones” and “explicit
  compatible tests” apply. erasure_catalog_paths returns origin-linked derived
  paths, but SourceOwner treats them as canonical SHA copies (storage_access.py:88;
  speech_source_erasure.py:251-279). Different-byte derivatives disable previously
  available bounded erasure. Reproducer: migrate a small origin-linked derivative
  and request existing SourceOwner inventory.
- R5 outage retry: “A systemic mount outage pauses dependent work rather than
  consuming every object's retry allowance.” Persisted attempting state survives
  mount/time/budget holds (storage_maintenance.py:532-565), so repeated outages
  exhaust three attempts. Reproducer: outage after admission, then repeat/resume.
- R6 consumer generation: “A legacy consumer not yet qualified ... holds its
  dependent media.” qualified checks installed bytes but ignores the recorded PID
  and process start identity (storage_maintenance.py:243-254); a restarted/stale API
  stays qualified. qualify_api also accepts an unrelated healthy store. Reproducer:
  stale process identity in an otherwise valid receipt, or wrong-store health body.
- R7 per-placement accounting: “every initial object ... with an explicit cause.”
  account keys held outcomes by logical ID, then overwrites them with qualified
  placements (initial_storage_migration.py:193-215). Native evidence showed 43 of48
  held WAV placements share IDs with retired placements. Reproducer: same ID in held
  and migrated inventory paths; retain both distinct outcomes.
- R8 rotation budget: “24 hourly ... seven daily ... four weekly” and “16 GiB ...
  per pass” must coexist. rotate calls full verify/unpack for every historical point
  (storage_backups.py:276-278). At the actual 870MB DB size, retained history alone
  exceeds the write ceiling, permanently preventing expiry as backups accumulate.
  Reproducer: rotation with published snapshots and zero remaining payload-write
  allowance; verify compressed custody without rewriting the whole history.
- R9 backup scheduling: “Take hourly consistent SQLite snapshots.” Maintenance
  migrates until the pass budget stops, then backs up (storage_maintenance.py:487-568),
  so sustained migration debt starves metadata backups. Reproducer: eligible payload
  reaches write bound; maintenance must publish its bounded metadata backup first.

Nonblocking backlog B1: reserve's nonoverlapping-time requirement can hold otherwise
valid wide-uncertainty observations. It fails closed and does not undermine completed
native evidence; do not add a new time bridge or widen this remediation.

Spec: nine accepted blocking findings; highest severity is unqualified placement
retirement (R1). No finding changes source-erasure authority, shared-service ownership,
backup failure-domain requirements or the no-canary decision.

## One bounded TDD pass

Public seams: catalog migrate/evict (eligibility, pins and proof), maintenance run
(scheduling/retry/disposal), API qualification (current process/store), initial account
(distinct placement outcomes), SourceOwner inventory (per-payload hashes), and
BackupManager rotate (retention without history rewrite). Real tiny files/SQLite;
external time/mount/process observations injected only at their supported boundary.
No provider calls, private-corpus fixtures or new harness. Each slice observes red,
then minimal green. Exact commands/results are appended below after execution.


### TDD execution and closed-world verification

All nine accepted blockers were remediated; no new broad review was run.
Interpreter: existing main `.venv/bin/python`. Managed imports use the installed
qualification SDK through explicit PYTHONPATH. Each public-seam command below
was observed failing for the named defect before its minimal fix, then passing:

- R1 `-m pytest -q tests/test_storage_catalog.py -k unclassified_unknown`: retired
  instead of held; green. Default is unclassified and retirement has an explicit
  retained-class allowlist.
- R2 `-m pytest -q tests/test_storage_catalog.py -k eligible_disposal`: missing evict
  interface; green. Controller test `tests/test_storage_maintenance.py::test_maintenance_evicts_eligible_cache_without_cold_copy` then failed held versus
  evicted; green after dispatch. Its existing check was extended to the subsequent
  restore drill: failed on intentionally absent cache, then green with evicted
  references/samples excluded. No retained media was excluded.
- R3 `-m pytest -q tests/test_storage_catalog.py -k persists_verified`: canonical
  retirement accepted absent custody; green with persisted proof before unlink.
  Delegated catalog R1-R3 worker: /root/storage_finish/catalog_review_remediation,
  complete, nine catalog tests passed. Primary integrated only its two owned files.
- R4 `-m pytest -q tests/test_storage_consumers.py::test_erasure_inventory_preserves_distinct_origin_linked_derivative_hash`: inventory unavailable; green,
  with different-byte derivative hashes verified separately from canonical SHA.
- R5 `-m pytest -q tests/test_storage_maintenance.py::test_outage_during_copy_preserves_retry_allowance_and_resumes`: outage consumed an attempt; green across
  three simulated filesystem outages and actual resumption.
- R6 `-m pytest -q tests/test_storage_maintenance.py::test_consumer_receipt_holds_when_recorded_api_process_identity_changed`: stale receipt stayed qualified; green.
  `tests/test_initial_storage_migration.py::test_api_qualification_rejects_health_from_another_store`: unrelated healthy store was accepted; green with a tiny
  local HTTP process, no provider operation.
- R7 `-m pytest -q tests/test_initial_storage_migration.py::test_account_preserves_held_placement_when_logical_id_also_retired`: missing distinct placement
  outcome; green with composite ID/path accounting and logical_path in each row.
- R8 `-m pytest -q tests/test_storage_backups.py::test_rotation_keeps_hour_day_week_points_and_holds_unknown_time`: rotation attempted payload writes; green.
  Delegated R8 worker /root/storage_finish/standards_review_once completed five
  backup tests in3.23s. Primary checked the exact two-file diff. Creation and
  full verify/restore remain strict; rotation reads full compressed checksums.
- R9 `-m pytest -q tests/test_storage_maintenance.py::test_hourly_backup_is_published_before_migration_can_exhaust_deadline`: missing backup after exhausted
  migration deadline; green with backup/rotation/monthly drill before migration.

The first combined selection was112 passed/1 failed: one pre-existing consumer
fixture assumed permissive default classification. Existing migration fixtures now
state their roles and independent custody and assert actual retirement, preventing
a passing source-only resolution from masquerading as a migrated-path check.
Consumer selection then passed7 tests. Final combined selection:113 passed in25.58s.
Command: `.venv/bin/python -m pytest -q tests/test_storage_catalog.py tests/test_storage_policy.py tests/test_storage_backups.py tests/test_storage_consumers.py tests/test_storage_maintenance.py tests/test_storage_admission.py tests/test_initial_storage_migration.py tests/test_managed_recording.py tests/test_assembly_audio_upload.py`.

Focused tier only; no hosted CI, paid capacity, inference, provider or GPU requests.
Local lint: zero new findings,56 accepted legacy; git diff --check passed.
Only the accepted findings and immediate regressions from their fixes were checked.
S1 README pointer is addressed during closeout; S2 identity-decoder refactor and B1
wide-uncertainty admission remain explicit nonblocking backlog.

Validated Python/systemd source digest: `bdf961783e1b4d33d8ec0d8c08b8109edd7e35e9f6050288dcc1b8d62d5a4df2` (ordered tracked path/NUL/content/NUL SHA256).
