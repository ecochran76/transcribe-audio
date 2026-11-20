# Replicated-share canonical source migration

State: OPEN
Lane: P11
Scope: recognize the operator-confirmed peer-replicated backups share as canonical
replica custody, match exact retained bytes, and migrate eligible originals using
the existing bounded controller.

## Objective and vision

Advance evidence preservation, replay and automatic historical access from built
local-drive replica qualification to operational replicated-share qualification
(level1 to level3 for this bounded placement capability). Measure actual retired
placements, exclusive NVMe blocks and cold access; do not claim the complete
contextualization vision or long-term backup reliability from one migration.

## Current State

The operator confirms Syncthing/backups is redundant across its peers. This is
reviewed topology/custody evidence, not proof that every registered recording is
present there. Existing replica validation only accepts local Windows drive
letters. Add an explicit replicated_share variant bound to the actual remote
mount identity; retain exact content/identity verification before retirement.
Plan0097 and its one review/one remediation pass remain closed and unchanged.

## Non-Goals

No source erasure, Syncthing deletion/configuration changes, new backup provisioning,
Speech/shared LocalAI/model changes, inference replay, or historical bundle cleanup.
No blanket per-directory qualification or bypass of missing-match holds.

## Acceptance Criteria

- Operator custody plus exact remote mount point/filesystem/source bind a private
  per-object proof; missing, changed or local fallback mounts do not qualify.
- The existing catalog performs full replica hash and stable identity checks,
  journals custody and rechecks before old-placement removal.
- Consecutive direct passes retain the existing single-worker 16GiB/30-minute
  limits, StableTime, pins and erasure fences. Stop when no eligible work progresses.
- Install/requalify the exact validated generation; report per-object results,
  unmatched holds, cold access, unchanged frozen identities and exclusive recovery.
- Keep hourly enforcement active after migration and preserve live producers.

## Validation

Focused public Maintenance.replica boundary test: replicated CIFS accepted;
absent/replaced mount and missing custody rejected. OS mount observations are the
only mocked boundary. Observed red before implementation, then green.
Run affected storage maintenance/catalog/initial adapter tests locally; no provider
requests or hosted CI. Native full hashes and actual cold access are separate proof.
Private host paths and recording inventories remain outside Git.


## Native reconciliation and mount repair

The first pass registered canonical bytes but retired none: the active CIFS share
and autofs placeholder share a mount point, and equal-depth selection chose the
placeholder. Preserve that held run as evidence; no invalid retirement occurred.
Replica mount selection now matches the file device before selecting the longest
covering path. Extended existing public mount regression red, then affected
maintenance/catalog/initial selection46passed in19.44s with the existing qualification
PYTHONPATH. Read-only actual share qualification passed. Resume uses all existing
registrations/journals; per-file full content and identity checks remain mandatory.
Native successor acceptance is pending.


## Four-pass native checkpoint

Successor on7aa8757 retired302 of826 matched canonical originals across four
bounded passes,54,626,726,502 logical bytes. Durable cumulative exclusive block
recovery is93,140,905,984 bytes including the earlier38,512,484,352 derivative
recovery; incremental canonical recovery54,628,421,632 bytes. The fourth pass
held on stable_time_client_invalid_evidence. Its original response was not retained,
so the exception alone does not establish a StableTime service outage or exact
field failure. Fresh authenticated supported responses subsequently qualified at
both primary and native worker. Resume remaining524 using unchanged trusted-time
validation and existing catalog/journal. One published placement can reconcile;
never copy already retired objects again. Seven unmatched originals stay protected.
Cold API206, exact WAV and all896 frozen identities passed. Hourly timer restored
enabled/active after the held run; another native resume pauses it only while the
single controller owns consecutive passes and restores it on finalization.
Plan remains OPEN until terminal native acceptance of all eligible candidates or
explicit final holds. No code change or repeated broad review/testing is needed
for fresh qualified time observations.
