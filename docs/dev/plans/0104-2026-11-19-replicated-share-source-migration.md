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
