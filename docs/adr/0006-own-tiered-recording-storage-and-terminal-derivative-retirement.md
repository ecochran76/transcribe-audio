# Own recording placement and terminal derivative retirement

Status: accepted lifecycle design; implementation qualification pending.

Transcribe Audio owns placement and retention for its recording sources, processing
derivatives, checkpoints, exports and private qualification artifacts. Canonical
original recordings, including screen video, remain retained on HDD and available
through ordinary application consumers. Text/index operations remain independently
available when HDD media is unavailable. Speech Receipts lifecycle is reserved for
a separate spec; shared model assets retain their existing owner's authority.

Terminal processing derivatives may retire after verified publication and verified
recovery while source identity, configuration, hashes, checkpoints and results remain
retained. Incomplete or uncertain work remains protected. This partially qualifies
[ADR0005](0005-resume-recordings-across-managed-inference-units.md)'s intermediate-
evidence preservation requirement: active resumability still requires its evidence,
but terminal work need not accumulate every working payload indefinitely. It does
not authorize original recording erasure, checkpoint reset, inference replay or
lossy source replacement.

After bounded acceptance, a versioned policy may perform migrations and eligible
cleanup unattended within its agreed scope, with per-item receipts. Stable logical
location and independent source recovery are specified by
[ADR0007](0007-separate-media-identity-placement-and-source-replica-protection.md).
Plan0097 Q6-Q17 accepts grace periods, quotas, migration protocol, backup history
and rollout, with no native canary or timed hold under the explicit Q16 override.
The [lifecycle contract](../recording-storage-lifecycle.md) consolidates the design;
no installed behavior is claimed.

Source: operator acceptance of Plan0097 interview Q1-Q5 on 2026-11-19, explicitly
reserving Speech Receipts for a separate spec.
