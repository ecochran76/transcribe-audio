# Own recording placement and terminal derivative retirement

Status: accepted foundations; implementation and remaining policy choices pending.

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
location, exact reconstruction versus cold-copy recovery, durations, quotas, backup
requirements and rollout details remain open in Plan0097; this ADR does not claim
those decisions are settled or deployed.

Source: operator acceptance of Plan0097 interview Q1-Q5 on 2026-11-19, explicitly
reserving Speech Receipts for a separate spec.
