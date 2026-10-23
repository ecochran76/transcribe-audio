# Recording-owner erasure for shared Speech

State: CLOSED
Lane: P09
Critical-Path Owner: Speech graph primary agent
Work item: CochranResearchGroup/speech-receipts#57

## Vision and maturity

Advance trustworthy, owned recordings and evidence preservation without silently
removing meeting context or accepting identity. Current: source export exists,
original-owner deletion is unavailable. Target: source-qualified optional owner
protocol with exact original/blob/playback inventories, separate authority,
durable fences and cleanup receipts. Installed acceptance remains separate.
Evidence: actual private synthetic stores/HTTP, restart and concurrent I/O tests.

## Current State

Source implementation and focused tests exist on plan-0037-campaign, baseline
bac86a66081bf3cc0c1f9c7a754cfd615681d782. No service profile or private recording
was touched. Optional source_owner configuration defaults disabled; it requires
an independent protected token, exact tenant/application principals and explicit
exclusive original roots. Unsupported platform ownership, unknown legacy copy
provenance, shared originals or unsafe physical paths remain unavailable.

## Scope and non-goals

Own registered canonical media, proven original copies and playback caches for
one exact source. Track original hashes before publication, preserve historical
copy paths and deny reingestion of known erased bytes before new media effects.
A common reentrant SQLite store lock spans registered ingestion and full HTTP
media/export publication, including separate processes. No provider/model run,
private/live deletion, service restart, person erasure, transcript rewrite or
Plan0072/frozen identity-pilot work is authorized by source qualification.

## Acceptance and validation

Real stores test independent token/principal denial, exact source revision/hash,
stale/unsafe originals, recorded original-container hashes, previous same-content
originals, permanent replay/read fences, actual HTTP, partial purge/restart and
unrelated-source preservation. Atomic quarantine verifies exact inode/content
before unlinking; changed originals stay unknown rather than being guessed away.
Default tests do not require Node/a sibling checkout. Opt-in actual Speech owner
client traverses supported authenticated HTTP. Preserve red and failed receipts;
validate public blob regressions, syntax and focused static checks. The shared
Speech primary owns coordinator/cache integration and ticket closeout.


## Source qualification

Primary actual joined owner/consumer/affected-API selection:19passed20.72seconds,
58 unrelated API cases deselected. This explicit opt-in selection exceeds the15s
focused-feedback target; no default gate or threshold was raised. Correctness
Ruff checks and compile checks pass. Default tests retain explicit skips for the
four sibling-Node integration cases. Source remains uncommitted/unpublished;
no service restart, owner profile enablement or live/private erasure occurred.

Two closed-world counterexamples were reproduced and remediated: a stream/export
could publish after a premature quiescence acknowledgement, and a separate ingest
process could create an unpreviewed canonical sibling. Publication and registered
ingest now share reentrant cross-process SQLite exclusion, including Windows-safe
ordinary imports; original-owner identity checks remain explicitly unavailable
where unsupported. New copies invalidate old previews. Historical same-content
originals and differently encoded originals use recorded digests; known erased
originals are denied before new media effects. Private atomic quarantine supports
same-ID partial-purge restart without deleting unrelated or changed files.

Meeting context, canonical person decisions, transcript history, Plan0072 and all
frozen identity/enrollment/calibration corpora remain untouched. Qualified Speech
integration and ticket closeout are recorded by Speech plan0049. Private evidence
is under the associated Speech qualification/2026-10-23-erasure-qualification/.
