# Scoped speaker comparison process

`speaker_match_bridge.py` adds read-side acoustic evidence to the scoped enrollment
store from Plan 0085. `ComparisonBridge(config, adapter)` is the direct test seam;
its production CLI obtains adapters only from the genuine pinned registry. The
configuration is the existing private enrollment configuration. The enrollment
store must already exist, belong to that tenant/application scope, and retain its
reviewed authority, private profile objects and immutable selected heads.

Run the process with `--config /absolute/private/config.json --parent-pid PID`.
The Linux parent-death guard is mandatory. JSON enters stdin and one bounded
`{ok:true,result:...}` or sanitized `{ok:false,error:...}` leaves stdout. Optional
library output is redirected to stderr. The parent owns CPU/thread/offline
limits, operation admission, durable intent, timeout, process-exit cancellation,
result retention and reconciliation. No raw probe or comparison result is stored
by this module. An interrupted comparison is not implicitly retried.

All requests include `tenant` and `app`. Supported actions are:

- `readiness`: the same metadata-only installed-asset verification as enrollment;
  configured/available remain separate from loaded state and qualification.
- `profiles`: the existing scoped selected-head projection, including profile
  SHA256, person revision, adapter/model and enrollment sample provenance.
- `compare`: `job_id`, exactly one `samples` entry in the enrollment protocol,
  and one to 32 explicitly selected `profiles` entries.

Each profile descriptor contains `person: {id, revision}` (revision is a nonempty
string), `profile_version_id`, `profile_sha256`, `adapter_id`, and `model_revision`.
The probe uses normalized mono PCM16 16 kHz WAV, 500–30000 ms, exact SHA256,
source recording ID/hash/timestamps and curated sample revision. No media path or
credential is accepted from the wire.

The boundary validates selected heads, current person/review/enrollment authority,
A3 active state and included reference samples, exact adapter/model pins and
private profile object hash. It verifies the centroid's dimension and finite
nonzero norm. Those guards run before embedding, after exclusive owned-worker
admission and after embedding; changed or withdrawn evidence returns no scores.
Only the new probe reaches the pinned adapter. Existing private centroids enter
`cosine_score`; no profile is built, promoted, relabeled or assigned to a recording.

The result schema is `transcribe-audio.speaker-comparison-process.v1`. It binds
`id`, `sample_manifest`, `sample_manifest_sha256` and `candidates`. Each candidate
contains the exact selected profile descriptor plus finite `score` in [-1,1],
`score_kind: "uncalibrated_cosine"` and `qualified: false`. Top-level
`identity_assigned`, `enrollment_performed` and `qualified` are all false. No
threshold or probability interpretation is supplied. Portable responses contain
no centroid, embedding, raw audio, credential or private filesystem path.

This Level 1 source boundary enables reviewable evidence for Speech Receipts #37.
It does not reuse campaign authority, qualify a recognition policy, establish
speaker identity, write Contacts, install a runtime, or authorize real-model
inference. Synthetic contract tests inject the existing deterministic adapter
only through the direct Python class; there is no production fake-mode option.
