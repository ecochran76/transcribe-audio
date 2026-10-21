# Scoped speaker enrollment process

`speaker_enrollment_bridge.py` supplies an additive local process contract for Speech Receipts. It accepts newly curated, explicitly reviewed samples. It does not borrow a campaign's P3/P4 enrollment grant, source corpus, calibration results or runtime directory. Existing campaign APIs and the experimental A3 custody component keep their existing boundaries. This bridge invokes the supported pinned verification adapter and A3 custody interfaces in a newly registered, dedicated operator directory.

This is source integration, not installed recognition acceptance. A profile contains a normalized centroid and exact provenance; `qualified:false` always means recognition thresholds and identity quality have not been calibrated. Explicit reviewed identity authority is required independently from a name or Contacts association. A local speaker ID can be the reviewed identity anchor; Contacts linking alone supplies no authority.

## Process and configuration

The caller invokes its explicitly configured Python interpreter and absolute bridge path, passes `--config /absolute/private/config.json`, and sends one bounded JSON request on stdin. One JSON envelope is written to stdout: `{ok:true,result:...}` or `{ok:false,error:"enrollment_..."}`. Raw library stdout is routed away from the protocol. The caller must bound/discard stderr and never display private library diagnostics. Input is limited to 12,000,000 bytes, up to eight reference samples and 30 seconds per reference.

On Linux the caller should always pass `--parent-pid <its own process id>`. The bridge installs `PR_SET_PDEATHSIG=SIGKILL` and rechecks its parent before any custody or inference effect. An unsupported platform or changed parent fails closed. The host caller owns process cancellation, exit/quiescence evidence and async progress; the bridge itself completes one synchronous request. A stopped process must be reconciled through `status`, never silently rerun.

The config must be a private regular file, not a symlink. Its product-safe shape is:

```json
{
  "directory": "/absolute/new/private/biometric-custody",
  "tenant": "operator-tenant",
  "apps": ["speech-operator"],
  "adapter_id": "speechbrain_ecapa_tdnn",
  "model_revision": "exact revision from the pinned adapter registry",
  "model_snapshot_root": "/absolute/approved/model-snapshot/models",
  "maximum_stored_bytes": 128000000,
  "authorities": {
    "reviewed-person-id": {
      "person_revision": "reviewed-binding-r1",
      "review_id": "identity-review-authority-id",
      "enrollment_id": "explicit-enrollment-authority-id"
    }
  }
}
```

The production CLI obtains only the genuine `adapter_registry` candidate and requires its exact model revision. It offers no fake-adapter/test-mode request. Registry asset verification uses its existing acquisition receipts and hashes; an explicitly configured model path is asset provenance, not enrollment authority. Neither model assets nor receipts are discovered from an implicit campaign default. Missing assets yield unavailable readiness without model load or download.

The custody directory must be empty on first registration and have `0700` permissions. The bridge records a fixed tenant/application owner marker and creates an isolated knowledge schema beneath it. It refuses to adopt an existing unregistered directory; this cannot silently migrate a live knowledge store. Private objects and the SQLite journal use owner-only access. The original source recording remains outside this copied-sample custody.

## Request contract v1

Every request carries `tenant`, `app` and an `action`. The config authorizes exact tenant/application scope before processing. Supported actions are `readiness`, `enroll`, `status`, `cancel`, and `profiles`.

`enroll` additionally carries:

```json
{
  "job_id": "persisted-operation-id",
  "person": {"id": "reviewed-person-id", "revision": "reviewed-binding-r1"},
  "authority": {
    "review_id": "identity-review-authority-id",
    "enrollment_id": "explicit-enrollment-authority-id",
    "reviewed_identity": true,
    "retained": true
  },
  "expected_predecessor": null,
  "samples": [{
    "id": "curated-sample-id",
    "revision": 1,
    "sha256": "64 lowercase hexadecimal characters",
    "data_base64": "private PCM WAV bytes",
    "source": {
      "recording_id": "retained-source-id",
      "sha256": "64 lowercase hexadecimal characters",
      "start_ms": 0,
      "end_ms": 4000
    }
  }]
}
```

The authority IDs and reviewed person revision must exactly match the private configured authority mapping. The browser must never manufacture those IDs. `retained:true` explicitly permits keeping private sample/profile copies until separately deleted under custody policy. This does not create rights or override applicable upstream terms. The host must obtain the sample bytes and metadata through its authorized, reviewed curation store. Each copied sample is hash checked and must be uncompressed mono 16-bit 16 kHz PCM WAV; frame count must match its exact 500–30,000 ms source interval. The source-media hash remains separate from the sample hash.

The bridge records exact job identity and request digest before copying bytes or invoking an embedding model. It reuses `BiometricCustodyLedger.register_sample`, profile families/versions and activation events. Normalized adapter embeddings must have the pinned dimension, finite values and nonzero norm. Their normalized mean is stored privately with exact person/sample/model/recipe provenance. Structural checks establish a well-formed profile, not speaker correctness or a measured recognition threshold.

Results include `schema_version:"transcribe-audio.speaker-enrollment-process.v1"`, `id`, `state`, `person`, `adapter_id`, `model_revision`, `recipe_revision`, `sample_manifest`, `sample_manifest_sha256`, `predecessor_profile_version_id` and `qualified:false`. Completed results also include immutable `profile_version_id` and `profile_sha256`. Raw audio, vectors and private paths never appear. The sample manifest removes `data_base64`, preserves the other exact sample fields and sorts by sample ID. Its SHA-256 hashes canonical UTF-8 JSON with recursively sorted keys, no optional whitespace and unescaped Unicode.

For an explicit rebuild, `expected_predecessor` must equal the currently selected profile ID. Failed or cancelled builds leave that head untouched. The new profile is checked and activated before the selected head and completed journal receipt change together. Predecessor versions and original sample provenance remain retained; older A3 activation history does not make an older version the selected head. `profiles` lists selected heads visible to the initiating app, marking a changed configured person revision stale. Selection changes in Speech must independently mark stale and require an explicit rebuild.

## Recovery, cancellation and retention

`status` and `cancel` take the original `job_id`. While the worker holds its exclusive process lock, `status` reports the durable running state. `cancel` records cancellation intent and returns `cancellation:"waiting_for_quiescence"`; it does not claim a native embedding call stopped. Before promotion, cancellation is checked in the journal transaction. The caller may terminate its owned process and await confirmed exit. Interrupted embedding is `unknown` and holds new enrollment, with no automatic replay.

Before activation the candidate profile ID/hash is durably recorded. If promotion's acknowledgement is lost, `status` reconciles the exact A3 activation idempotency event, active profile state and private-object hash, then atomically restores the matching selected head and terminal receipt. It never reruns embeddings. An unproven outcome remains unknown for operator inspection. Exact operation replays return the existing record; changed request digests conflict.

Private copied samples and immutable profile objects remain retained, including failed/cancelled-build evidence, under the explicit retention declaration. The original source is never deleted. Default private-object storage is capped at 128 MB, with 256 operation records; host quotas and retention review may further restrict use. Raw deletion is a separate explicit custody operation through the existing `BiometricCustodyLedger.preview_effect/apply_effect` protocol, including descendant invalidation and tombstones. This bridge does not erase historical evidence as incidental cleanup or automatically rotate a full custody root.

## Source validation

Focused tests exercise synthetic enrollment/rebuild, exact authority and tenant denial, hash/CAS failures, retained prior versions, interruption without replay, lost-promotion reconciliation without embedding, concurrent cancellation, parent guard, and metadata-only production CLI readiness with absent assets. Existing A3 custody and pinned verification regression tests remain part of acceptance. No real recording, embedding model, campaign data, provider or shared service is used. Transcription/DOCX behavior is unchanged.
