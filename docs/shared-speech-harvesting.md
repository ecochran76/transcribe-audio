# Shared speaker samples

The meeting console can ask Speech Receipts to extract playable speaker clips,
keep reviewed samples and propose a canonical Contacts person link. Speech owns
selection, retained audio, sample quality and sample associations. Transcribe
Audio owns original recordings, diarization, meeting context and participant
corrections. A recording speaker label is not a person identity.

## Optional configuration

The default API and frontend build need no Speech checkout, Node subprocess,
provider, model or credentials. Without configuration the panel reports that
shared tools are not connected. Start the existing API with
`--speech-harvest-config /absolute/private/speech-harvesting.json` to configure it.
The configuration and token files must belong to the current OS user, be regular
files with mode 0600 and remain outside Git. The state directory must be absolute
and private. It contains bounded operation/observation metadata, not media copies.
One active consumer holds its custody lock.

```json
{
  "state_directory": "/absolute/private/ta-shared-speech",
  "speech_url": "https://your-configured-speech-host/",
  "speech_token_file": "/absolute/private/speech-client-token",
  "source_token_file": "/absolute/private/ta-source-token",
  "observation_token_file": "/absolute/private/ta-observation-token",
  "owner": {"tenant": "your-tenant", "app": "your-ta-app", "account": "your-recordings"},
  "allowed_origins": ["https://your-meeting-console-host"]
}
```

Use distinct independently scoped tokens. Omit `observation_token_file` to keep
foreign observations disabled. Bind the exact same tenant/app/account in Speech's
`transcribeHarvestingSource` adapter. The synchronization peer's `ownerApp` is this
configured app, which may differ from the management caller app. The URL accepts
HTTPS or explicit loopback HTTP only, with no credentials/query/fragment or
redirects. No default host or fallback provider is inferred.

The browser uses the existing authenticated meeting ingress. Service source and
observation routes independently validate their bearer tokens; their configured
network route must reach the API without requiring a human browser cookie. This
source packet does not install or change ingress. Do not expose the general
meeting API by treating a source/observation token as a browser login.

## Workflow and limits

In a meeting's Speakers tab, choose Find speaker samples, listen, and explicitly
keep a reviewed sample. Person search sends an attributed proposal to Speech;
it does not accept a person or change meeting speaker choices. Incoming Speech
observations remain pending/conflicting evidence for separate meeting review.
No enrollment, clone pin, global person update or automatic replacement occurs.

The consumer admits at most 1,024 journal operations and 64 MB metadata; one
source is at most 32 MB, timing metadata 3 MB/1,000 turns/20,000 words, and a
service response 4 MB. Requests have bounded timeouts and no redirects. Export
references expire after ten minutes and bind an admitted operation, exact owner,
recording revision/hash and timing. Existing registered original media is read
only after admission. Unknown word timestamp units are omitted rather than
inferred by magnitude. Known Assembly millisecond and faster-whisper second
exports are normalized explicitly.

An exact request ID and canonical intent are persisted before writes. Missing
acknowledgements and interrupted pending operations become unknown. Checking the
same operation uses supported read-only status; it does not repeat extraction,
retention or person proposals. Changed intent under an existing ID is rejected.

## Supported service transport

All paths below are API-relative. `documentId` identifies the meeting and
`harvestOperationId` is the TA-admitted request ID, not the Speech request ID.

- `GET /api/speech-harvest/sources/:harvestOperationId`, source bearer:
  returns exact `{owner,source,audioBase64}` for the admitted short-lived source.
- `GET /api/speech-harvest/references`, observation bearer: returns
  `{revision,owner,references}`. Revision is an opaque SHA-256 equality token;
  this is a bounded snapshot, not an event stream. It lists actual retained
  membership even before observations and reads the current Speech sample and
  accepted association snapshot. Unavailable samples and pending associations
  are explicit states.
- `POST /api/conversations/:documentId/speech-harvest/:harvestOperationId/observe`,
  observation bearer: appends an exact idempotent foreign observation.
- `GET /api/conversations/:documentId/speech-harvest/:observationRequestId`,
  observation bearer: returns only that owner/document-bound observation receipt.
  Anonymous, wrong-scope and foreign-document access is rejected. Optional POST
  to the same receipt path plus `/reconcile` performs the same read.

An observation requires `requestId`, `sampleId`, `sampleRevision` (positive),
`associationRevision` (nonnegative), `sourceId`, `sourceRevision`, `sourceSha256`,
`diarizationSha256`, `actor:{tenant,app}`, `person` and nonempty `evidence`.
Person is null or `{kind:"entity",id:UUID,revision:string}`. Source and timing
must match locally journaled applied retention membership; timing is the exact
completed Speech job's digest, distinct from TA's preflight digest. The receipt
preserves these inputs plus document/harvest IDs, owner, `state` proposed/conflict
and `meetingAssignmentChanged:false`. Competing same-revision observations remain
append-only conflicts. No name matching or accepted meeting assignment occurs.

Original recording purge is unavailable through this adapter. Foreign observation
or derived-copy deletion cannot prove original-source erasure.

## Source qualification

Run hermetic boundary tests without a sibling checkout:

```sh
.venv/bin/python -m pytest -q tests/test_speech_harvesting.py
```

Three actual cross-service synthetic cases additionally require an explicitly
built Speech module. This opt-in path is not a default test dependency:

```sh
TRANSCRIBE_SPEECH_INTEGRATION_MODULE=/absolute/speech/dist/index.mjs \
  .venv/bin/python -m pytest -q tests/test_speech_harvesting.py
```

The integration lane owns disposable authenticated loopback services, synthetic
media and private temporary stores. It checks actual playback/retention/proposal
boundaries, source drift, cancellation, interrupted journals, lost write responses
and observer conflict/reconciliation. Source tests and a rendered synthetic panel
do not establish installed ingress, real media quality or authority for frozen
speaker pilots, provider inference, enrollment or deployment.

## Optional original-owner erasure

The consumer profile may separately configure `source_owner` with `enabled:true`,
a protected `token_file`, an explicit `principals` list of `{tenant,app}`, and
`exclusive_original_roots` of absolute owner-controlled recording directories.
The owner token must differ from Speech/export/observation tokens and stays
server-side. Ordinary application bearer credentials and browser cookies do not
replace it. Owner APIs fail unavailable when the optional profile is absent.
Do not enable a profile until all processes that own these recording paths are
accounted for; unregistered external copies are not silently declared erased.

Speech's `transcribeSourceOwner` participant calls authenticated POST
`/api/speech-harvest/source-erasure/{inventory,fence,purge,status}`. Each request
binds its configured principal and exact source `{service,account,id,revision,
sha256}`; effects also require the original requestId and inventory fingerprint.
Inventory lists opaque copy IDs, not private filesystem paths. The overall Speech
coordinator separately requires explicit owner approval and all other participant
fences before any purge. It purges the original owner last.

This owner covers registered blob files, their canonical pre-registration siblings,
recorded/historical original files and supported browser playback copies. Current
ingestion records original-container digests, allowing different encodings to be
bound without guessing from a filename. Unknown legacy original digests or an
original shared with another canonical source remain unavailable. Symlinks,
hardlinks, foreign-owned paths, changed content and copies outside explicit roots
are refused. Current limits are256 copies,32MB/file,128MB total and4096 owner
operation receipts; these are separate from general transcription media limits.

Durable source and known-copy hash fences block supported blob/export reads and
new ingestion before media effects. Full streaming/export publication and registered
ingest paths share a reentrant cross-process SQLite exclusion lock. A pending
reader/writer prevents a quiescence acknowledgement; a new copy invalidates an old
preview. Long ingestion may delay a fence; an HTTP timeout never proves cleanup.
Original-owner POSIX identity checks are unavailable on unsupported platforms;
ordinary non-owner imports use only portable standard-library dependencies.

Purge journals its phase before moving exact files into private quarantine and
checking their identity before unlink. Status resumes only that same recorded
purge after an uncertain acknowledgement or restart; it never performs inference.
Completed receipts retain hashes/IDs while private file lists and voice observations
are scrubbed. Original/container byte hashes remain fenced. Ordinary deletion and
meeting participant decisions retain their existing semantics. No live owner
profile is installed by adding this source capability.
