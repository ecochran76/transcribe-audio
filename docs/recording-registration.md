# Register an existing recording and transcript

Transcribe Audio can retain an existing raw recording and a verbatim JSON
transcript sidecar in its native transcript store. This admission supports the
VISION outcomes for durable transcripts and replayable source artifacts. It
moves import from an operator CLI step to an authenticated service operation.
The local acceptance gate verifies both exact byte streams by independent
readback and checks that replaying a request returns the same native IDs.

Enable the route explicitly with
`transcript_api.py --recording-registration-token-file /private/registration.token`.
Use a service-owned, absolute, mode-0600 file with a dedicated random bearer
token of at least 32 characters. The API must bind to loopback. For approved
remote callers, terminate TLS and authenticate access at the deployment's
private application ingress. Configure its request size and idle-time limits
for the expected recordings.

Send `POST /api/recording-registrations` with an
`application/octet-stream` body composed of the original transcript JSON
bytes followed immediately by the original recording bytes. The transcript
must include a string `transcript_text`. Use these headers:

| Header | Value |
| --- | --- |
| `Authorization` | Bearer token from the dedicated file |
| `Content-Length` | Combined byte count |
| `X-Transcript-Bytes` | Transcript prefix byte count, up to 16 MiB |
| `X-Transcript-Sha256` | Lowercase SHA-256 of the exact transcript JSON |
| `X-Recording-Sha256` | Lowercase SHA-256 of the exact raw recording, up to 2 GiB |
| `X-Recording-Extension` | Recording filename suffix such as `.wav` or `.m4a` |

The response returns `native_document_id`, `native_recording_id`,
`transcript_sha256` and `recording_sha256`. A new registration returns 201.
A valid replay returns 200 and the same identities. Each transcript version
bound to the same original recording gets its own document and reuses the
recording blob. Sidecar paths remain in the retained verbatim copy and are
removed from the locally actionable document projection. This operation
uses the server's configured indexing backend and runs no transcription or
speaker assignment.

Read the accepted binding at `GET /api/documents/{native_document_id}`.
Retrieve original recording bytes at `GET /api/blobs/{native_recording_id}`
through the regular private application access. Retrieve the exact transcript
bytes at `GET /api/recording-registrations/{native_document_id}/transcript`
using the same registration bearer. Keep derived speaker corrections
separately from the registered transcript bytes. The native media erasure fence
continues to apply.

## Correct transcript wording

The same private registration bearer can submit a committed human correction
at `POST /api/transcript-corrections`. Each JSON payload carries its
`document_id`, the two original artifact SHA-256 values, and a `target` with
`start_ms`, `end_ms`, `speaker`, and `machine_text`. The target must match
one native source-time utterance. Separate working-media timelines require
alignment review. A correction carries `text`, `workspace_id`,
`correction_id`, `revision`, `actor_id`, `authored_at`, and `reason`.
Compute `id` from SHA-256 of the compact UTF-8 JSON array
`[workspace_id, correction_id, revision]`. Send `previous_id` as null for
the first correction or the current accepted correction ID for that
utterance.

A compatible append returns 201. Replaying the same payload returns 200.
An incompatible target or prior revision returns 409. Retrieve the accepted
ledger at `GET /api/transcript-corrections?document_id={id}` with the same
bearer. An optional `target_key` filters to one utterance and adds target
verification. Compute it as SHA-256 of the compact UTF-8 target JSON
object with lexicographically sorted keys. Original transcript bytes
and machine wording remain available.
