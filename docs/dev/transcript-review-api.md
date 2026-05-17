# Transcript Review API

`transcript_api.py` is the local API for the planned React + Vite review console.

It serves only the configured user-scoped transcript store. It does not read arbitrary filesystem paths from request parameters, and blob playback is limited to blob ids registered in `~/.transcripts/transcripts.sqlite3`.

## Start

```bash
python transcript_api.py --store-dir ~/.transcripts --host 127.0.0.1 --port 18876
```

Development and test runs can use `--embedding-provider debug-hash` when search should avoid a live embedder. The default port is pinned to `18876` for cooper ingress as `transcripts.localhost` and `transcripts.ecochran.dyndns.org`.

The API reads operator workflow state from `--state-dir`, defaulting to `~/.local/state/transcribe-audio`. That state remains user-scoped runtime data, not tracked repo content.

When `frontend/dist/` exists, the same server also serves the built React console at `/`; API routes remain under `/api`.

## Endpoints

- `GET /api/health`: service and store path.
- `GET /api/library?kind=transcript&limit=50&offset=0`: paged stored document list.
- `GET /api/review-queue?limit=50`: read-only review queue aggregation over local route-review files, filename-conflict reviews, and first-pass summary queue counts.
- `POST /api/review-queue/first-pass-summaries/prepare`: create a dry-run first-pass summary batch manifest without submitting provider work.
- `POST /api/review-queue/first-pass-summaries/submit`: submit an existing prepared manifest. Requires `approval_token=SUBMIT_FIRST_PASS_SUMMARY_BATCH`.
- `POST /api/review-queue/first-pass-summaries/status`: poll a submitted manifest and optionally materialize completed readouts with `materialize=true`.
- `GET /api/search?q=<query>&kind=transcript&limit=10`: lexical/semantic search over stored artifacts.
- `GET /api/documents/<document_id>`: document detail, JSON payload, text content, metadata, and linked blobs.
- `GET /api/documents/<document_id>/context?chunk_index=5&context_chunks=1`: nearby transcript/readout context from stored chunks.
- `GET /api/blobs/<blob_id>`: registered blob playback/download endpoint with `Range` support.
- `GET /api/blobs/<blob_id>?download=1`: same blob as an attachment.

## Blob Contract

Transcript ingestion copies existing source recordings into:

```text
~/.transcripts/blobs/<prefix>/<blob-id>.<ext>
```

The SQLite store records:

- `blobs`: blob id, original path, stored path, hash, MIME type, byte size.
- `document_blobs`: document-to-blob links and roles such as `source_recording`.
- document metadata `media_blob`: compact frontend-facing playback/download URLs.

The UI should play recordings through `/api/blobs/<blob_id>` rather than using original `~/Downloads` paths. This keeps playback stable after the source file is moved or deleted and prevents arbitrary path streaming.

## Review Queue Contract

`/api/review-queue` returns:

- `buckets`: summary cards for route reviews, filename conflicts, first-pass summaries, memory harvest, and speaker ID work.
- `items`: currently route-review files from `~/.local/state/transcribe-audio/review-queue/`.
- `route_decision_exists`: whether a route-review item still points at a readable route-decision artifact.
- `status=stale_reference`: a local review item exists, but its referenced route decision is gone, commonly from earlier pytest/temp runs.

The queue aggregation endpoint is read-only and intentionally reports stale references instead of deleting or hiding them.

Use `review_queue_maintenance.py` for reviewed cleanup of stale local
route-review files. It is dry-run by default and requires
`--apply --approval-token ARCHIVE_STALE_ROUTE_REVIEWS` before moving files to
`~/.local/state/transcribe-audio/review-queue-archive/<run-id>/`.

First-pass summary preparation writes a dry-run manifest under
`~/.local/state/transcribe-audio/first-pass-summary-batches/`, returns the
manifest path and request count, and leaves `batch=null`. It does not submit
provider work. Submit and status actions are manifest-scoped: the API refuses
manifest paths outside that directory, submit requires an explicit approval
token, and status can materialize completed provider results back into the
store when requested.

## Security Boundary

This API is currently local and exposes manifest-scoped first-pass summary batch actions. Operator login and scoped share links are planned for a later P09 slice and should follow the `previews` model: single-operator guard for operator routes and revocable token-hash-backed share links for scoped reviewer access.

Do not expose this service publicly without an auth layer.
