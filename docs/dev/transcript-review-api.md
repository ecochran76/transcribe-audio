# Transcript Review API

`transcript_api.py` is the local read API for the planned React + Vite review console.

It serves only the configured user-scoped transcript store. It does not read arbitrary filesystem paths from request parameters, and blob playback is limited to blob ids registered in `~/.transcripts/transcripts.sqlite3`.

## Start

```bash
python transcript_api.py --store-dir ~/.transcripts --host 127.0.0.1 --port 18876
```

Development and test runs can use `--embedding-provider debug-hash` when search should avoid a live embedder. The default port is pinned to `18876` for cooper ingress as `transcripts.localhost` and `transcripts.ecochran.dyndns.org`.

When `frontend/dist/` exists, the same server also serves the built React console at `/`; API routes remain under `/api`.

## Endpoints

- `GET /api/health`: service and store path.
- `GET /api/library?kind=transcript&limit=50&offset=0`: paged stored document list.
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

## Security Boundary

This API is currently local and read-only. Operator login and scoped share links are planned for a later P09 slice and should follow the `previews` model: single-operator guard for operator routes and revocable token-hash-backed share links for scoped reviewer access.

Do not expose this service publicly without an auth layer.
