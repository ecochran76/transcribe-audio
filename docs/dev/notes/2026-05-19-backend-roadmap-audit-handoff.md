# Backend Roadmap Audit Handoff | 2026-05-19

## Purpose

Hand off the current backend development state after the AuraCall dispatch-pool
first-pass summary work cleared the pending queue. This note is for the next
agent working in `transcribe-audio`, especially on P09 backend/API work.

## Current Repo State

- Latest repo commit before this note: `039993c Record final dispatch pool readouts`.
- `ROADMAP.md` and `docs/dev/plans/0009-2026-05-12-react-vite-review-console.md`
  have been updated to reflect the current backend state.
- User-scoped runtime state remains outside the repo:
  - store: `~/.transcripts`
  - workflow state: `~/.local/state/transcribe-audio`
  - AuraCall client env: `~/.local/state/transcribe-audio/auracall-transcripts.env`
- `transcripts.service` and `transcribe-watch.service` are active.

## Live Backend Snapshot

Verified on 2026-05-19:

- `GET /api/health` returns `status=ok` for
  `/home/ecochran76/.transcripts/transcripts.sqlite3`.
- Live store counts:
  - `documents`: 240 total
  - `transcript`: 164
  - `readout`: 74
  - `contextual_readout`: 2
  - `blobs`: 122
  - `document_blobs`: 144
  - `document_chunks`: 6560
- `GET /api/library?limit=1` returns current readout rows from the AuraCall
  dispatch-pool batches.
- `GET /api/review-queue?limit=100` reports `total_open=0`.
- `transcript_store.py first-pass-summary-queue --format compact-json --limit 5`
  reports `selected_count=0`.
- `GET /api/search?q=SoyLei&limit=3` returns ranked readout results.

## Backend Surface Implemented

`transcript_api.py` currently supports:

- `GET /api/health`
- `GET /api/library`
- `GET /api/review-queue`
- `GET /api/search`
- `GET /api/documents/<document_id>`
- `GET /api/documents/<document_id>/context`
- `GET /api/blobs/<blob_id>` with range support
- `POST /api/review-queue/first-pass-summaries/prepare`
- `POST /api/review-queue/first-pass-summaries/submit`
- `POST /api/review-queue/first-pass-summaries/status`

The first-pass summary write path is manifest-scoped and gated:

- prepare creates a dry-run manifest under user-scoped state;
- prepare honors `AURACALL_DISPATCH_TEAM` and `AURACALL_DISPATCH_MODEL` from
  the configured AuraCall env file;
- submit requires `approval_token=SUBMIT_FIRST_PASS_SUMMARY_BATCH`;
- status may materialize completed readouts back into the store;
- paths outside the allowed first-pass summary batch directory are rejected.

## Roadmap Audit

P01, P02, P03, P07, and P08 remain closed.

P04 and P05 remain open because routing/contextual reread and deposition/memory
harvest have core CLI contracts, but not the full UI-backed workflow or all
external apply surfaces.

P06 remains open. The watcher and transcript API are running under systemd, but
readiness/blocked-reason observability is still mostly operational rather than
productized.

P09 remains the active backend/UI lane. Backend read surfaces and first-pass
batch operations are now substantially ahead of the original plan text, but
these pieces are still missing:

- operator login guard and scoped share-link model;
- contact, identity, speaker-assignment, and merge-audit tables;
- context-run/provenance API surfaces;
- intelligence provider readiness/status API surfaces;
- deposition and memory-harvest review/apply API surfaces;
- broad API auth/authorization boundary beyond local-only service posture;
- migration/backfill for older stored transcripts that still lack blob links.

## Recommended Next Slice

Start P09 contact/speaker backend foundations before adding more UI chrome:

1. Add SQLite tables for contacts, contact identities, speaker assignments, and
   contact merge events.
2. Add read/write-preview API endpoints for speaker assignment review with
   explicit reviewer/audit metadata.
3. Add tests in `tests/test_transcript_api.py` and `tests/test_transcript_store.py`.
4. Update `docs/dev/transcript-review-api.md`, `ROADMAP.md`, and this plan lane
   with the new contract.

This is the highest-value backend slice because the first-pass queue is clear,
the review queue is clear, and the next operator bottleneck is turning readouts
into reviewed people/matter context rather than generating more summaries.

## Verification Commands Used

```bash
curl -fsS http://127.0.0.1:18876/api/health | jq .
curl -fsS 'http://127.0.0.1:18876/api/library?limit=1' | jq '{total, first:(.items[0] // null)}'
curl -fsS 'http://127.0.0.1:18876/api/review-queue?limit=100' | jq '{total_open, buckets:[.buckets[] | {id,count,status,detail}], item_count:(.items|length)}'
curl -fsS 'http://127.0.0.1:18876/api/search?q=SoyLei&limit=3' | jq '{query, count:(.results|length), sample:[.results[] | {kind,title,score,best_chunk:.best_chunk.chunk_index}]}'
sqlite3 /home/ecochran76/.transcripts/transcripts.sqlite3 "select kind,count(*) from documents group by kind order by kind; select 'blobs',count(*) from blobs; select 'document_blobs',count(*) from document_blobs; select 'chunks',count(*) from document_chunks;"
systemctl --user is-active transcripts.service transcribe-watch.service
.venv/bin/python transcript_store.py first-pass-summary-queue --format compact-json --limit 5
.venv/bin/python -m pytest tests/test_transcript_api.py tests/test_transcript_store.py tests/test_review_queue_maintenance.py -q
python -m py_compile transcript_api.py transcript_store.py review_queue_maintenance.py
npm --prefix frontend run build
git diff --check
```
