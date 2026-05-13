# AuraCall Transcribe Ownership Handoff

Date: 2026-05-13

## Purpose

This note hands `transcribe-audio` back to its own repo ownership after the
AuraCall-side repair and readiness work. It records what changed, what was
verified live, and what the next transcribe-audio agent should do without
reconstructing state from AuraCall chat history.

## Policy Context

- `AGENTS.md` is the repo policy entrypoint.
- Durable status belongs in `RUNBOOK.md` and dated notes under `docs/dev/notes/`.
- Repo memory group is `transcribe_audio_main`.
- Graphiti runtime was checked before writing this note:
  `graphiti-runtime doctor` reported healthy.
- A focused Graphiti discovery query returned older P03/readout context, but no
  newer AuraCall batch-readiness episode than the repo docs and current runbook.

## What Changed

Transcribe Audio now has a first-class AuraCall batch path for legacy readout
enrichment:

- `scripts/auracall_legacy_enrichment_batch.py` submits pending legacy
  transcript readouts to AuraCall response batches.
- The default model is
  `agent:pro-extended-chatgpt-soylei-transcripts`.
- The scoped client env is user-scoped at:
  `/home/ecochran76/.local/state/transcribe-audio/auracall-transcripts.env`.
- `summarize_transcript.py` keeps the full-transcript path and shared readout
  materialization behavior.
- Transcript-length limiting was explicitly rejected as the wrong fix. Do not
  reintroduce caller-side transcript truncation as the default failure response.

AuraCall owns:

- large prompt transport
- request attachment fallback
- browser-backed provider execution
- project binding
- response-batch queueing
- browser interaction rate limiting
- honest API failure reporting

Transcribe Audio owns:

- choosing transcript queue items
- constructing readout prompts
- submitting bounded AuraCall batches
- polling batch status
- materializing completed readouts
- storing readout artifacts in the transcript store

## Current Verified State

The running AuraCall service currently advertises and accepts the transcript
agent:

```text
agent:pro-extended-chatgpt-soylei-transcripts
```

The agent is registry-backed and bound to the ChatGPT project:

```text
projectName=Transcripts
projectId=g-p-6a04628762ac8191894b16cfaddfd126
service=chatgpt
runtimeProfile=wsl-chrome-3
modelSelector=chatgpt:pro-extended
```

Live scoped-client smoke passed:

```bash
pnpm run smoke:scoped-client-env -- \
  /home/ecochran76/.local/state/transcribe-audio/auracall-transcripts.env \
  --prompt 'Reply exactly: auracall transcribe env ok' \
  --expect-output 'auracall transcribe env ok' \
  --timeout-ms 180000
```

Result:

```text
response=resp_45008e83347940909bcdba697b91fa2c
status=completed
output=auracall transcribe env ok
```

Previous one-item live batch also completed:

```text
batch=batch_0db1883c7905471c83d807411cfdee33
child=resp_1a4b0915303848a6ab68a48e286e563f
```

Materialized and stored readout:

```text
/home/ecochran76/.transcripts/legacy-artifacts/29/29ed3d64cca92a7cf5f5-2025-08-15 Dr Stefl Knee Replacement Consult.readout.json
```

At last recorded queue check, the de-duped pending legacy enrichment queue had
57 items.

## How To Resume

Start with a small bounded batch, not the full backlog.

Recommended dry run:

```bash
.venv/bin/python scripts/auracall_legacy_enrichment_batch.py \
  --env-file /home/ecochran76/.local/state/transcribe-audio/auracall-transcripts.env \
  enqueue \
  --limit 3 \
  --store \
  --dry-run
```

Recommended first live batch:

```bash
.venv/bin/python scripts/auracall_legacy_enrichment_batch.py \
  --env-file /home/ecochran76/.local/state/transcribe-audio/auracall-transcripts.env \
  enqueue \
  --limit 3 \
  --store \
  --max-concurrent-runs 2 \
  --max-browser-interactions-per-minute 8
```

Then poll and materialize from the emitted manifest:

```bash
.venv/bin/python scripts/auracall_legacy_enrichment_batch.py \
  --env-file /home/ecochran76/.local/state/transcribe-audio/auracall-transcripts.env \
  status <manifest-path> \
  --materialize \
  --store
```

The script prints `AURACALL_BATCH_MANIFEST=<path>` after enqueue. Use that
manifest path for status and materialization.

## Guardrails

- Do not print, commit, or copy API keys from the scoped env.
- Keep real runtime state under user-scoped paths such as
  `~/.local/state/transcribe-audio`, `~/.transcripts`, and `~/.auracall`.
- Do not downgrade the caller by shortening transcripts to make provider
  failures disappear.
- Treat AuraCall HTTP 502 or failed child responses as real provider/runtime
  failures that need retry or AuraCall diagnosis.
- Use small live batches until multiple rounds complete cleanly.
- Keep batch concurrency and browser interaction limits in the AuraCall batch
  request; do not implement separate browser pacing in this repo.

## Next Owner Actions

1. Run a three-item dry run and inspect the manifest for the expected model,
   JSON response-format metadata, full prompt payloads, and limits.
2. Run one three-item live batch with `maxConcurrentRuns=2` and
   `maxBrowserInteractionsPerMinute=8`.
3. Materialize completed children with `status --materialize --store`.
4. Re-check the de-duped pending queue count.
5. If all three complete, increase batch size gradually while keeping AuraCall
   responsible for concurrency and browser rate limiting.
6. If any child fails, preserve the manifest and response ids, then diagnose
   the AuraCall run rather than editing transcript length or prompt transport.

## Evidence Commands

These commands were used as proof points during the handoff. They intentionally
avoid printing secrets.

```bash
awk -F= '/^(OPENAI_BASE_URL|AURACALL_MODEL)=/{print $1"="$2} /^(OPENAI_API_KEY)=/{print $1"=<redacted>"}' \
  /home/ecochran76/.local/state/transcribe-audio/auracall-transcripts.env
```

```bash
pnpm run smoke:scoped-client-env -- \
  /home/ecochran76/.local/state/transcribe-audio/auracall-transcripts.env \
  --prompt 'Reply exactly: auracall transcribe env ok' \
  --expect-output 'auracall transcribe env ok' \
  --timeout-ms 180000
```

```bash
git diff --check
```

## Ownership Note

From this point, the next transcribe-audio slice should be driven from this
repo's queue, store, and runbook. AuraCall should be treated as the execution
service. If the batch path fails, capture the AuraCall batch id, child response
ids, manifest path, and exact HTTP failure before changing transcribe-audio
behavior.
