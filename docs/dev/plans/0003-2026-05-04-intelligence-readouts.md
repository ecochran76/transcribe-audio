# Plan 0003 | Intelligence Readouts

State: CLOSED

Lane: P03

## Scope

Generate automatic summaries and contextual readouts from transcript artifacts.

## Non-Goals

- No matter deposition.
- No automatic memory writes.
- No high-confidence routing claims without a route schema.

## Current State

Readout generation is implemented as a bounded post-processing stage. `summarize_transcript.py` reads `*.transcript.json`, calls an OpenAI-compatible chat completions API or `codex exec`, and emits `*.readout.json` plus `*.readout.md`. Watcher jobs can enable readout generation behind config. Calendar overlap context from `event.matching_calendars` is passed in a dedicated `calendar_context` prompt block for meeting-type and matter-candidate reasoning. The prompt duplicates the JSON-only contract in the user payload for browser-backed OpenAI-compatible providers that may ignore system messages. Local-compatible, real AuraCall, and `codex-exec` provider smokes passed.

## Work Items

- Done: define `Readout` JSON schema.
- Done: add `summarize_transcript.py`.
- Done: implement OpenAI-compatible provider first.
- Done: implement `codex-exec` provider.
- Done: define provider seams for `auracall` and `openclaw`.
- Done: emit `*.readout.json` and `*.readout.md`.
- Done: wire watcher post-processing behind config.
- Done: include `event.matching_calendars` in readout prompt context.
- Done: smoke against a real configured AuraCall provider/key.

## Acceptance Criteria

- Readouts include summary, participants, topics, action items, unresolved questions, matter candidates, and memory candidates.
- Provider credentials are resolved without committing secrets.
- Failures do not mark transcription itself as failed.

## Validation

- Mock OpenAI-compatible API tests.
- Manual readout generation from an existing transcript artifact.
- Real AuraCall readout smoke using `agent:pro-extended-chatgpt-soylei`.
- Real `codex-exec` readout smoke using `gpt-5.5`.
