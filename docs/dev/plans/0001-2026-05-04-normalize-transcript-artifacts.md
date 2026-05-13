# Plan 0001 | Normalize Transcript Artifacts

State: CLOSED

Lane: P01

## Scope

Create a stable machine-readable artifact contract for every completed transcription.

## Non-Goals

- No intelligence summarization.
- No matter routing.
- No deposition outside existing output directories.

## Current State

The CLIs generate DOCX/TXT/SRT outputs and emit `*.transcript.json` sidecars through the shared output path. The watcher parses `TRANSCRIPT_ARTIFACT_JSON=<path>` from backend stdout and persists artifact paths in processed state. A temp-location watcher smoke on a non-sensitive short recording validated DOCX/TXT/sidecar generation and watcher artifact capture.

## Work Items

- Done: add `TranscriptArtifact` and related serialization.
- Done: emit `*.transcript.json` sidecars from the shared output path.
- Done: include output paths, event metadata, transcript text, and structured utterances.
- Done: record sidecar paths in watcher state.
- Done: add tests for serialization and watcher state handling.
- Done: manual smoke on a short non-sensitive recording through watcher calendar config and `--text-output`.
- Done: keep v1 artifact schema as-is for the next lane; richer backend metadata can be added when readouts/deposition need it.

## Acceptance Criteria

- AssemblyAI and faster-whisper paths both emit equivalent artifact JSON.
- Existing CLI output filenames remain compatible.
- Watcher can find the artifact for a processed file.

## Validation

- Unit tests for artifact serialization.
- Manual smoke on a short recording with `--text-output --use-calendar`.
