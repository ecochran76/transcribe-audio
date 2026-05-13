# Policy | Architecture And Productization

## Policy

- Keep speech backends focused on transcription and normalized artifact generation.
- Put shared calendar, artifact, readout, routing, deposition, and provider abstractions in focused modules rather than growing the transcription scripts.
- Do not add a new provider, route target, or deposition surface without updating the roadmap or an active plan in the same slice.
- Treat live one-off operator workflows as fieldwork until they prove reusable.
- Before productizing fieldwork, classify the outcome as product code, runtime config, playbook/skill, local operator note, or discard.
- Keep provider-specific heuristics at the narrowest layer that can own them cleanly.
- Use behavior-preserving extraction when splitting existing large modules.
- Tests or smoke checks must protect current transcription, calendar matching, watcher retry, and output naming behavior during extraction.

## Local Boundaries

- `assembly_transcribe.py`: AssemblyAI backend.
- `faster_whisper_transcribe.py`: local faster-whisper backend.
- `watch_transcriptions.py`: watcher and backend orchestration.
- `transcribe_common.py`: current shared layer; expected to be decomposed as artifact, calendar, output, and post-processing responsibilities mature.
