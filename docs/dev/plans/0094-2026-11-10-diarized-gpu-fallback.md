# Restore AssemblyAI and diarized GPU fallback

State: CLOSED

## Scope
Restore AssemblyAI as primary using an explicitly selected environment key.
Replace active faster-whisper fallback with CUDA Parakeet TDT v3 and CUDA
Pyannote Community-1. Root session owns implementation and validation.

## Non-Goals
No automatic historical reprocessing, named speaker identification, or changes
to managed LocalAI pilot authority. No eviction of other GPU workloads.

## Current State
Voice watcher used faster-whisper alone without diarization; other watchers
also disabled diarization on fallback. VISION durable diarized transcript
outcome moves from absent (0) on this route to built (1), with an operational
route smoke as evidence toward level 3; broad measured reliability stays open.

## Acceptance Criteria
- AssemblyAI is primary and selects the requested key without account fallback.
- GPU fallback emits timed, diarized utterances and existing DOCX/TXT/JSON.
- Missing diarization fails the job; no unlabeled success.
- Configurations and running services select the new routes.

## Validation
Focused local key-selection, routing and speaker alignment tests; real short
speech CLI smoke on GPU and AssemblyAI; watcher readiness and process readback.
Hosted CI is skipped under operator policy. Sequential ASR and diarization,
45-second padded ASR chunks and a cross-watcher lock bound GPU concurrency.

## Result
AssemblyAI primary and Parakeet/Pyannote fallback are installed in both active
watchers. The separate managed pilot watcher is disabled and its 160 Downloads
processing records were preserved in the main watcher. Runtime configuration,
credentials, source media and validation artifacts remain outside Git.

Selected tests: `PYTHONPATH=/home/ecochran76/workspace.local/localai/qualification
.venv/bin/python -m pytest -q tests/test_diarized_fallback.py
tests/test_assembly_audio_upload.py tests/test_transcript_artifacts.py
tests/test_managed_transcription.py` => 58 passed (2.34s).
The initial managed regression command lacked that external qualification path;
its missing-module failure was resolved by using the documented environment.
Ruff correctness checks on changed backend/new files and new-file formatting,
compileall and diff whitespace checks pass. Existing watcher unused-import lint
is unrelated and retained.

Real public 7.435-second sample: AssemblyAI with selected account completed;
Parakeet/Pyannote completed; DOCX/TXT/JSON reopened successfully. Real private
120-second excerpt: 391 words, 13 utterances, three detected anonymous speaker
labels and UNKNOWN gaps, readable DOCX/TXT/JSON. This proves execution and
speaker separation, not measured attribution accuracy. A first additional
excerpt mistakenly selected an older similarly numbered recording; it returned
no timed speech and failed without publishing. That negative log was retained;
the corrected source was selected by exact current filename and completion time.
Final task process census and GPU memory confirmed the workers exited. Both
watchers report active and their loaded environment contains the selected key
and HF token (presence only inspected). No historical transcript was replaced.

Baseline source: dc3ca0def2a9cadc5439740e2869835611d5bba2. Final source commit is
recorded in the private validation receipt. Local RTX 5080 / CUDA / NeMo model
assets and provider-enabled smoke differ materially from hosted CI; hosted CI
was skipped under operator policy. Full operational yield and diarization
accuracy benchmarking remain outside this bounded change (manual evidence is
maturity level 2, not a dependable level 4 claim).
