# Managed LocalAI pilot consumer

State: CLOSED
Owner: LocalAI ticket0158 worker
Work item: LocalAI repo qualification/docs/tickets/0158-transcribe-audio-managed-route.md
Branch: codex/localai-pilot-0158
Base:3c561b380929f577070266c197982a7d14c107ee

## Vision outcome and maturity

Advances VISION outcome1 (durable trustworthy transcript) and autonomy through
shared resource admission while preserving this application's canonical output.
Managed local ASR moves from0Absent to1Built via public synthetic client/CLI/export
checks. Shadow/operational real-media and native GPU evidence belongs to the
coordinator's deployment ticket0160; this source slice makes no operational claim.
Speaker identity/contextualization/readout maturity is unchanged.

## Outcome and scope

Explicit --managed-config route with private settings/token/spool/state, pinned
profile and original controls. Direct mode remains unchanged. Whole-file bounded
staging, stable durable request IDs, safe queue retry, observe-only unknown recovery,
pre-export canonical effect guard and no mixed watcher fallback are implemented.
No installed service, live route, model/provider call or private recording changed.

## Validation

Red/green at tests/test_managed_transcription.py public seams; tests include actual
original main/output pipeline generating synthetic TXT/DOCX without Whisper load.
Watcher/export/readout focused regression results and environment exclusions are
recorded in the LocalAI private pilot0158-evidence receipt. Manual CLI subprocess
smoke uses synthetic short WAV and authenticated loopback test server, not a model.
Exact result commit and final commands are supplied to the primary coordinator.

## Boundaries

Canonical export is not made transactionally atomic across arbitrary existing
side effects. A crash/failed export remains unknown and is not replayed, preserving
existing output custody. Root deployment owns live acceptance and final review.

Final focused command: PATH=/home/linuxbrew/.linuxbrew/bin:$PATH
PYTHONPATH=/home/ecochran76/workspace.local/localai/qualification
/home/ecochran76/workspace.local/transcribe-audio/.venv/bin/python -m pytest
tests/test_managed_transcription.py tests/test_transcript_artifacts.py
tests/test_readouts.py -q =>63passed in14.42s. Initial unconfigured PATH run
failed one pre-existing readiness assertion because ffprobe was absent from PATH;
rerun used the existing explicit tool directory, no binary installation.
Real CLI subprocess against an authenticated disposable HTTP test server, short
synthetic PCM WAV and original TXT/DOCX exporters passed. First30second smoke
timed out under host pressure; preserved as negative evidence, bounded120second
smoke completed. No claim of native ASR quality/latency is made.
