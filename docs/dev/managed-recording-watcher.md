# Managed diarized recording watcher backend

Use the existing LocalAI-managed pilot recording endpoint and its private client
configuration. This backend submits ASR and diarization to that manager; it does
not start models or ask the operator to manage model residency. It requires the
pilot to register both stage profiles and authorize the recording principal.

The watcher route is an explicit opt-in. Existing AssemblyAI jobs and defaults
remain unchanged. Do not change live watcher configuration merely to install
this source capability.

```json
{
  "jobs": [{
    "name": "managed-recordings",
    "watch_dir": "/path/to/approved/recordings",
    "backend": "managed_recording",
    "cli_args": ["--config", "/private/managed-recording-client.json", "--text-output"]
  }]
}
```

Use the watcher's documented configuration envelope and scheduling fields.
`--config` points to the existing `managed_recording_cli.py` client configuration:
`endpoint`, `token_file`, optional `ca_file`, `processor` (private state/spool,
unit size, pinned ASR/DIA profile bindings, reviewed speaker thresholds), and
`output_directory`. Keep normal TLS verification and the pilot's CA. Config and
token files must be private owned files; never put token values in watcher args.
The manager config must match the installed pilot's returned stage bindings.

`managed_recording_transcribe.py MEDIA --config CONFIG` is also usable directly.
It always publishes DOCX, TXT and JSON; `--text-output` is accepted for watcher
compatibility. It deliberately rejects unsupported calendar/shared CLI flags.
For an existing private transcript use `--existing-transcript PATH`; accepted
IDs, calendar metadata and previous artifact provenance are preserved by the
existing recording CLI. Optional `--recording-id` and `--conversation-id` must
agree with an existing ingress/transcript identity.

Before the first manager request, a private locked ingress manifest persists the
recording/conversation UUID pair under `processor.state_directory/watcher-ingress`
(or explicit `--ingress-directory`). Canonical source path plus full content SHA
selects its identity. Restart/retry resumes the same CLI checkpoint and published
generation; changed content receives a new pair. Preserve these state directories.
Configuration/metadata conflicts fail rather than silently replacing a generation.

Only a complete publication emits `TRANSCRIPT_ARTIFACT_JSON=...`, allowing normal
watcher readout/store processing. Choose one store-ingestion owner: when watcher
store ingestion is enabled, omit the managed CLI's optional debug-hash store
configuration. A failure/partial result emits no success marker. Watcher backoff
and the existing bounded CLI stage retry policy apply; unknown requests are not
replayed as new work. Managed recording jobs cannot contain cross-route fallbacks.
This source integration does not activate a watcher or qualify native model
quality, performance or lifecycle.
