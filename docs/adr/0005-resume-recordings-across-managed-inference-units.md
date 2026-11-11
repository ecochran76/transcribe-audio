# Resume recordings across managed inference units

Status: accepted design; implementation and qualification pending.

Historical recordings may be too long to process as one GPU session while meeting
interactive responsiveness requirements. Process them as resumable bounded work
units admitted by the existing Shared GPU service; Transcribe Audio owns durable
recording progress, source timeline and canonical transcript artifacts. Completed
units survive interruption, while uncertain outcomes require reconciliation before
replay. The service retains sole GPU admission, residency and release authority.

Preserve raw and intermediate evidence, then reconcile anonymous speakers across
the recording before publishing its final transcript. Independent per-unit label
numbers are not speaker identity; unresolved attributions remain explicit. This
accepts added checkpoint/reconciliation complexity in exchange for interruption
and long-recording progress, rather than requiring one uninterrupted whole-file
GPU operation. It does not authorize cross-recording person identification.

Source: accepted interview decisions Q8, Q9 and Q12 in
[design notes](../dev/notes/0062-2026-11-10-managed-transcription-design-interview.md).
