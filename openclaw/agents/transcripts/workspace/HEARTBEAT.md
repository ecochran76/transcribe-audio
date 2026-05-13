# Heartbeat

Default response when no action is needed: `HEARTBEAT_OK`.

Periodic checks should stay light:

- Confirm whether `transcribe-watch.service` is active.
- Look for recent failed watcher records or queued review items.
- Report only actionable failures, blocked queues, or new review needs.

Do not post raw transcript content during heartbeat checks.

