# AuraCall ChatGPT Target Evidence Handoff

Date: 2026-05-17

## Failure Signature

- Transcribe first-pass retry manifest:
  `~/.local/state/transcribe-audio/auracall-batches/first-pass-summary-20260517-204443.json`.
- Batch id: `batch_b233f1defa434225abc95acf46fac534`.
- Response id: `resp_801d7fae735e4a348460029d8ca95ef0`.
- Runtime profile: `wsl-chrome-3`.
- The run remained `running` with repeated `chatgpt-passive-dom-probe` and
  `browser-runtime-hint` lease heartbeats.
- The active lease target id was `2DD81FEB230FEF239857872E722DEB56`.
- CDP `/json/list` showed that target on `https://chatgpt.com/library`, and no
  page target matched conversation `6a0a6f14-7a80-83ea-a77b-81f654b709aa`.
- AuraCall runtime inspection with `diagnostics=browser-state` confirmed target
  `ChatGPT - Library`, `document.url=https://chatgpt.com/library`, and
  `visibleCounts.modelResponses=0`.
- The run was cancelled cleanly through `POST /status` with
  `runControl.cancel-run`.

## Required AuraCall Fix

- A running ChatGPT browser response must own exactly one live conversation
  target for that running prompt.
- Passive DOM lease evidence must be tied to the actual target currently being
  probed. It must not renew a prompt lease from stored tab metadata when the CDP
  target has navigated to Library, project root, or another non-conversation
  page.
- If the expected conversation target is gone or has navigated away, the run
  should fail or escalate with a concrete target-mismatch diagnostic instead of
  renewing indefinitely.
- Completed or failed transcript conversation tabs should be cleaned up or
  excluded from active prompt ownership so stale tabs do not confuse runtime
  profile state.

## Transcribe Gate

Do not submit another private first-pass summary batch until AuraCall has a
non-private smoke that proves:

- active lease target URL matches the running conversation URL,
- the expected conversation id appears in the live CDP target list,
- Library/project/root pages do not count as running-prompt evidence, and
- cancellation/recovery leaves zero active, reclaimable, recoverable-stranded,
  and stranded runs.
