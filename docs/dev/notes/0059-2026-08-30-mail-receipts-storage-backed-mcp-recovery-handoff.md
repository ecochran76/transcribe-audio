# Mail Receipts storage-backed MCP recovery handoff

Date: 2026-08-30

Purpose: Give a fresh agent a restart-safe, evidence-bounded packet for the
Mail Receipts repository defect that blocks the approved Plan 0073
relationship-and-role discovery campaign. This note describes the dependency;
it does not authorize a private corpus read, a provider call, or a live repair.

## Outcome first

The `transcribe-audio` side of Plan 0073 P5 is implementation-ready at commit
`f0cc3fc` on branch `plan-0037-campaign`. Its approval-bound executor, exact
25-conversation/57-query cohort, private artifact controls, and provider-free
offline replay pass the focused 112-test selection.

P5 cannot produce an approvable private preview because the installed Mail
Receipts `operator-lite` workbench cannot currently return the exact
source-profile, account, tenant, account-address, and corpus selectors. Static
MCP initialization, tool discovery, and the Mail Live descriptor work, while
storage-backed registry reads do not return within bounded diagnostics.

The next agent should repair or safely fail the storage-backed public read path
in a new, scoped Mail Receipts worktree. Do not add a private-storage bypass to
`transcribe-audio`, enumerate Mail Receipts files directly, or reinterpret the
running service and socket as proof that storage-backed tools are healthy.

## Authority order

Read these sources in order before changing either repository or installed
runtime:

1. `/home/ecochran76/workspace.local/mail-receipts/AGENTS.md`
2. relevant Mail Receipts policies under `docs/dev/policies/`, especially
   runtime/product, runtime-state, tenant-isolation, validation/handoff,
   worktree, branch/integration, planning, and testing policies
3. Mail Receipts `ROADMAP.md` and latest `RUNBOOK.md` entries
4. `/home/ecochran76/workspace.local/transcribe-audio/AGENTS.md`
5. `transcribe-audio` Plan 0073:
   `docs/dev/plans/0073-2026-08-30-deterministic-mail-evidence-for-relationship-role-discovery.md`
6. `transcribe-audio` `RUNBOOK.md`, Turns 409 and 410
7. this note
8. Graphiti groups selected by each repository, advisory only

Current source, tests, installed-runtime readback, and immutable receipts are
authoritative. This handoff is a routing surface, not proof that an older
runtime observation is still current.

## Repository and installed-runtime anchors

### Consumer repository

- Repository: `/home/ecochran76/workspace.local/transcribe-audio`
- Branch: `plan-0037-campaign`
- P5 executor checkpoint: `f0cc3fc` (`Prepare Plan 0073 private pilot execution`)
- State immediately before this note: clean; 20 commits ahead of
  `origin/plan-0037-campaign`; not pushed
- Plan 0073: `OPEN`, P5 `awaiting-gate`

### Mail Receipts source repository

- Repository: `/home/ecochran76/workspace.local/mail-receipts`
- Current branch: `plan/227-device-verification-link-claim`
- Current commit after the later upgrade:
  `2e0dac8cf6e26b9e1584c0aa0d76b411d70e387b`
- Upstream divergence at readback: `0 0`
- The worktree now contains untracked generated benchmark attachment
  temporaries under `fixtures/benchmarks/corpus/.tmp_attachments/`. Treat them
  as unrelated user/runtime work and do not remove or absorb them.
- This branch belongs to unrelated Plan 227 work. Do not implement the MCP
  repair on it. Reconcile current default-ref and active-lane policy, then
  create a dedicated branch/worktree for the repair.

### Installed runtime

Observed on 2026-08-30; refresh before relying on process IDs or timestamps:

- CLI: `/home/ecochran76/.local/bin/mail-receipts`
- Installed version: `0.1.14`
- Codex registration: `mail_receipts`, stdio, profile `operator-lite`
- Bound namespace and allow-list: `default`
- Backend socket:
  `/home/ecochran76/.local/share/mail-receipts/storage/.runtime/mail-receipts-mcp-backend.sock`
- Stdio working directory:
  `/home/ecochran76/.local/share/mail-receipts`
- Backend unit: `mail-receipts-mcp-backend.service`
- Backend executable:
  `/home/ecochran76/.local/share/mail-receipts/venv/bin/mail-receipts`
- Installed source checkout after the later upgrade:
  `/home/ecochran76/.local/share/mail-receipts/source-checkouts/plan231-auth-challenge-runtime`
- Installed source commit: `c949ff9dfea37ea16b0142bc559236bf3812c642`
- Latest readback: backend `active/running`, PID `11867`, started
  `2026-08-30 17:31:25 CDT`; the Unix socket exists
- All four live-mailbox schedulers and both worker services were
  `active/running` after the upgrade recheck. Refresh this census before and
  after any authorized runtime change.
- The configured HTTP child target `http://127.0.0.1:8000` had no listening
  TCP socket; a three-second `/v1/service/profile` probe timed out with HTTP
  code `000`.

Do not copy environment values, tenant identifiers, message data, provider
identifiers, recipient lists, or storage contents into tracked notes.

## Confirmed symptoms

The Plan 0073 diagnostic used only the installed public workbench boundary.
It did not read Mail Receipts storage directly.

1. The originally registered MCP surface returned `Transport closed`.
2. A fresh, manually launched installed stdio shim initialized successfully
   and listed 45 tools.
3. `get_mail_live_view_descriptor` returned successfully. This method is
   static in `src/unified_mail/api/service.py` and does not read registry state.
4. Storage-backed `corpus_registry` and `list_corpus_operations` calls each
   exceeded a 10-second client ceiling.
5. One deliberately bounded `corpus_registry` attempt also exceeded 120
   seconds without returning selectors.
6. Restarting only `mail-receipts-mcp-backend.service` refreshed an older
   backend process but did not change the storage-backed timeout behavior.
7. After the diagnostic, the backend was refreshed again. A new stdio session
   still initialized and returned the static descriptor, and no further
   storage-backed request was intentionally left in flight.
8. `journalctl --user -u mail-receipts-mcp-backend.service` contained no
   entries for the diagnostic interval, so the stalled operation currently
   leaves no useful service-level trace.

No provider call, mailbox mutation, corpus-operation execution, corpus content
read, message-body read, private runtime artifact, schema migration, or
deployment occurred.

## Post-handoff upgrade recheck

The operator later reported that Mail Receipts had been upgraded. A fresh
read-only recheck established:

1. The installed executable and backend were replaced/restarted at
   approximately `2026-08-30 17:31 CDT`.
2. The installed package still reports `0.1.14`; its direct-install metadata
   points to Plan 231 source commit `c949ff9d`, whose purpose is the installed
   email authentication-challenge runtime.
3. The Codex-hosted `mail_receipts` MCP tools in the already-running session
   return `Transport closed` even for the static descriptor.
4. A fresh standalone installed shim using the same `operator-lite`, namespace,
   authentication, and backend-socket arguments returned the static Mail Live
   descriptor successfully with no stderr.
5. The same fresh shim's `corpus_registry` `list_corpora` request for namespace
   `default` still exceeded a single 30-second ceiling with no response and no
   stderr.
6. Inspection of the installed `UnifiedMailMcpBackendClient._request` confirms
   it still calls `client.settimeout(None)` immediately after connection.

Conclusion: the upgrade is installed and its unrelated Plan 231 runtime is
active, but it did not include or prove the storage-backed MCP repair. A new
Codex session may be needed to reacquire the hosted MCP transport after an
upgrade, but session reconnection alone will not fix the independently
reproduced registry timeout.

## Source-level fault boundary

CodeGraph was current for the Mail Receipts checkout at the handoff readback.
The relevant flow is:

```text
operator-lite MCP shim
  -> UnifiedMailMcpServer._call_tool
  -> UnifiedMailMcpBackendClient.call_tool
  -> Unix socket shared backend
  -> UnifiedMailMcpAdapter.call_tool
  -> UnifiedMailService storage-backed method
```

Important current behavior:

- `src/unified_mail/mcp/backend.py`,
  `UnifiedMailMcpBackendClient._request`, applies
  `connect_timeout_seconds` only while connecting. Immediately after connect
  it calls `client.settimeout(None)` and then waits for the complete backend
  response. A stalled tool therefore has no backend-response deadline or
  cancellation path. This explains why an internal stall can wait until the
  outer host kills the stdio request and reports `Transport closed`; it does
  not by itself identify the internal stall's root cause.
- `corpus_registry` dispatches to
  `UnifiedMailService.run_corpus_registry_tool`. Its `list_corpora` action
  enters `_registry_metadata_store`, then `list_corpus_registry_records`.
- `list_corpus_operations` enters the same registry metadata store and registry
  enumeration, then performs additional per-corpus operational summary reads.
- Because even the lighter `corpus_registry` listing did not return, start at
  registry metadata-store construction/enumeration and its locking/backend
  behavior. Do not begin by optimizing the later corpus-operations summary
  loop.
- `get_mail_live_view_descriptor` succeeding proves MCP framing, profile
  loading, backend connection, and static dispatch. It does not prove metadata
  backend health.

The shared fault boundary is confirmed. Lock contention, metadata-backend
configuration, filesystem behavior, or another internal cause remains a
hypothesis until reproduced with bounded instrumentation.

## Secondary HTTP gap

The Receipts-side child configuration points to `http://127.0.0.1:8000`, but
there is no listener. Treat this as a separate operational gap until ownership
is verified. The Mail Receipts workbench contract says HTTP may be used only
through an already configured authenticated listener; it expressly forbids
starting, proxying, publishing, or reconfiguring a listener merely to complete
a read. Do not use HTTP startup as a shortcut around the MCP defect.

## Consumer impact

Plan 0073 needs these exact selectors before it can bind its immutable P5
preview:

- `source_profile_id`
- `account_id`
- `tenant_id`
- normalized `account_address`
- `corpus_id`
- namespace `default` (already confirmed)

Without them, `transcribe-audio` correctly rejects preview creation. The
executor must not accept placeholders, fixture values, inferred tenant state,
or selectors copied from private storage. Once the selectors are returned by
an authorized public surface, Plan 0073 will create the exact 25/57 preview and
stop for literal operator approval before its first owned-corpus read.

## Recommended bounded repair packet

Open one Mail Receipts plan/packet whose outcome is: storage-backed
`operator-lite` metadata reads either return a tenant-safe typed response
within a declared deadline or fail with a structured, redacted timeout while
leaving the stdio session and shared backend usable.

Suggested sequence:

1. Re-anchor the Mail Receipts default ref, active lanes, worktrees, latest
   roadmap/runbook authority, installed release identity, and current service
   census. Create a dedicated repair worktree; leave Plan 227 untouched.
2. Reproduce provider-free with a temporary metadata backend at the narrowest
   seam: direct service call, direct shared-backend client call, then stdio MCP.
3. Add bounded timing/phase instrumentation around metadata-store construction
   and `list_corpus_registry_records`. Logs must stay redacted and must not emit
   roots, corpus contents, tenant IDs, or raw backend errors.
4. Identify and repair the actual stall before adding a response deadline.
   A timeout alone would convert an infinite wait into a useful failure but
   would not restore selector retrieval.
5. Add a separate backend-response deadline/cancellation contract so one
   stalled request cannot wait forever or force the host to report an opaque
   transport closure. Preserve longer legitimate corpus-search budgets through
   explicit per-tool/request policy rather than one unbounded socket.
6. Confirm whether the current public descriptor set can safely return all five
   selectors needed by a bound consumer. If it cannot, design a narrow,
   tenant-authorized source-binding descriptor rather than exposing storage
   paths or overloading a mutation/admin tool.
7. Validate against fixtures first. Only after explicit runtime authority,
   install the repaired build and restart the exact backend unit. Do not
   restart schedulers/workers unless the repair actually requires it.
8. Return the exact public selector bundle to Plan 0073. Do not execute the
   private 25/57 pilot from the Mail Receipts repair lane.

## Acceptance evidence

The repair is not complete until current evidence proves all of the following:

- installed stdio `initialize` and `tools/list` succeed with stdin held open;
- `get_mail_live_view_descriptor` returns promptly;
- `corpus_registry` `list_corpora` for authorized namespace `default` returns
  a typed result or typed empty result within the declared budget;
- `list_corpus_operations` returns a typed result within its declared budget;
- an intentionally stalled synthetic backend returns a structured, redacted
  timeout without killing the stdio session;
- a subsequent static call and storage-backed call still succeed, proving the
  shim/backend remain usable;
- repeated requests do not leave unbounded worker threads or socket sessions;
- operator-lite still exposes no mailbox mutation or corpus-operation
  execution;
- namespace denial and cross-tenant tests remain fail-closed;
- no provider interaction occurs in fixture or installed readback validation;
- the exact Plan 0073 selector bundle is available through the authorized
  public surface without exposing private corpus contents or storage paths;
- the exact backend, four schedulers, and two workers have a fresh post-change
  service/process census; and
- focused tests, relevant presubmit selection, `git diff --check`, plan wiring,
  and installed-shim smoke all pass with exact commands recorded in the Mail
  Receipts Runbook.

Likely test homes include `tests/test_investigation_mcp.py` for the shared MCP
backend transport and focused service/registry tests for metadata enumeration.
CodeGraph currently skips oversized `src/unified_mail/api/service.py` and
`tests/test_api_service.py`; use exact reads for those files or raise the index
cap only if a bounded task truly requires full structural indexing.

## Safe startup commands

Run these read-only checks before diagnosis:

```bash
cd /home/ecochran76/workspace.local/mail-receipts
git status --short --branch
git rev-parse HEAD
git worktree list
codegraph status
codex mcp get mail_receipts
mail-receipts --version
systemctl --user show mail-receipts-mcp-backend.service \
  -p ActiveState -p SubState -p MainPID -p ExecMainStartTimestamp -p ExecStart
systemctl --user --no-pager --plain --type=service --state=running \
  | rg 'mail-receipts.*(backend|scheduler|worker)'
ss -ltn '( sport = :8000 )'
```

Use the installed `mail_receipts` MCP registration for public workbench reads.
Do not improvise a direct-storage reproducer. Any timed diagnostic must have an
outer process ceiling and an explicit cleanup/readback step.

## Hard stops

- Do not modify the current Plan 227 branch for this repair.
- Do not enumerate or read private Mail Receipts storage to recover selectors.
- Do not switch from `operator-lite` to a mutation-capable MCP profile.
- Do not start or publish an HTTP listener merely to complete the read.
- Do not call Gmail, Outlook, msgcli, or another provider.
- Do not mutate a mailbox or execute/retry a corpus operation.
- Do not put tenant selectors, account addresses, message metadata, logs, or
  secrets in git, Graphiti, test fixtures, or broad terminal output.
- Do not restart schedulers or workers without action-specific authority and a
  demonstrated repair dependency.
- Do not treat a green unit test or listening socket as installed-runtime
  acceptance.
- Do not run the Plan 0073 private pilot until its exact preview is shown and
  literally approved.

## Suggested skills

- `repo-policy-selector` — load the Mail Receipts planning, runtime, worktree,
  and validation policies before opening the repair packet.
- `graphiti-discovery` — query the Mail Receipts repo group or memory atlas for
  recent MCP/runtime decisions, then verify them against current sources.
- `mail-receipts-workbench` — preserve the public-surface and operator-lite
  boundary during reproduction and readback.
- `diagnosing-bugs` — isolate the blocking phase and distinguish root cause
  from the missing response deadline.
- `tdd` — prove the hang and stdio-survival contracts before implementing the
  repair.
- `codebase-design` — use only if the bounded diagnosis shows that response
  deadlines or source-binding descriptors need a new product seam.

## Handoff state

This is a `blocked` dependency handoff, not a completion claim. The Mail
Receipts repository and installed runtime were not modified while preparing
it. Graphiti was healthy but returned no current Plan 0073 incident recall;
the current repositories and runtime readbacks above remain authoritative.
