# Policy | Codegraph Usage

## Policy

- Use the local codegraph before non-trivial source-code edits, architecture
  claims, trace analysis, impact analysis, or refactor planning when the index
  is available.
- Prefer codegraph for structural questions: symbol definitions, callers,
  callees, impact radius, route/component ownership, and unfamiliar subsystem
  maps.
- Treat codegraph as a discovery and impact-analysis aid, not proof that a
  change is correct. Verify behavior with source reads, tests, builds, browser
  checks, or runtime smokes as appropriate.
- Check index freshness before relying on graph results. If the index is stale
  or unavailable, either refresh it or state the fallback to ordinary repo
  inspection in the handoff.
- Keep private runtime data, credentials, raw transcripts, tenant payloads, and
  unrelated logs out of codegraph inputs and persisted analysis artifacts.
- Keep `.codegraph/` workstation-local. It is ignored through
  `.git/info/exclude` and must not be committed.

## Local Entrypoints

- CLI: `codegraph`
- Sibling checkout: `../codegraph`
- Repo-local index: `.codegraph/codegraph.db`

Useful checks:

```bash
codegraph status . --json
codegraph sync .
codegraph context -p . "<task or subsystem>"
codegraph query -p . "<symbol or phrase>" --json
codegraph callers -p . "<symbol>" --json
codegraph callees -p . "<symbol>" --json
codegraph impact -p . "<symbol>" --json
```

If `codegraph status . --json` reports pending changes, prefer
`codegraph sync .` before relying on graph structure. After active edits, direct
source reads and validation remain authoritative until the index refreshes.

## Source

Adopts the shared `codegraph-usage` module with repo-local entrypoints for the
installed CodeGraph CLI and the local `.codegraph/` index.
- When a repo has an available codegraph or indexed code-intelligence service, consult it before making non-trivial code changes, architecture claims, trace analysis, or refactor plans.
- Prefer codegraph context, trace, callers, callees, impact, or file-index tools for structural questions such as:
  - where a symbol is defined
  - what calls or depends on a function, class, route, or component
  - how one behavior flows into another
  - what a refactor is likely to affect
  - which files make up an unfamiliar subsystem
- Use the repo's documented codegraph entrypoint when one exists, such as a sibling `../codegraph` checkout, local MCP tools, CLI wrapper, or indexed workspace service.
- Resolve the intended repository or worktree root and inspect current index status before relying on graph results. A sibling checkout's index is not proof that a fresh worktree or different branch is indexed.
- When a repo has already adopted, configured, or explicitly declared codegraph as an expected development surface, treat a missing index in a verified local worktree as routine derived-state maintenance: run the documented initialization workflow and verify the resulting index status. Do not require a fresh approval solely because the worktree is new.
- Do not assume automatic refresh applies to every project. The active checkout may have a live watcher while secondary projects, explicit-path queries, and fresh worktrees require explicit synchronization.
- When status or a staleness banner reports pending files, disabled auto-sync, an unwatched project, or a stale index, run the documented explicit sync once and re-check status. Do not wait repeatedly on a watcher that is absent or disabled.
- Treat the codegraph as a discovery and impact-analysis aid, not as proof that a change is correct. Verify behavior with source reads, targeted tests, type checks, linters, browser checks, or runtime smoke as appropriate.
- Prefer codegraph lookups over broad manual grep loops for symbol, flow, caller/callee, and architecture questions. Use text search or direct file reads to confirm details the index does not cover.
- After editing code, inspect the reported staleness or pending-sync state instead of guessing a delay. Use direct reads for specifically flagged files until synchronization is confirmed.
- Keep secrets, credentials, private logs, and unrelated runtime data out of indexed codegraph inputs or persisted analysis artifacts.
- Before initialization, confirm the target root and repo-local exclusions. Stop and ask when codegraph has not been established for the repo, the target or allowed input scope is ambiguous, repo policy reserves indexing for an operator, or initialization would create unexpected tracked-file changes.
- If initialization or one explicit sync still leaves codegraph unavailable or stale, proceed with ordinary repo inspection and report the exact failed status or staleness evidence in the handoff when it affects confidence.

## Adoption Notes

Use this module when a repo contains code that agents edit, review, trace, or refactor and an indexed codegraph is available or expected in the working environment.

Keep exact commands, MCP tool names, sibling checkout paths, service repair, and project-specific index exclusions repo-local. The reusable contract is initialize an expected missing index, explicitly sync when automatic refresh is absent, and verify current status.
