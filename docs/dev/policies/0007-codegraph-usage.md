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
