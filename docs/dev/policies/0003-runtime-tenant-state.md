# Policy | Runtime And Tenant State

## Policy

- Keep reusable product code, schemas, examples, and redacted templates in the repo.
- Keep live operator data, tenant-specific state, credentials, raw transcripts that contain private content, caches, and deposition logs outside the repo unless they are explicitly redacted fixtures.
- Treat one runtime profile as one operator or tenant environment.
- Bind runtime state to explicit selectors such as tenant id, profile name, account email, calendar provider, Drive root, Odoo profile, or Graphiti group.
- Refuse silent reuse of one tenant's local state against another tenant or environment.
- Prefer structured JSON/YAML runtime records for durable state that must survive across service restarts.
- Keep secrets in environment variables, ignored local config, or user-scoped runtime homes; never in committed policy or plan files.
- Record private tenant facts in tenant-scoped runtime memory or operator notes, not public repo docs.

## Local Runtime Targets

- `watch_transcriptions.json` may define product-safe watcher behavior but must not embed secrets.
- `api_keys.json`, Google OAuth tokens, and service credentials are ignored runtime secrets.
- Future routing/deposition state should use a runtime home such as `~/.local/state/transcribe-audio/` or a configured equivalent, not `.openclaw/` inside the repo unless the content is non-sensitive and intentionally tracked.
