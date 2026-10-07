## Graphiti Discovery And Memory

- At the start of non-trivial planning, debugging, architecture, audit,
  handoff, or prior-context work, decide whether Graphiti discovery could
  prevent repeated investigation or recover durable decisions and runtime
  history. Use the `graphiti-discovery` skill when it could materially help.
- Skip Graphiti for trivial or self-contained tasks when supplied content or
  current authoritative sources are sufficient. If Graphiti is unavailable,
  continue from repo-native evidence and mention the fallback only when it
  materially limits confidence.
- Keep discovery read-first, narrowly scoped, and advisory. Verify retrieved
  claims against current files, artifacts, commits, tests, or live state before
  acting on them.
- At every substantive closeout, record exactly one explicit memory disposition:
  `queued`, `duplicate_noop`, `not_durable`, `forbidden`, or `unavailable`.
  Give a short reason for every non-write; silence is not a disposition.
- Judge admission by durability and future retrieval value. A canonical source artifact is expected provenance
  for a source-anchored memory; its existence
  is not by itself a reason to select `not_durable`. Use `duplicate_noop` only
  after exact reconciliation proves the same memory effect already exists.
- Keep semantic qualification separate from effect authority. If an outcome is
  durable and qualified but this task explicitly prohibits a memory write,
  select `forbidden` and name that boundary rather than relabeling the outcome
  `not_durable`. Do not backfill it later without a new qualified and authorized
  closeout.
- For a qualified durable outcome, use `graphiti-runtime remember --disposition
  queued` with an explicit group, stable name, source description, curated
  source artifact, reference time, and compact body. Queue acceptance is enough
  for ordinary closeout, but is not persistence or retrieval proof.
- Use `graphiti-runtime remember` with `--disposition not_durable`,
  `--disposition forbidden`, or `--disposition unavailable` plus `--reason`
  to produce a machine-readable non-write receipt without attempting a Graphiti
  write.
- Before retrying an ambiguous write or recording `duplicate_noop`, use
  `graphiti-runtime remember-reconcile RECEIPT`. Preserve the reconciliation
  receipt and do not retry merely because extraction is slow.
- Keep policy compliance, queue acceptance, terminal processing, grouped
  visibility, and retrieval as separate evidence boundaries. A missing
  disposition is a policy-observability failure; a failed queued job is a
  delivery failure.
- Route qualified singleton writes through `graphiti-runtime remember`. An
  unreceipted memory job is an observability failure even if it completes; do
  not reconstruct synthetic provenance or infer missing closeout eligibility.
- Do not seed secrets, raw private data, full logs, or unreviewed speculation.
