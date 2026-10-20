---
id: policy-context-pilot
title: Policy Context MCP
summary: Load current canonical policy through checked MCP profiles with direct Markdown fallback.
category: governance
status: standard
tags: [policy, mcp, context]
---

## Policy

- Use `gov_policy` as the standard policy-loading path when the repository has
  a reviewed manifest and a working MCP registration. Otherwise read canonical
  Markdown directly and continue within the same authority.
- Keep Markdown authoritative. The manifest inventories policy-bearing files,
  including AGENTS, with source hashes. Each profile declares supported task
  kinds, exact source sections, scope, reread triggers, fallback paths, and a
  versioned packet with cited obligations, precedence and unresolved conflicts.
- On first use and after context loss, pass `include_source_text:true` and
  consume the selected sections. Saving unread text or reading only a compact
  packet does not establish policy knowledge.
- Reuse a consumed packet only while sources, inventory, configuration, scope
  and action facts remain current. Use `known_decision_digest` only with that
  exact packet retained; a freshness response cannot restore lost context.
- On unavailable, stale, invalid, conflicted or out-of-scope results, read the
  named canonical fallback and any additional task-triggered policies. Never
  treat a partial packet as complete governance.
- Keep caller-supplied authorization, requested effects, lineage, exclusions,
  rollback requirements and runtime facts attributable and separate from policy.
  The tool must not infer effect authority from static text.
- Review applicability before rebuilding hashes or changing profile selections.
  Include every task-triggered obligation; another repository's ordinals are
  not a portable profile. Source changes invalidate existing admission.
- Before first use or after registration changes, verify installed identity,
  root, includes, fresh-session discovery, a current hydrated profile and
  readable fallback. Check source drift, inventory drift and unsupported tasks
  fail closed. Preserve the existing validation receipts; do not repeat an
  experiment on every task.
- Maintain direct Markdown access. Standard operating status does not establish
  token savings or downstream adoption; measure those claims separately.

## Adoption Notes

The legacy module ID `policy-context-pilot` is retained for upgrade compatibility.
Adopt this standard wherever reviewed policy-context MCP wiring is configured;
ordinary MCP use alone does not establish a policy manifest or registration.
