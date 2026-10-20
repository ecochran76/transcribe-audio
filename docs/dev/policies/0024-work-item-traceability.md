# Policy | Work Item Traceability

## Policy

- Drive substantive engineering work from a ticket in the configured tracker
  or governed repo-local backlog. Capture a user request there before starting;
  use a local ticket when provider creation is unavailable or unauthorized.
  Trivial one-step edits need no standalone ticket.
- Give the ticket a stable locator, problem and desired outcome, owner, priority,
  state, dependencies, scope and acceptance evidence. Refine only enough to make
  the next implementation decision; resolve material ambiguity before coding.
- Use `TRIAGE`, `READY`, `IN_PROGRESS`, `BLOCKED`, `DONE`, and `CANCELLED`, or an
  explicit local mapping. Keep ideas in triage until the outcome is testable.
- Split large tickets into small vertical slices that deliver observable
  behavior through the product. Each child has its own acceptance signal and
  dependencies. Avoid horizontal tickets whose sole deliverable is fixtures,
  mocks, scaffolding, or speculative infrastructure. Supporting work must serve
  a named product criterion or evidenced blocker in the same bounded outcome.
- Keep one source for each fact: the ticket owns outcome and acceptance; a
  linked plan adds execution detail only when needed; the lane catalog records
  concurrent custody; Git, validation and runtime evidence prove delivery.
  Reference these sources rather than duplicating their prose.
- Set an explicit substantive WIP limit. Finish or unblock current work before
  starting more; do not raise the limit to conceal contention.
- Preserve the ticket locator in plans, delivery changes and handoffs. Record
  blockers explicitly, and triage stale or duplicate items on a local cadence.
- Close from acceptance evidence for the intended delivery target. Distinguish
  implemented, validated, integrated and released; a label or closed issue is
  not proof. Use automatic closure only when integration satisfies the ticket.
  Reopen or link corrective work when regressions invalidate completion.
- Ticket creation does not authorize provider mutations, deployment or broader
  scope. Continue authorized local work without making tracker access a gate.

## Adoption Notes

Use in software engineering repositories, including small libraries and skill
repos. A repo-local ticket may share a plan file when its header provides a
stable work-item ID and its body holds the ticket fields above. Do not require
an external tracker or duplicate ticket and plan acceptance text.

With active-lane coordination, link tickets from substantive lanes and migrate
existing entries before enabling required work-item validation.
