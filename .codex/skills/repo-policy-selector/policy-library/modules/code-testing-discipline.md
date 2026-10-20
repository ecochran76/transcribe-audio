---
id: code-testing-discipline
title: Code Testing Discipline
summary: Protect important behavior with fast, trustworthy regression tests while governing suite latency, compute cost, duplication, and flakiness.
tags:
  - testing
  - regression
  - ci
  - performance
  - reliability
---

## Policy

- Treat tests as maintained product assets with both protective value and lifecycle cost. Test count, assertion count, and raw coverage percentage are not success metrics by themselves.
- Name the invariant or failure risk before adding a test. Place it at the cheapest layer that can prove it reliably: prefer a focused unit or contract test, use a narrow integration test for boundary behavior, and reserve end-to-end, live, soak, and exhaustive tests for risks that cheaper layers cannot establish.
- Before adding a regression test, inspect existing coverage for the invariant. Demonstrate that the new or changed test detects the defect before the fix when practical, then passes after the fix. Consolidate overlapping cases instead of accumulating historical duplicates.
- Put a regression test at a stable seam that exercises the real failure
  pattern. If no such seam exists, do not add a shallow or implementation-
  coupled proxy merely to claim coverage; record the unprotected risk and the
  architecture or testability gap, then route remediation as a separate bounded
  decision.
- Keep each test independent, deterministic, order-agnostic, and hermetic by default. Declare inputs, isolate writable state, use explicit readiness signals instead of arbitrary sleeps, and keep network, provider, browser, large-data, and live-system tests out of the default local lane unless their exact risk requires them.
- Define repo-local execution tiers and concrete wall-clock plus compute/resource budgets. At minimum distinguish focused development checks, blocking presubmit checks, periodic comprehensive regression, and opt-in live/soak/provider checks. A long comprehensive lane may remain valuable without blocking every change.
- Use affected-test selection or explicit changed-surface manifests for fast feedback only when the dependency mapping is trustworthy. Unknown impact must widen to a documented safe fallback, and a periodic comprehensive run must detect selection drift. Never describe a selected subset as the full suite.
- Measure suite economics over time: selection size, collection/startup cost, p50 and p95 wall time, total compute, peak constrained resources where material, slowest tests, flake rate, retry rate, and failure yield. Optimize repeated setup and collection costs before merely adding workers.
- Parallelize or shard only after tests are isolated and reproducible. Balance shards by observed duration when practical, retain exact shard identity in resumable receipts, and lower concurrency when contention increases failures or total resource cost.
- Treat retries as diagnostic or infrastructure-recovery evidence, not as erasure. Preserve the first failure, classify a pass-on-retry as flaky, and do not report the lane clean until policy-defined flake disposition is satisfied. Reconcile uncertain external effects before retrying any test that can mutate shared or live state.
- Quarantine a flaky test only with an owner, reason, issue or locator, quarantine date, expiry or service-level target, and replacement blocking coverage when the risk requires it. Repair, redesign, or remove quarantined tests promptly; quarantine is not permanent storage.
- Review expensive, redundant, obsolete, and low-yield tests on a recurring cadence. Every retained expensive test should protect a distinct named risk. Consolidation or deletion requires a retained-risk mapping and validation that the surviving suite still proves the intended contract.
- Use coverage to locate consequential gaps, not to chase a universal percentage. Prefer behavior, branch-risk, contract, and selectively applied mutation evidence over copy-pasted tests that only increase coverage.
- When a suite exceeds its local budget, profile before changing the gate. Prefer cheaper seams, shared-fixture optimization without weakened isolation, case consolidation, tier correction, trustworthy selection, caching on declared inputs, or duration-aware sharding. Raising a budget requires an explicit risk/economics decision and a follow-up date.
- Record exactly which tier, selection, environment, retries, shards, and exclusions ran. Validation claims must distinguish `focused`, `presubmit`, `comprehensive`, and `live_or_soak`, and must report any budget breach, flake, quarantine, or unexecuted risk.

## Implementation And Diagnosis Workflow

- Prefer the Matt Pocock `tdd` skill for behavior-changing implementation where
  a stable test seam exists. Name the observable behavior and seam, then work
  vertically: one failing test, the smallest implementation that passes, then
  the next behavior. Observe red before green; retain the exact commands and
  outcomes. Avoid batches of speculative tests followed by bulk implementation.
- Test behavior through the module's public interface with expectations from
  the specification, worked examples, or known-good fixtures. Avoid assertions
  that recompute the implementation, reach through private collaborators, or
  use side channels that miss the caller's actual behavior. Use the existing
  harness and consult `codebase-design` when the seam itself needs design.
- For hard bugs and performance regressions, prefer `diagnosing-bugs`: first
  build and run a tight feedback loop that detects the reported symptom, then
  minimize the reproducer before ranking falsifiable hypotheses. Test one
  prediction at a time; measure performance before changing it. Bound probes
  and stress to existing budgets and authority. If a trustworthy loop cannot
  be built, report the tried probes and exact missing evidence instead of
  treating an untested theory as diagnosis.
- Convert the minimal reproducer into a regression at a valid seam before the
  fix, then rerun the original scenario after the fix. Record a missing seam as
  an unprotected risk; preserve first failures and remove temporary diagnostic
  instrumentation before closeout. Keep structural refactoring in a separate
  bounded review/remediation step after green; do not expand the red/green
  cycle into unrelated cleanup. Trivial reversible edits need no new tests
  when they introduce no consequential behavior risk.

## Long Runs And Agent Suspension

- For long tests, CI, or builds that need no active intervention, prefer a durable background run and Codex Wake (or an equivalent verified resume mechanism). Once the run and wake are established, save a checkpoint and end the active turn instead of repeatedly polling, sleeping in the foreground, or ingesting full output merely to wait.
- Before yielding, record the exact command, repository/worktree and revision or dirty-state locator, tier and selection, environment, run/job identity, log and result paths, wake id and trigger, and next action. The job must survive the turn ending and preserve its exit status and full output outside model context. Use a unique completion artifact per run, written only after results are saved, or an exact CI run identity; a stale marker or vanished process is not passing evidence.
- Verify the actual resume target and persistent monitor/dispatcher readiness using the installed wake skill. Require monitor-backed scheduling for unattended wakes and inspect registration before yielding. Prefer a completion trigger; use a bounded timed status check when completion cannot be observed directly. If reliable delivery is unavailable, retain bounded foreground supervision or report the exact blocker; do not claim an unattended continuation is armed.
- On wake, first reconcile the checkpoint, current revision, exact run, and whether the task is already complete. Inspect terminal status, exit code, and a compact result summary, then read only relevant failure excerpts or artifact sections. Retain full logs for deeper diagnosis without automatically loading them into context. If the run is still active, re-arm a bounded wake rather than starting a duplicate job.
- When a useful wake type is missing, verify the gap against the installed skill and command capabilities, then suggest a Codex Wake feature with the concrete waiting workflow, desired trigger and completion evidence, exact run/target identity, and why supported triggers are insufficient or costly. Distinguish a missing feature from an unhealthy monitor or configuration gap. Continue authorized work with the best supported bounded fallback and state its limitation; keep the suggestion actionable in the handoff or closeout. Creating an external issue or implementing the feature follows the task's existing authority and scope.
- Suspension does not relax test selection, resource budgets, retry discipline, effect authority, or acceptance gates. Report the run as pending until its result is verified; a registered or delivered wake proves neither test completion nor a pass. Keep short feedback checks and jobs requiring active intervention in the foreground.

## Adoption Notes

Each adopting repo should define a local test-suite contract with concrete values for:

- `fast_feedback_target`
- `presubmit_blocking_budget`
- `presubmit_compute_budget`
- `comprehensive_lane_cadence`
- `unknown_impact_fallback`
- `flaky_test_disposition_sla`
- `retry_result_mode`

Keep exact commands, marker names, CI job names, hardware assumptions, provider gates, and risk-specific test inventories repo-local.
Document the local long-run launcher, durable artifact location, supported wake
transport, and checkpoint/resume procedure when asynchronous validation is used.
