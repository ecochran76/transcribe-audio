# Plan 0033 | GWS service PATH repair

State: OPEN

Lane: P09

Plan Version: 1

Execution Mode: `/goal`-compatible, bounded, checkpointed

Critical-Path Owner: primary agent

## Goal Contract

Objective:

> Make the restored GWS authorization usable by `transcripts.service`, then
> prove one served default immutable retrieval includes normalized GWS
> evidence without changing speaker, storage, or confirmation authority.

## Scope

- Install one non-secret user-systemd drop-in that gives
  `transcripts.service` an explicit executable PATH including
  `/home/ecochran76/.cargo/bin`.
- Validate the merged unit, daemon-reload, and restart the service once.
- Prove the restarted process inherited the intended PATH without exposing
  other process environment values.
- Execute one served default immutable retrieval on the already qualified
  non-frozen Plan 0032 target.
- Require at least one included evidence control from source profile
  `gws-default`.
- Document the service PATH requirement in README and close with one
  immutable terminal receipt.

## Non-Goals

- No credential rewrite, OAuth flow, provider configuration change, source
  code change, or target substitution.
- No model call, clue-generation pass, frozen-cohort prediction, gold
  review/read, or evidence-family scoring.
- No legacy rollback, speaker assignment, contact merge, CRM mutation,
  automatic confirmation, database-authority cutover, or provider write.

## Current State

Plan 0032 passed the general provider-yield gate with four included Odollo
snapshots. Its immutable request also proved that GWS authorization was not
the active failure: all four GWS capabilities returned
`provider_unavailable/gws executable unavailable`.

The installed interactive executable is `/home/ecochran76/.cargo/bin/gws`.
The user systemd manager PATH is
`/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:...` and omits
that directory. The merged service unit has no PATH override. Existing Codex
and Odollo drop-ins are independent and must be preserved.

## Authority And Bounds

Authority order:

1. this plan and its private receipt;
2. installed merged unit plus service/process readbacks;
3. Plan 0032 receipt and immutable GWS failure evidence;
4. current repo source, config, tests, and live API;
5. roadmap/runbook; Graphiti remains advisory.

Bounds:

- `max_systemd_dropins_created: 1`;
- `max_service_restarts: 1`;
- `max_default_retrieval_attempts: 1`;
- `max_source_scope_attempts: 1` within that request;
- `max_target_substitutions: 0`;
- `max_source_code_remediations: 0`;
- `max_model_calls: 0`;
- `max_frozen_cohorts_consumed: 0`;
- provider access remains read-only; no external write is authorized.

Rollback removes only
`~/.config/systemd/user/transcripts.service.d/30-gws-path.conf`, reloads the
user manager, and restarts the service. Do not alter either existing drop-in.

## Execution Packet

### P1 | Installed runtime repair and GWS-inclusive proof

Owner: primary agent

Write surface:

- one installed user-systemd drop-in;
- private Plan 0033 and product retrieval receipts;
- README, this plan, `ROADMAP.md`, and `RUNBOOK.md`.

Steps:

1. Revalidate the executable, merged service, target, frozen state, Git, and
   authority modes.
2. Install the explicit PATH drop-in and verify the unit before reload.
3. Reload and restart once; verify active state, PID change, zero restart-loop
   behavior, API health, and PATH membership.
4. Execute one served default immutable retrieval on document
   `158fe299a59444821675`.
5. Validate receipt hashes, permissions, GWS evidence controls, source
   failures, frozen state, and authority invariants.
6. Record one terminal `pass`, `refine`, or `stop` decision and push repo
   closeout.

Delegation:

- `not_spawned`: the service mutation and one dependent live proof are a
  serialized critical path owned by the primary agent.

## Acceptance Criteria

- The installed executable remains `/home/ecochran76/.cargo/bin/gws`.
- The merged service and restarted process PATH include
  `/home/ecochran76/.cargo/bin` while existing Codex and Odollo configuration
  remains loaded.
- `transcripts.service` is active, stable, and serves API health after one
  restart.
- The fixed target remains non-frozen with its six-term query plan.
- One new immutable served request includes at least one evidence control whose
  source profile is `gws-default`.
- GWS absence/failure is never hidden by Odollo yield or legacy fallback.
- The frozen cohort, gold, authority modes, automatic confirmation, and
  external-write state remain unchanged.
- Runtime and retrieval receipts are private and hash-verified.

## Terminal Decisions

- `pass`: the service PATH repair is active and one immutable bundle includes
  normalized GWS evidence with all safety checks intact.
- `refine`: the installed runtime remains safe but GWS evidence is still not
  included after the one authorized attempt.
- `stop`: scope, privacy, evidence integrity, frozen-cohort, gold,
  unexpected-write, or service-stability safety is violated.

## Validation

- `systemd-analyze --user verify` on the unit and drop-in.
- Merged unit, MainPID, `NRestarts`, process-PATH membership, and API health.
- Served immutable response and private receipt/shadow reconciliation.
- Receipt hashes/permissions, live knowledge-table count, and frozen-state
  hash/status.
- README contract, active planning audit, `git diff --check`, focused commit,
  push verification, and served-source verification.

## Definition Of Done

Plan 0033 is done when one immutable terminal receipt records `pass`, `refine`,
or `stop`; the installed service state and repo authorities agree; all bounds,
rollback, and authority states are explicit; and the closeout commit is
pushed.

