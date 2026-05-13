# Plan 0005 | Deposition And Memory Harvest

State: OPEN

Lane: P05

## Scope

Store transcripts/readouts in the selected repository and harvest reviewed memory candidates.

## Non-Goals

- No raw transcript memory writes.
- No tenant-crossing deposition.

## Current State

Outputs stay in Downloads unless an operator applies a separate deposition workflow. `deposition_preview.py` defines a no-write preview contract for local filesystem, Google Drive, Odoo, and Graphiti/OpenClaw memory-harvest intent. Preview actions are auditable and explicitly record `writes_enabled=false`. A live preview over the context-packet-generated SoyLei/Tempo contextual readout produced one local filesystem action and six Graphiti memory-harvest candidates without enabling writes. `deposition_apply.py` can apply only local filesystem actions from a preview; non-local actions are skipped. `memory_harvest_apply.py` now provides a reviewed Graphiti memory-harvest apply contract: dry-run by default, refusal on review-required or warning-bearing previews unless explicitly allowed after review, `--init-review` templates for per-candidate approve/reject/pending decisions, and live writes only with `--apply --approval-token APPROVE_GRAPHITI_MEMORY_HARVEST`. Review files limit live writes to approved candidates, while rejected/pending candidates and duplicate-check outcomes remain in the audit JSON. Memory harvest candidates come only from structured readout `memory_candidates`, not raw transcript text. Route-level provenance filtering keeps weak sources out of contextual rereads before deposition preview, and preview JSON carries contextual warnings for reviewers. There is still no Drive/Odoo apply path and no Odoo write target model contract.

## Work Items

- Done: define preview `DepositAction`, `MemoryHarvestCandidate`, and `DepositPreview` schemas.
- Done: implement no-write deposition/memory-harvest preview CLI over contextual readouts.
- Done: implement local filesystem depositor apply path over reviewed previews.
- Done: run deposition preview over a context-packet-generated contextual readout.
- Done: add upstream provenance-source quality filtering before contextual reread and deposition preview.
- Done: add explicit Graphiti memory-harvest approval/apply gates.
- Done: add per-candidate memory-harvest review files plus duplicate-check audit status.
- Implement Google Drive depositor through selected provider.
- Define Odoo depositor target model before implementing writes.
- Continue calibrating live Graphiti memory-harvest operations with reviewed candidate batches.
- Add dry-run mode for all non-local depositors.

## Acceptance Criteria

- Depositions are idempotent or explicitly versioned.
- Every external write records target id, profile, and evidence.
- Memory harvest excludes secrets and raw transcript content.
- Graphiti memory writes apply only approved candidates when a review file is supplied.

## Validation

- Dry-run deposition tests.
- Manual local depositor smoke.
- Preview smoke over a real contextual readout.
- Preview smoke over a context-packet-generated contextual readout.
- Local apply smoke over a real preview.
