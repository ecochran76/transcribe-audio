# Plan 0004 | Matter Routing And Contextual Reread

State: OPEN

Lane: P04

## Scope

Identify the likely matter/context repository for a transcript and rerun analysis with supporting context.

## Non-Goals

- No unattended external writes until route confidence and review behavior are implemented.
- No Odoo write implementation before target model selection.

## Current State

The first deterministic routing slice is implemented. `routing_artifacts.py` defines `ContextProvenancePack`, `RouteCandidate`, `RouteDecision`, and `ReviewQueueItem`. `route_transcript.py` reads existing transcript/readout artifacts and emits a dry-run `*.route.json`; low-confidence selections write a local review queue item. Current provenance is extracted from transcript calendar metadata, including `event.matching_calendars`, and route candidates come from structured readout `matter_candidates`. `context_sources.py` adds explicit read-only provenance adapters for live `gws` Calendar/Drive metadata, Graphiti/OpenClaw discovery, and Odollo/Odoo contacts and log notes. Non-calendar provenance is quality-filtered with source-type-specific profiles for Drive file identity, Odollo contact identity, Odollo note subjects, and Graphiti labels/previews before it can support selected routes or contextual rereads; weak sources are retained under `provenance_pack.excluded_sources` with warnings and quality metadata. `contextual_reread.py` generates upgraded `*.contextual.readout.*` artifacts from transcript, prior readout, route decision, and cited supporting provenance sources, and carries route warnings into contextualization metadata. `scripts/context_packet_apply.py` has completed a real reviewed apply over a known SoyLei/Tempo transcript/readout pair using read-only provenance, yielding a selected route, contextual readout, and sanitized run manifest. Graphiti facts and episodes are evidence-only; Graphiti nodes can add low-confidence advisory route candidates. Odollo sources are evidence-only until Odoo target models and write contracts are explicitly selected. Deeper Drive/Docs content fetch and source-specific scoring calibration remain future work.

## Work Items

- Done: define `RouteDecision` schema.
- Done: implement local review queue for low-confidence matches.
- Done: add dry-run CLI over existing transcript/readout artifacts.
- Done: add read-only `gws` provenance adapter for Calendar and Drive metadata.
- Done: add read-only Graphiti/OpenClaw advisory provenance and node-based candidate hints.
- Done: add read-only Odollo/Odoo provenance adapter for contacts and log notes across selectable profiles.
- Done: generate upgraded contextual readouts with cited route provenance sources.
- Done: run a reviewed context-packet apply smoke over a real stored transcript/readout pair.
- Add candidate source adapter for local index.
- Done: add provenance-source quality filtering and route/contextual warning propagation.
- Done: add source-type-specific quality scoring beyond generic compact term overlap.
- Calibrate source-specific thresholds against more known-good meetings.
- Define Google Drive/Docs and Odoo candidate/depositor contracts.
- Fetch deeper Google Drive/Docs content for selected route sources.

## Acceptance Criteria

- Route decisions are auditable and stored as JSON.
- Low-confidence decisions are queued, not deposited.
- Contextual reread records exactly which sources were used.

## Validation

- Fixture-based tests for route scoring.
- Manual route dry-run against known transcript/readout examples.
- Real context-packet apply smoke over a known transcript/readout pair.
- Live route smoke showing weak Graphiti/Odollo sources excluded from contextual support.
