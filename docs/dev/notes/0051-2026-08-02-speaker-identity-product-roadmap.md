# Speaker Identity Product Roadmap

Date: 2026-08-02

## Vision

Speaker identity is a conversation-level inference, not a standalone voice
match. The product should combine independently visible acoustic similarity,
calendar candidates, contacts, transcript clues, relationships, prior accepted
conversations, and conflicts. It should accept only policy-qualified cases,
route ambiguous cases to review, and preserve unresolved alternatives.

Each reviewed conversation should improve future work without silently turning
contextual guesses into biometric truth. A confirmed speaker may contribute
private candidate voice clips. Those clips form a provisional, clue-only
profile until they satisfy multi-conversation quality and lifecycle rules.
Rejections and corrections must prevent profile contamination.

## Current maturity

| Capability | Current | Target |
| --- | --- | --- |
| Contextual speaker proposals | Level 2, shadow/reviewed | Level 4, dependable |
| Acoustic speaker evidence | Level 1, built but not successfully evaluated | Level 3, operational with safe fallback |
| Reviewed profile learning | Level 0, absent as a product loop | Level 3, operational |
| Combined automatic identity | Level 0, not authorized | Level 3, bounded operational |
| Knowledge feedback | Level 1, storage foundations exist | Level 4, dependable |

## Planned progression

1. Qualify Generation-4 media before cohort or gold freeze.
2. Execute a valid unseen acoustic evaluation and calibrate accept/reject/abstain.
3. Add acoustic similarity as a separate, cited factor in the existing Plan
   0025 speaker-clue workflow.
4. Convert human confirmations into quarantined provisional enrollment
   evidence with multi-session promotion, correction, withdrawal, and deletion.
5. Shadow-evaluate combined voice and context chronologically, using only
   evidence available at each conversation time.
6. Enable limited automatic identity only for policies that meet a frozen
   high-confidence error bound; preserve review and abstention elsewhere.
7. Project accepted assignments into the private conversation knowledge store
   so later conversations gain useful, source-backed context.

## Evidence of progress

- Media qualification yield and explicit rejection reasons.
- Acoustic candidate recall, assignment correctness, false-match rate,
  abstention rate, and calibration by condition.
- Combined-evidence accuracy, high-confidence error rate, review rate, and
  conflict handling.
- Provisional-profile promotion, correction, contamination, withdrawal, and
  deletion outcomes.
- End-to-end share of eligible conversations receiving correct speaker-aware
  contextual readouts with complete provenance.

## Non-negotiable boundaries

- No automatic enrollment from context or model output alone.
- No hidden fusion score: acoustic and contextual evidence remain separately
  inspectable.
- No raw audio, transcripts, embeddings, or private provider bodies in broad
  memory surfaces.
- No reuse of evaluation media as training evidence.
- No retroactive historical evidence leakage.
- No promotion from passing tests or isolated examples; representative
  measured outcomes control maturity claims.

## Immediate critical path

Plan 0051 qualifies and freezes a fresh Generation-4 media pool. Profile-loop
contract design may proceed conceptually in parallel, but acoustic clues remain
non-authoritative until a valid unseen evaluation passes.
