"""Frozen contextual-visibility contract for Plan 0052 lane G1C."""

from __future__ import annotations

import hashlib
import json
from typing import Any

import speaker_identity_preprocess as plan0025


SCHEMA_VERSION = "transcribe-audio.generation4-context-contract.v1"
G0_PREVIEW_SHA256 = (
    "aa179741e735247e87cc6143c6526669670734c8c562ed166160eb0c6d605010"
)
G0_MANIFEST_SHA256 = (
    "ad9e26b59502508c8810e11648d519d99860579aea1ca731445459b196836d22"
)

PROMPT_CONTRACT = {
    "task": "paired_shadow_speaker_identity",
    "instructions": [
        "Use only prepared evidence and candidates.",
        "Cite every supporting or conflicting factor.",
        "Preserve unresolved, unlisted, split, mixed, and conflicting outcomes.",
        "Treat missing acoustic evidence as neutral.",
        "Never infer that an acoustic candidate is correct from presence alone.",
        "Return proposals only; do not apply assignments or mutate sources.",
    ],
}

RUBRIC_CONTRACT = {
    "status_vocabulary": [
        "candidate_match", "unlisted", "unresolved", "conflicting"
    ],
    "factor_directions": ["supporting", "conflicting", "neutral"],
    "factor_strengths": ["weak", "moderate", "strong"],
    "review_flags_preserved": True,
    "ready_to_confirm_requires_no_review_flags": True,
    "confidence_is_evidence_strength_not_probability": True,
    "acoustic_factor_scored_separately": True,
}


def _canonical_hash(value: Any) -> str:
    body = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


def build_generation4_context_contract() -> dict[str, Any]:
    """Return the path-free, deterministic G1C contract submitted to J1."""
    action_vector = {
        "submit_g1c_to_j1": True,
        "send_model_turn": False,
        "freeze_g2_envelope": False,
        "reveal_gold": False,
        "apply_assignments": False,
        "mutate_contacts": False,
        "mutate_profiles_or_references": False,
        "enable_default_integration": False,
        "run_historical_reprocessing": False,
    }
    core = {
        "schema_version": SCHEMA_VERSION,
        "status": "g1c_context_contract_complete",
        "plan_id": "0052",
        "plan_version": 1,
        "g0_authority": {
            "preview_content_sha256": G0_PREVIEW_SHA256,
            "manifest_sha256": G0_MANIFEST_SHA256,
        },
        "plan0025_contract": {
            "clue_discovery_packet_schema": plan0025.CLUE_DISCOVERY_PACKET_SCHEMA_VERSION,
            "clue_discovery_readout_schema": plan0025.CLUE_DISCOVERY_READOUT_SCHEMA_VERSION,
            "identity_evaluation_packet_schema": plan0025.IDENTITY_EVALUATION_PACKET_SCHEMA_VERSION,
            "identity_evaluation_readout_schema": plan0025.IDENTITY_EVALUATION_READOUT_SCHEMA_VERSION,
            "workflow": "clue_discovery_then_host_retrieval_then_identity_evaluation",
            "provider_calls_owned_by": "host",
            "requires_human_confirmation": True,
        },
        "prediction_families": [
            "context_only",
            "context_plus_separately_visible_acoustic",
        ],
        "acoustic_evidence_policy": {
            "visibility": "separate_cited_factor",
            "missing_evidence_effect": "neutral",
            "may_remove_context_candidate": False,
            "may_hide_conflict": False,
            "opaque_fusion_score_allowed": False,
        },
        "temporal_policy": {
            "as_of_field": "recording_start",
            "recording_start_required": True,
            "current_conversation_transcript": "allowed_as_current_evidence",
            "prior_conversation_requirement": "conversation_end_not_after_as_of",
            "provider_evidence_requirement": "source_event_time_not_after_as_of",
            "timeless_contact_records": "candidate_generation_only",
            "unknown_timestamp_evidence": "excluded_from_corroboration",
            "post_as_of_evidence": "excluded",
            "retrieval_time_recorded": True,
        },
        "candidate_policy": {
            "context_only_pool": "context_candidates",
            "augmented_pool": "stable_union_context_first_then_acoustic",
            "deduplication_key": "prepared_opaque_person_id",
            "preserve_source_affinity": True,
            "acoustic_may_remove_context_candidate": False,
            "allow_unlisted_suggestion": True,
            "measurements": [
                "context_candidate_recall",
                "union_candidate_recall",
                "assignment_correctness",
            ],
        },
        "prompt_contract": PROMPT_CONTRACT,
        "rubric_contract": RUBRIC_CONTRACT,
        "prompt_sha256": _canonical_hash(PROMPT_CONTRACT),
        "rubric_sha256": _canonical_hash(RUBRIC_CONTRACT),
        "comparison_policy": {
            "same_case_membership": True,
            "same_prompt_hash": True,
            "same_rubric_hash": True,
            "same_temporal_cutoff": True,
            "family_difference": "acoustic_factor_and_union_candidates_only",
            "predictions_before_gold_reveal": True,
            "post_prediction_regeneration_allowed": False,
        },
        "output_schema": {
            "allowed_statuses": [
                "candidate_match", "unlisted", "unresolved", "conflicting"
            ],
            "required_assignment_fields": [
                "speaker_labels", "status", "person_id", "factors",
                "credible_alternatives", "review_flags"
            ],
            "required_acoustic_fields_when_present": [
                "factor_id", "candidate_id", "direction", "strength",
                "evidence_receipt_sha256"
            ],
            "confidence_computed_by": "host_under_frozen_rubric",
            "will_apply_assignments": False,
        },
        "action_vector": action_vector,
        "delegation_receipt": {
            "status": "not_spawned",
            "lane": "G1C",
            "reason": "primary_owned_lane_preserves_three_agent_concurrency_cap",
            "runtime_handle": None,
        },
        "contains_paths": False,
        "contains_private_membership": False,
        "contains_raw_audio": False,
        "contains_transcript_text": False,
        "contains_embeddings_or_vectors": False,
        "contains_biometric_scores": False,
        "did_read_private_gold": False,
        "did_send_model_turn": False,
        "will_perform_external_write": False,
    }
    digest = _canonical_hash(core)
    return {**core, "content_sha256": digest}
