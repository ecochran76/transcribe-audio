"""Versioned, non-live Plan 0072 identity-learning contracts.

A0 freezes host-facing artifact names and safety policy before later packets
add persistence, provider adapters, background processing, or review effects.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any


CATALOG_VERSION = "transcribe-audio.identity-learning-contract-catalog.v1"

CONTRACT_VERSIONS = {
    "domain": "transcribe-audio.identity-learning-domain.v1",
    "correction": "transcribe-audio.identity-learning-correction.v1",
    "privacy": "transcribe-audio.identity-learning-privacy.v1",
    "threat_model": "transcribe-audio.identity-learning-threat-model.v1",
    "api": "transcribe-audio.identity-learning-api.v1",
    "adapter": "transcribe-audio.identity-learning-adapter.v1",
    "supervisor": "transcribe-audio.identity-learning-supervisor.v1",
}

ARTIFACT_KINDS = (
    "source_observation",
    "person",
    "external_identity",
    "source_record",
    "person_alias",
    "role_assertion",
    "relationship_assertion",
    "conversation_association_candidate",
    "participant_hypothesis",
    "speaker_identity_proposal",
    "speaker_review_decision",
    "voice_sample",
    "voice_profile_version",
    "correction_event",
    "transcript_correction_proposal",
    "normalized_transcript_generation",
    "terminology_entry",
    "processing_run",
    "identity_review_queue_item",
    "provider_adapter_request",
    "provider_adapter_result",
    "identity_review_submission",
    "effect_preview",
)

ARTIFACT_SCHEMAS = {
    kind: f"transcribe-audio.{kind.replace('_', '-')}.v1"
    for kind in ARTIFACT_KINDS
}

REQUIRED_FIELDS: dict[str, tuple[str, ...]] = {
    kind: ("schema_version",) for kind in ARTIFACT_KINDS
}
REQUIRED_FIELDS["identity_review_queue_item"] = (
    "schema_version",
    "queue_item_id",
    "conversation_id",
    "recording_id",
    "original_recording_filename",
    "source_artifact_sha256",
    "source_media_sha256",
    "processing_run_id",
    "model_versions",
    "rubric_versions",
    "profile_versions",
    "calendar_candidates",
    "participant_hypotheses",
    "speakers",
    "review_state",
    "decision_history",
    "effect_preview_ref",
    "projection_version",
    "created_at",
)
REQUIRED_FIELDS["provider_adapter_request"] = (
    "schema_version",
    "request_id",
    "processing_run_id",
    "conversation_id",
    "adapter_id",
    "capability",
    "source_scope",
    "as_of",
    "query",
    "budgets",
    "mode",
    "idempotency_key",
    "created_at",
)
REQUIRED_FIELDS["provider_adapter_result"] = (
    "schema_version",
    "result_id",
    "request_id",
    "processing_run_id",
    "source_scope",
    "status",
    "observations",
    "warnings",
    "failure",
    "retrieved_at",
    "provider_write_count",
    "consumed_budget",
)
REQUIRED_FIELDS["identity_review_submission"] = (
    "schema_version",
    "submission_id",
    "queue_item_id",
    "conversation_id",
    "proposal_id",
    "action",
    "expected_projection_version",
    "decision_payload",
    "comment",
    "idempotency_key",
    "reviewer",
    "decided_at",
)
REQUIRED_FIELDS["effect_preview"] = (
    "schema_version",
    "preview_id",
    "queue_item_id",
    "submission_id",
    "expected_projection_version",
    "effect_mode",
    "proposed_effects",
    "invalidations",
    "profile_rebuilds",
    "provider_write_count",
    "raw_deletion_count",
    "warnings",
    "created_at",
)
REQUIRED_FIELDS["voice_sample"] = (
    "schema_version",
    "sample_id",
    "recording_id",
    "conversation_id",
    "speaker_ref",
    "start_ms",
    "end_ms",
    "source_media_sha256",
    "sample_sha256",
    "quality",
    "preparation_lineage",
    "review_authority_id",
    "consent_authority",
    "person_id",
    "review_state",
    "exclusion_state",
    "private_audio_ref",
    "created_at",
)
REQUIRED_FIELDS["voice_profile_version"] = (
    "schema_version",
    "profile_version_id",
    "person_id",
    "profile_family",
    "predecessor_profile_version_id",
    "sample_allowlist",
    "evaluation_id",
    "status",
    "active_interval",
    "private_profile_ref",
    "created_at",
)
REQUIRED_FIELDS["transcript_correction_proposal"] = (
    "schema_version",
    "proposal_id",
    "conversation_id",
    "recording_id",
    "raw_transcript_sha256",
    "raw_span",
    "replacement_text",
    "correction_kind",
    "terminology_entry_id",
    "scope",
    "evidence_ids",
    "review_state",
    "correction_pass",
    "processing_version",
    "cascade_count",
    "created_at",
)
REQUIRED_FIELDS["processing_run"] = (
    "schema_version",
    "run_id",
    "conversation_id",
    "recording_id",
    "original_recording_filename",
    "source_artifact_sha256",
    "source_media_sha256",
    "operation_mode",
    "policy_version",
    "as_of",
    "capabilities",
    "budgets",
    "model_versions",
    "rubric_versions",
    "profile_versions",
    "stage",
    "state",
    "transcript_correction_passes",
    "identity_cascade_count",
    "provider_retry_count",
    "input_ids",
    "output_ids",
    "failures",
    "effect_counts",
    "created_at",
)
REQUIRED_FIELDS.update(
    {
        "source_observation": (
            "schema_version",
            "observation_id",
            "source_type",
            "source_scope",
            "observed_at",
            "retrieved_at",
            "as_of",
            "content_hash",
            "evidence_independence_group",
            "subject_refs",
            "payload_ref",
            "privacy_class",
            "created_at",
        ),
        "person": (
            "schema_version",
            "person_id",
            "resolution_state",
            "preferred_display_name",
            "created_at",
            "retired_at",
            "redirect_person_id",
            "version",
        ),
        "external_identity": (
            "schema_version",
            "external_identity_id",
            "source_record_id",
            "provider_kind",
            "account_id",
            "tenant_id",
            "identity_type",
            "identity_value_hash",
            "person_specific",
            "verified",
            "observed_at",
            "valid_from",
            "valid_to",
        ),
        "source_record": (
            "schema_version",
            "source_record_id",
            "person_id",
            "source_profile_id",
            "provider_kind",
            "account_id",
            "tenant_id",
            "record_type",
            "external_ref",
            "label",
            "source_observation_ids",
            "link_state",
            "link_reason",
            "content_hash",
            "observed_at",
        ),
        "person_alias": (
            "schema_version",
            "alias_id",
            "person_id",
            "alias",
            "scope",
            "valid_from",
            "valid_to",
            "source_observation_ids",
            "review_state",
            "supersedes_alias_id",
        ),
        "role_assertion": (
            "schema_version",
            "role_assertion_id",
            "subject_person_id",
            "role_type_id",
            "context_ref",
            "direction",
            "effective_from",
            "effective_to",
            "source_observation_ids",
            "review_state",
            "conflict_refs",
            "ontology_version",
        ),
        "relationship_assertion": (
            "schema_version",
            "relationship_assertion_id",
            "subject_ref",
            "object_ref",
            "relationship_type_id",
            "direction",
            "effective_from",
            "effective_to",
            "source_observation_ids",
            "review_state",
            "conflict_refs",
            "ontology_version",
        ),
        "conversation_association_candidate": (
            "schema_version",
            "candidate_id",
            "conversation_id",
            "recording_id",
            "event_source_record_id",
            "rank",
            "evidence_assessment_id",
            "evidence_strength_score",
            "evidence_strength_band",
            "positive_factors",
            "negative_factors",
            "alternatives",
            "provider_warnings",
            "rubric_version",
            "as_of",
            "status",
            "created_at",
        ),
        "participant_hypothesis": (
            "schema_version",
            "hypothesis_id",
            "conversation_id",
            "person_id",
            "source_record_id",
            "kind",
            "source_observation_ids",
            "evidence_assessment_id",
            "status",
            "created_at",
        ),
        "speaker_identity_proposal": (
            "schema_version",
            "proposal_id",
            "evaluation_id",
            "conversation_id",
            "speaker_ref",
            "utterance_refs",
            "proposed_person_id",
            "alternatives",
            "contextual_assessment_id",
            "acoustic_assessment_id",
            "combined_assessment_id",
            "abstention_state",
            "review_flags",
            "model_versions",
            "rubric_versions",
            "profile_versions",
            "created_at",
        ),
        "speaker_review_decision": (
            "schema_version",
            "decision_id",
            "proposal_id",
            "evaluation_id",
            "action",
            "payload",
            "reviewer",
            "method",
            "decided_at",
            "supersedes_decision_id",
            "comment",
        ),
        "correction_event": (
            "schema_version",
            "correction_event_id",
            "correction_type",
            "subject_refs",
            "decision_id",
            "prior_refs",
            "replacement_refs",
            "invalidations",
            "rebuild_requests",
            "effect_preview_id",
            "created_at",
        ),
        "normalized_transcript_generation": (
            "schema_version",
            "generation_id",
            "conversation_id",
            "recording_id",
            "raw_transcript_sha256",
            "predecessor_generation_id",
            "accepted_correction_ids",
            "normalized_transcript_sha256",
            "index_version",
            "correction_pass_count",
            "identity_cascade_count",
            "created_at",
        ),
        "terminology_entry": (
            "schema_version",
            "entry_id",
            "canonical_term",
            "expansion",
            "definition",
            "aliases",
            "asr_confusions",
            "pronunciation_hints",
            "scope",
            "source_observation_ids",
            "status",
            "version",
            "supersedes_entry_id",
            "created_at",
        ),
    }
)

REVIEW_ACTIONS = {
    "confirm",
    "choose_existing_person",
    "create_reviewed_provisional_person",
    "not_listed",
    "unresolved",
    "reject_event",
    "choose_event",
    "no_matching_event",
    "mixed_speaker",
    "group_labels",
    "split_label",
    "correct_source_record",
    "correct_role",
    "correct_relationship",
    "merge_people",
    "split_person",
    "supersede",
    "defer",
}

PRIVACY_CLASSES = {
    kind: "private_user_scoped" for kind in ARTIFACT_KINDS
}
PRIVACY_CLASSES.update(
    {
        "voice_sample": "restricted_biometric",
        "voice_profile_version": "restricted_biometric",
        "provider_adapter_request": "private_provider_request",
        "provider_adapter_result": "private_bounded_evidence",
        "identity_review_queue_item": "private_review_metadata",
        "identity_review_submission": "private_review_decision",
        "effect_preview": "private_review_metadata",
    }
)

PORTABLE_ARTIFACTS = {
    "identity_review_queue_item",
    "identity_review_submission",
    "effect_preview",
}

FORBIDDEN_PORTABLE_KEY_FAMILIES = (
    "raw_path",
    "filesystem_path",
    "provider_body",
    "raw_audio",
    "audio_bytes",
    "waveform",
    "embedding",
    "vector",
    "unrestricted_audio_url",
    "full_transcript",
)

EFFECT_POLICY = {
    "provider_writes": "prohibited",
    "live_store_migration": "prohibited_in_a0",
    "historical_processing": "prohibited_in_a0",
    "biometric_collection": "prohibited_in_a0",
    "dashboard_publication": "prohibited_in_a0",
}

SUPERVISOR_STAGES = (
    "bind_conversation",
    "pre_identity_correction",
    "calendar_candidate_generation",
    "participant_and_evidence_collection",
    "speaker_and_relationship_proposals",
    "post_identity_correction",
    "queue_projection",
    "complete",
)

SUPERVISOR_LIMITS = {
    "max_provider_retries": 1,
    "max_transcript_correction_passes": 2,
    "max_transcript_identity_cascades": 1,
    "max_model_reference_repairs_per_phase": 1,
    "expensive_enrichment_backlog_threshold": 500,
}

THREAT_CONTROLS = (
    "authenticated_private_route",
    "tenant_account_capability_scope",
    "bounded_provider_reads",
    "provider_write_prohibition",
    "raw_path_non_disclosure",
    "bounded_media_handles",
    "biometric_private_storage",
    "review_before_enrollment",
    "stale_write_rejection",
    "append_only_supersession",
    "source_disjoint_evaluation",
    "self_training_prohibition",
    "correction_cascade_limit",
    "deletion_invalidation_and_tombstone",
    "evidence_independence_groups",
    "hindsight_labeling",
)


class ContractError(ValueError):
    """Raised when an artifact violates the frozen A0 interface."""


def _forbidden_nested_keys(value: Any) -> set[str]:
    keys: set[str] = set()
    if isinstance(value, Mapping):
        for key, child in value.items():
            normalized = re.sub(r"[^a-z0-9]+", "_", str(key).lower())
            if any(family in normalized for family in FORBIDDEN_PORTABLE_KEY_FAMILIES):
                keys.add(str(key))
            keys.update(_forbidden_nested_keys(child))
    elif isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray)
    ):
        for child in value:
            keys.update(_forbidden_nested_keys(child))
    return keys


def _inline_biometric_keys(value: Any) -> set[str]:
    forbidden = {"embedding", "embedding_vector", "vector", "waveform", "audio_bytes"}
    keys: set[str] = set()
    if isinstance(value, Mapping):
        for key, child in value.items():
            normalized = re.sub(r"[^a-z0-9]+", "_", str(key).lower()).strip("_")
            if normalized in forbidden:
                keys.add(str(key))
            keys.update(_inline_biometric_keys(child))
    elif isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray)
    ):
        for child in value:
            keys.update(_inline_biometric_keys(child))
    return keys


def _require_sha256(payload: Mapping[str, Any], field_name: str) -> None:
    if not re.fullmatch(r"[a-f0-9]{64}", str(payload.get(field_name) or "")):
        raise ContractError(f"{field_name} must be a lowercase SHA-256.")


def _require_original_filename(payload: Mapping[str, Any]) -> None:
    filename = str(payload.get("original_recording_filename") or "")
    if (
        not filename
        or "/" in filename
        or "\\" in filename
        or filename.endswith((".transcript.json", ".processing.json"))
    ):
        raise ContractError(
            "original_recording_filename must be a filename, not a path or "
            "derived artifact."
        )


def _validate_source_scope(value: Any) -> None:
    required = {
        "provider_kind",
        "profile_id",
        "account_id",
        "tenant_id",
        "capabilities",
    }
    if not isinstance(value, Mapping) or not required.issubset(value):
        raise ContractError("Adapter source_scope is incomplete.")
    selector_fields = required - {"capabilities"}
    if not all(str(value.get(field) or "").strip() for field in selector_fields):
        raise ContractError("Adapter source_scope contains an empty selector.")
    capabilities = value.get("capabilities")
    if not isinstance(capabilities, list) or not capabilities:
        raise ContractError("Adapter source_scope requires capabilities.")


def contract_catalog() -> dict[str, Any]:
    """Return the stable, serializable A0 contract catalog."""
    return {
        "catalog_version": CATALOG_VERSION,
        "contract_versions": dict(CONTRACT_VERSIONS),
        "artifacts": {
            kind: {
                "schema_version": ARTIFACT_SCHEMAS[kind],
                "privacy_class": PRIVACY_CLASSES[kind],
                "required_fields": list(REQUIRED_FIELDS[kind]),
                "portable": kind in PORTABLE_ARTIFACTS,
            }
            for kind in ARTIFACT_KINDS
        },
        "effect_policy": dict(EFFECT_POLICY),
        "supervisor": {
            "stages": list(SUPERVISOR_STAGES),
            "limits": dict(SUPERVISOR_LIMITS),
            "operation_modes": [
                "contract_fixture",
                "shadow",
                "reviewed_learning",
                "policy_qualified_automation",
            ],
            "execution": "asynchronous_after_transcript_stabilization",
        },
        "threat_controls": list(THREAT_CONTROLS),
        "portable_forbidden_key_families": list(
            FORBIDDEN_PORTABLE_KEY_FAMILIES
        ),
    }


def validate_artifact(kind: str, payload: Mapping[str, Any]) -> dict[str, Any]:
    """Validate one artifact at the stable A0 host seam."""
    if kind not in ARTIFACT_SCHEMAS:
        raise ContractError(f"Unknown identity-learning artifact kind: {kind}.")
    if not isinstance(payload, Mapping):
        raise ContractError("Identity-learning artifact must be an object.")
    if payload.get("schema_version") != ARTIFACT_SCHEMAS[kind]:
        raise ContractError(
            f"{kind} schema_version must be {ARTIFACT_SCHEMAS[kind]}."
        )
    missing = [field for field in REQUIRED_FIELDS[kind] if field not in payload]
    if missing:
        raise ContractError(
            f"{kind} is missing required fields: {', '.join(missing)}."
        )
    if kind in PORTABLE_ARTIFACTS:
        forbidden = sorted(_forbidden_nested_keys(payload))
        if forbidden:
            raise ContractError(
                "Portable artifact contains forbidden private fields: "
                + ", ".join(forbidden)
                + "."
            )
    if kind == "identity_review_queue_item":
        _require_original_filename(payload)
        _require_sha256(payload, "source_artifact_sha256")
        _require_sha256(payload, "source_media_sha256")
    elif kind == "provider_adapter_request":
        if payload.get("mode") != "read_only":
            raise ContractError("Provider adapter mode must be read_only.")
        _validate_source_scope(payload.get("source_scope"))
        scope = payload["source_scope"]
        if payload.get("capability") not in scope["capabilities"]:
            raise ContractError("Requested capability is outside source scope.")
        budgets = payload.get("budgets")
        required_budgets = {
            "max_records",
            "max_characters",
            "max_calls",
            "max_latency_ms",
        }
        if not isinstance(budgets, Mapping) or not required_budgets.issubset(budgets):
            raise ContractError("Provider adapter budgets are incomplete.")
        if any(
            not isinstance(budgets[field], int) or budgets[field] < 1
            for field in required_budgets
        ):
            raise ContractError("Provider adapter budgets must be positive integers.")
    elif kind == "provider_adapter_result":
        _validate_source_scope(payload.get("source_scope"))
        if payload.get("status") not in {"complete", "partial", "unavailable"}:
            raise ContractError("Provider adapter result status is invalid.")
        if payload.get("provider_write_count") != 0:
            raise ContractError(
                "Provider adapter results must report zero provider writes."
            )
        if payload.get("status") != "complete" and not payload.get("failure"):
            raise ContractError(
                "Partial or unavailable adapter results require a failure."
            )
    elif kind == "identity_review_submission":
        if payload.get("action") not in REVIEW_ACTIONS:
            raise ContractError("Identity review action is invalid.")
        if not str(payload.get("expected_projection_version") or "").strip():
            raise ContractError(
                "Identity review requires a stale-write projection version."
            )
        if not str(payload.get("idempotency_key") or "").strip():
            raise ContractError("Identity review requires an idempotency key.")
        decision_payload = payload.get("decision_payload")
        if not isinstance(decision_payload, Mapping):
            raise ContractError("Identity review decision_payload must be an object.")
        if (
            payload.get("action") == "choose_existing_person"
            and not decision_payload.get("person_id")
        ):
            raise ContractError("Choosing an existing person requires person_id.")
    elif kind == "effect_preview":
        if payload.get("effect_mode") != "preview_only":
            raise ContractError("A0 effects must remain preview_only.")
        if payload.get("provider_write_count") != 0:
            raise ContractError("Effect previews must contain zero provider writes.")
        if payload.get("raw_deletion_count") != 0:
            raise ContractError("A0 effect previews must not delete raw material.")
    elif kind == "voice_sample":
        _require_sha256(payload, "source_media_sha256")
        _require_sha256(payload, "sample_sha256")
        start_ms = payload.get("start_ms")
        end_ms = payload.get("end_ms")
        if (
            not isinstance(start_ms, int)
            or not isinstance(end_ms, int)
            or start_ms < 0
            or end_ms <= start_ms
        ):
            raise ContractError("Voice sample range is invalid.")
        if payload.get("person_id") and (
            payload.get("review_state") != "reviewed"
            or not payload.get("review_authority_id")
            or not payload.get("consent_authority")
        ):
            raise ContractError(
                "A person-bound voice sample requires reviewed identity and authority."
            )
        private_ref = payload.get("private_audio_ref")
        if not isinstance(private_ref, Mapping) or not private_ref.get("sha256"):
            raise ContractError(
                "Voice sample requires a hashed private audio reference."
            )
        if _inline_biometric_keys(payload):
            raise ContractError("Voice sample contains inline biometric material.")
    elif kind == "voice_profile_version":
        if payload.get("status") not in {
            "pending",
            "active",
            "rejected",
            "superseded",
            "deleted",
        }:
            raise ContractError("Voice profile status is invalid.")
        samples = payload.get("sample_allowlist")
        if not isinstance(samples, list) or not samples:
            raise ContractError("Voice profile requires an exact sample allowlist.")
        if any(
            not isinstance(sample, Mapping)
            or not sample.get("sample_id")
            or not sample.get("review_authority_id")
            or not sample.get("consent_authority")
            for sample in samples
        ):
            raise ContractError("Every profile sample requires reviewed authority.")
        private_ref = payload.get("private_profile_ref")
        if payload.get("status") == "deleted":
            if private_ref not in (None, {}):
                raise ContractError(
                    "Deleted profiles must not retain biometric material."
                )
        elif not isinstance(private_ref, Mapping) or not private_ref.get("sha256"):
            raise ContractError("Voice profile requires a hashed private reference.")
        if _inline_biometric_keys(payload):
            raise ContractError("Voice profile contains inline biometric material.")
    elif kind == "transcript_correction_proposal":
        _require_sha256(payload, "raw_transcript_sha256")
        raw_span = payload.get("raw_span")
        if not isinstance(raw_span, Mapping):
            raise ContractError("Transcript correction raw span is incomplete.")
        start = raw_span.get("start")
        end = raw_span.get("end")
        if (
            not isinstance(start, int)
            or not isinstance(end, int)
            or start < 0
            or end <= start
        ):
            raise ContractError("Transcript correction raw span is invalid.")
        if not re.fullmatch(r"[a-f0-9]{64}", str(raw_span.get("text_sha256") or "")):
            raise ContractError("Transcript correction span hash is invalid.")
        scope = payload.get("scope")
        if (
            not isinstance(scope, Mapping)
            or scope.get("type")
            not in {
                "conversation",
                "project_matter",
                "organization",
                "domain",
                "global",
            }
            or not scope.get("id")
        ):
            raise ContractError("Transcript correction scope is invalid.")
        if payload.get("correction_pass") not in {"pre_identity", "post_identity"}:
            raise ContractError("Transcript correction pass is invalid.")
        if payload.get("cascade_count") not in {0, 1}:
            raise ContractError("Transcript correction violates the one-cascade limit.")
    elif kind == "processing_run":
        _require_original_filename(payload)
        _require_sha256(payload, "source_artifact_sha256")
        _require_sha256(payload, "source_media_sha256")
        if payload.get("operation_mode") not in {
            "contract_fixture",
            "shadow",
            "reviewed_learning",
            "policy_qualified_automation",
        }:
            raise ContractError("Processing run operation mode is invalid.")
        if payload.get("stage") not in SUPERVISOR_STAGES:
            raise ContractError("Processing run stage is invalid.")
        bounded_counters = {
            "transcript_correction_passes": SUPERVISOR_LIMITS[
                "max_transcript_correction_passes"
            ],
            "identity_cascade_count": SUPERVISOR_LIMITS[
                "max_transcript_identity_cascades"
            ],
            "provider_retry_count": SUPERVISOR_LIMITS["max_provider_retries"],
        }
        if any(
            not isinstance(payload.get(field), int)
            or payload[field] < 0
            or payload[field] > maximum
            for field, maximum in bounded_counters.items()
        ):
            raise ContractError("Processing run exceeded a supervisor limit.")
        effects = payload.get("effect_counts")
        if not isinstance(effects, Mapping):
            raise ContractError("Processing run effect counts are incomplete.")
        if payload.get("operation_mode") in {"contract_fixture", "shadow"} and any(
            not isinstance(value, int) or value != 0 for value in effects.values()
        ):
            raise ContractError(
                f"{payload['operation_mode']} processing runs require zero effects."
            )
    return dict(payload)


def validate_adapter_exchange(
    request: Mapping[str, Any],
    result: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Validate one bounded adapter request/result exchange."""
    validated_request = validate_artifact("provider_adapter_request", request)
    validated_result = validate_artifact("provider_adapter_result", result)
    if validated_result["request_id"] != validated_request["request_id"]:
        raise ContractError("Adapter result is not bound to the request.")
    if validated_result["processing_run_id"] != validated_request["processing_run_id"]:
        raise ContractError("Adapter result is not bound to the processing run.")
    if validated_result["source_scope"] != validated_request["source_scope"]:
        raise ContractError(
            "Adapter result source scope does not match the request scope."
        )
    consumed = validated_result.get("consumed_budget")
    if not isinstance(consumed, Mapping):
        raise ContractError("Adapter consumed_budget is incomplete.")
    limits = validated_request["budgets"]
    budget_pairs = {
        "records": "max_records",
        "characters": "max_characters",
        "calls": "max_calls",
    }
    if any(
        not isinstance(consumed.get(actual), int)
        or consumed[actual] < 0
        or consumed[actual] > limits[maximum]
        for actual, maximum in budget_pairs.items()
    ):
        raise ContractError("Adapter result exceeded its request budget.")
    return validated_request, validated_result
