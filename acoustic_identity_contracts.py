"""Versioned Plan 0037 acoustic identity artifact contracts.

The module deliberately contains no audio or model dependencies.  P0 freezes
the host-facing shapes and privacy invariants before later packets implement
the processing behind them.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any


DERIVED_AUDIO_SCHEMA = "transcribe-audio.derived-audio.v1"
AUDIO_QUALITY_SCHEMA = "transcribe-audio.audio-quality.v1"
BIOMETRIC_PROFILE_SCHEMA = "transcribe-audio.biometric-profile.v1"
VERIFICATION_TRIAL_SCHEMA = "transcribe-audio.verification-trial-manifest.v1"
ACOUSTIC_EVIDENCE_SCHEMA = "transcribe-audio.acoustic-evidence-bundle.v1"
REPROCESSING_MANIFEST_SCHEMA = "transcribe-audio.acoustic-reprocessing-manifest.v1"

SCHEMA_VERSIONS = {
    "derived_audio": DERIVED_AUDIO_SCHEMA,
    "audio_quality": AUDIO_QUALITY_SCHEMA,
    "biometric_profile": BIOMETRIC_PROFILE_SCHEMA,
    "verification_trial_manifest": VERIFICATION_TRIAL_SCHEMA,
    "acoustic_evidence_bundle": ACOUSTIC_EVIDENCE_SCHEMA,
    "reprocessing_manifest": REPROCESSING_MANIFEST_SCHEMA,
}

PRIVACY_CLASSES = {
    "derived_audio": "private_audio",
    "audio_quality": "portable_derived_metadata",
    "biometric_profile": "restricted_biometric",
    "verification_trial_manifest": "private_evaluation",
    "acoustic_evidence_bundle": "portable_bounded_evidence",
    "reprocessing_manifest": "private_operation",
}

REQUIRED_FIELDS: dict[str, tuple[str, ...]] = {
    "derived_audio": (
        "schema_version",
        "artifact_id",
        "source_blob_id",
        "source_sha256",
        "output_sha256",
        "source_duration_seconds",
        "output_duration_seconds",
        "recipe",
        "timestamp_map",
        "created_at",
    ),
    "audio_quality": (
        "schema_version",
        "assessment_id",
        "audio_artifact_id",
        "usable_speech_seconds",
        "metrics",
        "warnings",
        "abstention_reasons",
        "created_at",
    ),
    "biometric_profile": (
        "schema_version",
        "profile_id",
        "person_id",
        "status",
        "eligible_for_scoring",
        "confirmed_sources",
        "embedding_model_revision",
        "preprocessing_revision",
        "private_embedding_ref",
        "session_diversity",
        "dispersion",
        "audit",
    ),
    "verification_trial_manifest": (
        "schema_version",
        "manifest_id",
        "corpus_id",
        "conversation_split_policy",
        "recipe_revision",
        "model_revisions",
        "threshold_policy_revision",
        "trials",
        "denominators",
        "prediction_visibility",
        "created_at",
    ),
    "acoustic_evidence_bundle": (
        "schema_version",
        "bundle_id",
        "source_blob_id",
        "source_sha256",
        "recipe_revision",
        "model_revisions",
        "candidate_evidence",
        "same_person_label_evidence",
        "quality_summary",
        "abstention_reasons",
        "warnings",
        "created_at",
    ),
    "reprocessing_manifest": (
        "schema_version",
        "manifest_id",
        "mode",
        "status",
        "items",
        "source_lineage",
        "idempotency_key",
        "output_policy",
        "dry_run_predecessor",
        "resumable_checkpoint",
        "rollback",
        "recipe_revision",
        "model_revisions",
        "approval",
        "created_at",
    ),
}

PORTABLE_ARTIFACTS = {
    "audio_quality",
    "acoustic_evidence_bundle",
    "reprocessing_manifest",
}

FORBIDDEN_PORTABLE_KEY_FAMILIES = (
    "embedding",
    "vector",
    "enrollment_audio",
    "raw_audio",
    "audio_bytes",
    "waveform",
)

PROFILE_STATUSES = {"active", "superseded", "withdrawn", "deleted"}
REPROCESSING_MODES = {"dry_run", "apply", "rollback"}


class ContractError(ValueError):
    """Raised when an artifact violates a frozen acoustic contract."""


def contract_catalog() -> dict[str, Any]:
    """Return the stable, serializable P0 catalog used by tests and docs."""
    return {
        "catalog_version": "transcribe-audio.acoustic-contract-catalog.v1",
        "artifacts": {
            name: {
                "schema_version": SCHEMA_VERSIONS[name],
                "privacy_class": PRIVACY_CLASSES[name],
                "required_fields": list(REQUIRED_FIELDS[name]),
                "portable": name in PORTABLE_ARTIFACTS,
            }
            for name in SCHEMA_VERSIONS
        },
        "portable_forbidden_key_families": list(
            FORBIDDEN_PORTABLE_KEY_FAMILIES
        ),
        "invariants": [
            "original_audio_is_immutable",
            "derived_audio_is_content_addressed",
            "timestamps_map_to_original_audio",
            "operator_confirmation_required_for_enrollment",
            "raw_biometrics_remain_user_scoped",
            "voice_evidence_is_not_identity_authority",
            "portable_artifacts_exclude_raw_biometrics",
        ],
    }


def _forbidden_nested_keys(value: Any) -> set[str]:
    keys: set[str] = set()
    if isinstance(value, Mapping):
        for key, child in value.items():
            normalized = re.sub(r"[^a-z0-9]+", "_", str(key).strip().lower())
            if any(family in normalized for family in FORBIDDEN_PORTABLE_KEY_FAMILIES):
                keys.add(str(key))
            keys.update(_forbidden_nested_keys(child))
    elif isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray)
    ):
        for child in value:
            keys.update(_forbidden_nested_keys(child))
    return keys


def _require_nonempty_list(payload: Mapping[str, Any], field_name: str) -> None:
    value = payload.get(field_name)
    if not isinstance(value, list) or not value:
        raise ContractError(f"{field_name} must be a non-empty list.")


def validate_artifact(kind: str, payload: Mapping[str, Any]) -> dict[str, Any]:
    """Validate one artifact at the stable host seam and return a plain dict."""
    if kind not in SCHEMA_VERSIONS:
        raise ContractError(f"Unknown acoustic artifact kind: {kind}.")
    if not isinstance(payload, Mapping):
        raise ContractError("Acoustic artifact must be an object.")
    expected_schema = SCHEMA_VERSIONS[kind]
    if payload.get("schema_version") != expected_schema:
        raise ContractError(
            f"{kind} schema_version must be {expected_schema}."
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
                "Portable acoustic artifact contains forbidden biometric fields: "
                + ", ".join(forbidden)
                + "."
            )

    if kind == "derived_audio":
        for field_name in ("source_sha256", "output_sha256"):
            if not re.fullmatch(r"[a-f0-9]{64}", str(payload.get(field_name) or "")):
                raise ContractError(f"{field_name} must be a lowercase SHA-256.")
        recipe = payload.get("recipe")
        required_recipe_fields = {
            "revision",
            "operation",
            "decoder_revision",
            "parameters",
            "model_revisions",
        }
        if not isinstance(recipe, Mapping) or not required_recipe_fields.issubset(recipe):
            raise ContractError("derived_audio recipe is incomplete.")
        _require_nonempty_list(payload, "timestamp_map")
        source_duration = float(payload.get("source_duration_seconds") or 0.0)
        output_duration = float(payload.get("output_duration_seconds") or 0.0)
        if source_duration <= 0 or output_duration <= 0:
            raise ContractError("Audio durations must be positive.")
        prior_source_end = 0.0
        prior_output_end = 0.0
        for mapping in payload["timestamp_map"]:
            if not isinstance(mapping, Mapping):
                raise ContractError("Timestamp map entries must be objects.")
            try:
                source_start = float(mapping["source_start_seconds"])
                source_end = float(mapping["source_end_seconds"])
                output_start = float(mapping["output_start_seconds"])
                output_end = float(mapping["output_end_seconds"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ContractError("Timestamp map entry is incomplete.") from exc
            if (
                source_start < prior_source_end
                or output_start < prior_output_end
                or source_start < 0
                or output_start < 0
                or source_end <= source_start
                or output_end <= output_start
                or source_end > source_duration
                or output_end > output_duration
            ):
                raise ContractError("Timestamp map is not monotonic and in bounds.")
            prior_source_end = source_end
            prior_output_end = output_end
        if payload.get("source_sha256") == payload.get("output_sha256") and (
            recipe.get("operation") != "identity_copy"
        ):
            raise ContractError(
                "A transformed derived track must not claim the source hash."
            )
    elif kind == "biometric_profile":
        status = payload.get("status")
        if status not in PROFILE_STATUSES:
            raise ContractError("Biometric profile status is invalid.")
        _require_nonempty_list(payload, "confirmed_sources")
        for source in payload["confirmed_sources"]:
            if not isinstance(source, Mapping) or not source.get(
                "operator_confirmation_id"
            ):
                raise ContractError(
                    "Every biometric source requires operator-confirmed provenance."
                )
        private_ref = payload.get("private_embedding_ref")
        if status == "deleted":
            if private_ref not in (None, {}):
                raise ContractError(
                    "Deleted biometric profile must not retain usable biometric material."
                )
        else:
            if not isinstance(private_ref, Mapping) or not private_ref.get("sha256"):
                raise ContractError(
                    "Biometric profile requires a hashed private reference."
                )
            if "values" in private_ref or "vector" in private_ref:
                raise ContractError(
                    "Biometric profile stores embeddings by private reference, not inline."
                )
        eligible_for_scoring = payload.get("eligible_for_scoring")
        if status == "active" and eligible_for_scoring is not True:
            raise ContractError("Active biometric profile must be eligible for scoring.")
        if status in {"superseded", "withdrawn", "deleted"} and eligible_for_scoring is not False:
            raise ContractError(
                "Inactive biometric profiles must be ineligible for scoring."
            )
    elif kind == "verification_trial_manifest":
        if payload.get("prediction_visibility") != "excluded":
            raise ContractError("Verification trials must exclude predictions at freeze.")
        if payload.get("conversation_split_policy") != "conversation_id_disjoint":
            raise ContractError("Verification trials must split by conversation ID.")
        trials = payload.get("trials")
        denominators = payload.get("denominators")
        if not isinstance(trials, list) or not trials:
            raise ContractError("Verification trial manifest has no trials.")
        if not isinstance(denominators, Mapping):
            raise ContractError("Verification trial denominators are missing.")
        owners: dict[str, set[str]] = {}
        truth_counts = {"same_person": 0, "different_person": 0}
        for trial in trials:
            if not isinstance(trial, Mapping):
                raise ContractError("Verification trial must be an object.")
            split = str(trial.get("split") or "")
            truth = str(trial.get("truth") or "")
            test_conversation = str(trial.get("test_conversation_id") or "")
            enrollment_conversations = trial.get("enrollment_conversation_ids")
            if (
                split not in {"development", "calibration", "evaluation"}
                or truth not in truth_counts
                or not test_conversation
                or not isinstance(enrollment_conversations, list)
                or not enrollment_conversations
                or test_conversation in enrollment_conversations
            ):
                raise ContractError("Verification trial identity is invalid.")
            for conversation_id in [test_conversation, *enrollment_conversations]:
                owners.setdefault(str(conversation_id), set()).add(split)
            truth_counts[truth] += 1
        if any(len(splits) != 1 for splits in owners.values()):
            raise ContractError("Verification conversations cross splits.")
        if any(count < 1 for count in truth_counts.values()):
            raise ContractError("Verification trials need same and different pairs.")
        if int(denominators.get("trials") or 0) != len(trials):
            raise ContractError("Verification trial denominator is inconsistent.")
    elif kind == "acoustic_evidence_bundle":
        for candidate in payload.get("candidate_evidence") or []:
            if not isinstance(candidate, Mapping):
                raise ContractError("Candidate acoustic evidence must be an object.")
            if not candidate.get("prepared_candidate_id") or not candidate.get(
                "evidence_id"
            ):
                raise ContractError(
                    "Candidate acoustic evidence requires prepared candidate and evidence IDs."
                )
    elif kind == "reprocessing_manifest":
        if payload.get("mode") not in REPROCESSING_MODES:
            raise ContractError("Reprocessing mode is invalid.")
        _require_nonempty_list(payload, "items")
        _require_nonempty_list(payload, "source_lineage")
        if not str(payload.get("idempotency_key") or ""):
            raise ContractError("Reprocessing requires an idempotency key.")
        approval = payload.get("approval")
        dry_run = payload.get("dry_run_predecessor")
        checkpoint = payload.get("resumable_checkpoint")
        rollback = payload.get("rollback")
        if (
            not isinstance(dry_run, Mapping)
            or not dry_run.get("manifest_id")
            or not re.fullmatch(r"[a-f0-9]{64}", str(dry_run.get("sha256") or ""))
            or not isinstance(checkpoint, Mapping)
            or not checkpoint.get("checkpoint_id")
            or not isinstance(rollback, Mapping)
            or not rollback.get("rollback_manifest_id")
        ):
            raise ContractError(
                "Reprocessing requires dry-run, resumable, and rollback bindings."
            )
        if payload.get("mode") == "apply" and (
            not isinstance(approval, Mapping)
            or approval.get("status") != "approved"
            or not approval.get("approval_id")
            or approval.get("approved_manifest_id") != dry_run.get("manifest_id")
            or approval.get("approved_manifest_sha256") != dry_run.get("sha256")
        ):
            raise ContractError(
                "Apply reprocessing requires approval bound to the dry run."
            )
        output_policy = payload.get("output_policy")
        if (
            not isinstance(output_policy, Mapping)
            or output_policy.get("overwrite_original_audio") is not False
            or output_policy.get("overwrite_original_transcript") is not False
        ):
            raise ContractError("Reprocessing must preserve original audio and transcript.")

    return dict(payload)
