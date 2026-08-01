"""Deterministic, no-write generation-2 pre-reveal authority preview.

The preview freezes every input that can exist before successor evaluation
windows are revealed.  It deliberately cannot authorize model execution or
scoring: an exact-trial child authority is required after immutable window
selection and before the first model runs.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

import acoustic_audio_derivatives as audio_derivatives
import acoustic_device_provenance as device_provenance
import acoustic_source_device_metadata as source_device_metadata
import acoustic_speech_preparation as speech_preparation
import acoustic_successor_conditions as successor_conditions
import acoustic_verification as verification


PREVIEW_SCHEMA = (
    "transcribe-audio.verification-generation-2-pre-reveal-preview.v1"
)
SUCCESSOR_SEAL_SCHEMA = (
    "transcribe-audio.verification-generation-2-successor-seal.v1"
)
TRIAL_CHILD_SCHEMA = (
    "transcribe-audio.verification-generation-2-exact-trial-authority.v1"
)
EXPECTED_SUCCESSOR_CORPUS_ID = "acoustic-corpus-4a2b13e7bdc201f694af2f43"
EXPECTED_SUCCESSOR_CORPUS_CONTENT_SHA256 = (
    "4a2b13e7bdc201f694af2f43d4ab845749eeeb3ea06c7a97a40164cab40b83fe"
)
EXPECTED_SUCCESSOR_CORPUS_MANIFEST_SHA256 = (
    "4b77479d25d7b248cc62d500ed84c1604f105848da25ecef53661c5d9ea05a30"
)
EXPECTED_SUCCESSOR_GOLD_FREEZE_SHA256 = (
    "70ca36436a41a0a16c37eb295783e82f48cf8b2b57735c6d6db64c1e150d7d13"
)
EXPECTED_SUCCESSOR_GOLD_INDEX_SHA256 = (
    "59e443b41ea2b2fa9f4e1d7c33df3e80988750993cedb0ca4b99efb1c70e83df"
)
EXPECTED_SUCCESSOR_GOLD_BODY_AUTHORITY_SET_SHA256 = (
    "ec742d147936428ecd90db5d952e84a2fba8c60738ac8bd83de7d10abe1097b0"
)
EXPECTED_SUCCESSOR_REPOSITORY_COMMIT = (
    "50f34ab3fd36f7b00ece776c35c9d9e05c3571f3"
)
EXPECTED_SUCCESSOR_CORPUS_MODULE_SHA256 = (
    "f28f5b2a85aa13495ade291f9509497f2db6c825b79bab43caa086c98febe533"
)
EXPECTED_PRIOR_CORPORA = [
    {
        "corpus_id": "acoustic-corpus-1f93d1405f82676420571e1b",
        "manifest_sha256": (
            "73f0e04aab0274ddfeaa7f6b1567ecb135eebc0a0d6e5818cb3bd2ee5535dabf"
        ),
    },
    {
        "corpus_id": "acoustic-corpus-e81ea546dea777fa40e9d1c9",
        "manifest_sha256": (
            "bec631d8ad277a41801a359fdfbe79200fc85b1ca074cca63f052dad9a4e939a"
        ),
    },
]
EXPECTED_CONDITION_CONTENT_SHA256 = (
    "3ef3bcdabc776dfd80fb2002fa0b29377008c08ae9b2dc5f715e6155eb0f1a5e"
)
EXPECTED_CONDITION_MANIFEST_SHA256 = (
    "d9d2d1627f5ec069b088aaef44102e2850d796587646ef683bea0124e3bb7eba"
)
EXPECTED_CONDITION_PLAN_ID = "successor-conditions-b76095fdaf488f41930cc1f4"
EXPECTED_CONDITION_PLAN_CONTENT_SHA256 = (
    "b76095fdaf488f41930cc1f46309280d6f1de115377a8d0733fd27080432e76c"
)
EXPECTED_CONDITION_REPOSITORY_COMMIT = (
    "837edf02e67d113d38819937acf5833a2fbd0db3"
)
EXPECTED_CONDITION_MODULE_SHA256 = (
    "6c66b36a7a86d2cf6a8a3ae05d4dba7182c34d601acf61a20bb0585504122834"
)
EXPECTED_CONDITION_P1_MODULE_SHA256 = (
    "b3c71a170eca72dfd1c674499e59ca2c645622eded407809d0c2ec0e1f97896a"
)
EXPECTED_CONDITION_P2_MODULE_SHA256 = (
    "700e10d802a6443eab9d2bb9c6b9a7519cff26021ffec23acbdb767f12bcd595"
)
EXPECTED_CONDITION_READINESS_SHA256 = (
    "0ada7164d69de8c1f3ff8aa11d69c6524feff09f91c4f93875e09de0ddda335a"
)
EXPECTED_CONDITION_SAFE_PROJECTION_SHA256 = (
    "bbadd46c5b68d8a8210f20f4ec1f69cdee73f4efc5fe2c764d40bb70109befbd"
)
EXPECTED_CALIBRATION_APPLICATION_SHA256 = (
    "c00df454c799e5afa3993dec01c4f021e9236ced109b9bfcd6a44685a3f6a05b"
)
EXPECTED_CALIBRATION_SCORE_MATRIX_SHA256 = (
    "9bca1c323a4681536dffada1399fe591152c132e9e9073d299531d7ebed6fccb"
)
EXPECTED_GENERATION2_TERMINAL_POLICY_SHA256 = (
    "d741d8ef10594818646910b08a1dd925cfe40ffb04e3e8536a5c6d0ffad9330f"
)
DEFAULT_GENERATION2_TERMINAL_POLICY = Path(__file__).parent / (
    "docs/dev/fixtures/plan-0037-p4/generation-2-terminal-decision-policy.json"
)
SHA256_RE = re.compile(r"[a-f0-9]{64}")


class Generation2AuthorityError(ValueError):
    """Raised when a generation-2 preview input is incomplete or drifted."""


def _canonical_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()


def _exact_sha(value: Any, expected: str, label: str) -> str:
    selected = str(value or "")
    if not SHA256_RE.fullmatch(selected) or selected != expected:
        raise Generation2AuthorityError(f"{label} authority drifted.")
    return selected


def _safe_opaque_id(value: Any, label: str) -> str:
    selected = str(value or "")
    if not verification._OPAQUE_ID_RE.fullmatch(selected):
        raise Generation2AuthorityError(f"{label} is invalid.")
    return selected


def _condition_safe_projection(value: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": value.get("schema_version"),
        "status": value.get("status"),
        "content_sha256": value.get("content_sha256"),
        "corpus": dict(value.get("corpus") or {}),
        "denominators": dict(value.get("denominators") or {}),
        "units": [
            {
                key: item.get(key)
                for key in (
                    "recording_id", "conversation_id", "source_sha256", "split"
                )
            }
            for item in value.get("units") or []
            if isinstance(item, Mapping)
        ],
        "did_run_p1_p2": value.get("did_run_p1_p2"),
        "did_process_audio": value.get("did_process_audio"),
        "did_read_private_corpus_gold_authority": value.get(
            "did_read_private_corpus_gold_authority"
        ),
        "did_run_biometrics": value.get("did_run_biometrics"),
        "did_use_gold_for_condition_measurement": value.get(
            "did_use_gold_for_condition_measurement"
        ),
        "did_perform_external_write": value.get("did_perform_external_write"),
        "contains_raw_audio": value.get("contains_raw_audio"),
        "contains_transcript_text": value.get("contains_transcript_text"),
        "contains_names_or_emails": value.get("contains_names_or_emails"),
        "contains_embeddings_or_vectors": value.get(
            "contains_embeddings_or_vectors"
        ),
    }


def _composite_safe_projection(value: Mapping[str, Any]) -> dict[str, Any]:
    projected = {
        key: value.get(key)
        for key in (
            "schema_version",
            "status",
            "composite_id",
            "content_sha256",
            "condition_manifest_sha256",
            "condition_content_sha256",
            "recordings",
            "latest_attestation_count",
            "direct_observed_attestation_count",
            "condition_coverage",
            "will_run_models",
            "will_run_biometrics",
            "will_reveal_evaluation",
            "will_perform_external_write",
        )
    }
    if value.get("schema_version") in {
        source_device_metadata.AUGMENTED_COMPOSITE_PLAN_SCHEMA,
        source_device_metadata.AUGMENTED_COMPOSITE_MANIFEST_SCHEMA,
    }:
        projected.update(
            {
                key: value.get(key)
                for key in (
                    "authoritative_device_evidence_count",
                    "direct_operator_observed_count",
                    "source_metadata_observed_count",
                )
            }
        )
    return projected


def replay_historical_condition_campaign(
    manifest_path: Path,
    *,
    corpus_manifest_path: Path,
) -> dict[str, Any]:
    """Replay the closed Plan 0044 campaign against its own frozen authority."""
    selected = manifest_path.expanduser().resolve(strict=True)
    root = selected.parents[2]
    audio_derivatives.require_private_file(selected, root)
    if audio_derivatives.sha256_file(selected) != EXPECTED_CONDITION_MANIFEST_SHA256:
        raise Generation2AuthorityError("Historical condition manifest drifted.")
    manifest = audio_derivatives.read_private_object(selected)
    core = {
        key: value
        for key, value in manifest.items()
        if key not in {"content_sha256", "applied_at"}
    }
    expected_repository = {
        "clean": True,
        "commit": EXPECTED_CONDITION_REPOSITORY_COMMIT,
        "module_sha256": EXPECTED_CONDITION_MODULE_SHA256,
    }
    expected_modules = {
        "condition_sha256": EXPECTED_CONDITION_MODULE_SHA256,
        "p1_sha256": EXPECTED_CONDITION_P1_MODULE_SHA256,
        "p2_sha256": EXPECTED_CONDITION_P2_MODULE_SHA256,
    }
    selected_corpus = corpus_manifest_path.expanduser().resolve(strict=True)
    audio_derivatives.require_private_file(selected_corpus, selected_corpus.parent)
    if (
        manifest.get("schema_version") != successor_conditions.MANIFEST_SCHEMA
        or manifest.get("status") != "complete"
        or manifest.get("plan_id") != EXPECTED_CONDITION_PLAN_ID
        or manifest.get("plan_content_sha256")
        != EXPECTED_CONDITION_PLAN_CONTENT_SHA256
        or manifest.get("content_sha256") != EXPECTED_CONDITION_CONTENT_SHA256
        or _canonical_hash(core) != EXPECTED_CONDITION_CONTENT_SHA256
        or manifest.get("repository_authority") != expected_repository
        or manifest.get("module_authority") != expected_modules
        or manifest.get("readiness_sha256") != EXPECTED_CONDITION_READINESS_SHA256
        or manifest.get("corpus", {}).get("manifest_path") != str(selected_corpus)
        or manifest.get("corpus", {}).get("manifest_sha256")
        != EXPECTED_SUCCESSOR_CORPUS_MANIFEST_SHA256
        or audio_derivatives.sha256_file(selected_corpus)
        != EXPECTED_SUCCESSOR_CORPUS_MANIFEST_SHA256
        or audio_derivatives.sha256_file(Path(successor_conditions.__file__).resolve())
        != EXPECTED_CONDITION_MODULE_SHA256
        or audio_derivatives.sha256_file(Path(audio_derivatives.__file__).resolve())
        != EXPECTED_CONDITION_P1_MODULE_SHA256
        or audio_derivatives.sha256_file(Path(speech_preparation.__file__).resolve())
        != EXPECTED_CONDITION_P2_MODULE_SHA256
        or len(manifest.get("units") or []) != 7
    ):
        raise Generation2AuthorityError("Historical condition authority is invalid.")
    for unit in manifest["units"]:
        if not isinstance(unit, Mapping):
            raise Generation2AuthorityError("Historical condition unit is invalid.")
        for path_key, sha_key in (
            ("p1_manifest_path", "p1_manifest_sha256"),
            ("p2_comparison_path", "p2_comparison_sha256"),
            ("p1_replay_path", "p1_replay_sha256"),
            ("p2_replay_path", "p2_replay_sha256"),
        ):
            artifact = Path(str(unit.get(path_key) or "")).resolve(strict=True)
            audio_derivatives.require_private_file(artifact, root)
            if audio_derivatives.sha256_file(artifact) != unit.get(sha_key):
                raise Generation2AuthorityError(
                    "Historical condition lineage artifact drifted."
                )
        p1_manifest = audio_derivatives.read_private_object(
            Path(str(unit["p1_manifest_path"]))
        )
        p2_comparison = audio_derivatives.read_private_object(
            Path(str(unit["p2_comparison_path"]))
        )
        p1_artifact = Path(str(p1_manifest.get("artifact_path") or "")).resolve(
            strict=True
        )
        audio_derivatives.require_private_file(p1_artifact, root)
        derived_audio = p1_manifest.get("derived_audio")
        if (
            not isinstance(derived_audio, Mapping)
            or audio_derivatives.sha256_file(p1_artifact)
            != derived_audio.get("output_sha256")
        ):
            raise Generation2AuthorityError(
                "Historical P1 audio artifact drifted."
            )
        methods = p2_comparison.get("method_results") or []
        if (
            not isinstance(methods, list)
            or len(methods) != len(successor_conditions.METHOD_IDS)
            or [str(item.get("method_id") or "") for item in methods]
            != list(successor_conditions.METHOD_IDS)
            or any(item.get("status") != "success" for item in methods)
        ):
            raise Generation2AuthorityError(
                "Historical condition method set drifted."
            )
        if {
            str(item.get("method_id") or ""): _canonical_hash(item)
            for item in methods
            if isinstance(item, Mapping)
        } != unit.get("method_result_sha256"):
            raise Generation2AuthorityError(
                "Historical condition method lineage drifted."
            )
        for method in methods:
            output = Path(str(method.get("output_path") or "")).resolve(strict=True)
            audio_derivatives.require_private_file(output, root)
            if audio_derivatives.sha256_file(output) != method.get("output_sha256"):
                raise Generation2AuthorityError(
                    "Historical condition method output drifted."
                )
        if successor_conditions._conditions(p1_manifest, p2_comparison) != unit.get(
            "conditions"
        ):
            raise Generation2AuthorityError(
                "Historical condition measurement drifted."
            )
    if successor_conditions._aggregate_conditions(manifest["units"]) != manifest.get(
        "condition_coverage"
    ):
        raise Generation2AuthorityError("Historical condition coverage drifted.")
    receipt_path = selected.parent / "apply-receipt.json"
    audio_derivatives.require_private_file(receipt_path, root)
    receipt = audio_derivatives.read_private_object(receipt_path)
    expected_receipt = {
        "schema_version": successor_conditions.RECEIPT_SCHEMA,
        "plan_id": EXPECTED_CONDITION_PLAN_ID,
        "manifest_path": str(selected),
        "manifest_sha256": EXPECTED_CONDITION_MANIFEST_SHA256,
        "content_sha256": EXPECTED_CONDITION_CONTENT_SHA256,
        "denominators": manifest["denominators"],
        "condition_coverage": manifest["condition_coverage"],
        "mode": "0600",
        "will_perform_external_write": False,
    }
    if receipt != expected_receipt:
        raise Generation2AuthorityError("Historical condition receipt drifted.")
    return {
        "schema_version": successor_conditions.REPLAY_SCHEMA,
        "plan_id": EXPECTED_CONDITION_PLAN_ID,
        "manifest_sha256": EXPECTED_CONDITION_MANIFEST_SHA256,
        "content_sha256": EXPECTED_CONDITION_CONTENT_SHA256,
        "condition_coverage": manifest["condition_coverage"],
        "safe_projection_sha256": _canonical_hash(
            _condition_safe_projection(manifest)
        ),
        "full_body_match": True,
        "historical_authority_replay": True,
        "idempotent": True,
        "will_perform_external_write": False,
    }


def _successor_seal(
    condition_manifest: Mapping[str, Any],
    condition_replay: Mapping[str, Any],
) -> dict[str, Any]:
    """Project an exact metadata-only successor seal from replayed conditions."""
    corpus = condition_manifest.get("corpus")
    units = condition_manifest.get("units")
    denominators = condition_manifest.get("denominators")
    if not isinstance(corpus, Mapping) or not isinstance(denominators, Mapping):
        raise Generation2AuthorityError("Successor condition authority is invalid.")
    if not isinstance(units, list) or len(units) != 7:
        raise Generation2AuthorityError("Successor condition unit coverage changed.")
    _exact_sha(
        condition_replay.get("manifest_sha256"),
        EXPECTED_CONDITION_MANIFEST_SHA256,
        "Condition manifest",
    )
    if condition_replay.get("safe_projection_sha256") != _canonical_hash(
        _condition_safe_projection(condition_manifest)
    ) or condition_replay.get("safe_projection_sha256") != (
        EXPECTED_CONDITION_SAFE_PROJECTION_SHA256
    ):
        raise Generation2AuthorityError(
            "Successor condition projection binding changed."
        )
    _exact_sha(
        condition_replay.get("content_sha256"),
        EXPECTED_CONDITION_CONTENT_SHA256,
        "Condition content",
    )
    if (
        condition_replay.get("schema_version")
        != successor_conditions.REPLAY_SCHEMA
        or condition_replay.get("full_body_match") is not True
        or condition_replay.get("historical_authority_replay") is not True
        or condition_replay.get("will_perform_external_write") is not False
        or condition_manifest.get("schema_version")
        != successor_conditions.MANIFEST_SCHEMA
        or condition_manifest.get("status") != "complete"
        or condition_manifest.get("content_sha256")
        != EXPECTED_CONDITION_CONTENT_SHA256
        or condition_manifest.get("did_run_p1_p2") is not True
        or condition_manifest.get("did_process_audio") is not True
        or condition_manifest.get("did_read_private_corpus_gold_authority")
        is not True
        or condition_manifest.get("did_run_biometrics") is not False
        or condition_manifest.get("did_use_gold_for_condition_measurement")
        is not False
        or condition_manifest.get("did_perform_external_write") is not False
        or condition_manifest.get("contains_raw_audio") is not False
        or condition_manifest.get("contains_transcript_text") is not False
        or condition_manifest.get("contains_names_or_emails") is not False
        or condition_manifest.get("contains_embeddings_or_vectors") is not False
        or denominators.get("recordings") != 7
        or denominators.get("methods_per_recording") != 5
        or denominators.get("method_attempts") != 35
        or denominators.get("p1_successes") != 7
        or denominators.get("p2_method_successes") != 35
        or corpus.get("corpus_id") != EXPECTED_SUCCESSOR_CORPUS_ID
        or corpus.get("content_sha256")
        != EXPECTED_SUCCESSOR_CORPUS_CONTENT_SHA256
        or corpus.get("manifest_sha256")
        != EXPECTED_SUCCESSOR_CORPUS_MANIFEST_SHA256
    ):
        raise Generation2AuthorityError("Successor condition replay is invalid.")

    projected = []
    seen_dimensions: dict[str, set[str]] = {
        "recording_id": set(),
        "conversation_id": set(),
        "source_sha256": set(),
    }
    for unit in units:
        if not isinstance(unit, Mapping):
            raise Generation2AuthorityError("Successor condition unit is invalid.")
        split = str(unit.get("split") or "")
        recording_id = _safe_opaque_id(unit.get("recording_id"), "Recording ID")
        conversation_id = _safe_opaque_id(
            unit.get("conversation_id"), "Conversation ID"
        )
        source_sha256 = str(unit.get("source_sha256") or "")
        if split not in {"development", "calibration", "evaluation"}:
            raise Generation2AuthorityError("Successor split is invalid.")
        if not SHA256_RE.fullmatch(source_sha256):
            raise Generation2AuthorityError("Successor source authority is invalid.")
        projected.append(
            {
                "recording_id": recording_id,
                "conversation_id": conversation_id,
                "source_sha256": source_sha256,
                "split": split,
            }
        )
        seen_dimensions["recording_id"].add(recording_id)
        seen_dimensions["conversation_id"].add(conversation_id)
        seen_dimensions["source_sha256"].add(source_sha256)
    if any(len(values) != 7 for values in seen_dimensions.values()):
        raise Generation2AuthorityError("Successor seal is not pairwise disjoint.")
    projected.sort(key=lambda item: (item["split"], item["recording_id"]))
    split_counts = Counter(item["split"] for item in projected)
    if dict(split_counts) != {"calibration": 2, "development": 3, "evaluation": 2}:
        raise Generation2AuthorityError("Successor split denominator changed.")
    evaluation_records = [
        dict(item) for item in projected if item["split"] == "evaluation"
    ]
    seal_core = {
        "schema_version": SUCCESSOR_SEAL_SCHEMA,
        "status": "sealed",
        "corpus_id": EXPECTED_SUCCESSOR_CORPUS_ID,
        "corpus_content_sha256": EXPECTED_SUCCESSOR_CORPUS_CONTENT_SHA256,
        "corpus_manifest_sha256": EXPECTED_SUCCESSOR_CORPUS_MANIFEST_SHA256,
        "gold_freeze_sha256": EXPECTED_SUCCESSOR_GOLD_FREEZE_SHA256,
        "gold_index_sha256": EXPECTED_SUCCESSOR_GOLD_INDEX_SHA256,
        "gold_body_authority_set_sha256": (
            EXPECTED_SUCCESSOR_GOLD_BODY_AUTHORITY_SET_SHA256
        ),
        "prior_corpora": [dict(item) for item in EXPECTED_PRIOR_CORPORA],
        "prior_corpus_overlap_counts": {
            "document_id": 0,
            "recording_id": 0,
            "conversation_id": 0,
            "source_sha256": 0,
        },
        "corpus_repository_authority": {
            "clean": True,
            "commit": EXPECTED_SUCCESSOR_REPOSITORY_COMMIT,
            "module_sha256": EXPECTED_SUCCESSOR_CORPUS_MODULE_SHA256,
        },
        "condition_manifest_sha256": EXPECTED_CONDITION_MANIFEST_SHA256,
        "condition_content_sha256": EXPECTED_CONDITION_CONTENT_SHA256,
        "split_algorithm": "chronological_rank_quota_3_2_2.v1",
        "split_counts": {
            "development": 3,
            "calibration": 2,
            "evaluation": 2,
        },
        "recording_count": 7,
        "conversation_count": 7,
        "evaluation_recording_count": 2,
        "evaluation_conversation_count": 2,
        "known_subject_count": 10,
        "recurrent_subject_count": 3,
        "independent_same_person_subject_session_pair_count": 23,
        "different_person_session_pair_count": 114,
        "record_set_sha256": _canonical_hash(projected),
        "evaluation_record_set_sha256": _canonical_hash(evaluation_records),
        "evaluation_records": evaluation_records,
        "prediction_visibility": "excluded",
        "contains_gold_bodies": False,
        "contains_raw_audio": False,
        "contains_transcript_text": False,
        "contains_names_or_emails": False,
        "contains_embeddings_or_vectors": False,
    }
    return {**seal_core, "seal_sha256": _canonical_hash(seal_core)}


def _composite_binding(
    composite_manifest: Mapping[str, Any],
    composite_replay: Mapping[str, Any],
) -> dict[str, Any]:
    coverage = composite_replay.get("condition_coverage")
    fields = coverage.get("fields") if isinstance(coverage, Mapping) else None
    device = fields.get("device") if isinstance(fields, Mapping) else None
    augmented = (
        composite_manifest.get("schema_version")
        == source_device_metadata.AUGMENTED_COMPOSITE_MANIFEST_SCHEMA
    )
    plan_schema = (
        source_device_metadata.AUGMENTED_COMPOSITE_PLAN_SCHEMA
        if augmented
        else device_provenance.COMPOSITE_SCHEMA
    )
    manifest_schema = (
        source_device_metadata.AUGMENTED_COMPOSITE_MANIFEST_SCHEMA
        if augmented
        else device_provenance.COMPOSITE_MANIFEST_SCHEMA
    )
    replay_schema = (
        source_device_metadata.AUGMENTED_COMPOSITE_REPLAY_SCHEMA
        if augmented
        else device_provenance.COMPOSITE_RECEIPT_SCHEMA
    )
    content_body = {
        key: value
        for key, value in composite_manifest.items()
        if key not in {
            "applied_at",
            "composite_id",
            "content_sha256",
            "status",
        }
    }
    content_body["schema_version"] = plan_schema
    recomputed_content_sha256 = _canonical_hash(content_body)
    evidence_counts_valid = (
        composite_manifest.get("authoritative_device_evidence_count") == 7
        and composite_manifest.get("direct_operator_observed_count") == 2
        and composite_manifest.get("source_metadata_observed_count") == 5
        if augmented
        else composite_manifest.get("latest_attestation_count") == 7
        and composite_manifest.get("direct_observed_attestation_count") == 7
    )
    augmented_evidence_valid = True
    if augmented:
        evidence = composite_manifest.get("evidence")
        augmented_evidence_valid = isinstance(evidence, list) and len(evidence) == 7
        if augmented_evidence_valid:
            positions = []
            recording_ids = set()
            source_sha256s = set()
            device_ids = set()
            basis_positions: dict[str, list[int]] = {
                "direct_operator_knowledge": [],
                "source_embedded_manufacturer_hardware_model": [],
            }
            for item in evidence:
                if not isinstance(item, Mapping):
                    augmented_evidence_valid = False
                    break
                position = item.get("position")
                recording_id = str(item.get("recording_id") or "")
                source_sha256 = str(item.get("source_sha256") or "")
                device_id = str(item.get("device_id") or "")
                evidence_sha256 = str(item.get("evidence_sha256") or "")
                basis = str(item.get("evidence_basis") or "")
                if (
                    set(item)
                    != {
                        "position",
                        "recording_id",
                        "source_sha256",
                        "device_id",
                        "evidence_basis",
                        "evidence_sha256",
                    }
                    or not isinstance(position, int)
                    or not verification._OPAQUE_ID_RE.fullmatch(recording_id)
                    or not SHA256_RE.fullmatch(source_sha256)
                    or not verification._OPAQUE_ID_RE.fullmatch(device_id)
                    or not SHA256_RE.fullmatch(evidence_sha256)
                    or basis not in basis_positions
                ):
                    augmented_evidence_valid = False
                    break
                positions.append(position)
                recording_ids.add(recording_id)
                source_sha256s.add(source_sha256)
                device_ids.add(device_id)
                basis_positions[basis].append(position)
            augmented_evidence_valid = augmented_evidence_valid and (
                positions == list(range(1, 8))
                and len(recording_ids) == 7
                and len(source_sha256s) == 7
                and basis_positions["direct_operator_knowledge"] == [2, 4]
                and basis_positions[
                    "source_embedded_manufacturer_hardware_model"
                ]
                == [1, 3, 5, 6, 7]
                and sorted(device_ids) == device.get("observed_values")
                if isinstance(device, Mapping)
                else False
            )
    expected_prefix = "augmented-composite-" if augmented else "composite-conditions-"
    if (
        composite_manifest.get("schema_version")
        != manifest_schema
        or composite_manifest.get("status") != "complete"
        or composite_manifest.get("condition_manifest_sha256")
        != EXPECTED_CONDITION_MANIFEST_SHA256
        or composite_manifest.get("condition_content_sha256")
        != EXPECTED_CONDITION_CONTENT_SHA256
        or composite_manifest.get("recordings") != 7
        or not evidence_counts_valid
        or not augmented_evidence_valid
        or composite_manifest.get("condition_coverage") != coverage
        or composite_manifest.get("will_run_models") is not False
        or composite_manifest.get("will_run_biometrics") is not False
        or composite_manifest.get("will_reveal_evaluation") is not False
        or composite_manifest.get("will_perform_external_write") is not False
        or composite_replay.get("schema_version")
        != replay_schema
        or composite_replay.get("full_body_match") is not True
        or composite_replay.get("will_perform_external_write") is not False
        or composite_manifest.get("content_sha256")
        != recomputed_content_sha256
        or not isinstance(coverage, Mapping)
        or coverage.get("terminal_selection_eligible") is not True
        or coverage.get("blockers") != []
        or not isinstance(device, Mapping)
        or device.get("status") != "pass"
        or device.get("missing_recordings") != 0
        or int(device.get("observed_value_count") or 0) < 2
    ):
        raise Generation2AuthorityError("Composite condition authority is invalid.")
    for label, value in (
        ("Composite content", composite_replay.get("content_sha256")),
        ("Composite manifest", composite_replay.get("manifest_sha256")),
    ):
        selected = str(value or "")
        if not SHA256_RE.fullmatch(selected):
            raise Generation2AuthorityError(f"{label} hash is invalid.")
    if (
        composite_manifest.get("content_sha256")
        != composite_replay.get("content_sha256")
        or composite_manifest.get("composite_id")
        != composite_replay.get("composite_id")
        or composite_manifest.get("composite_id")
        != f"{expected_prefix}{recomputed_content_sha256[:24]}"
    ):
        raise Generation2AuthorityError("Composite condition binding changed.")
    common = {
        "composite_id": _safe_opaque_id(
            composite_replay.get("composite_id"), "Composite ID"
        ),
        "content_sha256": composite_replay["content_sha256"],
        "manifest_sha256": composite_replay["manifest_sha256"],
        "condition_coverage": dict(coverage),
        "minimum_distinct_device_count": 2,
    }
    if augmented:
        return {
            **common,
            "authoritative_device_evidence_count": 7,
            "direct_operator_observed_count": 2,
            "source_metadata_observed_count": 5,
        }
    return {**common, "direct_observed_attestation_count": 7}


def _calibration_binding(
    calibration: Mapping[str, Any], calibration_authority: Mapping[str, Any]
) -> tuple[
    dict[str, Any],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    calibration_body = {
        key: value
        for key, value in calibration.items()
        if key
        not in {
            "application_sha256",
            "private_application_path",
            "threshold_replay_mode",
        }
    }
    authority_body = {
        key: value
        for key, value in calibration_authority.items()
        if key not in {"authority_sha256", "private_authority_path"}
    }
    recomputed_application_sha256 = verification._calibration_stage_identity(
        calibration_body, "applied_at"
    )
    recomputed_authority_sha256 = audio_derivatives.canonical_artifact_hash(
        authority_body
    )
    if (
        calibration.get("schema_version") != verification.CALIBRATION_APPLICATION_SCHEMA
        or calibration.get("status") != "success"
        or calibration.get("intended_split") != "calibration"
        or calibration.get("application_sha256")
        != EXPECTED_CALIBRATION_APPLICATION_SHA256
        or recomputed_application_sha256
        != EXPECTED_CALIBRATION_APPLICATION_SHA256
        or calibration.get("authority_sha256")
        != verification.HISTORICAL_CALIBRATION_AUTHORITY_SHA256
        or calibration.get("score_matrix_sha256")
        != EXPECTED_CALIBRATION_SCORE_MATRIX_SHA256
        or calibration.get("threshold_unit_count") != 9
        or calibration.get("did_select_and_freeze_thresholds") is not True
        or calibration.get("did_read_evaluation") is not False
        or calibration.get("did_make_terminal_model_or_method_selection") is not False
        or calibration.get("permits_generalization_claim") is not False
        or calibration_authority.get("authority_sha256")
        != verification.HISTORICAL_CALIBRATION_AUTHORITY_SHA256
        or recomputed_authority_sha256
        != verification.HISTORICAL_CALIBRATION_AUTHORITY_SHA256
        or calibration_authority.get("status") != "authorized"
        or calibration_authority.get("intended_split") != "calibration"
        or calibration_authority.get("will_read_evaluation") is not False
    ):
        raise Generation2AuthorityError("Calibration replay authority is invalid.")
    thresholds = calibration.get("thresholds")
    profiles = calibration_authority.get("profiles")
    methods = list(calibration_authority.get("score_methods") or [])
    if not isinstance(thresholds, list) or len(thresholds) != 9:
        raise Generation2AuthorityError("Frozen threshold matrix is invalid.")
    if not isinstance(profiles, list) or len(profiles) != 6 or len(methods) != 3:
        raise Generation2AuthorityError("Frozen candidate inventory is invalid.")
    profile_projection = []
    candidate_ids = set()
    for profile in profiles:
        if not isinstance(profile, Mapping):
            raise Generation2AuthorityError("Frozen profile is invalid.")
        projection = {
            key: profile.get(key)
            for key in (
                "profile_id",
                "descendant_id",
                "person_ref_id",
                "candidate_id",
                "model_revision",
                "artifact_sha256",
                "profile_manifest_sha256",
                "generation_sha256",
                "lifecycle_state",
            )
        }
        if (
            projection["lifecycle_state"] != "active"
            or any(not projection[key] for key in projection)
        ):
            raise Generation2AuthorityError("Frozen profile binding is invalid.")
        profile_projection.append(projection)
        candidate_ids.add(str(projection["candidate_id"]))
    if len(candidate_ids) != 3:
        raise Generation2AuthorityError("Candidate-model denominator changed.")

    frozen_thresholds = []
    threshold_units = set()
    for item in thresholds:
        if not isinstance(item, Mapping):
            raise Generation2AuthorityError("Frozen threshold is invalid.")
        threshold = item.get("threshold")
        temperature = item.get("temperature")
        unit = (str(item.get("candidate_id") or ""), str(item.get("method_id") or ""))
        if (
            item.get("status") != "success"
            or unit[0] not in candidate_ids
            or unit[1] not in methods
            or isinstance(threshold, bool)
            or not isinstance(threshold, (int, float))
            or not math.isfinite(float(threshold))
            or isinstance(temperature, bool)
            or not isinstance(temperature, (int, float))
            or not math.isfinite(float(temperature))
            or float(temperature) <= 0
        ):
            raise Generation2AuthorityError("Frozen threshold binding is invalid.")
        threshold_units.add(unit)
        frozen_thresholds.append(
            {
                "candidate_id": unit[0],
                "method_id": unit[1],
                "threshold": threshold,
                "temperature": temperature,
                "calibration_status": "success",
            }
        )
    expected_units = {(candidate, method) for candidate in candidate_ids for method in methods}
    if threshold_units != expected_units:
        raise Generation2AuthorityError("Frozen threshold matrix is incomplete.")
    profile_projection.sort(key=lambda item: str(item["profile_id"]))
    frozen_thresholds.sort(key=lambda item: (item["candidate_id"], item["method_id"]))
    candidate_matrix = [
        {
            "candidate_id": candidate,
            "method_id": method,
            "profile_ids": sorted(
                str(item["profile_id"])
                for item in profile_projection
                if item["candidate_id"] == candidate
            ),
        }
        for candidate, method in sorted(expected_units)
    ]
    binding = {
        "application_sha256": EXPECTED_CALIBRATION_APPLICATION_SHA256,
        "authority_sha256": verification.HISTORICAL_CALIBRATION_AUTHORITY_SHA256,
        "score_matrix_sha256": EXPECTED_CALIBRATION_SCORE_MATRIX_SHA256,
        "historical_p2_replay_contract": verification.historical_p2_replay_contract(),
        "threshold_unit_count": 9,
        "profile_count": 6,
        "candidate_count": 3,
        "method_count": 3,
    }
    return binding, profile_projection, frozen_thresholds, candidate_matrix


def preview_generation2_pre_reveal_authority(
    *,
    calibration: Mapping[str, Any],
    calibration_authority: Mapping[str, Any],
    condition_manifest: Mapping[str, Any],
    condition_replay: Mapping[str, Any],
    composite_manifest: Mapping[str, Any],
    composite_replay: Mapping[str, Any],
    terminal_policy: Mapping[str, Any],
    terminal_policy_sha256: str,
) -> dict[str, Any]:
    """Build a deterministic private-data-free preview without filesystem writes."""
    successor_seal = _successor_seal(condition_manifest, condition_replay)
    composite = _composite_binding(composite_manifest, composite_replay)
    calibration_binding, profiles, thresholds, candidate_matrix = _calibration_binding(
        calibration, calibration_authority
    )
    expected_minimum_evidence = {
        "genuine_trials_per_model_method_unit": 20,
        "impostor_trials_per_model_method_unit": 100,
        "open_set_trials_per_model_method_unit": 20,
        "successor_corpus_recordings": 7,
        "successor_corpus_conversations": 7,
        "evaluation_recordings": 2,
        "evaluation_conversations": 2,
        "known_subjects": 5,
        "recurrent_subjects": 2,
        "independent_same_person_subject_session_pairs": 4,
        "all_declared_condition_slices_reported": True,
        "minimum_observed_values_per_condition": 2,
        "missing_condition_recordings": 0,
    }
    policy_path = DEFAULT_GENERATION2_TERMINAL_POLICY.resolve(strict=True)
    if audio_derivatives.sha256_file(policy_path) != terminal_policy_sha256:
        raise Generation2AuthorityError("Generation-2 terminal policy file drifted.")
    try:
        stored_terminal_policy = json.loads(policy_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise Generation2AuthorityError(
            "Generation-2 terminal policy is unreadable."
        ) from exc
    if (
        dict(terminal_policy) != stored_terminal_policy
        or
        terminal_policy.get("schema_version")
        != "transcribe-audio.verification-generation-2-terminal-decision-policy.v1"
        or terminal_policy.get("precedence") != ["stop", "reject", "select", "refine"]
        or terminal_policy.get("policy_changes_after_evaluation_unseal")
        != "forbidden_for_this_evaluation_generation"
        or terminal_policy.get("minimum_evidence") != expected_minimum_evidence
        or terminal_policy.get("exact_trial_child_may_change_parent_policy")
        is not False
    ):
        raise Generation2AuthorityError("Terminal decision policy is invalid.")
    _exact_sha(
        terminal_policy_sha256,
        EXPECTED_GENERATION2_TERMINAL_POLICY_SHA256,
        "Generation-2 terminal policy",
    )
    exact_trial_child_policy = {
        "required_schema_version": TRIAL_CHILD_SCHEMA,
        "required_before_model_or_score_execution": True,
        "must_bind_parent_content_sha256": True,
        "must_bind_immutable_window_manifest_sha256": True,
        "must_cover_every_candidate_matrix_unit": True,
        "must_freeze_exact_trial_ids": True,
        "must_freeze_per_unit_class_denominators": {
            "genuine": 20,
            "impostor": 100,
            "open_set": 20,
        },
        "may_change_parent_policy_threshold_margin_or_candidate": False,
        "missing_or_incomplete_child_action": "global_stop_before_model_execution",
    }
    core = {
        "schema_version": PREVIEW_SCHEMA,
        "status": "ready_for_independent_review",
        "reason_code": None,
        "authority_generation": 2,
        "intended_split": "evaluation",
        "canonicalization": "json_sort_keys_compact_utf8",
        "authority_module_sha256": audio_derivatives.sha256_file(
            Path(__file__).resolve()
        ),
        "successor_seal": successor_seal,
        "composite_condition_authority": composite,
        "calibration_authority": calibration_binding,
        "terminal_decision_policy_sha256": terminal_policy_sha256,
        "terminal_decision_policy": dict(terminal_policy),
        "profiles": profiles,
        "frozen_thresholds": thresholds,
        "fixed_abstention_margins": [
            {
                "candidate_id": item["candidate_id"],
                "method_id": item["method_id"],
                "margin": 0.0,
                "derivation": "fixed_before_evaluation_not_data_selected",
            }
            for item in thresholds
        ],
        "candidate_matrix": candidate_matrix,
        "preparation_contract": {
            "p1_module_sha256": audio_derivatives.sha256_file(
                Path(audio_derivatives.__file__).resolve()
            ),
            "p2_module_sha256": audio_derivatives.sha256_file(
                Path(speech_preparation.__file__).resolve()
            ),
            "p2_open_acquisition_manifest_sha256": (
                verification.EXPECTED_P2_OPEN_ACQUISITION_MANIFEST_SHA256
            ),
            "p2_pyannote_acquisition_manifest_sha256": (
                verification.EXPECTED_P2_PYANNOTE_ACQUISITION_MANIFEST_SHA256
            ),
            "preparation_methods": list(calibration_authority["preparation_methods"]),
            "score_methods": list(calibration_authority["score_methods"]),
            "window_policy": dict(calibration_authority["window_policy"]),
            "channel_policy": {
                **dict(calibration_authority["preparation_contract"]["channel_policy"]),
                "authority_binding": "generation_2_pre_reveal_content_sha256",
            },
            "no_fallback_method": True,
        },
        "trial_construction_policy": {
            "evaluation_record_set_sha256": successor_seal[
                "evaluation_record_set_sha256"
            ],
            "same_frozen_window_set_for_every_candidate_unit": True,
            "trial_score": "raw_cosine_against_fixed_enrollment_centroid",
            "same_person_class": "operator_gold_subject_matches_profile_person_ref",
            "different_person_class": "operator_gold_subject_differs_from_profile_person_ref",
            "open_set_class": "operator_gold_subject_has_no_frozen_profile",
            "mixed_or_unknown_gold": "excluded_before_scoring",
            "trial_id_derivation": (
                "canonical_hash(parent_content,window_id,method_id,profile_id,class)"
            ),
            "no_model_output_may_change_membership": True,
        },
        "exact_trial_child_policy": exact_trial_child_policy,
        "score_aggregation_policy": {
            "threshold_input": "raw_cosine_score",
            "profile_aggregation": "fixed_enrollment_centroid_only",
            "same_timestamp_bounds_across_score_methods": True,
            "ties_abstain_before_tie_break": True,
            "no_score_or_threshold_normalization_change": True,
        },
        "evaluation_metric_policy": {
            "trial_metrics": dict(calibration_authority["metric_policy"]),
            "thresholds_and_temperatures_are_frozen": True,
            "attempt_accounting": "attempted_success_failed_blocked_reported_separately",
            "all_declared_condition_slices_reported": True,
            "conversation_clustered_non_independent": True,
        },
        "minimum_evidence_policy": {
            **dict(terminal_policy["minimum_evidence"]),
            "applies_per_model_method_unit": True,
            "incomplete_cartesian_or_failed_or_blocked_cell": "global_stop",
            "nonfinite_score_or_required_metric": "global_stop",
        },
        "terminal_resolution_policy": {
            "unit_precedence": ["stop", "reject", "select", "refine"],
            "global_integrity_or_minimum_evidence_failure": "stop",
            "any_unit_stop": "global_stop_before_candidate_reduction",
            "evaluation_may_not_change_policy_threshold_margin_or_candidate": True,
        },
        "will_reveal_evaluation_after_apply": True,
        "will_prepare_evaluation_audio_after_apply": True,
        "will_freeze_evaluation_windows_after_apply": True,
        "will_run_models": False,
        "will_score_trials": False,
        "will_calculate_terminal_metrics": False,
        "will_make_terminal_decision": False,
        "will_select_or_change_thresholds": False,
        "will_mutate_profiles_or_references": False,
        "will_enable_default_integration": False,
        "will_automatically_confirm_identity": False,
        "will_run_historical_reprocessing": False,
        "will_perform_external_write": False,
        "production_apply_authorized": False,
        "requires_independent_review": True,
        "requires_clean_pushed_commit": True,
        "contains_biometric_scores": False,
        "contains_raw_audio": False,
        "contains_transcript_text": False,
        "contains_names_or_emails": False,
        "contains_gold_bodies": False,
        "contains_embeddings_or_vectors": False,
        "contains_raw_biometric_values": False,
    }
    if (
        core["preparation_contract"]["p1_module_sha256"]
        != EXPECTED_CONDITION_P1_MODULE_SHA256
        or core["preparation_contract"]["p2_module_sha256"]
        != EXPECTED_CONDITION_P2_MODULE_SHA256
    ):
        raise Generation2AuthorityError("Current preparation module authority drifted.")
    if verification._contains_forbidden_private_key(core):
        raise Generation2AuthorityError(
            "Generation-2 preview contains forbidden private data."
        )
    content_sha256 = _canonical_hash(core)
    return {
        **core,
        "preview_id": f"generation-2-pre-reveal-{content_sha256[:24]}",
        "content_sha256": content_sha256,
    }


def replay_generation2_pre_reveal_preview(
    stored_preview: Mapping[str, Any],
    **inputs: Mapping[str, Any],
) -> dict[str, Any]:
    """Recompute and compare a preview in memory; perform no writes."""
    expected = preview_generation2_pre_reveal_authority(**inputs)
    if dict(stored_preview) != expected:
        raise Generation2AuthorityError("Generation-2 preview replay mismatch.")
    return {
        "schema_version": PREVIEW_SCHEMA,
        "preview_id": expected["preview_id"],
        "content_sha256": expected["content_sha256"],
        "full_body_match": True,
        "will_reveal_evaluation": False,
        "will_run_models": False,
        "will_score_trials": False,
        "will_perform_external_write": False,
    }
