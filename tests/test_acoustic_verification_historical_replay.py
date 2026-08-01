from __future__ import annotations

import inspect
import json
from pathlib import Path

import pytest

import acoustic_verification as verification
from acoustic_verification import AcousticVerificationError


def _archived_authority() -> dict[str, object]:
    return {
        "preparation_contract": {
            "p2_module_sha256": (
                verification.HISTORICAL_CALIBRATION_P2_MODULE_SHA256
            )
        }
    }


def test_exact_historical_p2_replay_contract_is_narrow_and_accepted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        verification,
        "sha256_file",
        lambda path: verification.CURRENT_EVALUATION_SEAM_P2_MODULE_SHA256,
    )
    contract = verification.historical_p2_replay_contract()

    assert verification._validate_historical_p2_replay_contract(
        contract,
        authority=_archived_authority(),
        authority_sha256=verification.HISTORICAL_CALIBRATION_AUTHORITY_SHA256,
    ) == verification.HISTORICAL_CALIBRATION_P2_MODULE_SHA256
    assert contract["replay_only"] is True
    assert contract["permits_artifact_creation"] is False
    assert contract["permits_apply"] is False


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("schema_version", "forged"),
        ("policy_id", "any_module_transition"),
        ("reason", "unspecified"),
        ("calibration_authority_sha256", "0" * 64),
        ("archived_p2_module_sha256", "1" * 64),
        ("current_p2_module_sha256", "2" * 64),
        ("replay_only", False),
        ("permits_artifact_creation", True),
        ("permits_apply", True),
    ],
)
def test_historical_p2_replay_contract_rejects_every_policy_drift(
    monkeypatch: pytest.MonkeyPatch, field: str, value: object
) -> None:
    monkeypatch.setattr(
        verification,
        "sha256_file",
        lambda path: verification.CURRENT_EVALUATION_SEAM_P2_MODULE_SHA256,
    )
    contract = verification.historical_p2_replay_contract()
    contract[field] = value

    with pytest.raises(AcousticVerificationError, match="contract is invalid"):
        verification._validate_historical_p2_replay_contract(
            contract,
            authority=_archived_authority(),
            authority_sha256=verification.HISTORICAL_CALIBRATION_AUTHORITY_SHA256,
        )


def test_historical_p2_replay_contract_rejects_extra_fields(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        verification,
        "sha256_file",
        lambda path: verification.CURRENT_EVALUATION_SEAM_P2_MODULE_SHA256,
    )
    contract = verification.historical_p2_replay_contract()
    contract["compatibility_range"] = "*"

    with pytest.raises(AcousticVerificationError, match="contract is invalid"):
        verification._validate_historical_p2_replay_contract(
            contract,
            authority=_archived_authority(),
            authority_sha256=verification.HISTORICAL_CALIBRATION_AUTHORITY_SHA256,
        )


@pytest.mark.parametrize("binding", ["authority", "stored_module", "live_module"])
def test_historical_p2_replay_contract_rejects_binding_drift(
    monkeypatch: pytest.MonkeyPatch, binding: str
) -> None:
    live_sha = verification.CURRENT_EVALUATION_SEAM_P2_MODULE_SHA256
    authority_sha = verification.HISTORICAL_CALIBRATION_AUTHORITY_SHA256
    authority = _archived_authority()
    if binding == "authority":
        authority_sha = "3" * 64
    elif binding == "stored_module":
        authority["preparation_contract"]["p2_module_sha256"] = "4" * 64
    else:
        live_sha = "5" * 64
    monkeypatch.setattr(verification, "sha256_file", lambda path: live_sha)

    with pytest.raises(AcousticVerificationError, match="binding is invalid"):
        verification._validate_historical_p2_replay_contract(
            verification.historical_p2_replay_contract(),
            authority=authority,
            authority_sha256=authority_sha,
        )


@pytest.mark.parametrize(
    "missing_stage",
    ["split_reveal", "preparation", "window_selection", "score_matrix"],
)
def test_historical_replay_requires_every_named_private_stage(
    tmp_path: Path, missing_stage: str
) -> None:
    root = tmp_path / "runtime"
    stage_root = (
        root
        / "calibration-stages"
        / verification.HISTORICAL_CALIBRATION_AUTHORITY_SHA256
    )
    stage_root.mkdir(parents=True, mode=0o700)
    root.chmod(0o700)
    (root / "calibration-stages").chmod(0o700)
    stage_root.chmod(0o700)
    names = {
        "split_reveal": "split-reveal.json",
        "preparation": "preparation.json",
        "window_selection": "window-selection.json",
        "score_matrix": "score-matrix.json",
    }
    for stage, name in names.items():
        if stage == missing_stage:
            continue
        path = stage_root / name
        path.write_text("{}\n", encoding="utf-8")
        path.chmod(0o600)

    with pytest.raises(AcousticVerificationError, match="artifact is unavailable"):
        verification._require_historical_replay_artifacts(
            runtime_root=root,
            authority_sha256=verification.HISTORICAL_CALIBRATION_AUTHORITY_SHA256,
            stages=tuple(names),
        )


def test_historical_contract_is_not_exposed_by_calibration_write_paths() -> None:
    for function in (
        verification.build_calibration_apply_authority,
        verification.reveal_calibration_split,
        verification.prepare_calibration_split,
        verification.select_calibration_windows,
        verification.apply_calibration_scores,
        verification.apply_calibration_thresholds,
    ):
        assert "p2_replay_contract" not in inspect.signature(function).parameters


def test_full_authority_validation_stays_strict_except_for_exact_p2_field(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    authority = {
        **_archived_authority(),
        "authorized_at": "2026-07-31T12:00:00Z",
        "unchanged_binding": "exact",
    }

    def payload(*, authorized_at: str, p2_module_sha256=None, **kwargs):
        return {
            **authority,
            "authorized_at": authorized_at,
            "preparation_contract": {
                "p2_module_sha256": (
                    p2_module_sha256
                    if p2_module_sha256 is not None
                    else verification.CURRENT_EVALUATION_SEAM_P2_MODULE_SHA256
                )
            },
        }

    monkeypatch.setattr(
        verification, "_calibration_apply_authority_payload", payload
    )
    monkeypatch.setattr(
        verification,
        "sha256_file",
        lambda path: verification.CURRENT_EVALUATION_SEAM_P2_MODULE_SHA256,
    )
    inputs = {
        "development_application": {},
        "development_application_sha256": "6" * 64,
        "development_authority": {},
        "split_metadata": {},
    }

    with pytest.raises(AcousticVerificationError, match="authority is invalid"):
        verification._validate_calibration_apply_authority(authority, **inputs)

    assert verification._validate_calibration_apply_authority(
        authority,
        authority_sha256=verification.HISTORICAL_CALIBRATION_AUTHORITY_SHA256,
        p2_replay_contract=verification.historical_p2_replay_contract(),
        **inputs,
    ) == authority

    forged = {**authority, "unchanged_binding": "drifted"}
    with pytest.raises(AcousticVerificationError, match="authority is invalid"):
        verification._validate_calibration_apply_authority(
            forged,
            authority_sha256=verification.HISTORICAL_CALIBRATION_AUTHORITY_SHA256,
            p2_replay_contract=verification.historical_p2_replay_contract(),
            **inputs,
        )


def test_compatibility_score_replay_fails_before_authority_when_stage_missing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        verification,
        "replay_calibration_apply_authority",
        lambda *args, **kwargs: pytest.fail("authority replay entered"),
    )
    with pytest.raises(AcousticVerificationError, match="artifact is unavailable"):
        verification.replay_calibration_score_matrix(
            verification.HISTORICAL_CALIBRATION_AUTHORITY_SHA256,
            runtime_root=tmp_path / "runtime",
            p3_runtime_root=tmp_path / "p3",
            p2_replay_contract=verification.historical_p2_replay_contract(),
        )


def test_historical_stage_replay_uses_no_immutable_writer(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    root = tmp_path / "runtime"
    authority_sha = verification.HISTORICAL_CALIBRATION_AUTHORITY_SHA256
    stage_root = root / "calibration-stages" / authority_sha
    stage_root.mkdir(parents=True, mode=0o700)
    for directory in (root, root / "calibration-stages", stage_root):
        directory.chmod(0o700)
    record = {
        "recording_id": "recording-1",
        "conversation_id": "conversation-1",
        "source_blob": {
            "blob_id": "source-1",
            "sha256": "1" * 64,
            "bytes": 10,
        },
        "transcript_lineage": {"current_artifact_sha256": "2" * 64},
        "operator_gold": {
            "gold_id": "gold-1",
            "speaker_truth": [
                {"speaker_label": "A", "outcome": "person", "subject_id": "s1"}
            ],
        },
        "conditions": {},
    }
    authority = {
        "calibration_record_set_sha256": "3" * 64,
        "calibration_conversation_set_sha256": "4" * 64,
        "preparation_methods": ["method-1"],
    }
    reveal = {
        "schema_version": verification.CALIBRATION_SPLIT_REVEAL_SCHEMA,
        "status": "success",
        "reason_code": None,
        "authority_sha256": authority_sha,
        "intended_split": "calibration",
        "record_set_sha256": "3" * 64,
        "conversation_set_sha256": "4" * 64,
        "record_count": 1,
        "conversation_count": 1,
        "records": [{
            "recording_id": "recording-1",
            "conversation_id": "conversation-1",
            "source_blob_id": "source-1",
            "source_sha256": "1" * 64,
            "source_bytes": 10,
            "transcript_artifact_sha256": "2" * 64,
            "gold_id": "gold-1",
            "speaker_truth": record["operator_gold"]["speaker_truth"],
            "conditions": {},
        }],
        "development_disjoint": True,
        "evaluation_disjoint": True,
        "source_content_disjoint": True,
        "contains_opaque_gold_labels": True,
        "contains_raw_audio": False,
        "contains_transcript_text": False,
        "contains_names_or_emails": False,
        "contains_embeddings_or_vectors": False,
        "will_read_evaluation": False,
        "will_perform_external_write": False,
        "revealed_at": "2026-07-31T12:00:00Z",
    }
    reveal_sha = verification.canonical_artifact_hash(
        {key: value for key, value in reveal.items() if key != "revealed_at"}
    )
    preparation = {
        "schema_version": verification.CALIBRATION_PREPARATION_SCHEMA,
        "status": "success",
        "reason_code": None,
        "authority_sha256": authority_sha,
        "split_reveal_sha256": reveal_sha,
        "intended_split": "calibration",
        "record_count": 1,
        "method_attempts": 1,
        "method_successes": 1,
        "units": [{
            "recording_id": "recording-1",
            "conversation_id": "conversation-1",
            "source_sha256": "1" * 64,
            "methods": [{"method_id": "method-1"}],
        }],
        "did_run_p1_p2": True,
        "did_read_calibration_audio": True,
        "did_run_biometrics": False,
        "did_read_evaluation": False,
        "did_perform_external_write": False,
        "contains_raw_audio": False,
        "contains_transcript_text": False,
        "contains_embeddings_or_vectors": False,
        "prepared_at": "2026-07-31T12:01:00Z",
    }
    preparation_sha = verification._calibration_stage_identity(
        preparation, "prepared_at"
    )
    selection = {
        "schema_version": verification.CALIBRATION_WINDOW_SELECTION_SCHEMA,
        "status": "success",
        "reason_code": None,
        "authority_sha256": authority_sha,
        "split_reveal_sha256": reveal_sha,
        "preparation_sha256": preparation_sha,
        "intended_split": "calibration",
        "maximum_windows_per_speaker_per_conversation": 3,
        "window_count": 1,
        "windows": [{
            "window_id": "window-1",
            "recording_id": "recording-1",
            "conversation_id": "conversation-1",
            "source_sha256": "1" * 64,
            "transcript_artifact_sha256": "2" * 64,
            "start_seconds": 0.0,
            "end_seconds": 1.0,
        }],
        "did_read_calibration_gold": True,
        "did_run_biometrics": False,
        "did_read_evaluation": False,
        "did_perform_external_write": False,
        "contains_opaque_gold_labels": True,
        "contains_raw_audio": False,
        "contains_transcript_text": False,
        "contains_embeddings_or_vectors": False,
        "selected_at": "2026-07-31T12:02:00Z",
    }
    for name, value in (
        ("split-reveal.json", reveal),
        ("preparation.json", preparation),
        ("window-selection.json", selection),
    ):
        path = stage_root / name
        path.write_text(json.dumps(value), encoding="utf-8")
        path.chmod(0o600)
    monkeypatch.setattr(
        verification, "_calibration_records_after_authority", lambda *a, **k: [record]
    )
    monkeypatch.setattr(
        verification,
        "write_immutable_private_json",
        lambda *a, **k: pytest.fail("immutable writer entered"),
    )
    monkeypatch.setattr(
        verification,
        "ensure_private_tree",
        lambda *a, **k: pytest.fail("directory writer entered"),
    )

    replay = verification._replay_historical_calibration_window_selection(
        authority,
        authority_sha,
        runtime_root=root,
        parent_corpus_manifest_path=tmp_path / "unused.json",
    )
    assert replay["selection_replay_mode"] == "structural_without_writers"
