from __future__ import annotations

import json
from pathlib import Path

import pytest

import acoustic_generation3_recalibration as recalibration


def _dimensions(prefix: str) -> dict[str, set[str]]:
    return {
        "source_sha256": {prefix + "-source"},
        "recording_identity_sha256": {prefix + "-recording"},
        "conversation_identity_sha256": {prefix + "-conversation"},
        "derivative_identity_sha256": {prefix + "-derivative"},
    }


def _dimension_authority(values: dict[str, set[str]]) -> dict[str, object]:
    return {
        key: {
            "count": len(items),
            "set_sha256": recalibration._canonical_hash(sorted(items)),
        }
        for key, items in values.items()
    }


def _patch_context(monkeypatch: pytest.MonkeyPatch) -> None:
    calibration = _dimensions("calibration")
    evaluation = _dimensions("evaluation")
    historical = {
        "authority_sha256": "a" * 64,
        "application_sha256": "b" * 64,
        "authority_file_sha256": "c" * 64,
        "application_file_sha256": "d" * 64,
        "window_selection_sha256": "e" * 64,
        "preparation_sha256": "f" * 64,
        "window_count": 22,
        "calibration_dimensions": _dimension_authority(calibration),
        "prior_evaluation_dimensions": _dimension_authority(evaluation),
        "corpora": [],
        "score_methods": list(recalibration.METHOD_IDS),
        "threshold_policy": {
            "minimum_genuine_trials_per_unit": 9,
            "minimum_impostor_trials_per_unit": 35,
            "temperature_candidates": [0.05, 0.1],
        },
        "metric_policy": {"condition_slices": ["channel"]},
        "selection_objective": ["minimum_brier_score"],
    }
    monkeypatch.setattr(
        recalibration,
        "_historical_context",
        lambda **_kwargs: (
            historical,
            {
                "selection": {
                    "windows": [
                        {
                            "window_id": f"window-{index}",
                            "subject_id": (
                                f"subject-{index % 2}" if index < 9 else f"open-{index}"
                            ),
                        }
                        for index in range(22)
                    ]
                },
                "preparation": {},
            },
            calibration,
            evaluation,
        ),
    )
    training = _dimensions("training")
    monkeypatch.setattr(
        recalibration,
        "_training_dimensions",
        lambda: (
            {
                "intake_id": "training-test",
                "manifest_sha256": "1" * 64,
                "dimensions": _dimension_authority(training),
            },
            training,
        ),
    )
    generation3 = _dimensions("generation3")
    monkeypatch.setattr(
        recalibration,
        "_generation3_context",
        lambda: (
            {
                "authority_id": "cohort-test",
                "manifest_sha256": "2" * 64,
                "membership_sha256": "3" * 64,
                "dimensions": _dimension_authority(generation3),
            },
            generation3,
        ),
    )
    monkeypatch.setattr(
        recalibration,
        "_gold_receipt",
        lambda: {
            "gold_id": "gold-test",
            "receipt_sha256": "4" * 64,
            "manifest_sha256": "5" * 64,
            "gold_body_sha256": "6" * 64,
            "membership_sha256": "3" * 64,
            "gold_label_count": 28,
        },
    )
    profiles = []
    for candidate_index, candidate_id in enumerate(recalibration.CANDIDATE_IDS):
        for subject_index in range(2):
            profiles.append(
                {
                    "profile_id": f"profile-{candidate_index}-{subject_index}",
                    "descendant_id": f"descendant-{candidate_index}-{subject_index}",
                    "person_ref_id": f"subject-{subject_index}",
                    "p3_profile_id": f"p3-{subject_index}",
                    "generation_id": f"generation-{subject_index}",
                    "generation_sha256": f"{subject_index + 1:064x}",
                    "candidate_id": candidate_id,
                    "model_revision": f"revision-{candidate_index}",
                    "preprocessing": {"method_id": "no_enhancement"},
                    "artifact_sha256": f"{candidate_index + subject_index + 10:064x}",
                    "profile_manifest_sha256": f"{candidate_index + subject_index + 20:064x}",
                    "state_receipt_sha256": f"{candidate_index + subject_index + 30:064x}",
                    "vector_dimension": 192,
                    "window_count": 10,
                    "session_count": 4,
                }
            )
    monkeypatch.setattr(
        recalibration,
        "_active_profiles",
        lambda **_kwargs: (
            profiles,
            {
                "profile_count": 6,
                "subject_count": 2,
                "candidate_count": 3,
                "profile_set_sha256": recalibration._canonical_hash(profiles),
                "model_assets": {},
                "model_asset_set_sha256": "7" * 64,
            },
        ),
    )


def test_recalibration_preview_freezes_exact_prescore_inventory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_context(monkeypatch)
    preview = recalibration.preview_generation3_recalibration()
    assert preview["unit_count"] == 9
    assert preview["active_profile_authority"]["profile_count"] == 6
    assert preview["historical_calibration"]["window_count"] == 22
    assert preview["expected_trials_per_unit"] == 44
    assert preview["expected_genuine_trials_per_unit"] == 9
    assert preview["expected_impostor_trials_per_unit"] == 35
    assert preview["expected_open_set_trials_per_unit"] == 26
    assert preview["abstention_margin"] == 0.0
    assert preview["did_load_or_run_models"] is False
    assert preview["did_score_trials"] is False
    assert preview["did_select_thresholds"] is False
    assert not any(
        value
        for group in preview["disjointness_overlap_counts"].values()
        for value in group.values()
    )
    assert all(value is False for value in preview["action_vector"].values())

    portable = recalibration.portable_recalibration_projection(preview)
    serialized = json.dumps(portable, sort_keys=True)
    assert "profile-0-0" not in serialized
    assert "subject-0" not in serialized
    assert "private_historical" not in serialized
    assert portable["contains_private_membership"] is False
    assert portable["contains_profile_or_subject_ids"] is False
    assert portable["contains_paths"] is False
    assert portable["contains_biometric_scores"] is False


def test_recalibration_preview_rejects_training_or_generation3_overlap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_context(monkeypatch)
    with pytest.raises(
        recalibration.Generation3RecalibrationError,
        match="exact prior evaluation corpus inventory",
    ):
        recalibration.preview_generation3_recalibration(corpus_manifest_paths=[])

    overlapping = _dimensions("calibration")
    monkeypatch.setattr(
        recalibration,
        "_training_dimensions",
        lambda: (
            {
                "intake_id": "training-overlap",
                "manifest_sha256": "8" * 64,
                "dimensions": _dimension_authority(overlapping),
            },
            overlapping,
        ),
    )
    with pytest.raises(
        recalibration.Generation3RecalibrationError,
        match="overlaps training or Generation-3 evaluation",
    ):
        recalibration.preview_generation3_recalibration()


def test_recalibration_preview_rejects_false_denominators_and_profile_shape(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_context(monkeypatch)
    profiles, authority = recalibration._active_profiles(
        calibration_root=Path("."), p3_runtime_root=Path(".")
    )
    malformed = [dict(item) for item in profiles]
    malformed[-1]["person_ref_id"] = malformed[-2]["person_ref_id"]
    malformed_authority = {
        **authority,
        "profile_set_sha256": recalibration._canonical_hash(malformed),
    }
    monkeypatch.setattr(
        recalibration,
        "_active_profiles",
        lambda **_kwargs: (malformed, malformed_authority),
    )
    with pytest.raises(
        recalibration.Generation3RecalibrationError,
        match="Cartesian lineage",
    ):
        recalibration.preview_generation3_recalibration()

    _patch_context(monkeypatch)
    historical, private, calibration, evaluation = recalibration._historical_context(
        calibration_root=Path("."), corpus_manifest_paths=[]
    )
    private["selection"]["windows"] = [
        {"window_id": f"window-{index}", "subject_id": "subject-0"}
        for index in range(22)
    ]
    monkeypatch.setattr(
        recalibration,
        "_historical_context",
        lambda **_kwargs: (historical, private, calibration, evaluation),
    )
    with pytest.raises(
        recalibration.Generation3RecalibrationError,
        match="denominators changed",
    ):
        recalibration.preview_generation3_recalibration()

    _patch_context(monkeypatch)
    gold_receipt = recalibration._gold_receipt()
    monkeypatch.setattr(
        recalibration,
        "_gold_receipt",
        lambda: {**gold_receipt, "membership_sha256": "9" * 64},
    )
    with pytest.raises(
        recalibration.Generation3RecalibrationError,
        match="gold and cohort membership differ",
    ):
        recalibration.preview_generation3_recalibration()


def test_recalibration_authority_apply_and_replay_are_prescore_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_context(monkeypatch)
    repository = {
        "commit": "a" * 40,
        "module_sha256": {
            name: f"{index + 40:064x}"
            for index, name in enumerate(
                sorted(
                    {
                        "acoustic_generation3_recalibration.py",
                        "acoustic_generation3_gold.py",
                        "acoustic_generation3_authority.py",
                        "acoustic_verification.py",
                        "acoustic_speech_preparation.py",
                        "acoustic_audio_derivatives.py",
                        "acoustic_training_expansion.py",
                        "acoustic_biometric_references.py",
                    }
                )
            )
        },
        "clean": True,
        "upstream_ahead": 0,
        "upstream_behind": 0,
    }
    monkeypatch.setattr(recalibration, "_repository_authority", lambda: repository)
    monkeypatch.setattr(
        recalibration, "_validate_repository_authority", lambda value: dict(value)
    )
    preview = recalibration.preview_generation3_recalibration()
    receipt = recalibration.apply_generation3_recalibration_authority(
        preview,
        expected_preview_content_sha256=preview["content_sha256"],
        runtime_root=tmp_path,
    )
    assert receipt["status"] == "applied_recalibration_authority_scores_not_run"
    assert receipt["action_vector"]["freeze_recalibration_authority"] is True
    assert receipt["action_vector"]["run_calibration_models"] is True
    assert receipt["action_vector"]["freeze_thresholds_and_temperatures"] is False
    assert receipt["action_vector"]["reveal_evaluation"] is False
    assert Path(receipt["private_manifest_path"]).stat().st_mode & 0o777 == 0o600

    replay = recalibration.replay_generation3_recalibration_authority(
        Path(receipt["private_manifest_path"]), runtime_root=tmp_path
    )
    assert replay["idempotent_replay"] is True
    assert replay["manifest_sha256"] == receipt["manifest_sha256"]

    changed = dict(preview)
    changed["content_sha256"] = "0" * 64
    with pytest.raises(
        recalibration.Generation3RecalibrationError,
        match="stale",
    ):
        recalibration.apply_generation3_recalibration_authority(
            changed,
            expected_preview_content_sha256=preview["content_sha256"],
            runtime_root=tmp_path / "stale",
        )


def test_evaluation_semantic_lineage_requires_private_hash_matched_bytes(
    tmp_path: Path,
) -> None:
    transcript_path = tmp_path / "evaluation.transcript.json"
    transcript_path.write_text(
        json.dumps(
                {
                    "schema_version": 2,
                    "recording_id": "recording-one",
                    "conversation_id": "conversation-one",
                    "duration_seconds": 1.0,
                    "utterances": [
                        {
                            "speaker": "A",
                            "start": 0,
                            "end": 1000,
                            "text": "test",
                        }
                    ],
            }
        ),
        encoding="utf-8",
    )
    transcript_path.chmod(0o600)
    record = {
        "recording_id": "recording-one",
        "conversation_id": "conversation-one",
        "source_blob": {"sha256": "a" * 64},
        "transcript_lineage": {
            "current_artifact_path": str(transcript_path),
            "current_artifact_sha256": recalibration.sha256_file(transcript_path),
        },
    }
    key, identities = recalibration._evaluation_semantic_identity(
        record, transcripts_root=tmp_path
    )
    assert key == ("a" * 64, "recording-one", "conversation-one")
    assert identities is not None
    assert set(identities) == set(recalibration.DIMENSIONS[1:])

    missing = {key: value for key, value in record.items() if key != "transcript_lineage"}
    missing_key, missing_identities = recalibration._evaluation_semantic_identity(
        missing, transcripts_root=tmp_path
    )
    assert missing_identities is None
    with pytest.raises(
        recalibration.Generation3RecalibrationError,
        match="validated semantic lineage",
    ):
        recalibration._require_semantic_coverage({missing_key}, set())

    drifted = {
        **record,
        "transcript_lineage": {
            **record["transcript_lineage"],
            "current_artifact_sha256": "b" * 64,
        },
    }
    with pytest.raises(
        recalibration.Generation3RecalibrationError,
        match="lineage drifted",
    ):
        recalibration._evaluation_semantic_identity(
            drifted, transcripts_root=tmp_path
        )

    transcript_path.chmod(0o644)
    with pytest.raises(
        recalibration.Generation3RecalibrationError,
        match="not private",
    ):
        recalibration._evaluation_semantic_identity(
            record, transcripts_root=tmp_path
        )


def test_current_repository_replay_requires_clean_upstream_even(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        recalibration,
        "_git",
        lambda args: " M private.txt" if args[:2] == ["status", "--porcelain"] else "0\t0",
    )
    with pytest.raises(
        recalibration.Generation3RecalibrationError,
        match="clean and upstream-even",
    ):
        recalibration._require_current_repository_even()

    monkeypatch.setattr(
        recalibration,
        "_git",
        lambda args: "" if args[:2] == ["status", "--porcelain"] else "0\t1",
    )
    with pytest.raises(
        recalibration.Generation3RecalibrationError,
        match="clean and upstream-even",
    ):
        recalibration._require_current_repository_even()
