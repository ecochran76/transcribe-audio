from __future__ import annotations

import json
from pathlib import Path

import pytest

import acoustic_generation3_recalibration_execution as execution


def _frozen_preview() -> dict[str, object]:
    profiles = []
    for candidate_index, candidate_id in enumerate(execution.recalibration.CANDIDATE_IDS):
        for subject_index in range(2):
            profiles.append(
                {
                    "profile_id": f"profile-{candidate_index}-{subject_index}",
                    "descendant_id": f"descendant-{candidate_index}-{subject_index}",
                    "person_ref_id": f"subject-{subject_index}",
                    "candidate_id": candidate_id,
                    "model_revision": f"revision-{candidate_index}",
                    "artifact_sha256": f"{candidate_index * 2 + subject_index + 1:064x}",
                }
            )
    units = [
        {"candidate_id": candidate_id, "method_id": method_id}
        for candidate_id in execution.recalibration.CANDIDATE_IDS
        for method_id in execution.recalibration.METHOD_IDS
    ]
    return {
        "historical_calibration": {
            "calibration_dimensions": {"source_sha256": {"count": 22}},
            "window_selection_sha256": "1" * 64,
            "preparation_sha256": "2" * 64,
            "score_methods": list(execution.recalibration.METHOD_IDS),
            "window_count": 22,
            "threshold_policy": {
                "minimum_genuine_trials_per_unit": 9,
                "minimum_impostor_trials_per_unit": 35,
                "temperature_candidates": [0.05, 0.1],
            },
            "metric_policy": {"condition_slices": ["channel"]},
            "selection_objective": ["minimum_brier_score"],
        },
        "active_profile_authority": {
            "profile_count": 6,
            "candidate_count": 3,
            "profile_set_sha256": "3" * 64,
            "model_asset_set_sha256": "4" * 64,
        },
        "profiles": profiles,
        "units": units,
        "unit_count": 9,
        "expected_trials_per_unit": 44,
        "expected_genuine_trials_per_unit": 9,
        "expected_impostor_trials_per_unit": 35,
        "expected_open_set_trials_per_unit": 26,
        "abstention_margin": 0.0,
    }


def _context() -> dict[str, object]:
    return {
        "manifest_path": "/private/manifest.json",
        "manifest_sha256": "5" * 64,
        "authority_id": "private-authority-id",
        "authority_content_sha256": "6" * 64,
        "preview": _frozen_preview(),
    }


def _repository(*, clean: bool = True) -> dict[str, object]:
    return {
        "commit": "a" * 40,
        "module_name": execution.EXECUTOR_MODULE,
        "module_sha256": "b" * 64,
        "clean": clean,
        "upstream_ahead": 0,
        "upstream_behind": 0,
    }


def _windows() -> list[dict[str, object]]:
    return [
        {
            "window_id": f"window-{index}",
            "recording_id": f"recording-{index}",
            "conversation_id": f"conversation-{index}",
            "subject_id": f"subject-{index % 2}" if index < 9 else f"open-{index}",
            "conditions": {"channel": "near" if index % 2 else "far"},
        }
        for index in range(22)
    ]


def _trials(authority_sha: str = "c" * 64) -> list[dict[str, object]]:
    frozen = _frozen_preview()
    profiles = frozen["profiles"]
    trials = []
    for window in _windows():
        for method_id in execution.recalibration.METHOD_IDS:
            probe_sha = execution._canonical_hash(
                {"window": window["window_id"], "method": method_id}
            )
            for profile in profiles:
                expected_match = window["subject_id"] == profile["person_ref_id"]
                score = 0.85 if expected_match else (
                    -0.75 if str(window["subject_id"]).startswith("open-") else -0.25
                )
                score_identity = {
                    "profile_id": profile["profile_id"],
                    "descendant_id": profile["descendant_id"],
                    "artifact_sha256": profile["artifact_sha256"],
                    "candidate_id": profile["candidate_id"],
                    "model_revision": profile["model_revision"],
                    "probe_sha256": probe_sha,
                    "score": score,
                }
                score_trial_id = "verification-trial-" + execution.verification.canonical_artifact_hash(
                    score_identity
                )[:24]
                identity = {
                    "execution_authority_sha256": authority_sha,
                    "window_id": window["window_id"],
                    "method_id": method_id,
                    "profile_id": profile["profile_id"],
                    "score_trial_id": score_trial_id,
                }
                trials.append(
                    {
                        "trial_id": "generation3-calibration-trial-"
                        + execution._canonical_hash(identity)[:24],
                        "status": "success",
                        "reason_code": None,
                        "window_id": window["window_id"],
                        "recording_id": window["recording_id"],
                        "conversation_id": window["conversation_id"],
                        "probe_subject_id": window["subject_id"],
                        "profile_person_ref_id": profile["person_ref_id"],
                        "expected_match": expected_match,
                        "open_set_probe": str(window["subject_id"]).startswith("open-"),
                        "method_id": method_id,
                        "profile_id": profile["profile_id"],
                        "descendant_id": profile["descendant_id"],
                        "candidate_id": profile["candidate_id"],
                        "model_revision": profile["model_revision"],
                        "probe_sha256": probe_sha,
                        "score_trial_id": score_trial_id,
                        "score": score,
                        "conditions": window["conditions"],
                        "p4_state_verified_before_and_after": True,
                        "p3_eligibility_verified_before_and_after": True,
                        "contains_raw_biometric_values": False,
                    }
                )
    return sorted(trials, key=lambda item: str(item["trial_id"]))


def _matrix(authority_sha: str = "c" * 64) -> dict[str, object]:
    return {
        "schema_version": execution.SCORE_MATRIX_SCHEMA,
        "status": "success",
        "reason_code": None,
        "execution_authority_sha256": authority_sha,
        "recalibration_manifest_sha256": "5" * 64,
        "preparation_sha256": "2" * 64,
        "window_selection_sha256": "1" * 64,
        "logical_trial_count": 396,
        "trials": _trials(authority_sha),
        "did_run_biometrics": True,
        "did_select_thresholds": False,
        "did_read_generation3_gold_or_audio": False,
        "did_mutate_profiles_or_references": False,
        "contains_biometric_scores": True,
        "contains_raw_audio": False,
        "contains_transcript_text": False,
        "contains_embeddings_or_vectors": False,
        "contains_raw_biometric_values": False,
    }


def test_preview_is_aggregate_only_and_all_actions_are_false(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(execution, "_authority_context", lambda **_kwargs: _context())
    monkeypatch.setattr(execution, "_repository_authority", _repository)
    preview = execution.preview_generation3_recalibration_execution()
    serialized = json.dumps(preview, sort_keys=True)
    assert preview["expected_trial_count"] == 396
    assert preview["abstention_margin_is_zero"] is True
    assert all(value is False for value in preview["action_vector"].values())
    assert "profile-0-0" not in serialized
    assert "subject-0" not in serialized
    assert "/private/manifest.json" not in serialized
    assert preview["contains_biometric_scores"] is False


def test_dirty_or_unpushed_executor_fails_before_adapter_load(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(execution, "_authority_context", lambda **_kwargs: _context())
    monkeypatch.setattr(
        execution, "_repository_authority", lambda: _repository(clean=False)
    )
    loaded = False

    def adapters() -> dict[str, object]:
        nonlocal loaded
        loaded = True
        return {}

    monkeypatch.setattr(execution.verification, "adapter_registry", adapters)
    preview = execution.preview_generation3_recalibration_execution()
    with pytest.raises(
        execution.Generation3RecalibrationExecutionError,
        match="repository authority",
    ):
        execution.apply_generation3_recalibration_scores(
            preview,
            expected_preview_content_sha256=preview["content_sha256"],
            runtime_root=Path("/tmp/not-used"),
        )
    assert loaded is False


def test_exact_score_matrix_and_denominators_replay() -> None:
    matrix = _matrix()
    execution._validate_score_matrix(
        matrix,
        authority_sha="c" * 64,
        context=_context(),
        selection={"windows": _windows()},
    )
    execution._validate_unit_denominators(matrix["trials"], _frozen_preview())


def test_score_matrix_rejects_duplicate_and_nonfinite_trial() -> None:
    duplicate = _matrix()
    duplicate["trials"][-1] = dict(duplicate["trials"][0])
    with pytest.raises(
        execution.Generation3RecalibrationExecutionError,
        match="replay is invalid|denominators|binding",
    ):
        execution._validate_score_matrix(
            duplicate,
            authority_sha="c" * 64,
            context=_context(),
            selection={"windows": _windows()},
        )
    nonfinite = _matrix()
    nonfinite["trials"][0]["score"] = float("nan")
    with pytest.raises(
        execution.Generation3RecalibrationExecutionError,
        match="binding",
    ):
        execution._validate_score_matrix(
            nonfinite,
            authority_sha="c" * 64,
            context=_context(),
            selection={"windows": _windows()},
        )


def test_threshold_freeze_is_deterministic_for_all_nine_units() -> None:
    frozen = _frozen_preview()
    matrix = _matrix()
    first = execution._threshold_results(frozen, matrix)
    second = execution._threshold_results(frozen, matrix)
    assert first == second
    assert len(first) == 9
    assert all(item["status"] == "success" for item in first)
    assert all(item["threshold"] is not None for item in first)
    assert all(item["temperature"] in {0.05, 0.1} for item in first)
    receipt = execution._threshold_receipt(
        score_receipt={
            "execution_authority_sha256": "c" * 64,
            "score_matrix_sha256": "d" * 64,
            "action_vector": {
                "run_calibration_models": True,
                "persist_private_score_matrix": True,
                "freeze_thresholds_and_temperatures": True,
                "build_pre_reveal_envelope": False,
                "reveal_evaluation": False,
                "mutate_profiles_or_references": False,
                "enable_default_integration": False,
                "run_historical_reprocessing": False,
            },
        },
        application_sha="e" * 64,
        results=first,
    )
    assert receipt["threshold_unit_count"] == 9
    assert receipt["abstention_margin_is_zero"] is True
    assert receipt["action_vector"]["build_pre_reveal_envelope"] is True
    assert receipt["action_vector"]["reveal_evaluation"] is False
    assert receipt["contains_frozen_threshold_values"] is False


def test_threshold_freeze_rejects_missing_denominator() -> None:
    matrix = _matrix()
    matrix["trials"] = matrix["trials"][:-1]
    with pytest.raises(
        execution.Generation3RecalibrationExecutionError,
        match="incomplete, nonfinite, or fallback",
    ):
        execution._threshold_results(_frozen_preview(), matrix)


def test_apply_replay_and_threshold_freeze_end_to_end_with_test_adapters(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    tmp_path.chmod(0o700)
    context = _context()
    frozen = context["preview"]
    profiles = {item["profile_id"]: item for item in frozen["profiles"]}
    monkeypatch.setattr(execution, "_authority_context", lambda **_kwargs: context)
    monkeypatch.setattr(execution, "_repository_authority", _repository)
    monkeypatch.setattr(
        execution, "_validate_repository_authority", lambda value: dict(value)
    )
    monkeypatch.setattr(
        execution,
        "_private_historical_context",
        lambda **_kwargs: ({"windows": _windows()}, {"units": []}),
    )
    monkeypatch.setattr(
        execution.verification,
        "_calibration_pcm_window",
        lambda _preparation, window, method_id: (
            float(str(window["window_id"]).split("-")[-1]),
            float(execution.recalibration.METHOD_IDS.index(method_id)),
        ),
    )

    class Adapter:
        def __init__(self, candidate_id: str, revision_sha: str) -> None:
            self.candidate_id = candidate_id
            self.revision_sha = revision_sha

    adapters = {
        candidate_id: Adapter(candidate_id, f"revision-{candidate_index}")
        for candidate_index, candidate_id in enumerate(
            execution.recalibration.CANDIDATE_IDS
        )
    }

    def score_profile(
        profile_id: str, *, adapter: Adapter, probe_samples: tuple[float, ...],
        **_kwargs: object,
    ) -> dict[str, object]:
        profile = profiles[profile_id]
        window_index = int(probe_samples[0])
        subject_id = (
            f"subject-{window_index % 2}" if window_index < 9 else f"open-{window_index}"
        )
        score = 0.85 if subject_id == profile["person_ref_id"] else (
            -0.75 if subject_id.startswith("open-") else -0.25
        )
        probe_sha = execution._canonical_hash(list(probe_samples))
        identity = {
            "profile_id": profile_id,
            "descendant_id": profile["descendant_id"],
            "artifact_sha256": profile["artifact_sha256"],
            "candidate_id": adapter.candidate_id,
            "model_revision": adapter.revision_sha,
            "probe_sha256": probe_sha,
            "score": score,
        }
        return {
            "status": "success",
            "trial_id": "verification-trial-"
            + execution.verification.canonical_artifact_hash(identity)[:24],
            "probe_sha256": probe_sha,
            "score": score,
        }

    monkeypatch.setattr(execution.verification, "score_profile", score_profile)
    preview = execution.preview_generation3_recalibration_execution(
        runtime_root=tmp_path
    )
    applied = execution.apply_generation3_recalibration_scores(
        preview,
        expected_preview_content_sha256=preview["content_sha256"],
        runtime_root=tmp_path,
        adapters=adapters,
        test_mode=True,
    )
    assert applied["logical_trial_count"] == 396
    assert applied["idempotent_replay"] is False
    authority_sha = applied["execution_authority_sha256"]
    replayed = execution.replay_generation3_recalibration_scores(
        authority_sha, runtime_root=tmp_path
    )
    assert replayed["idempotent_replay"] is True
    assert replayed["score_matrix_sha256"] == applied["score_matrix_sha256"]
    frozen_receipt = execution.freeze_generation3_recalibration_thresholds(
        authority_sha, runtime_root=tmp_path
    )
    assert frozen_receipt["threshold_unit_count"] == 9
    assert frozen_receipt["action_vector"]["build_pre_reveal_envelope"] is True
    threshold_replay = execution.replay_generation3_recalibration_thresholds(
        authority_sha, runtime_root=tmp_path
    )
    assert threshold_replay["idempotent_replay"] is True
    assert (
        threshold_replay["threshold_application_sha256"]
        == frozen_receipt["threshold_application_sha256"]
    )
