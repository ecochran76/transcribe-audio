from __future__ import annotations

import json
from pathlib import Path

import pytest

import acoustic_generation3_pre_reveal as pre_reveal


def _context() -> dict[str, object]:
    profiles = []
    thresholds = []
    for candidate_index, candidate_id in enumerate(
        pre_reveal.execution.recalibration.CANDIDATE_IDS
    ):
        for subject_index in range(2):
            profiles.append(
                {
                    "profile_id": f"profile-{candidate_index}-{subject_index}",
                    "person_ref_id": f"subject-{subject_index}",
                    "candidate_id": candidate_id,
                    "model_revision": f"revision-{candidate_index}",
                    "artifact_sha256": f"{candidate_index + subject_index + 1:064x}",
                }
            )
        for method_index, method_id in enumerate(
            pre_reveal.execution.recalibration.METHOD_IDS
        ):
            thresholds.append(
                {
                    "candidate_id": candidate_id,
                    "method_id": method_id,
                    "status": "success",
                    "reason_code": None,
                    "threshold": -0.1 + candidate_index / 100,
                    "temperature": 0.05 + method_index / 100,
                }
            )
    return {
        "root": "/private/runtime",
        "cohort_manifest_sha256": "1" * 64,
        "cohort": {
            "membership_sha256": "2" * 64,
            "membership": {"conversation_count": 7, "speaker_label_count": 28},
            "window_policy": {
                "minimum_seconds": 0.75,
                "maximum_seconds": 15.0,
                "maximum_windows_per_speaker_per_conversation": 12,
                "preserve_original_timestamps": True,
                "exclude_overlap_and_speaker_change_regions": True,
                "exclude_mixed_or_unknown_gold": True,
                "same_frozen_window_set_for_every_candidate_unit": True,
            },
        },
        "gold_manifest_sha256": "3" * 64,
        "gold": {
            "content_sha256": "4" * 64,
            "membership_sha256": "2" * 64,
            "gold_label_count": 28,
            "known_subject_count": 12,
            "outcome_counts": {
                "enrolled": 10, "open_set": 10, "mixed": 2, "unknown": 6,
            },
            "enrolled_conversation_counts": {
                "subject-secret-a": 3,
                "subject-secret-b": 7,
            },
        },
        "recalibration_manifest_sha256": "5" * 64,
        "recalibration": {
            "profiles": profiles,
            "historical_calibration": {
                "calibration_dimensions": {"source_sha256": {"count": 22}},
                "metric_policy": {"condition_slices": ["channel", "noise"]},
            },
            "active_profile_authority": {
                "profile_set_sha256": "6" * 64,
                "model_asset_set_sha256": "7" * 64,
                "model_assets": {
                    "speechbrain_ecapa_tdnn": {"asset_sha256": "e" * 64},
                    "wespeaker_campplus": {"asset_sha256": "f" * 64},
                    "wespeaker_resnet34": {"asset_sha256": "0" * 64},
                },
            },
            "generation3_gold_commitment": {
                "receipt_sha256": "1" * 64,
                "gold_body_sha256": "2" * 64,
            },
        },
        "execution_authority_sha256": "8" * 64,
        "score_matrix_sha256": "9" * 64,
        "threshold_application_sha256": "a" * 64,
        "threshold_set_sha256": "b" * 64,
        "thresholds": thresholds,
    }


def _repository(*, clean: bool = True) -> dict[str, object]:
    return {
        "commit": "c" * 40,
        "module_sha256": {name: "d" * 64 for name in pre_reveal.MODULE_NAMES},
        "terminal_policy_sha256": "e" * 64,
        "clean": clean,
        "upstream_ahead": 0,
        "upstream_behind": 0,
    }


def test_preview_freezes_complete_generation3_envelope(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(pre_reveal, "_frozen_context", lambda **_kwargs: _context())
    preview = pre_reveal.preview_generation3_pre_reveal()
    assert preview["population_authority"][
        "independent_same_person_subject_session_pair_count"
    ] == 24
    assert preview["population_authority"]["gate_status"] == "pass"
    assert len(preview["profiles"]) == 6
    assert len(preview["candidate_matrix"]) == 9
    assert all(item["abstention_margin"] == 0.0 for item in preview["candidate_matrix"])
    assert preview["condition_policy"]["dimensions"] == list(
        pre_reveal.CONDITION_DIMENSIONS
    )
    assert preview["condition_policy"]["minimum_observed_values_per_dimension"] == 2
    assert preview["condition_policy"]["missing_recordings_allowed"] == 0
    assert preview["window_policy"]["maximum_windows_per_speaker_per_conversation"] == 12
    assert preview["terminal_resolution_policy"]["unit_precedence"] == [
        "stop", "reject", "select", "refine",
    ]
    assert all(value is False for value in preview["action_vector"].values())


def test_portable_projection_excludes_private_ids_thresholds_and_paths(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(pre_reveal, "_frozen_context", lambda **_kwargs: _context())
    preview = pre_reveal.preview_generation3_pre_reveal()
    portable = pre_reveal.portable_pre_reveal_projection(preview)
    serialized = json.dumps(portable, sort_keys=True)
    assert "subject-secret" not in serialized
    assert "profile-0-0" not in serialized
    assert "/private/runtime" not in serialized
    assert '"threshold":' not in serialized
    assert '"temperature":' not in serialized
    assert portable["contains_profile_or_subject_ids"] is False
    assert portable["contains_threshold_or_temperature_values"] is False


def test_population_gate_fails_closed() -> None:
    gold = _context()["gold"]
    gold["enrolled_conversation_counts"] = {"one": 1, "two": 2}
    with pytest.raises(pre_reveal.Generation3PreRevealError, match="population gate"):
        pre_reveal._population_authority(gold)


def test_candidate_matrix_rejects_failed_threshold() -> None:
    context = _context()
    context["thresholds"][0]["status"] = "not_run"
    with pytest.raises(pre_reveal.Generation3PreRevealError, match="not successful"):
        pre_reveal._candidate_matrix(context)


def test_candidate_matrix_rejects_duplicate_or_missing_unit() -> None:
    context = _context()
    context["thresholds"][-1] = dict(context["thresholds"][0])
    with pytest.raises(pre_reveal.Generation3PreRevealError, match="incomplete"):
        pre_reveal._candidate_matrix(context)


def test_apply_and_replay_authorize_only_reveal(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    tmp_path.chmod(0o700)
    monkeypatch.setattr(pre_reveal, "_frozen_context", lambda **_kwargs: _context())
    preview = pre_reveal.preview_generation3_pre_reveal(runtime_root=tmp_path)
    monkeypatch.setattr(pre_reveal, "_repository_authority", _repository)
    monkeypatch.setattr(
        pre_reveal, "_validate_repository_authority", lambda value: dict(value)
    )
    applied = pre_reveal.apply_generation3_pre_reveal(
        preview,
        expected_preview_content_sha256=preview["content_sha256"],
        runtime_root=tmp_path,
    )
    assert applied["action_vector"]["build_pre_reveal_envelope"] is True
    assert applied["action_vector"]["reveal_evaluation"] is True
    assert applied["action_vector"]["run_denominator_preflight"] is False
    assert applied["action_vector"]["prepare_evaluation_audio"] is False
    assert applied["action_vector"]["load_or_run_models"] is False
    assert applied["action_vector"]["score_evaluation_trials"] is False
    assert applied["action_vector"]["mutate_profiles_or_references"] is False
    replayed = pre_reveal.replay_generation3_pre_reveal(
        Path(applied["private_manifest_path"]), runtime_root=tmp_path
    )
    assert replayed["idempotent_replay"] is True
    assert replayed["full_body_match"] is True


def test_apply_rejects_dirty_repository_before_private_write(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    tmp_path.chmod(0o700)
    monkeypatch.setattr(pre_reveal, "_frozen_context", lambda **_kwargs: _context())
    preview = pre_reveal.preview_generation3_pre_reveal(runtime_root=tmp_path)
    monkeypatch.setattr(
        pre_reveal, "_repository_authority", lambda: _repository(clean=False)
    )
    with pytest.raises(
        pre_reveal.Generation3PreRevealError, match="clean upstream-even",
    ):
        pre_reveal.apply_generation3_pre_reveal(
            preview,
            expected_preview_content_sha256=preview["content_sha256"],
            runtime_root=tmp_path,
        )
    assert not (tmp_path / "pre-reveal-authorities").exists()


@pytest.mark.parametrize(
    ("field", "unsafe"),
    [
        ("condition_failure", "continue"),
        ("incomplete_cartesian_or_failed_or_blocked_cell", "refine"),
        ("nonfinite_score_or_required_metric", "ignore"),
    ],
)
def test_terminal_policy_rejects_stop_semantic_drift(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, field: str, unsafe: str,
) -> None:
    policy = json.loads(pre_reveal.DEFAULT_TERMINAL_POLICY.read_text())
    policy[field] = unsafe
    path = tmp_path / "policy.json"
    path.write_text(json.dumps(policy), encoding="utf-8")
    monkeypatch.setattr(pre_reveal, "DEFAULT_TERMINAL_POLICY", path)
    with pytest.raises(
        pre_reveal.Generation3PreRevealError, match="policy is invalid",
    ):
        pre_reveal._terminal_policy()


def test_condition_measurement_algorithms_bind_real_frozen_functions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(pre_reveal, "_frozen_context", lambda **_kwargs: _context())
    preview = pre_reveal.preview_generation3_pre_reveal()
    algorithms = preview["condition_policy"]["measurement_algorithms"]
    assert set(algorithms) == set(pre_reveal.CONDITION_DIMENSIONS)
    assert preview["condition_policy"]["implementation_functions"] == [
        "_conditions", "_aggregate_conditions",
    ]
    assert "_conditions_and_aggregate_conditions" not in json.dumps(algorithms)
    assert callable(pre_reveal.conditions._conditions)
    assert callable(pre_reveal.conditions._aggregate_conditions)
    assert preview["condition_policy"]["measurement_module_sha256"] == (
        pre_reveal.sha256_file(Path(pre_reveal.conditions.__file__).resolve())
    )
