from __future__ import annotations

import json
from pathlib import Path

import pytest

import acoustic_generation3_evaluation as evaluation
from acoustic_audio_derivatives import AudioDerivativeError


def _parent() -> dict[str, object]:
    profiles = [
        {"profile_id": f"profile-{candidate}-{subject}", "person_ref_id": f"subject-{subject}"}
        for candidate in range(3) for subject in range(2)
    ]
    matrix = [
        {"candidate_id": f"candidate-{candidate}", "method_id": f"method-{method}"}
        for candidate in range(3) for method in range(3)
    ]
    minimum = {
        "genuine_trials_per_model_method_unit": 20,
        "impostor_trials_per_model_method_unit": 100,
        "open_set_trials_per_model_method_unit": 20,
    }
    return {
        "manifest_path": "/private/parent.json",
        "manifest_sha256": "1" * 64,
        "authority_id": "generation3-pre-reveal-test",
        "content_sha256": "2" * 64,
        "preview": {
            "cohort_authority": {
                "manifest_sha256": "3" * 64,
                "membership_sha256": "4" * 64,
                "conversation_count": 7,
            },
            "gold_authority": {"manifest_sha256": "5" * 64, "gold_label_count": 28},
            "recalibration_authority": {"profile_set_sha256": "6" * 64},
            "window_policy": {"maximum_windows_per_speaker_per_conversation": 12},
            "terminal_decision_policy": {"minimum_evidence": minimum},
            "profiles": profiles,
            "candidate_matrix": matrix,
        },
    }


def _gold(*, enrolled: int = 10, open_set: int = 10) -> dict[str, object]:
    values = []
    for index in range(enrolled):
        values.append({"outcome": "enrolled", "subject_id": f"subject-{index % 2}"})
    for index in range(open_set):
        values.append({"outcome": "open_set", "subject_id": f"open-{index}"})
    while len(values) < 28:
        values.append({"outcome": "mixed" if len(values) % 2 else "unknown", "subject_id": None})
    return {"gold": values}


def _repository(*, clean: bool = True) -> dict[str, object]:
    return {
        "commit": "a" * 40,
        "module_sha256": {name: "b" * 64 for name in evaluation.MODULE_NAMES},
        "clean": clean,
        "upstream_ahead": 0,
        "upstream_behind": 0,
    }


def test_reveal_preview_does_not_read_gold(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(evaluation, "_parent_context", lambda _root: _parent())
    monkeypatch.setattr(evaluation, "_repository_authority", _repository)
    monkeypatch.setattr(
        evaluation, "_gold_preview",
        lambda *_args, **_kwargs: pytest.fail("preview read private gold"),
    )
    preview = evaluation.preview_generation3_reveal()
    assert preview["did_read_private_gold"] is False
    assert all(value is False for value in preview["action_vector"].values())
    assert preview["candidate_unit_count"] == 9
    assert preview["maximum_windows_per_speaker_per_conversation"] == 12


def test_real_parent_check_does_not_rebuild_gold_context(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    tmp_path.chmod(0o700)
    parent = _parent()
    preview = parent["preview"]
    repository = _repository()
    core = {
        "schema_version": "test-parent",
        "preview": preview,
        "repository_authority": repository,
    }
    content_sha = evaluation._canonical_hash(core)
    authority_id = f"generation3-pre-reveal-{content_sha[:24]}"
    directory = tmp_path / "pre-reveal-authorities" / authority_id
    directory.mkdir(parents=True, mode=0o700)
    (tmp_path / "pre-reveal-authorities").chmod(0o700)
    directory.chmod(0o700)
    manifest_path = directory / "private-manifest.json"
    manifest = {**core, "authority_id": authority_id, "content_sha256": content_sha}
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    manifest_path.chmod(0o600)
    receipt = {
        "action_vector": {
            "reveal_evaluation": True,
            "run_denominator_preflight": False,
            "prepare_evaluation_audio": False,
            "load_or_run_models": False,
        }
    }
    receipt_path = directory / "receipt.json"
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    receipt_path.chmod(0o600)
    monkeypatch.setattr(
        evaluation.pre_reveal, "_validate_repository_authority",
        lambda _value: repository,
    )
    monkeypatch.setattr(
        evaluation.pre_reveal, "_manifest_core",
        lambda _preview, _repository: core,
    )
    monkeypatch.setattr(
        evaluation.pre_reveal, "_receipt",
        lambda _preview, _authority_id, _manifest_sha: receipt,
    )
    monkeypatch.setattr(
        evaluation.pre_reveal, "replay_generation3_pre_reveal",
        lambda *_args, **_kwargs: pytest.fail("full parent replay opened gold"),
    )
    monkeypatch.setattr(
        evaluation.pre_reveal, "_frozen_context",
        lambda *_args, **_kwargs: pytest.fail("parent check rebuilt gold context"),
    )
    context = evaluation._parent_context(tmp_path)
    assert context["authority_id"] == authority_id
    manifest_path.chmod(0o644)
    with pytest.raises(AudioDerivativeError, match="0600"):
        evaluation._parent_context(tmp_path)


def test_denominator_preflight_passes_exact_structural_maxima() -> None:
    result = evaluation._denominator_preflight(_parent(), _gold(), "c" * 64)
    assert result["status"] == "pass"
    assert result["unit_count"] == 9
    assert all(item["maximum_genuine_trials"] == 120 for item in result["units"])
    assert all(item["maximum_impostor_trials"] == 120 for item in result["units"])
    assert all(item["maximum_open_set_trials"] == 240 for item in result["units"])
    assert result["did_read_audio"] is False
    assert result["did_load_or_run_models"] is False


def test_denominator_preflight_rejects_revealed_population_drift() -> None:
    with pytest.raises(evaluation.Generation3EvaluationError, match="population drifted"):
        evaluation._denominator_preflight(_parent(), _gold(enrolled=9), "c" * 64)


def test_pass_receipt_authorizes_only_prediction_blind_preparation() -> None:
    parent = _parent()
    repository = _repository()
    core = evaluation._preview_core(parent, repository)
    content = evaluation._canonical_hash(core)
    preview = {
        **core,
        "preview_id": f"generation3-reveal-preview-{content[:24]}",
        "content_sha256": content,
    }
    preflight = evaluation._denominator_preflight(parent, _gold(), "c" * 64)
    receipt = evaluation._receipt(preview, "c" * 64, "d" * 64, preflight)
    assert receipt["action_vector"]["run_prediction_blind_p1_p2"] is True
    assert receipt["action_vector"]["prepare_evaluation_audio"] is False
    assert receipt["action_vector"]["record_terminal_stop"] is False
    assert receipt["action_vector"]["measure_conditions"] is False
    assert receipt["action_vector"]["freeze_evaluation_windows"] is False
    assert receipt["action_vector"]["construct_exact_trial_child"] is False
    assert receipt["action_vector"]["load_or_run_models"] is False
    assert receipt["action_vector"]["make_terminal_decision"] is False


def test_stop_receipt_authorizes_terminal_stop_only() -> None:
    parent = _parent()
    parent["preview"]["terminal_decision_policy"]["minimum_evidence"][
        "impostor_trials_per_model_method_unit"
    ] = 121
    core = evaluation._preview_core(parent, _repository())
    content = evaluation._canonical_hash(core)
    preview = {
        **core,
        "preview_id": f"generation3-reveal-preview-{content[:24]}",
        "content_sha256": content,
    }
    preflight = evaluation._denominator_preflight(parent, _gold(), "c" * 64)
    assert preflight["status"] == "stop"
    receipt = evaluation._receipt(preview, "c" * 64, "d" * 64, preflight)
    assert receipt["action_vector"]["record_terminal_stop"] is True
    assert receipt["action_vector"]["make_terminal_decision"] is False
    assert receipt["action_vector"]["prepare_evaluation_audio"] is False
    assert receipt["action_vector"]["run_prediction_blind_p1_p2"] is False
    assert receipt["action_vector"]["load_or_run_models"] is False


def test_apply_replay_writes_authority_before_gold_read(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    tmp_path.chmod(0o700)
    parent = _parent()
    monkeypatch.setattr(evaluation, "_parent_context", lambda _root: parent)
    monkeypatch.setattr(evaluation, "_repository_authority", _repository)
    monkeypatch.setattr(
        evaluation, "_validate_repository_authority", lambda value: dict(value)
    )

    def gold(_parent: object, root: Path) -> dict[str, object]:
        authority_paths = list(root.glob("evaluation-reveals/*/reveal-authority.json"))
        assert len(authority_paths) == 1
        return _gold()

    monkeypatch.setattr(evaluation, "_gold_preview", gold)
    preview = evaluation.preview_generation3_reveal(runtime_root=tmp_path)
    applied = evaluation.apply_generation3_reveal_and_preflight(
        preview,
        expected_preview_content_sha256=preview["content_sha256"],
        runtime_root=tmp_path,
    )
    assert applied["status"] == "preflight_pass_prediction_blind_preparation_authorized"
    replayed = evaluation.replay_generation3_reveal_and_preflight(
        applied["reveal_authority_sha256"], runtime_root=tmp_path
    )
    assert replayed["idempotent_replay"] is True
    assert replayed["replay_mode"] == "structural_without_audio_or_model_execution"
