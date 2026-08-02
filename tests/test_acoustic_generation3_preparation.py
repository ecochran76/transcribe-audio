from __future__ import annotations

import json
from pathlib import Path

import pytest

import acoustic_generation3_preparation as preparation


def _reveal() -> dict[str, object]:
    return {
        "authority_sha256": "1" * 64,
        "preflight_sha256": "2" * 64,
        "gold_manifest_sha256": "3" * 64,
        "receipt": {},
    }


def _units(source_root: Path | None = None) -> list[dict[str, object]]:
    if source_root is not None:
        source_root.mkdir(mode=0o700)
        source_root.chmod(0o700)
        for index in range(7):
            path = source_root / f"source-{index}.mkv"
            path.write_bytes(f"source-{index}".encode())
    return [
        {
            "conversation_input_id": f"input-{index}",
            "recording_id": f"recording-{index}",
            "conversation_id": f"conversation-{index}",
            "source_sha256": (
                preparation.sha256_file(source_root / f"source-{index}.mkv")
                if source_root is not None else f"{index + 10:064x}"
            ),
            "source_path": str(
                source_root / f"source-{index}.mkv"
                if source_root is not None else f"/private/source-{index}.mkv"
            ),
            "split": "evaluation",
        }
        for index in range(7)
    ]


def _repository(*, clean: bool = True) -> dict[str, object]:
    return {
        "commit": "a" * 40,
        "module_sha256": {name: "b" * 64 for name in preparation.MODULE_NAMES},
        "clean": clean,
        "upstream_ahead": 0,
        "upstream_behind": 0,
    }


def _preview(
    monkeypatch: pytest.MonkeyPatch, units: list[dict[str, object]] | None = None,
) -> dict[str, object]:
    monkeypatch.setattr(preparation, "_reveal_context", lambda _root: _reveal())
    monkeypatch.setattr(
        preparation, "_cohort_units", lambda _root, _reveal_value: units or _units()
    )
    monkeypatch.setattr(preparation, "_repository_authority", _repository)
    return preparation.preview_generation3_preparation()


def test_preview_is_prediction_blind_and_authorizes_nothing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    preview = _preview(monkeypatch)
    assert preview["unit_count"] == 7
    assert len(preview["method_ids"]) == 5
    assert preview["condition_fields"] == list(preparation.conditions.CONDITION_FIELDS)
    assert preview["condition_minimum_observed_values"] == 2
    assert preview["condition_missing_recordings_allowed"] == 0
    assert preview["did_read_audio"] is False
    assert preview["did_load_or_run_models"] is False
    assert all(value is False for value in preview["action_vector"].values())


def test_portable_projection_hides_paths_and_membership(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    preview = _preview(monkeypatch)
    portable = preparation.portable_preparation_projection(preview)
    assert "/private/source" not in str(portable)
    assert "recording-0" not in str(portable)
    assert portable["contains_paths"] is False
    assert portable["contains_private_membership"] is False


def test_pass_receipt_authorizes_only_window_freeze(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    preview = _preview(monkeypatch)
    application = {
        "condition_coverage": {"terminal_selection_eligible": True, "blockers": []},
        "unit_count": 7,
        "method_attempt_count": 35,
        "method_success_count": 35,
    }
    receipt = preparation._receipt(preview, "c" * 64, "d" * 64, application)
    assert receipt["action_vector"]["freeze_evaluation_windows"] is True
    assert receipt["action_vector"]["record_terminal_stop"] is False
    assert receipt["action_vector"]["construct_exact_trial_child"] is False
    assert receipt["action_vector"]["load_or_run_models"] is False


def test_blocked_conditions_authorize_stop_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    preview = _preview(monkeypatch)
    application = {
        "condition_coverage": {
            "terminal_selection_eligible": False,
            "blockers": ["device_condition_coverage_below_policy"],
        },
        "unit_count": 7,
        "method_attempt_count": 35,
        "method_success_count": 35,
    }
    receipt = preparation._receipt(preview, "c" * 64, "d" * 64, application)
    assert receipt["action_vector"]["record_terminal_stop"] is True
    assert receipt["action_vector"]["freeze_evaluation_windows"] is False
    assert receipt["action_vector"]["make_terminal_decision"] is False
    assert receipt["action_vector"]["load_or_run_models"] is False


def test_apply_writes_authority_before_p1_p2(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    tmp_path.chmod(0o700)
    units = _units(tmp_path / "sources")
    preview = _preview(monkeypatch, units)
    monkeypatch.setattr(
        preparation, "_validate_repository_authority", lambda value: dict(value)
    )

    def execute(
        unit: dict[str, object], _preview_value: dict[str, object],
        paths: dict[str, Path],
    ) -> dict[str, object]:
        authorities = list(
            tmp_path.glob("evaluation-preparation/*/authority.json")
        )
        assert len(authorities) == 1
        preparation.ensure_private_tree(tmp_path, paths["p1"])
        preparation.ensure_private_tree(tmp_path, paths["p2"])
        index = str(unit["recording_id"]).split("-")[-1]
        p1_manifest = paths["p1"] / f"manifest-{index}.json"
        p1_replay = paths["p1"] / f"replay-{index}.json"
        p2_comparison = paths["p2"] / f"comparison-{index}.json"
        p2_replay = paths["p2"] / f"replay-{index}.json"
        methods = [
            {"method_id": method, "status": "success"}
            for method in preparation.conditions.METHOD_IDS
        ]
        preparation.write_immutable_private_json(p1_manifest, {"status": "success"})
        preparation.write_immutable_private_json(p1_replay, {"status": "success"})
        preparation.write_immutable_private_json(
            p2_comparison, {"status": "success", "method_results": methods}
        )
        preparation.write_immutable_private_json(p2_replay, {"status": "success"})
        return {
            "recording_id": unit["recording_id"],
            "conversation_id": unit["conversation_id"],
            "source_sha256": unit["source_sha256"],
            "split": "evaluation",
            "p1_manifest_path": str(p1_manifest),
            "p1_manifest_sha256": preparation.sha256_file(p1_manifest),
            "p1_replay_path": str(p1_replay),
            "p1_replay_sha256": preparation.sha256_file(p1_replay),
            "p2_comparison_path": str(p2_comparison),
            "p2_comparison_sha256": preparation.sha256_file(p2_comparison),
            "p2_replay_path": str(p2_replay),
            "p2_replay_sha256": preparation.sha256_file(p2_replay),
            "method_result_sha256": {
                item["method_id"]: preparation.conditions._canonical_hash(item)
                for item in methods
            },
            "conditions": {
                "channel": "source_mono" if unit["recording_id"] != "recording-0" else "source_stereo",
                "device": "device-a" if unit["recording_id"] != "recording-0" else "device-b",
                "device_observed": True,
                "noise": "low" if unit["recording_id"] != "recording-0" else "high",
                "telephone_bandwidth": "wide" if unit["recording_id"] != "recording-0" else "telephone",
                "usable_duration_band": "long" if unit["recording_id"] != "recording-0" else "short",
            },
        }

    monkeypatch.setattr(preparation.conditions, "_execute_unit", execute)
    applied = preparation.apply_generation3_preparation(
        preview,
        expected_preview_content_sha256=preview["content_sha256"],
        runtime_root=tmp_path,
    )
    assert applied["method_success_count"] == 35
    assert applied["action_vector"]["freeze_evaluation_windows"] is True
    replayed = preparation.replay_generation3_preparation(
        applied["preparation_authority_sha256"], runtime_root=tmp_path
    )
    assert replayed["idempotent_replay"] is True
    assert replayed["replay_mode"] == "structural_without_audio_or_preparation_execution"
    application_path = Path(str(applied["private_application_path"]))
    original_application = application_path.read_text(encoding="utf-8")
    drifted = json.loads(original_application)
    drifted["contains_paths"] = False
    application_path.chmod(0o600)
    application_path.write_text(json.dumps(drifted), encoding="utf-8")
    application_path.chmod(0o600)
    with pytest.raises(
        preparation.Generation3PreparationError, match="application drifted",
    ):
        preparation.replay_generation3_preparation(
            applied["preparation_authority_sha256"], runtime_root=tmp_path
        )
    drifted = json.loads(original_application)
    drifted["units"][0]["split"] = "development"
    application_path.chmod(0o600)
    application_path.write_text(json.dumps(drifted), encoding="utf-8")
    application_path.chmod(0o600)
    with pytest.raises(
        preparation.Generation3PreparationError, match="split binding drifted",
    ):
        preparation.replay_generation3_preparation(
            applied["preparation_authority_sha256"], runtime_root=tmp_path
        )
    application_path.write_text(original_application, encoding="utf-8")
    application_path.chmod(0o600)
    source = Path(str(units[0]["source_path"]))
    source.write_bytes(b"drift")
    with pytest.raises(
        preparation.Generation3PreparationError, match="source bytes drifted",
    ):
        preparation.replay_generation3_preparation(
            applied["preparation_authority_sha256"], runtime_root=tmp_path
        )


def test_method_hashes_reject_extra_failed_or_duplicate_rows() -> None:
    good = [
        {"method_id": method, "status": "success"}
        for method in preparation.conditions.METHOD_IDS
    ]
    assert set(preparation._method_hashes({"method_results": good})) == set(
        preparation.conditions.METHOD_IDS
    )
    with pytest.raises(
        preparation.Generation3PreparationError, match="incomplete",
    ):
        preparation._method_hashes(
            {"method_results": [*good, {"method_id": "extra", "status": "failure"}]}
        )
    duplicate = [*good[:-1], dict(good[0])]
    with pytest.raises(
        preparation.Generation3PreparationError, match="inventory drifted",
    ):
        preparation._method_hashes({"method_results": duplicate})


def test_dirty_repository_fails_before_authority_write(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    tmp_path.chmod(0o700)
    preview = _preview(monkeypatch)
    monkeypatch.setattr(
        preparation, "_validate_repository_authority",
        lambda _value: (_ for _ in ()).throw(
            preparation.Generation3PreparationError("repository drifted")
        ),
    )
    with pytest.raises(preparation.Generation3PreparationError, match="repository"):
        preparation.apply_generation3_preparation(
            preview,
            expected_preview_content_sha256=preview["content_sha256"],
            runtime_root=tmp_path,
        )
    assert not (tmp_path / "evaluation-preparation").exists()
