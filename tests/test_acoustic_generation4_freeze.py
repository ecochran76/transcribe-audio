import json

import pytest

import acoustic_generation4_freeze as freeze


REPOSITORY = {
    "commit": "1" * 40,
    "module_sha256": "2" * 64,
    "clean": True,
    "upstream_ahead": 0,
    "upstream_behind": 0,
}


def _cases() -> list[dict]:
    people = [
        ("p1", "enrolled-1"),
        ("p1", "enrolled-1"),
        ("p2", "enrolled-2"),
        ("p2", "enrolled-2"),
        ("p3", ""),
        ("p4", ""),
        ("p5", ""),
    ]
    return [
        {
            "source_sha256": f"{index + 1:064x}",
            "transcript_sha256": f"{index + 20:064x}",
            "conversation_id": f"conversation-{index}",
            "recording_id": f"recording-{index}",
            "speaker_gold": [
                {
                    "speaker_label": "A",
                    "person_id": person,
                    "enrolled_subject_id": enrolled,
                },
                {"speaker_label": "B", "person_id": "p6", "enrolled_subject_id": ""},
            ],
            "overlap_codes": [],
        }
        for index, (person, enrolled) in enumerate(people)
    ]


def _bind(monkeypatch) -> None:
    cases = _cases()
    monkeypatch.setattr(
        freeze,
        "_g1a",
        lambda: (
            {"authority": {"qualified_set_sha256": "a" * 64}},
            [dict(item) for item in cases],
        ),
    )
    monkeypatch.setattr(
        freeze,
        "_gold",
        lambda proposed: (
            {"content_sha256": freeze.GOLD_CONTENT_SHA256},
            [dict(item) for item in proposed],
        ),
    )
    monkeypatch.setattr(
        freeze,
        "_g1b",
        lambda: {
            "content_sha256": freeze.G1B_CONTENT_SHA256,
            "selected_factor_contract_sha256": "b" * 64,
            "full_matrix_unit_count": 9,
            "full_matrix_unit_set_sha256": "c" * 64,
            "contract_hashes": {"exact_trial_replay_sha256": "d" * 64},
        },
    )
    monkeypatch.setattr(
        freeze.context,
        "build_generation4_context_contract",
        lambda: {
            "content_sha256": freeze.G1C_CONTENT_SHA256,
            "prompt_sha256": "e" * 64,
            "rubric_sha256": "f" * 64,
        },
    )
    monkeypatch.setattr(freeze, "_repository_authority", lambda: dict(REPOSITORY))


def test_preview_freezes_exact_pre_model_authority(monkeypatch) -> None:
    _bind(monkeypatch)

    preview = freeze.preview_generation4_freeze()
    portable = freeze._portable(preview)

    assert preview["status"] == "immutable_pre_model_authority"
    assert preview["population"]["passing"] is True
    assert preview["cohort_count"] == 7
    assert preview["did_freeze_cohort"] is True
    assert preview["did_freeze_gold_commitment"] is True
    assert preview["did_reveal_gold_to_prediction_workers"] is False
    assert preview["did_load_or_run_models"] is False
    assert "private_evidence" not in portable
    assert portable["contains_private_membership"] is False
    assert portable["contains_private_gold"] is False


def test_preview_rejects_context_contract_drift(monkeypatch) -> None:
    _bind(monkeypatch)
    monkeypatch.setattr(
        freeze.context,
        "build_generation4_context_contract",
        lambda: {"content_sha256": "0" * 64},
    )

    with pytest.raises(freeze.Generation4FreezeError, match="G1C"):
        freeze.preview_generation4_freeze()


def test_apply_replay_is_private_and_detects_policy_drift(tmp_path, monkeypatch) -> None:
    _bind(monkeypatch)
    preview = freeze.preview_generation4_freeze()

    applied = freeze.apply_generation4_freeze(
        preview,
        expected_content_sha256=preview["content_sha256"],
        runtime_root=tmp_path,
    )
    replayed = freeze.replay_generation4_freeze(
        preview["content_sha256"], runtime_root=tmp_path
    )

    paths = freeze._paths(tmp_path, preview["content_sha256"])
    assert applied["idempotent_replay"] is False
    assert replayed["idempotent_replay"] is True
    assert paths["manifest"].stat().st_mode & 0o777 == 0o600
    assert paths["receipt"].stat().st_mode & 0o777 == 0o600
    assert "private_evidence" not in json.loads(paths["receipt"].read_text())

    monkeypatch.setattr(
        freeze,
        "_terminal_policy",
        lambda: {"precedence": ["advance_to_limited_pilot_plan"]},
    )
    with pytest.raises(freeze.Generation4FreezeError, match="drifted"):
        freeze.replay_generation4_freeze(
            preview["content_sha256"], runtime_root=tmp_path
        )
