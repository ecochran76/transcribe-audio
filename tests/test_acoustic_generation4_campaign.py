from __future__ import annotations

from acoustic_generation4_campaign import (
    Generation4CampaignError,
    apply_generation4_campaign,
    preview_generation4_campaign,
    replay_generation4_campaign,
)
import pytest


SHA_A = "a" * 64
SHA_B = "b" * 64
SHA_C = "c" * 64
SHA_D = "d" * 64
SHA_E = "e" * 64
SHA_F = "f" * 64
SHA_1 = "1" * 64
SHA_2 = "2" * 64


def _evidence() -> dict[str, object]:
    return {
        "media": {
            "preview_content_sha256": SHA_A,
            "manifest_sha256": SHA_B,
            "qualified_set_sha256": SHA_C,
            "candidate_count": 12,
            "qualified_count": 10,
            "rejected_count": 2,
            "reason_counts": {"qualified": 10, "duration_below_minimum": 2},
            "replay_mode": "full_body_with_source_redecode_no_retained_audio",
            "idempotent_replay": True,
        },
        "profiles": {
            "recalibration_content_sha256": SHA_D,
            "recalibration_manifest_sha256": SHA_E,
            "profile_set_sha256": SHA_F,
            "model_asset_set_sha256": SHA_1,
            "profile_count": 6,
            "subject_count": 2,
            "candidate_count": 3,
        },
        "thresholds": {
            "execution_authority_sha256": SHA_2,
            "score_matrix_sha256": SHA_A,
            "threshold_application_sha256": SHA_B,
            "threshold_set_sha256": SHA_C,
            "threshold_unit_count": 9,
            "replay_mode": "recomputed_from_persisted_scores_without_audio",
            "idempotent_replay": True,
        },
        "runtime": {"speechbrain": "1.1.0", "onnxruntime": "1.24.4"},
    }


def _repository() -> dict[str, object]:
    return {
        "commit": "1" * 40,
        "module_name": "acoustic_generation4_campaign.py",
        "module_sha256": SHA_D,
        "plan_sha256": SHA_E,
        "clean": True,
        "upstream_ahead": 0,
        "upstream_behind": 0,
    }


def test_preview_opens_only_the_three_design_lanes() -> None:
    preview = preview_generation4_campaign(
        collect_evidence=_evidence,
        collect_repository=_repository,
    )

    assert preview["status"] == "g0_ready_to_freeze"
    assert preview["action_vector"] == {
        "run_g1a_cohort_gold_feasibility": True,
        "run_g1b_acoustic_contract": True,
        "run_g1c_context_contract": True,
        "run_j1_design_reconciliation": False,
        "freeze_g2_envelope": False,
        "run_g3_blind_baseline": False,
        "run_g4_augmented_predictions": False,
        "run_j2_blindness_audit": False,
        "reveal_gold": False,
        "run_g5_scoring": False,
        "run_j3_result_audit": False,
        "make_g6_terminal_decision": False,
        "mutate_profiles_or_references": False,
        "enable_default_integration": False,
        "run_historical_reprocessing": False,
    }
    assert preview["contains_paths"] is False
    assert preview["contains_private_membership"] is False
    assert preview["contains_biometric_scores"] is False
    assert preview["did_read_private_gold"] is False
    assert preview["did_load_or_run_models"] is False
    assert preview["will_perform_external_write"] is False


def test_apply_then_replay_is_immutable_and_idempotent(tmp_path) -> None:
    preview = preview_generation4_campaign(
        collect_evidence=_evidence,
        collect_repository=_repository,
    )

    applied = apply_generation4_campaign(
        preview,
        expected_content_sha256=preview["content_sha256"],
        runtime_root=tmp_path,
        collect_evidence=_evidence,
        collect_repository=_repository,
    )
    replayed = replay_generation4_campaign(
        preview["content_sha256"],
        runtime_root=tmp_path,
        collect_evidence=_evidence,
        collect_repository=_repository,
    )

    assert applied["status"] == "g0_frozen_g1_design_lanes_authorized"
    assert applied["idempotent_replay"] is False
    assert replayed["idempotent_replay"] is True
    assert replayed["manifest_sha256"] == applied["manifest_sha256"]
    assert replayed["action_vector"] == preview["action_vector"]
    assert oct(tmp_path.stat().st_mode & 0o777) == "0o700"
    assert replayed["mode"] == "0600"


def test_preview_fails_closed_when_inherited_replay_is_not_exact() -> None:
    evidence = _evidence()
    evidence["media"]["idempotent_replay"] = False

    with pytest.raises(Generation4CampaignError, match="inherited evidence"):
        preview_generation4_campaign(
            collect_evidence=lambda: evidence,
            collect_repository=_repository,
        )
