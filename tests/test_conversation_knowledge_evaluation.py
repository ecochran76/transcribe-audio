from __future__ import annotations

import json
from pathlib import Path

import pytest

import conversation_knowledge_evaluation


def _write_json(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _campaign(tmp_path: Path) -> tuple[Path, str]:
    campaign_root = tmp_path / "campaigns"
    campaign_id = "campaign-9ef07483fd3f65a43c27"
    campaign_dir = campaign_root / campaign_id
    items = [
        {
            "chronological_rank": rank,
            "document_id": f"document-{rank:02d}",
            "artifact_sha256": f"{rank:064x}",
            "disposition": (
                "incomplete"
                if rank == 5
                else "duplicate_member"
                if rank == 8
                else "needs_operator_classification"
            ),
            "duplicate_cluster_id": "duplicate-1" if rank in {7, 8} else "",
            "utterance_count": rank * 10,
        }
        for rank in range(1, 15)
    ]
    _write_json(
        campaign_dir / "manifest.json",
        {
            "schema_version": "test-manifest.v1",
            "campaign_id": campaign_id,
            "manifest_id": "manifest-9ef07483fd3f65a43c27",
            "items": items,
        },
    )
    _write_json(
        campaign_dir / "gold" / "index.json",
        {
            "schema_version": "test-gold-index.v1",
            "records": [
                {
                    "gold_id": f"gold-{rank}",
                    "document_id": f"document-{rank:02d}",
                    "chronological_rank": rank,
                    "disposition": "eligible_known",
                }
                for rank in range(1, 5)
            ],
        },
    )
    return campaign_root, campaign_id


def test_freeze_selects_unseen_chronological_cases_without_gold_content(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    campaign_root, campaign_id = _campaign(tmp_path)
    monkeypatch.setattr(
        conversation_knowledge_evaluation,
        "_repository_state",
        lambda: {"commit": "a" * 40, "dirty_tree": False},
    )

    frozen = conversation_knowledge_evaluation.freeze_chronological_evaluation(
        campaign_id,
        campaign_root=campaign_root,
        evaluation_root=tmp_path / "evaluations",
        cohort_size=5,
        approval_token=(
            conversation_knowledge_evaluation.FREEZE_EVALUATION_TOKEN
        ),
    )

    assert [item["chronological_rank"] for item in frozen["cases"]] == [
        6,
        7,
        9,
        10,
        11,
    ]
    assert frozen["start_after_chronological_rank"] == 4
    assert frozen["prediction_visibility"] == "unseen"
    assert frozen["status"] == "frozen_pending_readiness"
    assert frozen["excluded_disposition_counts"] == {
        "duplicate_member": 1,
        "incomplete": 1,
    }
    serialized = json.dumps(frozen)
    assert "name" not in serialized
    assert "email" not in serialized
    assert "speaker_outcomes" not in serialized
    freeze_path = Path(frozen["freeze_path"])
    assert freeze_path.stat().st_mode & 0o777 == 0o600

    repeated = conversation_knowledge_evaluation.freeze_chronological_evaluation(
        campaign_id,
        campaign_root=campaign_root,
        evaluation_root=tmp_path / "evaluations",
        cohort_size=5,
        approval_token=(
            conversation_knowledge_evaluation.FREEZE_EVALUATION_TOKEN
        ),
    )
    assert repeated == frozen


def test_readiness_decision_is_aggregate_immutable_and_preserves_unseen_cohort(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    campaign_root, campaign_id = _campaign(tmp_path)
    monkeypatch.setattr(
        conversation_knowledge_evaluation,
        "_repository_state",
        lambda: {"commit": "b" * 40, "dirty_tree": False},
    )
    frozen = conversation_knowledge_evaluation.freeze_chronological_evaluation(
        campaign_id,
        campaign_root=campaign_root,
        evaluation_root=tmp_path / "evaluations",
        cohort_size=5,
        approval_token=(
            conversation_knowledge_evaluation.FREEZE_EVALUATION_TOKEN
        ),
    )

    decision = conversation_knowledge_evaluation.record_readiness_decision(
        frozen["freeze_id"],
        evaluation_root=tmp_path / "evaluations",
        decision="refine",
        reason_codes=(
            "provider_adapters_not_integrated",
            "family_ablation_not_runnable",
        ),
        gate_results=(
            {
                "gate": "retrieval_bundle_interface",
                "status": "pass",
                "evidence": "C5 focused tests",
            },
            {
                "gate": "live_provider_snapshot_adapters",
                "status": "fail",
                "evidence": "No concrete production adapters",
            },
        ),
        historical_metrics={
            "reviewed_cases": 20,
            "known_person_labels": 53,
            "calibrated_top_correct": 17,
            "calibrated_high_or_very_high_wrong": 0,
        },
        retrieval_metrics={
            "preview_bundles": 3,
            "included_evidence": 0,
            "production_provider_adapters": 0,
        },
        family_results={
            family: {
                "status": "not_run",
                "reason_code": "readiness_gate_failed",
            }
            for family in conversation_knowledge_evaluation.EVIDENCE_FAMILIES
        },
        successor_scope=(
            "Implement concrete bounded provider snapshot adapters.",
            "Run the preserved unseen cohort only after readiness passes.",
        ),
        approval_token=(
            conversation_knowledge_evaluation.RECORD_DECISION_TOKEN
        ),
    )

    assert decision["decision"] == "refine"
    assert decision["cohort_prediction_status"] == "not_started"
    assert decision["automatic_confirmation_enabled"] is False
    assert decision["database_authority_enabled"] is False
    assert decision["family_results"]["combined"]["status"] == "not_run"
    assert Path(decision["decision_path"]).stat().st_mode & 0o777 == 0o600

    with pytest.raises(ValueError, match="Immutable decision conflict"):
        conversation_knowledge_evaluation.record_readiness_decision(
            frozen["freeze_id"],
            evaluation_root=tmp_path / "evaluations",
            decision="accept",
            reason_codes=("different",),
            gate_results=(),
            historical_metrics={},
            retrieval_metrics={},
            family_results={
                family: {
                    "status": "not_run",
                    "reason_code": "different",
                }
                for family in conversation_knowledge_evaluation.EVIDENCE_FAMILIES
            },
            successor_scope=(),
            approval_token=(
                conversation_knowledge_evaluation.RECORD_DECISION_TOKEN
            ),
        )


def test_freeze_requires_approval_and_enough_cases(tmp_path: Path) -> None:
    campaign_root, campaign_id = _campaign(tmp_path)
    with pytest.raises(ValueError, match="approval token"):
        conversation_knowledge_evaluation.freeze_chronological_evaluation(
            campaign_id,
            campaign_root=campaign_root,
            evaluation_root=tmp_path / "evaluations",
            cohort_size=20,
            approval_token="wrong",
        )
    with pytest.raises(ValueError, match="needs 20"):
        conversation_knowledge_evaluation.freeze_chronological_evaluation(
            campaign_id,
            campaign_root=campaign_root,
            evaluation_root=tmp_path / "evaluations",
            cohort_size=20,
            approval_token=(
                conversation_knowledge_evaluation.FREEZE_EVALUATION_TOKEN
            ),
        )
