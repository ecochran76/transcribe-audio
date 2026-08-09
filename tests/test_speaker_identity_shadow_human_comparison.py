from __future__ import annotations

import stat
from pathlib import Path

import pytest

import speaker_identity_shadow_human_comparison as comparison
import speaker_identity_shadow_human_review as review


def _manifest() -> dict:
    cases = []
    evaluation_ordinal = 0
    for case_ordinal, speaker_count in enumerate((4, 3, 3), start=1):
        document_id = f"document-{case_ordinal}"
        candidates = [
            {
                "person_id": f"person-{case_ordinal}-a",
                "label": "Candidate A",
                "email": "private-a@example.com",
                "status": "separate_review_only",
            },
            {
                "person_id": f"person-{case_ordinal}-b",
                "label": "Candidate B",
                "email": "private-b@example.com",
                "status": "separate_review_only",
            },
        ]
        slots = []
        for speaker_ordinal in range(1, speaker_count + 1):
            conditions = []
            for condition in review.CONDITIONS:
                evaluation_ordinal += 1
                conditions.append(
                    {
                        "evaluation_id": f"evaluation-{evaluation_ordinal}",
                        "condition": condition,
                        "outcome": "abstained",
                        "proposed_person_id": None,
                        "alternative_person_ids": [item["person_id"] for item in candidates],
                        "base_confidence": 0.2,
                        "capped_confidence": 0.2,
                        "confidence_cap_reasons": [],
                        "abstention_reason": "review required",
                        "source_failures": [],
                        "factors": [
                            {
                                "factor_type": condition.split("_")[0],
                                "score": 0.2,
                                "evidence_ids": [f"evidence-{evaluation_ordinal}"],
                                "independence_groups": ["group-1"],
                            }
                        ],
                    }
                )
            slots.append(
                {
                    "speaker_ref": f"SPEAKER_{speaker_ordinal}",
                    "allowed_decisions": [
                        *(item["person_id"] for item in candidates),
                        "not_listed",
                        "unresolved",
                    ],
                    "selected_person_id": None,
                    "decision_status": "pending",
                    "acoustic": {
                        "disposition": "insufficient",
                        "confidence_band": "low",
                        "score": 0.2,
                        "supporting_unit_count": 1,
                        "opposing_unit_count": 0,
                        "insufficient_unit_count": 1,
                    },
                    "conditions": conditions,
                }
            )
        cases.append(
            {
                "document_id": document_id,
                "recording_id": f"recording-{case_ordinal}",
                "candidate_options": candidates,
                "speaker_slots": slots,
                "warnings": [],
                "source_failures": [],
                "scopes": [
                    {
                        "source_type": "fixture",
                        "capabilities": ["read"],
                        "max_provider_calls": 1,
                        "max_records": 2,
                    }
                ],
            }
        )
    return {
        "schema_version": review.plan0060.P4_MANIFEST_VERSION,
        "activation_sha256": review.PLAN0060_ACTIVATION_SHA256,
        "recording_count": 3,
        "speaker_slot_count": 10,
        "condition_count": 30,
        "human_decision_count": 0,
        "preselected_decision_count": 0,
        "apply_enabled": False,
        "human_gold_read": False,
        "negative_actions": {"apply": False, "write": False},
        "cases": cases,
    }


def _submission(manifest: dict) -> dict:
    rows = [
        f"PLAN0061_SCHEMA={review.DECISION_SUBMISSION_SCHEMA}",
        f"PLAN0061_P4_CONTENT_SHA256={review.PLAN0060_P4_CONTENT_SHA256}",
        f"PLAN0061_P4_MANIFEST_SHA256={review.PLAN0060_P4_MANIFEST_SHA256}",
    ]
    person_slots = 0
    for case in review.normalized_review_cases(manifest):
        for slot in case["slots"]:
            if person_slots < 3:
                decision = case["candidates"][0]["person_id"]
                person_slots += 1
            else:
                decision = "not_listed"
            rows.append(f"{slot['slot_id']}={decision}")
    return review.parse_decision_block("\n".join(rows), manifest)


def test_recompute_scores_complete_abstention_denominator() -> None:
    manifest = _manifest()
    result = comparison.recompute_comparison(manifest, _submission(manifest))

    assert result["gold_metrics"] == {
        "person_count": 3,
        "not_listed_count": 7,
        "unresolved_count": 0,
        "candidate_recalled_count": 3,
        "candidate_recall_denominator": 10,
        "candidate_recall": 0.3,
        "unresolved_rate": 0.0,
    }
    assert result["terminal_decision"] == "refine"
    for metrics in result["condition_metrics"].values():
        assert metrics["evaluation_count"] == 10
        assert metrics["proposal_count"] == 0
        assert metrics["correct_proposal_count"] == 0
        assert metrics["wrong_proposal_count"] == 0
        assert metrics["high_confidence_wrong_count"] == 0
        assert metrics["known_person_count"] == 3
        assert metrics["enrolled_recall"] == 0.0
        assert metrics["precision"] is None
        assert metrics["appropriate_abstention_rate"] == 1.0
        assert metrics["provenance_completeness"] == 1.0
        assert metrics["provider_failure_count"] == 0
        assert metrics["duplicate_person_fork_count"] == 0


def test_recompute_stops_on_high_confidence_wrong_proposal() -> None:
    manifest = _manifest()
    submission = _submission(manifest)
    condition = manifest["cases"][0]["speaker_slots"][0]["conditions"][1]
    condition["outcome"] = "proposed"
    condition["proposed_person_id"] = manifest["cases"][0]["candidate_options"][1][
        "person_id"
    ]
    condition["alternative_person_ids"] = [
        manifest["cases"][0]["candidate_options"][0]["person_id"]
    ]
    condition["capped_confidence"] = 0.9

    result = comparison.recompute_comparison(manifest, submission)

    assert result["terminal_decision"] == "stop"
    assert result["condition_metrics"]["acoustic_only"]["wrong_proposal_count"] == 1
    assert (
        result["condition_metrics"]["acoustic_only"]["high_confidence_wrong_count"]
        == 1
    )


def test_freeze_and_replay_are_private_immutable_and_non_applying(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = _manifest()
    submission = _submission(manifest)
    answer_rows = [
        f"PLAN0061_SCHEMA={review.DECISION_SUBMISSION_SCHEMA}",
        f"PLAN0061_P4_CONTENT_SHA256={review.PLAN0060_P4_CONTENT_SHA256}",
        f"PLAN0061_P4_MANIFEST_SHA256={review.PLAN0060_P4_MANIFEST_SHA256}",
        *(f"{item['slot_id']}={item['decision']}" for item in submission["decisions"]),
    ]
    bindings = {
        "plan0060_activation_sha256": review.PLAN0060_ACTIVATION_SHA256,
        "p4_content_sha256": review.PLAN0060_P4_CONTENT_SHA256,
        "p4_manifest_sha256": review.PLAN0060_P4_MANIFEST_SHA256,
        "terminal_manifest_sha256": review.PLAN0060_TERMINAL_MANIFEST_SHA256,
        "live": {"quick_check": "ok"},
    }
    repository = {
        "commit": "a" * 40,
        "modules": {
            comparison.MODULE_PATH: "b" * 64,
            review.MODULE_PATH: "c" * 64,
        },
        "clean": True,
        "upstream_ahead": 0,
        "upstream_behind": 0,
    }
    monkeypatch.setattr(comparison, "_repository_authority", lambda: repository)
    monkeypatch.setattr(review, "_validated_live_source", lambda **_: (manifest, bindings))
    runtime_root = tmp_path / "plan-0061"

    frozen = comparison.freeze_human_gold_and_comparison(
        "\n".join(answer_rows), runtime_root=runtime_root
    )
    replayed = comparison.replay_human_gold_and_comparison(
        frozen["submission_content_sha256"], runtime_root=runtime_root
    )

    assert frozen["idempotent_replay"] is False
    assert replayed["idempotent_replay"] is True
    assert replayed["comparison_content_sha256"] == frozen["comparison_content_sha256"]
    assert replayed["terminal_decision"] == "refine"
    assert replayed["live_mutation_count"] == 0
    run = Path(replayed["runtime_path"])
    assert stat.S_IMODE(run.stat().st_mode) == 0o700
    assert all(stat.S_IMODE(path.stat().st_mode) == 0o600 for path in run.iterdir())
