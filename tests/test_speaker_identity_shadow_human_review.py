from __future__ import annotations

import os
import re
from pathlib import Path

import pytest

import speaker_identity_shadow_human_review as review


def _manifest() -> dict:
    cases = []
    speaker_counts = (4, 3, 3)
    evaluation_ordinal = 0
    for case_ordinal, speaker_count in enumerate(speaker_counts, start=1):
        document_id = f"document-{case_ordinal}"
        candidates = [
            {
                "person_id": f"person-{case_ordinal}-a",
                "label": "Candidate A </script><script>alert(1)</script>",
                "email": "must-not-appear@example.com",
                "status": "candidate",
            },
            {
                "person_id": f"person-{case_ordinal}-b",
                "label": "Candidate B",
                "email": "also-private@example.com",
                "status": "candidate",
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
                        "alternative_person_ids": [candidates[0]["person_id"]],
                        "base_confidence": 0.2,
                        "capped_confidence": 0.1,
                        "confidence_cap_reasons": ["missing binding"],
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
                        candidates[0]["person_id"],
                        candidates[1]["person_id"],
                        "not_listed",
                        "unresolved",
                    ],
                    "selected_person_id": None,
                    "decision_status": "pending",
                    "acoustic": {
                        "acoustic_subject_id": f"subject-{case_ordinal}-{speaker_ordinal}",
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


def _complete_block(manifest: dict) -> str:
    cases = review.normalized_review_cases(manifest)
    rows = [
        f"PLAN0061_SCHEMA={review.DECISION_SUBMISSION_SCHEMA}",
        f"PLAN0061_P4_CONTENT_SHA256={review.PLAN0060_P4_CONTENT_SHA256}",
        f"PLAN0061_P4_MANIFEST_SHA256={review.PLAN0060_P4_MANIFEST_SHA256}",
    ]
    for case in cases:
        for slot in case["slots"]:
            rows.append(f"{slot['slot_id']}={slot['allowed_decisions'][0]}")
    return "\n".join(rows)


def test_renderer_is_complete_unselected_and_client_only() -> None:
    page = review.render_review_worksheet(_manifest())

    assert page.count("data-review-slot") == 10
    assert page.count("data-decision ") == 10
    assert page.count("Open this recording in the local transcript console") == 3
    assert "must-not-appear@example.com" not in page
    assert "also-private@example.com" not in page
    assert "</script><script>alert(1)</script>" not in page
    assert "Candidate A &lt;/script&gt;&lt;script&gt;alert(1)&lt;/script&gt;" in page
    assert not re.search(r"<option[^>]+selected", page)
    assert "fetch(" not in page
    assert "XMLHttpRequest" not in page
    assert "WebSocket" not in page
    assert "form-action 'none'" in page
    assert "Nothing has been submitted or applied." in page


def test_complete_decision_block_round_trips_exactly() -> None:
    manifest = _manifest()
    result = review.parse_decision_block(_complete_block(manifest), manifest)

    assert result["status"] == "complete_operator_decisions_preview"
    assert result["decision_count"] == 10
    assert len(result["decisions"]) == 10
    assert result["applied_assignments"] is False
    assert result["wrote_live_knowledge"] is False
    assert result["wrote_external_provider"] is False
    assert result["wrote_graphiti"] is False
    assert re.fullmatch(r"[a-f0-9]{64}", result["content_sha256"])


@pytest.mark.parametrize("mutation", ["partial", "duplicate", "out_of_set", "stale"])
def test_decision_parser_rejects_inexact_gold(mutation: str) -> None:
    manifest = _manifest()
    lines = _complete_block(manifest).splitlines()
    if mutation == "partial":
        lines.pop()
    elif mutation == "duplicate":
        lines[-1] = lines[-2]
    elif mutation == "out_of_set":
        lines[-1] = lines[-1].rsplit("=", 1)[0] + "=display-label-is-not-authority"
    else:
        lines[1] = "PLAN0061_P4_CONTENT_SHA256=" + "0" * 64

    with pytest.raises(review.Plan0061ReviewError):
        review.parse_decision_block("\n".join(lines), manifest)


def test_renderer_rejects_preselection_and_denominator_drift() -> None:
    manifest = _manifest()
    manifest["cases"][0]["speaker_slots"][0]["selected_person_id"] = "person-1-a"
    with pytest.raises(review.Plan0061ReviewError):
        review.render_review_worksheet(manifest)

    manifest = _manifest()
    manifest["cases"][0]["speaker_slots"].pop()
    with pytest.raises(review.Plan0061ReviewError):
        review.render_review_worksheet(manifest)


def test_private_worksheet_freezes_and_replays_without_decisions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _manifest()
    bindings = {
        "plan0060_activation_sha256": review.PLAN0060_ACTIVATION_SHA256,
        "p4_content_sha256": review.PLAN0060_P4_CONTENT_SHA256,
        "p4_manifest_sha256": review.PLAN0060_P4_MANIFEST_SHA256,
        "terminal_manifest_sha256": review.PLAN0060_TERMINAL_MANIFEST_SHA256,
        "live": {"quick_check": "ok"},
    }
    monkeypatch.setattr(
        review,
        "_repository_authority",
        lambda: {
            "commit": "a" * 40,
            "module_sha256": "b" * 64,
            "clean": True,
            "upstream_ahead": 0,
            "upstream_behind": 0,
        },
    )
    monkeypatch.setattr(
        review,
        "_validated_live_source",
        lambda **_: (source, bindings),
    )
    runtime_root = tmp_path / "plan-0061"

    frozen = review.prepare_live_worksheet(runtime_root=runtime_root)
    replay = review.replay_live_worksheet(
        frozen["worksheet_sha256"], runtime_root=runtime_root
    )

    assert frozen["idempotent_replay"] is False
    assert replay["idempotent_replay"] is True
    assert replay["content_sha256"] == frozen["content_sha256"]
    assert replay["preselected_decision_count"] == 0
    assert replay["human_decision_count"] == 0
    assert replay["apply_enabled"] is False
    for path_key in ("worksheet_path", "manifest_path"):
        assert os.stat(replay[path_key]).st_mode & 0o777 == 0o600
