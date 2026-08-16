from __future__ import annotations

import hashlib
import json
import stat
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from conversation_knowledge_store import ConversationKnowledgeStore
from identity_evidence_supervisor import IdentityEvidenceSupervisor
from identity_review_workflow import IdentityReviewWorkflow
from identity_shadow_campaign import (
    ACTIVATE_TOKEN,
    FINALIZE_TOKEN,
    IdentityShadowCampaignError,
    activate_shadow_campaign,
    finalize_shadow_campaign,
    preview_shadow_campaign,
    record_shadow_case,
    register_new_arrival,
    replay_shadow_campaign,
)


SHA_A = "a" * 64
SHA_B = "b" * 64


def _candidate(index: int, *, cohort: str = "historical") -> dict[str, object]:
    observed = datetime(2026, 8, 1, tzinfo=timezone.utc) + timedelta(hours=index)
    return {
        "conversation_id": f"conversation-{index:03d}",
        "recording_id": f"recording-{index:03d}",
        "original_recording_filename": f"Call {index:03d}.m4a",
        "source_artifact_sha256": SHA_A,
        "source_media_sha256": SHA_B,
        "conversation_at": observed.isoformat().replace("+00:00", "Z"),
        "artifact_stabilized_at": observed.isoformat().replace("+00:00", "Z"),
        "cohort": cohort,
        "eligible": True,
        "disposition": "eligible",
    }


def test_preview_freezes_oldest_forward_batch_and_new_arrival_window() -> None:
    candidates = [_candidate(index) for index in reversed(range(27))]
    candidates.append(
        {
            **_candidate(100, cohort="new_arrival"),
            "artifact_stabilized_at": "2026-08-16T13:00:00Z",
        }
    )
    candidates.append(
        {
            **_candidate(101, cohort="new_arrival"),
            "artifact_stabilized_at": "2026-08-24T12:00:00Z",
        }
    )

    preview = preview_shadow_campaign(
        candidates,
        activated_at="2026-08-16T12:00:00Z",
    )

    assert preview["schema_version"] == "transcribe-audio.identity-shadow-preview.v1"
    assert preview["operation_mode"] == "shadow"
    assert preview["effect_policy"] == {
        "accepted_identity_effect_count": 0,
        "accepted_profile_effect_count": 0,
        "provider_write_count": 0,
        "raw_deletion_count": 0,
    }
    assert [item["conversation_id"] for item in preview["historical_cases"]] == [
        f"conversation-{index:03d}" for index in range(25)
    ]
    assert preview["historical_inventory"] == {
        "candidate_count": 27,
        "selected_count": 25,
        "deferred_count": 2,
        "ineligible_count": 0,
    }
    assert [item["conversation_id"] for item in preview["new_arrival_cases"]] == [
        "conversation-100"
    ]
    assert preview["new_arrival_window"] == {
        "starts_at": "2026-08-16T12:00:00Z",
        "ends_at": "2026-08-23T12:00:00Z",
        "duration_days": 7,
    }
    assert preview == preview_shadow_campaign(
        list(reversed(candidates)),
        activated_at="2026-08-16T12:00:00Z",
    )
    serialized = repr(preview).lower()
    assert "source_path" not in serialized
    assert "stored_path" not in serialized
    assert "transcript_text" not in serialized


def test_preview_rejects_private_payload_fields() -> None:
    candidate = {**_candidate(1), "source_path": "/private/conversation.m4a"}

    with pytest.raises(IdentityShadowCampaignError, match="private payload field"):
        preview_shadow_campaign(
            [candidate],
            activated_at="2026-08-16T12:00:00Z",
        )

    with pytest.raises(IdentityShadowCampaignError, match="disposition"):
        preview_shadow_campaign(
            [{**_candidate(1), "disposition": "eligible because raw private note"}],
            activated_at="2026-08-16T12:00:00Z",
        )


def test_activation_is_private_immutable_bound_and_replayable(tmp_path) -> None:
    preview = preview_shadow_campaign(
        [_candidate(index) for index in range(25)],
        activated_at="2026-08-16T12:00:00Z",
    )
    preview_sha256 = hashlib.sha256(
        json.dumps(
            preview,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()

    activated = activate_shadow_campaign(
        preview,
        expected_preview_sha256=preview_sha256,
        reviewed_at="2026-08-16T12:05:00Z",
        runtime_root=tmp_path / "shadow",
        approval_token=ACTIVATE_TOKEN,
    )
    replayed = activate_shadow_campaign(
        preview,
        expected_preview_sha256=preview_sha256,
        reviewed_at="2026-08-16T12:05:00Z",
        runtime_root=tmp_path / "shadow",
        approval_token=ACTIVATE_TOKEN,
    )

    assert activated == replayed
    assert activated["status"] == "active_shadow_only"
    assert activated["preview_sha256"] == preview_sha256
    assert activated["effect_policy"] == preview["effect_policy"]
    assert stat.S_IMODE((tmp_path / "shadow").stat().st_mode) == 0o700
    assert stat.S_IMODE(activated["manifest_path"].stat().st_mode) == 0o600
    assert stat.S_IMODE(activated["activation_receipt_path"].stat().st_mode) == 0o600

    changed = {**preview, "operation_mode": "apply"}
    with pytest.raises(IdentityShadowCampaignError, match="preview hash"):
        activate_shadow_campaign(
            changed,
            expected_preview_sha256=preview_sha256,
            reviewed_at="2026-08-16T12:05:00Z",
            runtime_root=tmp_path / "shadow",
            approval_token=ACTIVATE_TOKEN,
        )


def test_activation_requires_exact_operator_token(tmp_path) -> None:
    preview = preview_shadow_campaign(
        [_candidate(index) for index in range(25)],
        activated_at="2026-08-16T12:00:00Z",
    )
    preview_sha256 = hashlib.sha256(
        json.dumps(preview, separators=(",", ":"), sort_keys=True).encode("utf-8")
    ).hexdigest()

    with pytest.raises(IdentityShadowCampaignError, match=ACTIVATE_TOKEN):
        activate_shadow_campaign(
            preview,
            expected_preview_sha256=preview_sha256,
            reviewed_at="2026-08-16T12:05:00Z",
            runtime_root=tmp_path / "shadow",
            approval_token="",
        )


def _queue_item(case: dict[str, object]) -> dict[str, object]:
    return {
        "schema_version": "transcribe-audio.identity-review-queue-item.v1",
        "queue_item_id": f"queue-{case['case_id']}",
        "conversation_id": case["conversation_id"],
        "recording_id": case["recording_id"],
        "original_recording_filename": case["original_recording_filename"],
        "source_artifact_sha256": case["source_artifact_sha256"],
        "source_media_sha256": case["source_media_sha256"],
        "processing_run_id": "shadow-run-001",
        "model_versions": ["identity-model-v1"],
        "rubric_versions": ["identity-rubric-v1"],
        "profile_versions": [],
        "calendar_candidates": [],
        "participant_hypotheses": [],
        "speakers": [],
        "review_state": "unreviewed",
        "decision_history": [],
        "effect_preview_ref": "",
        "projection_version": "1",
        "created_at": "2026-08-16T13:00:00Z",
    }


class _StubSupervisor:
    def __init__(self, run: dict[str, object]) -> None:
        self.run = run

    def load_run(self, run_id: str) -> dict[str, object]:
        assert run_id == self.run["run_id"]
        return self.run


def _supervisor_run(index: int) -> dict[str, object]:
    return {
        "run_id": f"shadow-run-{index:03d}",
        "content_hash": f"{index:064x}",
        "stage": "complete",
        "state": "complete",
        "event_id": f"event-shadow-run-{index:03d}",
        "effect_counts": {"identity": 0, "profile": 0, "provider": 0},
        "provider_retry_count": 1 if index == 24 else 0,
    }


def _complete_supervisor(
    store_root: Path, case: dict[str, object]
) -> IdentityEvidenceSupervisor:
    supervisor = IdentityEvidenceSupervisor(store_root)
    supervisor.start_run(
        {
            "schema_version": "transcribe-audio.processing-run.v1",
            "run_id": "shadow-run-001",
            "conversation_id": case["conversation_id"],
            "recording_id": case["recording_id"],
            "original_recording_filename": case["original_recording_filename"],
            "source_artifact_sha256": case["source_artifact_sha256"],
            "source_media_sha256": case["source_media_sha256"],
            "operation_mode": "shadow",
            "policy_version": "shadow-policy.synthetic.v1",
            "as_of": "2026-08-16T13:00:00Z",
            "capabilities": ["fixture_read"],
            "source_scopes": [
                {
                    "provider_kind": "synthetic",
                    "profile_id": "fixture-profile",
                    "account_id": "fixture-account",
                    "tenant_id": "fixture-tenant",
                    "capabilities": ["fixture_read"],
                }
            ],
            "budgets": {
                "max_records": 4,
                "max_characters": 120,
                "max_calls": 3,
                "max_latency_ms": 500,
            },
            "model_versions": {"context": "context.synthetic.v1"},
            "rubric_versions": {"combined": "combined.synthetic.v1"},
            "profile_versions": [],
            "stage": "bind_conversation",
            "state": "running",
            "transcript_correction_passes": 0,
            "identity_cascade_count": 0,
            "provider_retry_count": 0,
            "input_ids": [str(case["case_id"])],
            "output_ids": [],
            "failures": [],
            "effect_counts": {
                "accepted_identity": 0,
                "profile_activation": 0,
                "provider_write": 0,
            },
            "created_at": "2026-08-16T13:00:00Z",
        }
    )
    stages = (
        "pre_identity_correction",
        "calendar_candidate_generation",
        "participant_and_evidence_collection",
        "speaker_and_relationship_proposals",
        "post_identity_correction",
        "queue_projection",
        "complete",
    )
    for index, stage in enumerate(stages, start=1):
        supervisor.advance_stage(
            run_id="shadow-run-001",
            stage=stage,
            state="complete" if stage == "complete" else "running",
            output_ids=(f"shadow-output-{index}",),
            failures=(),
            effect_counts={
                "accepted_identity": 0,
                "profile_activation": 0,
                "provider_write": 0,
            },
            idempotency_key=f"shadow-stage-{index}",
            created_at=f"2026-08-16T13:{index:02d}:00Z",
        )
    return supervisor


def test_case_receipt_projects_queue_and_enforces_zero_effects(tmp_path) -> None:
    preview = preview_shadow_campaign(
        [_candidate(index) for index in range(25)],
        activated_at="2026-08-16T12:00:00Z",
    )
    preview_sha256 = hashlib.sha256(
        json.dumps(preview, separators=(",", ":"), sort_keys=True).encode("utf-8")
    ).hexdigest()
    runtime_root = tmp_path / "shadow"
    activate_shadow_campaign(
        preview,
        expected_preview_sha256=preview_sha256,
        reviewed_at="2026-08-16T12:05:00Z",
        runtime_root=runtime_root,
        approval_token=ACTIVATE_TOKEN,
    )
    store_root = tmp_path / "store"
    ConversationKnowledgeStore(store_root).migrate(backup=False)
    workflow = IdentityReviewWorkflow(store_root)
    case = preview["historical_cases"][0]
    supervisor = _complete_supervisor(store_root, case)
    supervisor_hash = supervisor.load_run("shadow-run-001")["content_hash"]
    result = {
        "case_id": case["case_id"],
        "conversation_id": case["conversation_id"],
        "recording_id": case["recording_id"],
        "source_artifact_sha256": case["source_artifact_sha256"],
        "source_media_sha256": case["source_media_sha256"],
        "processing_run_id": "shadow-run-001",
        "status": "complete",
        "attempt_count": 1,
        "provider_reads": {"succeeded": 2, "failed": 1, "transient_retries": 0},
        "latency_ms": 425,
        "duplicate_suppressed": False,
        "knowledge_integrity": "preserved",
        "effect_policy": preview["effect_policy"],
        "queue_item": _queue_item(case),
        "completed_at": "2026-08-16T13:00:00Z",
    }

    recorded = record_shadow_case(
        preview["campaign_id"],
        result,
        runtime_root=runtime_root,
        review_workflow=workflow,
        evidence_supervisor=supervisor,
    )
    replayed = record_shadow_case(
        preview["campaign_id"],
        result,
        runtime_root=runtime_root,
        review_workflow=workflow,
        evidence_supervisor=supervisor,
    )

    assert recorded == replayed
    assert recorded["status"] == "complete"
    assert recorded["supervisor_run_content_hash"] == supervisor_hash
    assert recorded["effect_policy"] == preview["effect_policy"]
    assert stat.S_IMODE(recorded["case_receipt_path"].stat().st_mode) == 0o600
    assert workflow.get_queue_item(result["queue_item"]["queue_item_id"])[
        "original_recording_filename"
    ] == case["original_recording_filename"]

    with pytest.raises(IdentityShadowCampaignError, match="zero-effect"):
        record_shadow_case(
            preview["campaign_id"],
            {
                **result,
                "effect_policy": {
                    **preview["effect_policy"],
                    "provider_write_count": 1,
                },
            },
            runtime_root=runtime_root,
            review_workflow=workflow,
            evidence_supervisor=supervisor,
        )
    with pytest.raises(IdentityShadowCampaignError, match="supervisor zero effects"):
        record_shadow_case(
            preview["campaign_id"],
            result,
            runtime_root=runtime_root,
            review_workflow=workflow,
            evidence_supervisor=_StubSupervisor(
                {
                    **_supervisor_run(1),
                    "content_hash": supervisor_hash,
                    "provider_retry_count": 0,
                    "effect_counts": {"identity": 1},
                }
            ),
        )


def test_new_arrival_registration_is_window_bound_and_replayable(tmp_path) -> None:
    preview = preview_shadow_campaign(
        [_candidate(index) for index in range(25)],
        activated_at="2026-08-16T12:00:00Z",
    )
    preview_sha256 = hashlib.sha256(
        json.dumps(preview, separators=(",", ":"), sort_keys=True).encode("utf-8")
    ).hexdigest()
    runtime_root = tmp_path / "shadow"
    activate_shadow_campaign(
        preview,
        expected_preview_sha256=preview_sha256,
        reviewed_at="2026-08-16T12:05:00Z",
        runtime_root=runtime_root,
        approval_token=ACTIVATE_TOKEN,
    )
    arrival = {
        **_candidate(100, cohort="new_arrival"),
        "artifact_stabilized_at": "2026-08-20T09:00:00Z",
    }

    registered = register_new_arrival(
        preview["campaign_id"], arrival, runtime_root=runtime_root
    )
    replayed = register_new_arrival(
        preview["campaign_id"], arrival, runtime_root=runtime_root
    )

    assert registered == replayed
    assert registered["status"] == "registered_for_shadow"
    assert stat.S_IMODE(registered["registration_path"].stat().st_mode) == 0o600

    with pytest.raises(IdentityShadowCampaignError, match="outside the frozen window"):
        register_new_arrival(
            preview["campaign_id"],
            {
                **_candidate(101, cohort="new_arrival"),
                "artifact_stabilized_at": "2026-08-23T12:00:00Z",
            },
            runtime_root=runtime_root,
        )


def _terminal_result(case: dict[str, object], index: int) -> dict[str, object]:
    return {
        "case_id": case["case_id"],
        "conversation_id": case["conversation_id"],
        "recording_id": case["recording_id"],
        "source_artifact_sha256": case["source_artifact_sha256"],
        "source_media_sha256": case["source_media_sha256"],
        "processing_run_id": f"shadow-run-{index:03d}",
        "status": "complete" if index < 24 else "partial",
        "attempt_count": 1,
        "provider_reads": {
            "succeeded": 2,
            "failed": 1 if index == 24 else 0,
            "transient_retries": 1 if index == 24 else 0,
        },
        "latency_ms": 100 + index,
        "duplicate_suppressed": index == 23,
        "knowledge_integrity": "preserved",
        "effect_policy": {
            "accepted_identity_effect_count": 0,
            "accepted_profile_effect_count": 0,
            "provider_write_count": 0,
            "raw_deletion_count": 0,
        },
        "completed_at": f"2026-08-16T13:{index:02d}:00Z",
    }


def test_finalize_requires_full_window_and_emits_replayable_scorecard(tmp_path) -> None:
    preview = preview_shadow_campaign(
        [_candidate(index) for index in range(25)],
        activated_at="2026-08-16T12:00:00Z",
    )
    preview_sha256 = hashlib.sha256(
        json.dumps(preview, separators=(",", ":"), sort_keys=True).encode("utf-8")
    ).hexdigest()
    runtime_root = tmp_path / "shadow"
    activate_shadow_campaign(
        preview,
        expected_preview_sha256=preview_sha256,
        reviewed_at="2026-08-16T12:05:00Z",
        runtime_root=runtime_root,
        approval_token=ACTIVATE_TOKEN,
    )
    registered = register_new_arrival(
        preview["campaign_id"],
        {
            **_candidate(100, cohort="new_arrival"),
            "artifact_stabilized_at": "2026-08-20T09:00:00Z",
        },
        runtime_root=runtime_root,
    )
    evaluation_metrics = {
        "candidate_recall": {"status": "unavailable", "reason": "No reviewed outcomes yet."},
        "correctness": {"status": "unavailable", "reason": "No reviewed outcomes yet."},
        "calibration": {"status": "unavailable", "reason": "Below the 30-outcome minimum."},
        "high_strength_errors": {"status": "measured", "value": 0, "denominator": 0},
        "abstention": {"status": "measured", "value": 4, "denominator": 25},
        "review_load": {"status": "measured", "value": 25, "denominator": 25},
        "workflow_usability": {
            "status": "unavailable",
            "reason": "Operator review has not been completed.",
        },
    }

    with pytest.raises(IdentityShadowCampaignError, match="seven-day window"):
        finalize_shadow_campaign(
            preview["campaign_id"],
            observed_through="2026-08-20T12:00:00Z",
            finalized_at="2026-08-23T12:05:00Z",
            evaluation_metrics=evaluation_metrics,
            runtime_root=runtime_root,
            approval_token=FINALIZE_TOKEN,
        )

    for index, case in enumerate(preview["historical_cases"]):
        record_shadow_case(
            preview["campaign_id"],
            _terminal_result(case, index),
            runtime_root=runtime_root,
            evidence_supervisor=_StubSupervisor(_supervisor_run(index)),
        )
    record_shadow_case(
        preview["campaign_id"],
        _terminal_result(registered["case"], 25),
        runtime_root=runtime_root,
        evidence_supervisor=_StubSupervisor(_supervisor_run(25)),
    )

    window_receipt = finalize_shadow_campaign(
        preview["campaign_id"],
        observed_through="2026-08-23T12:00:00Z",
        finalized_at="2026-08-23T12:05:00Z",
        evaluation_metrics=evaluation_metrics,
        runtime_root=runtime_root,
        approval_token=FINALIZE_TOKEN,
    )
    replayed_window = replay_shadow_campaign(
        preview["campaign_id"], runtime_root=runtime_root
    )

    assert window_receipt == replayed_window
    assert window_receipt["status"] == "shadow_window_complete_pending_review"
    assert window_receipt["scorecard"]["pipeline_yield"] == {
        "processed_count": 26,
        "total_count": 26,
        "rate": 1.0,
    }
    assert window_receipt["scorecard"]["provider_yield"] == {
        "succeeded": 52,
        "failed": 1,
        "transient_retries": 1,
    }
    assert window_receipt["scorecard"]["duplicate_control"] == {
        "suppressed_count": 1,
        "total_count": 26,
    }
    assert window_receipt["scorecard"]["knowledge_integrity"] == {
        "preserved_count": 26,
        "violation_count": 0,
    }
    assert window_receipt["evaluation_metrics"] == evaluation_metrics
    assert window_receipt["effect_policy"] == preview["effect_policy"]
    assert stat.S_IMODE(window_receipt["campaign_receipt_path"].stat().st_mode) == 0o600

    with pytest.raises(IdentityShadowCampaignError, match="already closed"):
        register_new_arrival(
            preview["campaign_id"],
            {
                **_candidate(102, cohort="new_arrival"),
                "artifact_stabilized_at": "2026-08-20T10:00:00Z",
            },
            runtime_root=runtime_root,
        )

    reviewed_metrics = {
        **evaluation_metrics,
        "candidate_recall": {"status": "measured", "value": 0.92, "denominator": 26},
        "correctness": {"status": "measured", "value": 0.96, "denominator": 26},
        "workflow_usability": {"status": "measured", "value": True},
    }
    terminal = finalize_shadow_campaign(
        preview["campaign_id"],
        observed_through="2026-08-23T12:00:00Z",
        finalized_at="2026-08-23T13:05:00Z",
        evaluation_metrics=reviewed_metrics,
        runtime_root=runtime_root,
        approval_token=FINALIZE_TOKEN,
    )
    replayed_terminal = replay_shadow_campaign(
        preview["campaign_id"], runtime_root=runtime_root
    )
    assert terminal == replayed_terminal
    assert terminal["status"] == "shadow_window_complete_reviewed"
    assert terminal["predecessor_scorecard_sha256"]


def test_redacted_cli_preview_is_stable_and_read_only() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [
            sys.executable,
            str(repo_root / "scripts/plan0072_a6_shadow.py"),
            "preview",
            str(
                repo_root
                / "docs/dev/fixtures/plan-0072-a6/redacted-candidates.json"
            ),
            "--activated-at",
            "2026-08-16T12:00:00Z",
        ],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )

    preview = json.loads(result.stdout)
    assert preview["campaign_id"] == "identity-shadow-062175592588b3529e27376b"
    assert len(preview["historical_cases"]) == 25
    assert len(preview["new_arrival_cases"]) == 1
    assert preview["effect_policy"]["provider_write_count"] == 0
