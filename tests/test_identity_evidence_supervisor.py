from __future__ import annotations

import copy
import json
import sqlite3
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import conversation_knowledge_store
from identity_evidence_supervisor import (
    EvidencePillarSpec,
    IdentityEvidenceSupervisor,
)


def _supervisor(tmp_path: Path) -> IdentityEvidenceSupervisor:
    conversation_knowledge_store.ConversationKnowledgeStore(tmp_path).migrate(
        backup=False
    )
    return IdentityEvidenceSupervisor(tmp_path)


def _a4_fixture() -> dict[str, object]:
    path = (
        Path(__file__).resolve().parents[1]
        / "docs"
        / "dev"
        / "fixtures"
        / "plan-0072-a4"
        / "supervisor-replay.json"
    )
    return json.loads(path.read_text(encoding="utf-8"))


def _run_payload() -> dict[str, object]:
    return {
        "schema_version": "transcribe-audio.processing-run.v1",
        "run_id": "run-a4-redacted-1",
        "conversation_id": "conversation-a4-redacted-1",
        "recording_id": "recording-a4-redacted-1",
        "original_recording_filename": "2026-08-16 Redacted A4 Example.m4a",
        "source_artifact_sha256": "a" * 64,
        "source_media_sha256": "b" * 64,
        "operation_mode": "contract_fixture",
        "policy_version": "supervisor-policy.synthetic.v1",
        "as_of": "2026-08-16T20:00:00Z",
        "capabilities": ["event_metadata_read", "contact_search"],
        "source_scopes": [
            {
                "provider_kind": "synthetic",
                "profile_id": "profile-redacted-a4",
                "account_id": "account-redacted-a4",
                "tenant_id": "tenant-redacted-a4",
                "capabilities": [
                    "event_metadata_read",
                    "contact_search",
                ],
            }
        ],
        "budgets": {
            "max_records": 4,
            "max_characters": 120,
            "max_calls": 3,
            "max_latency_ms": 500,
        },
        "model_versions": {"context": "context.synthetic.v1"},
        "rubric_versions": {
            "calendar_association": "calendar.synthetic.v1",
            "person_link": "person.synthetic.v1",
            "contextual_speaker": "context.synthetic.v1",
            "acoustic": "acoustic.synthetic.v1",
            "combined": "combined.synthetic.v1",
        },
        "profile_versions": [],
        "stage": "bind_conversation",
        "state": "running",
        "transcript_correction_passes": 0,
        "identity_cascade_count": 0,
        "provider_retry_count": 0,
        "input_ids": ["input-redacted-1"],
        "output_ids": [],
        "failures": [],
        "effect_counts": {
            "accepted_identity": 0,
            "profile_activation": 0,
            "provider_write": 0,
        },
        "created_at": "2026-08-16T20:00:00Z",
    }


def _adapter_exchange(
    *,
    request_id: str,
    capability: str,
    status: str = "complete",
    failure: dict[str, str] | None = None,
    records: int = 1,
    characters: int = 20,
    calls: int = 1,
    latency_ms: int = 40,
) -> tuple[dict[str, object], dict[str, object]]:
    scope = {
        "provider_kind": "synthetic",
        "profile_id": "profile-redacted-a4",
        "account_id": "account-redacted-a4",
        "tenant_id": "tenant-redacted-a4",
        "capabilities": [capability],
    }
    request = {
        "schema_version": "transcribe-audio.provider-adapter-request.v1",
        "request_id": request_id,
        "processing_run_id": "run-a4-redacted-1",
        "conversation_id": "conversation-a4-redacted-1",
        "adapter_id": "adapter.synthetic.v1",
        "capability": capability,
        "source_scope": scope,
        "as_of": "2026-08-16T20:00:00Z",
        "query": {"terms": ["redacted"]},
        "budgets": {
            "max_records": 4,
            "max_characters": 120,
            "max_calls": 1,
            "max_latency_ms": 500,
        },
        "mode": "read_only",
        "idempotency_key": f"adapter-{request_id}",
        "created_at": "2026-08-16T20:01:00Z",
    }
    result = {
        "schema_version": "transcribe-audio.provider-adapter-result.v1",
        "result_id": f"result-{request_id}",
        "request_id": request_id,
        "processing_run_id": "run-a4-redacted-1",
        "source_scope": scope,
        "status": status,
        "observations": [f"observation-{request_id}"] if records else [],
        "warnings": [],
        "failure": failure,
        "retrieved_at": "2026-08-16T20:01:01Z",
        "provider_write_count": 0,
        "consumed_budget": {
            "records": records,
            "characters": characters,
            "calls": calls,
            "latency_ms": latency_ms,
        },
    }
    return request, result


def test_run_ledger_replays_exactly_and_rejects_idempotency_drift(
    tmp_path: Path,
) -> None:
    supervisor = _supervisor(tmp_path)
    payload = _run_payload()

    inserted = supervisor.start_run(payload)
    replay = supervisor.start_run(payload)

    assert inserted.status == "inserted"
    assert replay.status == "unchanged"
    assert supervisor.load_run(str(payload["run_id"]))["artifact"] == payload
    drifted = copy.deepcopy(payload)
    drifted["as_of"] = "2026-08-16T20:02:00Z"
    with pytest.raises(ValueError, match="already has different inputs"):
        supervisor.start_run(drifted)


def test_adapter_exchanges_enforce_scope_budget_retry_and_partial_isolation(
    tmp_path: Path,
) -> None:
    supervisor = _supervisor(tmp_path)
    supervisor.start_run(_run_payload())
    first_request, first_result = _adapter_exchange(
        request_id="request-calendar-1",
        capability="event_metadata_read",
    )
    partial_request, partial_result = _adapter_exchange(
        request_id="request-contact-1",
        capability="contact_search",
        status="partial",
        failure={"reason_code": "transient_timeout", "detail": "synthetic"},
        records=1,
        characters=30,
    )

    supervisor.record_adapter_exchange(first_request, first_result, attempt=0)
    replay = supervisor.record_adapter_exchange(
        first_request,
        first_result,
        attempt=0,
    )
    assert replay.status == "unchanged"
    supervisor.record_adapter_exchange(
        partial_request,
        partial_result,
        attempt=0,
    )
    retry_request, retry_result = _adapter_exchange(
        request_id="request-contact-2",
        capability="contact_search",
        records=1,
        characters=30,
    )
    retry = supervisor.record_adapter_exchange(
        retry_request,
        retry_result,
        attempt=1,
    )

    exchanges = supervisor.load_adapter_exchanges("run-a4-redacted-1")
    assert retry.status == "inserted"
    assert [item["result"]["status"] for item in exchanges] == [
        "complete",
        "partial",
        "complete",
    ]
    assert exchanges[0]["result"]["observations"] == [
        "observation-request-calendar-1"
    ]
    too_many_request, too_many_result = _adapter_exchange(
        request_id="request-contact-3",
        capability="contact_search",
        records=0,
        characters=0,
    )
    with pytest.raises(ValueError, match="budget"):
        supervisor.record_adapter_exchange(
            too_many_request,
            too_many_result,
            attempt=0,
        )
    retry_result["provider_write_count"] = 1
    with pytest.raises(ValueError, match="provider writes"):
        supervisor.record_adapter_exchange(
            {**retry_request, "request_id": "request-contact-write"},
            {**retry_result, "request_id": "request-contact-write"},
            attempt=0,
        )


def test_participant_hypotheses_remain_nonbinding_and_adapter_scope_is_exact(
    tmp_path: Path,
) -> None:
    supervisor = _supervisor(tmp_path)
    supervisor.start_run(_run_payload())
    candidate = {
        "schema_version": "transcribe-audio.conversation-association-candidate.v1",
        "candidate_id": "calendar-candidate-a4-1",
        "conversation_id": "conversation-a4-redacted-1",
        "recording_id": "recording-a4-redacted-1",
        "event_source_record_id": "event-record-redacted-1",
        "rank": 1,
        "evidence_assessment_id": "assessment-calendar-redacted-1",
        "evidence_strength_score": 78,
        "evidence_strength_band": "strong",
        "positive_factors": ["time_overlap"],
        "negative_factors": [],
        "alternatives": ["no_matching_event"],
        "provider_warnings": [],
        "rubric_version": "calendar.synthetic.v1",
        "as_of": "2026-08-16T20:00:00Z",
        "status": "candidate",
        "created_at": "2026-08-16T20:02:00Z",
    }
    hypothesis = {
        "schema_version": "transcribe-audio.participant-hypothesis.v1",
        "hypothesis_id": "participant-hypothesis-a4-1",
        "conversation_id": "conversation-a4-redacted-1",
        "person_id": "person-redacted-a4-1",
        "source_record_id": "contact-redacted-a4-1",
        "kind": "calendar_attendee",
        "source_observation_ids": ["observation-redacted-a4-1"],
        "evidence_assessment_id": "assessment-participant-redacted-a4-1",
        "status": "hypothesis",
        "created_at": "2026-08-16T20:02:00Z",
    }
    purpose = {
        "schema_version": "transcribe-audio.conversation-purpose-hypothesis.v1",
        "hypothesis_id": "purpose-hypothesis-a4-1",
        "conversation_id": "conversation-a4-redacted-1",
        "label": "redacted planning discussion",
        "alternatives": ["unresolved"],
        "evidence_ids": ["observation-redacted-a4-1"],
        "status": "hypothesis",
        "created_at": "2026-08-16T20:02:00Z",
    }

    supervisor.record_conversation_candidate("run-a4-redacted-1", candidate)
    supervisor.record_purpose_hypothesis("run-a4-redacted-1", purpose)
    supervisor.record_participant_hypothesis("run-a4-redacted-1", hypothesis)

    loaded = supervisor.load_hypotheses("run-a4-redacted-1")
    assert loaded["participants"][0]["status"] == "hypothesis"
    assert loaded["purposes"][0]["status"] == "hypothesis"
    assert "assignment" not in json.dumps(loaded)
    request, result = _adapter_exchange(
        request_id="request-out-of-scope",
        capability="drive_search",
    )
    with pytest.raises(ValueError, match="outside run capabilities"):
        supervisor.record_adapter_exchange(request, result, attempt=0)
    scoped_request, scoped_result = _adapter_exchange(
        request_id="request-wrong-tenant",
        capability="contact_search",
    )
    wrong_scope = {
        **scoped_request["source_scope"],
        "tenant_id": "another-tenant",
    }
    scoped_request["source_scope"] = wrong_scope
    scoped_result["source_scope"] = wrong_scope
    with pytest.raises(ValueError, match="outside run source scopes"):
        supervisor.record_adapter_exchange(
            scoped_request,
            scoped_result,
            attempt=0,
        )


def _pillars(
    *,
    contradiction: bool = False,
    duplicate_group: bool = False,
) -> tuple[EvidencePillarSpec, ...]:
    return (
        EvidencePillarSpec(
            pillar="calendar_association",
            score=80,
            positive_factors=("calendar_time_fit",),
            negative_factors=(),
            evidence_ids=("evidence-calendar-1",),
            independence_groups=("calendar-source-1",),
        ),
        EvidencePillarSpec(
            pillar="person_link",
            score=70,
            positive_factors=("verified_identifier",),
            negative_factors=(),
            evidence_ids=("evidence-person-1",),
            independence_groups=("contact-source-1",),
        ),
        EvidencePillarSpec(
            pillar="contextual_speaker",
            score=90,
            positive_factors=("self_introduction",),
            negative_factors=("material_contradiction",) if contradiction else (),
            evidence_ids=("evidence-context-1",),
            independence_groups=("transcript-source-1",),
            material_contradiction=contradiction,
        ),
        EvidencePillarSpec(
            pillar="acoustic",
            score=60,
            positive_factors=("reviewed_profile_similarity",),
            negative_factors=(),
            evidence_ids=("evidence-acoustic-1",),
            independence_groups=(
                "transcript-source-1" if duplicate_group else "acoustic-source-1",
            ),
        ),
    )


def test_pillar_scores_keep_lineage_and_material_contradiction_caps_combined(
    tmp_path: Path,
) -> None:
    supervisor = _supervisor(tmp_path)
    supervisor.start_run(_run_payload())

    first = supervisor.score_candidate(
        run_id="run-a4-redacted-1",
        candidate_id="person-redacted-a4-1",
        pillars=_pillars(),
        rubric_version="combined.synthetic.v1",
        model_version="host-deterministic.v1",
        created_at="2026-08-16T20:03:00Z",
    )
    assert (
        supervisor.score_candidate(
            run_id="run-a4-redacted-1",
            candidate_id="person-redacted-a4-1",
            pillars=_pillars(),
            rubric_version="combined.synthetic.v1",
            model_version="host-deterministic.v1",
            created_at="2026-08-16T20:03:00Z",
        ).status
        == "unchanged"
    )
    second = supervisor.score_candidate(
        run_id="run-a4-redacted-1",
        candidate_id="person-redacted-a4-1",
        pillars=_pillars(contradiction=True),
        rubric_version="combined.synthetic.v2",
        model_version="host-deterministic.v1",
        created_at="2026-08-16T20:04:00Z",
        predecessor_assessment_id=first.combined_assessment_id,
    )
    third = supervisor.score_candidate(
        run_id="run-a4-redacted-1",
        candidate_id="person-redacted-a4-1",
        pillars=_pillars(duplicate_group=True),
        rubric_version="combined.synthetic.v3",
        model_version="host-deterministic.v1",
        created_at="2026-08-16T20:05:00Z",
        predecessor_assessment_id=second.combined_assessment_id,
    )

    history = supervisor.load_assessment_history(
        "run-a4-redacted-1",
        "person-redacted-a4-1",
    )
    assert first.combined_score == 75
    assert second.combined_score == 49
    assert second.review_required is True
    assert third.combined_score == 49
    assert third.review_required is True
    assert history[-1]["predecessor_assessment_id"] == (
        second.combined_assessment_id
    )
    assert "duplicate_evidence_group_cap" in history[-1]["reason_codes"]
    assert {item["pillar"] for item in history[-1]["pillar_assessments"]} == {
        "calendar_association",
        "person_link",
        "contextual_speaker",
        "acoustic",
    }


def test_calibrated_likelihood_requires_30_source_disjoint_outcomes(
    tmp_path: Path,
) -> None:
    supervisor = _supervisor(tmp_path)
    for index in range(29):
        supervisor.record_calibration_outcome(
            pillar="combined",
            score_band="70-79",
            correct=index < 23,
            source_disjoint_id=f"source-disjoint-{index:02d}",
            evaluation_version="evaluation.synthetic.v1",
            review_decision_id=f"decision-redacted-{index:02d}",
            created_at="2026-08-16T20:05:00Z",
        )
    unavailable = supervisor.calibrated_likelihood(
        pillar="combined",
        score_band="70-79",
        evaluation_version="evaluation.synthetic.v1",
        created_at="2026-08-16T20:06:00Z",
    )
    supervisor.record_calibration_outcome(
        pillar="combined",
        score_band="70-79",
        correct=True,
        source_disjoint_id="source-disjoint-29",
        evaluation_version="evaluation.synthetic.v1",
        review_decision_id="decision-redacted-29",
        created_at="2026-08-16T20:05:00Z",
    )
    available = supervisor.calibrated_likelihood(
        pillar="combined",
        score_band="70-79",
        evaluation_version="evaluation.synthetic.v1",
        created_at="2026-08-16T20:07:00Z",
    )

    assert unavailable.status == "insufficient_data"
    assert unavailable.sample_size == 29
    assert unavailable.likelihood is None
    assert available.status == "available"
    assert available.sample_size == 30
    assert available.likelihood == pytest.approx(0.8)
    assert 0 <= available.interval_low < available.likelihood
    assert available.likelihood < available.interval_high <= 1
    with pytest.raises(ValueError, match="source-disjoint outcome"):
        supervisor.record_calibration_outcome(
            pillar="combined",
            score_band="70-79",
            correct=False,
            source_disjoint_id="source-disjoint-29",
            evaluation_version="evaluation.synthetic.v1",
            review_decision_id="decision-drifted",
            created_at="2026-08-16T20:08:00Z",
        )


def test_supervisor_stage_history_is_sequential_and_zero_effect(
    tmp_path: Path,
) -> None:
    supervisor = _supervisor(tmp_path)
    supervisor.start_run(_run_payload())
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
        failures = (
            ({"adapter_id": "synthetic", "reason_code": "partial"},)
            if stage == "participant_and_evidence_collection"
            else ()
        )
        receipt = supervisor.advance_stage(
            run_id="run-a4-redacted-1",
            stage=stage,
            state="complete" if stage == "complete" else "running",
            output_ids=(f"output-{index}",),
            failures=failures,
            effect_counts={
                "accepted_identity": 0,
                "profile_activation": 0,
                "provider_write": 0,
            },
            idempotency_key=f"stage-a4-{index}",
            created_at=f"2026-08-16T20:{10 + index:02d}:00Z",
        )
        assert receipt.status == "inserted"
        assert (
            supervisor.advance_stage(
                run_id="run-a4-redacted-1",
                stage=stage,
                state="complete" if stage == "complete" else "running",
                output_ids=(f"output-{index}",),
                failures=failures,
                effect_counts={
                    "accepted_identity": 0,
                    "profile_activation": 0,
                    "provider_write": 0,
                },
                idempotency_key=f"stage-a4-{index}",
                created_at=f"2026-08-16T20:{10 + index:02d}:00Z",
            ).status
            == "unchanged"
        )
    state = supervisor.load_run("run-a4-redacted-1")
    history = supervisor.load_stage_history("run-a4-redacted-1")

    assert state["stage"] == "complete"
    assert state["state"] == "complete"
    assert state["effect_counts"] == {
        "accepted_identity": 0,
        "profile_activation": 0,
        "provider_write": 0,
    }
    assert any(item["failures"] for item in history)
    with pytest.raises(ValueError, match="sequential"):
        supervisor.advance_stage(
            run_id="run-a4-redacted-1",
            stage="calendar_candidate_generation",
            state="running",
            output_ids=(),
            failures=(),
            effect_counts={"provider_write": 0},
            idempotency_key="stage-out-of-order",
            created_at="2026-08-16T20:30:00Z",
        )
    with sqlite3.connect(tmp_path / "transcripts.sqlite3") as con:
        with pytest.raises(sqlite3.IntegrityError, match="append-only"):
            con.execute(
                "UPDATE knowledge_identity_supervisor_runs "
                "SET original_recording_filename = 'tampered.m4a'"
            )


def test_plan0072_a4_redacted_fixture_replays_exact_supervisor_score(
    tmp_path: Path,
) -> None:
    supervisor = _supervisor(tmp_path)
    fixture = _a4_fixture()
    run = fixture["run"]
    purpose = fixture["purpose"]
    pillars = tuple(
        EvidencePillarSpec(
            pillar=str(item["pillar"]),
            score=float(item["score"]),
            positive_factors=tuple(item["positive_factors"]),
            negative_factors=tuple(item["negative_factors"]),
            evidence_ids=tuple(item["evidence_ids"]),
            independence_groups=tuple(item["independence_groups"]),
        )
        for item in fixture["pillars"]
    )

    supervisor.start_run(run)
    supervisor.record_purpose_hypothesis(str(run["run_id"]), purpose)
    score = supervisor.score_candidate(
        run_id=str(run["run_id"]),
        candidate_id="person-a4-fixture-1",
        pillars=pillars,
        rubric_version="combined.synthetic.v1",
        model_version="host-deterministic.v1",
        created_at="2026-08-16T22:32:00Z",
    )
    expected = fixture["expected"]

    assert score.combined_score == expected["combined_score"]
    assert score.review_required is expected["review_required"]
    assert supervisor.load_run(str(run["run_id"]))["effect_counts"][
        "provider_write"
    ] == expected["provider_write_count"]
    assert supervisor.load_hypotheses(str(run["run_id"]))["purposes"] == (
        purpose,
    )
