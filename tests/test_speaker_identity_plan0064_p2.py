from __future__ import annotations

import json

import pytest

import speaker_identity_plan0064_p2 as p2


PEOPLE = {f"person-{index}" for index in range(6)}


def _prediction(*proposals):
    return {
        "prediction": {
            "evaluation_id": "evaluation-1",
            "status": "complete",
            "proposals": list(proposals),
            "warnings": [],
        },
        "run_references": {
            "clue_discovery_run_id": "run-clue",
            "identity_evaluation_run_id": "run-evaluation",
        },
    }


def _proposal(*, label="A", status="candidate_match", person_id="person-1"):
    return {
        "proposal_id": f"proposal-{label}",
        "status": status,
        "person_id": person_id,
        "speaker_labels": [label],
        "confidence": {"band": "high", "numeric": 80},
        "transcript_clue_ids": ["utterance-1"],
        "provenance_source_ids": ["source-1"],
        "review_flags": [],
        "factors": [],
    }


def test_slot_rows_emit_only_allowlisted_candidate_matches():
    rows = p2._proposal_slot_rows(
        document_id="doc",
        speaker_labels=["A", "B", "C"],
        prediction={"proposals": [
            _proposal(label="A", person_id="person-1"),
            _proposal(label="B", status="unlisted", person_id="outside"),
        ]},
        canonical_people=PEOPLE,
    )
    assert rows[0]["candidate_person_id"] == "person-1"
    assert rows[0]["disposition"] == "candidate"
    assert rows[1]["candidate_person_id"] is None
    assert rows[1]["disposition"] == "abstain"
    assert rows[2]["reason_code"] == "speaker_missing_from_context_evaluation"


def test_multiple_prepared_candidates_route_review_without_identity():
    rows = p2._proposal_slot_rows(
        document_id="doc",
        speaker_labels=["A"],
        prediction={"proposals": [
            _proposal(person_id="person-1"),
            {**_proposal(person_id="person-2"), "proposal_id": "proposal-2"},
        ]},
        canonical_people=PEOPLE,
    )
    assert rows[0]["disposition"] == "review"
    assert rows[0]["candidate_person_id"] is None


def test_failure_case_preserves_complete_slot_denominator():
    case = p2._failure_case(
        document_id="doc", speaker_labels=["A", "B"], stage="validation",
        message="bad output", run_references={"clue_discovery_run_id": "run-1"},
    )
    assert len(case["speaker_slots"]) == 2
    assert all(row["disposition"] == "unavailable" for row in case["speaker_slots"])
    assert case["action_counts"] == p2.ACTION_COUNTS


def test_execute_checkpoints_each_case_and_replays_without_new_model_turns(
    tmp_path, monkeypatch
):
    runtime = tmp_path / "runtime"
    runtime.mkdir(mode=0o700)
    manifest_path = runtime / "p0.json"
    manifest = {
        "evaluation_cohort": {
            "cohort_sha256": "c" * 64,
            "considered": [
                {
                    "disposition": "selected_evaluation_candidate",
                    "document_id": f"doc-{index}",
                    "speaker_labels": ["A", "B"],
                    "context_status": "local_calendar_context_available",
                }
                for index in range(12)
            ],
        },
        "canonical_bindings": {
            "binding_set_sha256": "b" * 64,
            "subject_bindings": [
                {
                    "person_id": f"person-{index}",
                    "identity_candidate_eligible": True,
                }
                for index in range(6)
            ],
        },
    }
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    manifest_path.chmod(0o600)
    monkeypatch.setattr(
        p2,
        "replay_p0",
        lambda **_kwargs: {
            "private_manifest_path": str(manifest_path),
            "receipt_content_sha256": "r" * 64,
        },
    )
    calls = []

    def factory():
        def runner(document_id):
            calls.append(document_id)
            return _prediction(_proposal(label="A"), _proposal(label="B", status="unresolved", person_id=""))

        return runner

    receipt = p2.execute_p2("a" * 64, runtime_root=runtime, runner_factory=factory)
    assert receipt["summary"]["recording_count"] == 12
    assert receipt["summary"]["speaker_slot_count"] == 24
    assert receipt["summary"]["app_intelligence_run_count"] == 24
    assert len(calls) == 12
    replayed = p2.replay_p2("a" * 64, runtime_root=runtime)
    assert replayed["idempotent_replay"] is True
    again = p2.execute_p2("a" * 64, runtime_root=runtime, runner_factory=factory)
    assert again["content_sha256"] == receipt["content_sha256"]
    assert len(calls) == 12
    case_paths = list((runtime / f"p2-{p2.build_p2_preview('a' * 64, runtime_root=runtime)['content_sha256'][:24]}" / "cases").glob("*.json"))
    assert len(case_paths) == 12
    assert all(path.stat().st_mode & 0o777 == 0o600 for path in case_paths)


def test_case_prediction_failure_fields_remain_bounded():
    failure = p2.CasePredictionFailure(
        "identity_evaluation_validation", "x" * 800,
        run_references={"identity_evaluation_run_id": "run-2"},
    )
    case = p2._failure_case(
        document_id="doc", speaker_labels=["A"], stage=failure.stage,
        message=str(failure), run_references=failure.run_references,
    )
    assert len(case["failure_detail"]) == 500
    assert case["failure_stage"] == "identity_evaluation_validation"


def test_unexpected_prediction_shape_fails_closed():
    with pytest.raises(p2.Plan0064P2Error, match="no prediction"):
        p2._successful_case(
            document_id="doc", speaker_labels=["A"], result={}, canonical_people=PEOPLE
        )
