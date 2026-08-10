from __future__ import annotations

import json
from types import SimpleNamespace
from uuid import uuid4

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


def test_openai_fallback_records_host_ledger_events(monkeypatch, tmp_path):
    events = []
    prompt_path = tmp_path / "prompt.txt"
    prompt_path.write_text("Return JSON.", encoding="utf-8")
    prompt_path.chmod(0o600)
    monkeypatch.setattr(
        p2.app_intelligence_ledger,
        "append_event",
        lambda **kwargs: events.append(kwargs) or {},
    )
    monkeypatch.setattr(
        p2.requests,
        "post",
        lambda *_args, **_kwargs: SimpleNamespace(
            status_code=200,
            json=lambda: {
                "choices": [{"message": {"content": '{"schema_version":"test"}'}}]
            },
        ),
    )
    runner = p2.OpenAICompatibleCaseRunner(
        api_key="test-key", base_url="https://example.invalid/v1",
        state_root=tmp_path,
    )
    readout = runner._direct_readout(
        {
            "run_id": "run-1",
            "prompt_path": str(prompt_path),
            "prompt_packet": {"prompt_path": str(prompt_path)},
        }
    )
    assert readout == {"schema_version": "test"}
    assert [event["event_type"] for event in events] == [
        "model_turn_fallback_started",
        "model_turn_fallback_completed",
    ]
    assert all(event["payload"]["provider"] == "openai-compatible" for event in events)


def test_openai_fallback_http_error_fails_closed(monkeypatch, tmp_path):
    prompt_path = tmp_path / "prompt.txt"
    prompt_path.write_text("Return JSON.", encoding="utf-8")
    prompt_path.chmod(0o600)
    monkeypatch.setattr(p2.app_intelligence_ledger, "append_event", lambda **_kwargs: {})
    monkeypatch.setattr(
        p2.requests,
        "post",
        lambda *_args, **_kwargs: SimpleNamespace(status_code=429),
    )
    runner = p2.OpenAICompatibleCaseRunner(api_key="test-key", state_root=tmp_path)
    with pytest.raises(p2.Plan0064P2Error, match="HTTP 429"):
        runner._direct_readout(
            {
                "run_id": "run-1",
                "prompt_path": str(prompt_path),
                "prompt_packet": {"prompt_path": str(prompt_path)},
            }
        )


def _hydration_fixture(tmp_path, monkeypatch):
    runtime = tmp_path / "runtime"
    frozen_root = runtime / "p0-frozen"
    frozen_root.mkdir(parents=True, mode=0o700)
    transcript_path = tmp_path / "transcript.json"
    conversation_id = str(uuid4())
    recording_id = str(uuid4())
    transcript_path.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "conversation_id": conversation_id,
                "recording_id": recording_id,
                "utterances": [{"speaker": "A", "text": "hello"}],
            }
        ),
        encoding="utf-8",
    )
    new_hash = p2.sha256_file(transcript_path)
    old_row = {
        "document_id": "doc-1",
        "disposition": "selected_evaluation_candidate",
        "speaker_labels": ["A"],
        "artifact_sha256": "a" * 64,
        "conversation_id": None,
        "recording_id": None,
        "transcript_artifact": {
            "path": str(transcript_path),
            "sha256": "a" * 64,
        },
    }
    new_row = {
        **old_row,
        "artifact_sha256": new_hash,
        "conversation_id": conversation_id,
        "recording_id": recording_id,
        "transcript_artifact": {
            "path": str(transcript_path),
            "sha256": new_hash,
        },
    }
    frozen = {
        "schema_version": "p0-test",
        "repository_authority": {"commit": "b" * 40},
        "evaluation_cohort": {
            "cohort_sha256": "c" * 64,
            "selected_count": 1,
            "considered": [old_row],
        },
        "content_sha256": "d" * 64,
    }
    current = {
        **frozen,
        "evaluation_cohort": {
            **frozen["evaluation_cohort"],
            "cohort_sha256": "e" * 64,
            "considered": [new_row],
        },
        "content_sha256": "f" * 64,
    }
    manifest_path = frozen_root / "manifest.json"
    manifest_path.write_text(json.dumps(frozen), encoding="utf-8")
    manifest_path.chmod(0o600)
    receipt = {
        "manifest_content_sha256": frozen["content_sha256"],
        "manifest_file_sha256": p2.sha256_file(manifest_path),
        "action_counts": dict(p2.ACTION_COUNTS),
        "content_sha256": "1" * 64,
    }
    receipt_path = frozen_root / "receipt.json"
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    receipt_path.chmod(0o600)
    monkeypatch.setattr(
        p2,
        "replay_p0",
        lambda **_kwargs: (_ for _ in ()).throw(
            p2.Plan0064P0Error("p0_live_state_drift", "expected hydration")
        ),
    )
    monkeypatch.setattr(
        p2,
        "p0_paths",
        lambda *_args: {"root": frozen_root, "manifest": manifest_path, "receipt": receipt_path},
    )
    monkeypatch.setattr(p2, "validate_p0_manifest", lambda _manifest: frozen["content_sha256"])
    monkeypatch.setattr(p2, "build_p0_manifest", lambda **_kwargs: current)
    return runtime, frozen, current


def test_phase_safe_p0_accepts_only_synchronized_identity_hydration(
    tmp_path, monkeypatch
):
    runtime, frozen, _current = _hydration_fixture(tmp_path, monkeypatch)
    replay, bridge = p2._phase_safe_p0(
        frozen["content_sha256"], runtime_root=runtime
    )
    assert replay["status"] == "p0_frozen_with_validated_identity_hydration"
    assert bridge["status"] == "validated_transcript_identity_hydration"
    assert bridge["changed_recording_count"] == 1
    assert bridge["preserved_speaker_slot_count"] == 1
    assert bridge["action_counts"] == p2.ACTION_COUNTS


def test_phase_safe_p0_rejects_non_identity_row_drift(tmp_path, monkeypatch):
    runtime, frozen, current = _hydration_fixture(tmp_path, monkeypatch)
    current["evaluation_cohort"]["considered"][0]["speaker_labels"] = ["B"]
    with pytest.raises(
        p2.Plan0064P2Error, match="outside sanctioned identity hydration"
    ):
        p2._phase_safe_p0(frozen["content_sha256"], runtime_root=runtime)
