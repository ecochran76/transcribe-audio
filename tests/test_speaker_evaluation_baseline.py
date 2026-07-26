from __future__ import annotations

import json

import pytest

import speaker_evaluation_baseline


def _prepared(run_id: str) -> dict[str, object]:
    return {
        "run_id": run_id,
        "route": {"provider": "codex-app-server", "model": "gpt-5.6-sol"},
        "prompt_packet": {"packet_id": f"packet-{run_id}"},
    }


def test_case_runner_attempts_only_one_clue_reference_repair(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = speaker_evaluation_baseline.LocalSpeakerCaseRunner()
    prepare_evaluation_calls = 0
    repair_calls = 0

    def fake_post(path: str, payload: dict[str, object]) -> dict[str, object]:
        nonlocal prepare_evaluation_calls, repair_calls
        if path.endswith("/prepare-discovery"):
            return _prepared("discovery-run")
        if path.endswith("/prepare-reference-repair"):
            repair_calls += 1
            assert payload["phase"] == "clue_discovery"
            assert payload["original_run_id"] == "discovery-run"
            return _prepared("discovery-repair-run")
        if path.endswith("/prepare-evaluation"):
            prepare_evaluation_calls += 1
            raise ValueError("unprepared transcript clues")
        raise AssertionError(f"Unexpected POST: {path}")

    outputs = iter(
        [
            {"output_text": json.dumps({"original": True})},
            {"output_text": json.dumps({"corrected": True})},
        ]
    )
    monkeypatch.setattr(runner, "_post", fake_post)
    monkeypatch.setattr(runner, "_execute_prepared", lambda prepared: next(outputs))

    with pytest.raises(
        speaker_evaluation_baseline.CasePredictionFailure,
        match="unprepared transcript clues",
    ) as caught:
        runner("doc-1")

    assert caught.value.stage == "clue_discovery_validation"
    assert prepare_evaluation_calls == 2
    assert repair_calls == 1
    assert caught.value.run_references == {
        "clue_discovery_run_id": "discovery-run",
        "clue_discovery_repair_run_id": "discovery-repair-run",
    }


def test_case_runner_does_not_prepare_repair_for_valid_first_pass(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = speaker_evaluation_baseline.LocalSpeakerCaseRunner()
    paths: list[str] = []

    def fake_post(path: str, payload: dict[str, object]) -> dict[str, object]:
        paths.append(path)
        if path.endswith("/prepare-discovery"):
            return _prepared("discovery-run")
        if path.endswith("/prepare-evaluation"):
            return _prepared("identity-run")
        if path.endswith("/capture-evaluation"):
            assert "readout" not in payload
            return {
                "record": {
                    "current_evaluation_id": "evaluation-1",
                    "evaluations": [
                        {
                            "evaluation_id": "evaluation-1",
                            "status": "awaiting_human_confirmation",
                        }
                    ],
                }
            }
        raise AssertionError(f"Unexpected POST: {path}")

    monkeypatch.setattr(runner, "_post", fake_post)
    monkeypatch.setattr(
        runner,
        "_execute_prepared",
        lambda prepared: {"output_text": json.dumps({"valid": True})},
    )

    result = runner("doc-1")

    assert result["prediction"]["evaluation_id"] == "evaluation-1"
    assert not any(path.endswith("/prepare-reference-repair") for path in paths)


def test_case_runner_attempts_only_one_identity_reference_repair(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = speaker_evaluation_baseline.LocalSpeakerCaseRunner()
    capture_calls = 0
    repair_calls = 0

    def fake_post(path: str, payload: dict[str, object]) -> dict[str, object]:
        nonlocal capture_calls, repair_calls
        if path.endswith("/prepare-discovery"):
            return _prepared("discovery-run")
        if path.endswith("/prepare-evaluation"):
            return _prepared("identity-run")
        if path.endswith("/prepare-reference-repair"):
            repair_calls += 1
            assert payload["phase"] == "identity_evaluation"
            assert payload["original_run_id"] == "identity-run"
            return _prepared("identity-repair-run")
        if path.endswith("/capture-evaluation"):
            capture_calls += 1
            raise ValueError("unprepared provenance source")
        raise AssertionError(f"Unexpected POST: {path}")

    outputs = iter(
        [
            {"output_text": json.dumps({"discovery": True})},
            {"output_text": json.dumps({"identity": True})},
            {"output_text": json.dumps({"corrected": True})},
        ]
    )
    monkeypatch.setattr(runner, "_post", fake_post)
    monkeypatch.setattr(runner, "_execute_prepared", lambda prepared: next(outputs))

    with pytest.raises(
        speaker_evaluation_baseline.CasePredictionFailure,
        match="unprepared provenance source",
    ) as caught:
        runner("doc-1")

    assert caught.value.stage == "identity_evaluation_validation"
    assert capture_calls == 2
    assert repair_calls == 1
    assert caught.value.run_references == {
        "clue_discovery_run_id": "discovery-run",
        "identity_evaluation_run_id": "identity-run",
        "identity_evaluation_repair_run_id": "identity-repair-run",
    }
