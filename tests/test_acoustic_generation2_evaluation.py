from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

import acoustic_generation2_evaluation as evaluation


def _write_private(path: Path, value: object) -> str:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    path.parent.chmod(0o700)
    path.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")),
        encoding="utf-8",
    )
    path.chmod(0o600)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fixture(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, *, overlapping_subject: bool = False
) -> dict[str, Path]:
    private = tmp_path / "private"
    private.mkdir(parents=True, mode=0o700)
    recordings = []
    specs = [
        ("development", "dev", "subject-dev"),
        ("calibration", "cal", "subject-cal"),
        ("evaluation", "eval-a", "subject-profile-a" if overlapping_subject else "subject-open-a"),
        ("evaluation", "eval-b", "subject-open-b"),
    ]
    for index, (split, stem, subject) in enumerate(specs):
        source_path = private / f"{stem}.wav"
        source_path.write_bytes((f"audio-{stem}-" * 4).encode())
        source_path.chmod(0o600)
        transcript_path = private / f"{stem}.json"
        transcript_sha = _write_private(
            transcript_path, {"segments": [{"speaker": "A", "start": 0, "end": 2}]}
        )
        source_sha = hashlib.sha256(source_path.read_bytes()).hexdigest()
        recordings.append({
            "recording_id": f"recording-{index}",
            "conversation_id": f"conversation-{index}",
            "split": split,
            "source_blob": {
                "sha256": source_sha,
                "bytes": source_path.stat().st_size,
                "stored_path": str(source_path),
            },
            "transcript_lineage": {
                "current_artifact_path": str(transcript_path),
                "current_artifact_sha256": transcript_sha,
            },
            "operator_gold": {
                "gold_id": f"gold-{index}",
                "speaker_truth": [{
                    "speaker_label": "A", "outcome": "person", "subject_id": subject,
                }],
                "same_person_label_groups": [],
            },
        })
    corpus = {
        "schema_version": "test-successor-corpus.v1",
        "corpus_id": "test-corpus",
        "content_sha256": "1" * 64,
        "recordings": recordings,
    }
    corpus_path = private / corpus["corpus_id"] / "manifest.json"
    corpus_file_sha = _write_private(corpus_path, corpus)
    evaluation_records = sorted(
        [item for item in recordings if item["split"] == "evaluation"],
        key=lambda item: item["recording_id"],
    )
    safe_membership = [{
        "recording_id": item["recording_id"],
        "conversation_id": item["conversation_id"],
        "source_sha256": item["source_blob"]["sha256"],
        "split": "evaluation",
    } for item in evaluation_records]
    preview = {
        "preview_id": "generation-2-pre-reveal-test",
        "content_sha256": "2" * 64,
        "successor_seal": {
            "corpus_manifest_sha256": corpus_file_sha,
            "corpus_content_sha256": corpus["content_sha256"],
            "corpus_id": corpus["corpus_id"],
            "split_counts": {"development": 1, "calibration": 1, "evaluation": 2},
            "evaluation_recording_count": 2,
            "evaluation_record_set_sha256": evaluation._canonical_hash(safe_membership),
        },
        "profiles": [
            {"profile_id": "profile-a", "person_ref_id": "subject-profile-a"},
            {"profile_id": "profile-b", "person_ref_id": "subject-profile-b"},
        ],
        "candidate_matrix": [{
            "candidate_id": "candidate-a", "method_id": "no_enhancement",
            "profile_ids": ["profile-a", "profile-b"],
        }],
        "minimum_evidence_policy": {
            "genuine_trials_per_model_method_unit": 20,
            "impostor_trials_per_model_method_unit": 100,
            "open_set_trials_per_model_method_unit": 20,
        },
        "terminal_decision_policy_sha256": "3" * 64,
    }
    actions = {
        "reveal_evaluation": True,
        "prepare_evaluation_audio": True,
        "freeze_evaluation_windows": True,
        "run_models": False,
        "score_trials": False,
        "calculate_terminal_metrics": False,
        "make_terminal_decision": False,
    }
    parent_core = {
        "schema_version": "transcribe-audio.verification-generation-2-pre-reveal-manifest.v1",
        "status": "applied_pre_reveal",
        "preview": preview,
        "repository_authority": {},
        "authorized_actions": actions,
        "exact_trial_child_required_before_model_or_score_execution": True,
        "contains_private_evaluation": False,
        "contains_raw_audio": False,
        "contains_biometric_scores": False,
        "will_perform_external_write": False,
    }
    parent_content = evaluation._canonical_hash(parent_core)
    parent_id = f"generation-2-pre-reveal-authority-{parent_content[:24]}"
    parent = {**parent_core, "authority_id": parent_id, "content_sha256": parent_content}
    parent_path = private / "authorities" / parent_id / "manifest.json"
    parent_file_sha = _write_private(parent_path, parent)
    _write_private(parent_path.parent / "apply-receipt.json", {
        "schema_version": "transcribe-audio.verification-generation-2-pre-reveal-receipt.v1",
        "authority_id": parent_id,
        "authority_content_sha256": parent_content,
        "preview_id": preview.get("preview_id"),
        "preview_content_sha256": preview["content_sha256"],
        "manifest_sha256": parent_file_sha,
        "evaluation_reveal_authorized": True,
        "model_execution_authorized": False,
        "trial_scoring_authorized": False,
        "contains_private_evaluation": False,
        "contains_device_labels": False,
        "mode": "0600",
        "will_perform_external_write": False,
    })
    for directory in (
        private / "authorities", parent_path.parent, corpus_path.parent
    ):
        directory.chmod(0o700)
    monkeypatch.setattr(evaluation, "EXPECTED_PARENT_AUTHORITY_ID", parent_id)
    monkeypatch.setattr(evaluation, "EXPECTED_PARENT_CONTENT_SHA256", parent_content)
    monkeypatch.setattr(evaluation, "EXPECTED_PREVIEW_CONTENT_SHA256", preview["content_sha256"])
    return {
        "root": private,
        "parent": parent_path,
        "parent_receipt": parent_path.parent / "apply-receipt.json",
        "corpus": corpus_path,
        "source": Path(recordings[-1]["source_blob"]["stored_path"]),
        "transcript": Path(
            recordings[-1]["transcript_lineage"]["current_artifact_path"]
        ),
    }


def _preview(paths: dict[str, Path]) -> dict:
    return evaluation.preview_generation2_evaluation_preflight(
        parent_manifest_path=paths["parent"],
        corpus_manifest_path=paths["corpus"],
        parent_runtime_root=paths["root"],
    )


def test_preflight_proves_global_stop_without_private_leak(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    paths = _fixture(monkeypatch, tmp_path)
    preview = _preview(paths)
    assert preview["status"] == "global_stop_required"
    assert preview["reason_code"] == "trial_class_denominator_below_policy"
    assert preview["matched_profile_subject_count"] == 0
    assert preview["units"][0]["maximum_genuine_trials"] == 0
    assert preview["units"][0]["maximum_impostor_trials"] == 0
    encoded = json.dumps(preview)
    assert "subject-open" not in encoded
    assert str(paths["root"]) not in encoded
    assert preview["did_read_audio"] is False
    assert preview["did_run_models"] is False


def test_preflight_rejects_source_byte_drift(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    paths = _fixture(monkeypatch, tmp_path)
    paths["source"].write_bytes(b"tampered")
    paths["source"].chmod(0o600)
    with pytest.raises(evaluation.Generation2EvaluationError, match="lineage drifted"):
        _preview(paths)


def test_profile_overlap_changes_reviewed_stop_result(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    paths = _fixture(monkeypatch, tmp_path, overlapping_subject=True)
    preview = _preview(paths)
    assert preview["status"] == "preflight_pass_requires_preparation"
    assert preview["reason_code"] is None
    assert preview["matched_profile_subject_count"] == 1
    assert preview["units"][0]["feasibility"] == "requires_window_freeze"


def test_apply_and_replay_are_private_idempotent_and_full_body(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    paths = _fixture(monkeypatch, tmp_path)
    preview = _preview(paths)
    repository = {
        "commit": "a" * 40, "module_sha256": "b" * 64,
        "clean": True, "upstream_ahead": 0, "upstream_behind": 0,
    }
    monkeypatch.setattr(evaluation, "_repository_authority", lambda: repository)
    monkeypatch.setattr(
        evaluation, "_validate_repository_authority", lambda value: dict(value)
    )
    runtime = tmp_path / "runtime"
    applied = evaluation.apply_generation2_evaluation_stop(
        preview, expected_preview_content_sha256=preview["content_sha256"],
        parent_manifest_path=paths["parent"], corpus_manifest_path=paths["corpus"],
        parent_runtime_root=paths["root"], runtime_root=runtime,
    )
    assert applied["status"] == "terminal_stop"
    assert applied["contains_subject_ids"] is False
    assert applied["audio_preparation_authorized"] is False
    assert applied["window_freeze_authorized"] is False
    assert applied["exact_trial_child_construction_authorized"] is False
    assert applied["model_execution_authorized"] is False
    assert applied["trial_scoring_authorized"] is False
    assert applied["terminal_metrics_authorized"] is False
    assert applied["terminal_model_or_method_selection_authorized"] is False
    assert Path(applied["manifest_path"]).stat().st_mode & 0o777 == 0o600
    assert Path(applied["receipt_path"]).stat().st_mode & 0o777 == 0o600
    assert runtime.stat().st_mode & 0o777 == 0o700
    replay = evaluation.replay_generation2_evaluation_stop(
        Path(applied["manifest_path"]), parent_manifest_path=paths["parent"],
        corpus_manifest_path=paths["corpus"], parent_runtime_root=paths["root"],
        runtime_root=runtime,
    )
    assert replay["full_body_match"] is True
    again = evaluation.apply_generation2_evaluation_stop(
        preview, expected_preview_content_sha256=preview["content_sha256"],
        parent_manifest_path=paths["parent"], corpus_manifest_path=paths["corpus"],
        parent_runtime_root=paths["root"], runtime_root=runtime,
    )
    assert again["idempotent"] is True


def test_replay_rejects_self_rehashed_extra_manifest_key(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    paths = _fixture(monkeypatch, tmp_path)
    preview = _preview(paths)
    repository = {
        "commit": "a" * 40, "module_sha256": "b" * 64,
        "clean": True, "upstream_ahead": 0, "upstream_behind": 0,
    }
    monkeypatch.setattr(evaluation, "_repository_authority", lambda: repository)
    monkeypatch.setattr(
        evaluation, "_validate_repository_authority", lambda value: dict(value)
    )
    runtime = tmp_path / "runtime"
    applied = evaluation.apply_generation2_evaluation_stop(
        preview, expected_preview_content_sha256=preview["content_sha256"],
        parent_manifest_path=paths["parent"], corpus_manifest_path=paths["corpus"],
        parent_runtime_root=paths["root"], runtime_root=runtime,
    )
    manifest_path = Path(applied["manifest_path"])
    manifest = json.loads(manifest_path.read_text())
    manifest["unexpected"] = True
    manifest["content_sha256"] = evaluation._canonical_hash({
        key: value for key, value in manifest.items()
        if key not in {"run_id", "content_sha256"}
    })
    _write_private(manifest_path, manifest)
    with pytest.raises(evaluation.Generation2EvaluationError, match="replay mismatch"):
        evaluation.replay_generation2_evaluation_stop(
            manifest_path, parent_manifest_path=paths["parent"],
            corpus_manifest_path=paths["corpus"], parent_runtime_root=paths["root"],
            runtime_root=runtime,
        )


def test_apply_rejects_partial_run_directory(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    paths = _fixture(monkeypatch, tmp_path)
    preview = _preview(paths)
    monkeypatch.setattr(evaluation, "_repository_authority", lambda: {})
    runtime = tmp_path / "runtime"
    partial = runtime / "runs" / "partial"
    partial.mkdir(parents=True, mode=0o700)
    partial.chmod(0o700)
    _write_private(partial / "private-manifest.json", {})
    with pytest.raises(evaluation.Generation2EvaluationError, match="Partial"):
        evaluation.apply_generation2_evaluation_stop(
            preview, expected_preview_content_sha256=preview["content_sha256"],
            parent_manifest_path=paths["parent"], corpus_manifest_path=paths["corpus"],
            parent_runtime_root=paths["root"], runtime_root=runtime,
        )


def test_preflight_rejects_parent_receipt_and_corpus_path_drift(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    paths = _fixture(monkeypatch, tmp_path)
    receipt = json.loads(paths["parent_receipt"].read_text())
    receipt["evaluation_reveal_authorized"] = False
    _write_private(paths["parent_receipt"], receipt)
    with pytest.raises(evaluation.Generation2EvaluationError, match="receipt drifted"):
        _preview(paths)

    paths = _fixture(monkeypatch, tmp_path / "second")
    wrong_corpus = paths["root"] / "wrong-corpus" / "manifest.json"
    wrong_corpus.parent.mkdir(parents=True, mode=0o700)
    wrong_corpus.parent.chmod(0o700)
    wrong_corpus.write_bytes(paths["corpus"].read_bytes())
    wrong_corpus.chmod(0o600)
    with pytest.raises(evaluation.Generation2EvaluationError, match="corpus drifted"):
        evaluation.preview_generation2_evaluation_preflight(
            parent_manifest_path=paths["parent"],
            corpus_manifest_path=wrong_corpus,
            parent_runtime_root=paths["root"],
        )


def test_preflight_rejects_transcript_byte_drift(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    paths = _fixture(monkeypatch, tmp_path)
    paths["transcript"].write_text("{}", encoding="utf-8")
    paths["transcript"].chmod(0o600)
    with pytest.raises(evaluation.Generation2EvaluationError, match="lineage drifted"):
        _preview(paths)


def test_replay_rejects_portable_receipt_tamper(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    paths = _fixture(monkeypatch, tmp_path)
    preview = _preview(paths)
    repository = {
        "commit": "a" * 40, "module_sha256": "b" * 64,
        "clean": True, "upstream_ahead": 0, "upstream_behind": 0,
    }
    monkeypatch.setattr(evaluation, "_repository_authority", lambda: repository)
    monkeypatch.setattr(
        evaluation, "_validate_repository_authority", lambda value: dict(value)
    )
    runtime = tmp_path / "runtime"
    applied = evaluation.apply_generation2_evaluation_stop(
        preview, expected_preview_content_sha256=preview["content_sha256"],
        parent_manifest_path=paths["parent"], corpus_manifest_path=paths["corpus"],
        parent_runtime_root=paths["root"], runtime_root=runtime,
    )
    receipt_path = Path(applied["receipt_path"])
    receipt = json.loads(receipt_path.read_text())
    receipt["audio_preparation_authorized"] = True
    _write_private(receipt_path, receipt)
    with pytest.raises(evaluation.Generation2EvaluationError, match="receipt replay mismatch"):
        evaluation.replay_generation2_evaluation_stop(
            Path(applied["manifest_path"]), parent_manifest_path=paths["parent"],
            corpus_manifest_path=paths["corpus"], parent_runtime_root=paths["root"],
            runtime_root=runtime,
        )
