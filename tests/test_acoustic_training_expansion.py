from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

import acoustic_training_expansion as training


def _write_json(path: Path, value: object, *, mode: int = 0o600) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")),
        encoding="utf-8",
    )
    path.chmod(mode)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fixture(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, *, count: int = 5,
    prior_overlap: bool = False,
) -> dict[str, object]:
    source_root = tmp_path / "sources"
    source_root.mkdir(parents=True, mode=0o700)
    conversations = []
    source_hashes = []
    for index in range(count):
        source = source_root / f"conversation-{index}.m4a"
        source.write_bytes((f"audio-{index}-" * 20).encode())
        source_hashes.append(hashlib.sha256(source.read_bytes()).hexdigest())
        transcript = source_root / f"conversation-{index} Transcript.transcript.json"
        _write_json(transcript, {
            "schema_version": 1,
            "duration_seconds": 30.0,
            "source_media_path": str(source),
            "working_media_path": str(source),
            "output_paths": {"artifact": str(transcript)},
            "utterance_count": 2,
            "utterances": [
                {"speaker": "A", "start": 0, "end": 10_000, "text": "private"},
                {"speaker": "B", "start": 11_000, "end": 25_000, "text": "private"},
            ],
        })
        conversations.append({
            "source_path": str(source), "transcript_path": str(transcript),
        })
    expected = {}
    corpus_paths = []
    for index in range(3):
        corpus_id = f"test-corpus-{index}"
        recordings = []
        if index == 0 and prior_overlap:
            recordings.append({"source_blob": {"sha256": source_hashes[0]}})
        else:
            recordings.append({"source_blob": {"sha256": hashlib.sha256(f"prior-{index}".encode()).hexdigest()}})
        corpus = {
            "corpus_id": corpus_id,
            "content_sha256": hashlib.sha256(f"content-{index}".encode()).hexdigest(),
            "recordings": recordings,
        }
        path = tmp_path / "corpora" / corpus_id / "manifest.json"
        manifest_sha = _write_json(path, corpus)
        expected[corpus_id] = {
            "content_sha256": corpus["content_sha256"],
            "manifest_sha256": manifest_sha,
        }
        corpus_paths.append(path)
    monkeypatch.setattr(training, "EXPECTED_CORPORA", expected)
    return {
        "source_root": source_root,
        "conversations": conversations,
        "corpus_paths": tuple(corpus_paths),
    }


def _preview(values: dict[str, object]) -> dict:
    return training.preview_training_intake(
        values["conversations"],
        source_root=values["source_root"],
        corpus_manifest_paths=values["corpus_paths"],
    )


def test_preview_freezes_five_novel_conversations_without_private_leak(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    values = _fixture(monkeypatch, tmp_path)
    preview = _preview(values)
    assert preview["status"] == "ready_for_independent_review"
    assert preview["conversation_count"] == 5
    assert preview["speaker_label_count"] == 10
    assert preview["prior_corpus_overlap_count"] == 0
    assert preview["identity_confirmation_required"] is True
    encoded = json.dumps(preview)
    assert str(values["source_root"]) not in encoded
    assert "private" not in encoded
    assert '"speaker_label"' not in encoded
    assert all(
        label["speaker_label_id"].startswith("diarized-label-")
        for conversation in preview["conversations"]
        for label in conversation["labels"]
    )
    assert preview["will_read_audio"] is False
    assert preview["will_infer_identity"] is False


def test_preview_rejects_more_than_five_and_duplicate_sources(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    values = _fixture(monkeypatch, tmp_path, count=6)
    with pytest.raises(training.TrainingExpansionError, match="one and five"):
        _preview(values)

    values = _fixture(monkeypatch, tmp_path / "duplicate", count=2)
    conversations = list(values["conversations"])
    conversations[1] = {
        "source_path": conversations[0]["source_path"],
        "transcript_path": conversations[0]["transcript_path"],
    }
    values["conversations"] = conversations
    with pytest.raises(training.TrainingExpansionError, match="duplicate"):
        _preview(values)


def test_preview_rejects_prior_corpus_overlap(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    values = _fixture(monkeypatch, tmp_path, prior_overlap=True)
    with pytest.raises(training.TrainingExpansionError, match="overlaps"):
        _preview(values)


def test_preview_rejects_transcript_source_path_swap(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    values = _fixture(monkeypatch, tmp_path, count=2)
    conversations = list(values["conversations"])
    conversations[0] = {
        "source_path": conversations[1]["source_path"],
        "transcript_path": conversations[0]["transcript_path"],
    }
    values["conversations"] = conversations
    with pytest.raises(training.TrainingExpansionError, match="binding"):
        _preview(values)


def test_preview_keeps_human_looking_diarization_label_private(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    values = _fixture(monkeypatch, tmp_path, count=1)
    transcript_path = Path(values["conversations"][0]["transcript_path"])
    original_preview = _preview(values)
    original_label_ids = [
        item["speaker_label_id"]
        for item in original_preview["conversations"][0]["labels"]
    ]
    transcript = json.loads(transcript_path.read_text())
    transcript["utterances"][0]["speaker"] = "Alice.Smith"
    _write_json(transcript_path, transcript)

    preview = _preview(values)
    encoded = json.dumps(preview)
    assert "Alice" not in encoded
    assert preview["contains_names_or_emails"] is False
    assert [
        item["speaker_label_id"]
        for item in preview["conversations"][0]["labels"]
    ] == original_label_ids
    source_sha = preview["conversations"][0]["source_sha256"]
    guess = "diarized-label-" + hashlib.sha256(
        f"{source_sha}\0Alice.Smith".encode()
    ).hexdigest()[:20]
    assert guess not in original_label_ids


def test_preview_rejects_nested_symlink_escape(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    values = _fixture(monkeypatch, tmp_path, count=1)
    outside = tmp_path / "outside"
    outside.mkdir()
    source = outside / "escaped.m4a"
    source.write_bytes(b"outside audio")
    transcript = outside / "escaped Transcript.transcript.json"
    _write_json(transcript, {
        "schema_version": 1,
        "duration_seconds": 2.0,
        "source_media_path": str(source),
        "working_media_path": str(source),
        "output_paths": {"artifact": str(transcript)},
        "utterance_count": 1,
        "utterances": [{"speaker": "A", "start": 0, "end": 1000}],
    })
    link = Path(values["source_root"]) / "linked"
    link.symlink_to(outside, target_is_directory=True)
    values["conversations"] = [{
        "source_path": link / source.name,
        "transcript_path": link / transcript.name,
    }]
    with pytest.raises(training.TrainingExpansionError, match="symlink"):
        _preview(values)


def test_preview_rejects_forged_transcript_artifact_lineage(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    values = _fixture(monkeypatch, tmp_path, count=1)
    transcript_path = Path(values["conversations"][0]["transcript_path"])
    transcript = json.loads(transcript_path.read_text())
    transcript["output_paths"]["artifact"] = str(transcript_path.with_name("other.json"))
    _write_json(transcript_path, transcript)
    with pytest.raises(training.TrainingExpansionError, match="binding"):
        _preview(values)


def test_apply_replay_and_portable_receipt_are_exact(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    values = _fixture(monkeypatch, tmp_path)
    preview = _preview(values)
    repository = {
        "commit": "a" * 40, "module_sha256": "b" * 64,
        "clean": True, "upstream_ahead": 0, "upstream_behind": 0,
    }
    monkeypatch.setattr(training, "_repository_authority", lambda: repository)
    monkeypatch.setattr(
        training, "_validate_repository_authority", lambda value: dict(value)
    )
    runtime = tmp_path / "runtime"
    applied = training.apply_training_intake(
        preview, expected_preview_content_sha256=preview["content_sha256"],
        conversations=values["conversations"], source_root=values["source_root"],
        corpus_manifest_paths=values["corpus_paths"], runtime_root=runtime,
    )
    assert applied["conversation_count"] == 5
    assert applied["audio_preparation_authorized"] is True
    assert applied["identity_confirmation_authorized"] is False
    assert applied["reference_registration_authorized"] is False
    assert applied["contains_paths"] is False
    assert Path(applied["manifest_path"]).stat().st_mode & 0o777 == 0o600
    assert Path(applied["receipt_path"]).stat().st_mode & 0o777 == 0o600
    replay = training.replay_training_intake(
        Path(applied["manifest_path"]), conversations=values["conversations"],
        source_root=values["source_root"], corpus_manifest_paths=values["corpus_paths"],
        runtime_root=runtime,
    )
    assert replay["full_body_match"] is True
    assert replay["idempotent"] is True
    again = training.apply_training_intake(
        preview, expected_preview_content_sha256=preview["content_sha256"],
        conversations=values["conversations"], source_root=values["source_root"],
        corpus_manifest_paths=values["corpus_paths"], runtime_root=runtime,
    )
    assert again["idempotent"] is True


def test_replay_rejects_self_rehashed_extra_manifest_key(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    values = _fixture(monkeypatch, tmp_path)
    preview = _preview(values)
    repository = {
        "commit": "a" * 40, "module_sha256": "b" * 64,
        "clean": True, "upstream_ahead": 0, "upstream_behind": 0,
    }
    monkeypatch.setattr(training, "_repository_authority", lambda: repository)
    monkeypatch.setattr(
        training, "_validate_repository_authority", lambda value: dict(value)
    )
    runtime = tmp_path / "runtime"
    applied = training.apply_training_intake(
        preview, expected_preview_content_sha256=preview["content_sha256"],
        conversations=values["conversations"], source_root=values["source_root"],
        corpus_manifest_paths=values["corpus_paths"], runtime_root=runtime,
    )
    path = Path(applied["manifest_path"])
    manifest = json.loads(path.read_text())
    manifest["unexpected"] = True
    manifest["content_sha256"] = training._canonical_hash({
        key: value for key, value in manifest.items()
        if key not in {"intake_id", "content_sha256"}
    })
    _write_json(path, manifest)
    with pytest.raises(training.TrainingExpansionError, match="replay mismatch"):
        training.replay_training_intake(
            path, conversations=values["conversations"],
            source_root=values["source_root"], corpus_manifest_paths=values["corpus_paths"],
            runtime_root=runtime,
        )


def test_apply_rejects_partial_runtime_directory(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    values = _fixture(monkeypatch, tmp_path)
    preview = _preview(values)
    monkeypatch.setattr(training, "_repository_authority", lambda: {})
    runtime = tmp_path / "runtime"
    partial = runtime / "intakes" / "partial"
    partial.mkdir(parents=True)
    _write_json(partial / "private-manifest.json", {})
    with pytest.raises(training.TrainingExpansionError, match="Partial"):
        training.apply_training_intake(
            preview, expected_preview_content_sha256=preview["content_sha256"],
            conversations=values["conversations"], source_root=values["source_root"],
            corpus_manifest_paths=values["corpus_paths"], runtime_root=runtime,
        )
