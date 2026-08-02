from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

import acoustic_generation3_authority as generation3


def _conversation(root: Path, index: int) -> dict[str, str]:
    source = root / f"recording-{index}.wav"
    transcript = root / f"recording-{index} Transcript.transcript.json"
    source.write_bytes(f"wave-{index}".encode())
    transcript.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "duration_seconds": 4.0,
                "recording_start": f"2026-01-{index + 1:02d}T10:00:00-06:00",
                "recording_end": f"2026-01-{index + 1:02d}T10:00:04-06:00",
                "event": {"event_id": f"event-{index}"},
                "utterance_count": 2,
                "utterances": [
                    {
                        "speaker": "A",
                        "start": 0,
                        "end": 1500,
                        "text": f"one {index}",
                    },
                    {
                        "speaker": "B",
                        "start": 1600,
                        "end": 3200,
                        "text": f"two {index}",
                    },
                ],
                "working_media_path": str(source),
                "source_media_path": str(source),
                "output_paths": {"artifact": str(transcript)},
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return {"source_path": str(source), "transcript_path": str(transcript)}


@pytest.fixture
def lineage(monkeypatch: pytest.MonkeyPatch) -> None:
    authority = {
        "prior_corpus_authorities": [],
        "active_reference_authorities": [],
        "active_training_authority": {},
        "active_reference_authority_sha256": generation3._canonical_hash([]),
        "prior_corpus_source_count": 0,
        "active_reference_source_count": 0,
        "excluded_source_count": 0,
        "excluded_source_set_sha256": generation3._canonical_hash([]),
        "dimensional_lineage": {
            key: {"count": 0, "set_sha256": generation3._canonical_hash([])}
            for key in generation3.LINEAGE_DIMENSIONS
        },
    }
    authority["content_sha256"] = generation3._canonical_hash(authority)
    monkeypatch.setattr(
        generation3,
        "_source_lineage_authority",
        lambda **_kwargs: (
            authority,
            generation3._empty_lineage_dimensions(),
        ),
    )


def test_preview_is_exact_seven_and_portable_is_aggregate_only(
    tmp_path: Path, lineage: None
) -> None:
    conversations = [_conversation(tmp_path, index) for index in range(7)]
    preview = generation3.preview_generation3_cohort(
        conversations, source_root=tmp_path
    )
    assert preview["membership"]["conversation_count"] == 7
    assert preview["membership"]["speaker_label_count"] == 14
    assert preview["source_overlap_count"] == 0
    assert preview["action_vector"]["load_or_run_models"] is False
    assert (
        preview["window_policy"]["maximum_windows_per_speaker_per_conversation"]
        == 12
    )

    portable = generation3.portable_cohort_projection(preview)
    serialized = json.dumps(portable, sort_keys=True)
    assert "conversations" not in serialized
    assert "source_sha256" not in serialized
    assert "person_ref_id" not in serialized
    assert portable["contains_source_membership"] is False
    assert portable["contains_subject_ids"] is False


def test_preview_rejects_wrong_count_duplicate_and_excluded_source(
    tmp_path: Path, lineage: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    conversations = [_conversation(tmp_path, index) for index in range(7)]
    with pytest.raises(generation3.Generation3AuthorityError, match="exactly seven"):
        generation3.preview_generation3_cohort(
            conversations[:6], source_root=tmp_path
        )

    duplicate = conversations[:6] + [conversations[0]]
    with pytest.raises(generation3.Generation3AuthorityError, match="duplicate"):
        generation3.preview_generation3_cohort(duplicate, source_root=tmp_path)

    authority, _ = generation3._source_lineage_authority()
    transcript = generation3._read_object(
        Path(conversations[0]["transcript_path"])
    )
    semantic_excluded = generation3._empty_lineage_dimensions()
    semantic_excluded["derivative_identity_sha256"].add(
        generation3._transcript_identities(transcript)[
            "derivative_identity_sha256"
        ]
    )
    monkeypatch.setattr(
        generation3,
        "_source_lineage_authority",
        lambda **_kwargs: (authority, semantic_excluded),
    )
    with pytest.raises(generation3.Generation3AuthorityError, match="non-disjoint"):
        generation3.preview_generation3_cohort(
            conversations, source_root=tmp_path
        )

    digest = generation3.sha256_file(Path(conversations[0]["source_path"]))
    excluded = generation3._empty_lineage_dimensions()
    excluded["source_sha256"].add(digest)
    monkeypatch.setattr(
        generation3,
        "_source_lineage_authority",
        lambda **_kwargs: (authority, excluded),
    )
    with pytest.raises(generation3.Generation3AuthorityError, match="overlaps"):
        generation3.preview_generation3_cohort(
            conversations, source_root=tmp_path
        )


def test_apply_and_replay_freeze_private_membership_only(
    tmp_path: Path, lineage: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    source_root = tmp_path / "sources"
    source_root.mkdir()
    runtime_root = tmp_path / "runtime"
    conversations = [_conversation(source_root, index) for index in range(7)]
    preview = generation3.preview_generation3_cohort(
        conversations, source_root=source_root
    )
    repository = {
        "commit": "b" * 40,
        "module_sha256": "c" * 64,
        "training_dependency_sha256": "d" * 64,
        "p3_dependency_sha256": "e" * 64,
        "private_io_dependency_sha256": "a" * 64,
        "clean": True,
        "upstream_ahead": 0,
        "upstream_behind": 0,
    }
    monkeypatch.setattr(generation3, "_repository_authority", lambda: repository)
    monkeypatch.setattr(
        generation3, "_validate_repository_authority", lambda value: dict(value)
    )

    receipt = generation3.apply_generation3_cohort(
        preview,
        expected_preview_content_sha256=preview["content_sha256"],
        conversations=conversations,
        source_root=source_root,
        runtime_root=runtime_root,
    )
    assert receipt["status"] == "applied_membership_only_gold_not_frozen"
    assert receipt["contains_source_membership"] is False
    assert receipt["action_vector"]["prepare_audio"] is False
    assert receipt["action_vector"]["freeze_cohort_membership"] is True
    assert receipt["action_vector"]["build_private_gold_review_packet"] is True
    assert Path(receipt["private_manifest_path"]).stat().st_mode & 0o777 == 0o600

    replay = generation3.replay_generation3_cohort(
        Path(receipt["private_manifest_path"]),
        conversations=conversations,
        source_root=source_root,
        runtime_root=runtime_root,
    )
    assert replay["idempotent_replay"] is True
    assert replay["manifest_sha256"] == receipt["manifest_sha256"]

    changed = copy.deepcopy(preview)
    changed["membership_sha256"] = "f" * 64
    with pytest.raises(generation3.Generation3AuthorityError, match="stale"):
        generation3.apply_generation3_cohort(
            changed,
            expected_preview_content_sha256=preview["content_sha256"],
            conversations=conversations,
            source_root=source_root,
            runtime_root=tmp_path / "other-runtime",
        )


def test_preview_rejects_each_semantic_lineage_dimension(
    tmp_path: Path, lineage: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    conversations = [_conversation(tmp_path, index) for index in range(7)]
    authority, _ = generation3._source_lineage_authority()
    first = generation3._read_object(Path(conversations[0]["transcript_path"]))
    identities = generation3._transcript_identities(first)
    for dimension in (
        "recording_identity_sha256",
        "conversation_identity_sha256",
        "derivative_identity_sha256",
    ):
        excluded = generation3._empty_lineage_dimensions()
        excluded[dimension].add(identities[dimension])
        monkeypatch.setattr(
            generation3,
            "_source_lineage_authority",
            lambda **_kwargs: (authority, excluded),
        )
        with pytest.raises(
            generation3.Generation3AuthorityError, match="non-disjoint"
        ):
            generation3.preview_generation3_cohort(
                conversations, source_root=tmp_path
            )


def test_repository_authority_binds_private_io_dependency(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_git(args: list[str]) -> str:
        if args[0] == "status":
            return ""
        if args[0] == "rev-list":
            return "0\t0"
        if args[0] == "rev-parse":
            return "a" * 40
        raise AssertionError(args)

    monkeypatch.setattr(generation3, "_git", fake_git)
    repository = generation3._repository_authority()
    assert repository["private_io_dependency_sha256"] == generation3.sha256_file(
        Path(generation3.derivatives.__file__).resolve()
    )


def test_transcript_identities_require_recording_and_conversation_evidence(
    tmp_path: Path,
) -> None:
    conversation = _conversation(tmp_path, 0)
    transcript = generation3._read_object(Path(conversation["transcript_path"]))
    transcript.pop("recording_start")
    transcript.pop("recording_end")
    with pytest.raises(
        generation3.Generation3AuthorityError,
        match="Recording identity evidence",
    ):
        generation3._transcript_identities(transcript)

    transcript["recording_id"] = "recording-authority-1"
    transcript["event"] = {}
    with pytest.raises(
        generation3.Generation3AuthorityError,
        match="Conversation identity evidence",
    ):
        generation3._transcript_identities(transcript)
