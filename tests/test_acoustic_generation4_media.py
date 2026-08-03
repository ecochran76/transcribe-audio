from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import acoustic_generation4_media as media


@pytest.fixture
def source_root(tmp_path: Path) -> Path:
    tmp_path.chmod(0o700)
    root = tmp_path / "sources"
    root.mkdir(mode=0o700)
    return root


def _candidate(root: Path, name: str = "candidate.m4a", body: bytes = b"audio") -> Path:
    path = root / name
    path.write_bytes(body)
    return path


def test_qualifies_exact_healthy_top_level_source(monkeypatch: pytest.MonkeyPatch, source_root: Path) -> None:
    path = _candidate(source_root)
    monkeypatch.setattr(media, "_probe", lambda path, tool: {
        "audio_stream_count": 1, "codec_name": "aac", "channels": 2,
        "sample_rate": 48000, "duration_seconds": 120.0,
    })
    monkeypatch.setattr(media, "_decoded_duration", lambda path, tool: 120.0)
    result = media._qualify_one(path, source_root=source_root, prior_hashes=set(),
                                seen=set(), ffmpeg="ffmpeg", ffprobe="ffprobe")
    assert result["status"] == "qualified"
    assert result["duration_drift_seconds"] == 0


def test_rejects_prior_source_overlap(monkeypatch: pytest.MonkeyPatch, source_root: Path) -> None:
    path = _candidate(source_root)
    digest = media.sha256_file(path)
    result = media._qualify_one(path, source_root=source_root, prior_hashes={digest},
                                seen=set(), ffmpeg="ffmpeg", ffprobe="ffprobe")
    assert result["reason_code"] == "prior_plan0037_overlap"


def test_rejects_duplicate_candidate_bytes(source_root: Path) -> None:
    first = _candidate(source_root, "one.m4a", b"same")
    second = _candidate(source_root, "two.m4a", b"same")
    seen = {media.sha256_file(first)}
    result = media._qualify_one(second, source_root=source_root, prior_hashes=set(),
                                seen=seen, ffmpeg="ffmpeg", ffprobe="ffprobe")
    assert result["reason_code"] == "duplicate_candidate_bytes"


def test_rejects_nested_or_symlink_source(source_root: Path) -> None:
    nested = source_root / "nested"
    nested.mkdir()
    path = _candidate(nested)
    result = media._qualify_one(path, source_root=source_root, prior_hashes=set(),
                                seen=set(), ffmpeg="ffmpeg", ffprobe="ffprobe")
    assert result["reason_code"] == "not_top_level_regular_file"


def test_rejects_actual_symlink_source(source_root: Path) -> None:
    target = _candidate(source_root, "target.m4a")
    link = source_root / "link.m4a"
    link.symlink_to(target)
    result = media._qualify_one(link, source_root=source_root, prior_hashes=set(),
                                seen=set(), ffmpeg="ffmpeg", ffprobe="ffprobe")
    assert result["reason_code"] == "not_top_level_regular_file"


def test_probe_rejects_multiple_audio_streams(monkeypatch: pytest.MonkeyPatch, source_root: Path) -> None:
    path = _candidate(source_root)
    body = {"streams": [
        {"codec_type": "audio", "channels": 1, "sample_rate": "16000", "duration": "60"},
        {"codec_type": "audio", "channels": 1, "sample_rate": "16000", "duration": "60"},
    ], "format": {"duration": "60"}}
    monkeypatch.setattr(media.subprocess, "run", lambda *args, **kwargs: SimpleNamespace(
        returncode=0, stdout=json.dumps(body), stderr=""
    ))
    with pytest.raises(media.Generation4MediaError, match="audio_stream_count_not_one"):
        media._probe(path, "ffprobe")


def test_rejects_short_source(monkeypatch: pytest.MonkeyPatch, source_root: Path) -> None:
    path = _candidate(source_root)
    monkeypatch.setattr(media, "_probe", lambda path, tool: {
        "audio_stream_count": 1, "codec_name": "aac", "channels": 1,
        "sample_rate": 16000, "duration_seconds": 59.99,
    })
    result = media._qualify_one(path, source_root=source_root, prior_hashes=set(),
                                seen=set(), ffmpeg="ffmpeg", ffprobe="ffprobe")
    assert result["reason_code"] == "duration_below_minimum"


def test_rejects_decoded_duration_drift(monkeypatch: pytest.MonkeyPatch, source_root: Path) -> None:
    path = _candidate(source_root)
    monkeypatch.setattr(media, "_probe", lambda path, tool: {
        "audio_stream_count": 1, "codec_name": "aac", "channels": 2,
        "sample_rate": 48000, "duration_seconds": 120.0,
    })
    monkeypatch.setattr(media, "_decoded_duration", lambda path, tool: 119.9)
    result = media._qualify_one(path, source_root=source_root, prior_hashes=set(),
                                seen=set(), ffmpeg="ffmpeg", ffprobe="ffprobe")
    assert result["reason_code"] == "decoded_duration_drift_exceeds_policy"


def test_rejects_decode_failure(monkeypatch: pytest.MonkeyPatch, source_root: Path) -> None:
    path = _candidate(source_root)
    monkeypatch.setattr(media, "_probe", lambda path, tool: {
        "audio_stream_count": 1, "codec_name": "aac", "channels": 2,
        "sample_rate": 48000, "duration_seconds": 120.0,
    })
    monkeypatch.setattr(
        media, "_decoded_duration",
        lambda path, tool: (_ for _ in ()).throw(media.Generation4MediaError("decode_failed")),
    )
    result = media._qualify_one(path, source_root=source_root, prior_hashes=set(),
                                seen=set(), ffmpeg="ffmpeg", ffprobe="ffprobe")
    assert result["reason_code"] == "decode_failed"


def _preview() -> dict:
    actions = {key: False for key in media.POST_QUALIFICATION_ACTIONS}
    actions["freeze_media_qualification"] = False
    core = {
        "schema_version": media.PREVIEW_SCHEMA, "status": "ready_to_freeze",
        "policy": {}, "tool_authority": {}, "prior_evidence": {"hash_set_sha256": "a" * 64},
        "candidate_count": 7, "qualified_count": 7, "rejected_count": 0,
        "reason_counts": {"qualified": 7}, "qualified_set_sha256": "b" * 64,
        "private_results": [],
        "repository_authority": {"commit": "c" * 40, "module_name": media.MODULE_NAME,
                                 "module_sha256": "d" * 64, "clean": True,
                                 "upstream_ahead": 0, "upstream_behind": 0},
        "action_vector": actions, "contains_paths": True, "contains_private_membership": True,
        "contains_raw_audio": False, "contains_transcript_text": False,
        "contains_embeddings_or_vectors": False, "contains_biometric_scores": False,
        "did_load_or_run_models": False, "did_retain_decoded_audio": False,
        "will_perform_external_write": False,
    }
    digest = media._canonical_hash(core)
    return {**core, "preview_id": f"generation4-media-preview-{digest[:24]}", "content_sha256": digest}


def test_portable_projection_removes_private_results() -> None:
    portable = media.portable_media_projection(_preview())
    assert "private_results" not in portable
    assert "repository_authority" not in portable
    assert portable["contains_paths"] is False


def test_apply_and_full_body_replay(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    tmp_path.chmod(0o700)
    preview = _preview()
    monkeypatch.setattr(media, "preview_generation4_media", lambda *args, **kwargs: preview)
    monkeypatch.setattr(media, "_validate_repository_authority", lambda value: dict(value))
    receipt = media.apply_generation4_media(
        preview, expected_content_sha256=preview["content_sha256"], candidates=[Path("x")],
        runtime_root=tmp_path,
    )
    assert receipt["action_vector"]["build_generation4_cohort_preview"] is True
    replay = media.replay_generation4_media(
        [Path("x")], expected_content_sha256=preview["content_sha256"], runtime_root=tmp_path,
    )
    assert replay["idempotent_replay"] is True
    assert replay["replay_mode"] == "full_body_with_source_redecode_no_retained_audio"


def test_apply_rejects_stale_preview(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    preview = _preview()
    monkeypatch.setattr(media, "preview_generation4_media", lambda *args, **kwargs: preview)
    stale = {**preview, "qualified_count": 6}
    with pytest.raises(media.Generation4MediaError, match="stale"):
        media.apply_generation4_media(
            stale, expected_content_sha256=preview["content_sha256"],
            candidates=[Path("x")], runtime_root=tmp_path,
        )


def test_replay_rejects_coordinated_manifest_mutation(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    tmp_path.chmod(0o700)
    preview = _preview()
    monkeypatch.setattr(media, "preview_generation4_media", lambda *args, **kwargs: preview)
    monkeypatch.setattr(media, "_validate_repository_authority", lambda value: dict(value))
    media.apply_generation4_media(
        preview, expected_content_sha256=preview["content_sha256"],
        candidates=[Path("x")], runtime_root=tmp_path,
    )
    paths = media._paths(tmp_path, preview["content_sha256"])
    manifest = json.loads(paths["manifest"].read_text())
    manifest["contains_raw_audio"] = True
    paths["manifest"].write_text(json.dumps(manifest))
    receipt = json.loads(paths["receipt"].read_text())
    receipt["manifest_sha256"] = media.sha256_file(paths["manifest"])
    paths["receipt"].write_text(json.dumps(receipt))
    with pytest.raises(media.Generation4MediaError, match="drifted"):
        media.replay_generation4_media(
            [Path("x")], expected_content_sha256=preview["content_sha256"], runtime_root=tmp_path,
        )


def test_authority_replay_recovers_private_candidates(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    tmp_path.chmod(0o700)
    preview = _preview()
    preview["private_results"] = [{"path": "/private/candidate.m4a"}]
    preview["candidate_count"] = 1
    monkeypatch.setattr(media, "preview_generation4_media", lambda *args, **kwargs: preview)
    monkeypatch.setattr(media, "_validate_repository_authority", lambda value: dict(value))
    media.apply_generation4_media(
        preview, expected_content_sha256=preview["content_sha256"],
        candidates=[Path("/private/candidate.m4a")], runtime_root=tmp_path,
    )
    replay = media.replay_generation4_media_authority(
        preview["content_sha256"], runtime_root=tmp_path,
    )
    assert replay["idempotent_replay"] is True
