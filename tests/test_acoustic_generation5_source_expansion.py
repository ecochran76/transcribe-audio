import hashlib
import json
from pathlib import Path

import pytest

import acoustic_generation5_source_expansion as s0


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    zoom = tmp_path / "zoom.m4a"
    zoom.write_bytes(b"zoom-new-source")
    archive = tmp_path / "archive"
    archive.mkdir()
    required = archive / "required.m4a"
    required.write_bytes(b"required-new-source")
    for index in range(7):
        (archive / f"candidate-{index}.m4a").write_bytes(f"candidate-{index}".encode())
    prior = tmp_path / "prior"
    prior.mkdir()
    (prior / "evidence.json").write_text(json.dumps({"source_sha256": "a" * 64}))
    monkeypatch.setattr(s0, "ZOOM_SHA256", _sha(zoom))
    monkeypatch.setattr(s0, "ARCHIVE_REQUIRED_SHA256", _sha(required))
    monkeypatch.setattr(
        s0.r0, "_evidence_hashes", lambda _path: ({"a" * 64}, "structured_json")
    )
    probe = lambda _path, _ffprobe: {
        "codec_name": "aac", "sample_rate": 32000, "channels": 1,
        "duration_seconds": 120.0,
    }
    authority = {
        "commit": "b" * 40, "module_sha256": "c" * 64,
        "plan_sha256": "d" * 64, "clean": True,
        "upstream_ahead": 0, "upstream_behind": 0,
    }
    return zoom, required, archive, prior, probe, authority


def test_preview_binds_two_required_and_ordered_additional_sources(tmp_path, monkeypatch):
    zoom, required, archive, prior, probe, authority = _fixture(tmp_path, monkeypatch)
    preview = s0.preview_generation5_source_expansion(
        zoom_source=zoom, archive_required=required, archive_root=archive,
        prior_root=prior, ffprobe_path="ffprobe", probe=probe,
        repository_authority=authority,
    )
    assert preview["status"] == "ready_for_independent_j0_review"
    assert preview["required_source_count"] == 2
    assert preview["additional_candidate_count"] == 7
    assert preview["candidate_count"] == 9
    assert preview["action_vector"]["transcribe_or_diarize"] is False
    assert preview["action_vector"]["run_models_or_predictions"] is False
    names = [row["archive_relative_path"] for row in preview["private_evidence"]["additional_candidates"]]
    assert names == sorted(names)


def test_preview_rejects_required_prior_overlap(tmp_path, monkeypatch):
    zoom, required, archive, prior, probe, authority = _fixture(tmp_path, monkeypatch)
    zoom_hash = _sha(zoom)
    monkeypatch.setattr(s0.r0, "_evidence_hashes", lambda _path: ({zoom_hash}, "structured_json"))
    with pytest.raises(s0.Generation5SourceExpansionError, match="required_zoom_prior_evidence_overlap"):
        s0.preview_generation5_source_expansion(
            zoom_source=zoom, archive_required=required, archive_root=archive,
            prior_root=prior, ffprobe_path="ffprobe", probe=probe,
            repository_authority=authority,
        )


def test_preview_rejects_required_hash_drift(tmp_path, monkeypatch):
    zoom, required, archive, prior, probe, authority = _fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(s0, "ZOOM_SHA256", "f" * 64)
    with pytest.raises(s0.Generation5SourceExpansionError, match="required_zoom_hash_drift"):
        s0.preview_generation5_source_expansion(
            zoom_source=zoom, archive_required=required, archive_root=archive,
            prior_root=prior, ffprobe_path="ffprobe", probe=probe,
            repository_authority=authority,
        )
