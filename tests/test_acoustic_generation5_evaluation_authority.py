import hashlib
import json
from pathlib import Path

import acoustic_generation5_evaluation_authority as e1


REPO = {
    "commit": "1" * 40,
    "module_sha256": hashlib.sha256(b"module").hexdigest(),
    "clean": True,
    "upstream_ahead": 0,
    "upstream_behind": 0,
}
J2 = {
    "j2_preview_sha256": e1.J2_PREVIEW_SHA256,
    "j2_manifest_sha256": e1.J2_MANIFEST_SHA256,
    "idempotent_replay": True,
}


def _row(tmp_path: Path, ordinal: int) -> dict:
    source = tmp_path / f"source-{ordinal}.m4a"
    transcript = tmp_path / f"transcript-{ordinal}.json"
    source.write_bytes(f"audio-{ordinal}".encode())
    transcript.write_text(
        json.dumps(
            {
                "utterances": [
                    {"speaker": "A", "start": 0, "end": 5000, "text": f"speaker clue {ordinal}"}
                ]
            }
        ),
        encoding="utf-8",
    )
    return {
        "path": str(source),
        "transcript_path": str(transcript),
        "source_sha256": e1.sha256_file(source),
        "transcript_sha256": e1.sha256_file(transcript),
        "recording_start_utc": f"2026-01-{ordinal:02d}T00:00:00+00:00",
    }


def test_preview_is_candidate_review_only(tmp_path):
    rows = [_row(tmp_path, index) for index in range(1, 8)]
    preview = e1.preview_generation5_evaluation_authority(
        candidate_rows=rows,
        j2_authority=J2,
        repository_authority=REPO,
        tool_identity={"ffmpeg_path": "/usr/bin/ffmpeg", "ffmpeg_revision": "test"},
    )
    assert preview["candidate_count"] == 7
    assert preview["speaker_label_count"] == 7
    assert preview["action_vector"]["request_operator_identity_review"] is True
    assert preview["action_vector"]["freeze_cohort_or_gold"] is False
    assert preview["did_load_or_run_models"] is False


def test_preview_rejects_only_candidate_without_usable_speech(tmp_path):
    rows = [_row(tmp_path, index) for index in range(1, 9)]
    transcript = Path(rows[3]["transcript_path"])
    transcript.write_text(
        json.dumps({"utterances": [{"speaker": "A", "start": 0, "end": 1000, "text": ""}]}),
        encoding="utf-8",
    )
    rows[3]["transcript_sha256"] = e1.sha256_file(transcript)
    preview = e1.preview_generation5_evaluation_authority(
        candidate_rows=rows,
        j2_authority=J2,
        repository_authority=REPO,
        tool_identity={"ffmpeg_path": "/usr/bin/ffmpeg", "ffmpeg_revision": "test"},
    )
    assert preview["enumerated_candidate_count"] == 8
    assert preview["candidate_count"] == 7
    assert preview["rejected_candidate_count"] == 1
    assert preview["private_evidence"]["candidate_rejection_ledger"][0]["reason_code"] == "candidate_has_no_usable_speaker_utterance"


def test_apply_and_replay_are_private_and_idempotent(tmp_path, monkeypatch):
    rows = [_row(tmp_path, index) for index in range(1, 8)]
    preview = e1.preview_generation5_evaluation_authority(
        candidate_rows=rows,
        j2_authority=J2,
        repository_authority=REPO,
        tool_identity={"ffmpeg_path": "/usr/bin/ffmpeg", "ffmpeg_revision": "test"},
    )
    monkeypatch.setattr(e1, "preview_generation5_evaluation_authority", lambda: preview)
    monkeypatch.setattr(e1, "_j2_authority", lambda: J2)
    monkeypatch.setattr(
        e1,
        "_git",
        lambda arguments, binary=False: b"module" if arguments[0] == "show" else "",
    )

    def fake_extract(card, target, ffmpeg_path):
        target.write_bytes(str(card["speaker_ref"]).encode())

    monkeypatch.setattr(e1, "_extract", fake_extract)
    applied = e1.apply_generation5_evaluation_authority(
        preview,
        expected_content_sha256=preview["content_sha256"],
        runtime_root=tmp_path / "runtime",
    )
    replayed = e1.replay_generation5_evaluation_authority(
        preview["content_sha256"], runtime_root=tmp_path / "runtime"
    )
    assert applied["clip_count"] == 7
    assert replayed["idempotent_replay"] is True
    paths = e1._paths(tmp_path / "runtime", preview["content_sha256"])
    assert paths["manifest"].stat().st_mode & 0o777 == 0o600
    assert paths["page"].stat().st_mode & 0o777 == 0o600
