import json
import os
from pathlib import Path

import acoustic_generation5_source_review as s1


def _private_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")
    os.chmod(path.parent, 0o700)
    os.chmod(path, 0o600)


def test_speaker_cards_reject_zero_duration_utterance(tmp_path):
    row = {"source_sha256": "a" * 64, "review_source_path": str(tmp_path / "a.wav")}
    job = {"source_sha256": "a" * 64, "transcript_id": "job"}
    result = {"source_sha256": "a" * 64, "provider_payload": {
        "id": "job", "status": "completed", "utterances": [
            {"speaker": "A", "start": 1000, "end": 1000, "text": "not playable"}
        ]}}
    try:
        s1._speaker_cards(row, 1, job, result)
    except s1.Generation5SourceReviewError as exc:
        assert "no playable utterance" in str(exc)
    else:
        raise AssertionError("zero-duration utterance was accepted")


def test_speaker_cards_build_required_refs_and_positive_clip(tmp_path):
    row = {"source_sha256": "a" * 64, "review_source_path": str(tmp_path / "a.wav")}
    job = {"source_sha256": "a" * 64, "transcript_id": "job"}
    result = {"source_sha256": "a" * 64, "provider_payload": {
        "id": "job", "status": "completed", "utterances": [
            {"speaker": "B", "start": 1000, "end": 9000, "text": "usable speech"}
        ]}}
    cards = s1._speaker_cards(row, 2, job, result)
    assert cards[0]["speaker_ref"] == "Required B / Speaker B"
    assert cards[0]["clip"]["duration_seconds"] > 0


def test_render_page_has_audio_and_copy_fallback():
    preview = {"private_evidence": {"cards": [{
        "ordinal": 1, "display_case": "Required A", "speaker_label": "A",
        "speaker_ref": "Required A / Speaker A", "source_path": "/private/a.m4a",
        "clip": {"start_seconds": 0.0, "duration_seconds": 4.0,
                 "snippets": [{"text": "hello"}]},
    }]}}
    page = s1._render_page(preview)
    assert "<audio controls" in page
    assert "Prepare answers" in page
    assert "Copy the selected block" in page
