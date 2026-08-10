from __future__ import annotations

from array import array
import hashlib
import json
import wave

import speaker_identity_plan0064_p4_review as p4


def test_people_uses_reviewed_canonical_profiles_only():
    manifest = {
        "canonical_bindings": {
            "current_person_profiles": [
                {
                    "person_id": "person-1",
                    "primary_name": "Person One",
                    "resolution_status": "reviewed",
                },
                {
                    "person_id": "person-2",
                    "primary_name": "Unreviewed",
                    "resolution_status": "pending",
                },
            ]
        }
    }
    assert p4._people(manifest) == [
        {"person_id": "person-1", "display_name": "Person One"}
    ]


def test_review_html_has_one_blind_audio_question_per_case():
    cases = [
        {
            "speaker_ref": f"doc::{label}",
            "recording_label": "Recording 1 of 1",
            "recording_filename": "Original & recording.m4a",
            "speaker_label": label,
            "clip_relative_path": f"clips/{label}.wav",
        }
        for label in ("A", "B")
    ]
    page = p4.build_review_html(
        authority_content_sha256="a" * 64,
        cases=cases,
        people=[{"person_id": "person-1", "display_name": "Person One"}],
    )
    assert page.count('<article class="card"') == 2
    assert page.count("<audio controls") == 2
    assert page.count('class="decision"') == 2
    assert "no model predictions shown" in page
    assert "Copy complete JSON" in page
    assert page.count("Original recording:") == 2
    assert page.count("Original &amp; recording.m4a") == 2


def test_original_recording_filename_uses_hash_bound_source_basename(tmp_path):
    transcript_path = tmp_path / "transcript.json"
    payload = {
        "source_media_path": r"C:\\Users\\operator\\Sound Recordings\\Original recording.m4a"
    }
    transcript_path.write_text(json.dumps(payload), encoding="utf-8")
    transcript_path.chmod(0o600)
    digest = hashlib.sha256(transcript_path.read_bytes()).hexdigest()

    assert p4._original_recording_filename(
        {
            "transcript_artifact": {
                "path": str(transcript_path),
                "sha256": digest,
            }
        },
        allowed_sha256=frozenset({digest}),
    ) == "Original recording.m4a"


def test_write_clip_creates_private_pcm16_wave(tmp_path):
    path = tmp_path / "clip.wav"
    p4._write_clip(path, array("f", [0.0, 0.5, -0.5]))
    with wave.open(str(path), "rb") as audio:
        assert audio.getnchannels() == 1
        assert audio.getsampwidth() == 2
        assert audio.getframerate() == p4.SAMPLE_RATE
        assert audio.getnframes() == 3
    assert path.stat().st_mode & 0o777 == 0o600
