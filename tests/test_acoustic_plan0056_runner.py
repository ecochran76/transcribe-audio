from __future__ import annotations

import wave
from pathlib import Path

import acoustic_plan0056_runner as runner


def test_select_review_segments_is_deterministic_and_bounded() -> None:
    timeline = [
        {"speaker": "raw-b", "start": 20.0, "end": 25.0},
        {"speaker": "raw-a", "start": 0.0, "end": 5.0},
        {"speaker": "raw-a", "start": 10.0, "end": 20.0},
        {"speaker": "raw-b", "start": 30.0, "end": 33.0},
        {"speaker": "raw-a", "start": 40.0, "end": 41.0},
        {"speaker": "raw-c", "start": 50.0, "end": 52.5},
    ]

    selected = runner.select_review_segments(
        timeline,
        minimum_turn_seconds=2.0,
        maximum_turn_seconds=8.0,
        maximum_turns_per_speaker=6,
        target_seconds_per_speaker=24.0,
        minimum_usable_seconds_per_speaker=6.0,
    )

    assert list(selected) == ["SPEAKER_1", "SPEAKER_2"]
    assert selected["SPEAKER_1"][0] == {"start": 0.0, "end": 5.0}
    assert selected["SPEAKER_1"][1] == {"start": 10.0, "end": 18.0}
    assert selected["SPEAKER_2"] == [
        {"start": 20.0, "end": 25.0},
        {"start": 30.0, "end": 33.0},
    ]
    assert "SPEAKER_3" not in selected


def test_write_speaker_clip_concatenates_only_selected_audio(tmp_path: Path) -> None:
    source = tmp_path / "source.wav"
    with wave.open(str(source), "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(10)
        audio.writeframes(b"\x01\x00" * 100)

    result = runner._write_speaker_clip(
        source,
        tmp_path / "private" / "speaker.wav",
        ({"start": 1.0, "end": 2.5}, {"start": 7.0, "end": 8.0}),
    )

    with wave.open(result["clip_path"], "rb") as clip:
        assert clip.getframerate() == 10
        assert clip.getnframes() == 25
    assert result["duration_seconds"] == 2.5
    assert len(result["clip_sha256"]) == 64
