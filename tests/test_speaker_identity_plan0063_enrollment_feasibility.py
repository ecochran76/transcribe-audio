from __future__ import annotations

import speaker_identity_plan0063_enrollment_feasibility as feasibility


def _window(recording: str, start: float, end: float) -> dict:
    return {
        "recording_id": recording,
        "start_seconds": start,
        "end_seconds": end,
    }


def test_select_windows_prefers_longest_and_rejects_cross_person_overlap() -> None:
    selected, conflicts = feasibility.select_nonoverlapping_windows(
        {
            "person-a": [
                _window("recording-1", 0.0, 4.0),
                _window("recording-1", 10.0, 18.0),
            ],
            "person-b": [
                _window("recording-1", 12.0, 16.0),
                _window("recording-1", 20.0, 25.0),
            ],
        }
    )

    assert selected["person-a"] == [
        _window("recording-1", 0.0, 4.0),
        _window("recording-1", 10.0, 18.0),
    ]
    assert selected["person-b"] == [_window("recording-1", 20.0, 25.0)]
    assert conflicts == [
        {
            "proposed_person_id": "person-b",
            "conflicting_person_id": "person-a",
            "recording_id": "recording-1",
            "start_seconds": 12.0,
            "end_seconds": 16.0,
        }
    ]


def test_select_windows_enforces_per_person_limit() -> None:
    selected, conflicts = feasibility.select_nonoverlapping_windows(
        {
            "person-a": [
                _window("recording-1", float(index * 10), float(index * 10 + index + 3))
                for index in range(8)
            ]
        },
        maximum_per_person=3,
    )

    assert len(selected["person-a"]) == 3
    assert [item["end_seconds"] - item["start_seconds"] for item in selected["person-a"]] == [
        8.0,
        9.0,
        10.0,
    ]
    assert conflicts == []


def test_slot_candidates_are_p1_bound_and_holdout_excluded() -> None:
    case = {
        "document_id": "document-1",
        "recording_id": "recording-1",
        "conversation_id": "conversation-1",
        "speaker_refs": ("SPEAKER_1", "SPEAKER_2"),
        "speaker_labels": ("A", "B"),
        "timeline": (
            {"speaker": "A", "start": 0.0, "end": 7.0},
            {"speaker": "B", "start": 7.0, "end": 12.0},
            {"speaker": "A", "start": 12.0, "end": 32.0},
        ),
    }
    lineage = {
        "authority": "p1_audio_derivative_replay",
        "source_blob_id": "source-1",
        "source_sha256": "a" * 64,
        "source_duration_seconds": 40.0,
        "audio_quality_sha256": "b" * 64,
    }

    windows = feasibility._slot_candidates(
        "document-1::SPEAKER_1", case, lineage
    )

    assert [(item["start_seconds"], item["end_seconds"]) for item in windows] == [
        (0.0, 7.0),
        (12.0, 27.0),
    ]
    assert all(item["lineage"]["authority"] == "p1_audio_derivative_replay" for item in windows)
    assert all(item["future_holdout_excluded"] is True for item in windows)
    assert all(item["data_split"] == "development_training_candidate" for item in windows)
