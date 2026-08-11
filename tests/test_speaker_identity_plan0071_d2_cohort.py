from __future__ import annotations

import speaker_identity_plan0071_d2_cohort as cohort


def _unit(index: int, *, speakers: int = 3) -> dict[str, object]:
    return {
        "chronological_rank": index + 1,
        "document_id": f"doc-{index}",
        "recording_time": f"2025-01-{index + 1:02d}T00:00:00Z",
        "original_recording_filename": f"original-{index}.m4a",
        "source_media_sha256": f"hash-{index}",
        "speaker_labels": [chr(65 + value) for value in range(speakers)],
        "transcript_artifact_valid": True,
        "source_media_artifact_valid": True,
    }


def test_structural_selection_is_oldest_forward_and_identity_blind() -> None:
    units = [
        _unit(0),
        _unit(1),
        _unit(2, speakers=2),
        _unit(3),
        _unit(4),
        _unit(5),
        _unit(6),
        _unit(7),
    ]

    result = cohort.select_structural_units(
        units,
        exposed_document_ids={"doc-1"},
        exposed_recording_hashes={"hash-4"},
        limit=4,
    )

    assert [item["document_id"] for item in result["selected"]] == [
        "doc-0",
        "doc-3",
        "doc-5",
        "doc-6",
    ]
    assert result["considered_count"] == 7
    assert result["selected_count"] == 4


def test_structural_selection_requires_original_filename() -> None:
    units = [_unit(index) for index in range(6)]
    units[0]["original_recording_filename"] = ""
    units.append(_unit(6))

    result = cohort.select_structural_units(
        units,
        exposed_document_ids=set(),
        exposed_recording_hashes=set(),
    )

    assert result["selected"][0]["document_id"] == "doc-1"
    assert "missing_original_recording_filename" in result["considered"][0][
        "reason_codes"
    ]
