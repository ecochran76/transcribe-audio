from __future__ import annotations

import speaker_identity_plan0071_d2_review as review


def _cases() -> list[dict[str, str]]:
    return [
        {
            "speaker_ref": f"document-1::{label}",
            "recording_label": "Recording 1 of 1",
            "speaker_label": label,
            "original_recording_filename": "Original recording name.m4a",
            "clip_relative_path": f"clips/{label}.wav",
        }
        for label in ("A", "B", "C")
    ]


def test_html_is_filename_bearing_blind_and_incomplete_by_default() -> None:
    rendered = review.build_review_html(
        authority_content_sha256="a" * 64,
        cases=_cases(),
        people=[{"person_id": "person-1", "display_name": "Person One"}],
    )

    assert "Original recording name.m4a" in rendered
    assert rendered.count("<audio controls") == 3
    assert "no model predictions shown" in rendered
    assert '<button id="export" disabled>' in rendered
    assert "complete !== rows.length" in rendered
    assert review.DECISION_SCHEMA in rendered
    assert "candidate_person_id" not in rendered


def test_review_binds_exact_prediction_terminal() -> None:
    assert review.PREDICTION_RECEIPT_CONTENT_SHA256 == (
        "8de26c83af3a2dc1da7c04633fad4c698adcccf3972d42d56f7a8aecf86971b6"
    )
    assert review.PREDICTION_RESOLUTION_CONTENT_SHA256 == (
        "bf1876e0610f668ea8eaa4f5a0c4f3748540df36523e39b4410eb8428ebfe931"
    )
    assert all(value == 0 for value in review.MUTATION_EFFECT_COUNTS.values())
