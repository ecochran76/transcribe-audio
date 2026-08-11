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


def test_prediction_authority_requires_frozen_six_person_roster(monkeypatch) -> None:
    monkeypatch.setattr(review.attempt2, "replay_attempt2", lambda **_kwargs: {
        "content_sha256": review.PREDICTION_RECEIPT_CONTENT_SHA256
    })
    monkeypatch.setattr(review.attempt2, "_paths", lambda _root: {
        "manifest": "manifest.json", "resolution": "resolution.json"
    })
    manifest = {
        "content_sha256": review.PREDICTION_MANIFEST_CONTENT_SHA256,
        "human_gold_read": False,
        "execution_counts": {"capture_evaluation_calls": 0},
        "mutation_effect_counts": review.MUTATION_EFFECT_COUNTS,
    }
    resolution = {
        "content_sha256": review.PREDICTION_RESOLUTION_CONTENT_SHA256
    }
    monkeypatch.setattr(
        review,
        "read_private_object",
        lambda path: manifest if str(path) == "manifest.json" else resolution,
    )
    monkeypatch.setattr(review, "_validate_content", lambda *_args: None)
    monkeypatch.setattr(review.predictions, "_bound_authorities", lambda _root: {
        "cohort_manifest": {}, "p0_binding": {"path": "p0.json"}
    })
    monkeypatch.setattr(review.predictions, "_selected", lambda _manifest: [])
    monkeypatch.setattr(
        review.plan0064_p4,
        "_people",
        lambda _manifest: [
            {"person_id": str(index), "display_name": f"Person {index}"}
            for index in range(6)
        ],
    )

    result = review._prediction_authority(review.DEFAULT_RUNTIME_ROOT)

    assert len(result["people"]) == 6
