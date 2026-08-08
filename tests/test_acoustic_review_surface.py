from __future__ import annotations

import re
import subprocess

import pytest

from acoustic_plan0057_review import (
    CHRIS_SUBJECT_ID,
    ERIC_SUBJECT_ID,
    parse_review_answers,
)
from acoustic_review_surface import (
    AcousticReviewSurfaceError,
    build_answer_block,
    render_review_surface,
)


ALLOWED = frozenset({CHRIS_SUBJECT_ID, ERIC_SUBJECT_ID})
OPTIONS = [
    {
        "machine_identity": CHRIS_SUBJECT_ID,
        "display_label": "Enrolled subject A",
        "export_identity": CHRIS_SUBJECT_ID,
    },
    {
        "machine_identity": ERIC_SUBJECT_ID,
        "display_label": "Enrolled subject B",
        "export_identity": ERIC_SUBJECT_ID,
    },
]


def _cards(count: int = 2) -> list[dict]:
    return [
        {
            "card_id": f"synthetic-{index:02d}::SPEAKER_{index}",
            "speaker_ref": f"SPEAKER_{index}",
            "proposal_label": "Synthetic abstention",
            "proposal_subject_id": None,
            "confidence_band": "none",
            "supporting_unit_count": 0,
            "opposing_unit_count": 0,
            "transcript": f"Synthetic card {index}",
            "audio_url": f"clips/card-{index:02d}.wav",
        }
        for index in range(1, count + 1)
    ]


def test_renderer_has_accessible_controls_lazy_audio_and_fallback() -> None:
    page = render_review_surface(
        title="Synthetic review",
        cards=_cards(),
        enrolled_options=OPTIONS,
        allowed_subject_ids=ALLOWED,
    )

    assert page.count('<article class="card" data-review-card') == 2
    assert page.count("<fieldset data-decision-group") == 2
    assert page.count('preload="none"') == 2
    assert 'preload="metadata"' not in page
    assert page.count("Open audio directly") == 2
    assert page.count("<fieldset") == 2
    assert page.count("<legend>") == 2
    assert "Prepare answers" in page
    assert "Copy answers" in page


def test_renderer_escapes_content_and_emits_valid_javascript() -> None:
    cards = _cards(1)
    cards[0]["transcript"] = "<script>alert('private')</script>"
    page = render_review_surface(
        title="Synthetic <review>",
        cards=cards,
        enrolled_options=OPTIONS,
        allowed_subject_ids=ALLOWED,
    )
    script = re.search(r"<script>(.*)</script>", page, re.DOTALL)

    assert script is not None
    assert "<script>alert" not in page
    result = subprocess.run(
        ["node", "--check", "-"],
        input=script.group(1),
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_answer_block_round_trips_existing_strict_importer() -> None:
    card_ids = [
        *[f"synthetic-a::SPEAKER_{value}" for value in range(1, 4)],
        *[f"synthetic-b::SPEAKER_{value}" for value in range(1, 7)],
        *[f"synthetic-c::SPEAKER_{value}" for value in range(1, 7)],
    ]
    decisions = {card_id: "unknown" for card_id in card_ids}
    decisions[card_ids[0]] = CHRIS_SUBJECT_ID
    decisions[card_ids[1]] = ERIC_SUBJECT_ID
    decisions[card_ids[2]] = "neither_enrolled"
    export_labels = {
        CHRIS_SUBJECT_ID: CHRIS_SUBJECT_ID,
        ERIC_SUBJECT_ID: ERIC_SUBJECT_ID,
        "neither_enrolled": "Neither enrolled person",
        "unknown": "UNKNOWN",
    }

    answer_block = build_answer_block(
        card_ids=card_ids,
        decisions=decisions,
        export_labels=export_labels,
        allowed_subject_ids=ALLOWED,
        review_display_labels={card_ids[2]: "Synthetic reviewer label"},
    )
    parsed = parse_review_answers(answer_block, expected_card_ids=card_ids)

    assert len(parsed) == 15
    assert parsed[card_ids[0]]["actual_identity"] == CHRIS_SUBJECT_ID
    assert parsed[card_ids[1]]["actual_identity"] == ERIC_SUBJECT_ID
    assert parsed[card_ids[2]] == {
        "actual_identity": "neither_enrolled",
        "review_display_label": "Synthetic reviewer label",
    }


@pytest.mark.parametrize(
    ("decisions", "labels"),
    [
        ({"synthetic-01::SPEAKER_1": "unknown"}, {}),
        (
            {
                "synthetic-01::SPEAKER_1": "unknown",
                "synthetic-02::SPEAKER_2": "not-allowlisted",
            },
            {},
        ),
        (
            {
                "synthetic-01::SPEAKER_1": "unknown",
                "synthetic-02::SPEAKER_2": "unknown",
            },
            {"synthetic-02::SPEAKER_2": "unsafe=label"},
        ),
    ],
)
def test_answer_block_fails_closed(decisions: dict, labels: dict) -> None:
    with pytest.raises(AcousticReviewSurfaceError):
        build_answer_block(
            card_ids=[
                "synthetic-01::SPEAKER_1",
                "synthetic-02::SPEAKER_2",
            ],
            decisions=decisions,
            export_labels={
                CHRIS_SUBJECT_ID: CHRIS_SUBJECT_ID,
                ERIC_SUBJECT_ID: ERIC_SUBJECT_ID,
                "neither_enrolled": "Neither enrolled person",
                "unknown": "UNKNOWN",
            },
            allowed_subject_ids=ALLOWED,
            review_display_labels=labels,
        )


def test_renderer_rejects_unsafe_audio_path_or_missing_subject_option() -> None:
    cards = _cards(1)
    cards[0]["audio_url"] = "../private.wav"
    with pytest.raises(AcousticReviewSurfaceError):
        render_review_surface(
            title="Synthetic review",
            cards=cards,
            enrolled_options=OPTIONS,
            allowed_subject_ids=ALLOWED,
        )
    with pytest.raises(AcousticReviewSurfaceError):
        render_review_surface(
            title="Synthetic review",
            cards=_cards(1),
            enrolled_options=OPTIONS[:1],
            allowed_subject_ids=ALLOWED,
        )
