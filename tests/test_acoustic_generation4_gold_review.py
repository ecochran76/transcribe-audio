import hashlib
import json
from pathlib import Path

import acoustic_generation4_gold_review as review


def _write_private(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    path.chmod(0o600)
    return path


def _sealed(schema: str, body: dict) -> dict:
    payload = {"schema_version": schema, **body}
    return {**payload, "content_sha256": review._canonical_hash(payload)}


def test_build_plan_preserves_case_numbers_and_marks_replacements(tmp_path: Path):
    rows = []
    gap_cases = []
    best_ids = []
    for index in range(1, 4):
        source = hashlib.sha256(f"source-{index}".encode()).hexdigest()
        transcript = hashlib.sha256(f"transcript-{index}".encode()).hexdigest()
        case_id = review._case_id(source, transcript)
        best_ids.append(case_id)
        transcript_path = _write_private(
            tmp_path / f"transcript-{index}.json",
            {
                "utterances": [
                    {"speaker": "A", "start": 1000, "end": 8000,
                     "text": f"recognizable sample {index}"},
                ]
            },
        )
        source_path = tmp_path / f"source-{index}.m4a"
        source_path.write_bytes(b"media")
        rows.append(
            {
                "source_sha256": source,
                "transcript_sha256": transcript,
                "source_path": str(source_path),
                "transcript_path": str(transcript_path),
                "speaker_labels": ["A"],
            }
        )
        if index < 3:
            gap_cases.append(
                {
                    "case_id": case_id,
                    "speaker_reviews": [
                        {"speaker_ref": f"opaque-ref-{index}", "review_status": "gap"}
                    ],
                }
            )
    gap = {
        "cases": gap_cases,
        "review_gaps": [
            {"speaker_ref": "opaque-ref-2", "speaker_label": "A"}
        ],
        "supported_operator_assertions": [
            {
                "speaker_ref": "opaque-ref-1",
                "speaker_label": "A",
                "person_display_name": "Known Person",
            }
        ],
    }
    swap = {"opaque_best_subset_case_ids": best_ids}

    plan = review.build_generation4_gold_review_plan(
        rows=rows, gap_packet=gap, swap_packet=swap
    )

    assert [card["display_case"] for card in plan["cards"]] == [
        "Case 1", "Case 2", "Replacement A"
    ]
    assert plan["cards"][0]["prefilled_name"] == "Known Person"
    assert plan["cards"][1]["prefilled_name"] == ""
    assert plan["manual_label_count"] == 2
    assert plan["cards"][0]["clip"]["duration_seconds"] <= 25


def test_apply_bundle_writes_private_html_audio_and_copy_template(tmp_path: Path):
    source_sha = hashlib.sha256(b"source").hexdigest()
    transcript_sha = hashlib.sha256(b"transcript").hexdigest()
    case_id = review._case_id(source_sha, transcript_sha)
    source = tmp_path / "source.m4a"
    source.write_bytes(b"media")
    transcript = _write_private(
        tmp_path / "transcript.json",
        {
            "utterances": [
                {"speaker": "A", "start": 2000, "end": 9000,
                 "text": "private transcript clue"}
            ]
        },
    )
    plan = review.build_generation4_gold_review_plan(
        rows=[{
            "source_sha256": source_sha,
            "transcript_sha256": transcript_sha,
            "source_path": str(source),
            "transcript_path": str(transcript),
            "speaker_labels": ["A"],
        }],
        gap_packet={
            "cases": [{
                "case_id": case_id,
                "speaker_reviews": [{"speaker_ref": "case-1:A"}],
            }],
            "supported_operator_assertions": [],
        },
        swap_packet={"opaque_best_subset_case_ids": [case_id]},
    )

    def fake_extract(_source, _start, _duration, target):
        target.write_bytes(b"RIFF-private-audio")

    receipt = review.apply_generation4_gold_review_bundle(
        plan, output_root=tmp_path / "output", extractor=fake_extract
    )

    page = Path(receipt["private_review_page_path"])
    clip = page.parent / "clips" / "case-1-a.wav"
    assert page.is_file() and clip.is_file()
    assert page.stat().st_mode & 0o777 == 0o600
    assert clip.stat().st_mode & 0o777 == 0o600
    html = page.read_text(encoding="utf-8")
    assert "Copy all answers" in html
    assert "private transcript clue" in html
    assert "Case 1 / Speaker A =" in html
    assert receipt["contains_transcript_text"] is True
    assert receipt["contains_audio_excerpts"] is True
    assert receipt["did_run_acoustic_models"] is False


def test_plan_rejects_missing_best_subset_case(tmp_path: Path):
    missing = "g1a-case-" + "a" * 20
    try:
        review.build_generation4_gold_review_plan(
            rows=[], gap_packet={"cases": [], "supported_operator_assertions": []},
            swap_packet={"opaque_best_subset_case_ids": [missing]},
        )
    except review.Generation4GoldReviewError as exc:
        assert "best-subset case" in str(exc)
    else:
        raise AssertionError("missing private membership must fail closed")


def test_existing_source_transcript_mode_does_not_block_private_derivative(tmp_path: Path):
    transcript = tmp_path / "source-transcript.json"
    transcript.write_text(
        json.dumps({
            "utterances": [
                {"speaker": "A", "start": 0, "end": 3000, "text": "sample"}
            ]
        }),
        encoding="utf-8",
    )
    transcript.chmod(0o644)

    result = review._utterance_plan(transcript, "A")

    assert result["duration_seconds"] == 4.5
