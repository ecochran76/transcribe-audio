import hashlib
import json
from pathlib import Path

import acoustic_generation4_gold_review as review
from acoustic_generation4_cohort import evaluate_population


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
        rows=rows, gap_packet=gap, swap_packet=swap,
        enrolled_identity_names=("Enrolled One", "Enrolled Two"),
    )

    assert [card["display_case"] for card in plan["cards"]] == [
        "Case 1", "Case 2", "Replacement A"
    ]
    assert plan["cards"][0]["prefilled_name"] == "Known Person"
    assert plan["cards"][1]["prefilled_name"] == ""
    assert plan["manual_label_count"] == 2
    assert plan["enrolled_identity_names"] == ["Enrolled One", "Enrolled Two"]
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
        enrolled_identity_names=("Enrolled One", "Enrolled Two"),
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
    assert "Enrolled people to look for:" in html
    assert "Enrolled One" in html and "Enrolled Two" in html
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


def _answer_plan() -> dict:
    identities = [
        "Jordan Katz", "Nacu Hernandez", "Jordan Katz", "Nacu Hernandez",
        "Person Three", "Person Three", "Person Four", "Person Four",
        "Person Five",
    ]
    cards = []
    case_numbers = [1, 2, 3, 4, 5, 6, 6, 7, 7]
    for index, (identity, case_number) in enumerate(zip(identities, case_numbers)):
        source_sha = hashlib.sha256(f"gold-source-{case_number}".encode()).hexdigest()
        transcript_sha = hashlib.sha256(
            f"gold-transcript-{case_number}".encode()
        ).hexdigest()
        cards.append({
            "case_id": review._case_id(source_sha, transcript_sha),
            "display_case": f"Case {case_number}",
            "speaker_label": chr(65 + index),
            "speaker_ref": f"opaque-{index}",
            "source_sha256": source_sha,
            "transcript_sha256": transcript_sha,
            "prefilled_name": identity if index < 2 else "",
        })
    core = {
        "schema_version": review.BUNDLE_SCHEMA,
        "status": "private_operator_review_ready",
        "case_count": 7,
        "speaker_label_count": len(cards),
        "manual_label_count": len(cards) - 2,
        "enrolled_identity_names": [],
        "cards": cards,
        "contains_paths": True,
        "contains_private_membership": True,
        "contains_transcript_text": True,
        "contains_audio_excerpts": False,
        "contains_acoustic_scores": False,
        "did_run_acoustic_models": False,
        "did_freeze_cohort_or_gold": False,
        "supplemental_media_consumed": False,
    }
    return {**core, "content_sha256": review._canonical_hash(core)}


def test_answers_build_exact_private_gold_and_preserve_repeated_people():
    plan = _answer_plan()
    names = [
        "Jordan Katz", "Nacu Hernandez", "Jordan Katz", "Nacu Hernandez",
        "Person Three", "Person Three", "Person Four", "Person Four",
        "Person Five",
    ]
    answers = "\n".join(
        f"{card['speaker_ref']} = {name}"
        for card, name in zip(plan["cards"], names)
    )
    packet = review.build_generation4_private_gold(
        plan=plan,
        answer_text=answers,
        enrolled_identity_map={"Jordan Katz": "enrolled-jordan", "Nacu Hernandez": "enrolled-nacu"},
    )

    assert packet["schema_version"] == review.GOLD_SCHEMA
    assert len(packet["cases"]) == 7
    speakers = [speaker for case in packet["cases"] for speaker in case["speaker_gold"]]
    by_name = {}
    for speaker in speakers:
        by_name.setdefault(speaker["private_identity_display"], set()).add(
            speaker["person_id"]
        )
    assert len(by_name["Jordan Katz"]) == 1
    assert len(by_name["Nacu Hernandez"]) == 1
    expected_sources = {case["source_sha256"] for case in packet["cases"]}
    population = evaluate_population(packet["cases"], expected_sources=expected_sources)
    assert population["passing"] is True
    assert population["enrolled_people_with_two_sessions_count"] == 2
    assert population["same_person_session_pair_count"] >= 4


def test_answers_reject_unknown_missing_and_duplicate_values():
    plan = _answer_plan()
    valid = [
        f"{card['speaker_ref']} = Person {index}"
        for index, card in enumerate(plan["cards"])
    ]
    for broken in (
        valid[:-1],
        [*valid[:-1], f"{plan['cards'][-1]['speaker_ref']} = UNKNOWN"],
        [*valid, valid[0]],
    ):
        try:
            review.build_generation4_private_gold(
                plan=plan, answer_text="\n".join(broken), enrolled_identity_map={}
            )
        except review.Generation4GoldReviewError:
            pass
        else:
            raise AssertionError("incomplete or duplicate answers must fail closed")


def test_apply_private_gold_is_0600_and_idempotent(tmp_path: Path):
    plan = _answer_plan()
    answers = "\n".join(
        f"{card['speaker_ref']} = Person {index}"
        for index, card in enumerate(plan["cards"])
    )
    packet = review.build_generation4_private_gold(
        plan=plan, answer_text=answers, enrolled_identity_map={}
    )

    first = review.apply_generation4_private_gold(packet, output_root=tmp_path)
    second = review.apply_generation4_private_gold(packet, output_root=tmp_path)

    path = Path(first["private_gold_path"])
    assert first == second
    assert path.stat().st_mode & 0o777 == 0o600
    assert json.loads(path.read_text(encoding="utf-8")) == packet
    assert first["did_freeze_cohort_or_gold"] is False


def test_load_generation3_enrolled_identity_map_requires_exact_private_manifest(
    tmp_path: Path,
):
    bindings = []
    for name, person_ref in (("Jordan Katz", "person-jordan"), ("Nacu Hernandez", "person-nacu")):
        normalized = review._normalized_identity(name)
        bindings.append({
            "identity_name": name,
            "identity_name_sha256": hashlib.sha256(normalized.encode()).hexdigest(),
            "person_ref_id": person_ref,
        })
    path = _write_private(
        tmp_path / "generation3-gold.json",
        {
            "schema_version": review.GENERATION3_GOLD_MANIFEST_SCHEMA,
            "status": "applied_gold_frozen_evaluation_not_revealed",
            "preview": {"enrolled_identity_bindings": bindings},
        },
    )
    expected = hashlib.sha256(path.read_bytes()).hexdigest()

    result = review.load_generation3_enrolled_identity_map(
        path, expected_manifest_sha256=expected
    )

    assert result == {"Jordan Katz": "person-jordan", "Nacu Hernandez": "person-nacu"}
    try:
        review.load_generation3_enrolled_identity_map(
            path, expected_manifest_sha256="a" * 64
        )
    except review.Generation4GoldReviewError as exc:
        assert "drifted" in str(exc)
    else:
        raise AssertionError("stale Generation-3 identity authority must fail closed")
