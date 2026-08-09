from __future__ import annotations

import base64

import pytest

import speaker_identity_context_human_review as human_review
import speaker_identity_plan0062_human_comparison as comparison
from tests.test_speaker_identity_context_join import human_review_inputs


def sources():
    _cases, p3, p3_sha256, packets, acoustic_bundles = human_review_inputs()
    p4 = human_review.build_review_packet(
        p3,
        p3_content_sha256=p3_sha256,
        identity_packets=packets,
        acoustic_bundles=acoustic_bundles,
        enrolled_subject_labels={"subject-alice": "Enrolled Example"},
    )
    bindings = comparison.build_enrolled_option_bindings(
        p4, acoustic_bundles=acoustic_bundles
    )
    return p3, p4, bindings


def answer_block(
    p3, p4, *, new_person_first: bool = False, enrolled_first: bool = False
) -> str:
    rows = []
    for index, card in enumerate(p4["cards"]):
        if index == 0 and new_person_first:
            encoded = base64.urlsafe_b64encode("Zoë Example".encode()).decode().rstrip("=")
            selected = f"new_person:{encoded}"
        elif index == 0 and enrolled_first:
            selected = next(
                option["token"]
                for option in card["options"]
                if option["source"] == "enrolled_voice_subject"
            )
        else:
            canonical = next(
                (
                    option["token"]
                    for option in card["options"]
                    if option["source"] == "canonical_context_proposal"
                ),
                None,
            )
            suggested = next(
                (
                    option["token"]
                    for option in card["options"]
                    if option["source"] == "contextual_unlisted_suggestion"
                ),
                None,
            )
            selected = canonical or suggested or "unresolved"
        rows.append(f'{card["slot_id"]}={selected}')
    return "\n".join(
        [
            f"PLAN0062_SCHEMA={human_review.SUBMISSION_SCHEMA}",
            f"PLAN0062_P3_CONTENT_SHA256={p4['p3_content_sha256']}",
            f"PLAN0062_P4_CONTENT_SHA256={p4['content_sha256']}",
            *rows,
        ]
    )


def test_exact_submission_preserves_typed_new_person() -> None:
    p3, p4, bindings = sources()

    submission = comparison.parse_human_submission(
        answer_block(p3, p4, new_person_first=True),
        p3_manifest=p3,
        p4_source=p4,
        enrolled_binding_source=bindings,
    )

    assert submission["decision_count"] == 10
    assert submission["decisions"][0]["decision_type"] == "new_person"
    assert submission["decisions"][0]["label"] == "Zoë Example"
    assert not any(submission["negative_actions"].values())


def test_enrolled_selection_preserves_exact_private_subject_binding() -> None:
    p3, p4, bindings = sources()

    submission = comparison.parse_human_submission(
        answer_block(p3, p4, enrolled_first=True),
        p3_manifest=p3,
        p4_source=p4,
        enrolled_binding_source=bindings,
    )

    selected = submission["decisions"][0]
    assert selected["decision_type"] == "enrolled_voice_subject"
    assert selected["acoustic_subject_id"] == "subject-alice"
    assert selected["binding_status"] == (
        "reviewed_voice_subject_selected_pending_person_apply"
    )


def test_private_enrolled_bindings_freeze_and_replay(tmp_path) -> None:
    _p3, _p4, bindings = sources()

    first = comparison.freeze_enrolled_option_bindings(
        bindings, runtime_root=tmp_path / "plan-0062"
    )
    replay = comparison.replay_enrolled_option_bindings(
        content_sha256=bindings["content_sha256"],
        runtime_root=tmp_path / "plan-0062",
    )

    assert first["idempotent_replay"] is False
    assert replay["idempotent_replay"] is True
    assert replay["binding_count"] == bindings["binding_count"]
    assert replay["live_mutation_count"] == 0


def test_submission_rejects_stale_header_and_changed_slot_order() -> None:
    p3, p4, bindings = sources()
    valid = answer_block(p3, p4)

    with pytest.raises(comparison.Plan0062HumanComparisonError, match="stale"):
        comparison.parse_human_submission(
            valid.replace(p4["content_sha256"], "f" * 64, 1),
                p3_manifest=p3,
                p4_source=p4,
                enrolled_binding_source=bindings,
        )

    lines = valid.splitlines()
    lines[3], lines[4] = lines[4], lines[3]
    with pytest.raises(comparison.Plan0062HumanComparisonError, match="slot order"):
        comparison.parse_human_submission(
            "\n".join(lines),
            p3_manifest=p3,
            p4_source=p4,
            enrolled_binding_source=bindings,
        )


def test_comparison_scores_three_conditions_and_recommends_separate_plan() -> None:
    p3, p4, bindings = sources()
    submission = comparison.parse_human_submission(
        answer_block(p3, p4),
        p3_manifest=p3,
        p4_source=p4,
        enrolled_binding_source=bindings,
    )

    result = comparison.recompute_comparison(p3, p4, bindings, submission)

    assert result["speaker_slot_count"] == 10
    assert result["condition_count"] == 3
    assert all(
        metrics["evaluation_count"] == 10
        for metrics in result["condition_metrics"].values()
    )
    assert result["terminal_decision"] == "advance"
    assert result["recommended_next_action"].startswith("prepare_separate_")
    assert result["apply_authorized"] is False


def test_comparison_rejects_rehashed_decision_meaning_drift() -> None:
    p3, p4, bindings = sources()
    submission = comparison.parse_human_submission(
        answer_block(p3, p4),
        p3_manifest=p3,
        p4_source=p4,
        enrolled_binding_source=bindings,
    )
    submission["decisions"][0]["label"] = "Altered label"
    core = {key: value for key, value in submission.items() if key != "content_sha256"}
    submission["content_sha256"] = comparison.canonical_artifact_hash(core)

    with pytest.raises(comparison.Plan0062HumanComparisonError, match="meanings drifted"):
        comparison.recompute_comparison(p3, p4, bindings, submission)


def test_private_comparison_freezes_and_replays_without_apply(tmp_path) -> None:
    p3, p4, bindings = sources()
    block = answer_block(p3, p4, new_person_first=True)

    first = comparison.freeze_human_comparison(
        block,
        p3_manifest=p3,
        p4_source=p4,
        enrolled_binding_source=bindings,
        runtime_root=tmp_path / "plan-0062",
    )
    submission = comparison.parse_human_submission(
        block,
        p3_manifest=p3,
        p4_source=p4,
        enrolled_binding_source=bindings,
    )
    replay = comparison.replay_human_comparison(
        submission_sha256=submission["content_sha256"],
        p3_manifest=p3,
        p4_source=p4,
        enrolled_binding_source=bindings,
        runtime_root=tmp_path / "plan-0062",
    )

    assert first["idempotent_replay"] is False
    assert replay["idempotent_replay"] is True
    assert replay["metrics_recomputed"] is True
    assert replay["apply_authorized"] is False
    assert replay["live_mutation_count"] == 0
