from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

import acoustic_generation3_gold as gold


def _fixture_context(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    cohort_manifest = tmp_path / "cohort.json"
    cohort_manifest.write_text("{}", encoding="utf-8")
    records = []
    for index in range(7):
        source_sha = f"{index + 1:064x}"
        transcript = {
            "utterances": [
                {
                    "speaker": "A",
                    "start": 0,
                    "end": 1000,
                    "text": f"I am Enrolled One in conversation {index}.",
                },
                {
                    "speaker": "B",
                    "start": 1100,
                    "end": 2100,
                    "text": f"I am Enrolled Two in conversation {index}.",
                },
            ]
        }
        records.append(
            {
                "source_sha256": source_sha,
                "transcript_sha256": f"{index + 101:064x}",
                "recording_id": f"recording-{index}",
                "conversation_id": f"conversation-{index}",
                "transcript": transcript,
                "bindings": [
                    {"speaker_label": "A", "speaker_label_id": f"label-{index}-a"},
                    {"speaker_label": "B", "speaker_label_id": f"label-{index}-b"},
                    {"speaker_label": "C", "speaker_label_id": f"label-{index}-c"},
                    {"speaker_label": "D", "speaker_label_id": f"label-{index}-d"},
                ],
            }
        )
    manifest = {
        "authority_id": "generation3-cohort-test",
        "preview": {"membership_sha256": "a" * 64},
    }
    monkeypatch.setattr(
        gold,
        "_cohort_context",
        lambda **_kwargs: (manifest, records),
    )
    monkeypatch.setattr(
        gold,
        "_enrolled_bindings",
        lambda _values: {
            "subject-one": {
                "person_ref_id": "subject-one",
                "identity_name": "Enrolled One",
                "identity_name_sha256": "b" * 64,
                "training_intake_id": "training-one",
                "source_sha256": "c" * 64,
                "speaker_label_id": "training-label-one",
                "evidence": {},
            },
            "subject-two": {
                "person_ref_id": "subject-two",
                "identity_name": "Enrolled Two",
                "identity_name_sha256": "d" * 64,
                "training_intake_id": "training-one",
                "source_sha256": "e" * 64,
                "speaker_label_id": "training-label-two",
                "evidence": {},
            },
        },
    )
    return cohort_manifest


def _outcomes() -> list[dict[str, object]]:
    values: list[dict[str, object]] = []
    for index in range(7):
        source_sha = f"{index + 1:064x}"
        if index < 2:
            values.extend(
                [
                    {
                        "source_sha256": source_sha,
                        "speaker_label": "A",
                        "outcome": "enrolled",
                        "identity_name": "Enrolled One",
                        "person_ref_id": "subject-one",
                        "evidence": [
                            {
                                "kind": "transcript",
                                "method": "self_identification",
                                "identity_claim": "Enrolled One",
                                "utterance_indices": [0],
                            }
                        ],
                    },
                    {
                        "source_sha256": source_sha,
                        "speaker_label": "B",
                        "outcome": "enrolled",
                        "identity_name": "Enrolled Two",
                        "person_ref_id": "subject-two",
                        "evidence": [
                            {
                                "kind": "transcript",
                                "method": "self_identification",
                                "identity_claim": "Enrolled Two",
                                "utterance_indices": [1],
                            }
                        ],
                    },
                ]
            )
        elif index < 5:
            values.extend(
                [
                    {
                        "source_sha256": source_sha,
                        "speaker_label": "A",
                        "outcome": "open_set",
                        "identity_name": f"Open Person {index}",
                        "person_ref_id": "",
                        "evidence": [
                            {
                                "kind": "operator_confirmation",
                                "statement": f"Case {index} A is Open Person {index}.",
                            }
                        ],
                    },
                    {
                        "source_sha256": source_sha,
                        "speaker_label": "B",
                        "outcome": "unknown",
                        "identity_name": "",
                        "person_ref_id": "",
                        "evidence": [],
                    },
                ]
            )
        else:
            values.extend(
                [
                    {
                        "source_sha256": source_sha,
                        "speaker_label": label,
                        "outcome": "unknown",
                        "identity_name": "",
                        "person_ref_id": "",
                        "evidence": [],
                    }
                    for label in ("A", "B")
                ]
            )
        values.extend(
            [
                {
                    "source_sha256": source_sha,
                    "speaker_label": label,
                    "outcome": "unknown",
                    "identity_name": "",
                    "person_ref_id": "",
                    "evidence": [],
                }
                for label in ("C", "D")
            ]
        )
    return values


def _preview_inputs(cohort_manifest: Path) -> dict[str, object]:
    return {
        "cohort_manifest_path": cohort_manifest,
        "conversations": [],
        "source_root": cohort_manifest.parent,
        "enrolled_identity_bindings": [],
        "outcomes": _outcomes(),
    }


def test_gold_preview_is_complete_private_and_portable_is_aggregate_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cohort_manifest = _fixture_context(tmp_path, monkeypatch)
    preview = gold.preview_generation3_gold(**_preview_inputs(cohort_manifest))
    assert preview["gold_label_count"] == 28
    assert preview["known_subject_count"] == 5
    assert preview["enrolled_conversation_counts"] == {
        "subject-one": 2,
        "subject-two": 2,
    }
    assert preview["action_vector"]["reveal_evaluation"] is False

    portable = gold.portable_gold_projection(preview)
    serialized = json.dumps(portable, sort_keys=True)
    assert "identity_name" not in serialized
    assert "bounded_text" not in serialized
    assert "source_sha256" not in serialized
    assert "subject-one" not in serialized
    assert "subject-two" not in serialized
    assert portable["contains_subject_ids"] is False
    assert portable["contains_transcript_text"] is False


def test_gold_preview_rejects_missing_unconfirmed_and_underpowered_gold(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cohort_manifest = _fixture_context(tmp_path, monkeypatch)
    inputs = _preview_inputs(cohort_manifest)
    inputs["outcomes"] = _outcomes()[:-1]
    with pytest.raises(gold.Generation3GoldError, match="Every cohort label"):
        gold.preview_generation3_gold(**inputs)

    outcomes = _outcomes()
    open_index = next(
        index for index, item in enumerate(outcomes)
        if item["outcome"] == "open_set"
    )
    outcomes[open_index] = {**outcomes[open_index], "evidence": []}
    inputs["outcomes"] = outcomes
    with pytest.raises(gold.Generation3GoldError, match="Open-set gold"):
        gold.preview_generation3_gold(**inputs)

    outcomes = _outcomes()
    for item in outcomes:
        if item["outcome"] == "open_set":
            item.update(
                {
                    "identity_name": "Same Open Person",
                    "evidence": [
                        {
                            "kind": "operator_confirmation",
                            "statement": "This label is Same Open Person.",
                        }
                    ],
                }
            )
    inputs["outcomes"] = outcomes
    with pytest.raises(gold.Generation3GoldError, match="population minimum"):
        gold.preview_generation3_gold(**inputs)

    outcomes = _outcomes()
    outcomes[0] = {
        **outcomes[0],
        "evidence": [
            {
                "kind": "transcript",
                "method": "self_identification",
                "identity_claim": "Unrelated Person",
                "utterance_indices": [0],
            }
        ],
    }
    inputs["outcomes"] = outcomes
    with pytest.raises(
        gold.Generation3GoldError, match="does not contain the identity claim"
    ):
        gold.preview_generation3_gold(**inputs)

    outcomes = _outcomes()
    outcomes[0] = {
        **outcomes[0],
        "evidence": [
            {
                "kind": "transcript",
                "method": "self_identification",
                "identity_claim": "I",
                "utterance_indices": [0],
            }
        ],
    }
    inputs["outcomes"] = outcomes
    with pytest.raises(
        gold.Generation3GoldError,
        match="Enrolled transcript evidence does not bind identity",
    ):
        gold.preview_generation3_gold(**inputs)


def test_gold_apply_replay_and_stale_preview(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cohort_manifest = _fixture_context(tmp_path, monkeypatch)
    runtime_root = tmp_path / "runtime"
    inputs = _preview_inputs(cohort_manifest)
    preview = gold.preview_generation3_gold(**inputs)
    repository = {
        "commit": "a" * 40,
        "module_sha256": "b" * 64,
        "training_dependency_sha256": "c" * 64,
        "p3_dependency_sha256": "d" * 64,
        "private_io_dependency_sha256": "e" * 64,
        "gold_module_sha256": "f" * 64,
        "clean": True,
        "upstream_ahead": 0,
        "upstream_behind": 0,
    }
    monkeypatch.setattr(gold, "_repository_authority", lambda: repository)
    monkeypatch.setattr(
        gold, "_validate_repository_authority", lambda value: dict(value)
    )
    receipt = gold.apply_generation3_gold(
        preview,
        expected_preview_content_sha256=preview["content_sha256"],
        runtime_root=runtime_root,
        **inputs,
    )
    assert receipt["status"] == "applied_gold_frozen_evaluation_not_revealed"
    assert receipt["action_vector"]["freeze_gold"] is True
    assert receipt["action_vector"]["reveal_evaluation"] is False
    assert Path(receipt["private_manifest_path"]).stat().st_mode & 0o777 == 0o600

    replay = gold.replay_generation3_gold(
        Path(receipt["private_manifest_path"]),
        runtime_root=runtime_root,
        **inputs,
    )
    assert replay["idempotent_replay"] is True
    assert replay["manifest_sha256"] == receipt["manifest_sha256"]

    changed = copy.deepcopy(preview)
    changed["known_subject_count"] += 1
    with pytest.raises(gold.Generation3GoldError, match="stale"):
        gold.apply_generation3_gold(
            changed,
            expected_preview_content_sha256=preview["content_sha256"],
            runtime_root=tmp_path / "other-runtime",
            **inputs,
        )
