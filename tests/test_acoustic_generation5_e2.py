from __future__ import annotations

import copy

import pytest

import acoustic_generation5_e2 as e2


def cards() -> list[dict]:
    values = []
    count = 0
    for ordinal, speaker_count in ((1, 5), (2, 3), (3, 3), (4, 3), (5, 4), (6, 2), (7, 2)):
        for index in range(speaker_count):
            count += 1
            label = chr(65 + index)
            values.append({
                "ordinal": ordinal,
                "display_case": f"Case {ordinal}",
                "speaker_label": label,
                "speaker_ref": f"Case {ordinal} / Speaker {label}",
                "clip": {"snippets": [{"text": f"clue {count}"}]},
            })
    return values


def matrices() -> list[dict]:
    return [
        {"candidate_id": candidate, "method_id": method,
         "speaker_count": e2.EXPECTED_SPEAKER_COUNT, "rows": []}
        for candidate in e2.acoustic_contract.CANDIDATE_IDS
        for method in e2.acoustic_contract.METHOD_IDS
    ]


def predictions(packet: dict) -> dict:
    return {"predictions": [{
        "speaker_ref": speaker["speaker_ref"],
        "identity_or_alias": "stable alias",
        "confidence_band": "low",
        "disposition": "review",
        "rationale": "Transcript evidence is limited.",
    } for speaker in packet["speakers"]]}


def test_context_packet_is_complete_and_gold_blind() -> None:
    packet = e2.build_context_worker_packet(cards())
    assert packet["speaker_count"] == 22
    assert packet["contains_gold"] is False
    assert packet["contains_acoustic_evidence"] is False
    serialized = e2.json.dumps(packet)
    for forbidden in e2.FORBIDDEN_WORKER_KEYS:
        assert forbidden not in serialized


def test_context_packet_rejects_a_gold_field() -> None:
    values = cards()
    values[0]["speaker_gold"] = [{"private_identity_display": "sealed"}]
    packet = e2.build_context_worker_packet(values)
    assert "speaker_gold" not in e2.json.dumps(packet)


def test_context_packet_requires_exact_22_speakers() -> None:
    with pytest.raises(e2.Generation5E2Error, match="Exactly 22"):
        e2.build_context_worker_packet(cards()[:-1])


def test_augmented_packet_requires_all_nine_units() -> None:
    packet = e2.build_context_worker_packet(cards())
    with pytest.raises(e2.Generation5E2Error, match="nine"):
        e2.build_augmented_worker_packet(packet, matrices()[:-1])


def test_augmented_packet_excludes_competing_predictions() -> None:
    packet = e2.build_augmented_worker_packet(e2.build_context_worker_packet(cards()), matrices())
    assert packet["acoustic_matrix_count"] == 9
    assert packet["contains_competing_worker_output"] is False
    assert "predictions" not in packet


def test_prediction_validation_freezes_exact_order() -> None:
    packet = e2.build_context_worker_packet(cards())
    value = predictions(packet)
    value["predictions"].reverse()
    frozen = e2.validate_predictions(
        value,
        expected_refs=[item["speaker_ref"] for item in packet["speakers"]],
        worker_lane="context_only",
    )
    assert frozen["predictions"][0]["speaker_ref"] == packet["speakers"][0]["speaker_ref"]
    assert frozen["speaker_count"] == 22


def test_prediction_validation_rejects_duplicates() -> None:
    packet = e2.build_context_worker_packet(cards())
    value = predictions(packet)
    value["predictions"][1] = copy.deepcopy(value["predictions"][0])
    with pytest.raises(e2.Generation5E2Error, match="duplicated"):
        e2.validate_predictions(
            value,
            expected_refs=[item["speaker_ref"] for item in packet["speakers"]],
            worker_lane="context_only",
        )


def test_live_preview_binds_frozen_denominators_without_gold() -> None:
    repository = {"commit": "a" * 40, "module_sha256": "b" * 64,
                  "clean": True, "upstream_ahead": 0, "upstream_behind": 0}
    preview = e2.preview_generation5_e2(repository_authority=repository)
    assert preview["speaker_count"] == 22
    assert preview["acoustic_matrix_count"] == 9
    assert preview["acoustic_trial_count"] == 396
    assert preview["contains_gold"] is False
    assert preview["did_run_workers_or_models"] is False
    assert preview["worker_runtime"]["provider"] == "openrouter"
    assert preview["worker_runtime"]["model"] == "openai/gpt-5.2"
    assert preview["worker_runtime"]["tools_enabled"] is False
    assert preview["superseded_no_output_attempt"]["prediction_captured"] is False
