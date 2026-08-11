from __future__ import annotations

import pytest

import speaker_identity_plan0066_a0 as a0


def _cases() -> list[dict]:
    return [
        {
            "document_id": f"doc-{index:02d}",
            "content_sha256": f"{index:064x}",
            "speaker_slots": [
                {"speaker_ref": f"doc-{index:02d}::{slot}"}
                for slot in range(4 if index < 3 else 3)
            ],
        }
        for index in range(12)
    ]


def test_activation_manifest_binds_closed_world_denominator_and_roster() -> None:
    cases = _cases()
    bindings = [
        {
            "document_id": case["document_id"],
            "source_sha256": "a" * 64,
            "stored_sha256": "a" * 64,
            "index_row_sha256": "b" * 64,
        }
        for case in cases
    ]
    roster = [
        {
            "person_id": f"00000000-0000-4000-8000-{index:012d}",
            "primary_name": f"Person {index}",
        }
        for index in range(6)
    ]

    manifest = a0.build_activation_manifest(
        terminal={
            "status": "withhold",
            "content_sha256": "c" * 64,
            "terminal_file_sha256": "d" * 64,
        },
        cases=cases,
        document_bindings=bindings,
        reviewed_roster=roster,
        provider_readiness={"did_send_model_turn": False},
    )

    assert manifest["development_denominator"]["case_count"] == 12
    assert manifest["development_denominator"]["speaker_slot_count"] == 39
    assert len(manifest["reviewed_roster"]) == 6
    assert manifest["effect_counts"] == a0.EFFECT_COUNTS
    assert manifest["content_sha256"]


def test_activation_manifest_rejects_incomplete_reviewed_roster() -> None:
    with pytest.raises(a0.Plan0066A0Error, match="six-person"):
        a0.build_activation_manifest(
            terminal={"status": "withhold"},
            cases=_cases(),
            document_bindings=[
                {"document_id": case["document_id"]} for case in _cases()
            ],
            reviewed_roster=[],
            provider_readiness={"did_send_model_turn": False},
        )
