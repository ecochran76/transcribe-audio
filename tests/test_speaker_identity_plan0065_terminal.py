from __future__ import annotations

import pytest

import speaker_identity_plan0065_terminal as terminal


def _receipts():
    d0 = {"status": "d0_frozen_zero_effect", "content_sha256": "a" * 64}
    d1 = {
        "status": "d1_pass_zero_effect",
        "content_sha256": "b" * 64,
        "development_gate": {"passed": True},
        "execution_counts": {"local_biometric_half_probe_count": 24},
    }
    d2 = {
        "content_sha256": terminal.D2_RECEIPT_SHA256,
        "context_gate": {
            "passed": False,
            "terminal_status": "context_recovery_failed",
        },
        "execution_counts": {
            "primary_model_turn_count": 4,
            "fallback_model_turn_count": 0,
        },
        "effect_counts": {"speaker_assignments": 0, "external_writes": 0},
    }
    reconciliation = {
        "status": "d2_local_identity_metadata_reconciled",
        "content_sha256": "c" * 64,
        "effect_accounting": {
            "d2_identity_container_mutation_document_count": 3,
            "restored_local_artifact_copy_count": 5,
            "reconciled_transcript_index_row_count": 3,
            "lasting_identity_container_mutation_count": 0,
            "speaker_assignments": 0,
            "external_writes": 0,
        },
    }
    return d0, d1, d2, reconciliation


def test_terminal_withholds_and_never_opens_fresh_evaluation():
    d0, d1, d2, reconciliation = _receipts()

    result = terminal.build_terminal(
        d0_receipt=d0,
        d1_receipt=d1,
        d2_receipt=d2,
        reconciliation_receipt=reconciliation,
    )

    assert result["terminal_decision"] == "withhold"
    assert result["reason_code"] == "context_recovery_failed"
    assert result["packet_state"]["d3"] == "not_opened"
    assert result["packet_state"]["e0"] == "not_opened"
    assert result["execution_counts"]["fresh_evaluation_run_count"] == 0
    assert not any(result["effect_counts"].values())
    assert result["local_reconciliation"]["detected_document_count"] == 3
    assert result["local_reconciliation"]["lasting_mutation_count"] == 0


def test_terminal_rejects_a_passing_or_effectful_d2_receipt():
    d0, d1, d2, reconciliation = _receipts()

    with pytest.raises(terminal.Plan0065TerminalError):
        terminal.build_terminal(
            d0_receipt=d0,
            d1_receipt=d1,
            d2_receipt={**d2, "context_gate": {"passed": True}},
            reconciliation_receipt=reconciliation,
        )
    with pytest.raises(terminal.Plan0065TerminalError):
        terminal.build_terminal(
            d0_receipt=d0,
            d1_receipt=d1,
            d2_receipt={**d2, "effect_counts": {"speaker_assignments": 1}},
            reconciliation_receipt=reconciliation,
        )


def test_terminal_rejects_missing_or_incomplete_local_reconciliation():
    d0, d1, d2, reconciliation = _receipts()

    with pytest.raises(terminal.Plan0065TerminalError):
        terminal.build_terminal(
            d0_receipt=d0,
            d1_receipt=d1,
            d2_receipt=d2,
            reconciliation_receipt={
                **reconciliation,
                "effect_accounting": {
                    **reconciliation["effect_accounting"],
                    "lasting_identity_container_mutation_count": 1,
                },
            },
        )
