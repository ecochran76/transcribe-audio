from __future__ import annotations

import pytest
from types import SimpleNamespace

from speaker_identity_evidence_execution import (
    adapt_acoustic_review,
    normalize_explicit_provider_scopes,
    normalize_provider_lineage,
    transcript_speaker_timeline,
)
from speaker_identity_orchestration import IdentityOrchestrationError


SHA_A = "a" * 64
SHA_B = "b" * 64
SHA_C = "c" * 64
NOW = "2026-08-08T12:00:00+00:00"


def _review() -> dict[str, object]:
    return {
        "status": "complete_pending_human_review",
        "source_media_sha256": SHA_A,
        "execution_content_sha256": SHA_B,
        "identity_state_sha256": SHA_C,
        "rows": [
            {
                "speaker_ref": "SPEAKER_1",
                "disposition": "review",
                "subject_id": "subject-enrolled-1",
                "confidence_band": "low",
                "supporting_unit_count": 4,
                "opposing_unit_count": 1,
            },
            {
                "speaker_ref": "SPEAKER_2",
                "disposition": "abstain",
                "subject_id": None,
                "confidence_band": "none",
                "supporting_unit_count": 0,
                "opposing_unit_count": 0,
            },
        ],
    }


def test_acoustic_adapter_preserves_exact_denominator_and_lineage() -> None:
    bundle = adapt_acoustic_review(
        _review(),
        conversation_id="conversation-redacted",
        recording_id="recording-redacted",
        document_id="document-redacted",
        transcript_sha256=SHA_A,
        model_versions=(("adapter", "test-v1"),),
        created_at=NOW,
    )

    assert bundle.speaker_refs == ("SPEAKER_1", "SPEAKER_2")
    assert bundle.evidence[0].score == pytest.approx(4 / 9)
    assert bundle.evidence[1].insufficient_unit_count == 9
    assert len(bundle.lineage) == 2
    assert bundle.negative_actions and not any(bundle.negative_actions.values())


def test_acoustic_adapter_fails_closed_on_incomplete_review() -> None:
    review = _review()
    review["status"] = "partial"
    with pytest.raises(IdentityOrchestrationError) as excinfo:
        adapt_acoustic_review(
            review,
            conversation_id="conversation-redacted",
            recording_id="recording-redacted",
            document_id="document-redacted",
            transcript_sha256=SHA_A,
            model_versions=(("adapter", "test-v1"),),
            created_at=NOW,
        )
    assert excinfo.value.reason_code == "acoustic_review_incomplete"


def test_provider_scope_normalization_uses_only_operator_context() -> None:
    resolved = normalize_explicit_provider_scopes(
        {
            "source_contexts": [
                {
                    "source_id": "gws-private",
                    "owner": {"id": "person-owner"},
                    "relationship_scope": "personal",
                }
            ],
            "retrieval_sources": [
                {
                    "source_id": "gws-private",
                    "source_profile_id": "gws-private",
                    "provider_kind": "gws",
                    "account_id": "",
                    "tenant_id": "",
                    "evidence_capabilities": ["people"],
                }
            ],
        }
    )
    source = resolved["retrieval_sources"][0]
    assert source["account_id"] == "person-owner"
    assert source["tenant_id"] == "personal"


def test_provider_scope_normalization_rejects_unbound_source() -> None:
    with pytest.raises(IdentityOrchestrationError) as excinfo:
        normalize_explicit_provider_scopes(
            {
                "source_contexts": [],
                "retrieval_sources": [
                    {
                        "source_id": "gws-private",
                        "source_profile_id": "gws-private",
                        "provider_kind": "gws",
                        "evidence_capabilities": ["people"],
                    }
                ],
            }
        )
    assert excinfo.value.reason_code == "provider_scope_incomplete"


def test_transcript_timeline_converts_milliseconds_and_preserves_first_appearance() -> None:
    timeline, labels = transcript_speaker_timeline(
        [
            {"speaker": "B", "start": 8_000, "end": 12_500},
            {"speaker": "A", "start": 13_000, "end": 21_000},
            {"speaker": "B", "start": 22_000, "end": 25_000},
        ]
    )
    assert labels == ("B", "A")
    assert timeline[0] == {"speaker": "B", "start": 8.0, "end": 12.5}


def test_provider_lineage_hides_empty_native_record_handle() -> None:
    item = SimpleNamespace(
        snapshot=SimpleNamespace(
            evidence_id="00000000-0000-4000-8000-000000000001",
            source_profile_id="gws-private",
            provider_kind="gws",
            source_record_id="",
            independence_group_id="00000000-0000-4000-8000-000000000002",
            source_type="gws_contact",
            source_event_at=NOW,
            observed_at=NOW,
            retrieved_at=NOW,
            content_hash=SHA_A,
        )
    )
    lineage = normalize_provider_lineage(item)
    assert lineage.source_record_id.startswith("provider-record-")
    assert lineage.source_record_id != item.snapshot.evidence_id
