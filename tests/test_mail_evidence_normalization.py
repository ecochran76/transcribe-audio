from __future__ import annotations

from conversation_knowledge_evidence import EvidenceSnapshotRecord
from mail_evidence_normalization import (
    classify_account_direction,
    classify_mail_temporal,
    mail_independence_key,
    normalize_mail_address,
    normalize_mail_evidence,
)
from mail_relationship_contracts import validate_mail_artifact


SCOPE = {
    "provider_kind": "mail_receipts",
    "profile_id": "mail-receipts-default",
    "account_id": "account-redacted",
    "tenant_id": "tenant-redacted",
    "namespace": "namespace-redacted",
    "corpus_id": "corpus-redacted",
    "capabilities": ["mail_metadata_read"],
}


def receipt(result_hashes: list[str]) -> dict[str, object]:
    return {
        "schema_version": "transcribe-audio.mail-query-receipt.v1",
        "receipt_id": "mail-query-redacted-1",
        "request_hash": "a" * 64,
        "source_scope": SCOPE,
        "capability": "mail_metadata_read",
        "query_mode": "exact_email_only",
        "exact_addresses": ["alex@example.test"],
        "as_of": "2026-01-07T16:00:00Z",
        "lookback_start": "2025-01-07T16:00:00Z",
        "budgets": {
            "max_records": 25,
            "max_characters": 1,
            "max_calls": 4,
            "max_latency_ms": 30_000,
            "max_pages": 10,
        },
        "status": "complete",
        "counts": {
            "selected": len(result_hashes),
            "excluded": 0,
            "truncated": 0,
            "provider_writes": 0,
        },
        "warnings": [],
        "failures": [],
        "result_hashes": result_hashes,
        "created_at": "2026-01-07T16:01:00Z",
    }


def snapshot(
    index: int,
    *,
    source_event_at: str,
    interaction: str,
    thread_hash: str,
    source_hash: str,
    signature_observations: list[dict[str, str]] | None = None,
) -> EvidenceSnapshotRecord:
    content_hash = f"{index:x}" * 64
    return EvidenceSnapshotRecord(
        evidence_id=f"snapshot-redacted-{index}",
        source_record_id=f"record-redacted-{index}",
        source_profile_id="mail-receipts-default",
        provider_kind="mail_receipts",
        account_id="account-redacted",
        tenant_id="tenant-redacted",
        source_type="mail_receipts_message_metadata",
        capability="mail_metadata_read",
        snippet="",
        structured_metadata={
            "evidence_id": f"evidence-redacted-{index}",
            "record_ref": f"record-redacted-{index}",
            "message_ref_hash": "b" * 64,
            "thread_ref_hash": thread_hash,
            "source_key_hash": source_hash,
            "from_addresses": ["alex@example.test"],
            "to_addresses": ["account@example.test"],
            "cc_addresses": [],
            "account_direction": "inbound",
            "contact_ids_by_address": {
                "alex@example.test": "contact-redacted-alex"
            },
            "signature_observations": signature_observations or [],
            "namespace": "namespace-redacted",
            "corpus_id": "corpus-redacted",
            "query_address": "alex@example.test",
            "provider_record_id": f"evidence-redacted-{index}",
        },
        source_event_at=source_event_at,
        observed_at="2026-01-07T16:01:00Z",
        retrieved_at="2026-01-07T16:01:00Z",
        temporal_class="later_retrieved",
        source_uri="",
        content_hash=content_hash,
        independence_group_id=interaction,
        freshness_state="current",
        redaction={
            "body_retained": False,
            "subject_retained": False,
            "provider_ids_hashed": True,
        },
        truncation={"snippet_characters": 0},
    )


def test_normalization_is_replay_and_input_order_independent() -> None:
    snapshots = (
        snapshot(
            1,
            source_event_at="2026-01-05T09:00:00Z",
            interaction="interaction-redacted-1",
            thread_hash="c" * 64,
            source_hash="d" * 64,
        ),
        snapshot(
            2,
            source_event_at="2026-01-05T09:00:00Z",
            interaction="interaction-redacted-1",
            thread_hash="c" * 64,
            source_hash="e" * 64,
        ),
        snapshot(
            3,
            source_event_at="2026-01-08T09:00:00Z",
            interaction="interaction-redacted-2",
            thread_hash="f" * 64,
            source_hash="d" * 64,
        ),
    )
    query_receipt = receipt([item.content_hash for item in snapshots])

    first = normalize_mail_evidence(snapshots, query_receipt=query_receipt)
    second = normalize_mail_evidence(
        tuple(reversed(snapshots)), query_receipt=query_receipt
    )

    assert first == second
    assert len(first.observations) == 3
    assert len(first.independence_groups) == 2
    duplicate_group = next(
        group for group in first.independence_groups if group["duplicate_count"] == 1
    )
    assert duplicate_group["source_count"] == 2
    assert duplicate_group["reason_code"] == "duplicate_interaction"
    hindsight = next(
        item for item in first.observations if item["temporal_class"] == "hindsight"
    )
    assert hindsight["excluded_reason_code"] == "temporal_after_as_of"
    for observation in first.observations:
        validate_mail_artifact("mail_observation", observation)
    for group in first.independence_groups:
        validate_mail_artifact("mail_independence_group", group)


def test_normalization_preserves_conflicting_structured_signatures() -> None:
    signatures = [
        {
            "address": "alex@example.test",
            "title": "Program Director",
            "organization": "Example Organization",
            "department": "Programs",
            "observed_at": "2025-12-01T09:00:00Z",
        },
        {
            "address": "alex@example.test",
            "title": "Acting Director",
            "organization": "Example Organization",
            "department": "Operations",
            "observed_at": "2026-01-06T09:00:00Z",
        },
    ]
    snapshots = tuple(
        snapshot(
            index,
            source_event_at=signature["observed_at"],
            interaction=f"interaction-redacted-{index}",
            thread_hash=("c" if index == 1 else "d") * 64,
            source_hash="e" * 64,
            signature_observations=[signature],
        )
        for index, signature in enumerate(signatures, start=1)
    )

    projection = normalize_mail_evidence(
        snapshots,
        query_receipt=receipt([item.content_hash for item in snapshots]),
    )

    assert [
        item["signature_observations"][0]["title"]
        for item in projection.observations
    ] == ["Program Director", "Acting Director"]


def test_pure_mail_identity_and_temporal_helpers_are_explicit() -> None:
    assert normalize_mail_address(" Alex@Example.Test ") == "alex@example.test"
    assert classify_account_direction(
        from_addresses=("account@example.test",),
        to_addresses=("alex@example.test",),
        cc_addresses=(),
        account_address="account@example.test",
    ) == "outbound"
    assert classify_account_direction(
        from_addresses=("alex@example.test",),
        to_addresses=("account@example.test",),
        cc_addresses=(),
        account_address="account@example.test",
    ) == "inbound"
    assert classify_mail_temporal(
        "2026-01-08T09:00:00Z", as_of="2026-01-07T16:00:00Z"
    ) == ("hindsight", "temporal_after_as_of")
    assert mail_independence_key(
        message_ref_hash="b" * 64,
        thread_ref_hash="c" * 64,
        source_event_at="2026-01-05T09:00:00Z",
        participants=("account@example.test", "alex@example.test"),
    ) == mail_independence_key(
        message_ref_hash="b" * 64,
        thread_ref_hash="d" * 64,
        source_event_at="2026-01-06T09:00:00Z",
        participants=("account@example.test", "blair@example.test"),
    )
