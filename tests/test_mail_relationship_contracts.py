from __future__ import annotations

import json
from pathlib import Path

import pytest

import mail_relationship_contracts


def query_receipt() -> dict:
    return {
        "schema_version": "transcribe-audio.mail-query-receipt.v1",
        "receipt_id": "mail-query-redacted-1",
        "request_hash": "a" * 64,
        "source_scope": {
            "provider_kind": "mail_receipts",
            "profile_id": "operator-lite",
            "account_id": "account-redacted",
            "tenant_id": "tenant-redacted",
            "namespace": "namespace-redacted",
            "corpus_id": "corpus-redacted",
            "capabilities": ["mail_metadata_read"],
        },
        "capability": "mail_metadata_read",
        "query_mode": "exact_email_only",
        "exact_addresses": ["alex@example.test"],
        "as_of": "2026-01-07T16:00:00Z",
        "lookback_start": "2025-01-07T16:00:00Z",
        "budgets": {
            "max_records": 25,
            "max_characters": 1,
            "max_calls": 2,
            "max_latency_ms": 5_000,
            "max_pages": 2,
        },
        "status": "complete",
        "counts": {
            "selected": 1,
            "excluded": 0,
            "truncated": 0,
            "provider_writes": 0,
        },
        "warnings": [],
        "failures": [],
        "result_hashes": ["b" * 64],
        "created_at": "2026-01-07T16:01:00Z",
    }


def mail_observation() -> dict:
    return {
        "schema_version": "transcribe-audio.mail-observation.v1",
        "observation_id": "mail-observation-redacted-1",
        "query_receipt_id": "mail-query-redacted-1",
        "source_scope": query_receipt()["source_scope"],
        "capability": "mail_metadata_read",
        "source_ref": {
            "evidence_id": "evidence-redacted-1",
            "record_ref": "record-redacted-1",
            "message_ref_hash": "c" * 64,
            "thread_ref_hash": "d" * 64,
        },
        "source_event_at": "2026-01-06T12:00:00Z",
        "retrieved_at": "2026-01-07T16:01:00Z",
        "as_of": "2026-01-07T16:00:00Z",
        "temporal_class": "contemporaneous",
        "participants": {
            "from": ["alex@example.test"],
            "to": ["account@example.test"],
            "cc": [],
        },
        "account_direction": "inbound",
        "contact_ids_by_address": {
            "alex@example.test": "contact-redacted-alex",
        },
        "signature_observations": [
            {
                "address": "alex@example.test",
                "title": "Program Director",
                "organization": "Example Organization",
                "department": "Programs",
                "observed_at": "2026-01-06T12:00:00Z",
            }
        ],
        "independence_group_id": "mail-interaction-redacted-1",
        "redaction": {"body_retained": False},
        "truncation": {"snippet_characters": 0},
        "excluded_reason_code": None,
    }


def test_contract_freezes_schema_versions_thresholds_and_reason_codes() -> None:
    contract = mail_relationship_contracts.contract()

    assert contract["schema_version"] == "transcribe-audio.mail-relationship-contract.v1"
    assert contract["artifact_schemas"] == {
        "mail_query_receipt": "transcribe-audio.mail-query-receipt.v1",
        "mail_observation": "transcribe-audio.mail-observation.v1",
        "mail_independence_group": "transcribe-audio.mail-independence-group.v1",
        "mail_relationship_hypothesis": "transcribe-audio.mail-relationship-hypothesis.v1",
    }
    assert contract["thresholds"] == {
        "min_correspondence_threads": 2,
        "min_coparticipant_threads": 2,
        "max_records": 250,
        "max_characters": 1,
        "max_calls": 4,
        "max_latency_ms": 30_000,
        "max_lookback_days": 365,
        "max_retries": 1,
        "max_pilot_conversations": 25,
        "max_page_size": 100,
        "max_pages": 10,
    }
    assert set(contract["reason_codes"]) == {
        "excluded_shared_address",
        "excluded_role_address",
        "excluded_mailing_list",
        "excluded_automated_sender",
        "excluded_unresolved_contact",
        "temporal_after_as_of",
        "provider_response_invalid",
        "duplicate_interaction",
        "budget_exhausted",
        "partial_source_failure",
        "namespace_scope_mismatch",
        "unsupported_capability",
        "opaque_reference_unavailable",
    }
    assert contract["effects"] == {
        "accepted_relationships": 0,
        "accepted_roles": 0,
        "provider_writes": 0,
        "person_merges": 0,
        "speaker_assignments": 0,
        "biometric_effects": 0,
        "graphiti_writes": 0,
    }


def test_query_receipt_accepts_an_exact_bounded_operator_lite_read() -> None:
    assert mail_relationship_contracts.validate_mail_artifact(
        "mail_query_receipt", query_receipt()
    ) == query_receipt()


def test_portable_mail_artifacts_reject_raw_message_content() -> None:
    payload = query_receipt()
    payload["failures"] = [{"body": "private message content"}]

    with pytest.raises(
        mail_relationship_contracts.MailRelationshipContractError,
        match="prohibited raw content fields: body",
    ):
        mail_relationship_contracts.validate_mail_artifact(
            "mail_query_receipt", payload
        )


def test_mail_observation_requires_complete_source_and_temporal_provenance() -> None:
    with pytest.raises(
        mail_relationship_contracts.MailRelationshipContractError,
        match="mail_observation is missing required fields",
    ):
        mail_relationship_contracts.validate_mail_artifact(
            "mail_observation",
            {"schema_version": "transcribe-audio.mail-observation.v1"},
        )


def test_mail_observation_accepts_exact_participants_and_structured_role_evidence() -> None:
    assert mail_relationship_contracts.validate_mail_artifact(
        "mail_observation", mail_observation()
    ) == mail_observation()


def test_independence_group_requires_membership_and_duplicate_accounting() -> None:
    with pytest.raises(
        mail_relationship_contracts.MailRelationshipContractError,
        match="mail_independence_group is missing required fields",
    ):
        mail_relationship_contracts.validate_mail_artifact(
            "mail_independence_group",
            {"schema_version": "transcribe-audio.mail-independence-group.v1"},
        )


def test_mail_hypothesis_requires_evidence_time_and_zero_effects() -> None:
    with pytest.raises(
        mail_relationship_contracts.MailRelationshipContractError,
        match="mail_relationship_hypothesis is missing required fields",
    ):
        mail_relationship_contracts.validate_mail_artifact(
            "mail_relationship_hypothesis",
            {
                "schema_version": (
                    "transcribe-audio.mail-relationship-hypothesis.v1"
                )
            },
        )


def test_redacted_p0_artifacts_validate_against_the_frozen_contract() -> None:
    fixture_root = Path("docs/dev/fixtures/plan-0073-p0")
    catalog = json.loads((fixture_root / "contract-catalog.json").read_text())
    artifacts = json.loads((fixture_root / "portable-artifacts.json").read_text())

    assert catalog == mail_relationship_contracts.contract()
    assert {
        kind: mail_relationship_contracts.validate_mail_artifact(kind, payload)
        for kind, payload in artifacts["artifacts"].items()
    } == artifacts["artifacts"]


def test_p0_publishes_one_closed_json_schema_per_portable_artifact() -> None:
    fixture_root = Path("docs/dev/fixtures/plan-0073-p0")
    contract = mail_relationship_contracts.contract()

    for kind, schema_version in contract["artifact_schemas"].items():
        schema = json.loads((fixture_root / "schemas" / f"{kind}.schema.json").read_text())
        assert schema["$schema"] == "https://json-schema.org/draft/2020-12/schema"
        assert schema["$id"] == schema_version
        assert schema["type"] == "object"
        assert schema["additionalProperties"] is False
        assert "schema_version" in schema["required"]


def test_p0_freezes_all_required_redacted_discovery_scenarios() -> None:
    fixture = json.loads(
        Path("docs/dev/fixtures/plan-0073-p0/discovery-scenarios.json").read_text()
    )

    assert fixture["synthetic_test_only"] is True
    assert {case["case_id"] for case in fixture["cases"]} == {
        "one_way_transmission",
        "bidirectional_correspondence",
        "recurring_thread_coparticipation",
        "cross_source_duplicate",
        "conflicting_structured_roles",
        "shared_address_exclusion",
        "post_as_of_exclusion",
    }
    serialized = json.dumps(fixture, sort_keys=True).casefold()
    assert '"body"' not in serialized
    assert '"body_text"' not in serialized
    assert '"body_html"' not in serialized
    assert "@example.test" in serialized
