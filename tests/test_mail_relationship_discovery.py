from __future__ import annotations

from mail_relationship_contracts import ZERO_EFFECTS, validate_mail_artifact
from relationship_role_discovery import discover_mail_relationship_hypotheses


CONTACTS = {
    "contact-account": {
        "contact_id": "contact-account",
        "label": "Account Contact",
        "email": "account@example.test",
        "contact_class": "person_candidate",
    },
    "contact-alex": {
        "contact_id": "contact-alex",
        "label": "Alex Example",
        "email": "alex@example.test",
        "contact_class": "person_candidate",
    },
    "contact-blair": {
        "contact_id": "contact-blair",
        "label": "Blair Example",
        "email": "blair@example.test",
        "contact_class": "person_candidate",
    },
    "contact-team": {
        "contact_id": "contact-team",
        "label": "Example Team",
        "email": "team@example.test",
        "contact_class": "shared_or_role_address",
    },
}


def observation(
    index: int,
    *,
    sender: str,
    recipients: list[str],
    group_id: str,
    observed_at: str,
    signature: dict[str, str] | None = None,
    excluded_reason_code: str | None = None,
) -> dict[str, object]:
    addresses = [sender, *recipients]
    contact_ids = {
        item["email"]: contact_id
        for contact_id, item in CONTACTS.items()
        if item["email"] in addresses
    }
    return {
        "schema_version": "transcribe-audio.mail-observation.v1",
        "observation_id": f"mail-observation-{index}",
        "query_receipt_id": "mail-query-redacted-1",
        "source_scope": {
            "provider_kind": "mail_receipts",
            "profile_id": "mail-receipts-default",
            "account_id": "account-redacted",
            "tenant_id": "tenant-redacted",
            "namespace": "namespace-redacted",
            "corpus_id": "corpus-redacted",
            "capabilities": ["mail_metadata_read"],
        },
        "capability": "mail_metadata_read",
        "source_ref": {
            "evidence_id": f"evidence-redacted-{index}",
            "record_ref": f"record-redacted-{index}",
            "message_ref_hash": f"{index:x}" * 64,
            "thread_ref_hash": f"{index + 5:x}" * 64,
        },
        "source_event_at": observed_at,
        "retrieved_at": "2026-01-07T16:01:00Z",
        "as_of": "2026-01-07T16:00:00Z",
        "temporal_class": (
            "hindsight" if excluded_reason_code == "temporal_after_as_of"
            else "contemporaneous"
        ),
        "participants": {"from": [sender], "to": recipients, "cc": []},
        "account_direction": (
            "outbound" if sender == "account@example.test" else "inbound"
        ),
        "contact_ids_by_address": contact_ids,
        "signature_observations": [signature] if signature else [],
        "independence_group_id": group_id,
        "redaction": {"body_retained": False},
        "truncation": {"snippet_characters": 0},
        "excluded_reason_code": excluded_reason_code,
    }


def groups(observations: list[dict[str, object]]) -> list[dict[str, object]]:
    return [
        {
            "schema_version": "transcribe-audio.mail-independence-group.v1",
            "group_id": item["independence_group_id"],
            "interaction_key_version": "mail-interaction-key.v1",
            "independent_thread_key": item["source_ref"]["thread_ref_hash"],
            "member_observation_ids": [item["observation_id"]],
            "duplicate_count": 0,
            "source_count": 1,
            "reason_code": None,
            "content_hash": f"{index + 10:x}" * 64,
        }
        for index, item in enumerate(observations)
    ]


def test_mail_discovery_builds_sent_and_bidirectional_correspondence() -> None:
    observations = [
        observation(
            1,
            sender="alex@example.test",
            recipients=["account@example.test"],
            group_id="mail-interaction-1",
            observed_at="2025-12-15T10:00:00Z",
        ),
        observation(
            2,
            sender="account@example.test",
            recipients=["alex@example.test"],
            group_id="mail-interaction-2",
            observed_at="2026-01-03T10:00:00Z",
        ),
    ]

    result = discover_mail_relationship_hypotheses(
        observations,
        groups(observations),
        contacts=CONTACTS,
        account_address="account@example.test",
        input_watermark="watermark-redacted-1",
    )

    assert [item["hypothesis_kind"] for item in result.hypotheses] == [
        "correspondence",
        "sent_mail",
        "sent_mail",
    ]
    correspondence = result.hypotheses[0]
    assert correspondence["independent_thread_count"] == 2
    assert correspondence["effect_counts"] == ZERO_EFFECTS
    assert correspondence["status"] == "proposed"
    assert "does not establish" in correspondence["why_not_accepted"]
    for hypothesis in result.hypotheses:
        validate_mail_artifact("mail_relationship_hypothesis", hypothesis)


def test_mail_discovery_finds_recurring_coparticipants_without_counting_account() -> None:
    observations = [
        observation(
            index,
            sender="account@example.test",
            recipients=["alex@example.test", "blair@example.test"],
            group_id=f"mail-interaction-{index}",
            observed_at=f"2026-01-0{index}T10:00:00Z",
        )
        for index in (1, 2)
    ]

    result = discover_mail_relationship_hypotheses(
        observations,
        groups(observations),
        contacts=CONTACTS,
        account_address="account@example.test",
        input_watermark="watermark-redacted-2",
    )

    coparticipants = [
        item
        for item in result.hypotheses
        if item["hypothesis_kind"] == "thread_coparticipation"
    ]
    assert len(coparticipants) == 1
    assert {
        coparticipants[0]["subject_contact_id"],
        coparticipants[0]["counterpart_id"],
    } == {"contact-alex", "contact-blair"}


def test_mail_discovery_preserves_role_conflicts_and_exclusions() -> None:
    first_signature = {
        "address": "alex@example.test",
        "title": "Program Director",
        "organization": "Example Organization",
        "department": "Programs",
        "observed_at": "2025-12-01T09:00:00Z",
    }
    second_signature = {
        "address": "alex@example.test",
        "title": "Acting Director",
        "organization": "Example Organization",
        "department": "Operations",
        "observed_at": "2026-01-06T09:00:00Z",
    }
    observations = [
        observation(
            1,
            sender="alex@example.test",
            recipients=["account@example.test"],
            group_id="mail-interaction-1",
            observed_at=first_signature["observed_at"],
            signature=first_signature,
        ),
        observation(
            2,
            sender="alex@example.test",
            recipients=["account@example.test"],
            group_id="mail-interaction-2",
            observed_at=second_signature["observed_at"],
            signature=second_signature,
        ),
        observation(
            3,
            sender="team@example.test",
            recipients=["account@example.test"],
            group_id="mail-interaction-3",
            observed_at="2026-01-06T10:00:00Z",
        ),
        observation(
            4,
            sender="alex@example.test",
            recipients=["account@example.test"],
            group_id="mail-interaction-4",
            observed_at="2026-01-08T10:00:00Z",
            excluded_reason_code="temporal_after_as_of",
        ),
    ]

    result = discover_mail_relationship_hypotheses(
        observations,
        groups(observations),
        contacts=CONTACTS,
        account_address="account@example.test",
        input_watermark="watermark-redacted-3",
    )

    roles = [
        item for item in result.hypotheses if item["hypothesis_kind"] == "contextual_role"
    ]
    affiliations = [
        item for item in result.hypotheses if item["hypothesis_kind"] == "affiliation"
    ]
    assert {item["counterpart_label"] for item in roles} == {
        "Acting Director",
        "Program Director",
    }
    assert sum(len(item["conflicts"]) for item in roles) == 2
    assert len(affiliations) == 1
    assert result.excluded_reason_counts == {
        "excluded_shared_address": 1,
        "temporal_after_as_of": 1,
    }
