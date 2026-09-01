from __future__ import annotations

import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from people_organization_activity import build_directory_index
import conversation_knowledge_store
from conversation_evidence_fabric import EvidenceAnchor, EvidenceFabric, EvidenceRequest
from conversation_knowledge_evidence import EvidenceScope
from identity_learning_ledger import IdentityLearningLedger


def test_name_overlap_is_one_unresolved_group_without_becoming_a_person() -> None:
    source_payload = {
        "schema_version": "transcribe-audio.identity-people.v1",
        "items": [
            {
                "person_id": "contact:baker-work",
                "identity_kind": "local_contact",
                "status": "review_required",
                "primary_name": "Baker Kuehl",
                "aliases": ["baker.work@example.test"],
                "source_records": [
                    {"source_record_id": "calendar:baker-work"},
                    {"source_record_id": "gws:baker-work"},
                    {"source_record_id": "crm:baker-work"},
                ],
                "calendar_occurrences": [
                    {
                        "document_id": "recording-a",
                        "recorded_at": "2026-07-01T12:00:00Z",
                        "recording_title": "Conversation A",
                        "evidence_source": "calendar.attendees",
                    }
                ],
                "review_occurrences": [],
                "relationship_hypotheses": [],
                "organizations": [],
            },
            {
                "person_id": "contact:baker-personal",
                "identity_kind": "local_contact",
                "status": "provisional",
                "primary_name": "Baker Kuehl",
                "aliases": [],
                "source_records": [
                    {"source_record_id": "calendar:baker-personal"},
                    {"source_record_id": "gws:baker-personal"},
                    {"source_record_id": "crm:baker-personal"},
                    {"source_record_id": "crm:baker-personal-duplicate-view"},
                ],
                "calendar_occurrences": [
                    {
                        "document_id": "recording-b",
                        "recorded_at": "2026-06-01T12:00:00Z",
                        "recording_title": "Conversation B",
                        "evidence_source": "calendar.attendees",
                    }
                ],
                "review_occurrences": [],
                "relationship_hypotheses": [],
                "organizations": [],
            },
            {
                "person_id": "review:person-baker-kuehl",
                "identity_kind": "reviewed_speaker",
                "status": "reviewed",
                "primary_name": "Baker Kuehl",
                "aliases": [],
                "source_records": [
                    {"source_record_id": "operator-review:baker"},
                ],
                "calendar_occurrences": [],
                "review_occurrences": [
                    {
                        "document_id": "recording-c",
                        "recorded_at": "2026-05-01T12:00:00Z",
                        "recording_title": "Conversation C",
                        "speaker_ref": "speaker-1",
                    }
                ],
                "relationship_hypotheses": [],
                "organizations": [],
            },
        ],
    }

    index = build_directory_index(source_payload)

    assert index["schema_version"] == "transcribe-audio.people-organization-activity-index.v3"
    assert index["counts"] == {
        "people": 0,
        "unresolved_groups": 1,
            "source_records": 8,
            "organizations": 0,
            "review_leads": 0,
        }
    group = index["people"][0]
    assert group["entity_kind"] == "unresolved_group"
    assert group["resolution_state"] == "review_required"
    assert group["accepted_person_id"] == ""
    assert [member["record_id"] for member in group["members"]] == [
        "contact:baker-personal",
        "contact:baker-work",
        "review:person-baker-kuehl",
    ]
    assert group["identity_health"]["member_count"] == 3
    assert group["identity_health"]["source_record_count"] == 8
    assert group["activity_summary"]["calendar"] == {
        "confirmed_count": 0,
        "proposed_count": 2,
        "coverage_state": "partial",
        "first_at": "2026-06-01T12:00:00Z",
        "last_at": "2026-07-01T12:00:00Z",
    }
    assert group["activity_summary"]["transcript"] == {
        "confirmed_count": 0,
        "proposed_count": 1,
        "coverage_state": "partial",
        "first_at": "2026-05-01T12:00:00Z",
        "last_at": "2026-05-01T12:00:00Z",
    }


def test_provider_organization_strings_create_one_proposed_entity_not_an_employer() -> None:
    source_payload = {
        "schema_version": "transcribe-audio.identity-people.v1",
        "items": [
            {
                "person_id": "contact:alex",
                "identity_kind": "local_contact",
                "status": "provisional",
                "primary_name": "Alex Example",
                "aliases": [],
                "source_records": [{"source_record_id": "gws:alex"}],
                "organizations": ["Acme Research"],
                "calendar_occurrences": [
                    {
                        "document_id": "meeting-a",
                        "recorded_at": "2026-04-01T12:00:00Z",
                        "event_summary": "Acme introduction",
                    }
                ],
                "review_occurrences": [],
                "relationship_hypotheses": [],
            },
            {
                "person_id": "contact:morgan",
                "identity_kind": "local_contact",
                "status": "provisional",
                "primary_name": "Morgan Example",
                "aliases": [],
                "source_records": [{"source_record_id": "crm:morgan"}],
                "organizations": [" ACME RESEARCH "],
                "calendar_occurrences": [],
                "review_occurrences": [],
                "relationship_hypotheses": [],
            },
        ],
    }

    index = build_directory_index(source_payload)

    assert index["counts"]["organizations"] == 1
    organization = index["organizations"][0]
    assert organization["primary_name"] == "Acme Research"
    assert organization["resolution_state"] == "proposed"
    assert organization["accepted_organization_id"] == ""
    assert organization["identity_health"] == {
        "source_name_count": 2,
        "affiliation_count": 2,
        "requires_review": True,
    }
    assert organization["activity_summary"]["calendar"]["confirmed_count"] == 0
    assert organization["activity_summary"]["calendar"]["proposed_count"] == 1
    for person in index["people"]:
        assert len(person["organizations"]) == 1
        affiliation = person["organizations"][0]
        assert affiliation["organization_id"] == organization["organization_id"]
        assert affiliation["primary_name"] == "Acme Research"
        assert affiliation["status"] == "proposed"
        assert affiliation["basis"] == "provider_organization_string"
        assert affiliation["roles"] == []
        assert affiliation["role_count"] == 0
        assert affiliation["affiliation_id"].startswith("affiliation:")
        assert person["primary_affiliation"] == affiliation
        assert person["additional_organization_count"] == 0


def test_directory_carries_provider_affiliation_and_role_review_leads() -> None:
    source_payload = {
        "schema_version": "transcribe-audio.identity-people.v1",
        "items": [
            {
                "person_id": "contact:alex",
                "identity_kind": "local_contact",
                "status": "provisional",
                "primary_name": "Alex Example",
                "aliases": [],
                "source_records": [{"source_record_id": "contact-alex"}],
                "organizations": ["Example Labs"],
                "calendar_occurrences": [],
                "review_occurrences": [],
                "role_hypotheses": [
                    {
                        "hypothesis_id": "role-hypothesis-alex",
                        "hypothesis_kind": "contextual_role",
                        "subject_contact_id": "contact-alex",
                        "display_value": "Research Director",
                        "organization": "Example Labs",
                        "department": "Research",
                        "status": "proposed",
                        "source_content_sha256": "a" * 64,
                        "projection_version": "1",
                        "review_state": "unreviewed",
                    }
                ],
                "relationship_hypotheses": [
                    {
                        "hypothesis_id": "affiliation-hypothesis-alex",
                        "hypothesis_kind": "affiliation",
                        "subject_contact_id": "contact-alex",
                        "relationship_type": "AFFILIATED_WITH",
                        "counterpart_id": "organization-example-labs",
                        "counterpart_label": "Example Labs",
                        "status": "proposed",
                        "source_content_sha256": "b" * 64,
                        "projection_version": "1",
                        "review_state": "unreviewed",
                    }
                ],
            }
        ],
    }

    index = build_directory_index(source_payload)

    assert index["schema_version"] == "transcribe-audio.people-organization-activity-index.v3"
    assert index["counts"]["review_leads"] == 2
    person = index["people"][0]
    assert [lead["hypothesis_kind"] for lead in person["review_leads"]] == [
        "affiliation",
        "contextual_role",
    ]
    assert person["review_leads"][0]["subject_contact_id"] == "contact-alex"
    assert person["review_leads"][1]["display_value"] == "Research Director"
    assert person["identity_health"]["review_lead_count"] == 2


def test_activity_counts_independent_evidence_and_preserves_participation_state() -> None:
    source_payload = {
        "schema_version": "transcribe-audio.identity-people.v1",
        "items": [
            {
                "person_id": "person:reviewed-alex",
                "identity_kind": "canonical_person",
                "status": "reviewed",
                "primary_name": "Alex Example",
                "aliases": [],
                "source_records": [{"source_record_id": "gws:alex"}],
                "organizations": [],
                "calendar_occurrences": [
                    {
                        "document_id": "meeting-a",
                        "recorded_at": "2026-04-01T12:00:00Z",
                        "event_summary": "Invitation copy one",
                    },
                    {
                        "document_id": "meeting-a",
                        "recorded_at": "2026-04-01T12:00:00Z",
                        "event_summary": "Invitation duplicate source copy",
                    },
                ],
                "review_occurrences": [
                    {
                        "document_id": "meeting-a",
                        "reviewed_at": "2026-04-02T12:00:00Z",
                        "recording_title": "Reviewed conversation",
                        "speaker_ref": "speaker-1",
                    }
                ],
                "relationship_hypotheses": [
                    {
                        "hypothesis_id": "mail-hypothesis-a",
                        "hypothesis_kind": "sent_mail",
                        "status": "accepted",
                        "review_state": "accepted",
                        "observation_count": 3,
                        "first_observed_at": "2026-03-01T12:00:00Z",
                        "last_observed_at": "2026-03-03T12:00:00Z",
                        "evidence_independence_group_ids": [
                            "mail-copy-a",
                            "mail-copy-a",
                            "mail-message-b",
                        ],
                    }
                ],
            }
        ],
    }

    person = build_directory_index(source_payload)["people"][0]

    assert person["activity_summary"]["calendar"] == {
        "confirmed_count": 0,
        "proposed_count": 1,
        "coverage_state": "partial",
        "first_at": "2026-04-01T12:00:00Z",
        "last_at": "2026-04-01T12:00:00Z",
    }
    assert person["activity_summary"]["transcript"]["confirmed_count"] == 1
    assert person["activity_summary"]["transcript"]["proposed_count"] == 0
    assert person["activity_summary"]["email"] == {
        "confirmed_count": 2,
        "proposed_count": 0,
        "coverage_state": "partial",
        "first_at": "2026-03-01T12:00:00Z",
        "last_at": "2026-03-03T12:00:00Z",
    }
    assert len(person["activities"]) == 3


def test_schema_v9_rebuilds_organization_activity_and_coverage_authority(
    tmp_path: Path,
) -> None:
    store = conversation_knowledge_store.ConversationKnowledgeStore(tmp_path)
    migration = store.migrate(backup=False)
    assert migration.to_version == 9
    ledger = IdentityLearningLedger(tmp_path)

    def append(event_type: str, payload: dict[str, object], ordinal: int) -> str:
        return ledger.append_event(
            event_type=event_type,
            payload=payload,
            actor_id="reviewer:test",
            occurred_at=f"2026-09-01T12:{ordinal:02d}:00Z",
            idempotency_key=f"plan76-event-{ordinal}",
        ).event_id

    person_id = "00000000-0000-4000-8000-000000000761"
    append(
        "person_created",
        {"person_id": person_id, "primary_name": "Alex Example", "status": "reviewed"},
        1,
    )
    append(
        "organization_created",
        {
            "organization_id": "organization-acme",
            "primary_name": "Acme Research",
            "status": "reviewed",
            "organization_type": "company",
            "domains": ["acme.example"],
            "websites": ["https://acme.example"],
            "locations": ["Ames, Iowa"],
        },
        2,
    )
    append(
        "organization_source_observed",
        {
            "source_record_id": "organization-source-acme",
            "organization_id": "organization-acme",
            "source_profile_id": "crm-company",
            "provider_kind": "odollo",
            "account_id": "company",
            "tenant_id": "tenant",
            "record_type": "organization",
            "external_ref": "res.partner:42",
            "label": "ACME Research",
            "observed_at": "2026-09-01T12:02:00Z",
            "content_hash": "organization-source-hash",
        },
        3,
    )
    append(
        "role_asserted",
        {
            "role_id": "role-alex-acme",
            "person_id": person_id,
            "role_type": "works_for",
            "organization_id": "organization-acme",
            "starts_at": "2025-01-01T00:00:00Z",
            "status": "reviewed",
            "evidence_ids": ["organization-source-acme"],
        },
        4,
    )
    activity_id = append(
        "activity_observed",
        {
            "observation_id": "activity-mail-1",
            "subject_type": "person",
            "subject_id": person_id,
            "channel": "email",
            "occurred_at": "2026-08-30T15:00:00Z",
            "direction": "outbound",
            "participation_status": "observed",
            "evidence_status": "accepted",
            "source_profile_id": "mail-default",
            "account_id": "account",
            "tenant_id": "tenant",
            "source_record_id": "mail-receipt-1",
            "independence_group_id": "logical-message-1",
            "content_hash": "activity-content-hash",
            "source_locator": {"evidence_id": "evidence-redacted-1"},
        },
        5,
    )
    append(
        "activity_coverage_observed",
        {
            "subject_type": "person",
            "subject_id": person_id,
            "channel": "email",
            "coverage_state": "partial",
            "observed_at": "2026-09-01T12:05:00Z",
            "source_profile_ids": ["mail-default"],
            "metadata": {"reason": "bounded_query"},
        },
        6,
    )

    first = ledger.rebuild()
    snapshot = ledger.projection_snapshot()

    assert snapshot["organizations"]["organization-acme"]["primary_name"] == "Acme Research"
    assert snapshot["organization_sources"]["organization-source-acme"]["organization_id"] == "organization-acme"
    assert snapshot["roles"]["role-alex-acme"]["organization_id"] == "organization-acme"
    assert snapshot["activities"]["activity-mail-1"]["independence_group_id"] == "logical-message-1"
    assert snapshot["activity_coverage"][f"person:{person_id}:email"]["coverage_state"] == "partial"
    assert ledger.rebuild().projection_hash == first.projection_hash

    # The generic helper cannot specify a reversed event, so reverse it explicitly.
    ledger.append_event(
        event_type="event_reversed",
        payload={"reason": "mail observation was associated with the wrong person"},
        actor_id="reviewer:test",
        occurred_at="2026-09-01T12:07:00Z",
        idempotency_key="plan76-event-7",
        reverses_event_id=activity_id,
    )
    ledger.rebuild()
    assert "activity-mail-1" not in ledger.projection_snapshot()["activities"]


def test_directory_groups_multiple_roles_without_collapsing_organizations() -> None:
    person_id = "00000000-0000-4000-8000-000000000771"
    source_payload = {
        "schema_version": "transcribe-audio.identity-people.v1",
        "items": [
            {
                "person_id": person_id,
                "identity_kind": "canonical_person",
                "status": "reviewed",
                "primary_name": "Jordan Example",
                "aliases": [],
                "source_records": [{"source_record_id": "gws:jordan"}],
                "organizations": [],
                "calendar_occurrences": [],
                "review_occurrences": [],
                "relationship_hypotheses": [],
            }
        ],
    }
    authority_snapshot = {
        "organizations": {
            "organization-acme": {
                "organization_id": "organization-acme",
                "primary_name": "Acme Research",
                "status": "reviewed",
                "aliases_json": "[]",
                "domains_json": "[]",
                "websites_json": "[]",
                "locations_json": "[]",
                "organization_type": "company",
                "merged_into_organization_id": "",
            },
            "organization-beta": {
                "organization_id": "organization-beta",
                "primary_name": "Beta Foundation",
                "status": "reviewed",
                "aliases_json": "[]",
                "domains_json": "[]",
                "websites_json": "[]",
                "locations_json": "[]",
                "organization_type": "nonprofit",
                "merged_into_organization_id": "",
            },
        },
        "organization_sources": {},
        "roles": {
            "role-acme-founder": {
                "role_id": "role-acme-founder",
                "person_id": person_id,
                "role_type": "founder",
                "organization_id": "organization-acme",
                "starts_at": "2020-01-01T00:00:00Z",
                "ends_at": "",
                "status": "reviewed",
                "evidence_ids_json": '["evidence-founder"]',
            },
            "role-acme-ceo": {
                "role_id": "role-acme-ceo",
                "person_id": person_id,
                "role_type": "chief_executive_officer",
                "organization_id": "organization-acme",
                "starts_at": "2021-01-01T00:00:00Z",
                "ends_at": "",
                "status": "accepted",
                "evidence_ids_json": '["evidence-ceo"]',
            },
            "role-beta-advisor": {
                "role_id": "role-beta-advisor",
                "person_id": person_id,
                "role_type": "advisor",
                "organization_id": "organization-beta",
                "starts_at": "2024-01-01T00:00:00Z",
                "ends_at": "",
                "status": "proposed",
                "evidence_ids_json": '["evidence-advisor"]',
            },
        },
        "activities": {},
        "activity_coverage": {},
    }

    first = build_directory_index(
        source_payload,
        authority_snapshot=authority_snapshot,
    )
    second = build_directory_index(
        source_payload,
        authority_snapshot=authority_snapshot,
    )
    person = first["people"][0]

    assert first["semantic_hash"] == second["semantic_hash"]
    assert [item["organization_id"] for item in person["organizations"]] == [
        "organization-acme",
        "organization-beta",
    ]
    assert [role["role_id"] for role in person["organizations"][0]["roles"]] == [
        "role-acme-ceo",
        "role-acme-founder",
    ]
    assert person["primary_affiliation"]["organization_id"] == "organization-acme"
    assert person["primary_affiliation"]["role_types"] == [
        "chief_executive_officer",
        "founder",
    ]
    assert person["additional_organization_count"] == 1
    assert person["identity_health"]["affiliation_count"] == 2
    assert person["identity_health"]["role_count"] == 3


def test_organization_alias_merge_split_and_reversal_preserve_history(
    tmp_path: Path,
) -> None:
    store = conversation_knowledge_store.ConversationKnowledgeStore(tmp_path)
    store.migrate(backup=False)
    ledger = IdentityLearningLedger(tmp_path)

    def append(
        event_type: str,
        payload: dict[str, object],
        ordinal: int,
        *,
        reverses_event_id: str = "",
    ) -> str:
        return ledger.append_event(
            event_type=event_type,
            payload=payload,
            actor_id="reviewer:test",
            occurred_at=f"2026-09-01T13:{ordinal:02d}:00Z",
            idempotency_key=f"plan76-organization-event-{ordinal}",
            reverses_event_id=reverses_event_id,
        ).event_id

    person_id = "00000000-0000-4000-8000-000000000762"
    append(
        "person_created",
        {"person_id": person_id, "primary_name": "Morgan Example", "status": "reviewed"},
        1,
    )
    for ordinal, organization_id, name in (
        (2, "organization-acme", "Acme Research"),
        (3, "organization-acme-labs", "ACME Labs"),
        (4, "organization-independent", "Independent Research"),
    ):
        append(
            "organization_created",
            {
                "organization_id": organization_id,
                "primary_name": name,
                "status": "reviewed",
                "organization_type": "company",
            },
            ordinal,
        )
    append(
        "organization_alias_added",
        {"organization_id": "organization-acme", "alias": "Acme"},
        5,
    )
    append(
        "organization_source_observed",
        {
            "source_record_id": "organization-source-labs",
            "organization_id": "organization-acme-labs",
            "source_profile_id": "crm-company",
            "provider_kind": "odollo",
            "record_type": "organization",
            "external_ref": "res.partner:77",
            "observed_at": "2026-08-01T00:00:00Z",
            "content_hash": "organization-source-labs-hash",
        },
        6,
    )
    append(
        "role_asserted",
        {
            "role_id": "role-morgan-labs",
            "person_id": person_id,
            "role_type": "works_for",
            "organization_id": "organization-acme-labs",
            "status": "reviewed",
            "evidence_ids": ["organization-source-labs"],
        },
        7,
    )
    append(
        "activity_observed",
        {
            "observation_id": "activity-labs-calendar",
            "subject_type": "organization",
            "subject_id": "organization-acme-labs",
            "channel": "calendar",
            "occurred_at": "2026-08-10T00:00:00Z",
            "participation_status": "candidate",
            "evidence_status": "proposed",
            "source_profile_id": "calendar-default",
            "source_record_id": "event-77",
            "independence_group_id": "event-77",
            "content_hash": "activity-labs-hash",
        },
        8,
    )
    merge_id = append(
        "organizations_merged",
        {
            "source_organization_ids": ["organization-acme-labs"],
            "target_organization_id": "organization-acme",
        },
        9,
    )

    first = ledger.rebuild()
    snapshot = ledger.projection_snapshot()
    assert snapshot["organizations"]["organization-acme"]["aliases_json"] == '["Acme"]'
    assert snapshot["organizations"]["organization-acme-labs"]["merged_into_organization_id"] == "organization-acme"
    assert snapshot["organization_sources"]["organization-source-labs"]["organization_id"] == "organization-acme"
    assert snapshot["roles"]["role-morgan-labs"]["organization_id"] == "organization-acme"
    assert snapshot["activities"]["activity-labs-calendar"]["subject_id"] == "organization-acme"
    assert ledger.rebuild().projection_hash == first.projection_hash

    append(
        "event_reversed",
        {"reason": "organizations are distinct"},
        10,
        reverses_event_id=merge_id,
    )
    ledger.rebuild()
    restored = ledger.projection_snapshot()
    assert restored["organizations"]["organization-acme-labs"]["merged_into_organization_id"] == ""
    assert restored["organization_sources"]["organization-source-labs"]["organization_id"] == "organization-acme-labs"

    split_id = append(
        "organization_split",
        {
            "source_organization_id": "organization-acme-labs",
            "target_organization_id": "organization-independent",
            "source_record_ids": ["organization-source-labs"],
            "role_ids": ["role-morgan-labs"],
            "activity_ids": ["activity-labs-calendar"],
        },
        11,
    )
    ledger.rebuild()
    split = ledger.projection_snapshot()
    assert split["organization_sources"]["organization-source-labs"]["organization_id"] == "organization-independent"
    assert split["roles"]["role-morgan-labs"]["organization_id"] == "organization-independent"
    assert split["activities"]["activity-labs-calendar"]["subject_id"] == "organization-independent"

    append(
        "event_reversed",
        {"reason": "split was incorrect"},
        12,
        reverses_event_id=split_id,
    )
    ledger.rebuild()
    restored_again = ledger.projection_snapshot()
    assert restored_again["organization_sources"]["organization-source-labs"]["organization_id"] == "organization-acme-labs"


def test_accepted_activity_is_eligible_for_scoped_as_of_context_without_self_corroboration(
    tmp_path: Path,
) -> None:
    conversation_knowledge_store.ConversationKnowledgeStore(tmp_path).migrate(
        backup=False
    )
    ledger = IdentityLearningLedger(tmp_path)
    person_id = "00000000-0000-4000-8000-000000000763"
    ledger.append_event(
        event_type="person_created",
        payload={"person_id": person_id, "primary_name": "Taylor Example", "status": "reviewed"},
        actor_id="reviewer:test",
        occurred_at="2026-09-01T14:00:00Z",
        idempotency_key="plan76-context-person",
    )
    for suffix, occurred_at, evidence_status, originating_conversation in (
        ("prior", "2026-08-01T12:00:00Z", "accepted", "conversation-prior"),
        ("current", "2026-08-02T12:00:00Z", "accepted", "conversation-current"),
        ("proposed", "2026-08-03T12:00:00Z", "proposed", "conversation-prior"),
        ("future", "2026-10-01T12:00:00Z", "accepted", "conversation-future"),
    ):
        ledger.append_event(
            event_type="activity_observed",
            payload={
                "observation_id": f"activity-{suffix}",
                "subject_type": "person",
                "subject_id": person_id,
                "channel": "email",
                "occurred_at": occurred_at,
                "source_event_at": occurred_at,
                "retrieved_at": "2026-09-01T14:01:00Z",
                "as_of": "2026-09-01T14:01:00Z",
                "participation_status": "observed",
                "evidence_status": evidence_status,
                "source_profile_id": "mail-default",
                "account_id": "account",
                "tenant_id": "tenant",
                "source_record_id": f"mail-{suffix}",
                "independence_group_id": f"logical-{suffix}",
                "content_hash": f"activity-{suffix}-hash",
                "source_locator": {"evidence_id": f"evidence-{suffix}"},
                "metadata": {
                    "accepted_at": "2026-08-05T12:00:00Z",
                    "originating_conversation_id": originating_conversation,
                },
            },
            actor_id="reviewer:test",
            occurred_at="2026-09-01T14:01:00Z",
            idempotency_key=f"plan76-context-{suffix}",
        )
    ledger.rebuild()

    bundle = EvidenceFabric(tmp_path).collect(
        EvidenceRequest(
            purpose="conversation_understanding",
            conversation_id="conversation-current",
            anchors=(EvidenceAnchor("person", person_id),),
            query_terms=(),
            scopes=(EvidenceScope("mail-default", "account", "tenant"),),
            capabilities=("accepted_activity_history",),
            as_of="2026-09-01T00:00:00Z",
            hindsight_policy="allow_hindsight",
            allowed_freshness_states=("current",),
            max_records=10,
            max_characters=10000,
            max_provider_calls=0,
            max_relationship_hops=0,
        )
    )

    assert [activity.observation_id for activity in bundle.activities] == [
        "activity-prior"
    ]
    assert "current_conversation_activity_excluded" in bundle.warnings
    assert "activity_after_as_of_excluded" in bundle.warnings


def test_accepted_roles_are_anchor_scoped_and_effective_at_context_time(
    tmp_path: Path,
) -> None:
    conversation_knowledge_store.ConversationKnowledgeStore(tmp_path).migrate(
        backup=False
    )
    ledger = IdentityLearningLedger(tmp_path)
    person_id = "00000000-0000-4000-8000-000000000772"
    ledger.append_event(
        event_type="person_created",
        payload={"person_id": person_id, "primary_name": "Casey Example", "status": "reviewed"},
        actor_id="reviewer:test",
        occurred_at="2026-09-01T15:00:00Z",
        idempotency_key="plan77-context-person",
    )
    ledger.append_event(
        event_type="organization_created",
        payload={
            "organization_id": "organization-acme",
            "primary_name": "Acme Research",
            "status": "reviewed",
        },
        actor_id="reviewer:test",
        occurred_at="2026-09-01T15:01:00Z",
        idempotency_key="plan77-context-organization",
    )
    for suffix, starts_at, ends_at, status, origin in (
        ("current", "2025-01-01T00:00:00Z", "", "reviewed", "conversation-prior"),
        ("self", "2025-01-01T00:00:00Z", "", "accepted", "conversation-current"),
        ("proposed", "2025-01-01T00:00:00Z", "", "proposed", "conversation-prior"),
        ("future", "2027-01-01T00:00:00Z", "", "accepted", "conversation-prior"),
        ("ended", "2024-01-01T00:00:00Z", "2026-01-01T00:00:00Z", "accepted", "conversation-prior"),
    ):
        ledger.append_event(
            event_type="role_asserted",
            payload={
                "role_id": f"role-{suffix}",
                "person_id": person_id,
                "organization_id": "organization-acme",
                "role_type": suffix,
                "starts_at": starts_at,
                "ends_at": ends_at,
                "status": status,
                "evidence_ids": [f"evidence-{suffix}"],
                "metadata": {
                    "accepted_at": "2026-02-01T00:00:00Z",
                    "originating_conversation_id": origin,
                },
            },
            actor_id="reviewer:test",
            occurred_at="2026-09-01T15:02:00Z",
            idempotency_key=f"plan77-context-role-{suffix}",
        )
    ledger.rebuild()

    bundle = EvidenceFabric(tmp_path).collect(
        EvidenceRequest(
            purpose="conversation_understanding",
            conversation_id="conversation-current",
            anchors=(EvidenceAnchor("person", person_id),),
            query_terms=(),
            scopes=(EvidenceScope("local-knowledge", "", ""),),
            capabilities=("accepted_role_appointments",),
            as_of="2026-09-01T00:00:00Z",
            hindsight_policy="exclude",
            allowed_freshness_states=("current",),
            max_records=10,
            max_characters=10000,
            max_provider_calls=0,
            max_relationship_hops=0,
        )
    )

    assert [role.role_id for role in bundle.role_appointments] == ["role-current"]
    assert bundle.role_appointments[0].organization_id == "organization-acme"
    assert "current_conversation_role_excluded" in bundle.warnings
    assert "role_after_as_of_excluded" in bundle.warnings
    assert "role_outside_effective_time_excluded" in bundle.warnings
