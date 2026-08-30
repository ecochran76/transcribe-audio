from __future__ import annotations

import json
from pathlib import Path

import relationship_role_discovery
import transcript_store


def _contact_metadata(
    *,
    appearances: list[dict[str, str]],
    organization: str = "",
    role: str = "",
    contact_class: str = "person_candidate",
) -> str:
    source = {
        "provider": "gws",
        "profile": "fixture",
        "record_type": "gws_contact",
        "source_record_id": f"people/{role or organization or contact_class}",
        "label": "Fixture Contact",
        "organizations": [organization] if organization else [],
        "roles": [
            {
                "title": role,
                "organization": organization,
                "department": "Research",
                "current": True,
            }
        ]
        if role
        else [],
        "phones": [],
        "match_basis": "exact_email",
    }
    return json.dumps(
        {
            "contact_class": contact_class,
            "calendar_attendee": {"appearances": appearances},
            "enrichment": {"source_records": [source]},
        },
        sort_keys=True,
    )


def _appearance(document_id: str, date: str) -> dict[str, str]:
    return {
        "document_id": document_id,
        "recording_filename": f"{document_id}.m4a",
        "recorded_at": date,
        "event_summary": f"Event {document_id}",
    }


def _insert_contact(
    root: Path,
    *,
    contact_id: str,
    label: str,
    email: str,
    metadata_json: str,
) -> None:
    with transcript_store.connect(root) as con:
        transcript_store.init_db(con)
        con.execute(
            """
            INSERT INTO contacts (
              id, label, email, external_ref, metadata_json, created_at, updated_at
            ) VALUES (?, ?, ?, '', ?, '2026-08-20T00:00:00Z', '2026-08-29T00:00:00Z')
            """,
            (contact_id, label, email, metadata_json),
        )
        con.commit()


def test_discovery_builds_role_affiliation_and_symmetric_recurring_invitation_hypotheses(
    tmp_path: Path,
) -> None:
    root = tmp_path / "store"
    shared = [
        _appearance("doc-1", "2026-07-01T12:00:00Z"),
        _appearance("doc-2", "2026-08-01T12:00:00Z"),
    ]
    _insert_contact(
        root,
        contact_id="contact-a",
        label="Alex Example",
        email="alex@example.test",
        metadata_json=_contact_metadata(
            appearances=shared,
            organization="Example Labs",
            role="Research Director",
        ),
    )
    _insert_contact(
        root,
        contact_id="contact-b",
        label="Blair Example",
        email="blair@example.test",
        metadata_json=_contact_metadata(
            appearances=shared,
            organization="Partner Lab",
        ),
    )
    _insert_contact(
        root,
        contact_id="contact-c",
        label="Shared Inbox",
        email="team@example.test",
        metadata_json=_contact_metadata(
            appearances=shared,
            organization="Example Labs",
            role="Operations",
            contact_class="shared_or_role_address",
        ),
    )

    first = relationship_role_discovery.discover_relationship_roles(root)
    second = relationship_role_discovery.discover_relationship_roles(root)

    assert first == second
    assert first["authority_mode"] == "shadow_hypotheses_only"
    assert first["role_hypothesis_count"] == 1
    assert first["affiliation_hypothesis_count"] == 2
    assert first["calendar_co_invitation_hypothesis_count"] == 1
    assert first["excluded_shared_address_count"] == 1
    assert first["accepted_effect_count"] == 0
    alex = first["by_contact_id"]["contact-a"]
    blair = first["by_contact_id"]["contact-b"]
    assert alex["role_hypotheses"][0]["display_value"] == "Research Director"
    assert alex["relationship_hypotheses"][0]["relationship_type"] == "AFFILIATED_WITH"
    alex_pair = next(
        item
        for item in alex["relationship_hypotheses"]
        if item["hypothesis_kind"] == "calendar_co_invitation"
    )
    blair_pair = next(
        item
        for item in blair["relationship_hypotheses"]
        if item["hypothesis_kind"] == "calendar_co_invitation"
    )
    assert alex_pair["hypothesis_id"] == blair_pair["hypothesis_id"]
    assert alex_pair["counterpart_label"] == "Blair Example"
    assert blair_pair["counterpart_label"] == "Alex Example"
    assert alex_pair["observation_count"] == 2
    assert "does not prove presence" in alex_pair["why_not_accepted"]
    assert first["by_contact_id"]["contact-c"] == {
        "role_hypotheses": [],
        "relationship_hypotheses": [],
    }


def test_discovery_rejects_single_invitation_as_recurring_relationship(
    tmp_path: Path,
) -> None:
    root = tmp_path / "store"
    appearance = [_appearance("doc-1", "2026-08-01T12:00:00Z")]
    for contact_id, label in (("contact-a", "Alex"), ("contact-b", "Blair")):
        _insert_contact(
            root,
            contact_id=contact_id,
            label=label,
            email=f"{label.lower()}@example.test",
            metadata_json=_contact_metadata(appearances=appearance),
        )

    result = relationship_role_discovery.discover_relationship_roles(root)

    assert result["calendar_co_invitation_hypothesis_count"] == 0
    assert result["contacts_with_candidates"] == 0
