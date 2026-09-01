from __future__ import annotations

import json
import sqlite3
import sys
from dataclasses import replace
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import conversation_knowledge_store
from identity_learning_ledger import (
    BaselineSourceRecord,
    IdentityLearningLedger,
    IdentityOntologyTerm,
    _stable_id,
)


def _ledger(tmp_path: Path) -> IdentityLearningLedger:
    store = conversation_knowledge_store.ConversationKnowledgeStore(tmp_path)
    store.migrate(backup=False)
    return IdentityLearningLedger(tmp_path)


def _append(
    ledger: IdentityLearningLedger,
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
        occurred_at=f"2026-08-16T12:{ordinal:02d}:00Z",
        idempotency_key=f"test-event-{ordinal}",
        reverses_event_id=reverses_event_id,
    ).event_id


def test_append_events_is_atomic_when_a_later_event_conflicts(
    tmp_path: Path,
) -> None:
    ledger = _ledger(tmp_path)
    common = {
        "actor_id": "reviewer:test",
        "occurred_at": "2026-09-01T12:00:00Z",
    }
    ledger.append_event(
        event_type="person_created",
        payload={
            "person_id": "person-existing",
            "primary_name": "Existing Person",
            "status": "reviewed",
        },
        idempotency_key="existing-key",
        **common,
    )

    with pytest.raises(ValueError, match="reused with different content"):
        ledger.append_events(
            (
                {
                    "event_type": "organization_created",
                    "payload": {
                        "organization_id": "organization-new",
                        "primary_name": "New Organization",
                        "status": "reviewed",
                    },
                    "idempotency_key": "new-key",
                    **common,
                },
                {
                    "event_type": "person_created",
                    "payload": {
                        "person_id": "person-different",
                        "primary_name": "Different Person",
                        "status": "reviewed",
                    },
                    "idempotency_key": "existing-key",
                    **common,
                },
            )
        )

    assert [event["idempotency_key"] for event in ledger.events()] == [
        "existing-key"
    ]


def test_append_and_rebuild_preserves_same_timestamp_dependency_order(
    tmp_path: Path,
) -> None:
    ledger = _ledger(tmp_path)
    occurred_at = "2026-09-01T12:00:00Z"
    person_key = "same-time-person"
    source_key = next(
        f"same-time-source-{ordinal}"
        for ordinal in range(100)
        if _stable_id("identity-event", f"same-time-source-{ordinal}")
        < _stable_id("identity-event", person_key)
    )

    ledger.append_events(
        (
            {
                "event_type": "person_created",
                "payload": {
                    "person_id": "person-same-time",
                    "primary_name": "Same Time Person",
                    "status": "reviewed",
                },
                "actor_id": "reviewer:test",
                "occurred_at": occurred_at,
                "idempotency_key": person_key,
            },
            {
                "event_type": "source_record_observed",
                "payload": {
                    "source_record_id": "source-same-time",
                    "person_id": "person-same-time",
                    "source_profile_id": "fixture",
                    "provider_kind": "local",
                    "record_type": "contact",
                    "external_ref": "local:source-same-time",
                    "label": "Same Time Person",
                    "observed_at": occurred_at,
                    "content_hash": "same-time-content",
                },
                "actor_id": "reviewer:test",
                "occurred_at": occurred_at,
                "idempotency_key": source_key,
            },
        ),
        rebuild=True,
    )

    snapshot = ledger.projection_snapshot()
    assert snapshot["people"]["person-same-time"]["primary_name"] == (
        "Same Time Person"
    )
    assert snapshot["sources"]["source-same-time"]["person_id"] == (
        "person-same-time"
    )


def test_append_and_rebuild_rolls_back_raw_events_when_projection_fails(
    tmp_path: Path,
) -> None:
    ledger = _ledger(tmp_path)

    with pytest.raises(ValueError, match="unknown person"):
        ledger.append_events(
            (
                {
                    "event_type": "source_record_observed",
                    "payload": {
                        "source_record_id": "source-orphan",
                        "person_id": "person-missing",
                        "source_profile_id": "fixture",
                        "provider_kind": "local",
                        "record_type": "contact",
                        "external_ref": "local:source-orphan",
                        "label": "Orphan Source",
                        "observed_at": "2026-09-01T12:00:00Z",
                        "content_hash": "orphan-content",
                    },
                    "actor_id": "reviewer:test",
                    "occurred_at": "2026-09-01T12:00:00Z",
                    "idempotency_key": "orphan-source",
                },
            ),
            rebuild=True,
        )

    assert ledger.events() == ()
    assert ledger.projection_snapshot()["sources"] == {}


def test_rebuild_coalesces_equivalent_deterministic_organization_creations(
    tmp_path: Path,
) -> None:
    ledger = _ledger(tmp_path)
    events = []
    for ordinal, hypothesis_id in enumerate(("hypothesis-a", "hypothesis-b")):
        events.append(
            {
                "event_type": "organization_created",
                "payload": {
                    "organization_id": "organization:shared",
                    "primary_name": "Shared Organization",
                    "status": "reviewed",
                    "metadata": {"source_hypothesis_id": hypothesis_id},
                },
                "actor_id": "reviewer:test",
                "occurred_at": f"2026-09-01T12:0{ordinal}:00Z",
                "idempotency_key": f"shared-organization-{ordinal}",
            }
        )

    ledger.append_events(tuple(events), rebuild=True)

    organization = ledger.projection_snapshot()["organizations"][
        "organization:shared"
    ]
    metadata = json.loads(organization["metadata_json"])
    assert metadata["source_hypothesis_ids"] == ["hypothesis-a", "hypothesis-b"]


def test_rebuild_rejects_conflicting_deterministic_organization_creations(
    tmp_path: Path,
) -> None:
    ledger = _ledger(tmp_path)
    common = {
        "actor_id": "reviewer:test",
        "occurred_at": "2026-09-01T12:00:00Z",
    }

    with pytest.raises(ValueError, match="conflicting definitions"):
        ledger.append_events(
            (
                {
                    "event_type": "organization_created",
                    "payload": {
                        "organization_id": "organization:shared",
                        "primary_name": "Shared Organization",
                        "status": "reviewed",
                    },
                    "idempotency_key": "shared-organization-a",
                    **common,
                },
                {
                    "event_type": "organization_created",
                    "payload": {
                        "organization_id": "organization:shared",
                        "primary_name": "Different Organization",
                        "status": "reviewed",
                    },
                    "idempotency_key": "shared-organization-b",
                    **common,
                },
            ),
            rebuild=True,
        )

    assert ledger.events() == ()


def test_append_only_events_rebuild_merge_split_and_reversal_deterministically(
    tmp_path: Path,
) -> None:
    ledger = _ledger(tmp_path)
    person_a = "00000000-0000-4000-8000-000000000101"
    person_b = "00000000-0000-4000-8000-000000000102"
    person_c = "00000000-0000-4000-8000-000000000103"
    for ordinal, person_id, name in (
        (1, person_a, "Alex North"),
        (2, person_b, "A. North"),
        (3, person_c, "Alex North Consulting"),
    ):
        _append(
            ledger,
            "person_created",
            {
                "person_id": person_id,
                "primary_name": name,
                "status": "reviewed",
            },
            ordinal,
        )
    for ordinal, source_record_id, person_id, external_ref in (
        (4, "source-gws-1", person_a, "people/1"),
        (5, "source-crm-1", person_b, "res.partner:1"),
    ):
        _append(
            ledger,
            "source_record_observed",
            {
                "source_record_id": source_record_id,
                "person_id": person_id,
                "source_profile_id": "profile-1",
                "provider_kind": "gws" if ordinal == 4 else "odollo",
                "account_id": "account-1",
                "tenant_id": "tenant-1",
                "record_type": "contact",
                "external_ref": external_ref,
                "label": "Alex North",
                "source_event_at": "2026-08-15T12:00:00Z",
                "observed_at": "2026-08-16T12:00:00Z",
                "content_hash": f"content-{ordinal}",
            },
            ordinal,
        )
    _append(
        ledger,
        "role_asserted",
        {
            "role_id": "role-1",
            "person_id": person_a,
            "role_type": "advisor",
            "organization_id": "organization-1",
            "status": "reviewed",
            "evidence_ids": ["evidence-1"],
        },
        6,
    )
    _append(
        ledger,
        "role_asserted",
        {
            "role_id": "role-2",
            "person_id": person_a,
            "role_type": "client",
            "matter_id": "matter-1",
            "status": "reviewed",
            "evidence_ids": ["evidence-2"],
        },
        7,
    )
    _append(
        ledger,
        "relationship_asserted",
        {
            "relationship_id": "relationship-1",
            "relationship_type": "advises",
            "subject_type": "person",
            "subject_id": person_a,
            "object_type": "organization",
            "object_id": "organization-1",
            "directionality": "directional",
            "inverse_relationship_id": "relationship-2",
            "status": "reviewed",
            "evidence_ids": ["evidence-1"],
        },
        8,
    )
    merge_id = _append(
        ledger,
        "people_merged",
        {"source_person_ids": [person_b], "target_person_id": person_a},
        9,
    )
    split_id = _append(
        ledger,
        "person_split",
        {
            "source_person_id": person_a,
            "target_person_id": person_c,
            "source_record_ids": ["source-crm-1"],
        },
        10,
    )

    first = ledger.rebuild()
    first_snapshot = ledger.projection_snapshot()

    assert first.event_count == 10
    assert first_snapshot["people"][person_b]["merged_into_person_id"] == person_a
    assert first_snapshot["sources"]["source-gws-1"]["person_id"] == person_a
    assert first_snapshot["sources"]["source-crm-1"]["person_id"] == person_c
    assert len(first_snapshot["roles"]) == 2
    assert first_snapshot["relationships"]["relationship-1"]["directionality"] == "directional"

    second = ledger.rebuild()
    assert second.projection_hash == first.projection_hash
    assert ledger.projection_snapshot() == first_snapshot

    _append(
        ledger,
        "event_reversed",
        {"reason": "split selected the wrong source record"},
        11,
        reverses_event_id=split_id,
    )
    ledger.rebuild()
    assert ledger.projection_snapshot()["sources"]["source-crm-1"]["person_id"] == person_a

    _append(
        ledger,
        "event_reversed",
        {"reason": "merge was not supported"},
        12,
        reverses_event_id=merge_id,
    )
    ledger.rebuild()
    snapshot = ledger.projection_snapshot()
    assert snapshot["people"][person_b]["merged_into_person_id"] == ""
    assert snapshot["sources"]["source-crm-1"]["person_id"] == person_b

    with sqlite3.connect(tmp_path / "transcripts.sqlite3") as con:
        with pytest.raises(sqlite3.IntegrityError, match="append-only"):
            con.execute(
                "UPDATE knowledge_identity_ledger_events SET actor_id = 'tampered'"
            )


def test_ontology_supports_hierarchy_direction_and_inverse_terms(
    tmp_path: Path,
) -> None:
    ledger = _ledger(tmp_path)

    receipt = ledger.register_ontology(
        schema_name="identity-contact-ontology",
        version="1",
        terms=(
            IdentityOntologyTerm("role", "professional"),
            IdentityOntologyTerm("role", "advisor", parent_term_key="professional"),
            IdentityOntologyTerm(
                "relationship",
                "advises",
                directionality="directional",
                inverse_term_key="advised_by",
            ),
            IdentityOntologyTerm(
                "relationship",
                "advised_by",
                directionality="directional",
                inverse_term_key="advises",
            ),
            IdentityOntologyTerm(
                "relationship",
                "colleague_of",
                directionality="symmetric",
            ),
        ),
    )

    assert receipt.term_count == 5
    assert (
        ledger.ontology_terms(receipt.ontology_version_id)[1]["parent_term_key"]
        == "professional"
    )
    with pytest.raises(ValueError, match="unknown parent"):
        ledger.register_ontology(
            schema_name="invalid-ontology",
            version="1",
            terms=(IdentityOntologyTerm("role", "advisor", parent_term_key="missing"),),
        )


def test_external_identity_and_corrections_replay_without_raw_identifier_leakage(
    tmp_path: Path,
) -> None:
    ledger = _ledger(tmp_path)
    person_id = "00000000-0000-4000-8000-000000000121"
    _append(
        ledger,
        "person_created",
        {"person_id": person_id, "primary_name": "Morgan Example", "status": "reviewed"},
        1,
    )
    _append(
        ledger,
        "source_record_observed",
        {
            "source_record_id": "source-1",
            "person_id": person_id,
            "source_profile_id": "gws-personal",
            "provider_kind": "gws",
            "account_id": "personal",
            "tenant_id": "",
            "record_type": "contact",
            "external_ref": "people/1",
            "label": "Morgan Example",
            "observed_at": "2026-08-16T12:00:00Z",
            "content_hash": "source-hash",
        },
        2,
    )
    _append(
        ledger,
        "external_identity_observed",
        {
            "external_identity_id": "external-1",
            "source_record_id": "source-1",
            "person_id": person_id,
            "provider_kind": "gws",
            "account_id": "personal",
            "tenant_id": "",
            "identity_type": "email",
            "identity_value_hash": "a" * 64,
            "person_specific": True,
            "verified": True,
            "shared_identifier": False,
            "observed_at": "2026-08-16T12:00:00Z",
        },
        3,
    )
    _append(
        ledger,
        "role_asserted",
        {
            "role_id": "role-1",
            "person_id": person_id,
            "role_type": "advisor",
            "status": "proposed",
            "evidence_ids": ["evidence-1"],
        },
        4,
    )
    _append(
        ledger,
        "relationship_asserted",
        {
            "relationship_id": "relationship-1",
            "relationship_type": "advises",
            "subject_type": "person",
            "subject_id": person_id,
            "object_type": "organization",
            "object_id": "organization-1",
            "status": "proposed",
        },
        5,
    )
    correction_id = _append(
        ledger,
        "source_record_corrected",
        {"source_record_id": "source-1", "changes": {"label": "Morgan A. Example"}},
        6,
    )
    _append(
        ledger,
        "role_corrected",
        {"role_id": "role-1", "changes": {"status": "reviewed"}},
        7,
    )
    _append(
        ledger,
        "relationship_corrected",
        {
            "relationship_id": "relationship-1",
            "changes": {
                "status": "reviewed_conflict",
                "metadata": {"conflict_refs": ["relationship-2"]},
            },
        },
        8,
    )

    ledger.rebuild()
    snapshot = ledger.projection_snapshot()

    assert snapshot["sources"]["source-1"]["label"] == "Morgan A. Example"
    assert "email" not in snapshot["sources"]["source-1"]
    assert snapshot["external_identities"]["external-1"]["identity_value_hash"] == "a" * 64
    assert snapshot["roles"]["role-1"]["status"] == "reviewed"
    assert snapshot["relationships"]["relationship-1"]["status"] == "reviewed_conflict"
    assert "relationship-2" in snapshot["relationships"]["relationship-1"]["metadata_json"]

    first = ledger.append_event(
        event_type="alias_added",
        payload={"person_id": person_id, "alias": "M. Example"},
        actor_id="reviewer:test",
        occurred_at="2026-08-16T12:09:00Z",
        idempotency_key="alias-idempotency",
    )
    second = ledger.append_event(
        event_type="alias_added",
        payload={"person_id": person_id, "alias": "M. Example"},
        actor_id="reviewer:test",
        occurred_at="2026-08-16T12:09:00Z",
        idempotency_key="alias-idempotency",
    )
    assert first.status == "inserted"
    assert second.status == "unchanged"
    with pytest.raises(ValueError, match="idempotency key"):
        ledger.append_event(
            event_type="alias_added",
            payload={"person_id": person_id, "alias": "Different Alias"},
            actor_id="reviewer:test",
            occurred_at="2026-08-16T12:09:00Z",
            idempotency_key="alias-idempotency",
        )

    _append(
        ledger,
        "event_reversed",
        {"reason": "source correction was inaccurate"},
        10,
        reverses_event_id=correction_id,
    )
    ledger.rebuild()
    assert ledger.projection_snapshot()["sources"]["source-1"]["label"] == "Morgan Example"


def test_role_correction_and_reversal_preserve_sibling_appointments(
    tmp_path: Path,
) -> None:
    ledger = _ledger(tmp_path)
    person_id = "00000000-0000-4000-8000-000000000773"
    _append(
        ledger,
        "person_created",
        {"person_id": person_id, "primary_name": "Casey Example"},
        1,
    )
    _append(
        ledger,
        "organization_created",
        {
            "organization_id": "organization-acme",
            "primary_name": "Acme Research",
        },
        2,
    )
    for ordinal, role_id, role_type in (
        (3, "role-acme-founder", "founder"),
        (4, "role-acme-ceo", "chief_executive_officer"),
    ):
        _append(
            ledger,
            "role_asserted",
            {
                "role_id": role_id,
                "person_id": person_id,
                "organization_id": "organization-acme",
                "role_type": role_type,
                "status": "reviewed",
                "evidence_ids": [f"evidence-{role_id}"],
            },
            ordinal,
        )
    ledger.rebuild()
    original = ledger.projection_snapshot()["roles"]

    def appointment_value(value: dict[str, object]) -> dict[str, object]:
        return {
            key: item
            for key, item in value.items()
            if key not in {"input_watermark", "built_at"}
        }

    correction_id = _append(
        ledger,
        "role_corrected",
        {
            "role_id": "role-acme-ceo",
            "changes": {"role_type": "president", "status": "accepted"},
        },
        5,
    )
    ledger.rebuild()
    corrected = ledger.projection_snapshot()["roles"]

    assert corrected["role-acme-ceo"]["role_type"] == "president"
    assert corrected["role-acme-ceo"]["status"] == "accepted"
    assert appointment_value(corrected["role-acme-founder"]) == appointment_value(
        original["role-acme-founder"]
    )

    _append(
        ledger,
        "event_reversed",
        {"reason": "role correction was inaccurate"},
        6,
        reverses_event_id=correction_id,
    )
    reversed_receipt = ledger.rebuild()
    restored = ledger.projection_snapshot()["roles"]
    replay_receipt = ledger.rebuild()

    assert appointment_value(restored["role-acme-ceo"]) == appointment_value(
        original["role-acme-ceo"]
    )
    assert appointment_value(restored["role-acme-founder"]) == appointment_value(
        original["role-acme-founder"]
    )
    assert replay_receipt.projection_hash == reversed_receipt.projection_hash


def test_baseline_reconciliation_deduplicates_exact_scope_and_fails_closed(
    tmp_path: Path,
) -> None:
    ledger = _ledger(tmp_path)
    person_id = "00000000-0000-4000-8000-000000000111"
    baseline = BaselineSourceRecord(
        source_record_id="source-existing",
        person_id=person_id,
        source_profile_id="gws-personal",
        provider_kind="gws",
        account_id="personal",
        tenant_id="",
        record_type="contact",
        external_ref="people/1",
        label="Taylor Example",
        email="taylor@example.test",
        email_verified=True,
        person_specific=True,
        observed_at="2026-08-16T12:00:00Z",
        content_hash="hash-1",
    )
    duplicate = replace(baseline, source_record_id="source-duplicate")
    exact_match = replace(
        baseline,
        source_record_id="source-new",
        person_id="",
        source_profile_id="odollo-company",
        provider_kind="odollo",
        tenant_id="company",
        external_ref="res.partner:1",
        content_hash="hash-2",
    )
    shared = replace(
        exact_match,
        source_record_id="source-shared",
        external_ref="res.partner:2",
        email="office@example.test",
        shared_identifier=True,
        content_hash="hash-3",
    )
    name_only = replace(
        exact_match,
        source_record_id="source-name-only",
        external_ref="res.partner:3",
        email="",
        email_verified=False,
        content_hash="hash-4",
    )

    plan = ledger.reconcile_baseline((
        baseline,
        duplicate,
        exact_match,
        shared,
        name_only,
    ))

    assert plan.kept_source_record_ids == (
        "source-existing",
        "source-name-only",
        "source-new",
        "source-shared",
    )
    assert plan.duplicate_source_record_ids == ("source-duplicate",)
    assert plan.auto_links == (("source-new", person_id, "verified_email"),)
    assert {item.source_record_id: item.reason_code for item in plan.proposals} == {
        "source-name-only": "insufficient_authoritative_identifier",
        "source-shared": "shared_identifier_requires_review",
    }
    assert plan.provider_write_count == 0


def test_baseline_reconciliation_preserves_conflicting_exact_identifiers(
    tmp_path: Path,
) -> None:
    ledger = _ledger(tmp_path)
    common = {
        "source_profile_id": "profile",
        "provider_kind": "provider",
        "account_id": "account",
        "tenant_id": "tenant",
        "record_type": "contact",
        "label": "Conflicted Contact",
        "email": "conflict@example.test",
        "email_verified": True,
        "person_specific": True,
        "observed_at": "2026-08-16T12:00:00Z",
    }
    plan = ledger.reconcile_baseline(
        (
            BaselineSourceRecord(
                source_record_id="source-a",
                person_id="00000000-0000-4000-8000-000000000131",
                external_ref="contact-a",
                content_hash="hash-a",
                **common,
            ),
            BaselineSourceRecord(
                source_record_id="source-b",
                person_id="00000000-0000-4000-8000-000000000132",
                external_ref="contact-b",
                content_hash="hash-b",
                **common,
            ),
            BaselineSourceRecord(
                source_record_id="source-unlinked",
                person_id="",
                external_ref="contact-c",
                content_hash="hash-c",
                **common,
            ),
        )
    )

    proposal = plan.proposals[0]
    assert proposal.source_record_id == "source-unlinked"
    assert proposal.reason_code == "conflicting_authoritative_identifiers"
    assert proposal.candidate_person_ids == (
        "00000000-0000-4000-8000-000000000131",
        "00000000-0000-4000-8000-000000000132",
    )
    assert plan.auto_links == ()
    assert plan.provider_write_count == 0


def test_source_ledger_rejects_raw_identifiers_and_failed_rebuild_is_atomic(
    tmp_path: Path,
) -> None:
    ledger = _ledger(tmp_path)
    with pytest.raises(ValueError, match="raw email or phone"):
        ledger.append_event(
            event_type="source_record_observed",
            payload={
                "source_record_id": "source-raw",
                "source_profile_id": "profile",
                "provider_kind": "provider",
                "record_type": "contact",
                "external_ref": "contact/1",
                "email": "raw@example.test",
                "observed_at": "2026-08-16T12:00:00Z",
                "content_hash": "source-hash",
            },
            actor_id="reviewer:test",
            occurred_at="2026-08-16T12:00:00Z",
            idempotency_key="raw-source-event",
        )

    person_a = "00000000-0000-4000-8000-000000000141"
    person_b = "00000000-0000-4000-8000-000000000142"
    _append(
        ledger,
        "person_created",
        {"person_id": person_a, "primary_name": "Person A", "status": "reviewed"},
        1,
    )
    _append(
        ledger,
        "person_created",
        {"person_id": person_b, "primary_name": "Person B", "status": "reviewed"},
        2,
    )
    ledger.rebuild()
    before = ledger.projection_snapshot()
    bad_split = _append(
        ledger,
        "person_split",
        {
            "source_person_id": person_a,
            "target_person_id": person_b,
            "source_record_ids": ["unknown-source"],
        },
        3,
    )

    with pytest.raises(ValueError, match="unknown source record"):
        ledger.rebuild()
    assert ledger.projection_snapshot() == before

    _append(
        ledger,
        "event_reversed",
        {"reason": "invalid split was appended but never projected"},
        4,
        reverses_event_id=bad_split,
    )
    ledger.rebuild()
    assert ledger.projection_snapshot()["people"][person_a]["status"] == "reviewed"
