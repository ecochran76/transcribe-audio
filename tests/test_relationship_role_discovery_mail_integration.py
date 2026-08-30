from __future__ import annotations

import json
from pathlib import Path

import relationship_role_discovery
import transcript_store
from mail_evidence_normalization import NormalizedMailEvidence


def insert_contact(
    root: Path,
    *,
    contact_id: str,
    label: str,
    email: str,
) -> None:
    metadata = json.dumps(
        {
            "contact_class": "person_candidate",
            "calendar_attendee": {"appearances": []},
            "enrichment": {"source_records": []},
        },
        sort_keys=True,
    )
    with transcript_store.connect(root) as con:
        transcript_store.init_db(con)
        con.execute(
            """
            INSERT INTO contacts (
              id, label, email, external_ref, metadata_json, created_at, updated_at
            ) VALUES (?, ?, ?, '', ?, '2026-01-01T00:00:00Z', '2026-01-07T00:00:00Z')
            """,
            (contact_id, label, email, metadata),
        )
        con.commit()


def test_discover_relationship_roles_merges_injected_mail_without_effects(
    tmp_path: Path,
) -> None:
    root = tmp_path / "store"
    insert_contact(
        root,
        contact_id="contact-account",
        label="Account Contact",
        email="account@example.test",
    )
    insert_contact(
        root,
        contact_id="contact-alex",
        label="Alex Example",
        email="alex@example.test",
    )
    observation = {
        "schema_version": "transcribe-audio.mail-observation.v1",
        "observation_id": "mail-observation-1",
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
            "evidence_id": "evidence-redacted-1",
            "record_ref": "record-redacted-1",
            "message_ref_hash": "b" * 64,
            "thread_ref_hash": "c" * 64,
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
            "alex@example.test": "contact-alex",
            "account@example.test": "contact-account",
        },
        "signature_observations": [],
        "independence_group_id": "mail-interaction-1",
        "redaction": {"body_retained": False},
        "truncation": {"snippet_characters": 0},
        "excluded_reason_code": None,
    }
    group = {
        "schema_version": "transcribe-audio.mail-independence-group.v1",
        "group_id": "mail-interaction-1",
        "interaction_key_version": "mail-interaction-key.v1",
        "independent_thread_key": "c" * 64,
        "member_observation_ids": ["mail-observation-1"],
        "duplicate_count": 0,
        "source_count": 1,
        "reason_code": None,
        "content_hash": "d" * 64,
    }
    evidence = NormalizedMailEvidence(
        observations=(observation,),
        independence_groups=(group,),
        input_watermark="mail-watermark-redacted-1",
    )

    result = relationship_role_discovery.discover_relationship_roles(
        root,
        mail_evidence=evidence,
        mail_account_address="account@example.test",
    )

    assert result["mail_hypothesis_count"] == 1
    assert result["mail_hypothesis_counts"] == {"sent_mail": 1}
    assert result["accepted_effect_count"] == 0
    assert result["provider_write_count"] == 0
    assert result["person_merge_count"] == 0
    assert result["speaker_assignment_apply_count"] == 0
    alex = result["by_contact_id"]["contact-alex"]
    account = result["by_contact_id"]["contact-account"]
    assert alex["relationship_hypotheses"][0]["mail_direction"] == "sent"
    assert account["relationship_hypotheses"][0]["mail_direction"] == "received"
