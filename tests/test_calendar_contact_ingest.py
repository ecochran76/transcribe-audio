from __future__ import annotations

import json
from pathlib import Path

import pytest

import calendar_contact_ingest
import transcript_store


def _insert_transcript(root: Path, *, doc_id: str, title: str, attendee: object) -> None:
    payload = {
        "recorded_at": "2026-08-20T14:30:00Z",
        "original_recording_filename": f"{title}.m4a",
        "event": {
            "summary": f"{title} calendar event",
            "attendees": [attendee],
        },
    }
    with transcript_store.connect(root) as con:
        transcript_store.init_db(con)
        con.execute(
            """
            INSERT INTO documents (
              id, kind, title, source_path, stored_path, artifact_sha256,
              generated_at, text_content, json_payload, metadata_json,
              embedding_json, embedding_provider, embedding_model, created_at, updated_at
            ) VALUES (?, 'transcript', ?, ?, ?, ?, ?, '', ?, '{}', '[]',
                      'debug-hash', 'debug-hash', ?, ?)
            """,
            (
                doc_id,
                title,
                f"/private/{title}.transcript.json",
                f"/private/store/{title}.transcript.json",
                doc_id.rjust(64, "a")[-64:],
                "2026-08-20T14:30:00Z",
                json.dumps(payload),
                "2026-08-20T14:30:00Z",
                "2026-08-20T14:30:00Z",
            ),
        )
        con.commit()


def test_collect_and_apply_calendar_contacts_is_exact_and_idempotent(tmp_path: Path) -> None:
    root = tmp_path / "store"
    state_root = tmp_path / "state"
    _insert_transcript(root, doc_id="doc-1", title="Planning", attendee="Alex Example <alex@example.test>")
    _insert_transcript(root, doc_id="doc-2", title="Follow up", attendee={"email": "ALEX@example.test", "displayName": "Alex Example"})

    enrichment = calendar_contact_ingest.EnrichmentResult(
        matches={
            "alex@example.test": [
                {
                    "provider": "gws",
                    "profile": "default",
                    "record_type": "gws_contact",
                    "source_record_id": "people/alex",
                    "label": "Alex Example",
                    "organizations": ["Example Labs"],
                    "roles": [
                        {
                            "title": "Research Director",
                            "organization": "Example Labs",
                            "department": "Research",
                            "current": True,
                        }
                    ],
                    "phones": ["+1 555 0100"],
                    "match_basis": "exact_email",
                }
            ]
        },
        read_calls=1,
        read_records=12,
        warnings=[],
    )
    plan = calendar_contact_ingest.build_ingest_plan(
        root,
        state_root=state_root,
        enrichment=enrichment,
    )

    assert plan["unique_attendee_email_count"] == 1
    assert plan["attendee_appearance_count"] == 2
    assert plan["counts"]["inserted"] == 1
    assert plan["provider_write_count"] == 0
    with pytest.raises(ValueError, match="Apply requires approval token"):
        calendar_contact_ingest.apply_ingest_plan(
            plan, root=root, state_root=state_root, approval_token="wrong"
        )

    receipt = calendar_contact_ingest.apply_ingest_plan(
        plan,
        root=root,
        state_root=state_root,
        approval_token=calendar_contact_ingest.APPLY_TOKEN,
    )
    assert Path(receipt["receipt_path"]).stat().st_mode & 0o777 == 0o600
    with transcript_store.connect(root) as con:
        row = con.execute("SELECT * FROM contacts").fetchone()
    assert row["email"] == "alex@example.test"
    metadata = json.loads(row["metadata_json"])
    assert metadata["calendar_attendee"]["recording_count"] == 2
    assert metadata["enrichment"]["organizations"] == ["Example Labs"]
    assert metadata["enrichment"]["roles"][0]["title"] == "Research Director"
    assert metadata["identity_boundary"] == "exact_email_source_join_not_person_or_speaker_proof"

    replay = calendar_contact_ingest.build_ingest_plan(
        root,
        state_root=state_root,
        enrichment=enrichment,
    )
    assert replay["counts"]["unchanged"] == 1
    assert replay["counts"].get("inserted", 0) == 0
    assert replay["counts"].get("enriched", 0) == 0


def test_role_address_stays_review_required_and_receipt_can_undo(tmp_path: Path) -> None:
    root = tmp_path / "store"
    state_root = tmp_path / "state"
    _insert_transcript(root, doc_id="doc-role", title="Support call", attendee="support@example.test")
    plan = calendar_contact_ingest.build_ingest_plan(root, state_root=state_root, enrich=False)
    metadata = json.loads(plan["operations"][0]["after"]["metadata_json"])
    assert metadata["contact_class"] == "shared_or_role_address"
    assert metadata["resolution_status"] == "review_required"

    receipt = calendar_contact_ingest.apply_ingest_plan(
        plan,
        root=root,
        state_root=state_root,
        approval_token=calendar_contact_ingest.APPLY_TOKEN,
    )
    undo = calendar_contact_ingest.undo_receipt(
        Path(receipt["receipt_path"]),
        root=root,
        state_root=state_root,
        approval_token=calendar_contact_ingest.UNDO_TOKEN,
    )
    assert undo["deleted_insert_count"] == 1
    with transcript_store.connect(root) as con:
        assert con.execute("SELECT COUNT(*) FROM contacts").fetchone()[0] == 0


def test_existing_contact_is_enriched_without_changing_manual_label(tmp_path: Path) -> None:
    root = tmp_path / "store"
    _insert_transcript(root, doc_id="doc-existing", title="Existing", attendee="Alex Example <alex@example.test>")
    with transcript_store.connect(root) as con:
        con.execute(
            """
            INSERT INTO contacts (id, label, email, external_ref, metadata_json, created_at, updated_at)
            VALUES ('manual-1', 'Alex E.', 'alex@example.test', 'manual-ref',
                    '{"source":"operator"}', '2026-08-01T00:00:00Z', '2026-08-01T00:00:00Z')
            """
        )
        con.commit()

    plan = calendar_contact_ingest.build_ingest_plan(root, state_root=tmp_path / "state", enrich=False)
    operation = plan["operations"][0]
    assert operation["action"] == "update"
    assert operation["after"]["label"] == "Alex E."
    assert operation["after"]["external_ref"] == "manual-ref"
    assert json.loads(operation["after"]["metadata_json"])["source"] == "calendar_attendee"


def test_apply_rejects_stale_existing_contact(tmp_path: Path) -> None:
    root = tmp_path / "store"
    state_root = tmp_path / "state"
    _insert_transcript(root, doc_id="doc-stale", title="Stale", attendee="Alex Example <alex@example.test>")
    with transcript_store.connect(root) as con:
        con.execute(
            """
            INSERT INTO contacts (id, label, email, external_ref, metadata_json, created_at, updated_at)
            VALUES ('manual-1', 'Alex', 'alex@example.test', '', '{}', '2026-08-01', '2026-08-01')
            """
        )
        con.commit()
    plan = calendar_contact_ingest.build_ingest_plan(root, state_root=state_root, enrich=False)
    with transcript_store.connect(root) as con:
        con.execute("UPDATE contacts SET label = 'Reviewed Alex' WHERE id = 'manual-1'")
        con.commit()

    with pytest.raises(ValueError, match="became stale before update"):
        calendar_contact_ingest.apply_ingest_plan(
            plan,
            root=root,
            state_root=state_root,
            approval_token=calendar_contact_ingest.APPLY_TOKEN,
        )
    failed = list((state_root / "contact-ingest").glob("*.json"))
    assert len(failed) == 1
    assert json.loads(failed[0].read_text())["mode"] == "apply_failed"


def test_gws_bulk_collection_joins_only_exact_email() -> None:
    calls: list[list[str]] = []

    def runner(command: list[str], *, config: object) -> dict[str, object]:
        calls.append(command)
        return {
            "connections": [
                {
                    "resourceName": "people/alex",
                    "names": [{"displayName": "Alex Example"}],
                    "emailAddresses": [{"value": "alex@example.test"}],
                    "organizations": [
                        {
                            "name": "Example Labs",
                            "title": "Research Director",
                            "department": "Research",
                            "current": True,
                        }
                    ],
                },
                {
                    "resourceName": "people/not-alex",
                    "names": [{"displayName": "Alex Similar"}],
                    "emailAddresses": [{"value": "alex.similar@example.test"}],
                },
            ]
        }

    result = calendar_contact_ingest.collect_gws_matches(
        {"alex@example.test"},
        config={"profiles": [{"label": "default", "surfaces": ["contacts"]}]},
        runner=runner,
    )

    assert list(result.matches) == ["alex@example.test"]
    assert result.matches["alex@example.test"][0]["source_record_id"] == "people/alex"
    assert result.matches["alex@example.test"][0]["roles"] == [
        {
            "title": "Research Director",
            "organization": "Example Labs",
            "department": "Research",
            "current": True,
        }
    ]
    assert result.read_calls == 1
    assert result.read_records == 2
    assert len(calls) == 1
