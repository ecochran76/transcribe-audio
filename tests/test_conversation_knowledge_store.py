from __future__ import annotations

import json
import sqlite3
import stat
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import conversation_knowledge_store
import identity_learning_ledger
import transcript_store


def _write_transcript(path: Path) -> Path:
    path.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "conversation_id": "00000000-0000-4000-8000-000000000001",
                "recording_id": "00000000-0000-4000-8000-000000000002",
                "transcript_title": "Migration compatibility conversation",
                "backend": "assembly",
                "recording_start": "2026-07-26T09:00:00-05:00",
                "duration_seconds": 60,
                "transcript_text": "Existing searchable transcript content.",
            }
        ),
        encoding="utf-8",
    )
    return path


def test_core_schema_migration_preserves_existing_store_and_sidecar_authority(
    tmp_path: Path,
) -> None:
    transcript_store.ingest_artifact(
        _write_transcript(tmp_path / "existing.transcript.json"),
        root=tmp_path,
        embedding_provider="debug-hash",
        embedding_model="debug-hash-v1",
    )
    store = conversation_knowledge_store.ConversationKnowledgeStore(tmp_path)

    before = store.schema_status()
    receipt = store.migrate(backup=False)
    after = store.schema_status()
    search_results = transcript_store.search_store(
        "searchable transcript",
        root=tmp_path,
        embedding_provider="debug-hash",
        embedding_model="debug-hash-v1",
    )

    assert before.schema_version == 0
    assert receipt.from_version == 0
    assert receipt.to_version == 5
    assert receipt.applied_versions == (1, 2, 3, 4, 5)
    assert after.schema_version == 5
    assert after.authority_mode == "sidecar"
    assert after.dirty is False
    assert search_results[0]["title"] == "Migration compatibility conversation"


def test_conversation_snapshot_round_trips_through_repository_interface(
    tmp_path: Path,
) -> None:
    store = conversation_knowledge_store.ConversationKnowledgeStore(tmp_path)
    store.migrate(backup=False)
    snapshot = conversation_knowledge_store.ConversationSnapshot(
        conversation=conversation_knowledge_store.ConversationRecord(
            conversation_id="00000000-0000-4000-8000-000000000011",
            title="Architecture review",
            starts_at="2026-07-26T09:00:00-05:00",
            ends_at="2026-07-26T10:00:00-05:00",
            calendar_association_state="proposed",
            metadata={"calendar_event_id": "event-1"},
        ),
        recordings=(
            conversation_knowledge_store.RecordingRecord(
                recording_id="00000000-0000-4000-8000-000000000012",
                conversation_id="00000000-0000-4000-8000-000000000011",
                transcript_document_id="document-1",
                backend="assembly",
                model="universal-3-pro",
                captured_at="2026-07-26T09:00:00-05:00",
            ),
        ),
        utterances=(
            conversation_knowledge_store.UtteranceRecord(
                utterance_id="00000000-0000-4000-8000-000000000013",
                conversation_id="00000000-0000-4000-8000-000000000011",
                recording_id="00000000-0000-4000-8000-000000000012",
                speaker_label="A",
                ordinal=0,
                start_ms=0,
                end_ms=1200,
                text="This is the first stored utterance.",
            ),
        ),
    )

    first = store.save_conversation_snapshot(snapshot)
    second = store.save_conversation_snapshot(snapshot)
    loaded = store.load_conversation_snapshot(
        "00000000-0000-4000-8000-000000000011"
    )

    assert first.status == "inserted"
    assert second.status == "unchanged"
    assert loaded == snapshot


def test_migration_backup_and_rollback_preserve_legacy_store(
    tmp_path: Path,
) -> None:
    transcript_store.ingest_artifact(
        _write_transcript(tmp_path / "rollback.transcript.json"),
        root=tmp_path,
        embedding_provider="debug-hash",
        embedding_model="debug-hash-v1",
    )
    store = conversation_knowledge_store.ConversationKnowledgeStore(tmp_path)

    migration = store.migrate()
    rollback = store.rollback(target_version=0)

    assert migration.backup_path
    assert Path(migration.backup_path).is_file()
    assert stat.S_IMODE(Path(migration.backup_path).parent.stat().st_mode) == 0o700
    assert stat.S_IMODE(Path(migration.backup_path).stat().st_mode) == 0o600
    assert rollback.from_version == 5
    assert rollback.to_version == 0
    assert rollback.rolled_back_versions == (5, 4, 3, 2, 1)
    assert rollback.backup_path
    assert store.schema_status().schema_version == 0
    assert transcript_store.search_store(
        "searchable transcript",
        root=tmp_path,
        embedding_provider="debug-hash",
        embedding_model="debug-hash-v1",
    )


def test_v4_identity_ledger_migration_is_additive_and_reversible(
    tmp_path: Path,
) -> None:
    transcript_store.ingest_artifact(
        _write_transcript(tmp_path / "identity-ledger.transcript.json"),
        root=tmp_path,
        embedding_provider="debug-hash",
        embedding_model="debug-hash-v1",
    )
    store = conversation_knowledge_store.ConversationKnowledgeStore(tmp_path)
    store.migrate(target_version=3, backup=False)
    legacy_person = conversation_knowledge_store.PersonSnapshot(
        person=conversation_knowledge_store.PersonRecord(
            person_id="00000000-0000-4000-8000-000000000099",
            status="reviewed",
            primary_name="Existing Identity",
        )
    )
    store.save_person_snapshot(legacy_person)

    migration = store.migrate(target_version=4, backup=False)

    assert migration.from_version == 3
    assert migration.to_version == 4
    assert migration.applied_versions == (4,)
    assert store.schema_status().schema_version == 4
    with sqlite3.connect(tmp_path / "transcripts.sqlite3") as con:
        ledger_table = con.execute(
            """
            SELECT 1
            FROM sqlite_master
            WHERE type = 'table'
              AND name = 'knowledge_identity_ledger_events'
            """
        ).fetchone()
    assert ledger_table is not None
    assert store.load_person_snapshot(legacy_person.person.person_id) == legacy_person
    assert transcript_store.search_store(
        "searchable transcript",
        root=tmp_path,
        embedding_provider="debug-hash",
        embedding_model="debug-hash-v1",
    )

    rollback = store.rollback(target_version=3, backup=False)

    assert rollback.from_version == 4
    assert rollback.to_version == 3
    assert rollback.rolled_back_versions == (4,)
    assert store.schema_status().schema_version == 3
    with sqlite3.connect(tmp_path / "transcripts.sqlite3") as con:
        ledger_table = con.execute(
            """
            SELECT 1
            FROM sqlite_master
            WHERE type = 'table'
              AND name = 'knowledge_identity_ledger_events'
            """
        ).fetchone()
    assert ledger_table is None
    assert store.load_person_snapshot(legacy_person.person.person_id) == legacy_person
    assert transcript_store.search_store(
        "searchable transcript",
        root=tmp_path,
        embedding_provider="debug-hash",
        embedding_model="debug-hash-v1",
    )


def test_v5_transcript_correction_migration_preserves_v4_identity_history(
    tmp_path: Path,
) -> None:
    store = conversation_knowledge_store.ConversationKnowledgeStore(tmp_path)
    store.migrate(target_version=4, backup=False)
    ledger = identity_learning_ledger.IdentityLearningLedger(tmp_path)
    person_id = "00000000-0000-4000-8000-000000000098"
    ledger.append_event(
        event_type="person_created",
        payload={
            "person_id": person_id,
            "primary_name": "Existing A1 Person",
            "status": "reviewed",
        },
        actor_id="reviewer:test",
        occurred_at="2026-08-16T13:00:00Z",
        idempotency_key="existing-a1-person",
    )

    migration = store.migrate(target_version=5, backup=False)

    assert migration.from_version == 4
    assert migration.to_version == 5
    assert migration.applied_versions == (5,)
    with sqlite3.connect(tmp_path / "transcripts.sqlite3") as con:
        correction_table = con.execute(
            """
            SELECT 1 FROM sqlite_master
            WHERE type = 'table'
              AND name = 'knowledge_raw_transcript_generations'
            """
        ).fetchone()
    assert correction_table is not None

    rollback = store.rollback(target_version=4, backup=False)

    assert rollback.rolled_back_versions == (5,)
    ledger.rebuild()
    assert person_id in ledger.projection_snapshot()["people"]
    with sqlite3.connect(tmp_path / "transcripts.sqlite3") as con:
        correction_table = con.execute(
            """
            SELECT 1 FROM sqlite_master
            WHERE type = 'table'
              AND name = 'knowledge_raw_transcript_generations'
            """
        ).fetchone()
    assert correction_table is None


def test_person_snapshot_preserves_cross_source_identity_context(
    tmp_path: Path,
) -> None:
    store = conversation_knowledge_store.ConversationKnowledgeStore(tmp_path)
    store.migrate(backup=False)
    person_id = "00000000-0000-4000-8000-000000000021"
    snapshot = conversation_knowledge_store.PersonSnapshot(
        person=conversation_knowledge_store.PersonRecord(
            person_id=person_id,
            status="reviewed",
            primary_name="Example Person",
        ),
        source_records=(
            conversation_knowledge_store.SourceRecord(
                source_record_id="source-record-gws-1",
                person_id=person_id,
                source_profile_id="gws-personal",
                provider_kind="gws",
                account_id="personal",
                tenant_id="",
                external_ref="people/1",
                label="Example Person",
                relationship_scope="personal_interaction",
                identifier_authority="email",
                observed_at="2026-07-26T14:00:00Z",
                content_hash="hash-gws",
            ),
            conversation_knowledge_store.SourceRecord(
                source_record_id="source-record-odollo-1",
                person_id=person_id,
                source_profile_id="odollo-company",
                provider_kind="odollo",
                account_id="",
                tenant_id="company-prod",
                external_ref="res.partner:1",
                label="Example Person",
                relationship_scope="company_interaction",
                identifier_authority="provider_id",
                observed_at="2026-07-26T14:01:00Z",
                content_hash="hash-odollo",
            ),
        ),
        external_identities=(
            conversation_knowledge_store.ExternalIdentityRecord(
                external_identity_id="external-identity-1",
                person_id=person_id,
                source_record_id="source-record-gws-1",
                identity_kind="email",
                normalized_value="person@example.com",
                display_value="person@example.com",
                authority="authoritative",
                verified=True,
            ),
        ),
    )

    first = store.save_person_snapshot(snapshot)
    second = store.save_person_snapshot(snapshot)
    loaded = store.load_person_snapshot(person_id)

    assert first.status == "inserted"
    assert second.status == "unchanged"
    assert loaded == snapshot
    assert {
        (record.source_profile_id, record.relationship_scope)
        for record in loaded.source_records
    } == {
        ("gws-personal", "personal_interaction"),
        ("odollo-company", "company_interaction"),
    }


def test_processing_history_round_trips_without_overwriting_evaluations(
    tmp_path: Path,
) -> None:
    store = conversation_knowledge_store.ConversationKnowledgeStore(tmp_path)
    store.migrate(backup=False)
    conversation_id = "00000000-0000-4000-8000-000000000031"
    store.save_conversation_snapshot(
        conversation_knowledge_store.ConversationSnapshot(
            conversation=conversation_knowledge_store.ConversationRecord(
                conversation_id=conversation_id,
                title="Processing history",
            )
        )
    )
    first_evaluation_id = "00000000-0000-4000-8000-000000000032"
    second_evaluation_id = "00000000-0000-4000-8000-000000000033"
    history = conversation_knowledge_store.ProcessingHistory(
        conversation_id=conversation_id,
        current_evaluation_id=second_evaluation_id,
        evaluations=(
            conversation_knowledge_store.EvaluationRecord(
                evaluation_id=first_evaluation_id,
                conversation_id=conversation_id,
                evaluation_type="speaker_identity",
                schema_version="speaker-identity.v1",
                status="completed",
                created_at="2026-07-26T14:00:00Z",
                payload={"proposal_ids": ["proposal-1"]},
            ),
            conversation_knowledge_store.EvaluationRecord(
                evaluation_id=second_evaluation_id,
                conversation_id=conversation_id,
                evaluation_type="speaker_identity",
                schema_version="speaker-identity.v2",
                status="completed",
                created_at="2026-07-26T14:05:00Z",
                payload={"proposal_ids": ["proposal-2"]},
            ),
        ),
        review_decisions=(
            conversation_knowledge_store.ReviewDecisionRecord(
                decision_id="00000000-0000-4000-8000-000000000034",
                evaluation_id=first_evaluation_id,
                proposal_id="proposal-1",
                action="reject",
                reviewer="operator",
                method="manual",
                decided_at="2026-07-26T14:03:00Z",
                note="Wrong person",
            ),
        ),
    )

    first = store.save_processing_history(history)
    second = store.save_processing_history(history)

    assert first.status == "inserted"
    assert second.status == "unchanged"
    assert store.load_processing_history(conversation_id) == history


def test_failed_migration_rolls_back_every_knowledge_table(
    tmp_path: Path,
    monkeypatch,
) -> None:
    store = conversation_knowledge_store.ConversationKnowledgeStore(tmp_path)
    original = store._apply_v1

    def fail_after_schema(con) -> None:
        original(con)
        raise RuntimeError("injected migration failure")

    monkeypatch.setattr(store, "_apply_v1", fail_after_schema)

    try:
        store.migrate(backup=False)
    except RuntimeError as exc:
        assert str(exc) == "injected migration failure"
    else:
        raise AssertionError("Migration failure was not raised.")

    assert store.schema_status() == (
        conversation_knowledge_store.KnowledgeSchemaStatus(
            schema_version=0,
            authority_mode="sidecar",
            dirty=False,
        )
    )
