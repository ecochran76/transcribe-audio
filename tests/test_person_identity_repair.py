from __future__ import annotations

import sys
import json
import threading
from pathlib import Path
from urllib.request import Request, urlopen

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from conversation_knowledge_store import ConversationKnowledgeStore
from identity_learning_ledger import IdentityLearningLedger
from person_identity_repair import (
    REPAIR_SUBMISSION_SCHEMA,
    StalePersonIdentityRepair,
    build_person_identity_repair_queue,
    record_person_identity_repair,
)
import transcript_api


def _ledger(tmp_path: Path) -> IdentityLearningLedger:
    ConversationKnowledgeStore(tmp_path).migrate(backup=False)
    return IdentityLearningLedger(tmp_path)


def _append(ledger: IdentityLearningLedger, event_type: str, payload: dict, ordinal: int) -> None:
    ledger.append_event(
        event_type=event_type,
        payload=payload,
        actor_id="reviewer:test",
        occurred_at=f"2026-09-02T12:{ordinal:02d}:00Z",
        idempotency_key=f"fixture-{ordinal}",
    )


def test_queue_separates_actionable_names_ambiguous_names_and_duplicate_candidates(
    tmp_path: Path,
) -> None:
    ledger = _ledger(tmp_path)
    for ordinal, person_id, name in (
        (1, "person-chris", "Rwilliam"),
        (2, "person-email", "gage@example.test"),
        (3, "person-ken-a", "Ken Anderson"),
        (4, "person-ken-b", "Ken Anderson"),
    ):
        _append(
            ledger,
            "person_created",
            {"person_id": person_id, "primary_name": name, "status": "reviewed"},
            ordinal,
        )
    ledger.rebuild()
    directory = {
        "items": [
            {
                "accepted_person_id": "person-chris",
                "display_name": "Chris Williams",
                "name_completeness": "complete",
                "person_name_candidates": ["Chris Williams", "R. Chris Williams"],
                "organizations": [{"primary_name": "Iowa State University"}],
                "source_records": [{"provider_kind": "gws", "label": "Chris Williams"}],
            },
            {
                "accepted_person_id": "person-email",
                "display_name": "gage@example.test",
                "name_completeness": "identifier_only",
                "person_name_candidates": [],
                "organizations": [{"primary_name": "Skyfleet"}],
                "source_records": [{"provider_kind": "odollo", "label": "gage@example.test"}],
            },
            *[
                {
                    "accepted_person_id": person_id,
                    "display_name": "Ken Anderson",
                    "name_completeness": "complete",
                    "person_name_candidates": ["Ken Anderson"],
                    "organizations": [],
                    "source_records": [{"provider_kind": "human_review", "label": "Ken Anderson"}],
                }
                for person_id in ("person-ken-a", "person-ken-b")
            ],
        ]
    }

    queue = build_person_identity_repair_queue(directory, ledger.projection_snapshot())

    assert queue["counts"] == {
        "all": 3,
        "actionable": 2,
        "canonical_name": 1,
        "identity_ambiguity": 1,
        "possible_duplicate": 1,
    }
    by_kind = {item["repair_kind"]: item for item in queue["items"]}
    assert by_kind["canonical_name"]["suggested_primary_name"] == "Chris Williams"
    assert by_kind["identity_ambiguity"]["allowed_actions"] == []
    assert set(by_kind["possible_duplicate"]["person_ids"]) == {
        "person-ken-a",
        "person-ken-b",
    }


def test_name_repair_is_append_only_idempotent_and_stale_safe(tmp_path: Path) -> None:
    ledger = _ledger(tmp_path)
    _append(
        ledger,
        "person_created",
        {"person_id": "person-chris", "primary_name": "Rwilliam", "status": "reviewed"},
        1,
    )
    ledger.rebuild()
    directory = {
        "items": [{
            "accepted_person_id": "person-chris",
            "display_name": "Chris Williams",
            "name_completeness": "complete",
            "person_name_candidates": ["Chris Williams", "R. Chris Williams"],
            "organizations": [{"primary_name": "Iowa State University"}],
            "source_records": [{"provider_kind": "gws", "label": "Chris Williams"}],
        }]
    }
    queue = build_person_identity_repair_queue(directory, ledger.projection_snapshot())
    repair = queue["items"][0]
    submission = {
        "schema_version": REPAIR_SUBMISSION_SCHEMA,
        "repair_id": repair["repair_id"],
        "repair_kind": "canonical_name",
        "action": "correct_name",
        "expected_content_sha256": repair["content_sha256"],
        "person_id": "person-chris",
        "replacement_primary_name": "Chris Williams",
        "reviewer": "operator",
        "decided_at": "2026-09-02T13:00:00Z",
        "idempotency_key": "repair-chris-1",
    }

    receipt = record_person_identity_repair(tmp_path, submission, queue)
    replay = record_person_identity_repair(tmp_path, submission, queue)

    assert receipt["status"] == "inserted"
    assert replay["idempotent_replay"] is True
    assert ledger.projection_snapshot()["people"]["person-chris"]["primary_name"] == "Chris Williams"
    assert [event["event_type"] for event in ledger.events()] == [
        "person_created",
        "person_corrected",
    ]

    stale = {**submission, "idempotency_key": "repair-chris-2"}
    with pytest.raises(StalePersonIdentityRepair, match="stale"):
        record_person_identity_repair(
            tmp_path,
            stale,
            build_person_identity_repair_queue(directory, ledger.projection_snapshot()),
        )


def test_merge_adopts_a_canonical_profile_missing_from_the_identity_ledger(
    tmp_path: Path,
) -> None:
    ledger = _ledger(tmp_path)
    _append(
        ledger,
        "person_created",
        {"person_id": "person-ledger", "primary_name": "Ken Anderson", "status": "reviewed"},
        1,
    )
    ledger.rebuild()
    directory = {
        "items": [
            {
                "accepted_person_id": person_id,
                "primary_name": "Ken Anderson",
                "display_name": "Ken Anderson",
                "name_completeness": "complete",
                "person_name_candidates": ["Ken Anderson"],
                "organizations": [],
                "source_records": [
                    {"provider_kind": provider, "label": "Ken Anderson"}
                ],
            }
            for person_id, provider in (
                ("person-ledger", "human_review"),
                ("person-profile", "directory_projection"),
            )
        ]
    }
    queue = build_person_identity_repair_queue(directory, ledger.projection_snapshot())
    repair = next(
        item for item in queue["items"] if item["repair_kind"] == "possible_duplicate"
    )
    assert [participant["in_identity_ledger"] for participant in repair["participants"]] == [
        True,
        False,
    ]

    receipt = record_person_identity_repair(
        tmp_path,
        {
            "schema_version": REPAIR_SUBMISSION_SCHEMA,
            "repair_id": repair["repair_id"],
            "repair_kind": "possible_duplicate",
            "action": "merge_people",
            "expected_content_sha256": repair["content_sha256"],
            "target_person_id": "person-ledger",
            "reviewer": "operator",
            "decided_at": "2026-09-02T13:30:00Z",
            "idempotency_key": "repair-ken-merge-1",
        },
        queue,
    )

    people = ledger.projection_snapshot()["people"]
    assert receipt["status"] == "inserted"
    assert people["person-profile"]["merged_into_person_id"] == "person-ledger"
    assert [event["event_type"] for event in ledger.events()] == [
        "person_created",
        "person_created",
        "people_merged",
    ]


def test_person_repair_api_lists_and_applies_a_name_correction(tmp_path: Path) -> None:
    root = tmp_path / "store"
    ledger = _ledger(root)
    _append(
        ledger,
        "person_created",
        {"person_id": "person-chris", "primary_name": "Rwilliam", "status": "reviewed"},
        1,
    )
    _append(
        ledger,
        "alias_added",
        {"person_id": "person-chris", "alias": "Chris Williams"},
        2,
    )
    ledger.rebuild()
    server = transcript_api.TranscriptApiServer(
        ("127.0.0.1", 0),
        transcript_api.TranscriptApiHandler,
        store_root=root,
        embedding_provider="debug-hash",
        embedding_model="debug-hash",
        state_root=tmp_path / "state",
        quiet=True,
        static_dir=None,
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address
    try:
        response = urlopen(f"http://{host}:{port}/api/person-repairs?limit=500", timeout=5)
        queue = json.loads(response.read())
        repair = queue["items"][0]
        submission = {
            "schema_version": REPAIR_SUBMISSION_SCHEMA,
            "repair_id": repair["repair_id"],
            "repair_kind": repair["repair_kind"],
            "action": "correct_name",
            "expected_content_sha256": repair["content_sha256"],
            "person_id": repair["person_id"],
            "replacement_primary_name": repair["suggested_primary_name"],
            "reviewer": "operator",
            "decided_at": "2026-09-02T14:00:00Z",
            "idempotency_key": "api-person-repair-1",
        }
        request = Request(
            f"http://{host}:{port}/api/person-repairs/{repair['repair_id']}",
            data=json.dumps(submission).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        applied_response = urlopen(request, timeout=5)
        applied = json.loads(applied_response.read())
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)

    assert response.status == 200
    assert queue["counts"]["canonical_name"] == 1
    assert applied_response.status == 201
    assert applied["action"] == "correct_name"
    assert applied["provider_write_count"] == 0
    assert ledger.projection_snapshot()["people"]["person-chris"]["primary_name"] == "Chris Williams"
