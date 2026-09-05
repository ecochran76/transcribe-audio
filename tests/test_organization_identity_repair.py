from __future__ import annotations

import json
import sys
import threading
from pathlib import Path
from urllib.request import Request, urlopen

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from conversation_knowledge_store import ConversationKnowledgeStore
from identity_learning_ledger import IdentityLearningLedger
from organization_identity_repair import (
    REPAIR_SUBMISSION_SCHEMA,
    build_organization_identity_repair_queue,
    record_organization_identity_repair,
)
import transcript_api


def _ledger(tmp_path: Path) -> IdentityLearningLedger:
    ConversationKnowledgeStore(tmp_path).migrate(backup=False)
    return IdentityLearningLedger(tmp_path)


def _append_organization(
    ledger: IdentityLearningLedger, organization_id: str, name: str, ordinal: int
) -> None:
    ledger.append_event(
        event_type="organization_created",
        payload={
            "organization_id": organization_id,
            "primary_name": name,
            "status": "reviewed",
        },
        actor_id="reviewer:test",
        occurred_at=f"2026-09-04T12:{ordinal:02d}:00Z",
        idempotency_key=f"organization-{ordinal}",
    )


def test_queue_separates_alias_unit_and_related_organization_candidates(
    tmp_path: Path,
) -> None:
    ledger = _ledger(tmp_path)
    organizations = (
        ("org-isu", "Iowa State University"),
        ("org-isu-legal", "The Iowa State University of Science and Technology"),
        (
            "org-cbe",
            "Iowa State University Department of Chemical and Biological Engineering",
        ),
        ("org-iprt", "IPRT, ISU"),
    )
    for ordinal, (organization_id, name) in enumerate(organizations, start=1):
        _append_organization(ledger, organization_id, name, ordinal)
    ledger.rebuild()
    directory = {
        "items": [
            {
                "organization_id": organization_id,
                "accepted_organization_id": organization_id,
                "primary_name": name,
                "aliases": [],
                "source_records": [],
            }
            for organization_id, name in organizations
        ]
    }

    queue = build_organization_identity_repair_queue(
        directory, ledger.projection_snapshot()
    )

    assert queue["counts"] == {
        "all": 3,
        "actionable": 3,
        "possible_alias": 1,
        "unit_candidate": 1,
        "related_candidate": 1,
    }
    by_kind = {item["repair_kind"]: item for item in queue["items"]}
    assert by_kind["possible_alias"]["organization_ids"] == [
        "org-isu",
        "org-isu-legal",
    ]
    assert by_kind["possible_alias"]["suggested_action"] == "merge_organizations"
    assert by_kind["possible_alias"]["suggested_target_organization_id"] == "org-isu"
    assert by_kind["unit_candidate"]["suggested_parent_id"] == "org-isu"
    assert by_kind["unit_candidate"]["suggested_child_id"] == "org-cbe"
    assert by_kind["unit_candidate"]["suggested_action"] == "set_parent"
    assert by_kind["related_candidate"]["organization_ids"] == [
        "org-iprt",
        "org-isu",
    ]
    assert by_kind["related_candidate"]["suggested_action"] == "relate_organizations"
    assert all(item["mutation_count"] == 0 for item in queue["items"])


def test_alias_review_merges_organizations_and_preserves_source_name(
    tmp_path: Path,
) -> None:
    ledger = _ledger(tmp_path)
    _append_organization(ledger, "org-isu", "Iowa State University", 1)
    _append_organization(
        ledger,
        "org-isu-legal",
        "The Iowa State University of Science and Technology",
        2,
    )
    ledger.rebuild()
    directory = {
        "items": [
            {
                "organization_id": organization_id,
                "accepted_organization_id": organization_id,
                "primary_name": name,
                "aliases": [],
                "source_records": [],
            }
            for organization_id, name in (
                ("org-isu", "Iowa State University"),
                (
                    "org-isu-legal",
                    "The Iowa State University of Science and Technology",
                ),
            )
        ]
    }
    queue = build_organization_identity_repair_queue(
        directory, ledger.projection_snapshot()
    )
    repair = queue["items"][0]
    submission = {
        "schema_version": REPAIR_SUBMISSION_SCHEMA,
        "repair_id": repair["repair_id"],
        "repair_kind": repair["repair_kind"],
        "action": "merge_organizations",
        "expected_content_sha256": repair["content_sha256"],
        "target_organization_id": "org-isu",
        "reviewer": "operator",
        "decided_at": "2026-09-04T22:30:00Z",
        "idempotency_key": "merge-isu-legal-1",
    }

    receipt = record_organization_identity_repair(tmp_path, submission, queue)
    replay = record_organization_identity_repair(tmp_path, submission, queue)
    organizations = ledger.projection_snapshot()["organizations"]

    assert receipt["status"] == "inserted"
    assert replay["idempotent_replay"] is True
    assert organizations["org-isu-legal"]["merged_into_organization_id"] == "org-isu"
    assert json.loads(organizations["org-isu"]["aliases_json"]) == [
        "The Iowa State University of Science and Technology"
    ]


def test_unit_review_sets_parent_and_typed_relationship_atomically(
    tmp_path: Path,
) -> None:
    ledger = _ledger(tmp_path)
    _append_organization(ledger, "org-isu", "Iowa State University", 1)
    _append_organization(
        ledger,
        "org-cbe",
        "Iowa State University Department of Chemical and Biological Engineering",
        2,
    )
    ledger.rebuild()
    directory = {
        "items": [
            {
                "organization_id": organization_id,
                "accepted_organization_id": organization_id,
                "primary_name": name,
                "aliases": [],
                "source_records": [],
            }
            for organization_id, name in (
                ("org-isu", "Iowa State University"),
                (
                    "org-cbe",
                    "Iowa State University Department of Chemical and Biological Engineering",
                ),
            )
        ]
    }
    queue = build_organization_identity_repair_queue(
        directory, ledger.projection_snapshot()
    )
    repair = queue["items"][0]

    receipt = record_organization_identity_repair(
        tmp_path,
        {
            "schema_version": REPAIR_SUBMISSION_SCHEMA,
            "repair_id": repair["repair_id"],
            "repair_kind": repair["repair_kind"],
            "action": "set_parent",
            "expected_content_sha256": repair["content_sha256"],
            "parent_organization_id": "org-isu",
            "child_organization_id": "org-cbe",
            "reviewer": "operator",
            "decided_at": "2026-09-04T22:40:00Z",
            "idempotency_key": "set-cbe-parent-1",
        },
        queue,
    )
    snapshot = ledger.projection_snapshot()

    assert receipt["status"] == "inserted"
    assert snapshot["organizations"]["org-cbe"]["parent_organization_id"] == "org-isu"
    relationship = next(iter(snapshot["relationships"].values()))
    assert relationship["relationship_type"] == "unit_of"
    assert relationship["subject_id"] == "org-cbe"
    assert relationship["object_id"] == "org-isu"
    assert relationship["status"] == "reviewed"


def test_related_review_records_selected_typed_relationship(tmp_path: Path) -> None:
    ledger = _ledger(tmp_path)
    _append_organization(ledger, "org-isu", "Iowa State University", 1)
    _append_organization(ledger, "org-iprt", "IPRT, ISU", 2)
    ledger.rebuild()
    directory = {
        "items": [
            {
                "organization_id": organization_id,
                "accepted_organization_id": organization_id,
                "primary_name": name,
                "aliases": [],
                "source_records": [],
            }
            for organization_id, name in (
                ("org-isu", "Iowa State University"),
                ("org-iprt", "IPRT, ISU"),
            )
        ]
    }
    queue = build_organization_identity_repair_queue(
        directory, ledger.projection_snapshot()
    )
    repair = queue["items"][0]

    record_organization_identity_repair(
        tmp_path,
        {
            "schema_version": REPAIR_SUBMISSION_SCHEMA,
            "repair_id": repair["repair_id"],
            "repair_kind": repair["repair_kind"],
            "action": "relate_organizations",
            "expected_content_sha256": repair["content_sha256"],
            "subject_organization_id": "org-iprt",
            "object_organization_id": "org-isu",
            "relationship_type": "related_to",
            "reviewer": "operator",
            "decided_at": "2026-09-04T22:50:00Z",
            "idempotency_key": "relate-iprt-isu-1",
        },
        queue,
    )
    relationship = next(iter(ledger.projection_snapshot()["relationships"].values()))

    assert relationship["relationship_type"] == "related_to"
    assert relationship["subject_id"] == "org-iprt"
    assert relationship["object_id"] == "org-isu"
    assert relationship["directionality"] == "symmetric"


def test_distinct_organization_review_suppresses_only_unchanged_candidate(
    tmp_path: Path,
) -> None:
    ledger = _ledger(tmp_path)
    _append_organization(ledger, "org-isu", "Iowa State University", 1)
    _append_organization(ledger, "org-iprt", "IPRT, ISU", 2)
    ledger.rebuild()
    directory = {
        "items": [
            {
                "organization_id": organization_id,
                "accepted_organization_id": organization_id,
                "primary_name": name,
                "aliases": [],
                "source_records": [],
            }
            for organization_id, name in (
                ("org-isu", "Iowa State University"),
                ("org-iprt", "IPRT, ISU"),
            )
        ]
    }
    queue = build_organization_identity_repair_queue(
        directory, ledger.projection_snapshot()
    )
    repair = queue["items"][0]
    submission = {
        "schema_version": REPAIR_SUBMISSION_SCHEMA,
        "repair_id": repair["repair_id"],
        "repair_kind": repair["repair_kind"],
        "action": "mark_distinct",
        "expected_content_sha256": repair["content_sha256"],
        "reviewer": "operator",
        "decided_at": "2026-09-04T23:00:00Z",
        "idempotency_key": "distinct-iprt-isu-1",
    }

    receipt = record_organization_identity_repair(tmp_path, submission, queue)
    replay = record_organization_identity_repair(tmp_path, submission, queue)
    refreshed = build_organization_identity_repair_queue(
        directory, ledger.projection_snapshot()
    )

    assert receipt["status"] == "inserted"
    assert replay["idempotent_replay"] is True
    assert refreshed["items"] == []
    assert [event["event_type"] for event in ledger.events()] == [
        "organization_created",
        "organization_created",
        "reconciliation_proposed",
        "reconciliation_decided",
    ]


def test_organization_repair_api_lists_and_records_exact_review(tmp_path: Path) -> None:
    ledger = _ledger(tmp_path)
    _append_organization(ledger, "org-isu", "Iowa State University", 1)
    _append_organization(ledger, "org-legal", "The Iowa State University of Science and Technology", 2)
    ledger.rebuild()
    server = transcript_api.TranscriptApiServer(
        ("127.0.0.1", 0),
        transcript_api.TranscriptApiHandler,
        store_root=tmp_path,
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
        response = urlopen(
            f"http://{host}:{port}/api/organization-repairs?limit=500", timeout=5
        )
        queue = json.loads(response.read())
        repair = queue["items"][0]
        submission = {
            "schema_version": REPAIR_SUBMISSION_SCHEMA,
            "repair_id": repair["repair_id"],
            "repair_kind": repair["repair_kind"],
            "action": "merge_organizations",
            "expected_content_sha256": repair["content_sha256"],
            "target_organization_id": "org-isu",
            "reviewer": "operator",
            "decided_at": "2026-09-04T23:30:00Z",
            "idempotency_key": "api-organization-repair-1",
        }
        request = Request(
            f"http://{host}:{port}/api/organization-repairs/{repair['repair_id']}",
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
    assert queue["counts"]["possible_alias"] == 1
    assert applied_response.status == 201
    assert applied["action"] == "merge_organizations"
    assert applied["provider_write_count"] == 0
    assert ledger.projection_snapshot()["organizations"]["org-legal"][
        "merged_into_organization_id"
    ] == "org-isu"
