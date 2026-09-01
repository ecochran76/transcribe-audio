from __future__ import annotations

import json
import sys
import threading
from pathlib import Path
from urllib.request import Request, urlopen

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from conversation_knowledge_store import ConversationKnowledgeStore
from directory_hypothesis_review import (
    SUBMISSION_SCHEMA,
    DirectoryHypothesisReviewError,
    StaleDirectoryHypothesisReview,
    project_directory_review_hypotheses,
    record_directory_hypothesis_review,
)
from identity_learning_ledger import IdentityLearningLedger
from identity_review_workflow import IdentityReviewWorkflow
import transcript_store
import transcript_api


def _root_with_contact(tmp_path: Path) -> Path:
    ConversationKnowledgeStore(tmp_path).migrate(backup=False)
    metadata = {
        "contact_class": "person_candidate",
        "calendar_attendee": {"appearances": []},
        "enrichment": {
            "source_records": [
                {
                    "provider": "gws",
                    "profile": "fixture",
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
                    "match_basis": "exact_email",
                }
            ]
        },
    }
    with transcript_store.connect(tmp_path) as con:
        transcript_store.init_db(con)
        con.execute(
            """
            INSERT INTO contacts (
              id, label, email, external_ref, metadata_json, created_at, updated_at
            ) VALUES (?, ?, ?, '', ?, ?, ?)
            """,
            (
                "contact-alex",
                "Alex Example",
                "alex@example.test",
                json.dumps(metadata),
                "2026-09-01T00:00:00Z",
                "2026-09-01T00:00:00Z",
            ),
        )
        con.commit()
    return tmp_path


def _submission(lead: dict[str, object], *, action: str = "accept") -> dict[str, object]:
    return {
        "schema_version": SUBMISSION_SCHEMA,
        "hypothesis_id": lead["hypothesis_id"],
        "action": action,
        "expected_projection_version": lead["projection_version"],
        "source_content_sha256": lead["source_content_sha256"],
        "reviewer": "reviewer:test",
        "decided_at": "2026-09-01T12:00:00Z",
        "idempotency_key": f"review-{lead['hypothesis_id']}-{action}",
        "person_target": {"mode": "create"},
        "organization_target": {"mode": "create"},
    }


def test_accept_role_creates_exact_person_organization_source_and_role(
    tmp_path: Path,
) -> None:
    root = _root_with_contact(tmp_path)
    projection = project_directory_review_hypotheses(root)
    lead = next(
        item
        for item in projection["by_contact_id"]["contact-alex"]["role_hypotheses"]
        if item["hypothesis_kind"] == "contextual_role"
    )

    receipt = record_directory_hypothesis_review(root, _submission(lead))
    snapshot = IdentityLearningLedger(root).projection_snapshot()

    assert receipt["accepted_person_effect_count"] == 1
    assert receipt["accepted_organization_effect_count"] == 1
    assert receipt["accepted_role_effect_count"] == 1
    assert receipt["accepted_relationship_effect_count"] == 0
    assert receipt["provider_write_count"] == 0
    assert len(snapshot["people"]) == 1
    assert len(snapshot["organizations"]) == 1
    assert len(snapshot["sources"]) == 1
    role = next(iter(snapshot["roles"].values()))
    assert role["role_type"] == "Research Director"
    assert role["status"] == "reviewed"
    assert role["person_id"] in snapshot["people"]
    assert role["organization_id"] in snapshot["organizations"]
    assert snapshot["sources"]["contact-alex"]["person_id"] == role["person_id"]
    directory = IdentityReviewWorkflow(root).list_directory_index(limit=100)
    assert directory["total"] == 1
    assert directory["items"][0]["accepted_person_id"] == role["person_id"]
    assert directory["items"][0]["organizations"][0]["roles"][0][
        "role_type"
    ] == "Research Director"

    replay = record_directory_hypothesis_review(root, _submission(lead))
    assert replay["idempotent_replay"] is True
    assert len(IdentityLearningLedger(root).events()) == 4


def test_accept_affiliation_renders_reviewed_membership_without_inventing_role(
    tmp_path: Path,
) -> None:
    root = _root_with_contact(tmp_path)
    lead = project_directory_review_hypotheses(root)["by_contact_id"][
        "contact-alex"
    ]["relationship_hypotheses"][0]

    record_directory_hypothesis_review(root, _submission(lead))
    directory = IdentityReviewWorkflow(root).list_directory_index(limit=100)

    assert directory["total"] == 1
    affiliation = directory["items"][0]["organizations"][0]
    assert affiliation["status"] == "reviewed"
    assert affiliation["basis"] == "identity_relationship_projection"
    assert affiliation["roles"] == []
    assert affiliation["role_count"] == 0


def test_reject_and_defer_create_no_accepted_graph_effects(tmp_path: Path) -> None:
    for action in ("reject", "defer"):
        root = _root_with_contact(tmp_path / action)
        projection = project_directory_review_hypotheses(root)
        lead = projection["by_contact_id"]["contact-alex"][
            "relationship_hypotheses"
        ][0]

        receipt = record_directory_hypothesis_review(
            root, _submission(lead, action=action)
        )
        snapshot = IdentityLearningLedger(root).projection_snapshot()

        assert receipt["accepted_person_effect_count"] == 0
        assert receipt["accepted_organization_effect_count"] == 0
        assert receipt["accepted_role_effect_count"] == 0
        assert receipt["accepted_relationship_effect_count"] == 0
        assert snapshot["people"] == {}
        assert snapshot["organizations"] == {}
        assert snapshot["roles"] == {}
        assert snapshot["relationships"] == {}
        assert len(snapshot["reconciliations"]) == 1


def test_review_requires_current_hash_version_and_explicit_accept_targets(
    tmp_path: Path,
) -> None:
    root = _root_with_contact(tmp_path)
    lead = project_directory_review_hypotheses(root)["by_contact_id"][
        "contact-alex"
    ]["relationship_hypotheses"][0]
    stale = _submission(lead)
    stale["expected_projection_version"] = "99"
    with pytest.raises(StaleDirectoryHypothesisReview, match="stale"):
        record_directory_hypothesis_review(root, stale)

    incomplete = _submission(lead)
    incomplete.pop("person_target")
    with pytest.raises(DirectoryHypothesisReviewError, match="person target"):
        record_directory_hypothesis_review(root, incomplete)


def test_directory_hypothesis_review_api_returns_created_receipt(
    tmp_path: Path,
) -> None:
    root = _root_with_contact(tmp_path / "store")
    lead = project_directory_review_hypotheses(root)["by_contact_id"][
        "contact-alex"
    ]["relationship_hypotheses"][0]
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
        request = Request(
            f"http://{host}:{port}/api/directory-hypotheses/{lead['hypothesis_id']}/reviews",
            data=json.dumps(_submission(lead)).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        response = urlopen(request, timeout=5)
        payload = json.loads(response.read())
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)

    assert response.status == 201
    assert payload["accepted_relationship_effect_count"] == 1
    assert payload["provider_write_count"] == 0
