from __future__ import annotations

import copy

import speaker_identity_plan0064_p3 as p3


def _acoustic(label, person=None, *, status="candidate", models=2):
    return {
        "speaker_ref": f"doc::{label}",
        "speaker_label": label,
        "status": status,
        "candidate_person_id": person,
        "supporting_model_count": models,
        "confidence_band": "high" if person else "none",
        "reason_code": "test_acoustic",
        "probe_sha256": "a" * 64,
        "model_rows": [],
    }


def _candidate(person, proposal, *, contradiction=False):
    return {
        "status": "candidate_match",
        "person_id": person,
        "prepared_person_id": person,
        "proposal_id": proposal,
        "transcript_clue_ids": ["clue-1"],
        "provenance_source_ids": ["source-1"],
        "review_flags": [],
        "factors": [
            {
                "direction": "contradict" if contradiction else "support",
                "strength": "strong",
                "evidence_ids": ["evidence-1"],
            }
        ],
    }


def _context(label, person=None, proposal="proposal-1", *, candidates=None):
    selected = candidates if candidates is not None else (
        [_candidate(person, proposal)] if person else []
    )
    return {
        "speaker_ref": f"doc::{label}",
        "speaker_label": label,
        "disposition": "candidate" if person else "abstain",
        "reason_code": "test_context",
        "candidate_person_id": person,
        "candidates": selected,
    }


def _recording(acoustic_slots, context_slots, *, provider_failures=None):
    return (
        {
            "document_id": "doc",
            "transcript_sha256": "b" * 64,
            "source_media_sha256": "c" * 64,
            "speaker_slots": acoustic_slots,
        },
        {
            "document_id": "doc",
            "speaker_slots": context_slots,
            "provider_failures": provider_failures or [],
        },
    )


def test_agreement_and_conflict_preserve_independent_pillars():
    recording, case = _recording(
        [_acoustic("A", "person-1"), _acoustic("B", "person-1")],
        [_context("A", "person-1"), _context("B", "person-2")],
    )
    rows = p3.resolve_conversation(recording, case)["speaker_slots"]
    assert rows[0]["combined"]["disposition"] == "candidate"
    assert rows[0]["combined"]["reason_code"] == "pillar_agreement"
    assert rows[1]["combined"]["disposition"] == "abstain"
    assert rows[1]["combined"]["reason_code"] == "pillar_conflict"
    assert rows[1]["acoustic"]["candidate_person_id"] == "person-1"
    assert rows[1]["context"]["candidate_person_id"] == "person-2"


def test_one_to_one_collision_routes_review_but_shared_proposal_allows_multilabel():
    recording, case = _recording(
        [_acoustic("A", "person-1"), _acoustic("B", "person-1")],
        [_context("A", "person-1", "proposal-a"), _context("B", "person-1", "proposal-b")],
    )
    rows = p3.resolve_conversation(recording, case)["speaker_slots"]
    assert {row["combined"]["reason_code"] for row in rows} == {"global_person_collision"}
    shared = copy.deepcopy(case)
    shared["speaker_slots"][1] = _context("B", "person-1", "proposal-a")
    rows = p3.resolve_conversation(recording, shared)["speaker_slots"]
    assert {row["combined"]["reason_code"] for row in rows} == {
        "pillar_agreement_same_person_multi_label"
    }


def test_residual_requires_two_known_and_independent_context_support():
    recording, case = _recording(
        [
            _acoustic("A", "person-1"),
            _acoustic("B", "person-2"),
            _acoustic("C", None, status="abstain", models=0),
        ],
        [
            _context("A", "person-1", "proposal-a"),
            _context("B", "person-2", "proposal-b"),
            _context("C", "person-3", "proposal-c"),
        ],
    )
    rows = p3.resolve_conversation(recording, case)["speaker_slots"]
    assert rows[2]["combined"]["reason_code"] == "context_only_support"
    assert rows[2]["residual_policy"]["candidate_person_id"] == "person-3"
    assert rows[2]["residual_policy"]["reason_code"] == (
        "two_known_plus_one_independently_supported_residual"
    )


def test_residual_abstains_on_material_contradiction_or_provider_failure():
    recording, case = _recording(
        [
            _acoustic("A", "person-1"),
            _acoustic("B", "person-2"),
            _acoustic("C", None, status="abstain", models=0),
        ],
        [
            _context("A", "person-1", "proposal-a"),
            _context("B", "person-2", "proposal-b"),
            _context(
                "C",
                "person-3",
                candidates=[_candidate("person-3", "proposal-c", contradiction=True)],
            ),
        ],
    )
    rows = p3.resolve_conversation(recording, case)["speaker_slots"]
    assert rows[2]["residual_policy"]["disposition"] == "review"
    clean_case = copy.deepcopy(case)
    clean_case["speaker_slots"][2] = _context("C", "person-3", "proposal-c")
    clean_case["provider_failures"] = [{"reason_code": "provider_failed"}]
    rows = p3.resolve_conversation(recording, clean_case)["speaker_slots"]
    assert rows[2]["residual_policy"]["disposition"] == "review"
