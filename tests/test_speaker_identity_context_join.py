from __future__ import annotations

from copy import deepcopy
from dataclasses import replace

import pytest

import speaker_identity_context_join as context_join
import speaker_identity_plan0062_execution as plan0062
import speaker_identity_preprocess
from speaker_identity_orchestration import (
    AcousticEvidenceBundle,
    AcousticSpeakerEvidence,
    EvidenceLineage,
    negative_action_vector,
)


NOW = "2026-08-08T21:30:00Z"
HASH_A = "a" * 64
HASH_B = "b" * 64
HASH_C = "c" * 64


def transcript(*, speakers: tuple[str, ...] = ("SPEAKER_1",)) -> dict:
    return {
        "conversation_id": "conversation-001",
        "recording_id": "recording-001",
        "transcript_title": "Proposal review",
        "recording_start": NOW,
        "utterances": [
            {
                "speaker": speaker,
                "start": index * 4,
                "end": index * 4 + 4,
                "text": f"Prepared clue for {speaker}.",
            }
            for index, speaker in enumerate(speakers)
        ],
        "event": {
            "summary": "Proposal review",
            "id": "event-001",
            "participants": [
                {"displayName": "Alice Example", "email": "alice@example.com"}
            ],
        },
    }


def packet(*, speakers: tuple[str, ...] = ("SPEAKER_1",)) -> dict:
    discovery = {
        "schema_version": "transcribe-audio.speaker-clue-discovery-readout.v1",
        "speaker_clues": [],
        "conversation_clues": [],
        "warnings": [],
    }
    result = speaker_identity_preprocess.build_identity_evaluation_packet(
        transcript=transcript(speakers=speakers),
        discovery_readout=discovery,
        person_records=[
            {
                "contact_id": "contact-alice",
                "label": "Alice Example",
                "email": "alice@example.com",
                "source_id": "gws-personal",
                "source_type": "gws_contact",
            },
            {
                "contact_id": "contact-bob",
                "label": "Bob Example",
                "email": "bob@example.com",
                "source_id": "gws-personal",
                "source_type": "gws_contact",
            },
        ],
        source_contexts=[
            {
                "source_id": "gws-personal",
                "source_type": "gws",
                "source_profile": "gws-personal",
                "account_id": "account-personal",
                "tenant_id": "tenant-personal",
                "capabilities": ["calendar", "people"],
            }
        ],
    )
    result["retrieval"] = {
        "bundle_content_hash": HASH_C,
        "as_of": NOW,
        "conversation_at": NOW,
        "retrieval_version": "retrieval-v1",
        "ranking_version": "ranking-v1",
        "budgets": {
            "max_records": 10,
            "max_characters": 1000,
            "max_per_source": 5,
            "max_provider_calls": 2,
            "max_relationship_hops": 0,
        },
        "evidence": [],
        "source_failures": [],
        "warnings": [],
    }
    return result


def split_scope_packet(*, speakers: tuple[str, ...] = ("SPEAKER_1",)) -> dict:
    prepared = packet(speakers=speakers)
    prepared["source_contexts"] = [
        {
            "source_id": "gws-personal",
            "owner": {"id": "account-personal", "type": "person"},
            "relationship_scope": "tenant-personal",
            "evidence_capabilities": ["calendar", "people"],
        },
        {
            "source_id": "gws-personal",
            "source_profile": "gws-personal",
            "account_id": "",
            "tenant_id": "",
            "capabilities": ["calendar", "people"],
            "retrieval_scope": "explicit",
        },
    ]
    return prepared


def factor(person_id: str, *, utterance_id: str = "utterance-1") -> dict:
    return {
        "factor": "direct_self_identification",
        "direction": "support",
        "strength": "decisive",
        "evidence_ids": [utterance_id, person_id],
        "rationale": "Prepared fixture evidence supports the candidate.",
    }


def readout(
    prepared: dict,
    *,
    status: str = "candidate_match",
    person_id: str | None = None,
    speaker_labels: tuple[str, ...] = ("SPEAKER_1",),
) -> dict:
    selected_person = person_id or prepared["people"][0]["person_id"]
    assignment = {
        "speaker_labels": list(speaker_labels),
        "status": status,
        "person_id": selected_person if status == "candidate_match" else "",
        "suggested_person": (
            {"name": "New Person", "email": "new@example.com", "organization": "Example"}
            if status == "unlisted"
            else {}
        ),
        "transcript_clue_ids": ["utterance-1"],
        "provenance_source_ids": ["gws-personal"],
        "factors": [factor(selected_person)],
        "utterance_assignments": [],
        "rationale": "Review-only fixture proposal.",
        "review_flags": [],
    }
    return {
        "schema_version": "transcribe-audio.speaker-identity-evaluation-readout.v1",
        "evaluation_id": prepared["evaluation_id"],
        "calendar_association": {"status": "matched", "factors": []},
        "person_links": [],
        "speaker_assignments": [assignment],
        "warnings": [],
    }


def acoustic(
    *,
    speakers: tuple[str, ...] = ("SPEAKER_1",),
    subject_id: str | None = "subject-alice",
    disposition: str = "review",
) -> AcousticEvidenceBundle:
    lineage = tuple(
        EvidenceLineage(
            evidence_id=f"acoustic-evidence-{index + 1}",
            source_record_id=f"acoustic-source-{index + 1}",
            independence_group=f"acoustic-session-{index + 1}",
            source_type="acoustic_verification",
            source_event_at=NOW,
            observed_at=NOW,
            retrieved_at=NOW,
            content_sha256=HASH_A,
        )
        for index, _speaker in enumerate(speakers)
    )
    return AcousticEvidenceBundle(
        conversation_id="conversation-001",
        recording_id="recording-001",
        document_id="document-001",
        speaker_refs=speakers,
        source_media_sha256=HASH_A,
        transcript_sha256=HASH_B,
        execution_sha256=HASH_C,
        identity_state_sha256=HASH_A,
        model_versions=(("speaker-model", "v1"),),
        created_at=NOW,
        evidence=tuple(
            AcousticSpeakerEvidence(
                speaker_ref=speaker,
                disposition=disposition,
                acoustic_subject_id=subject_id if disposition != "abstain" else None,
                score=0.88 if disposition != "abstain" else 0.0,
                confidence_band="high" if disposition != "abstain" else "none",
                supporting_unit_count=3 if disposition != "abstain" else 0,
                opposing_unit_count=0,
                insufficient_unit_count=0 if disposition != "abstain" else 1,
                evidence_ids=(lineage[index].evidence_id,),
            )
            for index, speaker in enumerate(speakers)
        ),
        lineage=lineage,
        negative_actions=negative_action_vector(),
    )


def join(
    prepared: dict,
    model_readout: dict,
    *,
    acoustic_bundle: AcousticEvidenceBundle | None = None,
    bindings: dict[str, str] | None = None,
) -> context_join.ContextualIdentityJoinResult:
    return context_join.join_contextual_identity(
        document_id="document-001",
        transcript_sha256=HASH_B,
        identity_packet=prepared,
        identity_readout=model_readout,
        acoustic_bundle=acoustic_bundle or acoustic(),
        speaker_ref_bindings={
            str(item["speaker_label"]): str(item["speaker_label"])
            for item in prepared["speakers"]
        },
        acoustic_subject_person_bindings=bindings or {},
        evaluated_at=NOW,
    )


def by_condition(result: context_join.ContextualIdentityJoinResult) -> dict[str, object]:
    return {item.condition: item for item in result.evaluations}


def test_candidate_match_and_explicit_acoustic_binding_propose_same_person() -> None:
    prepared = packet()
    person_id = prepared["people"][0]["person_id"]

    result = join(
        prepared,
        readout(prepared, person_id=person_id),
        bindings={"subject-alice": person_id},
    )

    evaluations = by_condition(result)
    assert len(result.candidate_snapshots) == 1
    assert [item.person_id for item in result.candidate_snapshots[0].candidates] == [person_id]
    assert all(item.outcome == "proposed" for item in evaluations.values())
    assert all(item.proposed_person_id == person_id for item in evaluations.values())
    assert result.review_outcomes[0].context_person_id == person_id
    assert result.negative_actions == negative_action_vector()
    assert len(result.content_sha256) == 64


def test_split_operator_authority_and_retrieval_scope_are_rejoined_by_source() -> None:
    prepared = split_scope_packet()
    person_id = prepared["people"][0]["person_id"]

    result = join(
        prepared,
        readout(prepared, person_id=person_id),
        bindings={"subject-alice": person_id},
    )

    assert len(result.context_bundle.scopes) == 1
    scope = result.context_bundle.scopes[0]
    assert scope.source_profile == "gws-personal"
    assert scope.account_id == "account-personal"
    assert scope.tenant_id == "tenant-personal"
    assert scope.capabilities == ("calendar", "people")


def plan0062_cases() -> tuple[plan0062.ContextualJoinCase, ...]:
    result = []
    for document_id in plan0062.EXPECTED_DOCUMENTS:
        count = plan0062.EXPECTED_SPEAKER_COUNTS[document_id]
        speakers = tuple(f"SPEAKER_{index}" for index in range(1, count + 1))
        prepared = packet(speakers=speakers)
        model_readout = readout(
            prepared,
            person_id=prepared["people"][0]["person_id"],
            speaker_labels=speakers,
        )
        acoustic_bundle = replace(
            acoustic(speakers=speakers, disposition="abstain", subject_id=None),
            document_id=document_id,
        )
        result.append(
            plan0062.ContextualJoinCase(
                document_id=document_id,
                transcript_sha256=HASH_B,
                identity_packet=prepared,
                identity_readout=model_readout,
                acoustic_bundle=acoustic_bundle,
                speaker_ref_bindings={speaker: speaker for speaker in speakers},
                acoustic_subject_person_bindings={},
                evaluated_at=NOW,
                run_references={"identity_evaluation_run_id": f"run-{document_id}"},
            )
        )
    return tuple(result)


def test_plan0062_manifest_has_exact_three_by_ten_by_thirty_denominator() -> None:
    manifest = plan0062.build_contextual_join_manifest(
        plan0062_cases(), activation_sha256=HASH_A, created_at=NOW
    )

    assert manifest["recording_count"] == 3
    assert manifest["speaker_count"] == 10
    assert manifest["evaluation_count"] == 30
    assert manifest["negative_actions"] == negative_action_vector()


def test_plan0062_private_freeze_replays_exactly(tmp_path) -> None:
    first = plan0062.freeze_contextual_join_manifest(
        plan0062_cases(),
        activation_sha256=HASH_A,
        created_at=NOW,
        runtime_root=tmp_path / "plan-0062",
    )
    replay = plan0062.freeze_contextual_join_manifest(
        plan0062_cases(),
        activation_sha256=HASH_A,
        created_at=NOW,
        runtime_root=tmp_path / "plan-0062",
    )

    assert first["idempotent_replay"] is False
    assert replay["idempotent_replay"] is True
    assert replay["content_sha256"] == first["content_sha256"]
    assert replay["manifest_sha256"] == first["manifest_sha256"]


def test_unlisted_suggestion_survives_but_cannot_become_a_person() -> None:
    prepared = packet()

    result = join(
        prepared,
        readout(prepared, status="unlisted"),
        acoustic_bundle=acoustic(disposition="abstain", subject_id=None),
    )

    evaluations = by_condition(result)
    assert evaluations["context_only"].outcome == "abstained"
    assert evaluations["context_only"].abstention_reason == "context_unlisted_person_requires_review"
    assert evaluations["combined"].outcome == "abstained"
    assert result.candidate_snapshots[0].candidates == ()
    assert result.review_outcomes[0].suggestions == (
        context_join.SuggestedPerson("New Person", "new@example.com", "Example"),
    )


def test_duplicate_speaker_coverage_fails_closed_without_discarding_suggestions() -> None:
    prepared = packet()
    model_readout = readout(prepared, status="unlisted")
    duplicate = deepcopy(model_readout["speaker_assignments"][0])
    duplicate["suggested_person"] = {
        "name": "Another Person",
        "email": "another@example.com",
        "organization": "Example",
    }
    model_readout["speaker_assignments"].append(duplicate)

    result = join(
        prepared,
        model_readout,
        acoustic_bundle=acoustic(disposition="abstain", subject_id=None),
    )

    evaluations = by_condition(result)
    assert evaluations["context_only"].abstention_reason == "context_duplicate_speaker_coverage"
    assert evaluations["combined"].abstention_reason == "context_duplicate_speaker_coverage"
    assert len(result.review_outcomes[0].suggestions) == 2
    assert result.review_outcomes[0].reason_code == "context_duplicate_speaker_coverage"


def test_context_and_acoustic_disagreement_abstains_with_both_alternatives() -> None:
    prepared = packet()
    context_person = prepared["people"][0]["person_id"]
    acoustic_person = prepared["people"][1]["person_id"]

    result = join(
        prepared,
        readout(prepared, person_id=context_person),
        bindings={"subject-alice": acoustic_person},
    )

    evaluations = by_condition(result)
    combined = evaluations["combined"]
    assert evaluations["context_only"].proposed_person_id == context_person
    assert evaluations["acoustic_only"].proposed_person_id == acoustic_person
    assert combined.outcome == "abstained"
    assert combined.abstention_reason == "pillar_identity_conflict"
    assert set(combined.alternative_person_ids) == {context_person, acoustic_person}
    assert combined.capped_confidence == 0.49
    assert combined.confidence_cap_reasons == ("material_contradiction",)


def test_required_provider_failure_forces_context_and_combined_abstention() -> None:
    prepared = packet()
    prepared["retrieval"]["source_failures"] = [
        {"adapter_id": "gws", "reason_code": "unavailable", "required": True}
    ]
    person_id = prepared["people"][0]["person_id"]

    result = join(
        prepared,
        readout(prepared, person_id=person_id),
        bindings={"subject-alice": person_id},
    )

    evaluations = by_condition(result)
    assert evaluations["context_only"].outcome == "abstained"
    assert evaluations["context_only"].abstention_reason == "required_provider_failure"
    assert evaluations["acoustic_only"].outcome == "proposed"
    assert evaluations["combined"].outcome == "abstained"
    assert evaluations["combined"].abstention_reason == "required_provider_failure"


def test_explicit_source_label_binding_joins_letter_labels_to_acoustic_refs() -> None:
    prepared = packet(speakers=("A", "B"))
    person_id = prepared["people"][0]["person_id"]
    model_readout = readout(
        prepared,
        person_id=person_id,
        speaker_labels=("A", "B"),
    )

    result = context_join.join_contextual_identity(
        document_id="document-001",
        transcript_sha256=HASH_B,
        identity_packet=prepared,
        identity_readout=model_readout,
        acoustic_bundle=acoustic(speakers=("SPEAKER_1", "SPEAKER_2")),
        speaker_ref_bindings={"A": "SPEAKER_1", "B": "SPEAKER_2"},
        acoustic_subject_person_bindings={"subject-alice": person_id},
        evaluated_at=NOW,
    )

    assert [item.speaker_ref for item in result.review_outcomes] == [
        "SPEAKER_1",
        "SPEAKER_2",
    ]
    assert [item.source_speaker_label for item in result.review_outcomes] == [
        "A",
        "B",
    ]
    assert len(result.evaluations) == 6


def test_incomplete_source_label_binding_fails_closed() -> None:
    prepared = packet(speakers=("A", "B"))

    with pytest.raises(
        context_join.ContextualIdentityJoinError,
        match="Every prepared speaker label",
    ) as error:
        context_join.join_contextual_identity(
            document_id="document-001",
            transcript_sha256=HASH_B,
            identity_packet=prepared,
            identity_readout=readout(
                prepared,
                speaker_labels=("A", "B"),
            ),
            acoustic_bundle=acoustic(speakers=("SPEAKER_1", "SPEAKER_2")),
            speaker_ref_bindings={"A": "SPEAKER_1"},
            acoustic_subject_person_bindings={},
            evaluated_at=NOW,
        )

    assert error.value.reason_code == "speaker_ref_binding_mismatch"


def test_unprepared_model_reference_is_rejected_before_join() -> None:
    prepared = packet()
    model_readout = readout(prepared)
    model_readout["speaker_assignments"][0]["factors"][0]["evidence_ids"] = [
        "invented-evidence"
    ]

    with pytest.raises(ValueError, match="unprepared evidence"):
        join(prepared, model_readout)
