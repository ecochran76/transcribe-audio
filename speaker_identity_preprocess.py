#!/usr/bin/env python3
"""Build bounded, review-gated evidence packets for speaker identification."""
from __future__ import annotations

import json
import hashlib
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence
from uuid import uuid4

import provenance_config
from context_sources import (
    GwsProvenanceConfig,
    OdolloProvenanceConfig,
    collect_gws_provenance,
    collect_odollo_provenance,
)
from participant_identity import extract_calendar_attendees, normalize_email
from routing_artifacts import ProvenanceSource, normalize_string
from transcribe_common import TranscriptionError

SPEAKER_CLUE_PACKET_SCHEMA_VERSION = "transcribe-audio.speaker-clue-packet.v1"
SPEAKER_IDENTITY_READOUT_SCHEMA_VERSION = "transcribe-audio.speaker-identity-readout.v1"
CLUE_DISCOVERY_PACKET_SCHEMA_VERSION = "transcribe-audio.speaker-clue-discovery-packet.v1"
CLUE_DISCOVERY_READOUT_SCHEMA_VERSION = "transcribe-audio.speaker-clue-discovery-readout.v1"
IDENTITY_EVALUATION_PACKET_SCHEMA_VERSION = "transcribe-audio.speaker-identity-evaluation-packet.v1"
IDENTITY_EVALUATION_READOUT_SCHEMA_VERSION = "transcribe-audio.speaker-identity-evaluation-readout.v1"
MAX_UTTERANCES_PER_SPEAKER = 12
MAX_UTTERANCE_CHARS = 1_200
MAX_PROVENANCE_SOURCES = 24
MAX_PROVENANCE_SNIPPET_CHARS = 600
PERSON_PROVENANCE_TYPES = {
    "gws_contact",
    "gws_other_contact",
    "gws_directory_person",
    "odollo_contact",
    "odollo_lead",
}
EVIDENCE_STRENGTH_POINTS = {
    "weak": 15,
    "moderate": 30,
    "strong": 50,
    "decisive": 70,
}
EVIDENCE_SCORE_CONTRACT = {
    "meaning": "rubric-based evidence strength, not probability",
    "strength_points": EVIDENCE_STRENGTH_POINTS,
    "direction_rule": "support adds, contradict subtracts, neutral contributes zero",
    "independence_rule": (
        "within one host-derived independence group, only the strongest assessment "
        "in each direction counts"
    ),
    "bands": {
        "low": {"minimum": 0, "maximum": 24},
        "medium": {"minimum": 25, "maximum": 59},
        "high": {"minimum": 60, "maximum": 84},
        "very_high": {"minimum": 85, "maximum": 100},
    },
}
EVIDENCE_RUBRICS = {
    "calendar_association": {
        "version": "calendar-association.v1",
        "question": "Does this calendar event describe the recorded conversation?",
        "factors": [
            "time_window_alignment",
            "event_title_topic_alignment",
            "attendee_or_organization_alignment",
            "explicit_contradiction_or_competing_event",
        ],
        "score_contract": EVIDENCE_SCORE_CONTRACT,
    },
    "person_link": {
        "version": "person-link.v1",
        "question": "Do these source records represent the same real person?",
        "factors": [
            "normalized_email_match",
            "name_and_organization_alignment",
            "cross_source_relationship_context",
            "identifier_or_role_contradiction",
        ],
        "score_contract": EVIDENCE_SCORE_CONTRACT,
    },
    "speaker_identity": {
        "version": "speaker-identity.v1",
        "question": "Does this person correspond to this diarized speaker or speaker group?",
        "factors": [
            "direct_self_identification",
            "verified_identifier_match",
            "addressed_by_name",
            "calendar_attendee_topic_alignment",
            "role_or_relationship_alignment",
            "speaker_mixing_or_contradiction",
        ],
        "score_contract": EVIDENCE_SCORE_CONTRACT,
    },
}

SPEAKER_IDENTITY_OUTPUT_SCHEMA = {
    "schema_version": SPEAKER_IDENTITY_READOUT_SCHEMA_VERSION,
    "speakers": [
        {
            "speaker_label": "prepared speaker label",
            "status": "proposed|unresolved|conflicting",
            "candidate_id": "prepared contact_id or empty",
            "suggested_person": {"name": "", "email": ""},
            "confidence": 0.0,
            "transcript_clue_ids": ["prepared utterance_id"],
            "provenance_source_ids": ["prepared source_id"],
            "rationale": "bounded explanation",
            "alternatives": [],
            "review_flags": [],
        }
    ],
    "warnings": [],
}


def _compact_candidate(value: dict[str, Any]) -> dict[str, Any]:
    return {
        "contact_id": normalize_string(value.get("contact_id") or value.get("id")),
        "label": normalize_string(value.get("label") or value.get("name")),
        "email": normalize_email(value.get("email")),
        "source_profile": normalize_string(value.get("source_profile") or value.get("profile")),
        "confidence": value.get("confidence"),
    }


def _ordered_candidates(
    candidates: Iterable[dict[str, Any]],
    attendees: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    attendee_order = {
        attendee["email"]: index
        for index, attendee in enumerate(attendees)
        if attendee.get("email")
    }
    compact = [_compact_candidate(item) for item in candidates if isinstance(item, dict)]
    return sorted(
        compact,
        key=lambda item: (
            0 if item["email"] in attendee_order else 1,
            attendee_order.get(item["email"], len(attendee_order)),
            -float(item["confidence"] or 0.0),
            item["label"].lower(),
        ),
    )


def _speaker_clues(transcript: dict[str, Any]) -> list[dict[str, Any]]:
    by_speaker: dict[str, list[dict[str, Any]]] = {}
    utterances = transcript.get("utterances")
    if not isinstance(utterances, list):
        return []
    for index, utterance in enumerate(utterances):
        if not isinstance(utterance, dict):
            continue
        speaker = normalize_string(utterance.get("speaker")) or "Speaker"
        clues = by_speaker.setdefault(speaker, [])
        if len(clues) >= MAX_UTTERANCES_PER_SPEAKER:
            continue
        text = normalize_string(utterance.get("text"))[:MAX_UTTERANCE_CHARS]
        if not text:
            continue
        clues.append(
            {
                "utterance_id": f"utterance-{index + 1}",
                "start": utterance.get("start"),
                "end": utterance.get("end"),
                "text": text,
            }
        )
    return [
        {"speaker_label": speaker, "utterance_clues": clues}
        for speaker, clues in by_speaker.items()
    ]


def build_clue_discovery_packet(
    *,
    transcript: dict[str, Any],
    source_contexts: Iterable[dict[str, Any]] = (),
) -> dict[str, Any]:
    """Build the transcript-only first pass used to discover retrieval clues."""
    event = transcript.get("event") if isinstance(transcript.get("event"), dict) else {}
    recording_ids = transcript.get("recording_ids")
    if not isinstance(recording_ids, list):
        recording_id = normalize_string(transcript.get("recording_id"))
        recording_ids = [recording_id] if recording_id else []
    return {
        "schema_version": CLUE_DISCOVERY_PACKET_SCHEMA_VERSION,
        "task": "speaker_clue_discovery",
        "conversation": {
            "conversation_id": normalize_string(transcript.get("conversation_id")),
            "recording_ids": [
                normalize_string(value)
                for value in recording_ids
                if normalize_string(value)
            ],
            "title": normalize_string(
                transcript.get("transcript_title") or event.get("summary")
            ),
        },
        "calendar_context": {
            "event_id": normalize_string(event.get("id") or event.get("event_id")),
            "title": normalize_string(event.get("summary")),
            "description": normalize_string(event.get("description"))[:MAX_PROVENANCE_SNIPPET_CHARS],
            "attendees": extract_calendar_attendees(transcript),
        },
        "source_contexts": [
            dict(item)
            for item in source_contexts
            if isinstance(item, dict)
        ],
        "speakers": _speaker_clues(transcript),
        "policy": {
            "retrieval_is_host_owned": True,
            "identify_people_in_this_pass": False,
            "defer_full_contextual_readout": True,
        },
    }


def validate_clue_discovery_readout(
    packet: dict[str, Any],
    readout: dict[str, Any],
) -> dict[str, Any]:
    """Reject discovery output that cites transcript material not in the packet."""
    if readout.get("schema_version") != CLUE_DISCOVERY_READOUT_SCHEMA_VERSION:
        raise ValueError(
            "Clue discovery readout schema_version must be "
            f"{CLUE_DISCOVERY_READOUT_SCHEMA_VERSION}."
        )
    prepared_by_speaker = {
        normalize_string(item.get("speaker_label")): {
            normalize_string(clue.get("utterance_id"))
            for clue in item.get("utterance_clues", [])
            if isinstance(clue, dict)
        }
        for item in packet.get("speakers", [])
        if isinstance(item, dict)
    }
    all_prepared_clues = set().union(*prepared_by_speaker.values()) if prepared_by_speaker else set()

    speaker_clues = readout.get("speaker_clues")
    conversation_clues = readout.get("conversation_clues")
    if not isinstance(speaker_clues, list) or not isinstance(conversation_clues, list):
        raise ValueError("Clue discovery readout clue collections must be lists.")

    for result in speaker_clues:
        if not isinstance(result, dict):
            raise ValueError("Clue discovery readout contains a non-object speaker clue.")
        label = normalize_string(result.get("speaker_label"))
        if label not in prepared_by_speaker:
            raise ValueError(f"Speaker clue references unprepared speaker: {label or '<empty>'}.")
        clue_ids = {
            normalize_string(value)
            for value in result.get("transcript_clue_ids", [])
            if normalize_string(value)
        }
        unknown = clue_ids - prepared_by_speaker[label]
        if unknown:
            raise ValueError(f"Speaker clue references unprepared transcript clues: {sorted(unknown)}.")

    for result in conversation_clues:
        if not isinstance(result, dict):
            raise ValueError("Clue discovery readout contains a non-object conversation clue.")
        clue_ids = {
            normalize_string(value)
            for value in result.get("transcript_clue_ids", [])
            if normalize_string(value)
        }
        unknown = clue_ids - all_prepared_clues
        if unknown:
            raise ValueError(
                f"Conversation clue references unprepared transcript clues: {sorted(unknown)}."
            )
    for group_hint in readout.get("speaker_group_hints", []):
        if not isinstance(group_hint, dict):
            raise ValueError("Clue discovery contains a non-object speaker group hint.")
        labels = {
            normalize_string(value)
            for value in group_hint.get("speaker_labels", [])
            if normalize_string(value)
        }
        if len(labels) < 2 or labels - set(prepared_by_speaker):
            raise ValueError(
                f"Speaker group hint references unprepared speakers: {sorted(labels)}."
            )
        clue_ids = {
            normalize_string(value)
            for value in group_hint.get("transcript_clue_ids", [])
            if normalize_string(value)
        }
        if clue_ids - all_prepared_clues:
            raise ValueError("Speaker group hint references unprepared transcript clues.")
    for mixed_hint in readout.get("mixed_speaker_hints", []):
        if not isinstance(mixed_hint, dict):
            raise ValueError("Clue discovery contains a non-object mixed speaker hint.")
        label = normalize_string(mixed_hint.get("speaker_label"))
        if label not in prepared_by_speaker:
            raise ValueError(f"Mixed speaker hint references unprepared speaker: {label}.")
        clue_ids = {
            normalize_string(value)
            for value in mixed_hint.get("transcript_clue_ids", [])
            if normalize_string(value)
        }
        if clue_ids - prepared_by_speaker[label]:
            raise ValueError("Mixed speaker hint references unprepared transcript clues.")
    return {"valid": True, "readout": readout}


def _person_group_key(value: dict[str, Any]) -> str:
    email = normalize_email(value.get("email"))
    if email:
        return f"email:{email}"
    source_id = normalize_string(value.get("source_id") or value.get("source_profile"))
    record_id = normalize_string(value.get("contact_id") or value.get("id"))
    return f"record:{source_id}:{record_id}"


def group_person_candidates(values: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Group cross-source records without losing their relationship provenance."""
    grouped: dict[str, dict[str, Any]] = {}
    for value in values:
        if not isinstance(value, dict):
            continue
        key = _person_group_key(value)
        if key == "record::":
            continue
        email = normalize_email(value.get("email"))
        label = normalize_string(value.get("label") or value.get("name"))
        source_id = normalize_string(value.get("source_id") or value.get("source_profile"))
        record_id = normalize_string(value.get("contact_id") or value.get("id"))
        person = grouped.setdefault(
            key,
            {
                "person_id": "person-" + hashlib.sha256(key.encode("utf-8")).hexdigest()[:16],
                "display_name": label,
                "emails": [],
                "source_records": [],
            },
        )
        if label and len(label) > len(person["display_name"]):
            person["display_name"] = label
        if email and email not in person["emails"]:
            person["emails"].append(email)
        source_record = {
            "source_id": source_id,
            "source_type": normalize_string(value.get("source_type")),
            "record_id": record_id,
            "label": label,
            "email": email,
        }
        if source_record not in person["source_records"]:
            person["source_records"].append(source_record)
    return list(grouped.values())


def score_evidence_factors(task: str, factors: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Compute a stable score from model-assessed, source-cited factor strengths."""
    rubric = EVIDENCE_RUBRICS.get(task)
    if rubric is None:
        raise ValueError(f"Unknown evidence rubric: {task}.")
    strongest_by_independence_key: dict[str, dict[str, tuple[int, dict[str, Any]]]] = {}
    for factor in factors:
        if not isinstance(factor, dict):
            continue
        direction = normalize_string(factor.get("direction")).lower()
        strength = normalize_string(factor.get("strength")).lower()
        if direction not in {"support", "contradict", "neutral"}:
            raise ValueError(f"Invalid evidence direction: {direction or '<empty>'}.")
        if strength not in EVIDENCE_STRENGTH_POINTS:
            raise ValueError(f"Invalid evidence strength: {strength or '<empty>'}.")
        key = normalize_string(factor.get("independence_key"))
        if not key:
            raise ValueError("Evidence factor must have an independence_key.")
        signed_points = EVIDENCE_STRENGTH_POINTS[strength]
        if direction == "contradict":
            signed_points *= -1
        elif direction == "neutral":
            signed_points = 0
        direction_key = "support" if signed_points > 0 else "contradict" if signed_points < 0 else "neutral"
        by_direction = strongest_by_independence_key.setdefault(key, {})
        previous = by_direction.get(direction_key)
        if previous is None or abs(signed_points) > abs(previous[0]):
            by_direction[direction_key] = (signed_points, dict(factor))

    numeric = max(
        0,
        min(
            100,
            sum(
                sum(item[0] for item in by_direction.values())
                for by_direction in strongest_by_independence_key.values()
            ),
        ),
    )
    if numeric >= 85:
        band = "very_high"
    elif numeric >= 60:
        band = "high"
    elif numeric >= 25:
        band = "medium"
    else:
        band = "low"
    return {
        "rubric": task,
        "rubric_version": rubric["version"],
        "numeric": numeric,
        "band": band,
        "band_label": band.replace("_", " ").title(),
        "counted_independence_keys": list(strongest_by_independence_key),
        "counted_factors": [
            item[1]
            for by_direction in strongest_by_independence_key.values()
            for item in by_direction.values()
        ],
    }


def build_identity_evaluation_packet(
    *,
    transcript: dict[str, Any],
    discovery_readout: dict[str, Any],
    person_records: Iterable[dict[str, Any]] = (),
    provenance_sources: Iterable[dict[str, Any]] = (),
    source_contexts: Iterable[dict[str, Any]] = (),
) -> dict[str, Any]:
    """Build the second pass after host-owned retrieval has completed."""
    discovery_packet = build_clue_discovery_packet(
        transcript=transcript,
        source_contexts=source_contexts,
    )
    validate_clue_discovery_readout(discovery_packet, discovery_readout)
    return {
        "schema_version": IDENTITY_EVALUATION_PACKET_SCHEMA_VERSION,
        "evaluation_id": str(uuid4()),
        "task": "speaker_identity_evaluation",
        "conversation": discovery_packet["conversation"],
        "calendar_context": discovery_packet["calendar_context"],
        "source_contexts": discovery_packet["source_contexts"],
        "speakers": discovery_packet["speakers"],
        "discovery_readout": discovery_readout,
        "people": group_person_candidates(person_records),
        "provenance_sources": _compact_provenance_sources(
            _source_dict(item) for item in provenance_sources
        ),
        "rubrics": EVIDENCE_RUBRICS,
        "policy": {
            "requires_human_confirmation": True,
            "will_apply_assignments": False,
            "model_must_not_emit_numeric_confidence": True,
            "host_computes_scores_from_factor_assessments": True,
            "allow_unlisted_person_proposals": True,
            "allow_speaker_groups_and_mixed_speaker_findings": True,
        },
    }


def _prepared_identity_references(
    packet: dict[str, Any],
) -> tuple[set[str], set[str], set[str], dict[str, str]]:
    speaker_labels: set[str] = set()
    evidence_ids: set[str] = set()
    person_ids: set[str] = set()
    independence_keys: dict[str, str] = {}
    conversation = (
        packet.get("conversation")
        if isinstance(packet.get("conversation"), dict)
        else {}
    )
    conversation_id = normalize_string(conversation.get("conversation_id"))
    if conversation_id:
        evidence_ids.add(conversation_id)
        independence_keys[conversation_id] = f"conversation:{conversation_id}"
    for recording_id_value in conversation.get("recording_ids", []):
        recording_id = normalize_string(recording_id_value)
        if recording_id:
            evidence_ids.add(recording_id)
            independence_keys[recording_id] = f"recording:{recording_id}"
    for speaker in packet.get("speakers", []):
        if not isinstance(speaker, dict):
            continue
        label = normalize_string(speaker.get("speaker_label"))
        if label:
            speaker_labels.add(label)
        for clue in speaker.get("utterance_clues", []):
            if not isinstance(clue, dict):
                continue
            clue_id = normalize_string(clue.get("utterance_id"))
            if clue_id:
                evidence_ids.add(clue_id)
                independence_keys[clue_id] = f"transcript:{clue_id}"
    calendar_context = (
        packet.get("calendar_context")
        if isinstance(packet.get("calendar_context"), dict)
        else {}
    )
    event_id = normalize_string(calendar_context.get("event_id"))
    if event_id:
        evidence_ids.add(event_id)
        independence_keys[event_id] = f"calendar:{event_id}"
    calendar_anchor = f"calendar:{event_id}" if event_id else "calendar:prepared-attendees"
    for attendee in calendar_context.get("attendees", []):
        if not isinstance(attendee, dict):
            continue
        attendee_id = normalize_string(attendee.get("id"))
        if attendee_id:
            evidence_ids.add(attendee_id)
            independence_keys[attendee_id] = calendar_anchor
    for source_context in packet.get("source_contexts", []):
        if not isinstance(source_context, dict):
            continue
        source_context_id = normalize_string(source_context.get("source_id"))
        if source_context_id:
            evidence_ids.add(source_context_id)
            independence_keys[source_context_id] = (
                f"source-context:{source_context_id}"
            )
    for person in packet.get("people", []):
        if not isinstance(person, dict):
            continue
        person_id = normalize_string(person.get("person_id"))
        if person_id:
            person_ids.add(person_id)
            evidence_ids.add(person_id)
            independence_keys[person_id] = f"person:{person_id}"
        person_anchor = f"person:{person_id}" if person_id else "person:prepared"
        for email_value in person.get("emails", []):
            email = normalize_email(email_value)
            if email:
                evidence_ids.add(email)
                independence_keys[email] = person_anchor
        for record in person.get("source_records", []):
            if not isinstance(record, dict):
                continue
            record_id = normalize_string(record.get("record_id"))
            source_id = normalize_string(record.get("source_id"))
            if record_id:
                evidence_ids.add(record_id)
                independence_keys[record_id] = f"source-record:{source_id}:{record_id}"
    for source in packet.get("provenance_sources", []):
        if not isinstance(source, dict):
            continue
        source_id = normalize_string(source.get("source_id"))
        if source_id:
            evidence_ids.add(source_id)
            independence_keys[source_id] = normalize_string(
                source.get("independence_key")
            ) or f"source:{source_id}"
    return speaker_labels, evidence_ids, person_ids, independence_keys


def _validated_factor_assessments(
    task: str,
    values: Any,
    *,
    evidence_ids: set[str],
    independence_keys: dict[str, str],
) -> list[dict[str, Any]]:
    if not isinstance(values, list):
        raise ValueError(f"{task} factors must be a list.")
    allowed_factors = set(EVIDENCE_RUBRICS[task]["factors"])
    result: list[dict[str, Any]] = []
    for value in values:
        if not isinstance(value, dict):
            raise ValueError(f"{task} contains a non-object factor.")
        factor_name = normalize_string(value.get("factor"))
        if factor_name not in allowed_factors:
            raise ValueError(f"{task} references an unknown rubric factor: {factor_name}.")
        cited = [
            normalize_string(item)
            for item in value.get("evidence_ids", [])
            if normalize_string(item)
        ]
        unknown = set(cited) - evidence_ids
        if unknown:
            raise ValueError(f"{task} factor references unprepared evidence: {sorted(unknown)}.")
        if not cited:
            raise ValueError(f"{task} factor must cite prepared evidence.")
        enriched = dict(value)
        enriched["evidence_ids"] = cited
        non_transcript_citations = [
            item for item in cited if not independence_keys[item].startswith("transcript:")
        ]
        anchor = non_transcript_citations[0] if non_transcript_citations else cited[0]
        enriched["independence_key"] = independence_keys[anchor]
        result.append(enriched)
    return result


def validate_and_score_identity_evaluation(
    packet: dict[str, Any],
    readout: dict[str, Any],
) -> dict[str, Any]:
    """Strictly validate model assessments and attach host-computed confidence."""
    if readout.get("schema_version") != IDENTITY_EVALUATION_READOUT_SCHEMA_VERSION:
        raise ValueError(
            "Identity evaluation readout schema_version must be "
            f"{IDENTITY_EVALUATION_READOUT_SCHEMA_VERSION}."
        )
    if normalize_string(readout.get("evaluation_id")) != normalize_string(
        packet.get("evaluation_id")
    ):
        raise ValueError("Identity evaluation readout references an unprepared evaluation_id.")
    speaker_labels, evidence_ids, person_ids, independence_keys = (
        _prepared_identity_references(packet)
    )
    result = json.loads(json.dumps(readout))

    calendar_association = result.get("calendar_association")
    if not isinstance(calendar_association, dict):
        raise ValueError("Identity evaluation calendar_association must be an object.")
    calendar_status = normalize_string(calendar_association.get("status"))
    if calendar_status not in {"matched", "unmatched", "ambiguous"}:
        raise ValueError(
            f"Calendar association has invalid status: {calendar_status or '<empty>'}."
        )
    calendar_factors = _validated_factor_assessments(
        "calendar_association",
        calendar_association.get("factors"),
        evidence_ids=evidence_ids,
        independence_keys=independence_keys,
    )
    calendar_association["factors"] = calendar_factors
    calendar_association["confidence"] = score_evidence_factors(
        "calendar_association",
        calendar_factors,
    )

    person_links = result.get("person_links")
    assignments = result.get("speaker_assignments")
    if not isinstance(person_links, list) or not isinstance(assignments, list):
        raise ValueError("Identity evaluation person_links and speaker_assignments must be lists.")
    for link in person_links:
        if not isinstance(link, dict):
            raise ValueError("Identity evaluation contains a non-object person link.")
        linked_person_ids = [
            normalize_string(value)
            for value in link.get("person_ids", [])
            if normalize_string(value)
        ]
        if len(linked_person_ids) < 2 or set(linked_person_ids) - person_ids:
            raise ValueError(
                f"Person link references unprepared person_ids: {linked_person_ids}."
            )
        link_status = normalize_string(link.get("status"))
        if link_status not in {"same_person", "different_people", "uncertain"}:
            raise ValueError(f"Person link has invalid status: {link_status or '<empty>'}.")
        link["person_ids"] = linked_person_ids
        factors = _validated_factor_assessments(
            "person_link",
            link.get("factors"),
            evidence_ids=evidence_ids,
            independence_keys=independence_keys,
        )
        link["factors"] = factors
        link["confidence"] = score_evidence_factors("person_link", factors)

    for assignment in assignments:
        if not isinstance(assignment, dict):
            raise ValueError("Identity evaluation contains a non-object speaker assignment.")
        labels = [
            normalize_string(value)
            for value in assignment.get("speaker_labels", [])
            if normalize_string(value)
        ]
        unknown_labels = set(labels) - speaker_labels
        if not labels or unknown_labels:
            raise ValueError(
                f"Speaker assignment references unprepared speakers: {sorted(unknown_labels)}."
            )
        status = normalize_string(assignment.get("status"))
        if status not in {"candidate_match", "unlisted", "unresolved", "conflicting"}:
            raise ValueError(f"Speaker assignment has invalid status: {status or '<empty>'}.")
        person_id = normalize_string(assignment.get("person_id"))
        if person_id and person_id not in person_ids:
            raise ValueError(f"Speaker assignment references unprepared person_id: {person_id}.")
        if status == "candidate_match" and not person_id:
            raise ValueError("Candidate Match speaker assignment must reference a person_id.")
        if status == "unlisted":
            suggested = (
                assignment.get("suggested_person")
                if isinstance(assignment.get("suggested_person"), dict)
                else {}
            )
            if not any(
                normalize_string(suggested.get(key))
                for key in ("name", "email", "organization")
            ):
                raise ValueError("Unlisted speaker assignment requires a suggested_person.")
        prepared_utterance_ids = {
            item
            for item, independence_key in independence_keys.items()
            if independence_key.startswith("transcript:")
        }
        transcript_clue_ids = {
            normalize_string(value)
            for value in assignment.get("transcript_clue_ids", [])
            if normalize_string(value)
        }
        unknown_transcript_clues = transcript_clue_ids - prepared_utterance_ids
        if unknown_transcript_clues:
            raise ValueError(
                "Speaker assignment references unprepared transcript clues: "
                f"{sorted(unknown_transcript_clues)}."
            )
        prepared_source_ids = {
            normalize_string(source.get("source_id"))
            for source in packet.get("provenance_sources", [])
            if isinstance(source, dict) and normalize_string(source.get("source_id"))
        }
        prepared_source_ids.update(
            normalize_string(source_context.get("source_id"))
            for source_context in packet.get("source_contexts", [])
            if isinstance(source_context, dict)
            and normalize_string(source_context.get("source_id"))
        )
        prepared_source_ids.update(
            normalize_string(record.get("source_id"))
            for person in packet.get("people", [])
            if isinstance(person, dict)
            for record in person.get("source_records", [])
            if isinstance(record, dict) and normalize_string(record.get("source_id"))
        )
        provenance_source_ids = {
            normalize_string(value)
            for value in assignment.get("provenance_source_ids", [])
            if normalize_string(value)
        }
        unknown_sources = provenance_source_ids - prepared_source_ids
        if unknown_sources:
            raise ValueError(
                f"Speaker assignment references unprepared provenance sources: {sorted(unknown_sources)}."
            )
        factors = _validated_factor_assessments(
            "speaker_identity",
            assignment.get("factors"),
            evidence_ids=evidence_ids,
            independence_keys=independence_keys,
        )
        assignment["factors"] = factors
        assignment["confidence"] = score_evidence_factors("speaker_identity", factors)
        for utterance_assignment in assignment.get("utterance_assignments", []):
            if not isinstance(utterance_assignment, dict):
                raise ValueError("Speaker assignment contains a non-object utterance assignment.")
            utterance_id = normalize_string(utterance_assignment.get("utterance_id"))
            if utterance_id not in evidence_ids or not utterance_id.startswith("utterance-"):
                raise ValueError(
                    f"Utterance assignment references unprepared evidence: {utterance_id}."
                )
            utterance_person_id = normalize_string(utterance_assignment.get("person_id"))
            if utterance_person_id and utterance_person_id not in person_ids:
                raise ValueError(
                    f"Utterance assignment references unprepared person_id: {utterance_person_id}."
                )

    safe_bulk_confirm_ready = bool(assignments) and all(
        assignment.get("status") == "candidate_match"
        and (assignment.get("confidence") or {}).get("numeric", 0) >= 85
        and not assignment.get("review_flags")
        for assignment in assignments
    )
    person_group_proposals = [
        {
            "person_ids": link["person_ids"],
            "status": (
                "ready_to_group"
                if link.get("status") == "same_person"
                and (link.get("confidence") or {}).get("numeric", 0) >= 60
                else "needs_review"
            ),
            "confidence": link.get("confidence") or {},
        }
        for link in person_links
        if isinstance(link, dict) and link.get("status") in {"same_person", "uncertain"}
    ]
    return {
        "valid": True,
        "requires_human_confirmation": True,
        "will_apply_assignments": False,
        "safe_bulk_confirm_ready": safe_bulk_confirm_ready,
        "person_group_proposals": person_group_proposals,
        "readout": result,
    }


def _compact_provenance_sources(values: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for value in values:
        if not isinstance(value, dict) or len(result) >= MAX_PROVENANCE_SOURCES:
            continue
        metadata = value.get("metadata") if isinstance(value.get("metadata"), dict) else {}
        result.append(
            {
                "source_type": normalize_string(value.get("source_type")),
                "source_id": normalize_string(value.get("source_id")),
                "label": normalize_string(value.get("label")),
                "snippet": normalize_string(value.get("snippet"))[:MAX_PROVENANCE_SNIPPET_CHARS],
                "profile": normalize_string(metadata.get("profile") or value.get("profile")),
                "tenant": normalize_string(metadata.get("tenant") or metadata.get("tenant_profile")),
                "email": normalize_email(metadata.get("email") or value.get("email")),
                "timestamp": normalize_string(metadata.get("timestamp") or value.get("timestamp")),
                "independence_key": normalize_string(
                    metadata.get("independence_key") or value.get("independence_key")
                ),
            }
        )
    return result


def _source_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, ProvenanceSource):
        return value.to_dict()
    return value if isinstance(value, dict) else {}


def collect_speaker_provenance(
    transcript: dict[str, Any],
    identity_bundle: dict[str, Any],
    *,
    discovery_readout: Optional[dict[str, Any]] = None,
    gws_configs: Sequence[GwsProvenanceConfig] = (),
    odollo_configs: Sequence[OdolloProvenanceConfig] = (),
) -> dict[str, Any]:
    """Collect bounded read-only evidence through host-owned adapters."""
    attendees = extract_calendar_attendees(transcript)
    candidates = identity_bundle.get("contact_candidates")
    if not isinstance(candidates, list):
        candidates = []
    candidate_people = [
        {
            "name": normalize_string(item.get("label") or item.get("name")),
            "email": normalize_email(item.get("email")),
        }
        for item in candidates if isinstance(item, dict)
    ]
    discovered_people: list[dict[str, str]] = []
    if isinstance(discovery_readout, dict):
        for speaker_clue in discovery_readout.get("speaker_clues", []):
            if not isinstance(speaker_clue, dict):
                continue
            for hint in speaker_clue.get("person_hints", []):
                if not isinstance(hint, dict):
                    continue
                person = {
                    "name": normalize_string(hint.get("name")),
                    "email": normalize_email(hint.get("email")),
                    "organization": normalize_string(hint.get("organization")),
                }
                if any(person.values()) and person not in discovered_people:
                    discovered_people.append(person)
    readout = {
        "title": normalize_string(transcript.get("transcript_title")),
        "participants": [
            {"name": item.get("name") or item.get("label") or "", "email": item.get("email") or ""}
            for item in attendees
        ]
        + candidate_people,
        "discovered_people": discovered_people,
    }
    readout["participants"].extend(discovered_people)
    sources: list[dict[str, Any]] = []
    warnings: list[str] = []
    for config in gws_configs:
        try:
            sources.extend(_source_dict(item) for item in collect_gws_provenance(transcript, readout, config=config))
        except (OSError, ValueError, TranscriptionError) as exc:
            warnings.append(f"GWS profile {config.profile_label or '<default>'} failed: {type(exc).__name__}")
    for config in odollo_configs:
        try:
            sources.extend(_source_dict(item) for item in collect_odollo_provenance(transcript, readout, config=config))
        except (OSError, ValueError, TranscriptionError) as exc:
            profile = config.profiles[0] if config.profiles else "<default>"
            warnings.append(f"Odollo profile {profile} failed: {type(exc).__name__}")
    return {"sources": sources, "warnings": warnings}


def collect_configured_identity_evidence(
    *,
    transcript: dict[str, Any],
    identity_bundle: dict[str, Any],
    discovery_readout: dict[str, Any],
    provenance_path: Optional[Path] = None,
    state_root: Optional[Path] = None,
    profile: Optional[str] = None,
) -> dict[str, Any]:
    """Run bounded host-owned retrieval after validated Clue Discovery."""
    discovery_packet = build_clue_discovery_packet(transcript=transcript)
    validate_clue_discovery_readout(discovery_packet, discovery_readout)
    configs = provenance_config.speaker_preprocessing_source_configs_from_provenance(
        path=provenance_path,
        state_root=state_root,
        profile=profile,
    )
    collected = collect_speaker_provenance(
        transcript,
        identity_bundle,
        discovery_readout=discovery_readout,
        gws_configs=configs.get("gws") or [],
        odollo_configs=configs.get("odollo") or [],
    )
    person_records = [
        {
            "contact_id": normalize_string(item.get("contact_id") or item.get("id")),
            "label": normalize_string(item.get("label") or item.get("name")),
            "email": normalize_email(item.get("email")),
            "source_id": normalize_string(item.get("source_id") or item.get("source_profile")),
            "source_type": normalize_string(item.get("source_type") or "prepared_contact_candidate"),
        }
        for item in identity_bundle.get("contact_candidates", [])
        if isinstance(item, dict)
    ]
    for source in collected.get("sources", []):
        if not isinstance(source, dict) or source.get("source_type") not in PERSON_PROVENANCE_TYPES:
            continue
        metadata = source.get("metadata") if isinstance(source.get("metadata"), dict) else {}
        person_records.append(
            {
                "contact_id": normalize_string(source.get("source_id")),
                "label": normalize_string(source.get("label")),
                "email": normalize_email(metadata.get("email") or source.get("email")),
                "source_id": normalize_string(
                    metadata.get("profile")
                    or metadata.get("tenant")
                    or metadata.get("tenant_profile")
                ),
                "source_type": normalize_string(source.get("source_type")),
            }
        )
    warnings = [
        normalize_string(item)
        for item in [*(configs.get("warnings") or []), *(collected.get("warnings") or [])]
        if normalize_string(item)
    ]
    return {
        "person_records": person_records,
        "provenance_sources": [
            _source_dict(item)
            for item in collected.get("sources", [])
            if isinstance(_source_dict(item), dict)
        ],
        "source_contexts": configs.get("source_contexts") or [],
        "warnings": warnings,
    }


def build_speaker_clue_packet(
    *,
    conversation_key: str,
    transcript: dict[str, Any],
    identity_bundle: dict[str, Any],
    provenance_sources: Iterable[dict[str, Any]] = (),
    source_contexts: Iterable[dict[str, Any]] = (),
) -> dict[str, Any]:
    """Return the bounded host-owned input for one App Intelligence clue pass."""
    attendees = extract_calendar_attendees(transcript)
    candidates = identity_bundle.get("contact_candidates")
    if not isinstance(candidates, list):
        candidates = []
    return {
        "schema_version": SPEAKER_CLUE_PACKET_SCHEMA_VERSION,
        "task": "speaker_disambiguation",
        "conversation": {
            "conversation_key": normalize_string(conversation_key),
            "source_document_id": normalize_string(identity_bundle.get("source_document_id")),
            "title": normalize_string(transcript.get("transcript_title")),
        },
        "calendar_attendees": attendees,
        "source_contexts": [
            dict(item)
            for item in source_contexts
            if isinstance(item, dict)
        ],
        "contact_candidates": _ordered_candidates(candidates, attendees),
        "speakers": _speaker_clues(transcript),
        "provenance_sources": _compact_provenance_sources(_source_dict(item) for item in provenance_sources),
        "policy": {
            "requires_human_review": True,
            "will_apply_assignments": False,
            "allow_unknown_person_suggestions": True,
            "defer_full_contextual_readout": True,
        },
    }


def build_configured_speaker_clue_packet(
    *,
    conversation_key: str,
    transcript: dict[str, Any],
    identity_bundle: dict[str, Any],
    provenance_path: Optional[Path] = None,
    state_root: Optional[Path] = None,
    profile: Optional[str] = None,
) -> dict[str, Any]:
    """Build a packet using the shared user-scoped context provenance profile."""
    configs = provenance_config.speaker_preprocessing_source_configs_from_provenance(
        path=provenance_path,
        state_root=state_root,
        profile=profile,
    )
    collected = collect_speaker_provenance(
        transcript,
        identity_bundle,
        gws_configs=configs.get("gws") or [],
        odollo_configs=configs.get("odollo") or [],
    )
    augmented_bundle = dict(identity_bundle)
    existing_candidates = identity_bundle.get("contact_candidates")
    candidate_values = [dict(item) for item in existing_candidates or [] if isinstance(item, dict)]
    seen_candidate_keys = {
        normalize_email(item.get("email")) or normalize_string(item.get("contact_id") or item.get("id"))
        for item in candidate_values
    }
    attendee_emails = {
        item.get("email")
        for item in extract_calendar_attendees(transcript)
        if item.get("email")
    }
    for source in collected["sources"]:
        if not isinstance(source, dict) or source.get("source_type") not in PERSON_PROVENANCE_TYPES:
            continue
        metadata = source.get("metadata") if isinstance(source.get("metadata"), dict) else {}
        email = normalize_email(metadata.get("email") or source.get("email"))
        contact_id = normalize_string(source.get("source_id"))
        key = email or contact_id
        if not key or key in seen_candidate_keys:
            continue
        candidate_values.append(
            {
                "contact_id": contact_id,
                "label": normalize_string(source.get("label")),
                "email": email,
                "source_profile": normalize_string(metadata.get("profile")),
                "confidence": 0.9 if email in attendee_emails else 0.65,
            }
        )
        seen_candidate_keys.add(key)
    augmented_bundle["contact_candidates"] = candidate_values
    packet = build_speaker_clue_packet(
        conversation_key=conversation_key,
        transcript=transcript,
        identity_bundle=augmented_bundle,
        provenance_sources=collected["sources"],
        source_contexts=configs.get("source_contexts") or [],
    )
    packet["collection_warnings"] = [
        normalize_string(item)
        for item in [*(configs.get("warnings") or []), *(collected.get("warnings") or [])]
        if normalize_string(item)
    ]
    return packet


def validate_speaker_identity_readout(
    packet: dict[str, Any],
    readout: dict[str, Any],
) -> dict[str, Any]:
    """Validate a model readout strictly against the host-prepared packet."""
    if readout.get("schema_version") != SPEAKER_IDENTITY_READOUT_SCHEMA_VERSION:
        raise ValueError(
            "Speaker identity readout schema_version must be "
            f"{SPEAKER_IDENTITY_READOUT_SCHEMA_VERSION}."
        )
    prepared_speakers = {
        normalize_string(item.get("speaker_label")): {
            normalize_string(clue.get("utterance_id"))
            for clue in item.get("utterance_clues", [])
            if isinstance(clue, dict)
        }
        for item in packet.get("speakers", [])
        if isinstance(item, dict)
    }
    prepared_candidates = {
        normalize_string(item.get("contact_id"))
        for item in packet.get("contact_candidates", [])
        if isinstance(item, dict) and normalize_string(item.get("contact_id"))
    }
    prepared_sources = {
        normalize_string(item.get("source_id"))
        for item in packet.get("provenance_sources", [])
        if isinstance(item, dict) and normalize_string(item.get("source_id"))
    }
    speakers = readout.get("speakers")
    if not isinstance(speakers, list):
        raise ValueError("Speaker identity readout speakers must be a list.")
    for result in speakers:
        if not isinstance(result, dict):
            raise ValueError("Speaker identity readout contains a non-object speaker result.")
        label = normalize_string(result.get("speaker_label"))
        if label not in prepared_speakers:
            raise ValueError(f"Speaker result references unprepared speaker: {label or '<empty>'}.")
        candidate_id = normalize_string(result.get("candidate_id"))
        if candidate_id and candidate_id not in prepared_candidates:
            raise ValueError(f"Speaker result references unprepared candidate_id: {candidate_id}.")
        clue_ids = {
            normalize_string(value)
            for value in result.get("transcript_clue_ids", [])
            if normalize_string(value)
        }
        unknown_clues = clue_ids - prepared_speakers[label]
        if unknown_clues:
            raise ValueError(f"Speaker result references unprepared transcript clues: {sorted(unknown_clues)}.")
        source_ids = {
            normalize_string(value)
            for value in result.get("provenance_source_ids", [])
            if normalize_string(value)
        }
        unknown_sources = source_ids - prepared_sources
        if unknown_sources:
            raise ValueError(f"Speaker result references unprepared provenance sources: {sorted(unknown_sources)}.")
        try:
            confidence = float(result.get("confidence"))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Speaker result for {label} has invalid confidence.") from exc
        if not 0.0 <= confidence <= 1.0:
            raise ValueError(f"Speaker result for {label} confidence must be between 0 and 1.")
    return {
        "valid": True,
        "requires_human_review": True,
        "will_apply_assignments": False,
        "readout": readout,
    }


def build_speaker_identity_prompt(packet: dict[str, Any]) -> str:
    """Render the reviewed JSON-only prompt for one speaker clue pass."""
    output_schema = json.dumps(SPEAKER_IDENTITY_OUTPUT_SCHEMA, sort_keys=True, ensure_ascii=False)
    packet_json = json.dumps(packet, sort_keys=True, ensure_ascii=False)
    return (
        "Identify the anonymous speakers using only the prepared transcript clues and provenance below. "
        "Do not summarize the conversation or perform the later contextual interpretation pass. "
        "Every identity proposal must cite prepared transcript_clue_ids and provenance_source_ids; "
        "never invent a speaker, candidate_id, or source_id. Mark uncertainty explicitly. "
        "Return JSON only, matching this output shape: "
        f"{output_schema}\n\nPrepared speaker clue packet:\n{packet_json}"
    )


def build_clue_discovery_prompt(packet: dict[str, Any]) -> str:
    """Render the JSON-only first-pass prompt."""
    output_schema = {
        "schema_version": CLUE_DISCOVERY_READOUT_SCHEMA_VERSION,
        "speaker_clues": [
            {
                "speaker_label": "prepared speaker label",
                "transcript_clue_ids": ["prepared utterance_id"],
                "observations": ["bounded identity-relevant observation"],
                "person_hints": [{"name": "", "email": "", "organization": ""}],
                "retrieval_terms": ["bounded term for host retrieval"],
            }
        ],
        "conversation_clues": [
            {
                "transcript_clue_ids": ["prepared utterance_id"],
                "observation": "bounded conversation-level identity clue",
                "retrieval_terms": [],
            }
        ],
        "speaker_group_hints": [
            {
                "speaker_labels": ["prepared speaker label", "prepared speaker label"],
                "transcript_clue_ids": ["prepared utterance_id"],
                "observation": "possible diarization split",
            }
        ],
        "mixed_speaker_hints": [
            {
                "speaker_label": "prepared speaker label",
                "transcript_clue_ids": ["prepared utterance_id"],
                "observation": "possible mixed diarization label",
            }
        ],
        "warnings": [],
    }
    return (
        "Find identity-relevant clues and bounded retrieval terms in the prepared transcript excerpts. "
        "Do not identify the speakers in this pass. Do not request or retrieve external data. "
        "Cite only prepared utterance_ids and return JSON only matching this output shape: "
        f"{json.dumps(output_schema, sort_keys=True, ensure_ascii=False)}\n\n"
        "Prepared clue discovery packet:\n"
        f"{json.dumps(packet, sort_keys=True, ensure_ascii=False)}"
    )


def build_identity_evaluation_prompt(packet: dict[str, Any]) -> str:
    """Render the JSON-only second-pass prompt."""
    output_schema = {
        "schema_version": IDENTITY_EVALUATION_READOUT_SCHEMA_VERSION,
        "evaluation_id": "prepared evaluation_id",
        "calendar_association": {
            "status": "matched|unmatched|ambiguous",
            "factors": [],
        },
        "person_links": [
            {
                "person_ids": ["prepared person_id", "prepared person_id"],
                "status": "same_person|different_people|uncertain",
                "factors": [],
            }
        ],
        "speaker_assignments": [
            {
                "speaker_labels": ["prepared speaker label"],
                "status": "candidate_match|unlisted|unresolved|conflicting",
                "person_id": "prepared person_id or empty",
                "suggested_person": {"name": "", "email": "", "organization": ""},
                "transcript_clue_ids": ["prepared utterance_id"],
                "provenance_source_ids": ["prepared source_id"],
                "factors": [
                    {
                        "factor": "rubric factor name",
                        "direction": "support|contradict|neutral",
                        "strength": "weak|moderate|strong|decisive",
                        "evidence_ids": ["prepared evidence id"],
                        "rationale": "bounded assessment",
                    }
                ],
                "utterance_assignments": [],
                "rationale": "bounded explanation",
                "review_flags": [],
            }
        ],
        "warnings": [],
    }
    return (
        "Evaluate calendar association, cross-source person links, and speaker identities using only "
        "the host-prepared evidence. The same person may span multiple diarization labels, and one "
        "label may contain mixed speakers; use grouped or utterance assignments when warranted. "
        "Do not emit numeric confidence; assess only rubric factor direction and strength because "
        "the host computes numeric scores and plain-English bands. Every assessment must cite prepared "
        "evidence IDs. Return JSON only. Speaker status must be "
        "candidate_match|unlisted|unresolved|conflicting. Output shape: "
        f"{json.dumps(output_schema, sort_keys=True, ensure_ascii=False)}\n\n"
        "Prepared identity evaluation packet:\n"
        f"{json.dumps(packet, sort_keys=True, ensure_ascii=False)}"
    )
