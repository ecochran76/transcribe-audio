"""Build and rehearse the reviewed Plan 0063 state transition on private copies.

The module deliberately has no live apply entry point.  It converts a complete
P4 submission into one deterministic canonical-person transition, then proves
the knowledge-store portion against a private SQLite backup and restores that
copy to its exact baseline bytes.  Biometric reference/profile materialization
is a later, exact-input step that consumes the same transition.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import sqlite3
import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping, Sequence
from uuid import UUID, uuid5

from acoustic_audio_derivatives import (
    canonical_artifact_hash,
    ensure_private_tree,
    read_private_object,
    require_private_file,
    sha256_file,
    write_immutable_private_json,
)
from conversation_knowledge_profiles import ConversationProfileProjector
from conversation_knowledge_store import (
    ConversationKnowledgeStore,
    ExternalIdentityRecord,
    ObservationRecord,
    PersonRecord,
    PersonSnapshot,
    SourceRecord,
)
import speaker_identity_orchestration as orchestration
import speaker_identity_plan0063_human_review as human_review
import transcript_store


TRANSITION_SCHEMA = "transcribe-audio.plan0063-reviewed-transition.v1"
REHEARSAL_SCHEMA = "transcribe-audio.plan0063-private-copy-rehearsal.v1"
REHEARSAL_RECEIPT_SCHEMA = (
    "transcribe-audio.plan0063-private-copy-rehearsal-receipt.v1"
)
DEFAULT_RUNTIME_ROOT = Path.home() / ".local/state/transcribe-audio/plan-0063"
PERSON_NAMESPACE = UUID("502e38b5-5e38-5f36-8432-ec1e76c6c8a7")
UTC_RE = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z")
SHA256_RE = re.compile(r"[a-f0-9]{64}")
MAX_ENROLLMENT_UNITS = 5


class Plan0063PrivateRehearsalError(ValueError):
    """Raised when a reviewed transition or private rehearsal is not exact."""


def _fail(message: str) -> None:
    raise Plan0063PrivateRehearsalError(message)


def _uuid(label: str) -> str:
    return str(uuid5(PERSON_NAMESPACE, label))


def _canonical_utc(value: Any) -> str:
    text = str(value or "")
    if not UTC_RE.fullmatch(text):
        _fail("The review time must be canonical UTC without fractional seconds.")
    return text


def _assert_content_hash(value: Mapping[str, Any], field: str) -> str:
    claimed = str(value.get("content_sha256") or "")
    core = {key: item for key, item in value.items() if key != "content_sha256"}
    if not SHA256_RE.fullmatch(claimed) or canonical_artifact_hash(core) != claimed:
        _fail(f"The {field} content hash is invalid.")
    return claimed


def _decision_map(
    submission: Mapping[str, Any], review_manifest: Mapping[str, Any]
) -> dict[str, str]:
    _assert_content_hash(submission, "review submission")
    if (
        submission.get("schema_version") != human_review.SUBMISSION_SCHEMA
        or submission.get("review_content_sha256")
        != review_manifest.get("content_sha256")
        or submission.get("live_mutation_count") != 0
    ):
        _fail("The review submission authority is invalid.")
    expected = {
        str(item["decision_key"]): set(item["choices"])
        for item in (
            *review_manifest.get("merge_reviews", []),
            *review_manifest.get("binding_reviews", []),
            *[
                window
                for person in review_manifest.get("source_reviews", [])
                for window in person.get("windows", [])
            ],
        )
    }
    raw = submission.get("decisions")
    if not isinstance(raw, list):
        _fail("The review submission decisions are unavailable.")
    decisions: dict[str, str] = {}
    for item in raw:
        if not isinstance(item, Mapping):
            _fail("A review decision is invalid.")
        key = str(item.get("decision_key") or "")
        decision = str(item.get("decision") or "")
        if key in decisions or key not in expected or decision not in expected[key]:
            _fail("A review decision is duplicated, unknown, or not allowlisted.")
        decisions[key] = decision
    if set(decisions) != set(expected):
        _fail("Every exact P4 decision is required.")
    return decisions


def _validated_inputs(
    *,
    review_manifest: Mapping[str, Any],
    reconciliation: Mapping[str, Any],
    feasibility: Mapping[str, Any],
    submission: Mapping[str, Any],
) -> dict[str, str]:
    if (
        review_manifest.get("schema_version") != human_review.REVIEW_SCHEMA
        or review_manifest.get("status") != "blank_human_review_pending"
        or review_manifest.get("decision_count") != 30
        or review_manifest.get("reconciliation_content_sha256")
        != reconciliation.get("content_sha256")
        or review_manifest.get("feasibility_content_sha256")
        != feasibility.get("content_sha256")
        or feasibility.get("reconciliation_content_sha256")
        != reconciliation.get("content_sha256")
        or any((review_manifest.get("negative_actions") or {}).values())
        or any((reconciliation.get("negative_actions") or {}).values())
        or any((feasibility.get("negative_actions") or {}).values())
    ):
        _fail("The P2/P3/P4 review authority is invalid.")
    _assert_content_hash(review_manifest, "P4 review")
    decisions = _decision_map(submission, review_manifest)
    if (
        len(reconciliation.get("slot_identities") or []) != 9
        or len(reconciliation.get("person_proposals") or []) != 6
        or len(reconciliation.get("merge_proposals") or []) != 3
        or len(reconciliation.get("voice_binding_proposals") or []) != 1
        or len(feasibility.get("person_source_proposals") or []) != 5
    ):
        _fail("The reviewed person or source denominator drifted.")
    return decisions


def _canonical_person_id(source_submission_sha256: str, slots: Sequence[str]) -> str:
    return _uuid(
        "canonical-person:"
        + source_submission_sha256
        + ":"
        + "|".join(sorted(slots))
    )


def build_reviewed_transition(
    *,
    review_manifest: Mapping[str, Any],
    reconciliation: Mapping[str, Any],
    feasibility: Mapping[str, Any],
    submission: Mapping[str, Any],
    reviewed_at: str,
) -> dict[str, Any]:
    """Resolve literal P4 decisions into one deterministic private apply plan."""

    decisions = _validated_inputs(
        review_manifest=review_manifest,
        reconciliation=reconciliation,
        feasibility=feasibility,
        submission=submission,
    )
    observed_at = _canonical_utc(reviewed_at)
    source_submission_sha256 = str(
        reconciliation.get("source_submission_sha256") or ""
    )
    if not SHA256_RE.fullmatch(source_submission_sha256):
        _fail("The source human-gold submission binding is invalid.")

    slot_rows: dict[str, dict[str, Any]] = {}
    for raw in reconciliation.get("slot_identities") or []:
        if not isinstance(raw, Mapping):
            _fail("A reviewed slot identity is invalid.")
        row = dict(raw)
        slot_id = str(row.get("slot_id") or "")
        slot_person_id = str(row.get("slot_person_id") or "")
        if (
            not slot_id
            or not slot_person_id
            or not str(row.get("name") or "").strip()
            or slot_id in slot_rows
        ):
            _fail("A reviewed named slot is incomplete or duplicated.")
        slot_rows[slot_id] = row

    merge_by_person = {
        str(item.get("proposed_person_id") or ""): dict(item)
        for item in reconciliation.get("merge_proposals") or []
        if isinstance(item, Mapping)
    }
    if len(merge_by_person) != 3:
        _fail("The merge-proposal mapping is invalid.")

    canonical_people: list[dict[str, Any]] = []
    slot_to_person: dict[str, str] = {}
    merge_outcomes: list[dict[str, Any]] = []
    for raw in reconciliation.get("person_proposals") or []:
        if not isinstance(raw, Mapping):
            _fail("A person proposal is invalid.")
        proposal = dict(raw)
        proposed_person_id = str(proposal.get("proposed_person_id") or "")
        member_slots = [str(value) for value in proposal.get("member_slot_ids") or []]
        member_slot_people = [
            str(value) for value in proposal.get("member_slot_person_ids") or []
        ]
        if (
            not proposed_person_id
            or not member_slots
            or len(member_slots) != len(member_slot_people)
            or any(slot not in slot_rows for slot in member_slots)
        ):
            _fail("A person proposal lost its reviewed slot bindings.")
        merge = merge_by_person.get(proposed_person_id)
        if merge is None:
            if len(member_slots) != 1:
                _fail("A multi-slot person proposal lacks a merge decision.")
            groups = [member_slots]
            merge_decision = "not_applicable"
        else:
            key = f"MERGE::{merge.get('merge_proposal_id')}"
            merge_decision = decisions.get(key, "")
            if merge_decision == "accept":
                groups = [member_slots]
            elif merge_decision == "reject":
                groups = [[slot] for slot in member_slots]
            else:
                _fail("A person merge decision is unavailable.")
            merge_outcomes.append(
                {
                    "merge_proposal_id": merge.get("merge_proposal_id"),
                    "proposed_person_id": proposed_person_id,
                    "decision": merge_decision,
                    "basis": merge.get("basis"),
                    "member_slot_ids": member_slots,
                }
            )
        for group in groups:
            rows = [slot_rows[slot] for slot in group]
            names = {" ".join(str(row.get("name") or "").split()) for row in rows}
            if len(names) != 1 or "" in names:
                _fail("Merged reviewed slots do not have one canonical name.")
            person_id = _canonical_person_id(source_submission_sha256, group)
            if any(slot in slot_to_person for slot in group):
                _fail("A reviewed slot maps to more than one canonical person.")
            for slot in group:
                slot_to_person[slot] = person_id
            external_identities = sorted(
                {
                    ("email", str(row.get("email") or "").strip().casefold())
                    for row in rows
                    if str(row.get("email") or "").strip()
                }
            )
            canonical_people.append(
                {
                    "person_id": person_id,
                    "primary_name": names.pop(),
                    "member_slot_ids": sorted(group),
                    "member_slot_person_ids": sorted(
                        str(slot_rows[slot]["slot_person_id"]) for slot in group
                    ),
                    "external_identities": [
                        {"kind": kind, "value": value}
                        for kind, value in external_identities
                    ],
                    "organizations": sorted(
                        {
                            " ".join(str(row.get("organization") or "").split())
                            for row in rows
                            if str(row.get("organization") or "").strip()
                        }
                    ),
                    "proposal_source_id": proposed_person_id,
                    "merge_decision": merge_decision,
                }
            )

    if set(slot_to_person) != set(slot_rows):
        _fail("The canonical person map does not cover all reviewed named slots.")

    slot_bindings = []
    for slot_id in sorted(slot_rows):
        document_id, speaker_label = slot_id.split("::", 1)
        slot_bindings.append(
            {
                "slot_id": slot_id,
                "document_id": document_id,
                "speaker_label_id": speaker_label,
                "person_id": slot_to_person[slot_id],
                "slot_person_id": slot_rows[slot_id]["slot_person_id"],
                "decision_type": slot_rows[slot_id].get("decision_type"),
                "email": str(slot_rows[slot_id].get("email") or "")
                .strip()
                .casefold(),
                "organization": " ".join(
                    str(slot_rows[slot_id].get("organization") or "").split()
                ),
            }
        )

    voice_outcomes = []
    for raw in reconciliation.get("voice_binding_proposals") or []:
        if not isinstance(raw, Mapping):
            _fail("The voice binding proposal is invalid.")
        binding = dict(raw)
        key = f"BINDING::{binding.get('binding_proposal_id')}"
        decision = decisions.get(key, "")
        slot_id = str(binding.get("slot_id") or "")
        if decision not in {"same_person", "different_person"} or slot_id not in slot_to_person:
            _fail("The voice/person decision lost its reviewed person binding.")
        voice_outcomes.append(
            {
                "binding_proposal_id": binding.get("binding_proposal_id"),
                "decision": decision,
                "active_binding": decision == "same_person",
                "acoustic_subject_id": binding.get("acoustic_subject_id"),
                "person_id": slot_to_person[slot_id],
                "slot_id": slot_id,
            }
        )

    feasibility_windows: dict[str, dict[str, Any]] = {}
    proposal_slots: dict[str, set[str]] = {}
    for raw in feasibility.get("person_source_proposals") or []:
        if not isinstance(raw, Mapping):
            _fail("An enrollment source proposal is invalid.")
        proposed_person_id = str(raw.get("proposed_person_id") or "")
        proposal_slots[proposed_person_id] = {
            str(value) for value in raw.get("member_slot_ids") or []
        }
        for window in raw.get("source_windows") or []:
            if not isinstance(window, Mapping):
                _fail("An enrollment source window is invalid.")
            reference_id = str(window.get("reference_id") or "")
            if reference_id in feasibility_windows:
                _fail("An enrollment source reference is duplicated.")
            feasibility_windows[reference_id] = dict(window)

    included_sources: list[dict[str, Any]] = []
    excluded_source_count = 0
    for raw_person in review_manifest.get("source_reviews") or []:
        if not isinstance(raw_person, Mapping):
            _fail("A P4 source-review group is invalid.")
        proposed_person_id = str(raw_person.get("proposed_person_id") or "")
        if set(raw_person.get("member_slot_ids") or []) != proposal_slots.get(
            proposed_person_id
        ):
            _fail("A source-review group lost its P3 proposal binding.")
        for raw_window in raw_person.get("windows") or []:
            if not isinstance(raw_window, Mapping):
                _fail("A P4 source-review window is invalid.")
            reference_id = str(raw_window.get("reference_id") or "")
            window = feasibility_windows.get(reference_id)
            decision = decisions.get(f"SOURCE::{reference_id}", "")
            if (
                window is None
                or decision not in {"include", "exclude"}
                or any(
                    raw_window.get(field) != window.get(field)
                    for field in (
                        "slot_id",
                        "speaker_label_id",
                        "start_seconds",
                        "end_seconds",
                        "source_sha256",
                    )
                )
            ):
                _fail("A P4 source decision lost its exact P3 window binding.")
            if decision == "exclude":
                excluded_source_count += 1
                continue
            slot_id = str(window.get("slot_id") or "")
            if slot_id not in slot_to_person:
                _fail("An included source has no reviewed canonical person.")
            included_sources.append(
                {
                    **window,
                    "person_id": slot_to_person[slot_id],
                    "proposed_person_id": proposed_person_id,
                    "decision": "include",
                }
            )

    enrollment_units = []
    candidate_proposals = set(proposal_slots)
    for person in sorted(canonical_people, key=lambda item: item["person_id"]):
        if person["proposal_source_id"] not in candidate_proposals:
            continue
        sources = sorted(
            (
                source
                for source in included_sources
                if source["person_id"] == person["person_id"]
            ),
            key=lambda item: item["reference_id"],
        )
        source_identity = [
            {
                key: source.get(key)
                for key in (
                    "reference_id",
                    "source_sha256",
                    "slot_id",
                    "start_seconds",
                    "end_seconds",
                )
            }
            for source in sources
        ]
        enrollment_units.append(
            {
                "person_id": person["person_id"],
                "primary_name": person["primary_name"],
                "member_slot_ids": person["member_slot_ids"],
                "status": (
                    "source_selected"
                    if sources
                    else "ineligible_no_selected_source"
                ),
                "source_count": len(sources),
                "source_set_sha256": canonical_artifact_hash(source_identity),
                "sources": sources,
            }
        )

    within_bound = len(enrollment_units) <= MAX_ENROLLMENT_UNITS
    core = {
        "schema_version": TRANSITION_SCHEMA,
        "status": (
            "reviewed_transition_ready_for_private_rehearsal"
            if within_bound
            else "reviewed_transition_exceeds_enrollment_bound"
        ),
        "reviewed_at": observed_at,
        "source_submission_sha256": source_submission_sha256,
        "review_content_sha256": review_manifest["content_sha256"],
        "review_submission_sha256": submission["content_sha256"],
        "reconciliation_content_sha256": reconciliation["content_sha256"],
        "feasibility_content_sha256": feasibility["content_sha256"],
        "canonical_people": sorted(
            canonical_people, key=lambda item: item["person_id"]
        ),
        "slot_bindings": slot_bindings,
        "merge_outcomes": sorted(
            merge_outcomes, key=lambda item: str(item["merge_proposal_id"])
        ),
        "voice_binding_outcomes": sorted(
            voice_outcomes, key=lambda item: str(item["binding_proposal_id"])
        ),
        "enrollment_units": enrollment_units,
        "metrics": {
            "canonical_person_count": len(canonical_people),
            "slot_binding_count": len(slot_bindings),
            "external_identity_count": sum(
                len(person["external_identities"])
                for person in canonical_people
            ),
            "accepted_merge_count": sum(
                item["decision"] == "accept" for item in merge_outcomes
            ),
            "rejected_merge_count": sum(
                item["decision"] == "reject" for item in merge_outcomes
            ),
            "active_voice_binding_count": sum(
                item["active_binding"] for item in voice_outcomes
            ),
            "reviewed_voice_binding_count": len(voice_outcomes),
            "included_source_count": len(included_sources),
            "excluded_source_count": excluded_source_count,
            "enrollment_unit_count": len(enrollment_units),
            "source_feasible_enrollment_unit_count": sum(
                item["status"] == "source_selected" for item in enrollment_units
            ),
        },
        "rehearsal_allowed": within_bound,
        "a1_authorized": False,
        "live_mutation_count": 0,
    }
    return {**core, "content_sha256": canonical_artifact_hash(core)}


def _database_snapshot(database: Path) -> dict[str, Any]:
    selected = database.expanduser().absolute()
    if not selected.is_file() or selected.is_symlink():
        _fail("The rehearsal database is unavailable.")
    try:
        with sqlite3.connect(f"file:{selected}?mode=ro", uri=True) as connection:
            connection.row_factory = sqlite3.Row
            quick = str(connection.execute("PRAGMA quick_check").fetchone()[0])
            table_names = [
                str(row[0])
                for row in connection.execute(
                    "SELECT name FROM sqlite_master "
                    "WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
                ).fetchall()
            ]
            tables: dict[str, Any] = {}
            for table in table_names:
                quoted = table.replace('"', '""')
                rows = []
                for row in connection.execute(f'SELECT * FROM "{quoted}"').fetchall():
                    body = {
                        key: (value.hex() if isinstance(value, bytes) else value)
                        for key, value in dict(row).items()
                    }
                    rows.append(body)
                rows.sort(
                    key=lambda item: json.dumps(
                        item,
                        sort_keys=True,
                        separators=(",", ":"),
                        ensure_ascii=False,
                    )
                )
                tables[table] = {
                    "count": len(rows),
                    "rows_sha256": canonical_artifact_hash(rows),
                }
    except sqlite3.Error as exc:
        raise Plan0063PrivateRehearsalError(
            "The rehearsal database could not be inventoried."
        ) from exc
    core = {
        "quick_check": quick,
        "database_sha256": sha256_file(selected),
        "tables": tables,
    }
    return {**core, "snapshot_sha256": canonical_artifact_hash(core)}


def _person_snapshot(
    person: Mapping[str, Any], *, transition: Mapping[str, Any]
) -> PersonSnapshot:
    person_id = str(person["person_id"])
    reviewed_at = str(transition["reviewed_at"])
    source_records = []
    identities = []
    seen_identities: set[tuple[str, str]] = set()
    slot_bindings = {
        str(item["slot_id"]): dict(item)
        for item in transition.get("slot_bindings") or []
        if isinstance(item, Mapping)
    }
    for slot_id in person.get("member_slot_ids") or []:
        slot = slot_bindings[str(slot_id)]
        source_id = _uuid(f"plan0063-person-source:{person_id}:{slot_id}")
        source_payload = {
            "person_id": person_id,
            "slot_id": slot_id,
            "review_submission_sha256": transition["review_submission_sha256"],
        }
        source_records.append(
            SourceRecord(
                source_record_id=source_id,
                person_id=person_id,
                source_profile_id="plan0062-human-review",
                provider_kind="human_review",
                account_id="",
                tenant_id="",
                external_ref=str(slot_id),
                label=str(person["primary_name"]),
                relationship_scope="speaker_identity",
                identifier_authority="operator_reviewed",
                observed_at=reviewed_at,
                content_hash=canonical_artifact_hash(source_payload),
                metadata={
                    "decision_type": slot.get("decision_type"),
                    "source_submission_sha256": transition[
                        "source_submission_sha256"
                    ],
                },
            )
        )
        value = str(slot.get("email") or "")
        identity_key = ("email", value.casefold())
        if value and identity_key not in seen_identities:
            kind = "email"
            seen_identities.add(identity_key)
            identities.append(
                ExternalIdentityRecord(
                    external_identity_id=_uuid(
                        f"plan0063-external:{person_id}:{slot_id}:{kind}:{value}"
                    ),
                    person_id=person_id,
                    source_record_id=source_id,
                    identity_kind=kind,
                    normalized_value=value.casefold(),
                    display_value=value,
                    authority="operator_reviewed_context",
                    verified=True,
                    metadata={
                        "review_submission_sha256": transition[
                            "review_submission_sha256"
                        ]
                    },
                )
            )
    return PersonSnapshot(
        person=PersonRecord(
            person_id=person_id,
            status="reviewed",
            primary_name=str(person["primary_name"]),
            metadata={
                "plan0063_transition_sha256": transition["content_sha256"],
                "merge_decision": person.get("merge_decision"),
                "organizations": list(person.get("organizations") or []),
            },
        ),
        source_records=tuple(source_records),
        external_identities=tuple(identities),
    )


def _observations(transition: Mapping[str, Any]) -> tuple[ObservationRecord, ...]:
    reviewed_at = str(transition["reviewed_at"])
    rows: list[ObservationRecord] = []
    for binding in transition.get("slot_bindings") or []:
        payload = {
            "slot_id": binding["slot_id"],
            "document_id": binding["document_id"],
            "speaker_label_id": binding["speaker_label_id"],
            "person_id": binding["person_id"],
            "review_submission_sha256": transition["review_submission_sha256"],
        }
        rows.append(
            ObservationRecord(
                observation_id=_uuid(
                    "plan0063-slot-binding:"
                    + str(transition["review_submission_sha256"])
                    + ":"
                    + str(binding["slot_id"])
                ),
                observation_type="reviewed_speaker_slot_binding",
                subject_type="person",
                subject_id=str(binding["person_id"]),
                source_type="human_review",
                source_id=str(binding["slot_id"]),
                conversation_id="",
                observed_at=reviewed_at,
                review_state="accepted",
                payload=payload,
                content_hash=canonical_artifact_hash(payload),
            )
        )
    for binding in transition.get("voice_binding_outcomes") or []:
        payload = {
            "binding_proposal_id": binding["binding_proposal_id"],
            "decision": binding["decision"],
            "active_binding": binding["active_binding"],
            "acoustic_subject_id": binding["acoustic_subject_id"],
            "person_id": binding["person_id"],
            "review_submission_sha256": transition["review_submission_sha256"],
        }
        rows.append(
            ObservationRecord(
                observation_id=_uuid(
                    "plan0063-voice-binding:"
                    + str(transition["review_submission_sha256"])
                    + ":"
                    + str(binding["binding_proposal_id"])
                ),
                observation_type=(
                    "reviewed_voice_subject_binding_confirmed"
                    if binding["active_binding"]
                    else "reviewed_voice_subject_binding_rejected"
                ),
                subject_type="person",
                subject_id=str(binding["person_id"]),
                source_type="human_review",
                source_id=str(binding["acoustic_subject_id"]),
                conversation_id="",
                observed_at=reviewed_at,
                review_state="accepted",
                payload=payload,
                content_hash=canonical_artifact_hash(payload),
            )
        )
    return tuple(sorted(rows, key=lambda item: item.observation_id))


def validate_reviewed_transition(transition: Mapping[str, Any]) -> str:
    """Return the hash of one transition eligible for A0 private rehearsal."""

    transition_sha256 = _assert_content_hash(transition, "reviewed transition")
    if (
        transition.get("schema_version") != TRANSITION_SCHEMA
        or transition.get("status")
        != "reviewed_transition_ready_for_private_rehearsal"
        or transition.get("rehearsal_allowed") is not True
        or transition.get("a1_authorized") is not False
        or transition.get("live_mutation_count") != 0
    ):
        _fail("The reviewed transition is not eligible for private rehearsal.")
    return transition_sha256


def rehearsal_paths(runtime_root: Path, transition_sha256: str) -> dict[str, Path]:
    root = runtime_root.expanduser().absolute()
    run = root / f"p5-private-copy-rehearsal-{transition_sha256[:24]}"
    return {
        "root": root,
        "run": run,
        "working_root": run / "working-store",
        "working_db": run / "working-store" / transcript_store.DEFAULT_DB_NAME,
        "baseline_db": run / "baseline.sqlite3",
        "transition": run / "reviewed-transition.json",
        "manifest": run / "rehearsal.json",
        "receipt": run / "receipt.json",
    }


def replay_knowledge_rehearsal(
    *,
    transition_sha256: str,
    live_store_root: Path,
    runtime_root: Path = DEFAULT_RUNTIME_ROOT,
) -> dict[str, Any]:
    paths = rehearsal_paths(runtime_root, transition_sha256)
    for key in ("working_db", "baseline_db", "transition", "manifest", "receipt"):
        require_private_file(paths[key], paths["root"])
    transition = read_private_object(paths["transition"])
    manifest = read_private_object(paths["manifest"])
    receipt = read_private_object(paths["receipt"])
    _assert_content_hash(transition, "reviewed transition")
    if transition["content_sha256"] != transition_sha256:
        _fail("The selected transition does not match the rehearsal.")
    live_snapshot = _database_snapshot(transcript_store.db_path(live_store_root))
    working_snapshot = _database_snapshot(paths["working_db"])
    baseline_snapshot = _database_snapshot(paths["baseline_db"])
    receipt_core = {
        key: value for key, value in receipt.items() if key != "content_sha256"
    }
    if (
        manifest.get("schema_version") != REHEARSAL_SCHEMA
        or manifest.get("transition_sha256") != transition_sha256
        or manifest.get("rolled_back_snapshot") != working_snapshot
        or manifest.get("baseline_snapshot") != baseline_snapshot
        or working_snapshot != baseline_snapshot
        or manifest.get("live_snapshot_before") != live_snapshot
        or manifest.get("live_snapshot_after") != live_snapshot
        or receipt.get("schema_version") != REHEARSAL_RECEIPT_SCHEMA
        or receipt.get("manifest_sha256") != sha256_file(paths["manifest"])
        or receipt.get("transition_file_sha256") != sha256_file(paths["transition"])
        or receipt.get("content_sha256") != canonical_artifact_hash(receipt_core)
        or receipt.get("live_mutation_count") != 0
    ):
        _fail("The private-copy rehearsal replay drifted.")
    return {
        **receipt,
        "manifest_path": str(paths["manifest"]),
        "receipt_path": str(paths["receipt"]),
        "transition_path": str(paths["transition"]),
        "idempotent_replay": True,
    }


def rehearse_knowledge_copy(
    transition: Mapping[str, Any],
    *,
    live_store_root: Path,
    runtime_root: Path = DEFAULT_RUNTIME_ROOT,
) -> dict[str, Any]:
    """Apply and roll back canonical people on one private transcript DB copy."""

    transition_sha256 = validate_reviewed_transition(transition)
    paths = rehearsal_paths(runtime_root, transition_sha256)
    if paths["receipt"].exists():
        return replay_knowledge_rehearsal(
            transition_sha256=transition_sha256,
            live_store_root=live_store_root,
            runtime_root=runtime_root,
        )
    if paths["run"].exists():
        _fail("A partial private-copy rehearsal already exists.")
    ensure_private_tree(paths["root"], paths["run"])
    ensure_private_tree(paths["root"], paths["working_root"])
    live_db = transcript_store.db_path(live_store_root)
    live_before = _database_snapshot(live_db)
    try:
        orchestration._sqlite_backup(live_db, paths["working_db"])
        shutil.copy2(paths["working_db"], paths["baseline_db"])
        paths["baseline_db"].chmod(0o600)
        baseline = _database_snapshot(paths["baseline_db"])
        baseline_knowledge = {
            name for name in baseline["tables"] if name.startswith("knowledge_")
        }
        if baseline["quick_check"] != "ok" or baseline_knowledge:
            _fail("Plan 0063 requires a clean schema-version-zero baseline copy.")

        store = ConversationKnowledgeStore(paths["working_root"])
        migration = store.migrate(backup=False)
        if migration.from_version != 0 or migration.to_version != 3:
            _fail("The private knowledge migration did not reach schema version 3.")
        person_receipts = [
            store.save_person_snapshot(
                _person_snapshot(person, transition=transition)
            )
            for person in transition.get("canonical_people") or []
        ]
        observation_receipt = store.save_observations("", _observations(transition))
        profile_receipt = ConversationProfileProjector(
            paths["working_root"]
        ).rebuild()
        applied = _database_snapshot(paths["working_db"])
        metrics = dict(transition.get("metrics") or {})
        expected_counts = {
            "knowledge_people": int(metrics.get("canonical_person_count") or 0),
            "knowledge_source_records": int(metrics.get("slot_binding_count") or 0),
            "knowledge_external_identities": int(
                metrics.get("external_identity_count") or 0
            ),
            "knowledge_observations": int(metrics.get("slot_binding_count") or 0)
            + int(metrics.get("reviewed_voice_binding_count") or 0),
            "knowledge_current_person_profiles": int(
                metrics.get("canonical_person_count") or 0
            ),
            "knowledge_affinity_profiles": 0,
            "knowledge_projection_state": 1,
        }
        for table, count in expected_counts.items():
            if applied["tables"].get(table, {}).get("count") != count:
                _fail("The private-copy apply counts did not reconcile.")
        if applied["quick_check"] != "ok":
            _fail("The private-copy apply failed SQLite quick_check.")

        rollback = store.rollback(target_version=0, backup=False)
        if rollback.from_version != 3 or rollback.to_version != 0:
            _fail("The knowledge schema rollback did not return to version 0.")
        logical_rollback = _database_snapshot(paths["working_db"])
        if (
            logical_rollback["quick_check"] != "ok"
            or logical_rollback["tables"] != baseline["tables"]
        ):
            _fail("The private-copy logical rollback did not reconcile.")

        restore_fd, restore_name = tempfile.mkstemp(
            prefix=".restore-", dir=paths["working_root"]
        )
        os.close(restore_fd)
        restore_stage = Path(restore_name)
        try:
            shutil.copy2(paths["baseline_db"], restore_stage)
            restore_stage.chmod(0o600)
            os.replace(restore_stage, paths["working_db"])
        finally:
            if restore_stage.exists():
                restore_stage.unlink()
        rolled_back = _database_snapshot(paths["working_db"])
        if rolled_back != baseline:
            _fail("The exact baseline database was not restored after rollback.")
        live_after = _database_snapshot(live_db)
        if live_after != live_before:
            _fail("Live transcript state changed during the private rehearsal.")

        write_immutable_private_json(paths["transition"], dict(transition))
        manifest_core = {
            "schema_version": REHEARSAL_SCHEMA,
            "status": "knowledge_private_apply_and_rollback_proved",
            "transition_sha256": transition_sha256,
            "live_snapshot_before": live_before,
            "baseline_snapshot": baseline,
            "applied_snapshot": applied,
            "logical_rollback_snapshot": logical_rollback,
            "rolled_back_snapshot": rolled_back,
            "live_snapshot_after": live_after,
            "apply_counts": expected_counts,
            "person_receipts": [asdict(receipt) for receipt in person_receipts],
            "observation_receipt": asdict(observation_receipt),
            "profile_receipt": asdict(profile_receipt),
            "copy_apply_count": 1,
            "copy_rollback_count": 1,
            "biometric_rehearsal_status": "pending_exact_source_application",
            "a1_authorized": False,
            "live_mutation_count": 0,
        }
        manifest = {
            **manifest_core,
            "content_sha256": canonical_artifact_hash(manifest_core),
        }
        write_immutable_private_json(paths["manifest"], manifest)
        receipt_core = {
            "schema_version": REHEARSAL_RECEIPT_SCHEMA,
            "status": "knowledge_private_apply_and_rollback_proved",
            "transition_sha256": transition_sha256,
            "transition_file_sha256": sha256_file(paths["transition"]),
            "manifest_sha256": sha256_file(paths["manifest"]),
            "baseline_database_sha256": baseline["database_sha256"],
            "rolled_back_database_sha256": rolled_back["database_sha256"],
            "copy_apply_count": 1,
            "copy_rollback_count": 1,
            "biometric_rehearsal_complete": False,
            "a1_authorized": False,
            "live_mutation_count": 0,
        }
        receipt = {
            **receipt_core,
            "content_sha256": canonical_artifact_hash(receipt_core),
        }
        write_immutable_private_json(paths["receipt"], receipt)
        return {
            **receipt,
            "manifest_path": str(paths["manifest"]),
            "receipt_path": str(paths["receipt"]),
            "transition_path": str(paths["transition"]),
            "idempotent_replay": False,
        }
    except Exception:
        if paths["run"].exists() and not paths["receipt"].exists():
            shutil.rmtree(paths["run"])
        raise


__all__ = [
    "Plan0063PrivateRehearsalError",
    "TRANSITION_SCHEMA",
    "build_reviewed_transition",
    "rehearsal_paths",
    "rehearse_knowledge_copy",
    "replay_knowledge_rehearsal",
    "validate_reviewed_transition",
]
