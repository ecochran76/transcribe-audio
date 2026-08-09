"""Validate and freeze Plan 0062 human gold without applying identity changes."""

from __future__ import annotations

import base64
import binascii
import re
import unicodedata
from collections import Counter
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Mapping

from acoustic_audio_derivatives import (
    canonical_artifact_hash,
    ensure_private_tree,
    read_private_object,
    require_private_file,
    sha256_file,
    write_immutable_private_json,
)
import speaker_identity_context_human_review as review
from speaker_identity_orchestration import negative_action_vector
from speaker_identity_plan0062_execution import (
    EXPECTED_CONDITIONS,
    EXPECTED_DOCUMENTS,
    EXPECTED_SPEAKER_COUNTS,
    MANIFEST_SCHEMA as P3_MANIFEST_SCHEMA,
)


DECISION_SCHEMA = "transcribe-audio.plan0062-human-gold.v1"
DECISION_RECEIPT_SCHEMA = "transcribe-audio.plan0062-human-gold-receipt.v1"
BINDING_SCHEMA = "transcribe-audio.plan0062-enrolled-option-bindings.v1"
BINDING_RECEIPT_SCHEMA = "transcribe-audio.plan0062-enrolled-option-bindings-receipt.v1"
COMPARISON_SCHEMA = "transcribe-audio.plan0062-three-condition-comparison.v1"
COMPARISON_RECEIPT_SCHEMA = "transcribe-audio.plan0062-comparison-receipt.v1"
TERMINAL_SCHEMA = "transcribe-audio.plan0062-terminal-audit.v1"
OPTION_RE = re.compile(r"^(enrolled|canonical|suggested)-[a-f0-9]{24}$")
NEW_PERSON_RE = re.compile(r"^new_person:([A-Za-z0-9_-]{1,512})$")
HIGH_CONFIDENCE_THRESHOLD = 0.8


class Plan0062HumanComparisonError(ValueError):
    """Raised when human gold or its frozen comparison is not exact."""


def _fail(message: str) -> None:
    raise Plan0062HumanComparisonError(message)


def _ratio(numerator: int, denominator: int) -> float | None:
    return round(numerator / denominator, 4) if denominator else None


def _expected_slots() -> list[str]:
    return [
        f"{document_id}::SPEAKER_{ordinal}"
        for document_id in EXPECTED_DOCUMENTS
        for ordinal in range(1, EXPECTED_SPEAKER_COUNTS[document_id] + 1)
    ]


def _review_packet(p4_source: Mapping[str, Any]) -> dict[str, Any]:
    p4 = dict(p4_source)
    if p4.get("schema_version") == review.MANIFEST_SCHEMA:
        if (
            p4.get("status") != "awaiting_literal_human_review"
            or any((p4.get("negative_actions") or {}).values())
            or not isinstance(p4.get("packet"), Mapping)
        ):
            _fail("The Plan 0062 P4 manifest is invalid.")
        packet = dict(p4["packet"])
    else:
        packet = p4
    packet_core = {
        key: value for key, value in packet.items() if key != "content_sha256"
    }
    if (
        packet.get("schema_version") != review.PACKET_SCHEMA
        or packet.get("status") != "awaiting_literal_human_review"
        or packet.get("content_sha256") != canonical_artifact_hash(packet_core)
        or int(packet.get("speaker_slot_count") or 0) != 10
        or int(packet.get("preselected_decision_count") or 0) != 0
        or int(packet.get("human_decision_count") or 0) != 0
        or any((packet.get("negative_actions") or {}).values())
    ):
        _fail("The Plan 0062 P4 review packet is invalid or stale.")
    return packet


def _bundle_value(raw: Any) -> tuple[dict[str, Any], str]:
    if is_dataclass(raw):
        bundle = asdict(raw)
        bundle_id = str(getattr(raw, "bundle_id", "") or "")
    elif isinstance(raw, Mapping):
        outer = dict(raw)
        nested = outer.get("bundle")
        bundle = dict(nested) if isinstance(nested, Mapping) else outer
        bundle_id = str(outer.get("bundle_id") or bundle.get("bundle_id") or "")
    else:
        _fail("An enrolled-option acoustic bundle is invalid.")
    evidence = bundle.get("evidence")
    if not isinstance(evidence, (list, tuple)):
        _fail("An enrolled-option acoustic bundle lacks speaker evidence.")
    return bundle, bundle_id


def build_enrolled_option_bindings(
    p4_source: Mapping[str, Any],
    *,
    acoustic_bundles: Mapping[str, Any],
) -> dict[str, Any]:
    """Bind opaque enrolled options to exact private acoustic subjects."""

    packet = _review_packet(p4_source)
    if set(acoustic_bundles) != set(EXPECTED_DOCUMENTS):
        _fail("Enrolled-option bundles do not cover the exact Plan 0062 cohort.")
    bundle_rows: dict[str, tuple[dict[str, Mapping[str, Any]], str, str]] = {}
    for document_id in EXPECTED_DOCUMENTS:
        bundle, bundle_id = _bundle_value(acoustic_bundles[document_id])
        if str(bundle.get("document_id") or "") != document_id:
            _fail("An enrolled-option bundle has the wrong document binding.")
        rows = {
            str(row.get("speaker_ref") or ""): row
            for row in bundle.get("evidence") or []
            if isinstance(row, Mapping)
        }
        if len(rows) != EXPECTED_SPEAKER_COUNTS[document_id]:
            _fail("An enrolled-option bundle has an incomplete speaker denominator.")
        bundle_rows[document_id] = (
            rows,
            bundle_id,
            canonical_artifact_hash(bundle),
        )

    bindings: list[dict[str, Any]] = []
    for card in packet.get("cards") or []:
        document_id = str(card.get("document_id") or "")
        speaker_ref = str(card.get("speaker_ref") or "")
        slot_id = str(card.get("slot_id") or "")
        rows, bundle_id, bundle_sha256 = bundle_rows.get(document_id, ({}, "", ""))
        row = rows.get(speaker_ref)
        if not isinstance(row, Mapping):
            _fail("An enrolled review option lacks its acoustic speaker row.")
        options = [
            option
            for option in card.get("options") or []
            if isinstance(option, Mapping)
            and option.get("source") == "enrolled_voice_subject"
        ]
        acoustic_subject_id = str(row.get("acoustic_subject_id") or "")
        if options:
            if (
                len(options) != 1
                or not acoustic_subject_id
                or str(row.get("disposition") or "") == "abstain"
            ):
                _fail("An enrolled review option has ambiguous acoustic authority.")
            token = str(options[0].get("token") or "")
            if token != review._option_token("enrolled", acoustic_subject_id):
                _fail("An enrolled review token lost its acoustic-subject binding.")
            bindings.append(
                {
                    "slot_id": slot_id,
                    "token": token,
                    "acoustic_subject_id": acoustic_subject_id,
                    "acoustic_bundle_id": bundle_id,
                    "acoustic_bundle_sha256": bundle_sha256,
                }
            )
        elif acoustic_subject_id and str(row.get("disposition") or "") != "abstain":
            _fail("A non-abstaining acoustic subject is missing from review options.")

    core = {
        "schema_version": BINDING_SCHEMA,
        "status": "private_enrolled_option_bindings_ready",
        "p4_content_sha256": packet["content_sha256"],
        "binding_count": len(bindings),
        "bindings": bindings,
        "negative_actions": negative_action_vector(),
    }
    return {**core, "content_sha256": canonical_artifact_hash(core)}


def _validated_enrolled_bindings(
    packet: Mapping[str, Any], binding_source: Mapping[str, Any]
) -> dict[str, Mapping[str, Any]]:
    bindings = binding_source.get("bindings")
    core = {
        key: value for key, value in binding_source.items() if key != "content_sha256"
    }
    if (
        binding_source.get("schema_version") != BINDING_SCHEMA
        or binding_source.get("status") != "private_enrolled_option_bindings_ready"
        or binding_source.get("p4_content_sha256") != packet.get("content_sha256")
        or binding_source.get("content_sha256") != canonical_artifact_hash(core)
        or not isinstance(bindings, list)
        or int(binding_source.get("binding_count") or 0) != len(bindings)
        or any((binding_source.get("negative_actions") or {}).values())
    ):
        _fail("The Plan 0062 enrolled-option binding authority is invalid.")
    expected = [
        (str(card.get("slot_id") or ""), str(option.get("token") or ""))
        for card in packet.get("cards") or []
        for option in card.get("options") or []
        if isinstance(card, Mapping)
        and isinstance(option, Mapping)
        and option.get("source") == "enrolled_voice_subject"
    ]
    actual: list[tuple[str, str]] = []
    by_slot: dict[str, Mapping[str, Any]] = {}
    for binding in bindings:
        if not isinstance(binding, Mapping):
            _fail("A Plan 0062 enrolled-option binding is invalid.")
        slot_id = str(binding.get("slot_id") or "")
        token = str(binding.get("token") or "")
        subject_id = str(binding.get("acoustic_subject_id") or "")
        bundle_sha256 = str(binding.get("acoustic_bundle_sha256") or "")
        if (
            slot_id in by_slot
            or not subject_id
            or token != review._option_token("enrolled", subject_id)
            or not re.fullmatch(r"[a-f0-9]{64}", bundle_sha256)
        ):
            _fail("A Plan 0062 enrolled-option binding drifted.")
        actual.append((slot_id, token))
        by_slot[slot_id] = binding
    if actual != expected:
        _fail("The enrolled-option binding denominator or order drifted.")
    return by_slot


def _binding_paths(runtime_root: Path, content_sha256: str) -> dict[str, Path]:
    root = runtime_root.expanduser().absolute()
    run = root / f"p5-source-bindings-{content_sha256[:24]}"
    return {
        "root": root,
        "run": run,
        "manifest": run / "private-manifest.json",
        "receipt": run / "receipt.json",
    }


def freeze_enrolled_option_bindings(
    binding_manifest: Mapping[str, Any], *, runtime_root: Path
) -> dict[str, Any]:
    """Freeze or replay the private token-to-acoustic-subject authority."""

    content_sha256 = str(binding_manifest.get("content_sha256") or "")
    if (
        not re.fullmatch(r"[a-f0-9]{64}", content_sha256)
        or content_sha256
        != canonical_artifact_hash(
            {
                key: value
                for key, value in binding_manifest.items()
                if key != "content_sha256"
            }
        )
    ):
        _fail("The enrolled-option binding content hash is invalid.")
    paths = _binding_paths(runtime_root, content_sha256)
    if paths["receipt"].exists():
        return replay_enrolled_option_bindings(
            content_sha256=content_sha256, runtime_root=runtime_root
        )
    if paths["run"].exists():
        _fail("A partial Plan 0062 source-binding directory already exists.")
    ensure_private_tree(paths["root"], paths["run"])
    write_immutable_private_json(paths["manifest"], dict(binding_manifest))
    receipt = {
        "schema_version": BINDING_RECEIPT_SCHEMA,
        "status": "private_enrolled_option_bindings_frozen",
        "content_sha256": content_sha256,
        "manifest_sha256": sha256_file(paths["manifest"]),
        "binding_count": int(binding_manifest.get("binding_count") or 0),
        "live_mutation_count": 0,
        "negative_actions_preserved": True,
    }
    write_immutable_private_json(paths["receipt"], receipt)
    return {
        **receipt,
        "manifest_path": str(paths["manifest"]),
        "receipt_path": str(paths["receipt"]),
        "idempotent_replay": False,
    }


def replay_enrolled_option_bindings(
    *, content_sha256: str, runtime_root: Path
) -> dict[str, Any]:
    """Verify an immutable private enrolled-option binding sidecar."""

    if not re.fullmatch(r"[a-f0-9]{64}", str(content_sha256 or "")):
        _fail("The enrolled-option binding SHA-256 is invalid.")
    paths = _binding_paths(runtime_root, content_sha256)
    require_private_file(paths["manifest"], paths["root"])
    require_private_file(paths["receipt"], paths["root"])
    manifest = read_private_object(paths["manifest"])
    receipt = read_private_object(paths["receipt"])
    manifest_core = {
        key: value for key, value in manifest.items() if key != "content_sha256"
    }
    if (
        manifest.get("schema_version") != BINDING_SCHEMA
        or manifest.get("content_sha256") != content_sha256
        or canonical_artifact_hash(manifest_core) != content_sha256
        or any((manifest.get("negative_actions") or {}).values())
        or receipt.get("schema_version") != BINDING_RECEIPT_SCHEMA
        or receipt.get("content_sha256") != content_sha256
        or receipt.get("manifest_sha256") != sha256_file(paths["manifest"])
        or receipt.get("binding_count") != manifest.get("binding_count")
        or receipt.get("live_mutation_count") != 0
        or receipt.get("negative_actions_preserved") is not True
    ):
        _fail("The frozen Plan 0062 source-binding evidence drifted.")
    return {
        **receipt,
        "manifest_path": str(paths["manifest"]),
        "receipt_path": str(paths["receipt"]),
        "idempotent_replay": True,
    }


def _validated_sources(
    p3_manifest: Mapping[str, Any], p4_source: Mapping[str, Any]
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Mapping[str, Any]]]:
    p3 = dict(p3_manifest)
    if (
        p3.get("schema_version") != P3_MANIFEST_SCHEMA
        or p3.get("status") != "joined_pending_human_review"
        or int(p3.get("recording_count") or 0) != 3
        or int(p3.get("speaker_count") or 0) != 10
        or int(p3.get("evaluation_count") or 0) != 30
        or any((p3.get("negative_actions") or {}).values())
    ):
        _fail("The Plan 0062 P3 comparison authority is invalid.")
    p3_content_sha256 = canonical_artifact_hash(p3)

    packet = _review_packet(p4_source)
    if (
        packet.get("p3_content_sha256") != p3_content_sha256
    ):
        _fail("The Plan 0062 P4 review packet lost its P3 binding.")

    raw_results = p3.get("results")
    if not isinstance(raw_results, list):
        _fail("The Plan 0062 P3 results are unavailable.")
    p3_slots: dict[str, Mapping[str, Any]] = {}
    for result in raw_results:
        if not isinstance(result, Mapping):
            _fail("A Plan 0062 P3 result is invalid.")
        document_id = str(result.get("document_id") or "")
        join = result.get("join") if isinstance(result.get("join"), Mapping) else {}
        outcomes = join.get("review_outcomes")
        evaluations = join.get("evaluations")
        if not isinstance(outcomes, (list, tuple)) or not isinstance(
            evaluations, (list, tuple)
        ):
            _fail("A Plan 0062 P3 result lacks joined review evidence.")
        evaluation_by_ref: dict[str, list[Mapping[str, Any]]] = {}
        for evaluation in evaluations:
            if not isinstance(evaluation, Mapping):
                _fail("A Plan 0062 P3 evaluation is invalid.")
            evaluation_by_ref.setdefault(
                str(evaluation.get("speaker_ref") or ""), []
            ).append(evaluation)
        for outcome in outcomes:
            if not isinstance(outcome, Mapping):
                _fail("A Plan 0062 P3 review outcome is invalid.")
            speaker_ref = str(outcome.get("speaker_ref") or "")
            slot_id = f"{document_id}::{speaker_ref}"
            slot_evaluations = evaluation_by_ref.get(speaker_ref) or []
            conditions = {str(item.get("condition") or "") for item in slot_evaluations}
            if slot_id in p3_slots or conditions != set(EXPECTED_CONDITIONS):
                _fail("The Plan 0062 P3 condition denominator drifted.")
            p3_slots[slot_id] = {
                "outcome": outcome,
                "evaluations": slot_evaluations,
            }

    cards = packet.get("cards")
    expected = _expected_slots()
    if (
        not isinstance(cards, list)
        or [str(card.get("slot_id") or "") for card in cards if isinstance(card, Mapping)]
        != expected
        or set(p3_slots) != set(expected)
    ):
        _fail("The Plan 0062 P3/P4 speaker denominator drifted.")

    cards_by_slot: dict[str, Mapping[str, Any]] = {}
    for card in cards:
        if not isinstance(card, Mapping):
            _fail("A Plan 0062 P4 review card is invalid.")
        slot_id = str(card.get("slot_id") or "")
        raw_options = card.get("options")
        if not isinstance(raw_options, list):
            _fail("A Plan 0062 P4 option set is invalid.")
        seen_tokens: set[str] = set()
        p3_outcome = p3_slots[slot_id]["outcome"]
        for option in raw_options:
            if not isinstance(option, Mapping):
                _fail("A Plan 0062 P4 decision option is invalid.")
            token = str(option.get("token") or "")
            source = str(option.get("source") or "")
            label = str(option.get("label") or "").strip()
            match = OPTION_RE.fullmatch(token)
            expected_prefix = {
                "enrolled_voice_subject": "enrolled",
                "canonical_context_proposal": "canonical",
                "contextual_unlisted_suggestion": "suggested",
            }.get(source)
            if (
                token in seen_tokens
                or match is None
                or match.group(1) != expected_prefix
                or not label
                or len(label) > 320
            ):
                _fail("A Plan 0062 P4 decision option drifted.")
            if source == "canonical_context_proposal":
                context_person_id = str(p3_outcome.get("context_person_id") or "")
                if not context_person_id or token != review._option_token(
                    "canonical", context_person_id
                ):
                    _fail("A canonical review option lost its P3 person binding.")
            if source == "contextual_unlisted_suggestion" and token not in {
                review._option_token("suggested", dict(suggestion))
                for suggestion in p3_outcome.get("suggestions") or []
                if isinstance(suggestion, Mapping)
            }:
                _fail("A suggested review option lost its P3 evidence binding.")
            if source == "enrolled_voice_subject":
                acoustic = card.get("acoustic")
                if not isinstance(acoustic, Mapping) or not bool(
                    acoustic.get("has_enrolled_subject")
                ):
                    _fail("An enrolled review option lacks acoustic evidence.")
            seen_tokens.add(token)
        cards_by_slot[slot_id] = card
    return p3, packet, p3_slots


def _decode_new_person(value: str) -> str:
    match = NEW_PERSON_RE.fullmatch(value)
    if match is None:
        _fail("A new-person decision has invalid encoding.")
    encoded = match.group(1)
    try:
        raw = base64.urlsafe_b64decode(encoded + "=" * (-len(encoded) % 4))
        name = raw.decode("utf-8")
    except (binascii.Error, ValueError, UnicodeDecodeError) as exc:
        _fail("A new-person decision is not valid UTF-8.")
    if (
        not name
        or name != name.strip()
        or len(name) > 160
        or not any(not character.isspace() for character in name)
        or any(unicodedata.category(character).startswith("C") for character in name)
    ):
        _fail("A new-person decision has an invalid display name.")
    return name


def _decision_for_selection(
    *,
    slot_id: str,
    selected: str,
    card: Mapping[str, Any],
    p3_outcome: Mapping[str, Any],
    enrolled_binding: Mapping[str, Any] | None,
) -> dict[str, Any]:
    option_by_token = {
        str(option["token"]): option for option in card.get("options") or []
    }
    decision: dict[str, Any] = {"slot_id": slot_id, "selected_token": selected}
    if selected == "unresolved":
        decision.update({"decision_type": "unresolved", "label": "Unresolved"})
    elif selected.startswith("new_person:"):
        decision.update(
            {"decision_type": "new_person", "label": _decode_new_person(selected)}
        )
    elif selected in option_by_token:
        option = option_by_token[selected]
        source = str(option["source"])
        decision.update({"decision_type": source, "label": str(option["label"])})
        if source == "canonical_context_proposal":
            decision["person_id"] = str(p3_outcome.get("context_person_id") or "")
        elif source == "contextual_unlisted_suggestion":
            matches = [
                dict(suggestion)
                for suggestion in p3_outcome.get("suggestions") or []
                if isinstance(suggestion, Mapping)
                and review._option_token("suggested", dict(suggestion)) == selected
            ]
            if len(matches) != 1:
                _fail("A selected contextual suggestion is ambiguous.")
            decision["suggestion"] = matches[0]
        else:
            if (
                not isinstance(enrolled_binding, Mapping)
                or enrolled_binding.get("token") != selected
            ):
                _fail("A selected enrolled voice lacks exact private binding authority.")
            decision["acoustic_subject_id"] = str(
                enrolled_binding["acoustic_subject_id"]
            )
            decision["acoustic_bundle_id"] = str(
                enrolled_binding.get("acoustic_bundle_id") or ""
            )
            decision["acoustic_bundle_sha256"] = str(
                enrolled_binding["acoustic_bundle_sha256"]
            )
            decision["binding_status"] = (
                "reviewed_voice_subject_selected_pending_person_apply"
            )
    else:
        _fail("A Plan 0062 decision is outside the frozen option set.")
    return decision


def parse_human_submission(
    answer_text: str,
    *,
    p3_manifest: Mapping[str, Any],
    p4_source: Mapping[str, Any],
    enrolled_binding_source: Mapping[str, Any],
) -> dict[str, Any]:
    """Parse exactly ten ordered choices from the client-only P4 export."""

    p3, packet, p3_slots = _validated_sources(p3_manifest, p4_source)
    enrolled_by_slot = _validated_enrolled_bindings(
        packet, enrolled_binding_source
    )
    expected = _expected_slots()
    lines = [line.strip() for line in str(answer_text or "").splitlines() if line.strip()]
    headers = [
        f"PLAN0062_SCHEMA={review.SUBMISSION_SCHEMA}",
        f"PLAN0062_P3_CONTENT_SHA256={canonical_artifact_hash(p3)}",
        f"PLAN0062_P4_CONTENT_SHA256={packet['content_sha256']}",
    ]
    if lines[:3] != headers or len(lines) != 3 + len(expected):
        _fail("The Plan 0062 decision block is stale, incomplete, or malformed.")

    cards_by_slot = {str(card["slot_id"]): card for card in packet["cards"]}
    decisions: list[dict[str, Any]] = []
    seen: set[str] = set()
    for expected_slot, line in zip(expected, lines[3:], strict=True):
        if "=" not in line:
            _fail("A Plan 0062 decision row is malformed.")
        slot_id, selected = line.rsplit("=", 1)
        if slot_id != expected_slot or slot_id in seen:
            _fail("The Plan 0062 decision block changed slot order or denominator.")
        seen.add(slot_id)
        decisions.append(
            _decision_for_selection(
                slot_id=slot_id,
                selected=selected,
                card=cards_by_slot[slot_id],
                p3_outcome=p3_slots[slot_id]["outcome"],
                enrolled_binding=enrolled_by_slot.get(slot_id),
            )
        )

    core = {
        "schema_version": DECISION_SCHEMA,
        "status": "human_gold_frozen_pending_comparison",
        "p3_content_sha256": canonical_artifact_hash(p3),
        "p4_content_sha256": packet["content_sha256"],
        "enrolled_binding_content_sha256": enrolled_binding_source[
            "content_sha256"
        ],
        "decision_count": len(decisions),
        "decisions": decisions,
        "negative_actions": negative_action_vector(),
    }
    return {**core, "content_sha256": canonical_artifact_hash(core)}


def recompute_comparison(
    p3_manifest: Mapping[str, Any],
    p4_source: Mapping[str, Any],
    enrolled_binding_source: Mapping[str, Any],
    submission: Mapping[str, Any],
) -> dict[str, Any]:
    """Independently compare all three joined conditions with reviewed gold."""

    p3, packet, p3_slots = _validated_sources(p3_manifest, p4_source)
    enrolled_by_slot = _validated_enrolled_bindings(
        packet, enrolled_binding_source
    )
    submission_core = {
        key: value for key, value in submission.items() if key != "content_sha256"
    }
    decisions = submission.get("decisions")
    if (
        submission.get("schema_version") != DECISION_SCHEMA
        or submission.get("status") != "human_gold_frozen_pending_comparison"
        or submission.get("p3_content_sha256") != canonical_artifact_hash(p3)
        or submission.get("p4_content_sha256") != packet["content_sha256"]
        or submission.get("enrolled_binding_content_sha256")
        != enrolled_binding_source.get("content_sha256")
        or int(submission.get("decision_count") or 0) != 10
        or submission.get("content_sha256") != canonical_artifact_hash(submission_core)
        or not isinstance(decisions, list)
        or len(decisions) != 10
        or [str(item.get("slot_id") or "") for item in decisions if isinstance(item, Mapping)]
        != _expected_slots()
        or any((submission.get("negative_actions") or {}).values())
    ):
        _fail("The Plan 0062 human-gold submission is invalid.")
    cards_by_slot = {str(card["slot_id"]): card for card in packet["cards"]}
    expected_decisions = [
        _decision_for_selection(
            slot_id=str(decision["slot_id"]),
            selected=str(decision.get("selected_token") or ""),
            card=cards_by_slot[str(decision["slot_id"])],
            p3_outcome=p3_slots[str(decision["slot_id"])]["outcome"],
            enrolled_binding=enrolled_by_slot.get(str(decision["slot_id"])),
        )
        for decision in decisions
    ]
    if [dict(decision) for decision in decisions] != expected_decisions:
        _fail("The Plan 0062 human-gold decision meanings drifted.")

    gold = Counter(str(item.get("decision_type") or "") for item in decisions)
    named_types = {
        "canonical_context_proposal",
        "enrolled_voice_subject",
        "contextual_unlisted_suggestion",
        "new_person",
    }
    counters = {condition: Counter() for condition in sorted(EXPECTED_CONDITIONS)}
    rows: list[dict[str, Any]] = []
    all_wrong = 0
    high_confidence_wrong = 0
    for decision in decisions:
        slot_id = str(decision["slot_id"])
        decision_type = str(decision["decision_type"])
        target_person_id = str(decision.get("person_id") or "")
        named = decision_type in named_types
        condition_rows = []
        for evaluation in p3_slots[slot_id]["evaluations"]:
            condition = str(evaluation.get("condition") or "")
            counter = counters[condition]
            proposed = str(evaluation.get("proposed_person_id") or "")
            abstained = not proposed and evaluation.get("outcome") == "abstained"
            correct = bool(
                proposed
                and decision_type == "canonical_context_proposal"
                and proposed == target_person_id
            )
            wrong = bool(proposed and not correct)
            safe_abstention = bool(
                abstained and decision_type != "canonical_context_proposal"
            )
            inappropriate_abstention = bool(
                abstained and decision_type == "canonical_context_proposal"
            )
            coverage_gap = bool(abstained and named)
            confidence = float(evaluation.get("capped_confidence") or 0.0)
            source_failures = evaluation.get("source_failures")
            if not isinstance(source_failures, (list, tuple)):
                _fail("A Plan 0062 comparison evaluation has invalid provenance.")
            counter["evaluation_count"] += 1
            counter["proposal_count"] += int(bool(proposed))
            counter["correct_proposal_count"] += int(correct)
            counter["wrong_proposal_count"] += int(wrong)
            counter["safe_abstention_count"] += int(safe_abstention)
            counter["inappropriate_abstention_count"] += int(inappropriate_abstention)
            counter["named_identity_count"] += int(named)
            counter["named_identity_recalled_count"] += int(correct)
            counter["canonical_person_count"] += int(
                decision_type == "canonical_context_proposal"
            )
            counter["canonical_person_recalled_count"] += int(correct)
            counter["coverage_gap_count"] += int(coverage_gap)
            counter["source_failure_count"] += len(source_failures)
            all_wrong += int(wrong)
            high_confidence_wrong += int(wrong and confidence >= HIGH_CONFIDENCE_THRESHOLD)
            condition_rows.append(
                {
                    "condition": condition,
                    "evaluation_id": evaluation.get("evaluation_id"),
                    "outcome": evaluation.get("outcome"),
                    "proposed_person_id": proposed or None,
                    "correct_proposal": correct,
                    "wrong_proposal": wrong,
                    "safe_abstention": safe_abstention,
                    "inappropriate_abstention": inappropriate_abstention,
                    "named_identity_coverage_gap": coverage_gap,
                    "capped_confidence": confidence,
                }
            )
        rows.append(
            {
                "slot_id": slot_id,
                "human_decision_type": decision_type,
                "selected_token": decision["selected_token"],
                "conditions": condition_rows,
            }
        )

    condition_metrics: dict[str, dict[str, Any]] = {}
    for condition, counter in counters.items():
        metrics = dict(counter)
        metrics.update(
            {
                "canonical_recall": _ratio(
                    counter["canonical_person_recalled_count"],
                    counter["canonical_person_count"],
                ),
                "named_identity_recall": _ratio(
                    counter["named_identity_recalled_count"],
                    counter["named_identity_count"],
                ),
                "proposal_precision": _ratio(
                    counter["correct_proposal_count"], counter["proposal_count"]
                ),
            }
        )
        condition_metrics[condition] = metrics

    named_count = sum(gold[value] for value in named_types)
    new_enrollment_count = gold["contextual_unlisted_suggestion"] + gold["new_person"]
    existing_voice_binding_count = gold["enrolled_voice_subject"]
    if all_wrong:
        terminal_decision = "refine"
        recommendation = "refine_before_any_identity_or_biometric_apply"
    elif not named_count:
        terminal_decision = "refine"
        recommendation = "refine_identity_evidence_for_unresolved_speakers"
    elif new_enrollment_count or existing_voice_binding_count:
        terminal_decision = "advance"
        recommendation = "prepare_separate_identity_binding_and_biometric_enrollment_plan"
    else:
        terminal_decision = "advance"
        recommendation = "prepare_separate_reviewed_identity_apply_plan"

    def delta(metric: str) -> float | None:
        left = condition_metrics["combined"][metric]
        right = condition_metrics["context_only"][metric]
        if left is None or right is None:
            return None
        return round(float(left) - float(right), 4)

    core = {
        "schema_version": COMPARISON_SCHEMA,
        "status": "comparison_complete_no_apply",
        "submission_content_sha256": submission["content_sha256"],
        "p3_content_sha256": canonical_artifact_hash(p3),
        "p4_content_sha256": packet["content_sha256"],
        "enrolled_binding_content_sha256": enrolled_binding_source[
            "content_sha256"
        ],
        "recording_count": 3,
        "speaker_slot_count": 10,
        "condition_count": 3,
        "gold_metrics": {
            "named_identity_count": named_count,
            "canonical_person_count": gold["canonical_context_proposal"],
            "existing_voice_binding_candidate_count": existing_voice_binding_count,
            "new_biometric_enrollment_candidate_count": new_enrollment_count,
            "unresolved_count": gold["unresolved"],
            "decision_type_counts": dict(sorted(gold.items())),
        },
        "condition_metrics": condition_metrics,
        "combined_minus_context_only": {
            "canonical_recall": delta("canonical_recall"),
            "named_identity_recall": delta("named_identity_recall"),
            "proposal_precision": delta("proposal_precision"),
        },
        "wrong_proposal_count": all_wrong,
        "high_confidence_wrong_count": high_confidence_wrong,
        "terminal_decision": terminal_decision,
        "recommended_next_action": recommendation,
        "apply_authorized": False,
        "rows": rows,
        "negative_actions": negative_action_vector(),
    }
    return {**core, "content_sha256": canonical_artifact_hash(core)}


def _paths(runtime_root: Path, submission_sha256: str) -> dict[str, Path]:
    root = runtime_root.expanduser().absolute()
    run = root / f"p5-human-comparison-{submission_sha256[:24]}"
    return {
        "root": root,
        "run": run,
        "decision": run / "human-gold.json",
        "decision_receipt": run / "human-gold-receipt.json",
        "comparison": run / "comparison.json",
        "comparison_receipt": run / "comparison-receipt.json",
        "terminal": run / "terminal-audit.json",
    }


def freeze_human_comparison(
    answer_text: str,
    *,
    p3_manifest: Mapping[str, Any],
    p4_source: Mapping[str, Any],
    enrolled_binding_source: Mapping[str, Any],
    runtime_root: Path,
) -> dict[str, Any]:
    """Freeze exact P5 gold and comparison while preserving the no-apply boundary."""

    submission = parse_human_submission(
        answer_text,
        p3_manifest=p3_manifest,
        p4_source=p4_source,
        enrolled_binding_source=enrolled_binding_source,
    )
    comparison = recompute_comparison(
        p3_manifest, p4_source, enrolled_binding_source, submission
    )
    paths = _paths(runtime_root, submission["content_sha256"])
    if paths["terminal"].exists():
        return replay_human_comparison(
            submission_sha256=submission["content_sha256"],
            p3_manifest=p3_manifest,
            p4_source=p4_source,
            enrolled_binding_source=enrolled_binding_source,
            runtime_root=runtime_root,
        )
    if paths["run"].exists():
        _fail("A partial Plan 0062 P5 directory already exists.")
    ensure_private_tree(paths["root"], paths["run"])
    write_immutable_private_json(paths["decision"], submission)
    decision_receipt = {
        "schema_version": DECISION_RECEIPT_SCHEMA,
        "status": "human_gold_frozen",
        "submission_content_sha256": submission["content_sha256"],
        "decision_manifest_sha256": sha256_file(paths["decision"]),
        "decision_count": 10,
        "live_mutation_count": 0,
    }
    write_immutable_private_json(paths["decision_receipt"], decision_receipt)
    write_immutable_private_json(paths["comparison"], comparison)
    comparison_receipt = {
        "schema_version": COMPARISON_RECEIPT_SCHEMA,
        "status": "comparison_frozen",
        "submission_content_sha256": submission["content_sha256"],
        "comparison_content_sha256": comparison["content_sha256"],
        "comparison_manifest_sha256": sha256_file(paths["comparison"]),
        "terminal_decision": comparison["terminal_decision"],
        "live_mutation_count": 0,
    }
    write_immutable_private_json(paths["comparison_receipt"], comparison_receipt)
    independently_recomputed = (
        recompute_comparison(
            p3_manifest, p4_source, enrolled_binding_source, submission
        )
        == comparison
    )
    if not independently_recomputed:
        _fail("Independent Plan 0062 comparison recomputation disagreed.")
    terminal = {
        "schema_version": TERMINAL_SCHEMA,
        "status": "complete_no_apply",
        "submission_content_sha256": submission["content_sha256"],
        "comparison_content_sha256": comparison["content_sha256"],
        "terminal_decision": comparison["terminal_decision"],
        "recommended_next_action": comparison["recommended_next_action"],
        "metrics_recomputed": independently_recomputed,
        "decision_manifest_sha256": sha256_file(paths["decision"]),
        "comparison_manifest_sha256": sha256_file(paths["comparison"]),
        "apply_authorized": False,
        "live_mutation_count": 0,
        "negative_actions": negative_action_vector(),
    }
    write_immutable_private_json(paths["terminal"], terminal)
    return {
        **terminal,
        "terminal_path": str(paths["terminal"]),
        "idempotent_replay": False,
    }


def replay_human_comparison(
    *,
    submission_sha256: str,
    p3_manifest: Mapping[str, Any],
    p4_source: Mapping[str, Any],
    enrolled_binding_source: Mapping[str, Any],
    runtime_root: Path,
) -> dict[str, Any]:
    """Verify a frozen P5 comparison against current immutable source values."""

    if not re.fullmatch(r"[a-f0-9]{64}", str(submission_sha256 or "")):
        _fail("The Plan 0062 submission SHA-256 is invalid.")
    paths = _paths(runtime_root, submission_sha256)
    for key in ("decision", "decision_receipt", "comparison", "comparison_receipt", "terminal"):
        require_private_file(paths[key], paths["root"])
    submission = read_private_object(paths["decision"])
    comparison = read_private_object(paths["comparison"])
    decision_receipt = read_private_object(paths["decision_receipt"])
    comparison_receipt = read_private_object(paths["comparison_receipt"])
    terminal = read_private_object(paths["terminal"])
    expected = recompute_comparison(
        p3_manifest, p4_source, enrolled_binding_source, submission
    )
    if (
        submission.get("content_sha256") != submission_sha256
        or comparison != expected
        or decision_receipt.get("schema_version") != DECISION_RECEIPT_SCHEMA
        or decision_receipt.get("decision_manifest_sha256") != sha256_file(paths["decision"])
        or decision_receipt.get("submission_content_sha256") != submission_sha256
        or decision_receipt.get("decision_count") != 10
        or decision_receipt.get("live_mutation_count") != 0
        or comparison_receipt.get("schema_version") != COMPARISON_RECEIPT_SCHEMA
        or comparison_receipt.get("comparison_manifest_sha256") != sha256_file(paths["comparison"])
        or comparison_receipt.get("comparison_content_sha256")
        != comparison["content_sha256"]
        or comparison_receipt.get("terminal_decision")
        != comparison["terminal_decision"]
        or comparison_receipt.get("live_mutation_count") != 0
        or terminal.get("schema_version") != TERMINAL_SCHEMA
        or terminal.get("comparison_content_sha256") != comparison["content_sha256"]
        or terminal.get("decision_manifest_sha256") != sha256_file(paths["decision"])
        or terminal.get("comparison_manifest_sha256") != sha256_file(paths["comparison"])
        or terminal.get("terminal_decision") != comparison["terminal_decision"]
        or terminal.get("recommended_next_action")
        != comparison["recommended_next_action"]
        or terminal.get("metrics_recomputed") is not True
        or terminal.get("apply_authorized") is not False
        or terminal.get("live_mutation_count") != 0
        or any((terminal.get("negative_actions") or {}).values())
    ):
        _fail("The frozen Plan 0062 P5 evidence drifted.")
    return {
        **terminal,
        "terminal_path": str(paths["terminal"]),
        "idempotent_replay": True,
    }
