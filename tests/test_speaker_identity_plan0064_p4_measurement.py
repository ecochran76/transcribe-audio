from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest

import speaker_identity_plan0064_p4_measurement as p4m


def _view(disposition: str = "abstain", *, reason: str = "no_candidate"):
    return {
        "disposition": disposition,
        "reason_code": reason,
        "candidate_person_id": None,
        "alternative_person_ids": [],
        "contradiction_evidence_ids": [],
    }


def _sources(*, accepted_pattern: bool = False):
    people = [
        {"person_id": f"person-{index}", "display_name": f"Person {index}"}
        for index in range(1, 4)
    ]
    cases = []
    slots = []
    for index in range(39):
        speaker_ref = f"document-{index // 3:02d}::SPEAKER_{index % 3 + 1}"
        cases.append({"speaker_ref": speaker_ref})
        acoustic = {
            **_view(),
            "confidence_band": "none",
            "supporting_model_count": 0,
            "probe_sha256": "",
        }
        context = {**_view(), "candidates": []}
        combined = _view()
        residual = _view()
        if accepted_pattern and index < 3:
            person_id = f"person-{index + 1}"
            acoustic.update(
                {
                    "disposition": "candidate",
                    "reason_code": "multi_model_acoustic_support",
                    "candidate_person_id": person_id,
                    "alternative_person_ids": [person_id],
                    "confidence_band": "high",
                    "supporting_model_count": 2,
                    "probe_sha256": f"{index + 1}" * 64,
                }
            )
            context.update(
                {
                    "disposition": "candidate",
                    "reason_code": "context_candidate",
                    "candidate_person_id": person_id,
                    "alternative_person_ids": [person_id],
                    "candidates": [
                        {
                            "status": "candidate_match",
                            "prepared_person_id": person_id,
                            "transcript_clue_ids": [f"clue-{index}"],
                            "provenance_source_ids": [f"source-{index}"],
                        }
                    ],
                }
            )
            if index < 2:
                combined.update(
                    {
                        "disposition": "candidate",
                        "reason_code": "pillar_agreement",
                        "candidate_person_id": person_id,
                        "alternative_person_ids": [person_id],
                    }
                )
                residual = deepcopy(combined)
            else:
                combined.update(
                    {
                        "disposition": "review",
                        "reason_code": "context_only_support",
                        "alternative_person_ids": [person_id],
                    }
                )
                residual.update(
                    {
                        "disposition": "candidate",
                        "reason_code": "two_known_plus_one_independently_supported_residual",
                        "candidate_person_id": person_id,
                        "alternative_person_ids": [person_id],
                    }
                )
        slots.append(
            {
                "speaker_ref": speaker_ref,
                "speaker_label": f"SPEAKER_{index % 3 + 1}",
                "acoustic": acoustic,
                "context": context,
                "combined": combined,
                "residual_policy": residual,
            }
        )
    authority = p4m._content_addressed(
        {
            "schema_version": "authority",
            "cases": cases,
            "people": people,
            "action_counts": dict(p4m.ACTION_COUNTS),
        }
    )
    resolution = p4m._content_addressed(
        {
            "schema_version": "resolution",
            "recordings": [
                {
                    "document_id": "synthetic",
                    "speaker_slots": slots,
                }
            ],
            "action_counts": dict(p4m.ACTION_COUNTS),
        }
    )
    decisions = [
        {
            "speaker_ref": case["speaker_ref"],
            "decision": "canonical_person" if index < 3 else "unresolved",
            "person_id": f"person-{index + 1}" if index < 3 else None,
            "note": "",
        }
        for index, case in enumerate(cases)
    ]
    submission = {
        "schema_version": p4m.DECISION_SCHEMA,
        "authority_content_sha256": authority["content_sha256"],
        "decisions": decisions,
    }
    return authority, resolution, submission


def _development_gate(*, passed: bool = True):
    return p4m._content_addressed(
        {
            "schema_version": p4m.DEVELOPMENT_GATE_SCHEMA,
            "source_corpus": "plan0063_reviewed_three_conversation",
            "replay_exact": passed,
            "high_support_wrong_count": 0,
            "combined_correct_count": 2,
            "residual_correct_count": 1,
            "action_counts": dict(p4m.ACTION_COUNTS),
        }
    )


def test_normalize_human_gold_requires_exact_complete_ordered_allowlisted_export():
    authority, resolution, submission = _sources()
    gold = p4m.normalize_human_gold(
        submission, authority=authority, resolution=resolution
    )
    assert gold["decision_count"] == 39
    assert gold["decision_type_counts"] == {
        "canonical_person": 3,
        "unresolved": 36,
    }

    incomplete = deepcopy(submission)
    incomplete["decisions"].pop()
    with pytest.raises(p4m.Plan0064P4MeasurementError):
        p4m.normalize_human_gold(
            incomplete, authority=authority, resolution=resolution
        )

    escaped = deepcopy(submission)
    escaped["decisions"][0]["person_id"] = "person-not-allowed"
    with pytest.raises(p4m.Plan0064P4MeasurementError):
        p4m.normalize_human_gold(
            escaped, authority=authority, resolution=resolution
        )

    reordered = deepcopy(submission)
    reordered["decisions"][0], reordered["decisions"][1] = (
        reordered["decisions"][1],
        reordered["decisions"][0],
    )
    with pytest.raises(p4m.Plan0064P4MeasurementError):
        p4m.normalize_human_gold(
            reordered, authority=authority, resolution=resolution
        )


def test_bridge_human_gold_authority_allows_only_filename_display_revision():
    authority, resolution, submission = _sources()
    source_cases = []
    target_cases = []
    filename_rows = []
    seen_documents = set()
    for index, case in enumerate(authority["cases"]):
        speaker_ref = case["speaker_ref"]
        _, speaker_label = speaker_ref.split("::", 1)
        document_id = f"document-{min(index // 3, 11):02d}"
        base = {
            "speaker_ref": speaker_ref,
            "document_id": document_id,
            "speaker_label": speaker_label,
            "recording_label": f"Recording {index // 3 + 1}",
            "clip_relative_path": f"clips/{index:02d}.wav",
            "clip_sha256": f"{index + 1:064x}",
        }
        source_cases.append(base)
        filename = f"recording-{min(index // 3, 11) + 1}.m4a"
        target_cases.append({**base, "recording_filename": filename})
        if document_id not in seen_documents:
            filename_rows.append(
                {"document_id": document_id, "recording_filename": filename}
            )
            seen_documents.add(document_id)
    shared = {
        "status": "p4_human_gold_required",
        "human_decision_count": 0,
        "model_predictions_visible": False,
        "p3_receipt_content_sha256": "b" * 64,
        "p3_resolution_content_sha256": resolution["content_sha256"],
        "people": authority["people"],
        "action_counts": dict(p4m.ACTION_COUNTS),
    }
    source = p4m._content_addressed(
        {
            "schema_version": p4m.LEGACY_REVIEW_AUTHORITY_SCHEMA,
            **shared,
            "preview_content_sha256": "c" * 64,
            "cases": source_cases,
        }
    )
    target = p4m._content_addressed(
        {
            "schema_version": p4m.CURRENT_REVIEW_AUTHORITY_SCHEMA,
            **shared,
            "preview_content_sha256": "d" * 64,
            "recording_filename_set_sha256": p4m._hash(filename_rows),
            "cases": target_cases,
        }
    )
    submission["authority_content_sha256"] = source["content_sha256"]

    rebound, bridge = p4m.bridge_human_gold_authority(
        submission,
        source_authority=source,
        target_authority=target,
        resolution=resolution,
    )

    assert rebound["decisions"] == submission["decisions"]
    assert rebound["authority_content_sha256"] == target["content_sha256"]
    assert bridge["changed_fields"] == ["authority_content_sha256"]
    assert bridge["decision_count"] == 39
    assert not any(bridge["action_counts"].values())

    changed_case = deepcopy(target)
    changed_case["cases"][0]["clip_sha256"] = "e" * 64
    changed_case = p4m._content_addressed(changed_case)
    with pytest.raises(p4m.Plan0064P4MeasurementError):
        p4m.bridge_human_gold_authority(
            submission,
            source_authority=source,
            target_authority=changed_case,
            resolution=resolution,
        )


def test_measurement_forbids_vacuous_zero_candidate_acceptance():
    authority, resolution, submission = _sources()
    gold = p4m.normalize_human_gold(
        submission, authority=authority, resolution=resolution
    )
    measurement = p4m.recompute_measurement(
        authority=authority,
        resolution=resolution,
        gold=gold,
        development_gate=_development_gate(),
    )
    gate = measurement["acceptance_gate"]
    assert gate["automatic_local_acceptance_ready"] is False
    assert "combined_correct_acceptance_observed" in gate["failed_checks"]
    assert "residual_correct_acceptance_observed" in gate["failed_checks"]
    assert measurement["terminal_decision"] == "withhold_p5"
    assert measurement["apply_authorized"] is False


def test_measurement_treats_single_model_review_identity_as_non_candidate():
    authority, resolution, submission = _sources()
    acoustic = resolution["recordings"][0]["speaker_slots"][0]["acoustic"]
    acoustic.update(
        {
            "disposition": "review",
            "reason_code": "single_model_acoustic_support",
            "candidate_person_id": "person-1",
            "alternative_person_ids": ["person-1"],
            "confidence_band": "medium",
            "supporting_model_count": 1,
            "probe_sha256": "1" * 64,
        }
    )
    resolution = p4m._content_addressed(resolution)
    gold = p4m.normalize_human_gold(
        submission, authority=authority, resolution=resolution
    )

    measurement = p4m.recompute_measurement(
        authority=authority,
        resolution=resolution,
        gold=gold,
        development_gate=_development_gate(),
    )

    assert measurement["condition_metrics"]["acoustic"]["review_count"] == 1
    assert measurement["condition_metrics"]["acoustic"]["candidate_count"] == 0
    assert measurement["condition_metrics"]["acoustic"]["correct_candidate_count"] == 0
    assert measurement["rows"][0]["conditions"][0]["proposed_person_id"] is None


def test_measurement_advances_only_with_correct_join_residual_lineage_and_development():
    authority, resolution, submission = _sources(accepted_pattern=True)
    gold = p4m.normalize_human_gold(
        submission, authority=authority, resolution=resolution
    )
    measurement = p4m.recompute_measurement(
        authority=authority,
        resolution=resolution,
        gold=gold,
        development_gate=_development_gate(),
    )
    assert measurement["condition_metrics"]["combined"]["correct_candidate_count"] == 2
    assert measurement["condition_metrics"]["residual_policy"]["correct_candidate_count"] == 3
    assert measurement["condition_metrics"]["residual_policy"][
        "residual_rule_correct_count"
    ] == 1
    assert measurement["condition_metrics"]["residual_policy"][
        "candidate_lineage_completeness"
    ] == 1.0
    assert measurement["acceptance_gate"]["failed_checks"] == []
    assert measurement["acceptance_gate"]["automatic_local_acceptance_ready"] is True
    assert measurement["terminal_decision"] == "advance_to_p5"
    assert measurement["apply_authorized"] is False

    without_development = p4m.recompute_measurement(
        authority=authority,
        resolution=resolution,
        gold=gold,
    )
    assert without_development["terminal_decision"] == "withhold_p5"
    assert "reviewed_development_replay_passed" in without_development[
        "acceptance_gate"
    ]["failed_checks"]


def test_freeze_and_replay_are_private_immutable_and_zero_effect(tmp_path, monkeypatch):
    authority, resolution, submission = _sources(accepted_pattern=True)
    monkeypatch.setattr(
        p4m,
        "_authorities",
        lambda _p0, *, runtime_root: (authority, {"content_sha256": "p3"}, resolution),
    )
    runtime_root = tmp_path / "plan-0064"
    first = p4m.freeze_human_gold_and_measurement(
        submission,
        p0_content_sha256="a" * 64,
        runtime_root=runtime_root,
        development_gate=_development_gate(),
    )
    replay = p4m.replay_human_gold_and_measurement(
        gold_content_sha256=first["human_gold_content_sha256"],
        p0_content_sha256="a" * 64,
        runtime_root=runtime_root,
        development_gate=_development_gate(),
    )
    assert first["idempotent_replay"] is False
    assert replay["idempotent_replay"] is True
    assert replay["terminal_decision"] == "advance_to_p5"
    assert replay["apply_authorized"] is False
    assert not any(replay["action_counts"].values())
    assert Path(first["private_terminal_path"]).stat().st_mode & 0o777 == 0o600
