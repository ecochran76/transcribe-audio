from __future__ import annotations

from copy import deepcopy

import pytest

import speaker_identity_plan0065_d0 as d0


def _content(value):
    return d0._content_addressed(value)


def _sources():
    selected = []
    p1_recordings = []
    p2_cases = []
    p3_recordings = []
    review_cases = []
    decisions = []
    measurement_rows = []
    for recording_index in range(12):
        document_id = f"doc-{recording_index:02d}"
        labels = ["A", "B", "C"] + (["D", "E", "F"] if recording_index == 11 else [])
        selected.append(
            {
                "document_id": document_id,
                "disposition": "selected_evaluation_candidate",
                "source_media_sha256": f"{recording_index + 1:064x}",
                "artifact_sha256": f"{recording_index + 101:064x}",
                "speaker_labels": labels,
            }
        )
        p1_slots = []
        p2_slots = []
        p3_slots = []
        for label in labels:
            speaker_ref = f"{document_id}::{label}"
            p1_slots.append(
                {"speaker_ref": speaker_ref, "probe_sha256": d0._hash(speaker_ref)}
            )
            p2_slots.append({"speaker_ref": speaker_ref})
            p3_slots.append({"speaker_ref": speaker_ref})
            review_cases.append(
                {
                    "speaker_ref": speaker_ref,
                    "clip_sha256": d0._hash(f"clip:{speaker_ref}"),
                    "recording_filename": f"recording-{recording_index}.m4a",
                }
            )
            decisions.append(
                {
                    "speaker_ref": speaker_ref,
                    "decision": "unresolved",
                    "person_id": None,
                    "note": "",
                }
            )
            measurement_rows.append({"speaker_ref": speaker_ref})
        p1_recordings.append(
            {"document_id": document_id, "speaker_slots": p1_slots}
        )
        p2_cases.append(
            {"document_id": document_id, "speaker_slots": p2_slots}
        )
        p3_recordings.append(
            {"document_id": document_id, "speaker_slots": p3_slots}
        )
    p0_manifest = {
        "evaluation_cohort": {"considered": selected},
        "reference_inventory": {
            "development_recording_hashes": ["f" * 64],
            "development_sources": [
                {
                    "source_sha256": "f" * 64,
                    "start_seconds": 1,
                    "end_seconds": 2,
                }
            ],
        },
        "prior_identity_exposure": {"document_ids": ["prior-doc"]},
    }
    return {
        "p0_manifest": p0_manifest,
        "p1_evidence": {"recordings": p1_recordings},
        "p2_cases": p2_cases,
        "p3_resolution": {"recordings": p3_recordings},
        "review_authority": {"cases": review_cases},
        "gold": {"decisions": decisions},
        "measurement": {"rows": measurement_rows},
    }


def test_exposure_set_aligns_all_39_rows_and_permanently_excludes_sources():
    exposure = d0.build_exposure_set(**_sources())

    assert len(exposure["full_recordings"]) == 12
    assert len(exposure["review_clips"]) == 39
    assert len(exposure["decision_rows"]) == 39
    assert len(exposure["recording_hashes"]) == 13
    assert exposure["document_ids"][0] == "doc-00"
    assert exposure["document_ids"][-1] == "prior-doc"
    assert not any(exposure["action_counts"].values())


def test_exposure_set_rejects_one_reordered_or_missing_condition_row():
    values = _sources()
    values["p3_resolution"]["recordings"][0]["speaker_slots"].reverse()

    with pytest.raises(d0.Plan0065D0Error, match="denominators do not align"):
        d0.build_exposure_set(**values)


def test_d0_manifest_requires_exact_zero_effect_activation_authority():
    plan64 = _content(
        {
            "lineage": {
                "terminal_content_sha256": d0.TERMINAL_SHA256,
                "terminal_decision": "withhold_p5",
            },
            "action_counts": dict(d0.ACTION_COUNTS),
        }
    )
    inventory = _content({"action_counts": dict(d0.ACTION_COUNTS)})
    provider = _content(
        {
            "did_start_session": False,
            "did_send_model_turn": False,
            "action_counts": dict(d0.ACTION_COUNTS),
        }
    )
    repository = {
        "clean": True,
        "module_blob_matches": True,
        "upstream_ahead": 0,
        "upstream_behind": 0,
    }

    manifest = d0.build_d0_manifest(
        plan0064_authority=plan64,
        inventory=inventory,
        provider=provider,
        repository=repository,
    )

    assert manifest["ready_packets"] == [
        "d1_acoustic_safety",
        "d2_contextual_evidence",
    ]
    assert manifest["blocked_packets"] == [
        "d3_joined_residual",
        "e0_fresh_authority",
    ]
    assert not any(manifest["action_counts"].values())

    unsafe = deepcopy(provider)
    unsafe["did_send_model_turn"] = True
    unsafe = _content(unsafe)
    with pytest.raises(d0.Plan0065D0Error, match="incomplete or unsafe"):
        d0.build_d0_manifest(
            plan0064_authority=plan64,
            inventory=inventory,
            provider=unsafe,
            repository=repository,
        )


def test_provider_readiness_contract_does_not_authorize_a_turn(monkeypatch):
    responses = {
        "/api/intelligence/providers": {
            "providers": [
                {
                    "id": "codex-app-server",
                    "status": "ready",
                    "ready": True,
                    "capabilities": {"persistent_sessions": True},
                }
            ]
        },
        "/api/intelligence/config": {
            "tasks": {
                "speaker_disambiguation": {
                    "task": "speaker_disambiguation",
                    "provider": "codex-app-server",
                    "model": "model-a",
                }
            }
        },
    }

    def fake_get(url):
        return responses[next(path for path in responses if url.endswith(path))]

    monkeypatch.setattr(d0, "_get_json", fake_get)
    readiness = d0.provider_readiness(base_url="http://local")

    assert readiness["status"] == "ready_for_bounded_context_execution"
    assert readiness["primary"]["ready"] is True
    assert readiness["fallback"] is None
    assert readiness["did_start_session"] is False
    assert readiness["did_send_model_turn"] is False
    assert not any(readiness["action_counts"].values())
