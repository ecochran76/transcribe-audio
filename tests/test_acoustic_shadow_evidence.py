from __future__ import annotations

import json
import stat
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import acoustic_shadow_evidence as shadow


def valid_bundle() -> dict:
    return shadow.build_shadow_bundle(
        document_id="document-1",
        conversation_key="conversation-1",
        source_path="/private/transcript-1.json",
        source_media_sha256="a" * 64,
        execution_content_sha256="b" * 64,
        identity_state_sha256="c" * 64,
        rows=[
            {
                "speaker_ref": "SPEAKER_1",
                "disposition": "assign",
                "subject_id": "subject-df34bc192c07bd86566fff12",
                "confidence_band": "medium",
                "supporting_unit_count": 7,
                "supporting_candidate_family_count": 3,
                "opposing_unit_count": 0,
                "rationale": "Frozen consensus evidence.",
            },
            {
                "speaker_ref": "SPEAKER_2",
                "disposition": "abstain",
                "subject_id": None,
                "confidence_band": "none",
                "supporting_unit_count": 0,
                "supporting_candidate_family_count": 0,
                "opposing_unit_count": 0,
                "rationale": "No threshold support.",
            },
        ],
    )


def test_build_shadow_bundle_accepts_only_enrolled_subject_ids() -> None:
    bundle = valid_bundle()

    assert bundle["speaker_count"] == 2
    assert bundle["allowlisted_subject_ids"] == sorted(shadow.ALLOWLISTED_SUBJECT_IDS)
    assert bundle["contains_display_names"] is False
    assert bundle["action_vector"] == shadow.NEGATIVE_ACTION_VECTOR
    assert shadow.validate_shadow_bundle(
        bundle,
        document_id="document-1",
        conversation_key="conversation-1",
        source_path="/private/transcript-1.json",
    )["content_sha256"] == bundle["content_sha256"]


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("subject_id", "Eric Cochran"),
        ("subject_id", "contact-123"),
        ("subject_id", "subject-not-enrolled"),
        ("speaker_ref", "Eric"),
    ],
)
def test_shadow_bundle_rejects_name_contact_and_unbound_identity_text(
    field: str,
    value: str,
) -> None:
    bundle = valid_bundle()
    bundle["rows"][0][field] = value
    bundle["content_sha256"] = shadow.canonical_hash(
        {key: item for key, item in bundle.items() if key != "content_sha256"}
    )

    with pytest.raises(shadow.AcousticShadowEvidenceError):
        shadow.validate_shadow_bundle(
            bundle,
            document_id="document-1",
            conversation_key="conversation-1",
            source_path="/private/transcript-1.json",
        )


def test_shadow_bundle_rejects_mutation_flags_and_binding_mismatch() -> None:
    bundle = valid_bundle()
    bundle["action_vector"]["apply_speaker_assignments"] = True
    bundle["content_sha256"] = shadow.canonical_hash(
        {key: item for key, item in bundle.items() if key != "content_sha256"}
    )

    with pytest.raises(shadow.AcousticShadowEvidenceError):
        shadow.validate_shadow_bundle(
            bundle,
            document_id="document-1",
            conversation_key="conversation-1",
            source_path="/private/transcript-1.json",
        )

    with pytest.raises(shadow.AcousticShadowEvidenceError):
        shadow.validate_shadow_bundle(
            valid_bundle(),
            document_id="wrong-document",
            conversation_key="conversation-1",
            source_path="/private/transcript-1.json",
        )


def test_publish_load_and_replay_are_private_and_idempotent(tmp_path: Path) -> None:
    state_root = tmp_path / "state"
    bundle = valid_bundle()

    first = shadow.publish_shadow_bundle(
        bundle,
        source_path="/private/transcript-1.json",
        state_root=state_root,
    )
    second = shadow.publish_shadow_bundle(
        bundle,
        source_path="/private/transcript-1.json",
        state_root=state_root,
    )
    loaded = shadow.load_for_review(
        document_id="document-1",
        conversation_key="conversation-1",
        source_path="/private/transcript-1.json",
        state_root=state_root,
    )

    assert first["idempotent_replay"] is False
    assert second["idempotent_replay"] is True
    assert loaded["status"] == "available"
    assert loaded["non_authoritative"] is True
    assert loaded["speaker_count"] == 2
    assert loaded["will_apply_speaker_assignments"] is False
    assert stat.S_IMODE(Path(first["path"]).stat().st_mode) == 0o600
    assert stat.S_IMODE(Path(first["path"]).parent.stat().st_mode) == 0o700


def test_load_fails_closed_on_tamper_or_ambiguous_versions(tmp_path: Path) -> None:
    state_root = tmp_path / "state"
    bundle = valid_bundle()
    published = shadow.publish_shadow_bundle(
        bundle,
        source_path="/private/transcript-1.json",
        state_root=state_root,
    )
    path = Path(published["path"])
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["rows"][0]["subject_id"] = "name-derived-identity"
    path.write_text(json.dumps(payload), encoding="utf-8")
    path.chmod(0o600)

    rejected = shadow.load_for_review(
        document_id="document-1",
        conversation_key="conversation-1",
        source_path="/private/transcript-1.json",
        state_root=state_root,
    )
    assert rejected["status"] == "rejected"
    assert rejected["rows"] == []

    path.unlink()
    shadow.publish_shadow_bundle(
        bundle,
        source_path="/private/transcript-1.json",
        state_root=state_root,
    )
    second_bundle = shadow.build_shadow_bundle(
        document_id="document-1",
        conversation_key="conversation-1",
        source_path="/private/transcript-1.json",
        source_media_sha256="a" * 64,
        execution_content_sha256="d" * 64,
        identity_state_sha256="c" * 64,
        rows=bundle["rows"],
    )
    shadow.publish_shadow_bundle(
        second_bundle,
        source_path="/private/transcript-1.json",
        state_root=state_root,
    )
    ambiguous = shadow.load_for_review(
        document_id="document-1",
        conversation_key="conversation-1",
        source_path="/private/transcript-1.json",
        state_root=state_root,
    )
    assert ambiguous["status"] == "rejected"
    assert ambiguous["reason"] == "ambiguous_evidence_versions"


def test_load_fails_closed_on_missing_review_binding(tmp_path: Path) -> None:
    loaded = shadow.load_for_review(
        document_id="",
        conversation_key="conversation-1",
        source_path="/private/transcript-1.json",
        state_root=tmp_path / "state",
    )

    assert loaded["status"] == "rejected"
    assert loaded["reason"] == "invalid_review_binding"


def test_load_rejects_rehashed_activation_with_drifted_binding(tmp_path: Path) -> None:
    state_root = tmp_path / "state"
    bundle = valid_bundle()
    shadow.publish_shadow_bundle(
        bundle,
        source_path="/private/transcript-1.json",
        state_root=state_root,
    )
    activation_path = (
        state_root
        / shadow.EVIDENCE_DIRNAME
        / "active-batches"
        / f"{bundle['execution_content_sha256']}.json"
    )
    activation = json.loads(activation_path.read_text(encoding="utf-8"))
    activation["bindings"][0]["conversation_key"] = "another-conversation"
    activation["content_sha256"] = shadow.canonical_hash(
        {
            key: value
            for key, value in activation.items()
            if key != "content_sha256"
        }
    )
    activation_path.write_text(json.dumps(activation), encoding="utf-8")
    activation_path.chmod(0o600)

    loaded = shadow.load_for_review(
        document_id="document-1",
        conversation_key="conversation-1",
        source_path="/private/transcript-1.json",
        state_root=state_root,
    )

    assert loaded["status"] == "rejected"
    assert loaded["reason"] == "invalid_activation_binding"
