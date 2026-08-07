from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from acoustic_audio_derivatives import (
    ensure_private_tree,
    read_private_object,
    require_private_file,
    write_immutable_private_json,
)


BUNDLE_SCHEMA = "transcribe-audio.acoustic-shadow-evidence.v1"
REVIEW_SCHEMA = "transcribe-audio.acoustic-shadow-review-evidence.v1"
BATCH_SCHEMA = "transcribe-audio.acoustic-shadow-evidence-batch.v1"
DEFAULT_STATE_ROOT = Path("~/.local/state/transcribe-audio")
EVIDENCE_DIRNAME = "acoustic-shadow-evidence"
ALLOWLISTED_SUBJECT_IDS = frozenset(
    {
        "subject-7c24e8f41409c6f517291fe7",
        "subject-df34bc192c07bd86566fff12",
    }
)
NEGATIVE_ACTION_VECTOR = {
    "apply_speaker_assignments": False,
    "create_or_mutate_identities": False,
    "mutate_profiles_or_references": False,
    "write_external_provider": False,
    "enable_default_integration": False,
    "run_historical_reprocessing": False,
}
SHA256_RE = re.compile(r"^[a-f0-9]{64}$")
SPEAKER_REF_RE = re.compile(r"^SPEAKER_[1-9][0-9]*$")


class AcousticShadowEvidenceError(ValueError):
    """Raised when acoustic shadow evidence is unsafe or not source-bound."""


def canonical_hash(value: Any) -> str:
    body = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(body).hexdigest()


def _path_binding(path: str) -> str:
    return hashlib.sha256(path.encode("utf-8")).hexdigest()


def _document_key(document_id: str) -> str:
    if not document_id.strip():
        raise AcousticShadowEvidenceError("A source document ID is required.")
    return hashlib.sha256(document_id.encode("utf-8")).hexdigest()[:32]


def _normalize_row(raw: Mapping[str, Any]) -> dict[str, Any]:
    speaker_ref = str(raw.get("speaker_ref") or "")
    disposition = str(raw.get("disposition") or "")
    subject_id = raw.get("subject_id")
    confidence_band = str(raw.get("confidence_band") or "")
    if not SPEAKER_REF_RE.fullmatch(speaker_ref):
        raise AcousticShadowEvidenceError("A recording-local speaker reference is invalid.")
    if disposition not in {"assign", "review", "abstain"}:
        raise AcousticShadowEvidenceError("An acoustic disposition is invalid.")
    if disposition == "abstain":
        if subject_id is not None or confidence_band != "none":
            raise AcousticShadowEvidenceError("Abstention evidence must not carry an identity.")
    elif subject_id not in ALLOWLISTED_SUBJECT_IDS:
        raise AcousticShadowEvidenceError("A non-abstaining identity is not enrolled.")
    elif confidence_band not in {"low", "medium", "high"}:
        raise AcousticShadowEvidenceError("A proposal confidence band is invalid.")
    counts: dict[str, int] = {}
    for key in (
        "supporting_unit_count",
        "supporting_candidate_family_count",
        "opposing_unit_count",
    ):
        value = raw.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise AcousticShadowEvidenceError("Acoustic support counts are invalid.")
        counts[key] = value
    rationale = str(raw.get("rationale") or "").strip()
    if not rationale or len(rationale) > 500:
        raise AcousticShadowEvidenceError("Acoustic rationale is missing or oversized.")
    return {
        "speaker_ref": speaker_ref,
        "disposition": disposition,
        "subject_id": subject_id,
        "confidence_band": confidence_band,
        **counts,
        "rationale": rationale,
    }


def build_shadow_bundle(
    *,
    document_id: str,
    conversation_key: str,
    source_path: str,
    source_media_sha256: str,
    execution_content_sha256: str,
    identity_state_sha256: str,
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Build one read-only, document-bound acoustic evidence projection."""

    if not document_id.strip() or not conversation_key.strip() or not source_path:
        raise AcousticShadowEvidenceError("Evidence bindings are incomplete.")
    for digest in (
        source_media_sha256,
        execution_content_sha256,
        identity_state_sha256,
    ):
        if not SHA256_RE.fullmatch(digest):
            raise AcousticShadowEvidenceError("An evidence hash is invalid.")
    normalized_rows = [_normalize_row(item) for item in rows]
    if not normalized_rows:
        raise AcousticShadowEvidenceError("At least one eligible speaker is required.")
    refs = [item["speaker_ref"] for item in normalized_rows]
    if len(set(refs)) != len(refs):
        raise AcousticShadowEvidenceError("Speaker evidence is duplicated.")
    normalized_rows.sort(key=lambda item: int(item["speaker_ref"].split("_")[-1]))
    core = {
        "schema_version": BUNDLE_SCHEMA,
        "status": "complete_pending_human_review",
        "document_id": document_id,
        "conversation_key": conversation_key,
        "source_path_sha256": _path_binding(source_path),
        "source_media_sha256": source_media_sha256,
        "execution_content_sha256": execution_content_sha256,
        "identity_state_sha256": identity_state_sha256,
        "allowlisted_subject_ids": sorted(ALLOWLISTED_SUBJECT_IDS),
        "speaker_count": len(normalized_rows),
        "rows": normalized_rows,
        "contains_display_names": False,
        "non_authoritative": True,
        "requires_human_review": True,
        "action_vector": dict(NEGATIVE_ACTION_VECTOR),
    }
    return {**core, "content_sha256": canonical_hash(core)}


def validate_shadow_bundle(
    raw: Mapping[str, Any],
    *,
    document_id: str,
    conversation_key: str,
    source_path: str,
) -> dict[str, Any]:
    """Validate an immutable projection against the requested review context."""

    bundle = dict(raw)
    content_sha256 = bundle.pop("content_sha256", None)
    if (
        bundle.get("schema_version") != BUNDLE_SCHEMA
        or bundle.get("status") != "complete_pending_human_review"
        or bundle.get("document_id") != document_id
        or bundle.get("conversation_key") != conversation_key
        or bundle.get("source_path_sha256") != _path_binding(source_path)
        or set(bundle.get("allowlisted_subject_ids") or []) != ALLOWLISTED_SUBJECT_IDS
        or bundle.get("contains_display_names") is not False
        or bundle.get("non_authoritative") is not True
        or bundle.get("requires_human_review") is not True
        or bundle.get("action_vector") != NEGATIVE_ACTION_VECTOR
        or not SHA256_RE.fullmatch(str(bundle.get("source_media_sha256") or ""))
        or not SHA256_RE.fullmatch(str(bundle.get("execution_content_sha256") or ""))
        or not SHA256_RE.fullmatch(str(bundle.get("identity_state_sha256") or ""))
    ):
        raise AcousticShadowEvidenceError("Acoustic shadow evidence is unbound or unsafe.")
    rows = bundle.get("rows")
    if not isinstance(rows, list):
        raise AcousticShadowEvidenceError("Acoustic speaker evidence is missing.")
    normalized_rows = [_normalize_row(item) for item in rows if isinstance(item, Mapping)]
    if len(normalized_rows) != len(rows) or len(normalized_rows) != bundle.get("speaker_count"):
        raise AcousticShadowEvidenceError("The acoustic speaker denominator is incomplete.")
    if len({item["speaker_ref"] for item in normalized_rows}) != len(normalized_rows):
        raise AcousticShadowEvidenceError("Speaker evidence is duplicated.")
    if content_sha256 != canonical_hash(bundle):
        raise AcousticShadowEvidenceError("Acoustic shadow evidence hash drifted.")
    return {**bundle, "content_sha256": content_sha256}


def _document_dir(*, document_id: str, state_root: Path) -> tuple[Path, Path]:
    root = state_root.expanduser().absolute()
    directory = root / EVIDENCE_DIRNAME / "by-document" / _document_key(document_id)
    return root, directory


def publish_shadow_bundle(
    bundle: Mapping[str, Any],
    *,
    source_path: str,
    state_root: Path = DEFAULT_STATE_ROOT,
    activate: bool = True,
) -> dict[str, Any]:
    """Publish one immutable private projection for ordinary review reads."""

    validated = validate_shadow_bundle(
        bundle,
        document_id=str(bundle.get("document_id") or ""),
        conversation_key=str(bundle.get("conversation_key") or ""),
        source_path=source_path,
    )
    root, directory = _document_dir(
        document_id=str(validated["document_id"]),
        state_root=state_root,
    )
    ensure_private_tree(root, directory)
    path = directory / f"{validated['content_sha256']}.json"
    existed = path.exists()
    write_immutable_private_json(path, dict(validated))
    result = {
        "schema_version": REVIEW_SCHEMA,
        "status": "published",
        "document_id": validated["document_id"],
        "conversation_key": validated["conversation_key"],
        "execution_content_sha256": validated["execution_content_sha256"],
        "content_sha256": validated["content_sha256"],
        "path": str(path),
        "idempotent_replay": existed,
        "will_apply_speaker_assignments": False,
    }
    if activate:
        activation = activate_shadow_batch(
            [result],
            execution_content_sha256=validated["execution_content_sha256"],
            state_root=state_root,
        )
        result["activation_content_sha256"] = activation["content_sha256"]
    return result


def activate_shadow_batch(
    publications: Sequence[Mapping[str, Any]],
    *,
    execution_content_sha256: str,
    state_root: Path = DEFAULT_STATE_ROOT,
) -> dict[str, Any]:
    """Atomically make a complete set of already-published bundles visible."""

    if not _is_sha256(execution_content_sha256) or not publications:
        raise AcousticShadowEvidenceError("A complete execution binding is required.")
    bindings = []
    seen_documents: set[str] = set()
    root = state_root.expanduser().absolute()
    for raw in publications:
        document_id = str(raw.get("document_id") or "")
        conversation_key = str(raw.get("conversation_key") or "")
        content_sha256 = str(raw.get("content_sha256") or "")
        path = Path(str(raw.get("path") or ""))
        if (
            not document_id
            or document_id in seen_documents
            or not conversation_key
            or not _is_sha256(content_sha256)
            or raw.get("execution_content_sha256") != execution_content_sha256
        ):
            raise AcousticShadowEvidenceError("A batch publication binding is invalid.")
        require_private_file(path, root)
        bundle = read_private_object(path)
        if (
            bundle.get("document_id") != document_id
            or bundle.get("conversation_key") != conversation_key
            or bundle.get("content_sha256") != content_sha256
            or bundle.get("execution_content_sha256") != execution_content_sha256
        ):
            raise AcousticShadowEvidenceError("A published bundle is unbound.")
        seen_documents.add(document_id)
        bindings.append(
            {
                "document_id": document_id,
                "document_key": _document_key(document_id),
                "conversation_key": conversation_key,
                "content_sha256": content_sha256,
            }
        )
    bindings.sort(key=lambda item: item["document_id"])
    core = {
        "schema_version": BATCH_SCHEMA,
        "status": "complete_active_batch",
        "execution_content_sha256": execution_content_sha256,
        "document_count": len(bindings),
        "bindings": bindings,
        "complete": True,
        "will_apply_speaker_assignments": False,
        "will_mutate_identity_state": False,
    }
    payload = {**core, "content_sha256": canonical_hash(core)}
    directory = root / EVIDENCE_DIRNAME / "active-batches"
    ensure_private_tree(root, directory)
    path = directory / f"{execution_content_sha256}.json"
    existed = path.exists()
    write_immutable_private_json(path, payload)
    return {
        "schema_version": BATCH_SCHEMA,
        "status": "activated",
        "content_sha256": payload["content_sha256"],
        "execution_content_sha256": execution_content_sha256,
        "document_count": len(bindings),
        "path": str(path),
        "idempotent_replay": existed,
    }


def _empty_review(status: str, reason: str) -> dict[str, Any]:
    return {
        "schema_version": REVIEW_SCHEMA,
        "status": status,
        "reason": reason,
        "rows": [],
        "speaker_count": 0,
        "non_authoritative": True,
        "requires_human_review": status == "available",
        "will_apply_speaker_assignments": False,
        "will_mutate_identity_state": False,
    }


def load_for_review(
    *,
    document_id: str,
    conversation_key: str,
    source_path: str,
    state_root: Path = DEFAULT_STATE_ROOT,
) -> dict[str, Any]:
    """Load accepted evidence or a fail-closed read-side status."""

    if not document_id.strip() or not conversation_key.strip() or not source_path:
        return _empty_review("rejected", "invalid_review_binding")
    root, directory = _document_dir(document_id=document_id, state_root=state_root)
    active_dir = root / EVIDENCE_DIRNAME / "active-batches"
    if not active_dir.is_dir() or active_dir.is_symlink():
        return _empty_review("absent", "no_evidence")
    matches: list[tuple[Mapping[str, Any], Mapping[str, Any]]] = []
    try:
        for index_path in sorted(active_dir.glob("*.json")):
            require_private_file(index_path, root)
            index = read_private_object(index_path)
            candidate = dict(index)
            content_sha256 = candidate.pop("content_sha256", None)
            bindings = candidate.get("bindings")
            execution_content_sha256 = str(
                candidate.get("execution_content_sha256") or ""
            )
            if (
                candidate.get("schema_version") != BATCH_SCHEMA
                or candidate.get("status") != "complete_active_batch"
                or candidate.get("complete") is not True
                or candidate.get("will_apply_speaker_assignments") is not False
                or candidate.get("will_mutate_identity_state") is not False
                or not _is_sha256(execution_content_sha256)
                or index_path.name != f"{execution_content_sha256}.json"
                or not isinstance(bindings, list)
                or not bindings
                or candidate.get("document_count") != len(bindings)
                or any(not isinstance(binding, Mapping) for binding in bindings)
                or content_sha256 != canonical_hash(candidate)
            ):
                continue
            normalized_keys = {
                (
                    str(binding.get("document_id") or ""),
                    str(binding.get("document_key") or ""),
                    str(binding.get("conversation_key") or ""),
                    str(binding.get("content_sha256") or ""),
                )
                for binding in bindings
            }
            if (
                len(normalized_keys) != len(bindings)
                or len({item[0] for item in normalized_keys}) != len(bindings)
                or any(
                    not bound_document
                    or bound_document_key != _document_key(bound_document)
                    or not bound_conversation
                    or not _is_sha256(bound_content)
                    for (
                        bound_document,
                        bound_document_key,
                        bound_conversation,
                        bound_content,
                    ) in normalized_keys
                )
            ):
                continue
            for binding in bindings:
                if binding.get("document_id") == document_id:
                    matches.append((candidate, binding))
    except (OSError, ValueError):
        return _empty_review("rejected", "invalid_activation_index")
    if not matches:
        return _empty_review("absent", "no_evidence")
    if len(matches) != 1:
        return _empty_review("rejected", "ambiguous_evidence_versions")
    activation, binding = matches[0]
    if (
        binding.get("document_key") != _document_key(document_id)
        or binding.get("conversation_key") != conversation_key
    ):
        return _empty_review("rejected", "invalid_activation_binding")
    path = directory / f"{binding.get('content_sha256')}.json"
    try:
        require_private_file(path, root)
        bundle = validate_shadow_bundle(
            read_private_object(path),
            document_id=document_id,
            conversation_key=conversation_key,
            source_path=source_path,
        )
    except (AcousticShadowEvidenceError, OSError, ValueError):
        return _empty_review("rejected", "invalid_or_drifted_evidence")
    if (
        bundle["content_sha256"] != binding.get("content_sha256")
        or bundle["execution_content_sha256"]
        != activation.get("execution_content_sha256")
    ):
        return _empty_review("rejected", "invalid_activation_binding")
    return {
        "schema_version": REVIEW_SCHEMA,
        "status": "available",
        "reason": "validated_private_shadow_evidence",
        "content_sha256": bundle["content_sha256"],
        "source_media_sha256": bundle["source_media_sha256"],
        "execution_content_sha256": bundle["execution_content_sha256"],
        "identity_state_sha256": bundle["identity_state_sha256"],
        "allowlisted_subject_ids": bundle["allowlisted_subject_ids"],
        "rows": bundle["rows"],
        "speaker_count": bundle["speaker_count"],
        "non_authoritative": True,
        "requires_human_review": True,
        "will_apply_speaker_assignments": False,
        "will_mutate_identity_state": False,
    }


def review_fingerprint(
    *,
    document_id: str,
    conversation_key: str,
    source_path: str,
    state_root: Path = DEFAULT_STATE_ROOT,
) -> str:
    """Bind participant-identity caches to accepted shadow evidence state."""

    return canonical_hash(
        load_for_review(
            document_id=document_id,
            conversation_key=conversation_key,
            source_path=source_path,
            state_root=state_root,
        )
    )


def _is_sha256(value: Any) -> bool:
    return bool(SHA256_RE.fullmatch(str(value or "")))
