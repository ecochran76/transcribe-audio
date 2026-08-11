"""Plan 0066 A0 zero-effect activation and source-integrity freeze."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import speaker_identity_plan0065_d0 as plan0065_d0
import transcript_store
from acoustic_audio_derivatives import (
    ensure_private_tree,
    read_private_object,
    require_private_file,
    sha256_file,
    write_immutable_private_json,
)


ACTIVATION_COMMIT = "9471a896714a6ba99c8a58f0348faff4ab123e58"
SCHEMA_VERSION = "transcribe-audio.plan0066-a0-activation.v1"
RECEIPT_SCHEMA_VERSION = "transcribe-audio.plan0066-a0-receipt.v1"
DEFAULT_RUNTIME_ROOT = Path("~/.local/state/transcribe-audio/plan-0066")
DEFAULT_PLAN0065_D2_CASE_ROOT = Path(
    "~/.local/state/transcribe-audio/plan-0065/"
    "d2-execution-ef76ba3392ca28a27c695e54/cases"
)
DEFAULT_PLAN0065_TERMINAL = Path(
    "~/.local/state/transcribe-audio/plan-0065/"
    "terminal-d2-8d65f6be10259cd54a8e1c8b/terminal.json"
)
DEFAULT_STORE_ROOT = Path("~/.transcripts")
EFFECT_COUNTS = {
    "model_turns": 0,
    "source_transcript_writes": 0,
    "stored_transcript_writes": 0,
    "transcript_index_writes": 0,
    "speaker_assignment_writes": 0,
    "identity_writes": 0,
    "knowledge_writes": 0,
    "biometric_writes": 0,
    "provider_writes": 0,
    "graphiti_writes": 0,
    "external_writes": 0,
}


class Plan0066A0Error(ValueError):
    """Raised when activation authority cannot be frozen exactly."""


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _hash(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _content(value: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(value)
    result["content_sha256"] = _hash(result)
    return result


def build_activation_manifest(
    *,
    terminal: Mapping[str, Any],
    cases: Sequence[Mapping[str, Any]],
    document_bindings: Sequence[Mapping[str, Any]],
    reviewed_roster: Sequence[Mapping[str, Any]],
    provider_readiness: Mapping[str, Any],
) -> dict[str, Any]:
    """Build the closed-world A0 manifest from already-read authority."""

    slot_count = sum(len(case.get("speaker_slots") or []) for case in cases)
    if len(cases) != 12 or slot_count != 39:
        raise Plan0066A0Error("Plan 0065 development denominator drifted.")
    if len(document_bindings) != 12:
        raise Plan0066A0Error("A0 requires one source/index binding per case.")
    if len(reviewed_roster) != 6:
        raise Plan0066A0Error("A0 requires the complete six-person reviewed roster.")
    if any(not str(item.get("primary_name") or "").strip() for item in reviewed_roster):
        raise Plan0066A0Error("Every reviewed person requires a primary name.")
    if terminal.get("status") not in {"withhold", "plan0065_closed_withhold"}:
        raise Plan0066A0Error("Plan 0065 terminal is not the expected withhold.")
    if provider_readiness.get("did_send_model_turn") is not False:
        raise Plan0066A0Error("Readiness must not send a model turn.")
    ordered_cases = sorted(cases, key=lambda item: str(item.get("document_id") or ""))
    ordered_bindings = sorted(
        document_bindings,
        key=lambda item: str(item.get("document_id") or ""),
    )
    ordered_roster = sorted(
        reviewed_roster,
        key=lambda item: str(item.get("person_id") or ""),
    )
    return _content(
        {
            "schema_version": SCHEMA_VERSION,
            "status": "a0_authority_frozen_zero_effect",
            "activation_commit": ACTIVATION_COMMIT,
            "plan0065_terminal_content_sha256": terminal.get("content_sha256"),
            "plan0065_terminal_file_sha256": terminal.get("terminal_file_sha256"),
            "development_denominator": {
                "case_count": len(ordered_cases),
                "speaker_slot_count": slot_count,
                "case_content_sha256s": sorted(
                    str(item.get("content_sha256") or "") for item in ordered_cases
                ),
            },
            "document_bindings": ordered_bindings,
            "document_binding_set_sha256": _hash(ordered_bindings),
            "reviewed_roster": ordered_roster,
            "reviewed_roster_sha256": _hash(ordered_roster),
            "provider_readiness": dict(provider_readiness),
            "accepted_findings": ["F1", "F2", "F3", "F4"],
            "review_verification_mode": "closed_world",
            "effect_counts": dict(EFFECT_COUNTS),
        }
    )


def _document_bindings(
    cases: Sequence[Mapping[str, Any]], *, store_root: Path
) -> list[dict[str, Any]]:
    bindings: list[dict[str, Any]] = []
    with transcript_store.connect(store_root) as con:
        for case in cases:
            document_id = str(case.get("document_id") or "")
            row = con.execute(
                "SELECT * FROM documents WHERE id = ?",
                (document_id,),
            ).fetchone()
            if row is None:
                raise Plan0066A0Error(f"Missing transcript index row: {document_id}.")
            row_value = dict(row)
            source_path = Path(str(row_value.get("source_path") or "")).expanduser()
            stored_path = Path(str(row_value.get("stored_path") or "")).expanduser()
            bindings.append(
                {
                    "document_id": document_id,
                    "source_path": str(source_path),
                    "source_sha256": (
                        sha256_file(source_path) if source_path.is_file() else ""
                    ),
                    "stored_path": str(stored_path),
                    "stored_sha256": (
                        sha256_file(stored_path) if stored_path.is_file() else ""
                    ),
                    "index_row_sha256": _hash(row_value),
                }
            )
    return bindings


def _reviewed_roster(*, store_root: Path) -> list[dict[str, Any]]:
    with transcript_store.connect(store_root) as con:
        people = con.execute(
            """
            SELECT id, status, primary_name, metadata_json
            FROM knowledge_people
            WHERE status = 'reviewed'
            ORDER BY id
            """
        ).fetchall()
        result: list[dict[str, Any]] = []
        for person in people:
            sources = con.execute(
                """
                SELECT id, source_profile_id, provider_kind, account_id, tenant_id,
                       external_ref, label, relationship_scope,
                       identifier_authority, content_hash
                FROM knowledge_source_records
                WHERE person_id = ?
                ORDER BY id
                """,
                (str(person["id"]),),
            ).fetchall()
            identities = con.execute(
                """
                SELECT id, source_record_id, identity_kind, normalized_value,
                       display_value, authority, verified
                FROM knowledge_external_identities
                WHERE person_id = ?
                ORDER BY id
                """,
                (str(person["id"]),),
            ).fetchall()
            result.append(
                {
                    "person_id": str(person["id"]),
                    "status": str(person["status"]),
                    "primary_name": str(person["primary_name"]),
                    "metadata_json": str(person["metadata_json"]),
                    "source_records": [dict(item) for item in sources],
                    "external_identities": [dict(item) for item in identities],
                }
            )
    return result


def _read_plan0065_terminal(path: Path = DEFAULT_PLAN0065_TERMINAL) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    require_private_file(resolved, resolved.parent.parent)
    terminal = read_private_object(resolved)
    core = {key: value for key, value in terminal.items() if key != "content_sha256"}
    if terminal.get("content_sha256") != _hash(core):
        raise Plan0066A0Error("Plan 0065 terminal content drifted.")
    return {
        **terminal,
        "terminal_file_sha256": sha256_file(resolved),
    }


def freeze_activation(
    *,
    runtime_root: Path = DEFAULT_RUNTIME_ROOT,
    case_root: Path = DEFAULT_PLAN0065_D2_CASE_ROOT,
    store_root: Path = DEFAULT_STORE_ROOT,
    api_base_url: str = plan0065_d0.DEFAULT_API_BASE_URL,
) -> dict[str, Any]:
    root = runtime_root.expanduser().resolve()
    cases = [
        read_private_object(path)
        for path in sorted(case_root.expanduser().resolve().glob("*.json"))
    ]
    terminal = _read_plan0065_terminal()
    readiness = plan0065_d0.provider_readiness(base_url=api_base_url)
    manifest = build_activation_manifest(
        terminal=terminal,
        cases=cases,
        document_bindings=_document_bindings(cases, store_root=store_root),
        reviewed_roster=_reviewed_roster(store_root=store_root),
        provider_readiness=readiness,
    )
    run_root = root / f"a0-{manifest['content_sha256'][:24]}"
    manifest_path = run_root / "private-manifest.json"
    receipt_path = run_root / "receipt.json"
    if receipt_path.exists():
        receipt = read_private_object(receipt_path)
        require_private_file(manifest_path, root)
        if read_private_object(manifest_path) != manifest:
            raise Plan0066A0Error("A0 live authority drifted from its frozen manifest.")
        return {**receipt, "manifest_path": str(manifest_path), "idempotent_replay": True}
    ensure_private_tree(root, run_root)
    write_immutable_private_json(manifest_path, manifest)
    receipt = _content(
        {
            "schema_version": RECEIPT_SCHEMA_VERSION,
            "status": "a0_frozen_zero_effect",
            "activation_content_sha256": manifest["content_sha256"],
            "activation_file_sha256": sha256_file(manifest_path),
            "case_count": 12,
            "speaker_slot_count": 39,
            "reviewed_person_count": 6,
            "model_turn_count": 0,
            "effect_counts": dict(EFFECT_COUNTS),
        }
    )
    write_immutable_private_json(receipt_path, receipt)
    return {**receipt, "manifest_path": str(manifest_path), "idempotent_replay": False}


if __name__ == "__main__":
    print(json.dumps(freeze_activation(), indent=2, sort_keys=True))
