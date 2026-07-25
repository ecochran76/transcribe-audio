"""Oldest-forward evaluation campaign orchestration for speaker identity."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
import unicodedata
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import intelligence_config
import provenance_config
import speaker_identity_preprocess
from transcript_store import connect, init_db, store_dir

MANIFEST_SCHEMA_VERSION = "transcribe-audio.speaker-evaluation-campaign-manifest.v1"
DEFAULT_CAMPAIGN_ROOT = Path(
    "~/.local/state/transcribe-audio/speaker-evaluation-campaigns"
)
DEFAULT_STATE_ROOT = Path("~/.local/state/transcribe-audio")
RUBRIC_VERSIONS = {
    "calendar_association": "calendar-association.v1",
    "person_link": "person-link.v1",
    "speaker_identity": "speaker-identity.v1",
}
DISPOSITION_RULE_VERSION = "oldest-forward-disposition.v1"


def _json_object(value: Any) -> dict[str, Any]:
    try:
        payload = json.loads(str(value or "{}"))
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def _recording_time(row: Any, payload: dict[str, Any]) -> tuple[datetime, str, str]:
    candidates = (
        ("recording_start", payload.get("recording_start")),
        ("generated_at", row["generated_at"]),
        ("updated_at", row["updated_at"]),
    )
    for source, value in candidates:
        text = str(value or "").strip()
        if not text:
            continue
        try:
            parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
        except ValueError:
            continue
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc), text, source
    return datetime.max.replace(tzinfo=timezone.utc), "", "unavailable"


def _artifact_access(row: Any) -> dict[str, Any]:
    source_path = Path(str(row["source_path"] or "")).expanduser()
    stored_path = Path(str(row["stored_path"] or "")).expanduser()
    source_accessible = source_path.is_file()
    stored_accessible = stored_path.is_file()
    return {
        "source_path": str(row["source_path"] or ""),
        "stored_path": str(row["stored_path"] or ""),
        "source_accessible": source_accessible,
        "stored_accessible": stored_accessible,
        "selected_location": (
            "source"
            if source_accessible
            else "stored"
            if stored_accessible
            else "unavailable"
        ),
    }


def _content_fingerprint(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text).lower()
    normalized = re.sub(r"[^a-z0-9]+", " ", normalized)
    normalized = " ".join(normalized.split())
    if not normalized:
        return ""
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _repository_state() -> dict[str, Any]:
    repo_root = Path(__file__).resolve().parent
    commit_result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        check=False,
        capture_output=True,
        text=True,
    )
    status_result = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=repo_root,
        check=False,
        capture_output=True,
        text=True,
    )
    return {
        "commit": commit_result.stdout.strip() or "unavailable",
        "dirty_tree": bool(status_result.stdout.strip()),
    }


def _sha256_json(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def preview_campaign(
    *,
    store_root: Optional[Path] = None,
    runtime_root: Optional[Path] = None,
    state_root: Optional[Path] = None,
    batch_size: int = 10,
) -> dict[str, Any]:
    """Return a deterministic, read-only oldest-forward campaign manifest."""
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1.")
    selected_store_root = store_dir(store_root)
    selected_runtime_root = (runtime_root or DEFAULT_CAMPAIGN_ROOT).expanduser()
    selected_state_root = (state_root or DEFAULT_STATE_ROOT).expanduser()
    with connect(selected_store_root) as con:
        init_db(con)
        rows = con.execute(
            "SELECT * FROM documents WHERE kind = 'transcript'"
        ).fetchall()

    prepared: list[tuple[datetime, str, dict[str, Any]]] = []
    for row in rows:
        payload = _json_object(row["json_payload"])
        sort_time, recording_time, recording_time_source = _recording_time(
            row, payload
        )
        access = _artifact_access(row)
        utterances = (
            payload.get("utterances")
            if isinstance(payload.get("utterances"), list)
            else []
        )
        transcript_text = " ".join(
            str(utterance.get("text") or "").strip()
            for utterance in utterances
            if isinstance(utterance, dict)
        ).strip()
        if access["selected_location"] == "unavailable":
            disposition = "artifact_unavailable"
            disposition_reason = "no_accessible_source_or_stored_artifact"
        elif len(utterances) <= 1:
            disposition = "incomplete"
            disposition_reason = "one_or_zero_utterances"
        elif len(transcript_text) < 250:
            disposition = "incomplete"
            disposition_reason = "transcript_text_under_250_characters"
        else:
            disposition = "needs_operator_classification"
            disposition_reason = "operator_review_required"
        item = {
            "document_id": str(row["id"]),
            "recording_time": recording_time,
            "recording_time_source": recording_time_source,
            "artifact_sha256": str(row["artifact_sha256"] or ""),
            "artifact": access,
            "utterance_count": len(utterances),
            "transcript_text_chars": len(transcript_text),
            "content_fingerprint": _content_fingerprint(transcript_text),
            "speaker_labels": sorted(
                {
                    str(utterance.get("speaker") or "").strip()
                    for utterance in utterances
                    if isinstance(utterance, dict)
                    and str(utterance.get("speaker") or "").strip()
                }
            ),
            "disposition": disposition,
            "disposition_reason": disposition_reason,
        }
        prepared.append((sort_time, str(row["id"]), item))

    prepared.sort(key=lambda entry: (entry[0], entry[1]))
    items = [entry[2] for entry in prepared]
    for rank, item in enumerate(items, start=1):
        item["chronological_rank"] = rank

    fingerprint_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in items:
        fingerprint = str(item["content_fingerprint"])
        if fingerprint:
            fingerprint_groups[fingerprint].append(item)
    duplicate_cluster_count = 0
    for fingerprint, group in sorted(fingerprint_groups.items()):
        if len(group) < 2:
            continue
        duplicate_cluster_count += 1
        canonical = group[0]
        cluster_id = f"duplicate-{fingerprint[:16]}"
        for item in group:
            item["duplicate_cluster_id"] = cluster_id
            item["duplicate_evidence"] = "exact_normalized_transcript"
        for item in group[1:]:
            if item["disposition"] == "needs_operator_classification":
                item["disposition"] = "duplicate_member"
                item["disposition_reason"] = "exact_normalized_transcript_duplicate"
                item["duplicate_of_document_id"] = canonical["document_id"]

    reviewable_items = [
        item
        for item in items
        if item["disposition"] == "needs_operator_classification"
    ]
    for index, item in enumerate(reviewable_items):
        if index < batch_size:
            item["candidate_role"] = "gold_review_candidate"
        elif index < batch_size * 2:
            item["candidate_role"] = "blind_holdout_candidate"
        else:
            item["candidate_role"] = "future_candidate"
    for item in items:
        item.setdefault("candidate_role", "excluded_pending_disposition")

    cursor_item = next(
        (
            item
            for item in items
            if item["disposition"] == "needs_operator_classification"
        ),
        None,
    )
    disposition_counts = Counter(item["disposition"] for item in items)
    algorithm = _repository_state()
    model_route = intelligence_config.resolve_task_config(
        intelligence_config.TASK_SPEAKER_DISAMBIGUATION
    ).to_dict()
    provenance_snapshot = provenance_config.all_config(
        state_root=selected_state_root
    )
    provenance_config_fingerprint = _sha256_json(provenance_snapshot)
    manifest_basis = {
        "algorithm": algorithm,
        "batch_size": batch_size,
        "documents": [
            {
                "artifact_sha256": item["artifact_sha256"],
                "disposition": item["disposition"],
                "document_id": item["document_id"],
                "recording_time": item["recording_time"],
            }
            for item in items
        ],
        "model_route": model_route,
        "provenance_config_fingerprint": provenance_config_fingerprint,
        "rubric_versions": RUBRIC_VERSIONS,
        "rule_version": DISPOSITION_RULE_VERSION,
    }
    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "manifest_id": f"manifest-{_sha256_json(manifest_basis)[:20]}",
        "mode": "preview",
        "store_root": str(selected_store_root),
        "runtime_root": str(selected_runtime_root),
        "state_root": str(selected_state_root),
        "batch_size": batch_size,
        "algorithm": algorithm,
        "model_route": model_route,
        "provenance_config_fingerprint": provenance_config_fingerprint,
        "provenance_snapshot_policy": "fresh_retrieval",
        "rubric_versions": dict(RUBRIC_VERSIONS),
        "schema_versions": {
            "clue_packet": (
                speaker_identity_preprocess.SPEAKER_CLUE_PACKET_SCHEMA_VERSION
            ),
            "clue_discovery_packet": (
                speaker_identity_preprocess.CLUE_DISCOVERY_PACKET_SCHEMA_VERSION
            ),
            "clue_discovery_readout": (
                speaker_identity_preprocess.CLUE_DISCOVERY_READOUT_SCHEMA_VERSION
            ),
            "identity_evaluation_packet": (
                speaker_identity_preprocess.IDENTITY_EVALUATION_PACKET_SCHEMA_VERSION
            ),
            "speaker_identity_readout": (
                speaker_identity_preprocess.SPEAKER_IDENTITY_READOUT_SCHEMA_VERSION
            ),
            "disposition_rules": DISPOSITION_RULE_VERSION,
        },
        "cursor": {
            "chronological_rank": (
                int(cursor_item["chronological_rank"]) if cursor_item else None
            ),
            "document_id": str(cursor_item["document_id"]) if cursor_item else "",
        },
        "summary": {
            "total_rows": len(items),
            "disposition_counts": dict(sorted(disposition_counts.items())),
            "duplicate_cluster_count": duplicate_cluster_count,
            "gold_review_candidate_count": min(
                len(reviewable_items), batch_size
            ),
            "blind_holdout_candidate_count": min(
                max(len(reviewable_items) - batch_size, 0),
                batch_size,
            ),
        },
        "items": items,
        "will_write_campaign_state": False,
        "will_execute_app_intelligence": False,
        "will_perform_external_write": False,
    }


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the oldest-forward speaker identity evaluation campaign."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    preview = subparsers.add_parser(
        "preview",
        help="Print a deterministic campaign manifest without writing state.",
    )
    preview.add_argument("--store-root", type=Path)
    preview.add_argument("--runtime-root", type=Path)
    preview.add_argument("--state-root", type=Path)
    preview.add_argument("--batch-size", type=int, default=10)
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    if args.command == "preview":
        manifest = preview_campaign(
            store_root=args.store_root,
            runtime_root=args.runtime_root,
            state_root=args.state_root,
            batch_size=args.batch_size,
        )
        print(json.dumps(manifest, indent=2, sort_keys=True))
        return 0
    raise ValueError(f"Unsupported campaign command: {args.command}")


if __name__ == "__main__":
    sys.exit(main())
