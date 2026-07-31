"""Build and freeze the private Plan 0037 acoustic evaluation corpus."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sqlite3
import stat
import tempfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Optional

import transcript_store


CORPUS_SCHEMA_VERSION = "transcribe-audio.acoustic-evaluation-corpus.v1"
FREEZE_TOKEN = "FREEZE_ACOUSTIC_EVALUATION_CORPUS"
HARDEN_TOKEN = "HARDEN_ACOUSTIC_EVALUATION_SOURCES"
DEFAULT_RUNTIME_ROOT = Path("~/.local/state/transcribe-audio/plan-0037")
DEFAULT_CAMPAIGN_ROOT = Path(
    "~/.local/state/transcribe-audio/speaker-evaluation-campaigns"
)
SPLITS = ("development", "calibration", "evaluation")
CONDITION_FIELDS = (
    "channel",
    "device",
    "noise",
    "overlap",
    "telephone_bandwidth",
    "usable_duration_band",
)
LEGACY_OPERATOR_REVIEW_METHODS = {"transcript_and_calendar"}


class CorpusError(ValueError):
    """Raised when a corpus cannot be frozen without weakening its gates."""


def _utc_now() -> str:
    return (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def _read_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CorpusError(f"Corpus source is not valid JSON: {path}") from exc
    if not isinstance(value, dict):
        raise CorpusError(f"Corpus source must contain an object: {path}")
    return value


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _ensure_private_tree(root: Path, leaf: Path) -> None:
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(root, 0o700)
    relative = leaf.relative_to(root)
    current = root
    for part in relative.parts:
        current /= part
        current.mkdir(exist_ok=True, mode=0o700)
        os.chmod(current, 0o700)


def _write_private_json(path: Path, payload: dict[str, Any]) -> Path:
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True, ensure_ascii=False)
            stream.write("\n")
        os.replace(temporary_name, path)
        os.chmod(path, 0o600)
    except Exception:
        try:
            Path(temporary_name).unlink()
        except OSError:
            pass
        raise
    return path


def _sha256_file(path: Path) -> str:
    return transcript_store.sha256_file(path)


def _latest_gold_records(index: dict[str, Any]) -> list[dict[str, Any]]:
    records = index.get("records")
    if not isinstance(records, list):
        raise CorpusError("Gold index records must be a list.")
    latest: dict[str, dict[str, Any]] = {}
    for record in records:
        if not isinstance(record, dict):
            raise CorpusError("Gold index record must be an object.")
        document_id = str(record.get("document_id") or "")
        if not document_id:
            raise CorpusError("Gold index record is missing document_id.")
        latest[document_id] = record
    return sorted(
        latest.values(),
        key=lambda item: (
            int(item.get("chronological_rank") or 0),
            str(item.get("document_id") or ""),
        ),
    )


def _split_for_conversation(conversation_id: str) -> str:
    bucket = int(hashlib.sha256(conversation_id.encode("utf-8")).hexdigest()[:8], 16)
    percentile = bucket % 100
    if percentile < 60:
        return "development"
    if percentile < 80:
        return "calibration"
    return "evaluation"


def _subject_id(person_ground_truth_id: str) -> str:
    digest = hashlib.sha256(
        f"plan-0037-acoustic-subject:{person_ground_truth_id}".encode("utf-8")
    ).hexdigest()
    return f"subject-{digest[:24]}"


def _duration_band(duration_seconds: float) -> str:
    if duration_seconds < 30:
        return "under_30_seconds"
    if duration_seconds < 180:
        return "30_to_179_seconds"
    if duration_seconds < 900:
        return "3_to_14_minutes"
    return "15_minutes_or_more"


def _store_bounded_blob_path(path: Path, store_root: Path) -> Path:
    try:
        resolved = path.expanduser().resolve(strict=True)
        blob_root = (store_root.expanduser().resolve(strict=True) / "blobs")
        resolved.relative_to(blob_root)
    except (OSError, ValueError) as exc:
        raise CorpusError(
            f"Selected source blob escapes the transcript blob store: {path}"
        ) from exc
    return resolved


def _has_overlap(payload: dict[str, Any]) -> bool:
    spans: list[tuple[float, float, str]] = []
    for utterance in payload.get("utterances") or []:
        if not isinstance(utterance, dict):
            continue
        start = transcript_store.utterance_seconds(utterance.get("start"))
        end = transcript_store.utterance_seconds(utterance.get("end"))
        label = str(utterance.get("speaker") or "")
        if end > start:
            spans.append((start, end, label))
    spans.sort()
    for index, (start, _end, label) in enumerate(spans[1:], start=1):
        prior_start, prior_end, prior_label = spans[index - 1]
        if start < prior_end and label != prior_label and prior_start < prior_end:
            return True
    return False


def _speaker_truth(gold: dict[str, Any]) -> list[dict[str, str]]:
    outcomes = gold.get("speaker_outcomes")
    if not isinstance(outcomes, list) or not outcomes:
        raise CorpusError("Eligible gold record has no speaker outcomes.")
    result: list[dict[str, str]] = []
    for outcome in outcomes:
        if not isinstance(outcome, dict):
            raise CorpusError("Speaker outcome must be an object.")
        label = str(outcome.get("speaker_label") or "")
        disposition = str(outcome.get("outcome") or "")
        person_id = str(outcome.get("person_ground_truth_id") or "")
        if not label or not disposition:
            raise CorpusError("Speaker outcome identity is incomplete.")
        item = {"speaker_label": label, "outcome": disposition}
        if person_id:
            item["subject_id"] = _subject_id(person_id)
        result.append(item)
    return result


def collect_candidates(
    campaign_id: str,
    *,
    campaign_root: Optional[Path] = None,
    store_root: Optional[Path] = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Collect only latest eligible operator gold with accessible source blobs."""
    if not re.fullmatch(r"campaign-[a-f0-9]{20}", campaign_id):
        raise CorpusError("Campaign ID is invalid.")
    selected_campaign_root = (campaign_root or DEFAULT_CAMPAIGN_ROOT).expanduser()
    selected_store_root = transcript_store.store_dir(store_root)
    campaign_dir = selected_campaign_root / campaign_id
    campaign_manifest_path = campaign_dir / "manifest.json"
    gold_index_path = campaign_dir / "gold" / "index.json"
    campaign = _read_object(campaign_manifest_path)
    index = _read_object(gold_index_path)
    latest = _latest_gold_records(index)
    manifest_id = str(campaign.get("manifest_id") or "")
    if (
        campaign.get("campaign_id") != campaign_id
        or not manifest_id
        or not isinstance(campaign.get("items"), list)
    ):
        raise CorpusError("Campaign manifest identity is invalid.")
    campaign_items: dict[str, dict[str, Any]] = {}
    for item in campaign["items"]:
        if not isinstance(item, dict):
            raise CorpusError("Campaign manifest item must be an object.")
        document_id = str(item.get("document_id") or "")
        if not document_id or document_id in campaign_items:
            raise CorpusError("Campaign manifest document identity is invalid.")
        campaign_items[document_id] = item

    connection = transcript_store.connect(selected_store_root)
    candidates: list[dict[str, Any]] = []
    excluded = Counter()
    try:
        for record in latest:
            if record.get("disposition") != "eligible_known":
                excluded["not_eligible_known"] += 1
                continue
            raw_gold_path = Path(str(record.get("path") or "")).expanduser()
            try:
                gold_path = raw_gold_path.resolve(strict=True)
                gold_path.relative_to((campaign_dir / "gold").resolve(strict=True))
            except (OSError, ValueError) as exc:
                raise CorpusError(
                    f"Gold record escapes the campaign gold directory: {raw_gold_path}"
                ) from exc
            gold = _read_object(gold_path)
            review_method = str(gold.get("review_method") or "")
            reviewer = str(gold.get("reviewer") or "")
            operator_confirmed = review_method.startswith("operator_") or (
                review_method in LEGACY_OPERATOR_REVIEW_METHODS and bool(reviewer)
            )
            if (
                gold.get("prediction_visibility") != "excluded"
                or not operator_confirmed
            ):
                raise CorpusError(
                    "Corpus gold must be operator-confirmed and prediction-excluded."
                )
            document_id = str(record["document_id"])
            campaign_item = campaign_items.get(document_id)
            if campaign_item is None:
                raise CorpusError("Gold document is absent from the campaign manifest.")
            reviewed_artifact_sha256 = str(gold.get("artifact_sha256") or "")
            if (
                gold.get("schema_version")
                != "transcribe-audio.speaker-evaluation-gold.v1"
                or gold.get("gold_id") != record.get("gold_id")
                or gold.get("document_id") != document_id
                or gold.get("disposition") != record.get("disposition")
                or gold.get("campaign_id") != campaign_id
                or gold.get("manifest_id") != manifest_id
                or int(gold.get("chronological_rank") or 0)
                != int(record.get("chronological_rank") or 0)
                or int(campaign_item.get("chronological_rank") or 0)
                != int(record.get("chronological_rank") or 0)
                or reviewed_artifact_sha256
                != str(campaign_item.get("artifact_sha256") or "")
                or not re.fullmatch(r"[a-f0-9]{64}", reviewed_artifact_sha256)
            ):
                raise CorpusError("Gold, index, and campaign provenance do not match.")
            row = connection.execute(
                """
                SELECT artifact_sha256, stored_path, json_payload
                FROM documents
                WHERE id = ? AND kind = 'transcript'
                """,
                (document_id,),
            ).fetchone()
            if row is None:
                excluded["missing_transcript_row"] += 1
                continue
            transcript = json.loads(str(row["json_payload"]))
            if not isinstance(transcript, dict):
                excluded["invalid_transcript_payload"] += 1
                continue
            current_transcript_path = Path(str(row["stored_path"])).expanduser()
            if (
                not current_transcript_path.is_file()
                or _sha256_file(current_transcript_path)
                != str(row["artifact_sha256"])
            ):
                raise CorpusError(
                    "Current indexed transcript artifact is unavailable or changed."
                )
            blob = connection.execute(
                """
                SELECT b.id, b.stored_path, b.sha256, b.mime_type, b.bytes
                FROM document_blobs AS db
                JOIN blobs AS b ON b.id = db.blob_id
                WHERE db.document_id = ? AND db.role = 'source_recording'
                ORDER BY b.id
                LIMIT 1
                """,
                (document_id,),
            ).fetchone()
            if blob is None:
                excluded["missing_source_blob"] += 1
                continue
            raw_blob_path = Path(str(blob["stored_path"])).expanduser()
            if not raw_blob_path.is_file():
                excluded["source_blob_unavailable"] += 1
                continue
            blob_path = _store_bounded_blob_path(raw_blob_path, selected_store_root)
            actual_sha256 = _sha256_file(blob_path)
            if actual_sha256 != str(blob["sha256"]):
                raise CorpusError(f"Source blob hash mismatch: {blob_path}")
            conversation_id = str(transcript.get("conversation_id") or "")
            recording_id = str(transcript.get("recording_id") or "")
            if not conversation_id or not recording_id:
                excluded["missing_durable_identity"] += 1
                continue
            duration = float(transcript.get("duration_seconds") or 0.0)
            candidates.append(
                {
                    "document_id": document_id,
                    "chronological_rank": int(record.get("chronological_rank") or 0),
                    "conversation_id": conversation_id,
                    "recording_id": recording_id,
                    "split": _split_for_conversation(conversation_id),
                    "source_blob": {
                        "blob_id": str(blob["id"]),
                        "stored_path": str(blob_path.resolve()),
                        "sha256": actual_sha256,
                        "bytes": int(blob["bytes"]),
                        "mime_type": str(blob["mime_type"]),
                        "mode": stat.S_IMODE(blob_path.stat().st_mode),
                    },
                    "operator_gold": {
                        "gold_id": str(record.get("gold_id") or ""),
                        "reviewed_at": str(record.get("reviewed_at") or ""),
                        "review_method": review_method,
                        "prediction_visibility": "excluded",
                        "speaker_truth": _speaker_truth(gold),
                        "same_person_label_groups": gold.get(
                            "same_person_label_groups"
                        )
                        or [],
                    },
                    "transcript_lineage": {
                        "reviewed_artifact_sha256": reviewed_artifact_sha256,
                        "current_artifact_sha256": str(row["artifact_sha256"]),
                        "current_artifact_path": str(
                            current_transcript_path.resolve()
                        ),
                    },
                    "conditions": {
                        "channel": "unassessed_until_p1",
                        "device": "unassessed_until_p1",
                        "noise": "unassessed_until_p2",
                        "overlap": (
                            "observed_in_diarized_turns"
                            if _has_overlap(transcript)
                            else "not_observed_in_diarized_turns"
                        ),
                        "telephone_bandwidth": "unassessed_until_p1",
                        "usable_duration_band": "unassessed_until_p1",
                    },
                    "reported_recording_duration_seconds": duration,
                }
            )
    finally:
        connection.close()

    authority_hashes = {
        "campaign_manifest_sha256": _sha256_file(campaign_manifest_path),
        "gold_index_sha256": _sha256_file(gold_index_path),
    }
    metadata = {
        "campaign_id": campaign_id,
        "manifest_id": manifest_id,
        "authority_hashes": authority_hashes,
        "runtime_readback": {
            "transcript_store_db_sha256": _sha256_file(
                transcript_store.db_path(selected_store_root)
            )
        },
        "excluded_counts": dict(sorted(excluded.items())),
    }
    return candidates, metadata


def harden_candidate_sources(
    candidates: Iterable[dict[str, Any]],
    *,
    store_root: Optional[Path] = None,
    approval_token: str,
) -> dict[str, Any]:
    """Restrict only selected source blobs; never modify their content."""
    if approval_token != HARDEN_TOKEN:
        raise CorpusError(f"Source hardening requires approval token {HARDEN_TOKEN}.")
    selected_store_root = transcript_store.store_dir(store_root)
    paths: list[str] = []
    for candidate in candidates:
        source = candidate.get("source_blob") or {}
        path = _store_bounded_blob_path(
            Path(str(source.get("stored_path") or "")), selected_store_root
        )
        expected_hash = str(source.get("sha256") or "")
        if _sha256_file(path) != expected_hash:
            raise CorpusError(f"Selected source changed before hardening: {path}")
        os.chmod(path, 0o600)
        if stat.S_IMODE(path.stat().st_mode) != 0o600:
            raise CorpusError(f"Selected source mode is not 0600: {path}")
        paths.append(str(path))
    return {
        "schema_version": "transcribe-audio.acoustic-source-hardening-receipt.v1",
        "selected_source_count": len(paths),
        "paths": paths,
        "content_modified": False,
        "final_mode": "0600",
        "hardened_at": _utc_now(),
    }


def _validate_split_integrity(candidates: list[dict[str, Any]]) -> None:
    owners: dict[str, set[str]] = defaultdict(set)
    for candidate in candidates:
        owners[str(candidate["conversation_id"])].add(str(candidate["split"]))
    crossing = sorted(key for key, values in owners.items() if len(values) != 1)
    if crossing:
        raise CorpusError(
            "Conversations cross evaluation splits: " + ", ".join(crossing)
        )


def freeze_corpus(
    candidates: list[dict[str, Any]],
    metadata: dict[str, Any],
    *,
    runtime_root: Optional[Path] = None,
    approval_token: str,
) -> dict[str, Any]:
    """Write one immutable private manifest and return its receipt."""
    if approval_token != FREEZE_TOKEN:
        raise CorpusError(f"Corpus freeze requires approval token {FREEZE_TOKEN}.")
    if not candidates:
        raise CorpusError("Acoustic evaluation corpus has no eligible recordings.")
    _validate_split_integrity(candidates)
    for candidate in candidates:
        source = candidate["source_blob"]
        path = Path(str(source["stored_path"]))
        actual_mode = stat.S_IMODE(path.stat().st_mode)
        if actual_mode & 0o077:
            raise CorpusError(
                f"Selected source blob is not private (mode {actual_mode:04o}): {path}"
            )
        source["mode"] = actual_mode

    split_counts = Counter(str(item["split"]) for item in candidates)
    conversation_counts = Counter(
        str(item["conversation_id"]) for item in candidates
    )
    subject_sessions: dict[str, set[str]] = defaultdict(set)
    split_subject_sessions: dict[str, dict[str, set[str]]] = {
        split: defaultdict(set) for split in SPLITS
    }
    for item in candidates:
        conversation_id = str(item["conversation_id"])
        split = str(item["split"])
        for truth in item["operator_gold"]["speaker_truth"]:
            subject_id = str(truth.get("subject_id") or "")
            if not subject_id:
                continue
            subject_sessions[subject_id].add(conversation_id)
            split_subject_sessions[split][subject_id].add(conversation_id)
    same_person_pair_count = sum(
        len(sessions) * (len(sessions) - 1) // 2
        for sessions in subject_sessions.values()
    )
    subject_items = sorted(subject_sessions.items())
    different_person_pair_count = 0
    for index, (_left_subject, left_sessions) in enumerate(subject_items):
        for _right_subject, right_sessions in subject_items[index + 1 :]:
            different_person_pair_count += sum(
                1
                for left in left_sessions
                for right in right_sessions
                if left != right
            )
    selected_input_projection = [
        {
            "document_id": item["document_id"],
            "conversation_id": item["conversation_id"],
            "recording_id": item["recording_id"],
            "split": item["split"],
            "source_blob_id": item["source_blob"]["blob_id"],
            "source_sha256": item["source_blob"]["sha256"],
            "gold_id": item["operator_gold"]["gold_id"],
            "speaker_truth": item["operator_gold"]["speaker_truth"],
        }
        for item in candidates
    ]
    selected_input_sha256 = hashlib.sha256(
        _canonical_bytes(selected_input_projection)
    ).hexdigest()
    split_coverage = {
        split: {
            "recordings": split_counts.get(split, 0),
            "subjects": len(split_subject_sessions[split]),
            "subject_sessions": sum(
                len(sessions)
                for sessions in split_subject_sessions[split].values()
            ),
        }
        for split in SPLITS
    }
    readiness_blockers = [
        blocker
        for blocker, failed in (
            ("one_or_more_splits_empty", any(not split_counts.get(split) for split in SPLITS)),
            ("no_cross_session_same_person_pairs", same_person_pair_count < 1),
            ("no_different_person_pairs", different_person_pair_count < 1),
        )
        if failed
    ]
    core = {
        "schema_version": CORPUS_SCHEMA_VERSION,
        "source_campaign": {
            "campaign_id": metadata.get("campaign_id"),
            "manifest_id": metadata.get("manifest_id"),
            "authority_hashes": metadata.get("authority_hashes"),
            "excluded_counts": metadata.get("excluded_counts"),
        },
        "selection_policy": {
            "gold": "latest_eligible_known_operator_confirmed",
            "prediction_visibility": "excluded",
            "source_audio": "stored_blob_hash_matched",
            "conversation_split_policy": "conversation_id_disjoint",
            "split_algorithm": "sha256_mod100_60_20_20.v1",
        },
        "denominators": {
            "recordings": len(candidates),
            "conversations": len(conversation_counts),
            "subjects": len(subject_sessions),
            "speaker_labels": sum(
                len(item["operator_gold"]["speaker_truth"])
                for item in candidates
            ),
            "split_recordings": {
                split: split_counts.get(split, 0) for split in SPLITS
            },
            "sessions_per_subject": {
                subject_id: len(sessions)
                for subject_id, sessions in sorted(subject_sessions.items())
            },
            "feasible_same_person_pairs": same_person_pair_count,
            "feasible_different_person_pairs": different_person_pair_count,
            "split_coverage": split_coverage,
        },
        "selected_input_sha256": selected_input_sha256,
        "condition_contract": {
            "fields": list(CONDITION_FIELDS),
            "unassessed_values_are_explicit": True,
            "promotion_requires_measured_p1_p2_values": True,
        },
        "recordings": candidates,
        "metric_definitions": {
            "false_acceptance_rate": "false_accepts / different_person_trials",
            "false_rejection_rate": "false_rejects / same_person_trials",
            "abstention_rate": "abstained_trials / all_trials",
            "brier_score": "mean((calibrated_probability - binary_truth)^2)",
            "label_group_precision": "correct_proposed_merges / proposed_merges",
            "label_group_recall": "correct_proposed_merges / true_merges",
            "word_error_rate": "(substitutions + deletions + insertions) / reference_words",
            "diarization_error_rate": "(missed + false_alarm + confusion) / scored_speaker_time",
        },
        "prediction_visibility": "excluded",
        "gold_content_in_prompts": False,
        "will_execute_models": False,
        "will_perform_external_write": False,
        "promotion_eligible": False,
        "benchmark_readiness": {
            "status": "ready_for_p1_measurement" if not readiness_blockers else "insufficient",
            "blockers": readiness_blockers,
        },
        "promotion_blockers": [
            "p1_source_quality_and_channel_measurement_not_run",
            "p2_vad_enhancement_and_diarization_comparison_not_run",
            "p4_model_calibration_not_run",
        ],
    }
    digest = hashlib.sha256(_canonical_bytes(core)).hexdigest()
    corpus_id = f"acoustic-corpus-{digest[:24]}"
    manifest = {
        **core,
        "corpus_id": corpus_id,
        "content_sha256": digest,
        "runtime_readback_at_freeze": metadata.get("runtime_readback") or {},
        "frozen_at": _utc_now(),
    }
    selected_runtime_root = (runtime_root or DEFAULT_RUNTIME_ROOT).expanduser()
    manifest_path = selected_runtime_root / "corpora" / corpus_id / "manifest.json"
    _ensure_private_tree(selected_runtime_root, manifest_path.parent)
    if manifest_path.exists():
        existing = _read_object(manifest_path)
        comparable = dict(existing)
        comparable.pop("frozen_at", None)
        comparable.pop("runtime_readback_at_freeze", None)
        expected = dict(manifest)
        expected.pop("frozen_at", None)
        expected.pop("runtime_readback_at_freeze", None)
        if comparable != expected:
            raise CorpusError(f"Immutable corpus conflict: {manifest_path}")
        manifest = existing
    else:
        _write_private_json(manifest_path, manifest)

    if stat.S_IMODE(manifest_path.stat().st_mode) != 0o600:
        raise CorpusError("Frozen corpus manifest mode is not 0600.")
    receipt = {
        "schema_version": "transcribe-audio.acoustic-corpus-freeze-receipt.v1",
        "corpus_id": corpus_id,
        "manifest_path": str(manifest_path),
        "manifest_sha256": _sha256_file(manifest_path),
        "content_sha256": digest,
        "denominators": manifest["denominators"],
        "prediction_visibility": "excluded",
        "will_execute_models": False,
        "will_perform_external_write": False,
        "mode": "0600",
    }
    receipt_path = manifest_path.parent / "freeze-receipt.json"
    if receipt_path.exists():
        if _read_object(receipt_path) != receipt:
            raise CorpusError(f"Immutable corpus receipt conflict: {receipt_path}")
    else:
        _write_private_json(receipt_path, receipt)
    if stat.S_IMODE(receipt_path.stat().st_mode) != 0o600:
        raise CorpusError("Frozen corpus receipt mode is not 0600.")
    return {**receipt, "receipt_path": str(receipt_path)}


def main(argv: Optional[Iterable[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Freeze the private Plan 0037 acoustic evaluation corpus."
    )
    parser.add_argument("campaign_id")
    parser.add_argument("--campaign-root", type=Path)
    parser.add_argument("--store-root", type=Path)
    parser.add_argument("--runtime-root", type=Path)
    parser.add_argument("--harden", action="store_true")
    parser.add_argument("--harden-approval-token", default="")
    parser.add_argument("--freeze-approval-token", default="")
    args = parser.parse_args(list(argv) if argv is not None else None)

    candidates, metadata = collect_candidates(
        args.campaign_id,
        campaign_root=args.campaign_root,
        store_root=args.store_root,
    )
    output: dict[str, Any] = {
        "candidate_count": len(candidates),
        "source_campaign": metadata,
        "will_execute_models": False,
        "will_perform_external_write": False,
    }
    if args.harden:
        output["hardening"] = harden_candidate_sources(
            candidates,
            store_root=args.store_root,
            approval_token=args.harden_approval_token,
        )
        candidates, metadata = collect_candidates(
            args.campaign_id,
            campaign_root=args.campaign_root,
            store_root=args.store_root,
        )
    if args.freeze_approval_token:
        output["freeze"] = freeze_corpus(
            candidates,
            metadata,
            runtime_root=args.runtime_root,
            approval_token=args.freeze_approval_token,
        )
    print(json.dumps(output, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
