"""Metadata-only readiness evidence for a new Plan 0037 terminal cohort."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence

from acoustic_audio_derivatives import (
    canonical_artifact_hash,
    ensure_private_tree,
    require_private_file,
    sha256_file,
    utc_now,
    write_immutable_private_json,
)
from acoustic_evaluation_corpus import DEFAULT_CAMPAIGN_ROOT, DEFAULT_RUNTIME_ROOT, SPLITS
import transcript_store


READINESS_SCHEMA = "transcribe-audio.p4e2-successor-readiness.v1"
DEFAULT_OUTPUT_ROOT = DEFAULT_RUNTIME_ROOT / "verification-calibration"


class SuccessorReadinessError(ValueError):
    """Raised when successor readiness cannot be assessed safely."""


def _read_object(path: Path, *, label: str = "Readiness source") -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SuccessorReadinessError(
            f"{label} is invalid: {path}"
        ) from exc
    if not isinstance(value, dict):
        raise SuccessorReadinessError(f"{label} is invalid: {path}")
    return value


def _set_hash(values: Iterable[str]) -> str:
    return canonical_artifact_hash(sorted(set(values)))


def _prior_identity_sets(
    manifests: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, set[str]], dict[str, dict[str, str]]]:
    identities = {
        "document": set(),
        "recording": set(),
        "conversation": set(),
        "source": set(),
    }
    by_document: dict[str, dict[str, str]] = {}
    for manifest in manifests:
        recordings = manifest.get("recordings")
        if not isinstance(recordings, list):
            raise SuccessorReadinessError("Prior corpus recordings are invalid.")
        for item in recordings:
            if not isinstance(item, Mapping):
                raise SuccessorReadinessError("Prior corpus recording is invalid.")
            source = item.get("source_blob") or {}
            values = {
                "document": item.get("document_id"),
                "recording": item.get("recording_id"),
                "conversation": item.get("conversation_id"),
                "source": source.get("sha256") if isinstance(source, Mapping) else None,
            }
            if any(
                not isinstance(value, str) or not value
                for value in values.values()
            ):
                raise SuccessorReadinessError("Prior corpus identity is incomplete.")
            for field, value in values.items():
                identities[field].add(value)
            existing = by_document.get(values["document"])
            if existing is not None and existing != values:
                raise SuccessorReadinessError(
                    "Prior corpus document identity is inconsistent."
                )
            by_document[values["document"]] = values
    return identities, by_document


def _latest_eligible_index_records(index: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    records = index.get("records")
    if not isinstance(records, list):
        raise SuccessorReadinessError("Gold index records are invalid.")
    latest: dict[str, Mapping[str, Any]] = {}
    for record in records:
        if not isinstance(record, Mapping):
            raise SuccessorReadinessError("Gold index record is invalid.")
        document_id = str(record.get("document_id") or "")
        if not document_id:
            raise SuccessorReadinessError("Gold index document identity is missing.")
        latest[document_id] = record
    return sorted(
        (
            item
            for item in latest.values()
            if item.get("disposition") == "eligible_known"
        ),
        key=lambda item: (
            int(item.get("chronological_rank") or 0),
            str(item.get("document_id") or ""),
        ),
    )


def collect_metadata_candidates(
    campaign_id: str,
    prior_manifests: Sequence[Mapping[str, Any]],
    *,
    campaign_root: Optional[Path] = None,
    store_root: Optional[Path] = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Bind campaign and blob-index metadata without opening transcript or audio."""
    if not re.fullmatch(r"campaign-[a-f0-9]{20}", campaign_id):
        raise SuccessorReadinessError("Campaign ID is invalid.")
    selected_campaign_root = (campaign_root or DEFAULT_CAMPAIGN_ROOT).expanduser()
    campaign_dir = selected_campaign_root / campaign_id
    manifest_path = campaign_dir / "manifest.json"
    gold_index_path = campaign_dir / "gold" / "index.json"
    manifest = _read_object(manifest_path, label="Campaign manifest")
    gold_index = _read_object(gold_index_path, label="Gold index")
    items = manifest.get("items")
    if manifest.get("campaign_id") != campaign_id or not isinstance(items, list):
        raise SuccessorReadinessError("Campaign manifest identity is invalid.")
    campaign_items = {
        str(item.get("document_id") or ""): item
        for item in items
        if isinstance(item, Mapping) and item.get("document_id")
    }
    if len(campaign_items) != len(items):
        raise SuccessorReadinessError("Campaign item identity is invalid.")
    _prior_sets, prior_by_document = _prior_identity_sets(prior_manifests)
    connection = transcript_store.connect(transcript_store.store_dir(store_root))
    candidates: list[dict[str, Any]] = []
    store_metadata_projection: list[dict[str, str]] = []
    try:
        for record in _latest_eligible_index_records(gold_index):
            document_id = str(record["document_id"])
            campaign_item = campaign_items.get(document_id)
            if campaign_item is None or int(
                campaign_item.get("chronological_rank") or 0
            ) != int(record.get("chronological_rank") or 0):
                raise SuccessorReadinessError(
                    "Campaign and gold-index provenance do not match."
                )
            row = connection.execute(
                """
                SELECT d.artifact_sha256, b.sha256
                FROM documents AS d
                JOIN document_blobs AS db ON db.document_id = d.id
                JOIN blobs AS b ON b.id = db.blob_id
                WHERE d.id = ? AND d.kind = 'transcript'
                  AND db.role = 'source_recording'
                ORDER BY b.id
                LIMIT 1
                """,
                (document_id,),
            ).fetchone()
            reviewed_artifact_sha256 = str(
                campaign_item.get("artifact_sha256") or ""
            )
            current_artifact_sha256 = str(row["artifact_sha256"] or "") if row else ""
            if row is None or not re.fullmatch(
                r"[a-f0-9]{64}", reviewed_artifact_sha256
            ) or not re.fullmatch(r"[a-f0-9]{64}", current_artifact_sha256):
                raise SuccessorReadinessError(
                    "Campaign or transcript-store metadata is invalid."
                )
            prior = prior_by_document.get(document_id)
            source_sha256 = str(row["sha256"] or "")
            if not re.fullmatch(r"[a-f0-9]{64}", source_sha256):
                raise SuccessorReadinessError("Source metadata hash is invalid.")
            store_metadata_projection.append(
                {
                    "document_id": document_id,
                    "reviewed_artifact_sha256": reviewed_artifact_sha256,
                    "current_artifact_sha256": current_artifact_sha256,
                    "source_sha256": source_sha256,
                }
            )
            candidates.append(
                {
                    "document_id": document_id,
                    "recording_id": prior.get("recording") if prior else None,
                    "conversation_id": prior.get("conversation") if prior else None,
                    "split": None,
                    "source_blob": {"sha256": source_sha256},
                    "identity_source": (
                        "prior_frozen_corpus" if prior else "unresolved_metadata_only"
                    ),
                }
            )
    finally:
        connection.close()
    return candidates, {
        "authority_hashes": {
            "campaign_manifest_sha256": sha256_file(manifest_path),
            "gold_index_sha256": sha256_file(gold_index_path),
            "transcript_store_metadata_projection_sha256": canonical_artifact_hash(
                store_metadata_projection
            ),
        }
    }


def build_readiness_receipt(
    candidates: Sequence[Mapping[str, Any]],
    prior_manifests: Sequence[Mapping[str, Any]],
    *,
    campaign_id: str,
    campaign_authority_hashes: Mapping[str, Any],
    prior_manifest_hashes: Sequence[str],
) -> dict[str, Any]:
    """Return counts and opaque hashes without portable record or gold content."""
    prior, prior_by_document = _prior_identity_sets(prior_manifests)
    overlap_counts = Counter()
    disjoint: list[Mapping[str, Any]] = []
    unresolved_identity_count = 0
    for item in candidates:
        source = item.get("source_blob") or {}
        prior_item = prior_by_document.get(str(item.get("document_id") or ""))
        values = {
            "document": item.get("document_id"),
            "recording": item.get("recording_id") or (
                prior_item.get("recording") if prior_item else None
            ),
            "conversation": item.get("conversation_id") or (
                prior_item.get("conversation") if prior_item else None
            ),
            "source": source.get("sha256") if isinstance(source, Mapping) else None,
        }
        if not isinstance(values["document"], str) or not values["document"]:
            raise SuccessorReadinessError("Candidate identity is incomplete.")
        if not isinstance(values["source"], str) or not values["source"]:
            raise SuccessorReadinessError("Candidate source identity is incomplete.")
        overlapping = False
        for field, value in values.items():
            if isinstance(value, str) and value in prior[field]:
                overlap_counts[field] += 1
                overlapping = True
        identity_complete = all(
            isinstance(value, str) and bool(value) for value in values.values()
        )
        if not overlapping and not identity_complete:
            unresolved_identity_count += 1
        elif not overlapping:
            disjoint.append(item)

    split_counts = Counter(str(item.get("split") or "") for item in disjoint)
    subject_sessions: dict[str, set[str]] = defaultdict(set)
    speaker_labels = 0
    for item in disjoint:
        gold = item.get("operator_gold") or {}
        for truth in gold.get("speaker_truth") or []:
            if not isinstance(truth, Mapping):
                raise SuccessorReadinessError("Candidate speaker truth is invalid.")
            speaker_labels += 1
            subject_id = truth.get("subject_id")
            if isinstance(subject_id, str) and subject_id:
                subject_sessions[subject_id].add(str(item["conversation_id"]))
    same_person_pairs = sum(
        len(sessions) * (len(sessions) - 1) // 2
        for sessions in subject_sessions.values()
    )
    subject_items = list(subject_sessions.items())
    different_person_pairs = sum(
        1
        for index, (_left, left_sessions) in enumerate(subject_items)
        for _right, right_sessions in subject_items[index + 1 :]
        for left in left_sessions
        for right in right_sessions
        if left != right
    )
    blockers = []
    if not disjoint:
        blockers.append("no_fully_disjoint_operator_confirmed_candidates")
    if unresolved_identity_count:
        blockers.append("candidate_identity_requires_governed_materialization")
    if disjoint and any(split_counts.get(split, 0) == 0 for split in SPLITS):
        blockers.append("successor_split_coverage_not_yet_demonstrated")
    if same_person_pairs < 1:
        blockers.append("same_person_pair_feasibility_not_demonstrated")
    if different_person_pairs < 1:
        blockers.append("different_person_pair_feasibility_not_demonstrated")

    candidate_projection = sorted(
        [
            {
                "document_id": item["document_id"],
                "recording_id": item["recording_id"],
                "conversation_id": item["conversation_id"],
                "source_sha256": item["source_blob"]["sha256"],
            }
            for item in disjoint
        ],
        key=lambda item: (
            item["document_id"],
            item["recording_id"],
            item["conversation_id"],
            item["source_sha256"],
        ),
    )
    core = {
        "schema_version": READINESS_SCHEMA,
        "campaign_id": campaign_id,
        "campaign_authority_hashes": dict(campaign_authority_hashes),
        "prior_corpus_manifest_hashes": sorted(prior_manifest_hashes),
        "status": "blocked" if blockers else "ready_for_successor_cohort_design",
        "blockers": blockers,
        "counts": {
            "latest_eligible_candidates": len(candidates),
            "fully_disjoint_candidates": len(disjoint),
            "identity_unresolved_candidates": unresolved_identity_count,
            "prior_corpus_recordings": len(prior["recording"]),
            "overlap_by_identity": dict(sorted(overlap_counts.items())),
            "disjoint_split_recordings": {
                split: split_counts.get(split, 0) for split in SPLITS
            },
            "disjoint_conversations": len(
                {str(item["conversation_id"]) for item in disjoint}
            ),
            "disjoint_subjects": len(subject_sessions),
            "disjoint_speaker_labels": speaker_labels,
            "feasible_same_person_session_pairs": same_person_pairs,
            "feasible_different_person_session_pairs": different_person_pairs,
        },
        "opaque_set_hashes": {
            "prior_documents": _set_hash(prior["document"]),
            "prior_recordings": _set_hash(prior["recording"]),
            "prior_conversations": _set_hash(prior["conversation"]),
            "prior_sources": _set_hash(prior["source"]),
            "disjoint_candidate_projection": canonical_artifact_hash(
                candidate_projection
            ),
        },
        "prediction_visibility": "excluded",
        "collection_method": "campaign_and_transcript_store_metadata_only",
        "source_hash_evidence": "transcript_store_metadata_not_blob_rehashed",
        "contains_gold_body": False,
        "contains_source_paths": False,
        "will_read_audio": False,
        "will_run_models": False,
        "will_reveal_split": False,
        "will_perform_external_write": False,
    }
    return {
        **core,
        "readiness_id": "p4e2-readiness-" + canonical_artifact_hash(core)[:24],
        "assessed_at": utc_now(),
    }


def assess_successor_readiness(
    campaign_id: str,
    *,
    campaign_root: Optional[Path] = None,
    store_root: Optional[Path] = None,
    corpus_root: Optional[Path] = None,
    output_root: Optional[Path] = None,
) -> dict[str, Any]:
    selected_corpus_root = (
        corpus_root or DEFAULT_RUNTIME_ROOT / "corpora"
    ).expanduser().absolute()
    manifest_paths = sorted(
        selected_corpus_root.glob("acoustic-corpus-*/manifest.json")
    )
    if not manifest_paths:
        raise SuccessorReadinessError("No prior frozen acoustic corpus exists.")
    manifests = []
    manifest_hashes = []
    for path in manifest_paths:
        require_private_file(path, selected_corpus_root)
        manifests.append(_read_object(path, label="Prior corpus manifest"))
        manifest_hashes.append(sha256_file(path))
    candidates, metadata = collect_metadata_candidates(
        campaign_id,
        manifests,
        campaign_root=campaign_root or DEFAULT_CAMPAIGN_ROOT,
        store_root=store_root,
    )
    receipt = build_readiness_receipt(
        candidates,
        manifests,
        campaign_id=campaign_id,
        campaign_authority_hashes=metadata.get("authority_hashes") or {},
        prior_manifest_hashes=manifest_hashes,
    )
    selected_output_root = (output_root or DEFAULT_OUTPUT_ROOT).expanduser().absolute()
    receipt_dir = selected_output_root / "successor-readiness"
    ensure_private_tree(selected_output_root, receipt_dir)
    receipt_path = receipt_dir / f"{receipt['readiness_id']}.json"
    stored = write_immutable_private_json(
        receipt_path, receipt, volatile_fields=("assessed_at",)
    )
    return {
        **stored,
        "receipt_path": str(receipt_path),
        "receipt_sha256": sha256_file(receipt_path),
    }


def main(argv: Optional[Iterable[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Assess metadata-only readiness for a new terminal cohort."
    )
    parser.add_argument("campaign_id")
    parser.add_argument("--campaign-root", type=Path)
    parser.add_argument("--store-root", type=Path)
    parser.add_argument("--corpus-root", type=Path)
    parser.add_argument("--output-root", type=Path)
    args = parser.parse_args(list(argv) if argv is not None else None)
    result = assess_successor_readiness(
        args.campaign_id,
        campaign_root=args.campaign_root,
        store_root=args.store_root,
        corpus_root=args.corpus_root,
        output_root=args.output_root,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
