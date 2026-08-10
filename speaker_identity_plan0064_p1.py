"""Plan 0064 P1 dynamic acoustic evidence over the frozen P0 corpus.

The executor is deliberately shadow-only.  It consumes the exact P0 manifest,
the active governed profile matrix, and an already-frozen calibration file.  It
never changes references, profiles, thresholds, speaker assignments, or
knowledge state.
"""

from __future__ import annotations

import argparse
from array import array
from collections import Counter, defaultdict
from collections.abc import Callable, Mapping, Sequence
import hashlib
import json
import math
from pathlib import Path
import subprocess
from typing import Any

import acoustic_verification as verification
from acoustic_audio_derivatives import (
    canonical_artifact_hash,
    ensure_private_tree,
    read_private_object,
    require_private_file,
    sha256_file,
    write_immutable_private_json,
)
from speaker_identity_plan0064_p0 import DEFAULT_RUNTIME_ROOT, replay_p0


P1_SCHEMA = "transcribe-audio.plan0064-p1-acoustic-evidence.v1"
RECEIPT_SCHEMA = "transcribe-audio.plan0064-p1-acoustic-receipt.v1"
DEFAULT_THRESHOLD_APPLICATION = Path(
    "~/.local/state/transcribe-audio/plan-0037/generation-3/"
    "recalibration-executions/"
    "generation3-recalibration-execution-39298c74aab4a773945268cd/"
    "threshold-application.json"
)
DEFAULT_REFERENCE_ROOT = Path(
    "~/.local/state/transcribe-audio/plan-0037/biometric-references"
)
DEFAULT_PROFILE_ROOT = Path(
    "~/.local/state/transcribe-audio/plan-0037/verification-calibration"
)
METHOD_ID = "no_enhancement"
SAMPLE_RATE = 16_000
MIN_PROBE_SECONDS = 3.0
MAX_PROBE_SECONDS = 30.0
ACTION_COUNTS = {
    "speaker_assignments": 0,
    "new_enrollments": 0,
    "profile_mutations": 0,
    "threshold_mutations": 0,
    "knowledge_writes": 0,
    "provider_writes": 0,
    "external_writes": 0,
}


class Plan0064P1Error(ValueError):
    """Raised when frozen P1 authority or evidence drifts."""


class _CachingAdapter:
    """Reuse one governed probe embedding across the seven profile scores."""

    def __init__(self, wrapped: Any) -> None:
        self._wrapped = wrapped
        self.candidate_id = wrapped.candidate_id
        self.revision_sha = wrapped.revision_sha
        self.embedding_dimension = wrapped.embedding_dimension
        self._source: Sequence[float] | None = None
        self._embedding: tuple[float, ...] | None = None

    @property
    def model_loaded(self) -> bool:
        return bool(self._wrapped.model_loaded)

    def embed(
        self, samples: Sequence[float], *, sample_rate: int
    ) -> tuple[float, ...]:
        if samples is not self._source:
            self._embedding = tuple(
                self._wrapped.embed(samples, sample_rate=sample_rate)
            )
            self._source = samples
        if self._embedding is None:
            raise Plan0064P1Error("The cached adapter did not produce an embedding.")
        return self._embedding


def _hash(value: Any) -> str:
    return canonical_artifact_hash(value)


def _content_addressed(value: Mapping[str, Any]) -> dict[str, Any]:
    body = dict(value)
    body.pop("content_sha256", None)
    return {**body, "content_sha256": _hash(body)}


def _read_object(path: Path) -> dict[str, Any]:
    value = read_private_object(path)
    if not isinstance(value, dict):
        raise Plan0064P1Error("A governed JSON artifact is not an object.")
    return value


def _thresholds(path: Path, candidate_ids: Sequence[str]) -> dict[str, Any]:
    selected = path.expanduser().absolute()
    require_private_file(selected, selected.parent.parent)
    raw = _read_object(selected)
    if (
        raw.get("status") != "success"
        or raw.get("contains_frozen_thresholds") is not True
        or raw.get("did_enable_default_integration") is not False
        or raw.get("did_mutate_profiles_or_references") is not False
    ):
        raise Plan0064P1Error("The calibration authority is not frozen shadow evidence.")
    units = {
        str(item.get("candidate_id")): {
            "candidate_id": str(item.get("candidate_id")),
            "method_id": str(item.get("method_id")),
            "threshold": float(item.get("threshold")),
            "temperature": float(item.get("temperature")),
        }
        for item in raw.get("thresholds") or []
        if isinstance(item, Mapping) and item.get("method_id") == METHOD_ID
    }
    if set(units) != set(candidate_ids) or any(
        not math.isfinite(item["threshold"])
        or not math.isfinite(item["temperature"])
        or item["temperature"] <= 0
        for item in units.values()
    ):
        raise Plan0064P1Error("The no-enhancement threshold matrix is incomplete.")
    return {
        "path": str(selected),
        "sha256": sha256_file(selected),
        "execution_authority_sha256": str(raw["execution_authority_sha256"]),
        "score_matrix_sha256": str(raw["score_matrix_sha256"]),
        "method_id": METHOD_ID,
        "units": [units[key] for key in sorted(units)],
        "threshold_set_sha256": _hash([units[key] for key in sorted(units)]),
    }


def build_p1_preview(
    p0_content_sha256: str,
    *,
    runtime_root: Path = DEFAULT_RUNTIME_ROOT,
    threshold_application: Path = DEFAULT_THRESHOLD_APPLICATION,
) -> dict[str, Any]:
    """Replay P0 and bind the exact active matrix and calibration authority."""
    p0 = replay_p0(content_sha256=p0_content_sha256, runtime_root=runtime_root)
    manifest = _read_object(Path(p0["private_manifest_path"]))
    profiles = manifest["profile_inventory"]["active_profiles"]
    selected = [
        item
        for item in manifest["evaluation_cohort"]["considered"]
        if item["disposition"] == "selected_evaluation_candidate"
    ]
    candidate_ids = manifest["profile_inventory"]["candidate_ids"]
    if len(selected) != 12 or len(profiles) != 21 or len(candidate_ids) != 3:
        raise Plan0064P1Error("The frozen P0 denominator is incomplete.")
    thresholds = _thresholds(threshold_application, candidate_ids)
    return _content_addressed(
        {
            "schema_version": P1_SCHEMA,
            "status": "ready_for_private_shadow_scoring",
            "p0_content_sha256": p0_content_sha256,
            "p0_receipt_content_sha256": p0["receipt_content_sha256"],
            "active_profile_set_sha256": manifest["profile_inventory"][
                "active_profile_set_sha256"
            ],
            "binding_set_sha256": manifest["canonical_bindings"][
                "binding_set_sha256"
            ],
            "cohort_sha256": manifest["evaluation_cohort"]["cohort_sha256"],
            "recording_count": len(selected),
            "speaker_slot_count": sum(len(item["speaker_labels"]) for item in selected),
            "active_profile_count": len(profiles),
            "identity_ready_profile_count": sum(
                bool(item["identity_candidate_eligible"]) for item in profiles
            ),
            "candidate_ids": list(candidate_ids),
            "threshold_authority": thresholds,
            "action_counts": dict(ACTION_COUNTS),
            "will_read_gold": False,
            "will_change_thresholds": False,
            "will_mutate_runtime_state": False,
            "will_perform_external_write": False,
        }
    )


def _decode(path: Path) -> array:
    result = subprocess.run(
        [
            "ffmpeg", "-v", "error", "-nostdin", "-i", str(path),
            "-ac", "1", "-ar", str(SAMPLE_RATE), "-f", "s16le", "pipe:1",
        ],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if result.returncode:
        raise Plan0064P1Error("Source media decode failed.")
    pcm = array("h")
    pcm.frombytes(result.stdout)
    if pcm.itemsize != 2:
        raise Plan0064P1Error("Unexpected signed PCM sample width.")
    return array("f", (sample / 32768.0 for sample in pcm))


def _slot_probe(transcript: Mapping[str, Any], speaker: str, samples: array) -> array:
    limit = int(MAX_PROBE_SECONDS * SAMPLE_RATE)
    probe = array("f")
    for utterance in transcript.get("utterances") or []:
        if not isinstance(utterance, Mapping) or str(utterance.get("speaker")) != speaker:
            continue
        start = max(0, int(float(utterance.get("start") or 0) * SAMPLE_RATE / 1000))
        end = min(len(samples), int(float(utterance.get("end") or 0) * SAMPLE_RATE / 1000))
        if end <= start:
            continue
        remaining = limit - len(probe)
        if remaining <= 0:
            break
        probe.extend(samples[start : min(end, start + remaining)])
    return probe


def _score_slot(
    *,
    document_id: str,
    speaker: str,
    probe: Sequence[float],
    profiles: Sequence[Mapping[str, Any]],
    thresholds: Mapping[str, float],
    adapters: Mapping[str, Any],
    score_fn: Callable[..., Mapping[str, Any]],
    profile_root: Path,
    reference_root: Path,
) -> dict[str, Any]:
    speaker_ref = f"{document_id}::{speaker}"
    probe_sha256 = hashlib.sha256(array("f", probe).tobytes()).hexdigest()
    duration = len(probe) / SAMPLE_RATE
    if duration < MIN_PROBE_SECONDS:
        return {
            "speaker_ref": speaker_ref,
            "speaker_label": speaker,
            "status": "unavailable",
            "reason_code": "insufficient_speaker_audio",
            "probe_sha256": probe_sha256,
            "probe_duration_seconds": duration,
            "model_rows": [],
            "candidate_person_id": None,
            "candidate_acoustic_subject_id": None,
            "confidence_band": "none",
        }
    rows: list[dict[str, Any]] = []
    for candidate_id in sorted(adapters):
        model_scores = []
        for profile in profiles:
            if profile["candidate_id"] != candidate_id:
                continue
            scored = score_fn(
                str(profile["profile_id"]),
                adapter=adapters[candidate_id],
                probe_samples=probe,
                sample_rate=SAMPLE_RATE,
                runtime_root=profile_root,
                p3_runtime_root=reference_root,
            )
            value = float(scored["score"])
            if not math.isfinite(value):
                raise Plan0064P1Error("An acoustic score is nonfinite.")
            model_scores.append(
                {
                    "profile_id": str(profile["profile_id"]),
                    "acoustic_subject_id": str(profile["person_ref_id"]),
                    "canonical_person_id": profile.get("canonical_person_id"),
                    "identity_candidate_eligible": bool(
                        profile["identity_candidate_eligible"]
                    ),
                    "score": value,
                    "trial_id": str(scored["trial_id"]),
                    "profile_artifact_sha256": profile["artifact"]["sha256"],
                    "model_revision": str(profile["model_revision"]),
                }
            )
        if len(model_scores) != 7:
            raise Plan0064P1Error("A model did not score the complete active subject set.")
        ranked = sorted(
            model_scores,
            key=lambda item: (item["score"], item["acoustic_subject_id"]),
            reverse=True,
        )
        top = ranked[0]
        threshold = thresholds[candidate_id]
        rows.append(
            {
                "candidate_id": candidate_id,
                "threshold": threshold,
                "scores": sorted(
                    model_scores, key=lambda item: item["acoustic_subject_id"]
                ),
                "top_acoustic_subject_id": top["acoustic_subject_id"],
                "top_canonical_person_id": (
                    top["canonical_person_id"]
                    if top["identity_candidate_eligible"]
                    else None
                ),
                "top_score": top["score"],
                "runner_up_score": ranked[1]["score"],
                "score_margin": top["score"] - ranked[1]["score"],
                "threshold_pass": top["score"] >= threshold,
                "binding_eligible": top["identity_candidate_eligible"],
            }
        )
    passing = [
        row for row in rows
        if row["threshold_pass"] and row["binding_eligible"]
        and row["top_canonical_person_id"]
    ]
    votes = Counter(row["top_canonical_person_id"] for row in passing)
    if votes:
        person_id, support = votes.most_common(1)[0]
        tied = sum(count == support for count in votes.values()) > 1
    else:
        person_id, support, tied = None, 0, False
    if person_id and support >= 2 and not tied:
        status, reason, band = "candidate", "multi_model_acoustic_support", "high"
    elif person_id and support == 1 and not tied:
        status, reason, band = "review", "single_model_acoustic_support", "medium"
    elif passing:
        status, reason, band = "review", "conflicting_acoustic_support", "low"
        person_id = None
    else:
        status, reason, band = "abstain", "no_bound_profile_threshold_pass", "none"
    subject = None
    if person_id:
        subjects = {
            row["top_acoustic_subject_id"]
            for row in passing
            if row["top_canonical_person_id"] == person_id
        }
        subject = next(iter(subjects)) if len(subjects) == 1 else None
    return {
        "speaker_ref": speaker_ref,
        "speaker_label": speaker,
        "status": status,
        "reason_code": reason,
        "probe_sha256": probe_sha256,
        "probe_duration_seconds": duration,
        "model_rows": rows,
        "candidate_person_id": person_id,
        "candidate_acoustic_subject_id": subject,
        "supporting_model_count": support,
        "confidence_band": band,
    }


def execute_p1(
    p0_content_sha256: str,
    *,
    runtime_root: Path = DEFAULT_RUNTIME_ROOT,
    threshold_application: Path = DEFAULT_THRESHOLD_APPLICATION,
    profile_root: Path = DEFAULT_PROFILE_ROOT,
    reference_root: Path = DEFAULT_REFERENCE_ROOT,
    score_fn: Callable[..., Mapping[str, Any]] = verification.score_profile,
    adapter_factory: Callable[[], Mapping[str, Any]] = verification.adapter_registry,
    decode_fn: Callable[[Path], array] = _decode,
) -> dict[str, Any]:
    """Execute complete private shadow scoring, or replay an existing result."""
    preview = build_p1_preview(
        p0_content_sha256,
        runtime_root=runtime_root,
        threshold_application=threshold_application,
    )
    p0 = replay_p0(content_sha256=p0_content_sha256, runtime_root=runtime_root)
    p0_manifest = _read_object(Path(p0["private_manifest_path"]))
    root = runtime_root.expanduser().absolute() / f"p1-{preview['content_sha256'][:24]}"
    result_path = root / "private-acoustic-evidence.json"
    receipt_path = root / "receipt.json"
    if result_path.exists() or receipt_path.exists():
        return replay_p1(p0_content_sha256, runtime_root=runtime_root,
                         threshold_application=threshold_application)
    profiles = p0_manifest["profile_inventory"]["active_profiles"]
    selected = [item for item in p0_manifest["evaluation_cohort"]["considered"]
                if item["disposition"] == "selected_evaluation_candidate"]
    adapters = {
        key: _CachingAdapter(value) for key, value in dict(adapter_factory()).items()
    }
    if set(adapters) != set(preview["candidate_ids"]):
        raise Plan0064P1Error("The adapter registry does not match P0.")
    thresholds = {
        item["candidate_id"]: float(item["threshold"])
        for item in preview["threshold_authority"]["units"]
    }
    recordings = []
    for index, recording in enumerate(selected, start=1):
        print(f"Scoring Plan 0064 P1 recording {index}/{len(selected)}...", flush=True)
        transcript_path = Path(recording["transcript_artifact"]["path"])
        media_path = Path(recording["source_media_artifact"]["path"])
        if sha256_file(transcript_path) != recording["artifact_sha256"]:
            raise Plan0064P1Error("A frozen transcript changed before scoring.")
        if sha256_file(media_path) != recording["source_media_sha256"]:
            raise Plan0064P1Error("Frozen source media changed before scoring.")
        transcript = json.loads(transcript_path.read_text(encoding="utf-8"))
        decoded = decode_fn(media_path)
        slots = [
            _score_slot(
                document_id=recording["document_id"], speaker=speaker,
                probe=_slot_probe(transcript, speaker, decoded), profiles=profiles,
                thresholds=thresholds, adapters=adapters, score_fn=score_fn,
                profile_root=profile_root.expanduser().absolute(),
                reference_root=reference_root.expanduser().absolute(),
            )
            for speaker in recording["speaker_labels"]
        ]
        recordings.append({
            "document_id": recording["document_id"],
            "recording_time": recording["recording_time"],
            "transcript_sha256": recording["artifact_sha256"],
            "source_media_sha256": recording["source_media_sha256"],
            "speaker_slots": slots,
        })
    slot_rows = [slot for item in recordings for slot in item["speaker_slots"]]
    evidence = _content_addressed({
        "schema_version": P1_SCHEMA,
        "status": "complete_private_shadow_acoustic_evidence",
        "preview_content_sha256": preview["content_sha256"],
        "p0_content_sha256": p0_content_sha256,
        "threshold_authority": preview["threshold_authority"],
        "recordings": recordings,
        "summary": {
            "recording_count": len(recordings),
            "speaker_slot_count": len(slot_rows),
            "trial_count": sum(len(row["scores"]) for slot in slot_rows for row in slot["model_rows"]),
            "status_counts": dict(sorted(Counter(slot["status"] for slot in slot_rows).items())),
            "reason_code_counts": dict(sorted(Counter(slot["reason_code"] for slot in slot_rows).items())),
        },
        "contains_biometric_scores": True,
        "contains_embeddings_or_vectors": False,
        "contains_raw_audio": False,
        "contains_gold": False,
        "did_change_thresholds": False,
        "action_counts": dict(ACTION_COUNTS),
        "will_perform_external_write": False,
    })
    ensure_private_tree(root, result_path, receipt_path)
    write_immutable_private_json(result_path, evidence)
    receipt = _content_addressed({
        "schema_version": RECEIPT_SCHEMA,
        "status": "p1_complete_zero_effect",
        "p0_content_sha256": p0_content_sha256,
        "preview_content_sha256": preview["content_sha256"],
        "evidence_content_sha256": evidence["content_sha256"],
        "evidence_file_sha256": sha256_file(result_path),
        "summary": evidence["summary"],
        "action_counts": dict(ACTION_COUNTS),
    })
    write_immutable_private_json(receipt_path, receipt)
    return {**receipt, "private_evidence_path": str(result_path),
            "private_receipt_path": str(receipt_path), "idempotent_replay": False}


def replay_p1(
    p0_content_sha256: str,
    *,
    runtime_root: Path = DEFAULT_RUNTIME_ROOT,
    threshold_application: Path = DEFAULT_THRESHOLD_APPLICATION,
) -> dict[str, Any]:
    preview = build_p1_preview(p0_content_sha256, runtime_root=runtime_root,
                               threshold_application=threshold_application)
    root = runtime_root.expanduser().absolute() / f"p1-{preview['content_sha256'][:24]}"
    result_path, receipt_path = root / "private-acoustic-evidence.json", root / "receipt.json"
    require_private_file(result_path, root)
    require_private_file(receipt_path, root)
    evidence, receipt = _read_object(result_path), _read_object(receipt_path)
    if (
        evidence.get("preview_content_sha256") != preview["content_sha256"]
        or evidence.get("content_sha256") != _hash({k: v for k, v in evidence.items() if k != "content_sha256"})
        or receipt.get("evidence_content_sha256") != evidence["content_sha256"]
        or receipt.get("evidence_file_sha256") != sha256_file(result_path)
        or receipt.get("content_sha256") != _hash({k: v for k, v in receipt.items() if k != "content_sha256"})
        or receipt.get("action_counts") != ACTION_COUNTS
    ):
        raise Plan0064P1Error("The frozen P1 evidence or receipt drifted.")
    return {**receipt, "private_evidence_path": str(result_path),
            "private_receipt_path": str(receipt_path), "idempotent_replay": True}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("preview", "execute", "replay"))
    parser.add_argument("--p0-content-sha256", required=True)
    parser.add_argument("--runtime-root", type=Path, default=DEFAULT_RUNTIME_ROOT)
    parser.add_argument("--threshold-application", type=Path, default=DEFAULT_THRESHOLD_APPLICATION)
    args = parser.parse_args(argv)
    kwargs = {"runtime_root": args.runtime_root,
              "threshold_application": args.threshold_application}
    if args.action == "preview":
        result = build_p1_preview(args.p0_content_sha256, **kwargs)
    elif args.action == "execute":
        result = execute_p1(args.p0_content_sha256, **kwargs)
    else:
        result = replay_p1(args.p0_content_sha256, **kwargs)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
