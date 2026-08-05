"""Plan 0055 E3 single gold reveal and deterministic paired scoring."""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
import unicodedata
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import acoustic_generation5_e2 as e2
import acoustic_generation5_source_gold as source_gold
import acoustic_generation5_source_j1_acceptance as j1
import acoustic_generation5_source_review as s1
from acoustic_audio_derivatives import (
    ensure_private_tree,
    require_private_file,
    sha256_file,
    write_immutable_private_json,
)


REVEAL_SCHEMA = "transcribe-audio.generation5-e3-reveal.v1"
SCORE_SCHEMA = "transcribe-audio.generation5-e3-score.v1"
RECEIPT_SCHEMA = "transcribe-audio.generation5-e3-receipt.v1"
REPLAY_SCHEMA = "transcribe-audio.generation5-e3-replay.v1"
E2_AUTHORITY_SHA256 = "9d5762fab9aea852835f4dbfd0575f33aeb36df90e66a46dcd4b69b3b140fef6"
E2_EXECUTION_SHA256 = "3b00b9462c0aae1d8016e9e6f7e4c9b0e35d75ad838ce19ac6386c2d609e0d82"
J1_PREVIEW_SHA256 = e2.J1_PREVIEW_SHA256
J1_MANIFEST_SHA256 = e2.J1_MANIFEST_SHA256
DEFAULT_RUNTIME_ROOT = Path("~/.local/state/transcribe-audio/plan-0055/e3")
MODULE_NAME = Path(__file__).name


class Generation5E3Error(ValueError):
    """Raised when the one-reveal paired score cannot remain exact."""


def _canonical_hash(value: Any) -> str:
    body = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise Generation5E3Error("Private E3 evidence is unreadable.") from exc
    if not isinstance(value, dict):
        raise Generation5E3Error("Private E3 evidence must be an object.")
    return value


def _git(arguments: Sequence[str], *, binary: bool = False) -> str | bytes:
    result = subprocess.run(["git", *arguments], cwd=Path(__file__).resolve().parent,
                            capture_output=True, text=not binary, check=False)
    if result.returncode:
        raise Generation5E3Error("Repository authority is unavailable.")
    return result.stdout if binary else result.stdout.strip()


def _repository_authority() -> dict[str, Any]:
    if _git(["status", "--porcelain=v1", "--untracked-files=normal"]):
        raise Generation5E3Error("Repository must be clean.")
    if str(_git(["rev-list", "--left-right", "--count", "@{upstream}...HEAD"])).split() != ["0", "0"]:
        raise Generation5E3Error("Repository must be upstream-even.")
    commit = str(_git(["rev-parse", "HEAD"]))
    body = _git(["show", f"{commit}:{MODULE_NAME}"], binary=True)
    if not isinstance(body, bytes) or hashlib.sha256(body).hexdigest() != sha256_file(Path(__file__).resolve()):
        raise Generation5E3Error("Committed E3 module drifted.")
    return {"commit": commit, "module_sha256": hashlib.sha256(body).hexdigest(),
            "clean": True, "upstream_ahead": 0, "upstream_behind": 0}


def _normalized_identity(value: Any) -> str:
    display = " ".join(unicodedata.normalize("NFKC", str(value or "")).split())
    return source_gold._normalized_identity(source_gold._canonical_identity(display))


def score_prediction_pair(
    *, expected_refs: Sequence[str], gold_by_ref: Mapping[str, Mapping[str, Any]],
    context_predictions: Mapping[str, Any], augmented_predictions: Mapping[str, Any],
) -> dict[str, Any]:
    """Compute complete paired metrics without returning private identities."""
    context = {str(item["speaker_ref"]): dict(item) for item in context_predictions.get("predictions") or []}
    augmented = {str(item["speaker_ref"]): dict(item) for item in augmented_predictions.get("predictions") or []}
    if (len(expected_refs) != e2.EXPECTED_SPEAKER_COUNT or set(context) != set(expected_refs)
            or set(augmented) != set(expected_refs) or set(gold_by_ref) != set(expected_refs)):
        raise Generation5E3Error("Paired prediction or gold denominator is incomplete.")

    rows = []
    for reference in expected_refs:
        gold = gold_by_ref[reference]
        truth = _normalized_identity(gold.get("private_identity_display"))
        if not truth:
            raise Generation5E3Error("A gold identity is empty.")
        lanes = {}
        for lane, prediction in (("context_only", context[reference]), ("voice_augmented", augmented[reference])):
            match = _normalized_identity(prediction.get("identity_or_alias")) == truth
            assigned = prediction.get("disposition") == "assign"
            lanes[lane] = {
                "identity_match": match,
                "correct_assignment": match and assigned,
                "wrong_assignment": assigned and not match,
                "candidate_recalled": match,
                "high_confidence_wrong": assigned and not match and prediction.get("confidence_band") == "high",
                "review": prediction.get("disposition") == "review",
                "abstention": prediction.get("disposition") == "abstain",
            }
        rows.append({
            "speaker_ref_sha256": hashlib.sha256(reference.encode()).hexdigest(),
            "is_enrolled": bool(gold.get("enrolled_subject_id")),
            "context_only": lanes["context_only"],
            "voice_augmented": lanes["voice_augmented"],
            "prediction_changed": _normalized_identity(context[reference].get("identity_or_alias"))
                                  != _normalized_identity(augmented[reference].get("identity_or_alias")),
            "corrected_baseline_error": (not lanes["context_only"]["correct_assignment"]
                                         and lanes["voice_augmented"]["correct_assignment"]),
            "introduced_error": (lanes["context_only"]["correct_assignment"]
                                 and not lanes["voice_augmented"]["correct_assignment"]),
            "safe_review_resolution": ((lanes["context_only"]["review"] or lanes["context_only"]["abstention"])
                                       and lanes["voice_augmented"]["correct_assignment"]),
        })

    def metrics(lane: str, subset: list[dict[str, Any]]) -> dict[str, Any]:
        denominator = len(subset)
        counts = {key: sum(bool(row[lane][key]) for row in subset) for key in (
            "correct_assignment", "wrong_assignment", "candidate_recalled",
            "high_confidence_wrong", "review", "abstention",
        )}
        return {
            "speaker_count": denominator, **counts,
            "assignment_correctness": counts["correct_assignment"] / denominator if denominator else 0.0,
            "candidate_recall": counts["candidate_recalled"] / denominator if denominator else 0.0,
        }

    enrolled_rows = [row for row in rows if row["is_enrolled"]]
    context_metrics = metrics("context_only", rows)
    augmented_metrics = metrics("voice_augmented", rows)
    paired = {
        "prediction_changed_count": sum(row["prediction_changed"] for row in rows),
        "corrected_baseline_error_count": sum(row["corrected_baseline_error"] for row in rows),
        "introduced_error_count": sum(row["introduced_error"] for row in rows),
        "safe_review_resolution_count": sum(row["safe_review_resolution"] for row in rows),
        "assignment_correctness_delta": augmented_metrics["assignment_correctness"] - context_metrics["assignment_correctness"],
        "candidate_recall_delta": augmented_metrics["candidate_recall"] - context_metrics["candidate_recall"],
    }
    core = {
        "speaker_count": len(rows), "enrolled_speaker_count": len(enrolled_rows),
        "context_only": context_metrics, "voice_augmented": augmented_metrics,
        "enrolled_context_only": metrics("context_only", enrolled_rows),
        "enrolled_voice_augmented": metrics("voice_augmented", enrolled_rows),
        "paired": paired, "private_rows": rows,
    }
    return {**core, "content_sha256": _canonical_hash(core)}


def _paths(runtime_root: Path) -> dict[str, Path]:
    root = runtime_root.expanduser().absolute()
    run = root / f"generation5-e3-{E2_EXECUTION_SHA256[:24]}"
    return {"root": root, "run": run, "reveal": run / "single-reveal.json",
            "score": run / "private-score.json", "receipt": run / "receipt.json"}


def _e2_execution() -> dict[str, Any]:
    replay = e2.replay_generation5_e2_execution(E2_AUTHORITY_SHA256)
    if replay.get("idempotent_replay") is not True or replay.get("content_sha256") != E2_EXECUTION_SHA256:
        raise Generation5E3Error("Frozen E2 execution drifted.")
    path = e2._paths(e2.DEFAULT_RUNTIME_ROOT, E2_AUTHORITY_SHA256)["execution"]
    execution = _read_json(path)
    if execution.get("contains_gold") is not False or execution.get("did_reveal_gold") is not False:
        raise Generation5E3Error("E2 blindness evidence failed.")
    return execution


def _gold_by_ref() -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    replay = j1.replay_generation5_source_j1_acceptance(J1_PREVIEW_SHA256)
    paths = j1._paths(j1.DEFAULT_RUNTIME_ROOT, J1_PREVIEW_SHA256)
    if replay.get("idempotent_replay") is not True or sha256_file(paths["manifest"]) != J1_MANIFEST_SHA256:
        raise Generation5E3Error("Frozen J1 gold drifted.")
    preview = _read_json(paths["manifest"])["preview"]
    selected = preview["private_gold"]["selected_cases"]
    by_key = {(int(case["enumerated_ordinal"]), str(item["speaker_label"])): dict(item)
              for case in selected for item in case["speaker_gold"]}
    s1_paths = s1._paths(s1.DEFAULT_RUNTIME_ROOT, e2.S1_PREVIEW_SHA256)
    cards = _read_json(s1_paths["manifest"])["preview"]["private_evidence"]["cards"]
    result = {str(card["speaker_ref"]): by_key[(int(card["ordinal"]), str(card["speaker_label"]))]
              for card in cards if int(card["ordinal"]) in e2.SELECTED_ORDINALS}
    if len(result) != e2.EXPECTED_SPEAKER_COUNT:
        raise Generation5E3Error("Private gold join is incomplete.")
    return result, preview


def execute_generation5_e3(*, runtime_root: Path = DEFAULT_RUNTIME_ROOT) -> dict[str, Any]:
    """Reveal the frozen J1 gold once and freeze complete paired metrics."""
    paths = _paths(runtime_root)
    if paths["receipt"].exists():
        return replay_generation5_e3(runtime_root=runtime_root)
    execution = _e2_execution()
    gold_by_ref, j1_preview = _gold_by_ref()
    ensure_private_tree(paths["root"], paths["run"])
    reveal_core = {
        "schema_version": REVEAL_SCHEMA, "status": "gold_revealed_to_scoring_custodian",
        "reveal_count": 1, "e2_execution_sha256": E2_EXECUTION_SHA256,
        "j1_preview_sha256": J1_PREVIEW_SHA256, "j1_manifest_sha256": J1_MANIFEST_SHA256,
        "gold_commitment_sha256": _canonical_hash(gold_by_ref),
        "speaker_count": len(gold_by_ref), "revealed_to_prediction_workers": False,
        "revealed_to_scoring_custodian": True,
    }
    reveal = {**reveal_core, "content_sha256": _canonical_hash(reveal_core)}
    write_immutable_private_json(paths["reveal"], reveal)

    context = execution["private_evidence"]["context_predictions"]
    augmented = execution["private_evidence"]["augmented_predictions"]
    refs = [str(item["speaker_ref"]) for item in execution["private_evidence"]["augmented_worker_packet"]["speakers"]]
    metrics = score_prediction_pair(
        expected_refs=refs, gold_by_ref=gold_by_ref,
        context_predictions=context, augmented_predictions=augmented,
    )
    score_core = {
        "schema_version": SCORE_SCHEMA, "status": "paired_scoring_complete",
        "repository_authority": _repository_authority(),
        "e2_authority_sha256": E2_AUTHORITY_SHA256,
        "e2_execution_sha256": E2_EXECUTION_SHA256,
        "reveal_content_sha256": reveal["content_sha256"], "reveal_count": 1,
        "selected_case_ids_sha256": j1_preview["selected_case_ids_sha256"],
        "matrix_set_sha256": execution["matrix_set_sha256"],
        "context_predictions_sha256": execution["context_predictions_sha256"],
        "augmented_predictions_sha256": execution["augmented_predictions_sha256"],
        "metrics": metrics,
        "did_reveal_gold_once": True, "did_reveal_gold_to_workers": False,
        "did_regenerate_predictions": False, "did_change_thresholds": False,
        "did_mutate_profiles_or_references": False,
        "did_enable_default_integration": False,
        "did_run_historical_reprocessing": False,
    }
    score = {**score_core, "content_sha256": _canonical_hash(score_core)}
    write_immutable_private_json(paths["score"], score)
    receipt = {
        "schema_version": RECEIPT_SCHEMA, "status": "paired_scoring_complete",
        "score_content_sha256": score["content_sha256"],
        "score_file_sha256": sha256_file(paths["score"]),
        "reveal_content_sha256": reveal["content_sha256"],
        "reveal_file_sha256": sha256_file(paths["reveal"]),
        "reveal_count": 1, "speaker_count": metrics["speaker_count"],
        "enrolled_speaker_count": metrics["enrolled_speaker_count"],
        "context_only": metrics["context_only"],
        "voice_augmented": metrics["voice_augmented"],
        "enrolled_context_only": metrics["enrolled_context_only"],
        "enrolled_voice_augmented": metrics["enrolled_voice_augmented"],
        "paired": metrics["paired"],
        "did_reveal_gold_to_workers": False, "did_regenerate_predictions": False,
        "did_mutate_profiles_or_references": False,
        "did_enable_default_integration": False,
        "did_run_historical_reprocessing": False, "mode": "0600",
    }
    write_immutable_private_json(paths["receipt"], receipt)
    return {**receipt, "idempotent_replay": False}


def replay_generation5_e3(*, runtime_root: Path = DEFAULT_RUNTIME_ROOT) -> dict[str, Any]:
    paths = _paths(runtime_root)
    for key in ("reveal", "score", "receipt"):
        require_private_file(paths[key], paths["root"])
    reveal, score, receipt = (_read_json(paths[key]) for key in ("reveal", "score", "receipt"))
    reveal_core = {key: value for key, value in reveal.items() if key != "content_sha256"}
    score_core = {key: value for key, value in score.items() if key != "content_sha256"}
    metrics = score.get("metrics")
    expected_receipt = {
        "schema_version": RECEIPT_SCHEMA, "status": "paired_scoring_complete",
        "score_content_sha256": score.get("content_sha256"), "score_file_sha256": sha256_file(paths["score"]),
        "reveal_content_sha256": reveal.get("content_sha256"), "reveal_file_sha256": sha256_file(paths["reveal"]),
        "reveal_count": 1, "speaker_count": metrics.get("speaker_count"),
        "enrolled_speaker_count": metrics.get("enrolled_speaker_count"),
        "context_only": metrics.get("context_only"), "voice_augmented": metrics.get("voice_augmented"),
        "enrolled_context_only": metrics.get("enrolled_context_only"),
        "enrolled_voice_augmented": metrics.get("enrolled_voice_augmented"),
        "paired": metrics.get("paired"), "did_reveal_gold_to_workers": False,
        "did_regenerate_predictions": False, "did_mutate_profiles_or_references": False,
        "did_enable_default_integration": False, "did_run_historical_reprocessing": False,
        "mode": "0600",
    }
    if (
        _canonical_hash(reveal_core) != reveal.get("content_sha256")
        or reveal.get("reveal_count") != 1
        or _canonical_hash(score_core) != score.get("content_sha256")
        or not isinstance(metrics, Mapping) or metrics.get("speaker_count") != e2.EXPECTED_SPEAKER_COUNT
        or receipt != expected_receipt
    ):
        raise Generation5E3Error("Frozen E3 scoring evidence drifted.")
    return {**receipt, "replay_schema_version": REPLAY_SCHEMA, "idempotent_replay": True}
