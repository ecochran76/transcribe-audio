"""Fail-closed generation-2 evaluation reveal and feasibility preflight."""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
from pathlib import Path
from typing import Any, Mapping, Optional

import acoustic_generation2_authority as generation2
from acoustic_audio_derivatives import (
    ensure_private_tree,
    require_private_file,
    sha256_file,
    write_immutable_private_json,
)


PREVIEW_SCHEMA = (
    "transcribe-audio.verification-generation-2-evaluation-preflight-preview.v1"
)
MANIFEST_SCHEMA = (
    "transcribe-audio.verification-generation-2-evaluation-stop-manifest.v1"
)
RECEIPT_SCHEMA = (
    "transcribe-audio.verification-generation-2-evaluation-stop-receipt.v1"
)
REPLAY_SCHEMA = (
    "transcribe-audio.verification-generation-2-evaluation-stop-replay.v1"
)
EXPECTED_PARENT_AUTHORITY_ID = (
    "generation-2-pre-reveal-authority-e36736a176600d5536c7c668"
)
EXPECTED_PARENT_CONTENT_SHA256 = (
    "e36736a176600d5536c7c6688ce00d04165955cf09d69cd67d2bb1b082ef61ad"
)
EXPECTED_PREVIEW_CONTENT_SHA256 = (
    "b83368b7bca2c5634f98c511844e82d78e87a954e99468a611b23efc5c0ff169"
)
DEFAULT_PARENT_RUNTIME_ROOT = Path(
    "~/.local/state/transcribe-audio/plan-0037/generation-2-pre-reveal"
)
DEFAULT_PARENT_MANIFEST = DEFAULT_PARENT_RUNTIME_ROOT / (
    "authorities/generation-2-pre-reveal-authority-e36736a176600d5536c7c668/"
    "manifest.json"
)
DEFAULT_CORPUS_MANIFEST = Path(
    "~/.local/state/transcribe-audio/plan-0037/corpora/"
    "acoustic-corpus-4a2b13e7bdc201f694af2f43/manifest.json"
)
DEFAULT_RUNTIME_ROOT = Path(
    "~/.local/state/transcribe-audio/plan-0037/generation-2-evaluation"
)
SHA256_RE = re.compile(r"[a-f0-9]{64}")
COMMIT_RE = re.compile(r"[a-f0-9]{40}")


class Generation2EvaluationError(ValueError):
    """Raised when the successor reveal or stop authority is invalid."""


def _canonical_hash(value: Any) -> str:
    return generation2._canonical_hash(value)


def _private_object(path: Path, root: Path) -> tuple[dict[str, Any], str]:
    selected = path.expanduser().resolve(strict=True)
    require_private_file(selected, root.expanduser().absolute())
    body = selected.read_bytes()
    try:
        value = json.loads(body)
    except json.JSONDecodeError as exc:
        raise Generation2EvaluationError("Private authority is invalid JSON.") from exc
    if not isinstance(value, dict):
        raise Generation2EvaluationError("Private authority body is invalid.")
    return value, hashlib.sha256(body).hexdigest()


def _git(args: list[str]) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=Path(__file__).resolve().parent,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise Generation2EvaluationError("Repository authority is unavailable.")
    return result.stdout.strip()


def _repository_authority() -> dict[str, Any]:
    status = _git(["status", "--porcelain=v1", "--untracked-files=normal"])
    upstream = _git(["rev-list", "--left-right", "--count", "HEAD...@{upstream}"])
    if status or upstream.split() != ["0", "0"]:
        raise Generation2EvaluationError("Repository must be clean and upstream-even.")
    return {
        "commit": _git(["rev-parse", "HEAD"]),
        "module_sha256": sha256_file(Path(__file__).resolve()),
        "clean": True,
        "upstream_ahead": 0,
        "upstream_behind": 0,
    }


def _validate_repository_authority(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != {
        "commit", "module_sha256", "clean", "upstream_ahead", "upstream_behind"
    }:
        raise Generation2EvaluationError("Frozen repository authority is invalid.")
    commit = str(value.get("commit") or "")
    current = _repository_authority()
    if (
        not COMMIT_RE.fullmatch(commit)
        or value.get("clean") is not True
        or value.get("upstream_ahead") != 0
        or value.get("upstream_behind") != 0
        or _git(["merge-base", "--is-ancestor", commit, current["commit"]])
    ):
        raise Generation2EvaluationError("Frozen repository authority is invalid.")
    blob = subprocess.run(
        ["git", "show", f"{commit}:acoustic_generation2_evaluation.py"],
        cwd=Path(__file__).resolve().parent,
        check=False,
        capture_output=True,
    )
    if (
        blob.returncode != 0
        or hashlib.sha256(blob.stdout).hexdigest() != value.get("module_sha256")
        or sha256_file(Path(__file__).resolve()) != value.get("module_sha256")
    ):
        raise Generation2EvaluationError("Evaluation module authority drifted.")
    return dict(value)


def _parent_authority(
    path: Path, *, parent_runtime_root: Path
) -> tuple[dict[str, Any], str]:
    root = parent_runtime_root.expanduser().absolute()
    selected = path.expanduser().resolve(strict=True)
    expected = (
        root / "authorities" / EXPECTED_PARENT_AUTHORITY_ID / "manifest.json"
    ).resolve(strict=True)
    if selected != expected:
        raise Generation2EvaluationError("Applied pre-reveal parent path drifted.")
    parent, file_sha256 = _private_object(selected, root)
    core = {
        key: value
        for key, value in parent.items()
        if key not in {"authority_id", "content_sha256"}
    }
    actions = parent.get("authorized_actions")
    preview = parent.get("preview")
    if (
        parent.get("schema_version")
        != "transcribe-audio.verification-generation-2-pre-reveal-manifest.v1"
        or parent.get("authority_id") != EXPECTED_PARENT_AUTHORITY_ID
        or parent.get("content_sha256") != EXPECTED_PARENT_CONTENT_SHA256
        or _canonical_hash(core) != EXPECTED_PARENT_CONTENT_SHA256
        or not isinstance(preview, Mapping)
        or preview.get("content_sha256") != EXPECTED_PREVIEW_CONTENT_SHA256
        or not isinstance(actions, Mapping)
        or actions.get("reveal_evaluation") is not True
        or actions.get("prepare_evaluation_audio") is not True
        or actions.get("freeze_evaluation_windows") is not True
        or any(
            actions.get(key) is not False
            for key in (
                "run_models", "score_trials", "calculate_terminal_metrics",
                "make_terminal_decision",
            )
        )
        or parent.get("exact_trial_child_required_before_model_or_score_execution")
        is not True
    ):
        raise Generation2EvaluationError("Applied pre-reveal parent drifted.")
    receipt, _ = _private_object(selected.parent / "apply-receipt.json", root)
    expected_receipt = {
        "schema_version": (
            "transcribe-audio.verification-generation-2-pre-reveal-receipt.v1"
        ),
        "authority_id": EXPECTED_PARENT_AUTHORITY_ID,
        "authority_content_sha256": EXPECTED_PARENT_CONTENT_SHA256,
        "preview_id": preview["preview_id"],
        "preview_content_sha256": preview["content_sha256"],
        "manifest_sha256": file_sha256,
        "evaluation_reveal_authorized": True,
        "model_execution_authorized": False,
        "trial_scoring_authorized": False,
        "contains_private_evaluation": False,
        "contains_device_labels": False,
        "mode": "0600",
        "will_perform_external_write": False,
    }
    if receipt != expected_receipt:
        raise Generation2EvaluationError("Applied pre-reveal receipt drifted.")
    return parent, file_sha256


def _source_binding(record: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    source = record.get("source_blob")
    lineage = record.get("transcript_lineage")
    if not isinstance(source, Mapping) or not isinstance(lineage, Mapping):
        raise Generation2EvaluationError("Evaluation source lineage is invalid.")
    source_path = Path(str(source.get("stored_path") or "")).expanduser().absolute()
    transcript_path = Path(
        str(lineage.get("current_artifact_path") or "")
    ).expanduser().absolute()
    require_private_file(source_path, source_path.parent)
    require_private_file(transcript_path, transcript_path.parent)
    if (
        not SHA256_RE.fullmatch(str(source.get("sha256") or ""))
        or sha256_file(source_path) != source.get("sha256")
        or source_path.stat().st_size != source.get("bytes")
        or not SHA256_RE.fullmatch(str(lineage.get("current_artifact_sha256") or ""))
        or sha256_file(transcript_path) != lineage.get("current_artifact_sha256")
    ):
        raise Generation2EvaluationError("Evaluation source lineage drifted.")
    return dict(source), dict(lineage)


def _gold_projection(record: Mapping[str, Any]) -> tuple[dict[str, Any], set[str]]:
    gold = record.get("operator_gold")
    if not isinstance(gold, Mapping) or not isinstance(gold.get("speaker_truth"), list):
        raise Generation2EvaluationError("Evaluation gold is invalid.")
    truths = []
    subjects: set[str] = set()
    labels: set[str] = set()
    for raw in gold["speaker_truth"]:
        if not isinstance(raw, Mapping):
            raise Generation2EvaluationError("Evaluation speaker truth is invalid.")
        label = str(raw.get("speaker_label") or "")
        outcome = str(raw.get("outcome") or "")
        subject = raw.get("subject_id")
        if not label or label in labels or outcome not in {"person", "mixed", "unknown"}:
            raise Generation2EvaluationError("Evaluation speaker truth is invalid.")
        labels.add(label)
        item = {"speaker_label": label, "outcome": outcome, "subject_id": subject}
        if outcome == "person":
            if not generation2.verification._OPAQUE_ID_RE.fullmatch(str(subject or "")):
                raise Generation2EvaluationError("Evaluation subject ID is invalid.")
            subjects.add(str(subject))
        elif subject is not None:
            raise Generation2EvaluationError("Excluded gold may not carry a subject ID.")
        truths.append(item)
    groups = gold.get("same_person_label_groups")
    if not isinstance(groups, list):
        raise Generation2EvaluationError("Evaluation label groups are invalid.")
    return {
        "gold_id": gold.get("gold_id"),
        "speaker_truth": truths,
        "same_person_label_groups": json.loads(json.dumps(groups)),
    }, subjects


def _evaluate(
    *, parent_manifest_path: Path, corpus_manifest_path: Path,
    parent_runtime_root: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    parent, parent_manifest_sha256 = _parent_authority(
        parent_manifest_path, parent_runtime_root=parent_runtime_root
    )
    preview = parent["preview"]
    seal = preview.get("successor_seal")
    if not isinstance(seal, Mapping):
        raise Generation2EvaluationError("Successor seal is missing.")
    corpus_path = corpus_manifest_path.expanduser().absolute()
    corpus, corpus_file_sha256 = _private_object(corpus_path, corpus_path.parent)
    if (
        corpus_path.name != "manifest.json"
        or corpus_path.parent.name != seal.get("corpus_id")
        or corpus_file_sha256 != seal.get("corpus_manifest_sha256")
        or corpus.get("content_sha256") != seal.get("corpus_content_sha256")
        or corpus.get("corpus_id") != seal.get("corpus_id")
        or not isinstance(corpus.get("recordings"), list)
    ):
        raise Generation2EvaluationError("Successor corpus drifted.")
    by_split: dict[str, list[Mapping[str, Any]]] = {
        name: [item for item in corpus["recordings"]
               if isinstance(item, Mapping) and item.get("split") == name]
        for name in ("development", "calibration", "evaluation")
    }
    if {name: len(values) for name, values in by_split.items()} != seal.get("split_counts"):
        raise Generation2EvaluationError("Successor split counts drifted.")
    for key in ("recording_id", "conversation_id"):
        sets = [{str(item.get(key) or "") for item in by_split[name]}
                for name in ("development", "calibration", "evaluation")]
        if any(sets[a] & sets[b] for a in range(3) for b in range(a + 1, 3)):
            raise Generation2EvaluationError(f"Successor {key} overlaps another split.")
    source_sets = [
        {str((item.get("source_blob") or {}).get("sha256") or "") for item in by_split[name]}
        for name in ("development", "calibration", "evaluation")
    ]
    if any(source_sets[a] & source_sets[b] for a in range(3) for b in range(a + 1, 3)):
        raise Generation2EvaluationError("Successor source content overlaps another split.")
    evaluation = sorted(by_split["evaluation"], key=lambda item: str(item.get("recording_id")))
    safe_membership = [
        {
            "recording_id": item.get("recording_id"),
            "conversation_id": item.get("conversation_id"),
            "source_sha256": (item.get("source_blob") or {}).get("sha256"),
            "split": "evaluation",
        }
        for item in evaluation
    ]
    if (
        len(evaluation) != seal.get("evaluation_recording_count")
        or _canonical_hash(safe_membership) != seal.get("evaluation_record_set_sha256")
    ):
        raise Generation2EvaluationError("Evaluation membership drifted.")
    private_records = []
    evaluation_subjects: set[str] = set()
    eligible_label_count = 0
    for record in evaluation:
        source, lineage = _source_binding(record)
        gold, subjects = _gold_projection(record)
        evaluation_subjects.update(subjects)
        eligible_label_count += sum(
            item["outcome"] == "person" for item in gold["speaker_truth"]
        )
        private_records.append({
            "recording_id": record["recording_id"],
            "conversation_id": record["conversation_id"],
            "source_sha256": source["sha256"],
            "source_bytes": source["bytes"],
            "transcript_artifact_sha256": lineage["current_artifact_sha256"],
            "operator_gold": gold,
        })
    profiles = preview.get("profiles")
    matrix = preview.get("candidate_matrix")
    minimum = preview.get("minimum_evidence_policy")
    if not isinstance(profiles, list) or not isinstance(matrix, list) or not isinstance(minimum, Mapping):
        raise Generation2EvaluationError("Frozen evaluation policy is incomplete.")
    profile_by_id = {
        str(item.get("profile_id")): dict(item)
        for item in profiles if isinstance(item, Mapping)
    }
    profile_subjects = {str(item.get("person_ref_id")) for item in profiles if isinstance(item, Mapping)}
    matched_subjects = evaluation_subjects & profile_subjects
    units = []
    for raw in matrix:
        if not isinstance(raw, Mapping) or not isinstance(raw.get("profile_ids"), list):
            raise Generation2EvaluationError("Candidate matrix is invalid.")
        selected_profiles = [profile_by_id.get(str(value)) for value in raw["profile_ids"]]
        if any(item is None for item in selected_profiles):
            raise Generation2EvaluationError("Candidate matrix profile binding drifted.")
        unit_subjects = {str(item["person_ref_id"]) for item in selected_profiles if item}
        matched = evaluation_subjects & unit_subjects
        units.append({
            "candidate_id": raw.get("candidate_id"),
            "method_id": raw.get("method_id"),
            "profile_count": len(selected_profiles),
            "matched_evaluation_subject_count": len(matched),
            "maximum_genuine_trials": 0 if not matched else None,
            "maximum_impostor_trials": 0 if not matched else None,
            "required_genuine_trials": minimum.get("genuine_trials_per_model_method_unit"),
            "required_impostor_trials": minimum.get("impostor_trials_per_model_method_unit"),
            "required_open_set_trials": minimum.get("open_set_trials_per_model_method_unit"),
            "feasibility": "blocked_required_known_subject_class_absent" if not matched else "requires_window_freeze",
        })
    global_stop = any(item["matched_evaluation_subject_count"] == 0 for item in units)
    if not global_stop:
        status = "preflight_pass_requires_preparation"
        reason_code = None
    else:
        status = "global_stop_required"
        reason_code = "trial_class_denominator_below_policy"
    core = {
        "schema_version": PREVIEW_SCHEMA,
        "status": status,
        "reason_code": reason_code,
        "parent_authority_id": parent["authority_id"],
        "parent_content_sha256": parent["content_sha256"],
        "parent_manifest_sha256": parent_manifest_sha256,
        "parent_preview_content_sha256": preview["content_sha256"],
        "corpus_id": corpus["corpus_id"],
        "corpus_content_sha256": corpus["content_sha256"],
        "corpus_manifest_sha256": corpus_file_sha256,
        "evaluation_record_set_sha256": seal["evaluation_record_set_sha256"],
        "evaluation_recording_count": len(evaluation),
        "evaluation_conversation_count": len({item["conversation_id"] for item in private_records}),
        "eligible_person_label_count": eligible_label_count,
        "evaluation_subject_count": len(evaluation_subjects),
        "profile_subject_count": len(profile_subjects),
        "matched_profile_subject_count": len(matched_subjects),
        "evaluation_subject_set_sha256": _canonical_hash(sorted(evaluation_subjects)),
        "profile_subject_set_sha256": _canonical_hash(sorted(profile_subjects)),
        "candidate_unit_count": len(units),
        "units": units,
        "terminal_policy_sha256": preview.get("terminal_decision_policy_sha256"),
        "minimum_evidence_policy": dict(minimum),
        "did_reveal_private_gold": True,
        "did_read_audio": False,
        "did_prepare_audio": False,
        "did_freeze_windows": False,
        "did_build_exact_trial_child": False,
        "did_run_models": False,
        "did_score_trials": False,
        "did_calculate_terminal_metrics": False,
        "did_select_model_or_method": False,
        "will_perform_external_write": False,
        "contains_subject_ids": False,
        "contains_names_or_emails": False,
        "contains_paths": False,
        "contains_raw_audio": False,
        "contains_embeddings_or_vectors": False,
        "contains_biometric_scores": False,
    }
    content_sha256 = _canonical_hash(core)
    portable = {
        **core,
        "preview_id": f"generation-2-evaluation-preflight-{content_sha256[:24]}",
        "content_sha256": content_sha256,
    }
    private = {
        "records": private_records,
        "evaluation_subjects": sorted(evaluation_subjects),
        "profile_subjects": sorted(profile_subjects),
    }
    return portable, private


def preview_generation2_evaluation_preflight(
    *, parent_manifest_path: Path = DEFAULT_PARENT_MANIFEST,
    corpus_manifest_path: Path = DEFAULT_CORPUS_MANIFEST,
    parent_runtime_root: Path = DEFAULT_PARENT_RUNTIME_ROOT,
) -> dict[str, Any]:
    """Reveal in memory and return only a portable structural preflight."""
    portable, _ = _evaluate(
        parent_manifest_path=parent_manifest_path,
        corpus_manifest_path=corpus_manifest_path,
        parent_runtime_root=parent_runtime_root,
    )
    return portable


def _paths(root: Path, run_id: str = "") -> dict[str, Path]:
    selected_root = root.expanduser().absolute()
    base = selected_root / "runs"
    run = base / run_id if run_id else base
    return {
        "root": selected_root,
        "base": base,
        "run": run,
        "manifest": run / "private-manifest.json",
        "receipt": run / "stop-receipt.json",
    }


def _existing_runs(root: Path) -> list[Path]:
    paths = _paths(root)
    if not paths["base"].exists():
        return []
    if not paths["base"].is_dir() or paths["base"].is_symlink():
        raise Generation2EvaluationError("Evaluation run root is invalid.")
    manifests = []
    for child in sorted(paths["base"].iterdir()):
        if not child.is_dir() or child.is_symlink():
            raise Generation2EvaluationError("Unknown evaluation run entry exists.")
        if {item.name for item in child.iterdir()} != {
            "private-manifest.json", "stop-receipt.json"
        }:
            raise Generation2EvaluationError("Partial or unknown evaluation run exists.")
        manifests.append(child / "private-manifest.json")
    return manifests


def _manifest_core(
    preview: Mapping[str, Any], private: Mapping[str, Any], repository: Mapping[str, Any]
) -> dict[str, Any]:
    return {
        "schema_version": MANIFEST_SCHEMA,
        "status": "terminal_stop",
        "reason_code": "trial_class_denominator_below_policy",
        "preview": dict(preview),
        "private_reveal": dict(private),
        "repository_authority": dict(repository),
        "authorized_actions_after_stop": {
            "prepare_evaluation_audio": False,
            "freeze_evaluation_windows": False,
            "build_exact_trial_child": False,
            "run_models": False,
            "score_trials": False,
            "calculate_terminal_metrics": False,
            "make_terminal_model_or_method_selection": False,
        },
        "contains_private_evaluation": True,
        "contains_raw_audio": False,
        "contains_embeddings_or_vectors": False,
        "contains_biometric_scores": False,
        "will_perform_external_write": False,
    }


def _receipt(
    preview: Mapping[str, Any], run_id: str, content_sha256: str,
    manifest_sha256: str,
) -> dict[str, Any]:
    return {
        "schema_version": RECEIPT_SCHEMA,
        "run_id": run_id,
        "status": "terminal_stop",
        "reason_code": "trial_class_denominator_below_policy",
        "authority_content_sha256": content_sha256,
        "manifest_sha256": manifest_sha256,
        "preview_id": preview["preview_id"],
        "preview_content_sha256": preview["content_sha256"],
        "evaluation_recording_count": preview["evaluation_recording_count"],
        "evaluation_subject_count": preview["evaluation_subject_count"],
        "matched_profile_subject_count": preview["matched_profile_subject_count"],
        "candidate_unit_count": preview["candidate_unit_count"],
        "minimum_genuine_trials": preview["minimum_evidence_policy"]["genuine_trials_per_model_method_unit"],
        "minimum_impostor_trials": preview["minimum_evidence_policy"]["impostor_trials_per_model_method_unit"],
        "audio_preparation_authorized": False,
        "window_freeze_authorized": False,
        "exact_trial_child_construction_authorized": False,
        "model_execution_authorized": False,
        "trial_scoring_authorized": False,
        "terminal_metrics_authorized": False,
        "terminal_model_or_method_selection_authorized": False,
        "contains_subject_ids": False,
        "contains_names_or_emails": False,
        "contains_paths": False,
        "contains_raw_audio": False,
        "contains_embeddings_or_vectors": False,
        "contains_biometric_scores": False,
        "mode": "0600",
        "will_perform_external_write": False,
    }


def apply_generation2_evaluation_stop(
    reviewed_preview: Mapping[str, Any], *, expected_preview_content_sha256: str,
    parent_manifest_path: Path = DEFAULT_PARENT_MANIFEST,
    corpus_manifest_path: Path = DEFAULT_CORPUS_MANIFEST,
    parent_runtime_root: Path = DEFAULT_PARENT_RUNTIME_ROOT,
    runtime_root: Optional[Path] = None,
) -> dict[str, Any]:
    """Persist the reviewed private reveal and mandatory pre-model STOP."""
    preview, private = _evaluate(
        parent_manifest_path=parent_manifest_path,
        corpus_manifest_path=corpus_manifest_path,
        parent_runtime_root=parent_runtime_root,
    )
    if (
        dict(reviewed_preview) != preview
        or preview.get("content_sha256") != expected_preview_content_sha256
        or preview.get("status") != "global_stop_required"
        or preview.get("reason_code") != "trial_class_denominator_below_policy"
    ):
        raise Generation2EvaluationError("Reviewed evaluation preflight is stale.")
    repository = _repository_authority()
    core = _manifest_core(preview, private, repository)
    content_sha256 = _canonical_hash(core)
    run_id = f"generation-2-evaluation-stop-{content_sha256[:24]}"
    root = runtime_root or DEFAULT_RUNTIME_ROOT
    existing = _existing_runs(root)
    if len(existing) > 1:
        raise Generation2EvaluationError("Multiple generation-2 evaluation runs exist.")
    if existing:
        return replay_generation2_evaluation_stop(
            existing[0], parent_manifest_path=parent_manifest_path,
            corpus_manifest_path=corpus_manifest_path,
            parent_runtime_root=parent_runtime_root, runtime_root=root,
        )
    paths = _paths(root, run_id)
    ensure_private_tree(paths["root"], paths["run"])
    manifest = {**core, "run_id": run_id, "content_sha256": content_sha256}
    write_immutable_private_json(paths["manifest"], manifest)
    receipt = _receipt(preview, run_id, content_sha256, sha256_file(paths["manifest"]))
    write_immutable_private_json(paths["receipt"], receipt)
    return {
        **receipt,
        "manifest_path": str(paths["manifest"]),
        "receipt_path": str(paths["receipt"]),
        "idempotent": False,
    }


def replay_generation2_evaluation_stop(
    manifest_path: Path, *, parent_manifest_path: Path = DEFAULT_PARENT_MANIFEST,
    corpus_manifest_path: Path = DEFAULT_CORPUS_MANIFEST,
    parent_runtime_root: Path = DEFAULT_PARENT_RUNTIME_ROOT,
    runtime_root: Optional[Path] = None,
) -> dict[str, Any]:
    """Replay the full private reveal and portable terminal-stop receipt."""
    root = (runtime_root or DEFAULT_RUNTIME_ROOT).expanduser().absolute()
    selected = manifest_path.expanduser().resolve(strict=True)
    manifest, manifest_sha256 = _private_object(selected, root)
    preview, private = _evaluate(
        parent_manifest_path=parent_manifest_path,
        corpus_manifest_path=corpus_manifest_path,
        parent_runtime_root=parent_runtime_root,
    )
    repository = _validate_repository_authority(manifest.get("repository_authority"))
    core = _manifest_core(preview, private, repository)
    content_sha256 = _canonical_hash(core)
    run_id = f"generation-2-evaluation-stop-{content_sha256[:24]}"
    expected_manifest = {**core, "run_id": run_id, "content_sha256": content_sha256}
    if (
        manifest != expected_manifest
        or selected != _paths(root, run_id)["manifest"]
        or _existing_runs(root) != [selected]
    ):
        raise Generation2EvaluationError("Evaluation stop manifest replay mismatch.")
    receipt_path = selected.parent / "stop-receipt.json"
    receipt, _ = _private_object(receipt_path, root)
    expected_receipt = _receipt(preview, run_id, content_sha256, manifest_sha256)
    if receipt != expected_receipt:
        raise Generation2EvaluationError("Evaluation stop receipt replay mismatch.")
    return {
        "schema_version": REPLAY_SCHEMA,
        **expected_receipt,
        "full_body_match": True,
        "idempotent": True,
    }
