"""Private Generation-3 speaker-gold preview, freeze, and replay authority."""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
import subprocess
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import acoustic_audio_derivatives as derivatives
import acoustic_biometric_references as references
import acoustic_generation3_authority as cohort
import acoustic_training_expansion as training
from acoustic_audio_derivatives import (
    ensure_private_tree,
    require_private_file,
    sha256_file,
    write_immutable_private_json,
)


PREVIEW_SCHEMA = "transcribe-audio.generation3-gold-preview.v1"
MANIFEST_SCHEMA = "transcribe-audio.generation3-gold-manifest.v1"
RECEIPT_SCHEMA = "transcribe-audio.generation3-gold-receipt.v1"
REPLAY_SCHEMA = "transcribe-audio.generation3-gold-replay.v1"
DEFAULT_RUNTIME_ROOT = cohort.DEFAULT_RUNTIME_ROOT
ALLOWED_OUTCOMES = ("enrolled", "open_set", "mixed", "unknown")


class Generation3GoldError(ValueError):
    """Raised when Generation-3 gold cannot remain exact and private."""


def _canonical_hash(value: Any) -> str:
    return cohort._canonical_hash(value)


def _normalized_identity(value: Any) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", str(value or "").casefold()))


def _contains_token_sequence(text: Any, claim: Any) -> bool:
    text_tokens = _normalized_identity(text).split()
    claim_tokens = _normalized_identity(claim).split()
    if not claim_tokens:
        return False
    size = len(claim_tokens)
    return any(
        text_tokens[index:index + size] == claim_tokens
        for index in range(len(text_tokens) - size + 1)
    )


def _valid_enrolled_claim(identity_name: Any, claim: Any) -> bool:
    identity = _normalized_identity(identity_name)
    normalized_claim = _normalized_identity(claim)
    first_token = identity.split()[0] if identity else ""
    return bool(normalized_claim) and normalized_claim in {identity, first_token}


def _read_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise Generation3GoldError("Generation-3 gold JSON is unreadable.") from exc
    if not isinstance(value, dict):
        raise Generation3GoldError("Generation-3 gold JSON must be an object.")
    return value


def _transcript_evidence(
    transcript: Mapping[str, Any], *, indices: Any, target_label: str,
    method: str, identity_claim: Any,
) -> dict[str, Any]:
    utterances = transcript.get("utterances")
    if (
        method not in {"self_identification", "direct_address_response"}
        or not isinstance(indices, list)
        or not indices
        or not isinstance(utterances, list)
    ):
        raise Generation3GoldError("Transcript gold evidence is invalid.")
    records = []
    target_present = False
    claim = " ".join(str(identity_claim or "").split())
    if not claim:
        raise Generation3GoldError("Transcript identity claim is empty.")
    for raw_index in indices:
        if not isinstance(raw_index, int) or not 0 <= raw_index < len(utterances):
            raise Generation3GoldError("Transcript evidence index is invalid.")
        utterance = utterances[raw_index]
        if not isinstance(utterance, Mapping):
            raise Generation3GoldError("Transcript evidence utterance is invalid.")
        label = str(utterance.get("speaker") or "")
        text = " ".join(str(utterance.get("text") or "").split())
        start = utterance.get("start")
        end = utterance.get("end")
        if not text or not isinstance(start, int) or not isinstance(end, int):
            raise Generation3GoldError("Transcript evidence body is invalid.")
        target_present = target_present or label == target_label
        records.append(
            {
                "utterance_index": raw_index,
                "speaker_label": label,
                "start_milliseconds": start,
                "end_milliseconds": end,
                "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                "bounded_text": text[:500],
            }
        )
    if not target_present:
        raise Generation3GoldError(
            "Transcript evidence does not include the target label."
        )
    joined_text = " ".join(record["bounded_text"] for record in records)
    if not _contains_token_sequence(joined_text, claim):
        raise Generation3GoldError(
            "Transcript evidence does not contain the identity claim."
        )
    return {
        "kind": "transcript",
        "method": method,
        "identity_claim": claim,
        "identity_claim_sha256": hashlib.sha256(
            _normalized_identity(claim).encode("utf-8")
        ).hexdigest(),
        "utterances": records,
    }


def _operator_evidence(value: Mapping[str, Any]) -> dict[str, Any]:
    statement = " ".join(str(value.get("statement") or "").split())
    if not statement:
        raise Generation3GoldError("Operator confirmation is empty.")
    return {
        "kind": "operator_confirmation",
        "statement_sha256": hashlib.sha256(statement.encode("utf-8")).hexdigest(),
        "bounded_statement": statement[:500],
        "campaign": "plan-0050-generation-3",
    }


def _cohort_context(
    *, cohort_manifest_path: Path, conversations: Sequence[Mapping[str, Any]],
    source_root: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    try:
        replay = cohort.replay_generation3_cohort(
            cohort_manifest_path,
            conversations=conversations,
            source_root=source_root,
        )
    except (cohort.Generation3AuthorityError, ValueError) as exc:
        raise Generation3GoldError("Generation-3 cohort replay failed.") from exc
    if (
        replay.get("status") != "applied_membership_only_gold_not_frozen"
        or replay.get("idempotent_replay") is not True
        or replay.get("action_vector", {}).get(
            "build_private_gold_review_packet"
        ) is not True
        or replay.get("action_vector", {}).get("freeze_gold") is not False
    ):
        raise Generation3GoldError(
            "Generation-3 cohort does not authorize gold review."
        )
    path = cohort_manifest_path.expanduser().absolute()
    require_private_file(path, DEFAULT_RUNTIME_ROOT.expanduser().absolute())
    manifest = _read_object(path)
    preview = manifest.get("preview")
    private = manifest.get("private_inputs")
    safe_units = (
        preview.get("membership", {}).get("conversations")
        if isinstance(preview, Mapping)
        else None
    )
    private_units = (
        private.get("conversations") if isinstance(private, Mapping) else None
    )
    if not isinstance(safe_units, list) or not isinstance(private_units, list):
        raise Generation3GoldError("Generation-3 cohort membership is incomplete.")
    private_by_input = {
        str(item.get("conversation_input_id") or ""): item
        for item in private_units
        if isinstance(item, Mapping)
    }
    records = []
    for safe in safe_units:
        if not isinstance(safe, Mapping):
            raise Generation3GoldError("Generation-3 safe membership is invalid.")
        private_unit = private_by_input.get(
            str(safe.get("conversation_input_id") or "")
        )
        if not isinstance(private_unit, Mapping):
            raise Generation3GoldError("Generation-3 private membership is invalid.")
        transcript_path = Path(str(private_unit.get("transcript_path") or ""))
        transcript = _read_object(transcript_path)
        bindings = private_unit.get("diarized_label_bindings")
        if not isinstance(bindings, list):
            raise Generation3GoldError("Generation-3 label bindings are unavailable.")
        records.append(
            {
                "source_sha256": safe["source_sha256"],
                "transcript_sha256": safe["transcript_sha256"],
                "recording_id": safe["recording_id"],
                "conversation_id": safe["conversation_id"],
                "transcript": transcript,
                "bindings": bindings,
            }
        )
    return manifest, records


def _training_context() -> tuple[dict[str, Any], dict[str, Any]]:
    try:
        paths = training._existing_manifests(cohort.DEFAULT_TRAINING_RUNTIME_ROOT)
    except training.TrainingExpansionError as exc:
        raise Generation3GoldError(str(exc)) from exc
    if len(paths) != 1:
        raise Generation3GoldError("Active training intake authority is unavailable.")
    manifest = _read_object(paths[0])
    preview = manifest.get("preview")
    private = manifest.get("private_inputs")
    safe_units = preview.get("conversations") if isinstance(preview, Mapping) else None
    private_units = (
        private.get("conversations") if isinstance(private, Mapping) else None
    )
    if not isinstance(safe_units, list) or not isinstance(private_units, list):
        raise Generation3GoldError("Active training intake lineage is incomplete.")
    private_by_input = {
        str(item.get("conversation_input_id") or ""): item
        for item in private_units
        if isinstance(item, Mapping)
    }
    return manifest, {
        str(item["source_sha256"]): {
            "safe": item,
            "private": private_by_input[str(item["conversation_input_id"])],
        }
        for item in safe_units
        if isinstance(item, Mapping)
    }


def _active_subject_sources() -> dict[str, set[tuple[str, str]]]:
    database = references.DEFAULT_RUNTIME_ROOT.expanduser() / "references.sqlite3"
    try:
        connection = sqlite3.connect(f"file:{database}?mode=ro", uri=True)
        rows = connection.execute(
            """
            SELECT p.person_ref_id, g.manifest_json
            FROM generations g JOIN profiles p ON p.profile_id = g.profile_id
            WHERE g.status = 'active' AND p.status = 'active'
            ORDER BY p.person_ref_id
            """
        ).fetchall()
    except sqlite3.Error as exc:
        raise Generation3GoldError("Active subject lineage is unavailable.") from exc
    finally:
        if "connection" in locals():
            connection.close()
    result: dict[str, set[tuple[str, str]]] = {}
    for person_ref_id, raw_manifest in rows:
        try:
            manifest = json.loads(raw_manifest)
        except json.JSONDecodeError as exc:
            raise Generation3GoldError("Active subject manifest is invalid.") from exc
        sources = manifest.get("sources") if isinstance(manifest, Mapping) else None
        if not isinstance(sources, list):
            raise Generation3GoldError("Active subject sources are invalid.")
        result[str(person_ref_id)] = {
            (
                str(item.get("source_sha256") or ""),
                str(item.get("speaker_label_id") or ""),
            )
            for item in sources
            if isinstance(item, Mapping)
        }
    if len(result) != 2:
        raise Generation3GoldError("Exactly two active enrolled subjects are required.")
    return result


def _enrolled_bindings(
    values: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    training_manifest, training_units = _training_context()
    active_sources = _active_subject_sources()
    result: dict[str, dict[str, Any]] = {}
    for value in values:
        if not isinstance(value, Mapping):
            raise Generation3GoldError("Enrolled identity binding is invalid.")
        person_ref_id = str(value.get("person_ref_id") or "")
        identity_name = " ".join(str(value.get("identity_name") or "").split())
        source_sha = str(value.get("source_sha256") or "")
        speaker_label = str(value.get("speaker_label") or "")
        unit = training_units.get(source_sha)
        if (
            person_ref_id not in active_sources
            or not identity_name
            or not isinstance(unit, Mapping)
        ):
            raise Generation3GoldError("Enrolled identity binding is unavailable.")
        private = unit["private"]
        binding = next(
            (
                item for item in private.get("diarized_label_bindings") or []
                if isinstance(item, Mapping)
                and item.get("speaker_label") == speaker_label
            ),
            None,
        )
        if (
            not isinstance(binding, Mapping)
            or (source_sha, str(binding.get("speaker_label_id") or ""))
            not in active_sources[person_ref_id]
        ):
            raise Generation3GoldError("Enrolled identity is not active-P3-bound.")
        transcript_path = Path(str(private.get("transcript_path") or ""))
        transcript = _read_object(transcript_path)
        evidence = _transcript_evidence(
            transcript,
            indices=value.get("utterance_indices"),
            target_label=speaker_label,
            method="direct_address_response",
            identity_claim=value.get("identity_claim"),
        )
        if not _valid_enrolled_claim(
            identity_name, evidence["identity_claim"]
        ):
            raise Generation3GoldError(
                "Enrolled identity evidence does not bind the identity name."
            )
        result[person_ref_id] = {
            "person_ref_id": person_ref_id,
            "identity_name": identity_name,
            "identity_name_sha256": hashlib.sha256(
                _normalized_identity(identity_name).encode("utf-8")
            ).hexdigest(),
            "training_intake_id": training_manifest["intake_id"],
            "source_sha256": source_sha,
            "speaker_label_id": binding["speaker_label_id"],
            "evidence": evidence,
        }
    if set(result) != set(active_sources):
        raise Generation3GoldError("Both active enrolled identities must be bound.")
    return result


def _evaluate(
    *, cohort_manifest_path: Path, conversations: Sequence[Mapping[str, Any]],
    source_root: Path, enrolled_identity_bindings: Sequence[Mapping[str, Any]],
    outcomes: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    cohort_manifest, records = _cohort_context(
        cohort_manifest_path=cohort_manifest_path,
        conversations=conversations,
        source_root=source_root,
    )
    enrolled = _enrolled_bindings(enrolled_identity_bindings)
    proposals = {
        (
            str(item.get("source_sha256") or ""),
            str(item.get("speaker_label") or ""),
        ): item
        for item in outcomes
        if isinstance(item, Mapping)
    }
    expected_keys = set()
    gold = []
    subject_conversations: dict[str, set[str]] = {}
    counts = {key: 0 for key in ALLOWED_OUTCOMES}
    membership_sha = cohort_manifest["preview"]["membership_sha256"]
    for record in records:
        binding_by_label = {
            str(item.get("speaker_label") or ""): item
            for item in record["bindings"]
            if isinstance(item, Mapping)
        }
        for speaker_label, binding in sorted(binding_by_label.items()):
            key = (record["source_sha256"], speaker_label)
            expected_keys.add(key)
            proposal = proposals.get(key)
            if not isinstance(proposal, Mapping):
                raise Generation3GoldError(
                    "Every cohort label requires one gold outcome."
                )
            outcome = str(proposal.get("outcome") or "")
            if outcome not in ALLOWED_OUTCOMES:
                raise Generation3GoldError("Generation-3 gold outcome is invalid.")
            identity_name = " ".join(str(proposal.get("identity_name") or "").split())
            person_ref_id = str(proposal.get("person_ref_id") or "")
            evidence_values = proposal.get("evidence")
            if not isinstance(evidence_values, list):
                raise Generation3GoldError("Generation-3 gold evidence is invalid.")
            evidence = []
            for evidence_value in evidence_values:
                if not isinstance(evidence_value, Mapping):
                    raise Generation3GoldError("Generation-3 evidence item is invalid.")
                kind = evidence_value.get("kind")
                if kind == "transcript":
                    evidence.append(
                        _transcript_evidence(
                            record["transcript"],
                            indices=evidence_value.get("utterance_indices"),
                            target_label=speaker_label,
                            method=str(evidence_value.get("method") or ""),
                            identity_claim=evidence_value.get("identity_claim"),
                        )
                    )
                elif kind == "operator_confirmation":
                    evidence.append(_operator_evidence(evidence_value))
                else:
                    raise Generation3GoldError("Generation-3 evidence kind is invalid.")
            if outcome == "enrolled":
                binding_authority = enrolled.get(person_ref_id)
                if (
                    not isinstance(binding_authority, Mapping)
                    or _normalized_identity(identity_name)
                    != _normalized_identity(binding_authority["identity_name"])
                    or not evidence
                ):
                    raise Generation3GoldError("Enrolled gold binding is invalid.")
                for item in evidence:
                    if item["kind"] == "operator_confirmation" and not (
                        _contains_token_sequence(
                            item["bounded_statement"], identity_name
                        )
                    ):
                        raise Generation3GoldError(
                            "Operator evidence does not bind enrolled identity."
                        )
                    if item["kind"] == "transcript" and not (
                        _valid_enrolled_claim(
                            identity_name, item["identity_claim"]
                        )
                    ):
                        raise Generation3GoldError(
                            "Enrolled transcript evidence does not bind identity."
                        )
                subject_id = person_ref_id
            elif outcome == "open_set":
                if not identity_name or not evidence or person_ref_id:
                    raise Generation3GoldError("Open-set gold binding is invalid.")
                for item in evidence:
                    if item["kind"] == "operator_confirmation" and not (
                        _contains_token_sequence(
                            item["bounded_statement"], identity_name
                        )
                    ):
                        raise Generation3GoldError(
                            "Operator evidence does not bind open-set identity."
                        )
                    if item["kind"] == "transcript" and (
                        _normalized_identity(item["identity_claim"])
                        != _normalized_identity(identity_name)
                    ):
                        raise Generation3GoldError(
                            "Open-set transcript evidence does not bind identity."
                        )
                subject_id = "generation3-open-subject-" + _canonical_hash(
                    [membership_sha, _normalized_identity(identity_name)]
                )[:24]
            else:
                if identity_name or person_ref_id or evidence:
                    raise Generation3GoldError(
                        "Mixed/unknown gold must remain unassigned."
                    )
                subject_id = outcome
            counts[outcome] += 1
            if outcome in {"enrolled", "open_set"}:
                subject_conversations.setdefault(subject_id, set()).add(
                    record["conversation_id"]
                )
            gold.append(
                {
                    "source_sha256": record["source_sha256"],
                    "transcript_sha256": record["transcript_sha256"],
                    "recording_id": record["recording_id"],
                    "conversation_id": record["conversation_id"],
                    "speaker_label": speaker_label,
                    "speaker_label_id": binding["speaker_label_id"],
                    "outcome": outcome,
                    "subject_id": subject_id,
                    "identity_name": identity_name or None,
                    "evidence": evidence,
                }
            )
    if (
        len(expected_keys) != 28
        or set(proposals) != expected_keys
        or len(proposals) != len(outcomes)
    ):
        raise Generation3GoldError("Generation-3 gold has duplicate or foreign labels.")
    enrolled_conversations = {
        subject_id: len(subject_conversations.get(subject_id, set()))
        for subject_id in enrolled
    }
    known_subject_count = len(subject_conversations)
    if (
        any(value < 2 for value in enrolled_conversations.values())
        or known_subject_count < 5
        or counts["open_set"] < 1
    ):
        raise Generation3GoldError("Generation-3 gold population minimum failed.")
    gold = sorted(gold, key=lambda item: (item["source_sha256"], item["speaker_label"]))
    core = {
        "schema_version": PREVIEW_SCHEMA,
        "status": "ready_for_independent_review",
        "cohort_authority_id": cohort_manifest["authority_id"],
        "cohort_manifest_sha256": sha256_file(cohort_manifest_path),
        "membership_sha256": membership_sha,
        "enrolled_identity_bindings": sorted(
            enrolled.values(), key=lambda item: item["person_ref_id"]
        ),
        "gold": gold,
        "gold_label_count": len(gold),
        "outcome_counts": counts,
        "known_subject_count": known_subject_count,
        "enrolled_conversation_counts": enrolled_conversations,
        "gold_status": "reviewed_not_frozen",
        "action_vector": {
            "freeze_gold": False,
            "reveal_evaluation": False,
            "prepare_audio": False,
            "freeze_windows": False,
            "construct_exact_trial_child": False,
            "load_or_run_models": False,
            "score_trials": False,
            "calculate_metrics": False,
            "make_terminal_decision": False,
            "mutate_profiles_or_references": False,
            "enable_default_integration": False,
            "run_historical_reprocessing": False,
        },
        "contains_private_gold": True,
        "contains_names": True,
        "contains_bounded_transcript_evidence": True,
        "contains_paths": False,
        "contains_raw_audio": False,
        "contains_embeddings_or_vectors": False,
        "contains_biometric_scores": False,
        "will_perform_external_write": False,
    }
    content_sha = _canonical_hash(core)
    return {
        **core,
        "preview_id": f"generation3-gold-preview-{content_sha[:24]}",
        "content_sha256": content_sha,
    }


def preview_generation3_gold(**kwargs: Any) -> dict[str, Any]:
    """Build the exact private gold packet without freezing or revealing it."""
    return _evaluate(**kwargs)


def portable_gold_projection(preview: Mapping[str, Any]) -> dict[str, Any]:
    """Return a count/hash/action-only projection of a private gold preview."""
    if preview.get("schema_version") != PREVIEW_SCHEMA:
        raise Generation3GoldError("Generation-3 gold preview is invalid.")
    enrolled_counts = list(preview["enrolled_conversation_counts"].values())
    return {
        "schema_version": "transcribe-audio.generation3-gold-portable.v1",
        "status": preview["status"],
        "preview_content_sha256": preview["content_sha256"],
        "membership_sha256": preview["membership_sha256"],
        "gold_body_sha256": _canonical_hash(preview["gold"]),
        "gold_label_count": preview["gold_label_count"],
        "outcome_counts": dict(preview["outcome_counts"]),
        "known_subject_count": preview["known_subject_count"],
        "enrolled_subject_count": len(enrolled_counts),
        "minimum_enrolled_conversation_count": min(enrolled_counts),
        "maximum_enrolled_conversation_count": max(enrolled_counts),
        "action_vector": dict(preview["action_vector"]),
        "contains_private_gold": False,
        "contains_names": False,
        "contains_subject_ids": False,
        "contains_source_membership": False,
        "contains_transcript_text": False,
        "contains_paths": False,
        "contains_raw_audio": False,
        "contains_embeddings_or_vectors": False,
        "contains_biometric_scores": False,
        "will_perform_external_write": False,
    }


def _repository_authority() -> dict[str, Any]:
    current = cohort._repository_authority()
    return {
        **current,
        "gold_module_sha256": sha256_file(Path(__file__).resolve()),
    }


def _validate_repository_authority(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping) or "gold_module_sha256" not in value:
        raise Generation3GoldError("Frozen gold repository authority is invalid.")
    cohort_authority = {
        key: item for key, item in value.items() if key != "gold_module_sha256"
    }
    cohort._validate_repository_authority(cohort_authority)
    commit = str(value.get("commit") or "")
    blob = subprocess.run(
        ["git", "show", f"{commit}:acoustic_generation3_gold.py"],
        cwd=Path(__file__).resolve().parent,
        check=False,
        capture_output=True,
    )
    if (
        blob.returncode != 0
        or hashlib.sha256(blob.stdout).hexdigest()
        != value.get("gold_module_sha256")
        or sha256_file(Path(__file__).resolve()) != value.get("gold_module_sha256")
    ):
        raise Generation3GoldError("Frozen gold module authority drifted.")
    return dict(value)


def _paths(root: Path, gold_id: str = "") -> dict[str, Path]:
    base = root.expanduser().absolute() / "gold-authorities"
    authority = base / gold_id if gold_id else base
    return {
        "root": root.expanduser().absolute(),
        "base": base,
        "authority": authority,
        "manifest": authority / "private-manifest.json",
        "receipt": authority / "receipt.json",
    }


def _existing_manifest(root: Path) -> Optional[Path]:
    base = _paths(root)["base"]
    if not base.exists():
        return None
    if not base.is_dir() or base.is_symlink():
        raise Generation3GoldError("Generation-3 gold authority root is invalid.")
    children = sorted(base.iterdir())
    if len(children) > 1:
        raise Generation3GoldError("Multiple Generation-3 gold authorities exist.")
    if not children:
        return None
    child = children[0]
    if (
        not child.is_dir()
        or child.is_symlink()
        or {item.name for item in child.iterdir()}
        != {"private-manifest.json", "receipt.json"}
    ):
        raise Generation3GoldError("Partial Generation-3 gold authority exists.")
    return child / "private-manifest.json"


def _manifest_core(
    preview: Mapping[str, Any], repository: Mapping[str, Any]
) -> dict[str, Any]:
    actions = dict(preview["action_vector"])
    actions["freeze_gold"] = True
    actions["build_successor_recalibration_authority"] = True
    return {
        "schema_version": MANIFEST_SCHEMA,
        "status": "applied_gold_frozen_evaluation_not_revealed",
        "preview": dict(preview),
        "repository_authority": dict(repository),
        "authorized_actions": actions,
        "contains_private_gold": True,
        "contains_bounded_transcript_evidence": True,
        "will_perform_external_write": False,
    }


def _receipt(
    preview: Mapping[str, Any], gold_id: str, manifest_sha256: str
) -> dict[str, Any]:
    portable = portable_gold_projection(preview)
    actions = dict(portable["action_vector"])
    actions["freeze_gold"] = True
    actions["build_successor_recalibration_authority"] = True
    return {
        **portable,
        "schema_version": RECEIPT_SCHEMA,
        "status": "applied_gold_frozen_evaluation_not_revealed",
        "gold_id": gold_id,
        "manifest_sha256": manifest_sha256,
        "action_vector": actions,
        "mode": "0600",
    }


def apply_generation3_gold(
    reviewed_preview: Mapping[str, Any], *, expected_preview_content_sha256: str,
    runtime_root: Path = DEFAULT_RUNTIME_ROOT, **preview_inputs: Any,
) -> dict[str, Any]:
    """Freeze independently reviewed gold without revealing evaluation."""
    preview = _evaluate(**preview_inputs)
    if (
        dict(reviewed_preview) != preview
        or preview["content_sha256"] != expected_preview_content_sha256
    ):
        raise Generation3GoldError("Reviewed Generation-3 gold preview is stale.")
    existing = _existing_manifest(runtime_root)
    if existing is not None:
        return replay_generation3_gold(
            existing, runtime_root=runtime_root, **preview_inputs
        )
    repository = _repository_authority()
    core = _manifest_core(preview, repository)
    content_sha = _canonical_hash(core)
    gold_id = f"generation3-gold-{content_sha[:24]}"
    paths = _paths(runtime_root, gold_id)
    ensure_private_tree(paths["root"], paths["authority"])
    manifest = {**core, "gold_id": gold_id, "content_sha256": content_sha}
    write_immutable_private_json(paths["manifest"], manifest)
    receipt = _receipt(preview, gold_id, sha256_file(paths["manifest"]))
    write_immutable_private_json(paths["receipt"], receipt)
    return {
        **receipt,
        "private_manifest_path": str(paths["manifest"]),
        "private_receipt_path": str(paths["receipt"]),
        "idempotent_replay": False,
    }


def replay_generation3_gold(
    manifest_path: Path, *, runtime_root: Path = DEFAULT_RUNTIME_ROOT,
    **preview_inputs: Any,
) -> dict[str, Any]:
    """Replay exact private gold, repository, and negative action authority."""
    root = runtime_root.expanduser().absolute()
    path = manifest_path.expanduser().absolute()
    require_private_file(path, root)
    manifest = _read_object(path)
    preview = _evaluate(**preview_inputs)
    repository = _validate_repository_authority(manifest.get("repository_authority"))
    core = _manifest_core(preview, repository)
    content_sha = _canonical_hash(core)
    gold_id = f"generation3-gold-{content_sha[:24]}"
    expected = {**core, "gold_id": gold_id, "content_sha256": content_sha}
    if manifest != expected or path != _paths(root, gold_id)["manifest"]:
        raise Generation3GoldError("Generation-3 private gold manifest drifted.")
    receipt_path = _paths(root, gold_id)["receipt"]
    require_private_file(receipt_path, root)
    receipt = _read_object(receipt_path)
    expected_receipt = _receipt(preview, gold_id, sha256_file(path))
    if receipt != expected_receipt:
        raise Generation3GoldError("Generation-3 gold receipt drifted.")
    return {
        **receipt,
        "private_manifest_path": str(path),
        "private_receipt_path": str(receipt_path),
        "replay_schema_version": REPLAY_SCHEMA,
        "idempotent_replay": True,
    }
