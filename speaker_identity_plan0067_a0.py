"""Freeze Plan 0067 retained-output replay authority without model or live effects."""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
from pathlib import Path
from typing import Any, Mapping

import app_intelligence_ledger
import speaker_identity_preprocess
import speaker_identity_plan0066_a2 as plan0066_a2
from acoustic_audio_derivatives import (
    ensure_private_tree,
    read_private_object,
    require_private_file,
    sha256_file,
    write_immutable_private_json,
)


SCHEMA_VERSION = "transcribe-audio.plan0067-a0-activation.v1"
RECEIPT_SCHEMA_VERSION = "transcribe-audio.plan0067-a0-receipt.v1"
PLAN_ACTIVATION_COMMIT = "1601717"
DEFAULT_RUNTIME_ROOT = Path("~/.local/state/transcribe-audio/plan-0067")
DEFAULT_PLAN0066_ROOT = Path("~/.local/state/transcribe-audio/plan-0066")
DEFAULT_SOURCE_STATE_ROOT = Path("~/.local/state/transcribe-audio")
DEFAULT_GOLD_PATH = Path(
    "~/.local/state/transcribe-audio/plan-0064/"
    "p4-submission-6df988b11c152b78f9da59ab/submitted-decisions.json"
)
CALENDAR_ID_RE = re.compile(r"calendar-[a-z-]+-[a-f0-9]+")
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


class Plan0067A0Error(ValueError):
    """Raised when retained Plan 0066 replay authority is incomplete or drifts."""


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _hash(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _content(value: Mapping[str, Any]) -> dict[str, Any]:
    core = dict(value)
    core.pop("content_sha256", None)
    return {**core, "content_sha256": _hash(core)}


def _validate_content(value: Mapping[str, Any], label: str) -> None:
    core = {key: item for key, item in value.items() if key != "content_sha256"}
    if value.get("content_sha256") != _hash(core):
        raise Plan0067A0Error(f"{label} content hash drifted.")


def _git(*args: str) -> str:
    result = subprocess.run(
        ["git", *args], check=False, capture_output=True, text=True
    )
    if result.returncode:
        raise Plan0067A0Error(result.stderr.strip() or "Git authority read failed.")
    return result.stdout.strip()


def _one(paths: list[Path], label: str) -> Path:
    if len(paths) != 1:
        raise Plan0067A0Error(f"Expected one {label}; found {len(paths)}.")
    return paths[0]


def _artifact_binding(path: Path, root: Path) -> dict[str, str]:
    resolved = path.expanduser().resolve()
    require_private_file(resolved, root.expanduser().resolve())
    return {"path": str(resolved), "file_sha256": sha256_file(resolved)}


def _require_bound_input(path: Path, root: Path) -> Path:
    """Require an existing contained regular input without mutating legacy modes."""

    expanded = path.expanduser().absolute()
    if expanded.is_symlink() or not expanded.is_file():
        raise Plan0067A0Error(f"Bound input is not a regular file: {expanded}.")
    resolved = expanded.resolve()
    try:
        resolved.relative_to(root.expanduser().resolve())
    except ValueError as exc:
        raise Plan0067A0Error(f"Bound input is outside its authority root: {resolved}.") from exc
    return resolved


def _calendar_citations(value: Any) -> list[str]:
    result: list[str] = []
    if isinstance(value, Mapping):
        evidence_ids = value.get("evidence_ids")
        if isinstance(evidence_ids, list):
            result.extend(
                str(item) for item in evidence_ids if str(item).startswith("calendar-")
            )
        for item in value.values():
            result.extend(_calendar_citations(item))
    elif isinstance(value, list):
        for item in value:
            result.extend(_calendar_citations(item))
    return result


def build_case_binding(
    *,
    document_id: str,
    a1_case: Mapping[str, Any],
    prepared: Mapping[str, Any],
    expected_packet: Mapping[str, Any],
    failed_case: Mapping[str, Any],
    status: Mapping[str, Any],
    calendar_evidence: list[dict[str, Any]],
) -> dict[str, Any]:
    """Validate one retained packet/output pair against explicit calendar evidence."""

    if prepared.get("document_id") != document_id or failed_case.get("document_id") != document_id:
        raise Plan0067A0Error("Retained case document binding drifted.")
    if prepared.get("run_id") != status.get("run_id") or status.get("completed") is not True:
        raise Plan0067A0Error("Retained model status is incomplete or bound to another run.")
    if prepared.get("packet") != expected_packet:
        raise Plan0067A0Error("Plan 0066 A1-to-A2 packet transformation drifted.")
    if prepared.get("packet_sha256") != _hash(prepared.get("packet")):
        raise Plan0067A0Error("Retained packet hash drifted.")
    filename = str(prepared.get("original_recording_filename") or "")
    if not filename or filename != str(a1_case.get("original_recording_filename") or ""):
        raise Plan0067A0Error("Original recording filename binding drifted.")

    output_text = str(status.get("output_text") or "")
    if not output_text:
        raise Plan0067A0Error("Retained model output is empty.")
    readout = app_intelligence_ledger.extract_json_object(output_text)
    if readout.get("evaluation_id") != prepared.get("packet", {}).get("evaluation_id"):
        raise Plan0067A0Error("Retained model output references another evaluation.")

    catalog_ids = {
        str(item.get("evidence_id") or "")
        for item in calendar_evidence
        if isinstance(item, Mapping) and item.get("evidence_id")
    }
    cited_ids = _calendar_citations(readout)
    unknown = set(cited_ids) - catalog_ids
    if unknown:
        raise Plan0067A0Error(
            f"Retained output cites calendar evidence outside the host catalog: {sorted(unknown)}."
        )
    failure_ids = CALENDAR_ID_RE.findall(str(failed_case.get("reason") or ""))
    if not failure_ids or set(failure_ids) - catalog_ids:
        raise Plan0067A0Error("Plan 0066 rejected ID is not in the host calendar catalog.")

    return {
        "document_id": document_id,
        "run_id": str(prepared["run_id"]),
        "original_recording_filename": filename,
        "packet_content_sha256": str(prepared["packet_sha256"]),
        "output_text_sha256": hashlib.sha256(output_text.encode("utf-8")).hexdigest(),
        "codex_thread_id": str(status.get("codex_thread_id") or ""),
        "codex_turn_id": str(status.get("codex_turn_id") or ""),
        "calendar_evidence": calendar_evidence,
        "calendar_evidence_sha256": _hash(calendar_evidence),
        "retained_calendar_citation_count": len(cited_ids),
        "retained_calendar_evidence_ids": sorted(set(cited_ids)),
        "plan0066_rejected_calendar_evidence_ids": sorted(set(failure_ids)),
    }


def _repository_authority() -> dict[str, Any]:
    root = Path(_git("rev-parse", "--show-toplevel")).resolve()
    head = _git("rev-parse", "HEAD")
    upstream = _git("rev-parse", "@{upstream}")
    if head != upstream or _git("status", "--porcelain=v1"):
        raise Plan0067A0Error("A0 requires a clean, upstream-even repository.")
    ancestor = subprocess.run(
        ["git", "merge-base", "--is-ancestor", PLAN_ACTIVATION_COMMIT, head],
        check=False,
    )
    if ancestor.returncode:
        raise Plan0067A0Error("Plan 0067 activation commit is not in repository history.")
    source_bindings = []
    for relative in (
        "speaker_identity_preprocess.py",
        "speaker_identity_plan0066_a0.py",
        "speaker_identity_plan0066_a1.py",
        "speaker_identity_plan0066_a2.py",
        "speaker_identity_plan0067_a0.py",
    ):
        path = root / relative
        source_bindings.append(
            {
                "path": relative,
                "file_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "last_commit": _git("log", "-1", "--format=%H", "--", relative),
            }
        )
    return {
        "plan_activation_commit": PLAN_ACTIVATION_COMMIT,
        "freeze_commit": head,
        "upstream_commit": upstream,
        "source_bindings": source_bindings,
    }


def build_activation_manifest(
    *,
    plan0066_root: Path,
    gold_path: Path,
) -> dict[str, Any]:
    """Build the six-case immutable authority manifest from retained artifacts."""

    source_root = plan0066_root.expanduser().resolve()
    terminal_path = _one(list(source_root.glob("terminal-*/terminal.json")), "Plan 0066 terminal")
    a0_manifest_path = _one(list(source_root.glob("a0-*/private-manifest.json")), "Plan 0066 A0 manifest")
    terminal = read_private_object(terminal_path)
    plan66_a0 = read_private_object(a0_manifest_path)
    _validate_content(terminal, "Plan 0066 terminal")
    _validate_content(plan66_a0, "Plan 0066 A0 manifest")
    if terminal.get("status") != "plan0066_closed_withhold":
        raise Plan0067A0Error("Plan 0066 terminal disposition drifted.")

    gold_resolved = gold_path.expanduser().resolve()
    require_private_file(gold_resolved, gold_resolved.parent)
    gold = read_private_object(gold_resolved)
    if gold.get("authority_content_sha256") != "6df988b11c152b78f9da59ab6d2324516082196d70d0340ecba2298051582f67":
        raise Plan0067A0Error("Human-gold authority drifted.")

    bindings = {
        str(item.get("document_id") or ""): item
        for item in plan66_a0.get("document_bindings") or []
        if isinstance(item, Mapping)
    }
    cases: list[dict[str, Any]] = []
    for prepared_path in sorted((source_root / "a2/prepared").glob("*.json")):
        document_id = prepared_path.stem
        a1_path = source_root / "a1/cases" / f"{document_id}.json"
        failed_path = source_root / "a2/cases" / f"{document_id}.json"
        prepared = read_private_object(prepared_path)
        a1_case = read_private_object(a1_path)
        failed_case = read_private_object(failed_path)
        binding = bindings.get(document_id)
        if not binding:
            raise Plan0067A0Error(f"Missing Plan 0066 source binding: {document_id}.")
        transcript_path = _require_bound_input(
            Path(str(binding.get("stored_path") or "")),
            Path("~/.transcripts/artifacts"),
        )
        if sha256_file(transcript_path) != binding.get("stored_sha256"):
            raise Plan0067A0Error(f"Stored transcript binding drifted: {document_id}.")
        transcript = json.loads(transcript_path.read_text(encoding="utf-8"))
        discovery_packet = speaker_identity_preprocess.build_clue_discovery_packet(
            transcript=transcript,
            source_contexts=a1_case.get("packet", {}).get("source_contexts") or (),
        )
        speaker_identity_preprocess.validate_clue_discovery_readout(
            discovery_packet,
            a1_case.get("packet", {}).get("discovery_readout") or {},
        )
        status_path = _one(
            list(
                (
                    source_root
                    / "a2/state/app-intelligence-runs"
                    / str(prepared.get("run_id") or "")
                    / "artifacts/model-turn-readouts"
                ).glob("*.status.json")
            ),
            f"retained status for {document_id}",
        )
        status = read_private_object(status_path)
        prior_run_id = plan0066_a2.PRIOR_IDENTITY_RUNS[document_id]
        prior_packet_path = (
            app_intelligence_ledger.run_dir(
                DEFAULT_SOURCE_STATE_ROOT.expanduser().resolve(), prior_run_id
            )
            / "artifacts/speaker-preprocessing/identity_evaluation.input.json"
        )
        prior_packet_path = _require_bound_input(
            prior_packet_path,
            DEFAULT_SOURCE_STATE_ROOT,
        )
        prior_packet = json.loads(prior_packet_path.read_text(encoding="utf-8"))
        expected_packet = plan0066_a2.build_a2_packet(
            a1_case["packet"],
            prior_packet,
        )
        case = build_case_binding(
            document_id=document_id,
            a1_case=a1_case,
            prepared=prepared,
            expected_packet=expected_packet,
            failed_case=failed_case,
            status=status,
            calendar_evidence=discovery_packet["calendar_evidence"],
        )
        case.update(
            {
                "a1_case": _artifact_binding(a1_path, source_root),
                "prepared_case": _artifact_binding(prepared_path, source_root),
                "failed_case": _artifact_binding(failed_path, source_root),
                "status_artifact": _artifact_binding(status_path, source_root),
                "prior_packet_artifact": {
                    "path": str(prior_packet_path),
                    "file_sha256": sha256_file(prior_packet_path),
                },
                "transcript_artifact": {
                    "path": str(transcript_path),
                    "file_sha256": sha256_file(transcript_path),
                },
            }
        )
        cases.append(case)

    rejected_count = sum(
        len(case["plan0066_rejected_calendar_evidence_ids"]) for case in cases
    )
    if len(cases) != 6 or rejected_count != 7:
        raise Plan0067A0Error("Plan 0067 requires six cases and seven rejected calendar IDs.")
    artifact_bindings = {
        "plan0066_terminal": _artifact_binding(terminal_path, source_root),
        "plan0066_a0_manifest": _artifact_binding(a0_manifest_path, source_root),
        "plan0066_a0_receipt": _artifact_binding(a0_manifest_path.parent / "receipt.json", source_root),
        "plan0066_a1_manifest": _artifact_binding(source_root / "a1/private-manifest.json", source_root),
        "plan0066_a1_receipt": _artifact_binding(source_root / "a1/receipt.json", source_root),
        "plan0066_a2_manifest": _artifact_binding(source_root / "a2/private-manifest.json", source_root),
        "plan0066_a2_receipt": _artifact_binding(source_root / "a2/receipt.json", source_root),
        "human_gold": {"path": str(gold_resolved), "file_sha256": sha256_file(gold_resolved)},
    }
    return _content(
        {
            "schema_version": SCHEMA_VERSION,
            "status": "a0_retained_output_authority_frozen_zero_effect",
            "repository_authority": _repository_authority(),
            "plan0066_terminal_content_sha256": terminal["content_sha256"],
            "human_gold_authority_content_sha256": gold["authority_content_sha256"],
            "artifact_bindings": artifact_bindings,
            "cases": cases,
            "case_count": len(cases),
            "rejected_calendar_evidence_id_count": rejected_count,
            "model_turn_count": 0,
            "reference_repair_count": 0,
            "fresh_retrieval_count": 0,
            "effect_counts": dict(EFFECT_COUNTS),
        }
    )


def _validate_artifact_bindings(manifest: Mapping[str, Any], runtime_root: Path) -> None:
    for binding in manifest.get("artifact_bindings", {}).values():
        path = Path(str(binding["path"])).expanduser().resolve()
        if sha256_file(path) != binding["file_sha256"]:
            raise Plan0067A0Error(f"Frozen artifact drifted: {path}.")
    for case in manifest.get("cases") or []:
        for key in (
            "a1_case",
            "prepared_case",
            "failed_case",
            "status_artifact",
            "prior_packet_artifact",
            "transcript_artifact",
        ):
            binding = case[key]
            path = Path(str(binding["path"])).expanduser().resolve()
            if sha256_file(path) != binding["file_sha256"]:
                raise Plan0067A0Error(f"Frozen case artifact drifted: {path}.")
    if manifest.get("effect_counts") != EFFECT_COUNTS:
        raise Plan0067A0Error("A0 effect budget drifted.")
    ensure_private_tree(runtime_root)


def replay_activation(*, runtime_root: Path = DEFAULT_RUNTIME_ROOT) -> dict[str, Any]:
    root = runtime_root.expanduser().resolve()
    manifest_path = _one(list(root.glob("a0-*/private-manifest.json")), "Plan 0067 A0 manifest")
    receipt_path = manifest_path.parent / "receipt.json"
    manifest = read_private_object(manifest_path)
    receipt = read_private_object(receipt_path)
    _validate_content(manifest, "Plan 0067 A0 manifest")
    _validate_content(receipt, "Plan 0067 A0 receipt")
    if receipt.get("activation_content_sha256") != manifest["content_sha256"]:
        raise Plan0067A0Error("A0 receipt lost its manifest content binding.")
    if receipt.get("activation_file_sha256") != sha256_file(manifest_path):
        raise Plan0067A0Error("A0 receipt lost its manifest file binding.")
    _validate_artifact_bindings(manifest, root)
    return {**receipt, "manifest_path": str(manifest_path), "idempotent_replay": True}


def freeze_activation(
    *,
    runtime_root: Path = DEFAULT_RUNTIME_ROOT,
    plan0066_root: Path = DEFAULT_PLAN0066_ROOT,
    gold_path: Path = DEFAULT_GOLD_PATH,
) -> dict[str, Any]:
    root = runtime_root.expanduser().resolve()
    if list(root.glob("a0-*/receipt.json")):
        return replay_activation(runtime_root=root)
    manifest = build_activation_manifest(
        plan0066_root=plan0066_root,
        gold_path=gold_path,
    )
    run_root = root / f"a0-{manifest['content_sha256'][:24]}"
    manifest_path = run_root / "private-manifest.json"
    receipt_path = run_root / "receipt.json"
    ensure_private_tree(root, run_root)
    write_immutable_private_json(manifest_path, manifest)
    receipt = _content(
        {
            "schema_version": RECEIPT_SCHEMA_VERSION,
            "status": "a0_frozen_zero_effect",
            "activation_content_sha256": manifest["content_sha256"],
            "activation_file_sha256": sha256_file(manifest_path),
            "case_count": 6,
            "rejected_calendar_evidence_id_count": 7,
            "model_turn_count": 0,
            "effect_counts": dict(EFFECT_COUNTS),
        }
    )
    write_immutable_private_json(receipt_path, receipt)
    return {**receipt, "manifest_path": str(manifest_path), "idempotent_replay": False}


if __name__ == "__main__":
    print(json.dumps(freeze_activation(), indent=2, sort_keys=True))
