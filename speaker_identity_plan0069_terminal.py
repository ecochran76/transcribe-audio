"""Close Plan 0069 from immutable grouped-normalization measurement."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any, Mapping

import speaker_identity_plan0069_a0 as a0
import speaker_identity_plan0069_a2 as a2
from acoustic_audio_derivatives import (
    ensure_private_tree,
    read_private_object,
    sha256_file,
    write_immutable_private_json,
)


SCHEMA_VERSION = "transcribe-audio.plan0069-terminal.v1"
DEFAULT_RUNTIME_ROOT = a0.DEFAULT_RUNTIME_ROOT
EFFECT_COUNTS = dict(a0.EFFECT_COUNTS)


class Plan0069TerminalError(ValueError):
    """Raised when the terminal cannot bind or replay exact A0/A2 evidence."""


def _content(value: Mapping[str, Any]) -> dict[str, Any]:
    return a0._content(value)


def _validate_content(value: Mapping[str, Any], label: str) -> None:
    try:
        a0._validate_content(value, label)
    except a0.Plan0069A0Error as exc:
        raise Plan0069TerminalError(str(exc)) from exc


def _git(*args: str) -> str:
    result = subprocess.run(
        ["git", *args], check=False, capture_output=True, text=True
    )
    if result.returncode:
        raise Plan0069TerminalError(result.stderr.strip() or "Git authority read failed.")
    return result.stdout.strip()


def build_terminal(
    *,
    a0_manifest: Mapping[str, Any],
    a2_manifest: Mapping[str, Any],
    source_commit: str,
) -> dict[str, Any]:
    measurement = dict(a2_manifest.get("measurement") or {})
    execution_counts = dict(a2_manifest.get("execution_counts") or {})
    if (
        a0_manifest.get("case_count") != 6
        or a0_manifest.get("original_recording_filename_count") != 6
        or execution_counts.get("retained_output_replays") != 6
    ):
        raise Plan0069TerminalError("Terminal requires all six filename-bearing outputs.")
    if execution_counts.get("primary_model_turns") != 0 or any(
        int(execution_counts.get(key) or 0)
        for key in (
            "fallback_model_turns",
            "retries",
            "model_reference_repairs",
            "fresh_retrievals",
            "fresh_evaluations",
        )
    ):
        raise Plan0069TerminalError("Terminal model/retry/repair/retrieval budget drifted.")
    expected_measurement = {
        "status": "context_candidate_recovered",
        "passed": True,
        "correct_prepared_candidate_count": 5,
        "wrong_prepared_candidate_count": 0,
        "abstained_slot_count": 17,
        "incomplete_candidate_provenance_count": 0,
        "unavailable_case_count": 0,
        "validation_failure_count": 0,
    }
    if measurement != expected_measurement:
        raise Plan0069TerminalError("A2 measurement differs from the accepted result.")
    if (
        a2_manifest.get("normalized_group_count") != 10
        or a2_manifest.get("expanded_utterance_assignment_count") != 28
        or a2_manifest.get("retained_output_change_count") != 0
        or a2_manifest.get("source_store_index_change_count") != 0
        or a2_manifest.get("effect_counts") != EFFECT_COUNTS
    ):
        raise Plan0069TerminalError("A2 normalization or effect bounds drifted.")
    return _content(
        {
            "schema_version": SCHEMA_VERSION,
            "status": "plan0069_closed_pass",
            "decision": "pass",
            "reason_code": "grouped_assignment_schema_reconciled",
            "source_commit": source_commit,
            "a0_activation_content_sha256": a0_manifest["content_sha256"],
            "a2_manifest_content_sha256": a2_manifest["content_sha256"],
            "measurement": measurement,
            "execution_counts": execution_counts,
            "validated_case_count": 6,
            "validation_failure_count": 0,
            "original_recording_filename_count": 6,
            "normalized_group_count": 10,
            "expanded_utterance_assignment_count": 28,
            "retained_output_change_count": 0,
            "source_store_index_change_count": 0,
            "joined_or_residual_gate_opened": False,
            "fresh_evaluation_opened": False,
            "will_apply_assignments": False,
            "effect_counts": dict(EFFECT_COUNTS),
        }
    )


def close_plan(*, runtime_root: Path = DEFAULT_RUNTIME_ROOT) -> dict[str, Any]:
    root = runtime_root.expanduser().resolve()
    existing = list(root.glob("terminal-*/terminal.json"))
    if existing:
        return replay_terminal(runtime_root=root)
    a0_receipt = a0.replay_activation(runtime_root=root)
    a2_receipt = a2.replay_a2(runtime_root=root)
    a0_manifest = read_private_object(Path(a0_receipt["manifest_path"]))
    a2_manifest = read_private_object(Path(a2_receipt["manifest_path"]))
    head = _git("rev-parse", "HEAD")
    if head != _git("rev-parse", "@{upstream}") or _git("status", "--porcelain=v1"):
        raise Plan0069TerminalError("Terminal requires clean, upstream-even source authority.")
    terminal = build_terminal(
        a0_manifest=a0_manifest,
        a2_manifest=a2_manifest,
        source_commit=head,
    )
    run_root = root / f"terminal-{terminal['content_sha256'][:24]}"
    terminal_path = run_root / "terminal.json"
    ensure_private_tree(root, run_root)
    write_immutable_private_json(terminal_path, terminal)
    return {
        **terminal,
        "terminal_path": str(terminal_path),
        "terminal_file_sha256": sha256_file(terminal_path),
        "idempotent_replay": False,
    }


def replay_terminal(*, runtime_root: Path = DEFAULT_RUNTIME_ROOT) -> dict[str, Any]:
    root = runtime_root.expanduser().resolve()
    a0.replay_activation(runtime_root=root)
    a2.replay_a2(runtime_root=root)
    paths = list(root.glob("terminal-*/terminal.json"))
    if len(paths) != 1:
        raise Plan0069TerminalError("Expected one Plan 0069 terminal.")
    terminal = read_private_object(paths[0])
    _validate_content(terminal, "Plan 0069 terminal")
    if terminal.get("effect_counts") != EFFECT_COUNTS:
        raise Plan0069TerminalError("Terminal effect counts drifted.")
    return {
        **terminal,
        "terminal_path": str(paths[0]),
        "terminal_file_sha256": sha256_file(paths[0]),
        "idempotent_replay": True,
    }


if __name__ == "__main__":
    print(json.dumps(close_plan(), indent=2, sort_keys=True))
