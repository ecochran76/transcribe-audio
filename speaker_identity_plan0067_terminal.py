"""Close Plan 0067 after its bounded A0 authority attempts are exhausted."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any, Mapping

import speaker_identity_plan0067_a0 as a0
from acoustic_audio_derivatives import (
    ensure_private_tree,
    read_private_object,
    sha256_file,
    write_immutable_private_json,
)


SCHEMA_VERSION = "transcribe-audio.plan0067-terminal.v1"
DEFAULT_RUNTIME_ROOT = a0.DEFAULT_RUNTIME_ROOT
FAILURE_REASONS = (
    "legacy_transcript_mode_not_0600",
    "a1_a2_transformation_not_byte_equal",
    "legacy_status_mode_not_0600",
)


class Plan0067TerminalError(ValueError):
    """Raised when the bounded Plan 0067 terminal cannot replay exactly."""


def _hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _content(value: Mapping[str, Any]) -> dict[str, Any]:
    core = dict(value)
    core.pop("content_sha256", None)
    return {**core, "content_sha256": _hash(core)}


def _git(*args: str) -> str:
    result = subprocess.run(
        ["git", *args], check=False, capture_output=True, text=True
    )
    if result.returncode:
        raise Plan0067TerminalError(result.stderr.strip() or "Git read failed.")
    return result.stdout.strip()


def build_terminal(*, source_commit: str) -> dict[str, Any]:
    return _content(
        {
            "schema_version": SCHEMA_VERSION,
            "status": "plan0067_closed_withhold",
            "decision": "withhold",
            "reason_code": "a0_legacy_artifact_mode_contract_mismatch",
            "source_commit": source_commit,
            "a0_attempt_count": 3,
            "a0_artifact_written": False,
            "failure_reasons": list(FAILURE_REASONS),
            "product_contract_changed": False,
            "retained_output_replay_count": 0,
            "fresh_evaluation_opened": False,
            "will_apply_assignments": False,
            "effect_counts": dict(a0.EFFECT_COUNTS),
        }
    )


def close_plan(*, runtime_root: Path = DEFAULT_RUNTIME_ROOT) -> dict[str, Any]:
    root = runtime_root.expanduser().resolve()
    existing = list(root.glob("terminal-*/terminal.json"))
    if existing:
        return replay_terminal(runtime_root=root)
    if list(root.glob("a0-*/private-manifest.json")):
        raise Plan0067TerminalError("A0 artifact exists; failure terminal is inapplicable.")
    head = _git("rev-parse", "HEAD")
    if head != _git("rev-parse", "@{upstream}") or _git("status", "--porcelain=v1"):
        raise Plan0067TerminalError("Terminal requires clean, upstream-even source authority.")
    terminal = build_terminal(source_commit=head)
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
    paths = list(root.glob("terminal-*/terminal.json"))
    if len(paths) != 1:
        raise Plan0067TerminalError("Expected one Plan 0067 terminal.")
    terminal = read_private_object(paths[0])
    core = {key: value for key, value in terminal.items() if key != "content_sha256"}
    if terminal.get("content_sha256") != _hash(core):
        raise Plan0067TerminalError("Plan 0067 terminal content drifted.")
    if terminal.get("effect_counts") != a0.EFFECT_COUNTS:
        raise Plan0067TerminalError("Plan 0067 terminal effect counts drifted.")
    return {
        **terminal,
        "terminal_path": str(paths[0]),
        "terminal_file_sha256": sha256_file(paths[0]),
        "idempotent_replay": True,
    }


if __name__ == "__main__":
    print(json.dumps(close_plan(), indent=2, sort_keys=True))
