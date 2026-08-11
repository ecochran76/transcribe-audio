"""Close Plan 0070 after its bounded D0 authority attempts are exhausted."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any, Mapping

import speaker_identity_plan0070_d0 as d0
from acoustic_audio_derivatives import (
    ensure_private_tree,
    read_private_object,
    sha256_file,
    write_immutable_private_json,
)


SCHEMA_VERSION = "transcribe-audio.plan0070-terminal.v1"
DEFAULT_RUNTIME_ROOT = d0.DEFAULT_RUNTIME_ROOT
FAILURE_REASONS = (
    "plan0065_policy_hash_not_exposed_as_source_constant",
    "exposure_lists_compared_as_integer_counts",
)


class Plan0070TerminalError(ValueError):
    """Raised when the bounded Plan 0070 terminal cannot replay exactly."""


def _content(value: Mapping[str, Any]) -> dict[str, Any]:
    return d0._content(value)


def _validate_content(value: Mapping[str, Any], label: str) -> None:
    try:
        d0._validate_content(value, label)
    except d0.Plan0070D0Error as exc:
        raise Plan0070TerminalError(str(exc)) from exc


def _git(*args: str) -> str:
    result = subprocess.run(
        ["git", *args], check=False, capture_output=True, text=True
    )
    if result.returncode:
        raise Plan0070TerminalError(result.stderr.strip() or "Git read failed.")
    return result.stdout.strip()


def build_terminal(*, source_commit: str) -> dict[str, Any]:
    return _content(
        {
            "schema_version": SCHEMA_VERSION,
            "status": "plan0070_closed_withhold",
            "decision": "withhold",
            "reason_code": "d0_authority_harness_shape_mismatch",
            "source_commit": source_commit,
            "d0_attempt_count": 2,
            "d0_artifact_written": False,
            "failure_reasons": list(FAILURE_REASONS),
            "d3_counterfactual": dict(d0.EXPECTED_D3_START),
            "packet_state": {
                "d0": "attempts_exhausted_no_artifact",
                "d1": "not_opened",
                "d2": "not_opened",
                "d3": "not_opened",
                "e0": "not_opened",
                "e1": "not_run",
                "e2": "not_published",
                "e3": "not_run",
            },
            "supplemental_development_opened": False,
            "fresh_evaluation_opened": False,
            "will_apply_assignments": False,
            "effect_counts": dict(d0.EFFECT_COUNTS),
        }
    )


def close_plan(*, runtime_root: Path = DEFAULT_RUNTIME_ROOT) -> dict[str, Any]:
    root = runtime_root.expanduser().resolve()
    existing = list(root.glob("terminal-*/terminal.json"))
    if existing:
        return replay_terminal(runtime_root=root)
    if list(root.glob("d0-*/private-manifest.json")):
        raise Plan0070TerminalError("D0 artifact exists; failure terminal is inapplicable.")
    head = _git("rev-parse", "HEAD")
    if head != _git("rev-parse", "@{upstream}") or _git("status", "--porcelain=v1"):
        raise Plan0070TerminalError(
            "Terminal requires clean, upstream-even source authority."
        )
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
        raise Plan0070TerminalError("Expected one Plan 0070 terminal.")
    terminal = read_private_object(paths[0])
    _validate_content(terminal, "Plan 0070 terminal")
    if (
        terminal.get("d0_attempt_count") != 2
        or terminal.get("d0_artifact_written") is not False
        or terminal.get("d3_counterfactual") != d0.EXPECTED_D3_START
        or terminal.get("effect_counts") != d0.EFFECT_COUNTS
    ):
        raise Plan0070TerminalError("Plan 0070 terminal contract drifted.")
    return {
        **terminal,
        "terminal_path": str(paths[0]),
        "terminal_file_sha256": sha256_file(paths[0]),
        "idempotent_replay": True,
    }


if __name__ == "__main__":
    print(json.dumps(close_plan(), indent=2, sort_keys=True))
