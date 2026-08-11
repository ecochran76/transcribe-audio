"""Plan 0066 terminal close and exact replay."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import speaker_identity_plan0066_a0 as a0
import speaker_identity_plan0066_a1 as a1
import speaker_identity_plan0066_a2 as a2
from acoustic_audio_derivatives import (
    ensure_private_tree,
    read_private_object,
    require_private_file,
    sha256_file,
    write_immutable_private_json,
)


SCHEMA_VERSION = "transcribe-audio.plan0066-terminal.v1"
DEFAULT_RUNTIME_ROOT = Path("~/.local/state/transcribe-audio/plan-0066")


class Plan0066TerminalError(ValueError):
    """Raised when Plan 0066 cannot close or replay exactly."""


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _hash(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def build_terminal(
    *,
    a0_receipt: Mapping[str, Any],
    a1_receipt: Mapping[str, Any],
    a2_receipt: Mapping[str, Any],
) -> dict[str, Any]:
    measurement = dict(a2_receipt.get("measurement") or {})
    if a0_receipt.get("status") != "a0_frozen_zero_effect":
        raise Plan0066TerminalError("A0 is not terminal-ready.")
    if a1_receipt.get("status") != "a1_passed_zero_source_mutation":
        raise Plan0066TerminalError("A1 source/roster gate did not pass.")
    if measurement.get("passed") is True:
        status = "plan0066_closed_context_candidate_recovered"
        reason_code = "context_candidate_recovered"
    else:
        status = "plan0066_closed_withhold"
        reason_code = (
            "evidence_reference_compliance_failed"
            if int(measurement.get("validation_failure_count") or 0) > 0
            else "context_candidate_not_recovered"
        )
    terminal = {
        "schema_version": SCHEMA_VERSION,
        "status": status,
        "decision": "pass" if measurement.get("passed") else "withhold",
        "reason_code": reason_code,
        "a0_activation_content_sha256": a0_receipt["activation_content_sha256"],
        "a1_manifest_content_sha256": a1_receipt["manifest_content_sha256"],
        "a2_manifest_content_sha256": a2_receipt["manifest_content_sha256"],
        "measurement": measurement,
        "execution_counts": dict(a2_receipt["execution_counts"]),
        "source_store_index_change_count": int(
            a2_receipt["source_store_index_change_count"]
        ),
        "reviewed_person_count": int(a1_receipt["reviewed_person_count"]),
        "joined_or_residual_gate_opened": False,
        "fresh_evaluation_opened": False,
        "will_apply_assignments": False,
        "effect_counts": dict(a0.EFFECT_COUNTS),
    }
    if (
        terminal["source_store_index_change_count"] != 0
        or any(terminal["effect_counts"].values())
        or terminal["execution_counts"]
        != {"primary_model_turns": 6, "fallback_model_turns": 0, "retries": 0}
    ):
        raise Plan0066TerminalError("Plan 0066 terminal effect or budget drifted.")
    terminal["content_sha256"] = _hash(terminal)
    return terminal


def close_plan(
    *, runtime_root: Path = DEFAULT_RUNTIME_ROOT
) -> dict[str, Any]:
    root = runtime_root.expanduser().resolve()
    a0_receipt = a0.freeze_activation(runtime_root=root)
    a1_receipt = a1.replay_a1(runtime_root=root)
    a2_receipt = a2.replay_a2(runtime_root=root)
    terminal = build_terminal(
        a0_receipt=a0_receipt,
        a1_receipt=a1_receipt,
        a2_receipt=a2_receipt,
    )
    terminal_root = root / f"terminal-{terminal['content_sha256'][:24]}"
    terminal_path = terminal_root / "terminal.json"
    if terminal_path.exists():
        return replay_terminal(runtime_root=root)
    ensure_private_tree(root, terminal_root)
    write_immutable_private_json(terminal_path, terminal)
    return {
        **terminal,
        "terminal_path": str(terminal_path),
        "terminal_file_sha256": sha256_file(terminal_path),
        "idempotent_replay": False,
    }


def replay_terminal(
    *, runtime_root: Path = DEFAULT_RUNTIME_ROOT
) -> dict[str, Any]:
    root = runtime_root.expanduser().resolve()
    paths = sorted(root.glob("terminal-*/terminal.json"))
    if len(paths) != 1:
        raise Plan0066TerminalError("Plan 0066 requires exactly one terminal.")
    require_private_file(paths[0], root)
    terminal = read_private_object(paths[0])
    core = {key: value for key, value in terminal.items() if key != "content_sha256"}
    if terminal.get("content_sha256") != _hash(core):
        raise Plan0066TerminalError("Plan 0066 terminal content drifted.")
    expected = build_terminal(
        a0_receipt=a0.freeze_activation(runtime_root=root),
        a1_receipt=a1.replay_a1(runtime_root=root),
        a2_receipt=a2.replay_a2(runtime_root=root),
    )
    if terminal != expected:
        raise Plan0066TerminalError("Plan 0066 terminal lost its packet binding.")
    return {
        **terminal,
        "terminal_path": str(paths[0]),
        "terminal_file_sha256": sha256_file(paths[0]),
        "idempotent_replay": True,
    }


if __name__ == "__main__":
    print(json.dumps(close_plan(), indent=2, sort_keys=True))
