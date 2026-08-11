#!/usr/bin/env python3
"""Emit and replay the fail-safe Plan 0065 terminal after D2."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

from acoustic_audio_derivatives import (
    canonical_artifact_hash,
    ensure_private_tree,
    read_private_object,
    require_private_file,
    sha256_file,
    write_immutable_private_json,
)
import speaker_identity_plan0065_d0 as d0
import speaker_identity_plan0065_d1 as d1
import speaker_identity_plan0065_d2 as d2


SCHEMA = "transcribe-audio.plan0065-terminal.v1"
DEFAULT_RUNTIME_ROOT = d0.DEFAULT_RUNTIME_ROOT
D0_MANIFEST_SHA256 = d1.D0_MANIFEST_SHA256
D1_POLICY_SHA256 = d2.D1_POLICY_SHA256
D2_ACTIVATION_SHA256 = "ef76ba3392ca28a27c695e547765cf03ef2ea062d0d8bc67292549d182009959"
D2_RECEIPT_SHA256 = "8d65f6be10259cd54a8e1c8bb3112dcd7db4c9838ca70a89daadfda509e86ad7"


class Plan0065TerminalError(ValueError):
    """Raised when the fail-safe terminal lacks exact upstream authority."""


def _hash(value: Any) -> str:
    return canonical_artifact_hash(value)


def _content(value: Mapping[str, Any]) -> dict[str, Any]:
    core = dict(value)
    core.pop("content_sha256", None)
    return {**core, "content_sha256": _hash(core)}


def build_terminal(
    *,
    d0_receipt: Mapping[str, Any],
    d1_receipt: Mapping[str, Any],
    d2_receipt: Mapping[str, Any],
) -> dict[str, Any]:
    if (
        d0_receipt.get("status") != "d0_frozen_zero_effect"
        or d1_receipt.get("status") != "d1_pass_zero_effect"
        or d1_receipt.get("development_gate", {}).get("passed") is not True
        or d2_receipt.get("content_sha256") != D2_RECEIPT_SHA256
        or d2_receipt.get("context_gate", {}).get("passed") is not False
        or d2_receipt.get("context_gate", {}).get("terminal_status")
        != "context_recovery_failed"
        or any((d2_receipt.get("effect_counts") or {}).values())
    ):
        raise Plan0065TerminalError("Plan 0065 terminal authority is incomplete.")
    return _content(
        {
            "schema_version": SCHEMA,
            "status": "plan0065_closed_withhold",
            "terminal_decision": "withhold",
            "reason_code": "context_recovery_failed",
            "d0_receipt_content_sha256": d0_receipt["content_sha256"],
            "d1_receipt_content_sha256": d1_receipt["content_sha256"],
            "d2_receipt_content_sha256": d2_receipt["content_sha256"],
            "packet_state": {
                "d0": "complete",
                "d1": "complete_pass",
                "d2": "complete_failed_nonvacuous_gate",
                "d3": "not_opened",
                "e0": "not_opened",
                "e1": "not_run",
                "e2": "not_published",
                "e3": "not_run",
            },
            "acceptance": {
                "acoustic_development_gate": True,
                "context_development_gate": False,
                "joined_residual_development_gate": False,
                "fresh_blind_evaluation": False,
                "ready_for_separate_local_acceptance_plan": False,
            },
            "execution_counts": {
                "local_biometric_half_probe_count": d1_receipt[
                    "execution_counts"
                ]["local_biometric_half_probe_count"],
                "primary_provider_model_turn_count": d2_receipt[
                    "execution_counts"
                ]["primary_model_turn_count"],
                "fallback_provider_model_turn_count": d2_receipt[
                    "execution_counts"
                ]["fallback_model_turn_count"],
                "fresh_evaluation_run_count": 0,
            },
            "effect_counts": dict(d2_receipt["effect_counts"]),
        }
    )


def _paths(runtime_root: Path) -> dict[str, Path]:
    root = runtime_root.expanduser().absolute()
    run = root / f"terminal-d2-{D2_RECEIPT_SHA256[:24]}"
    return {"root": root, "run": run, "terminal": run / "terminal.json"}


def close_plan(*, runtime_root: Path = DEFAULT_RUNTIME_ROOT) -> dict[str, Any]:
    d0_receipt = d0.replay_d0(
        manifest_content_sha256=D0_MANIFEST_SHA256,
        runtime_root=runtime_root,
    )
    d1_receipt = d1.replay_d1(
        policy_content_sha256=D1_POLICY_SHA256,
        runtime_root=runtime_root,
    )
    d2_receipt = d2.replay_d2(D2_ACTIVATION_SHA256, runtime_root=runtime_root)
    terminal = build_terminal(
        d0_receipt=d0_receipt,
        d1_receipt=d1_receipt,
        d2_receipt=d2_receipt,
    )
    paths = _paths(runtime_root)
    if paths["terminal"].exists():
        return replay_terminal(runtime_root=runtime_root)
    ensure_private_tree(paths["root"], paths["run"])
    write_immutable_private_json(paths["terminal"], terminal)
    return {
        **terminal,
        "terminal_file_sha256": sha256_file(paths["terminal"]),
        "private_terminal_path": str(paths["terminal"]),
        "idempotent_replay": False,
    }


def replay_terminal(*, runtime_root: Path = DEFAULT_RUNTIME_ROOT) -> dict[str, Any]:
    paths = _paths(runtime_root)
    require_private_file(paths["terminal"], paths["root"])
    terminal = read_private_object(paths["terminal"])
    core = {key: value for key, value in terminal.items() if key != "content_sha256"}
    if terminal.get("content_sha256") != _hash(core):
        raise Plan0065TerminalError("Plan 0065 terminal drifted.")
    d2_receipt = d2.replay_d2(D2_ACTIVATION_SHA256, runtime_root=runtime_root)
    if terminal.get("d2_receipt_content_sha256") != d2_receipt.get("content_sha256"):
        raise Plan0065TerminalError("Plan 0065 terminal lost its D2 binding.")
    return {
        **terminal,
        "terminal_file_sha256": sha256_file(paths["terminal"]),
        "private_terminal_path": str(paths["terminal"]),
        "idempotent_replay": True,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("close", "replay"))
    args = parser.parse_args()
    result = close_plan() if args.mode == "close" else replay_terminal()
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
