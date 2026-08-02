"""Plan 0049 training-only P1/P2 preparation in an isolated worker process."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Optional


SHARED_DURATION_TOLERANCE_SECONDS = 0.05
TRAINING_DURATION_TOLERANCE_SECONDS = 0.1
RESULT_PREFIX = "TRAINING_PREPARATION_RESULT="
PATH_ARGUMENTS = {"source_audio", "runtime_root", "p1_runtime_root"}


class TrainingPreparationError(ValueError):
    """Raised when isolated training preparation cannot remain bounded."""


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value.expanduser().absolute())
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _run_worker(operation: str, arguments: Mapping[str, Any]) -> dict[str, Any]:
    request = json.dumps(
        {"operation": operation, "arguments": _json_safe(arguments)},
        ensure_ascii=False,
        separators=(",", ":"),
    )
    result = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), "--worker"],
        input=request,
        text=True,
        capture_output=True,
        check=False,
        cwd=Path(__file__).resolve().parent,
    )
    payload_line = next(
        (
            line.removeprefix(RESULT_PREFIX)
            for line in reversed(result.stdout.splitlines())
            if line.startswith(RESULT_PREFIX)
        ),
        "",
    )
    try:
        payload = json.loads(payload_line)
    except json.JSONDecodeError as exc:
        raise TrainingPreparationError(
            "Training preparation worker returned no valid result."
        ) from exc
    if result.returncode != 0 or payload.get("ok") is not True:
        error_type = str(payload.get("error_type") or "worker_error")
        message = str(payload.get("message") or "training preparation failed")
        raise TrainingPreparationError(f"{error_type}: {message}")
    value = payload.get("result")
    if not isinstance(value, dict):
        raise TrainingPreparationError("Training preparation result is invalid.")
    return value


def dry_run(source_audio: Path, **kwargs: Any) -> dict[str, Any]:
    return _run_worker("p1_dry_run", {"source_audio": source_audio, **kwargs})


def apply_derivative(source_audio: Path, **kwargs: Any) -> dict[str, Any]:
    return _run_worker("p1_apply", {"source_audio": source_audio, **kwargs})


def replay_derivative(run_id: str, **kwargs: Any) -> dict[str, Any]:
    return _run_worker("p1_replay", {"run_id": run_id, **kwargs})


def speech_dry_run(
    p1_run_id: str,
    *,
    p1_runtime_root: Optional[Path] = None,
    runtime_root: Optional[Path] = None,
    intended_split: str = "development",
    readiness: Optional[Mapping[str, Mapping[str, Any]]] = None,
    test_mode: bool = False,
) -> dict[str, Any]:
    return _run_worker(
        "p2_dry_run",
        {
            "p1_run_id": p1_run_id,
            "p1_runtime_root": p1_runtime_root,
            "runtime_root": runtime_root,
            "intended_split": intended_split,
            "readiness": readiness,
            "test_mode": test_mode,
        },
    )


def apply_comparison(
    p1_run_id: str,
    *,
    p1_runtime_root: Optional[Path] = None,
    runtime_root: Optional[Path] = None,
    intended_split: str = "development",
    readiness: Optional[Mapping[str, Mapping[str, Any]]] = None,
    test_mode: bool = False,
) -> dict[str, Any]:
    return _run_worker(
        "p2_apply",
        {
            "p1_run_id": p1_run_id,
            "p1_runtime_root": p1_runtime_root,
            "runtime_root": runtime_root,
            "intended_split": intended_split,
            "readiness": readiness,
            "test_mode": test_mode,
        },
    )


def replay_comparison(
    run_id: str,
    *,
    runtime_root: Optional[Path] = None,
) -> dict[str, Any]:
    return _run_worker(
        "p2_replay", {"run_id": run_id, "runtime_root": runtime_root}
    )


def _worker_arguments(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TrainingPreparationError("Worker arguments must be an object.")
    arguments = dict(value)
    for key in PATH_ARGUMENTS & set(arguments):
        if arguments[key] is not None:
            arguments[key] = Path(str(arguments[key]))
    return arguments


def _execute_worker(request: Any) -> dict[str, Any]:
    if not isinstance(request, Mapping) or set(request) != {"operation", "arguments"}:
        raise TrainingPreparationError("Worker request shape is invalid.")
    operation = str(request["operation"])
    arguments = _worker_arguments(request["arguments"])

    import acoustic_audio_derivatives as p1
    import acoustic_speech_preparation as p2

    if p1.DURATION_TOLERANCE_SECONDS != SHARED_DURATION_TOLERANCE_SECONDS:
        raise TrainingPreparationError(
            "Shared P1 duration tolerance drifted from its frozen authority."
        )
    p1.DURATION_TOLERANCE_SECONDS = TRAINING_DURATION_TOLERANCE_SECONDS
    try:
        delay = os.environ.get("TRANSCRIBE_AUDIO_TRAINING_WORKER_TEST_DELAY", "")
        ready_path = os.environ.get(
            "TRANSCRIBE_AUDIO_TRAINING_WORKER_TEST_READY_PATH", ""
        )
        if os.environ.get("PYTEST_CURRENT_TEST"):
            if ready_path:
                Path(ready_path).touch()
            if delay:
                time.sleep(float(delay))
        dispatch = {
            "p1_dry_run": p1.dry_run,
            "p1_apply": p1.apply_derivative,
            "p1_replay": p1.replay_derivative,
            "p2_dry_run": p2.dry_run,
            "p2_apply": p2.apply_comparison,
            "p2_replay": p2.replay_comparison,
        }
        function = dispatch.get(operation)
        if function is None:
            raise TrainingPreparationError("Worker operation is invalid.")
        return function(**arguments)
    finally:
        p1.DURATION_TOLERANCE_SECONDS = SHARED_DURATION_TOLERANCE_SECONDS


def _worker_main() -> int:
    try:
        request = json.loads(sys.stdin.read())
        result = _execute_worker(request)
        payload = {"ok": True, "result": result}
        code = 0
    except Exception as exc:
        payload = {
            "ok": False,
            "error_type": type(exc).__name__,
            "message": str(exc),
        }
        code = 1
    print(
        RESULT_PREFIX
        + json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    )
    return code


if __name__ == "__main__":
    if sys.argv[1:] != ["--worker"]:
        raise SystemExit("This module accepts only --worker.")
    raise SystemExit(_worker_main())
