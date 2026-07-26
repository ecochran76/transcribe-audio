#!/usr/bin/env python3
"""Execute blind speaker-evaluation predictions through the local host API."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Iterable, Optional
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
from uuid import uuid4

import app_intelligence_ledger
import speaker_evaluation_campaign


DEFAULT_API_BASE_URL = "http://127.0.0.1:18876"


class CasePredictionFailure(ValueError):
    """A model result failed host validation for one case, not the harness."""

    def __init__(
        self,
        stage: str,
        message: str,
        *,
        run_references: Optional[dict[str, Any]] = None,
    ) -> None:
        super().__init__(message)
        self.stage = stage
        self.run_references = dict(run_references or {})


class LocalSpeakerCaseRunner:
    """Adapter from one campaign case to the existing two-phase local workflow."""

    def __init__(
        self,
        *,
        base_url: str = DEFAULT_API_BASE_URL,
        timeout_seconds: float = 600,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds

    def _post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        request = Request(
            f"{self.base_url}{path}",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urlopen(request, timeout=self.timeout_seconds) as response:
                result = json.loads(response.read())
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise ValueError(
                f"Local transcript API returned {exc.code} for {path}: {detail}"
            ) from exc
        except (URLError, TimeoutError) as exc:
            raise ValueError(f"Local transcript API failed for {path}: {exc}") from exc
        if not isinstance(result, dict):
            raise ValueError(f"Local transcript API returned a non-object for {path}.")
        return result

    def _execute_prepared(self, prepared: dict[str, Any]) -> dict[str, Any]:
        run_id = str(prepared.get("run_id") or "")
        prompt_packet = (
            prepared.get("prompt_packet")
            if isinstance(prepared.get("prompt_packet"), dict)
            else {}
        )
        packet_id = str(prompt_packet.get("packet_id") or "")
        if not run_id or not packet_id:
            raise ValueError("Prepared App Intelligence phase lacks run and packet IDs.")
        session = self._post(
            f"/api/intelligence/runs/{run_id}/session-start",
            {
                "transport": "stdio",
                "approval_token": (
                    app_intelligence_ledger.SESSION_START_APPROVAL_TOKEN
                ),
            },
        )
        if not session.get("ok"):
            raise ValueError(f"App Intelligence session start failed for {run_id}.")
        send = self._post(
            f"/api/intelligence/runs/{run_id}/prompt-packets/{packet_id}/send",
            {
                "approval_token": app_intelligence_ledger.MODEL_TURN_SEND_TOKEN,
                "timeout_seconds": self.timeout_seconds,
            },
        )
        if not send.get("ok"):
            raise ValueError(f"App Intelligence send failed for {run_id}.")
        status = self._post(
            f"/api/intelligence/runs/{run_id}/turn-status",
            {
                "thread_id": send.get("codex_thread_id") or "",
                "turn_id": send.get("codex_turn_id") or "",
                "approval_token": app_intelligence_ledger.MODEL_TURN_STATUS_TOKEN,
                "timeout_seconds": self.timeout_seconds,
            },
        )
        if not status.get("completed"):
            raise ValueError(f"App Intelligence turn did not complete for {run_id}.")
        return status

    @staticmethod
    def _captured_json(status: dict[str, Any]) -> dict[str, Any]:
        return app_intelligence_ledger.extract_json_object(
            str(status.get("output_text") or "")
        )

    def _execute_reference_repair(
        self,
        document_id: str,
        *,
        phase: str,
        original: dict[str, Any],
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        repair = self._post(
            (
                f"/api/conversations/{document_id}/speaker-preprocessing/"
                "prepare-reference-repair"
            ),
            {
                "phase": phase,
                "original_run_id": original["run_id"],
                "route": original.get("route") or {},
            },
        )
        status = self._execute_prepared(repair)
        return repair, self._captured_json(status)

    def __call__(self, document_id: str) -> dict[str, Any]:
        discovery = self._post(
            f"/api/conversations/{document_id}/speaker-preprocessing/prepare-discovery",
            {},
        )
        self._execute_prepared(discovery)
        clue_run_references = {
            "clue_discovery_run_id": discovery["run_id"],
        }
        try:
            evaluation = self._post(
                f"/api/conversations/{document_id}/speaker-preprocessing/prepare-evaluation",
                {"clue_discovery_run_id": discovery["run_id"]},
            )
        except ValueError as original_exc:
            try:
                repair, corrected_readout = self._execute_reference_repair(
                    document_id,
                    phase="clue_discovery",
                    original=discovery,
                )
                clue_run_references["clue_discovery_repair_run_id"] = repair[
                    "run_id"
                ]
                evaluation = self._post(
                    (
                        f"/api/conversations/{document_id}/speaker-preprocessing/"
                        "prepare-evaluation"
                    ),
                    {
                        "clue_discovery_run_id": discovery["run_id"],
                        "discovery_readout": corrected_readout,
                    },
                )
            except ValueError as repair_exc:
                raise CasePredictionFailure(
                    "clue_discovery_validation",
                    str(repair_exc),
                    run_references=clue_run_references,
                ) from original_exc
        self._execute_prepared(evaluation)
        run_references = {
            **clue_run_references,
            "identity_evaluation_run_id": evaluation["run_id"],
        }
        try:
            persisted = self._post(
                f"/api/conversations/{document_id}/speaker-preprocessing/capture-evaluation",
                run_references,
            )
        except ValueError as original_exc:
            try:
                repair, corrected_readout = self._execute_reference_repair(
                    document_id,
                    phase="identity_evaluation",
                    original=evaluation,
                )
                run_references["identity_evaluation_repair_run_id"] = repair[
                    "run_id"
                ]
                persisted = self._post(
                    (
                        f"/api/conversations/{document_id}/speaker-preprocessing/"
                        "capture-evaluation"
                    ),
                    {**run_references, "readout": corrected_readout},
                )
            except ValueError as repair_exc:
                raise CasePredictionFailure(
                    "identity_evaluation_validation",
                    str(repair_exc),
                    run_references=run_references,
                ) from original_exc
        record = (
            persisted.get("record")
            if isinstance(persisted.get("record"), dict)
            else {}
        )
        current_id = str(record.get("current_evaluation_id") or "")
        prediction = next(
            (
                item
                for item in record.get("evaluations") or []
                if isinstance(item, dict)
                and str(item.get("evaluation_id") or "") == current_id
            ),
            None,
        )
        if prediction is None:
            raise ValueError("Captured speaker evaluation did not expose its current record.")
        return {
            "prediction": prediction,
            "run_references": run_references,
        }


def execute_blind_baseline(
    campaign_id: str,
    baseline_id: str,
    *,
    runtime_root: Optional[Path] = None,
    case_runner: Optional[LocalSpeakerCaseRunner] = None,
) -> dict[str, Any]:
    """Run pending cases serially and capture each result before any reveal."""
    runner = case_runner or LocalSpeakerCaseRunner()
    status = speaker_evaluation_campaign.blind_baseline_status(
        campaign_id,
        baseline_id=baseline_id,
        runtime_root=runtime_root,
    )
    if status.get("status") == "comparison_complete":
        return status
    for case in status.get("cases") or []:
        if (
            not isinstance(case, dict)
            or case.get("status") == "prediction_captured"
        ):
            continue
        document_id = str(case.get("document_id") or "")
        print(
            f"Running blind speaker prediction for {document_id}...",
            file=sys.stderr,
            flush=True,
        )
        try:
            result = runner(document_id)
        except CasePredictionFailure as exc:
            print(
                f"Capturing validated model failure for {document_id}: "
                f"{exc.stage}.",
                file=sys.stderr,
                flush=True,
            )
            result = {
                "prediction": {
                    "evaluation_id": f"failed-evaluation-{uuid4()}",
                    "status": "model_output_rejected",
                    "failure_stage": exc.stage,
                    "failure_class": (
                        "transcript_clue_discovery"
                        if exc.stage == "clue_discovery_validation"
                        else "prompt_reasoning"
                    ),
                    "error": str(exc)[:2_000],
                    "calendar_association": {
                        "status": "ambiguous",
                        "confidence": {"numeric": 0, "band": "none"},
                    },
                    "people": [],
                    "proposals": [],
                    "warnings": ["Host validation rejected the model output."],
                },
                "run_references": exc.run_references,
            }
        status = speaker_evaluation_campaign.capture_blind_prediction(
            campaign_id,
            baseline_id=baseline_id,
            document_id=document_id,
            artifact_sha256=str(case.get("artifact_sha256") or ""),
            prediction=result["prediction"],
            run_references=result.get("run_references") or {},
            runtime_root=runtime_root,
            approval_token=(
                speaker_evaluation_campaign.CAPTURE_BLIND_PREDICTION_TOKEN
            ),
        )
        print(
            "Captured blind prediction "
            f"{status['captured_prediction_count']}/{status['batch_size']} "
            f"for {document_id}.",
            file=sys.stderr,
            flush=True,
        )
    return status


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Execute or reveal one private blind speaker-evaluation baseline."
    )
    parser.add_argument("campaign_id")
    parser.add_argument("baseline_id")
    parser.add_argument(
        "--runtime-root",
        type=Path,
        default=speaker_evaluation_campaign.DEFAULT_CAMPAIGN_ROOT,
    )
    parser.add_argument("--api-base-url", default=DEFAULT_API_BASE_URL)
    parser.add_argument("--timeout-seconds", type=float, default=600)
    parser.add_argument(
        "--reveal",
        action="store_true",
        help="Reveal frozen gold and write comparison after all predictions exist.",
    )
    parser.add_argument(
        "--reveal-reviewed-holdout-replay",
        action="store_true",
        help=(
            "Compare a completed rerun of an already-reviewed holdout only when "
            "an exact prior holdout comparison exists."
        ),
    )
    return parser.parse_args(argv)


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = parse_args(argv)
    try:
        if args.reveal or args.reveal_reviewed_holdout_replay:
            result = speaker_evaluation_campaign.reveal_blind_baseline_comparison(
                args.campaign_id,
                baseline_id=args.baseline_id,
                runtime_root=args.runtime_root,
                approval_token=(
                    speaker_evaluation_campaign.REVEAL_GOLD_COMPARISON_TOKEN
                ),
                allow_reviewed_holdout_replay=(
                    args.reveal_reviewed_holdout_replay
                ),
            )
        else:
            result = execute_blind_baseline(
                args.campaign_id,
                args.baseline_id,
                runtime_root=args.runtime_root,
                case_runner=LocalSpeakerCaseRunner(
                    base_url=args.api_base_url,
                    timeout_seconds=args.timeout_seconds,
                ),
            )
    except (OSError, ValueError, KeyError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
