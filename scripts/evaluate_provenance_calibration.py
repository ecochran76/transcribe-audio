#!/usr/bin/env python3
"""
Evaluate P04 provenance-source quality decisions against reviewed manifests.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from context_sources import (  # noqa: E402
    SOURCE_QUALITY_DEFAULT_MIN_SCORE,
    SOURCE_QUALITY_PROFILE_ID,
    provenance_quality_profile_summary,
    provenance_quality_terms,
    quality_for_source,
    source_quality_profile,
    source_type_min_score,
)
from routing_artifacts import ProvenanceSource, normalize_string  # noqa: E402
from transcribe_common import TranscriptionError  # noqa: E402

REPORT_SCHEMA_VERSION = "transcribe-audio.p04-calibration-report.v1"
MANIFEST_SCHEMA_VERSION = 1
DEFAULT_MANIFEST_DIR = Path("~/.local/state/transcribe-audio/p04-calibration/manifests")
FORBIDDEN_TRANSCRIPT_KEYS = {"text", "transcript_text", "raw_text", "raw_transcript", "utterances", "words"}


@dataclass
class EvaluationResult:
    manifest_id: str
    case_id: str
    decision_id: str
    source_family: str
    source_type: str
    expected: str
    actual: str
    outcome: str
    score: float
    required_score: int
    matched_terms: list[str]
    quality_profile: str
    reason: str
    rationale: str
    evidence_label: str

    def to_report_dict(self) -> dict[str, Any]:
        return {
            "manifest_id": self.manifest_id,
            "case_id": self.case_id,
            "decision_id": self.decision_id,
            "source_family": self.source_family,
            "source_type": self.source_type,
            "expected": self.expected,
            "actual": self.actual,
            "outcome": self.outcome,
            "score": self.score,
            "required_score": self.required_score,
            "matched_terms": self.matched_terms,
            "quality_profile": self.quality_profile,
            "reason": self.reason,
            "rationale": self.rationale,
            "evidence_label": self.evidence_label,
        }


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate P04 provenance calibration manifests.")
    parser.add_argument(
        "inputs",
        nargs="*",
        type=Path,
        help="Manifest JSON files or directories. Defaults to the local P04 calibration manifest directory.",
    )
    parser.add_argument(
        "--manifest-dir",
        action="append",
        type=Path,
        default=[],
        help="Additional directory containing *.json calibration manifests.",
    )
    parser.add_argument("--output", type=Path, help="Write a sanitized JSON report to this path.")
    parser.add_argument(
        "--min-score",
        type=int,
        default=SOURCE_QUALITY_DEFAULT_MIN_SCORE,
        help="Default non-calendar compact-term threshold.",
    )
    parser.add_argument(
        "--require-decision-count",
        type=int,
        default=0,
        help="Fail unless at least this many reviewed source decisions are evaluated.",
    )
    parser.add_argument(
        "--require-source-families",
        type=int,
        default=0,
        help="Fail unless at least this many source families are represented.",
    )
    parser.add_argument("--fail-on-mismatch", action="store_true", help="Return non-zero when mismatches exist.")
    parser.add_argument("--include-passed", action="store_true", help="Include all passed decisions in the report.")
    return parser.parse_args(argv)


def load_json_object(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.expanduser().read_text(encoding="utf-8"))
    except OSError as exc:
        raise TranscriptionError(f"Failed to read {path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise TranscriptionError(f"{path} is not valid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise TranscriptionError(f"{path} must contain a JSON object.")
    return payload


def iter_manifest_paths(paths: list[Path]) -> list[Path]:
    discovered: list[Path] = []
    for path in paths:
        expanded = path.expanduser()
        if expanded.is_dir():
            discovered.extend(sorted(item for item in expanded.glob("*.json") if item.is_file()))
        elif expanded.is_file():
            discovered.append(expanded)
        else:
            raise TranscriptionError(f"Manifest path does not exist: {path}")
    return discovered


def expected_decision(value: Any) -> str:
    normalized = normalize_string(value).lower()
    if normalized in {"include", "included", "expected_include"}:
        return "include"
    if normalized in {"exclude", "excluded", "expected_exclude"}:
        return "exclude"
    raise TranscriptionError(f"Expected decision must be include or exclude, got {value!r}.")


def actual_decision(status: str) -> str:
    return "include" if status in {"included", "included_unfiltered"} else "exclude"


def source_family_for_type(source_type: str) -> str:
    if source_type in {"calendar_event", "gws_calendar_overlap", "gws_calendar_event_detail"}:
        return "calendar"
    if source_type in {"gws_drive_file", "gws_docs_file"}:
        return "drive_docs"
    if source_type.startswith("graphiti_"):
        return "graphiti"
    if source_type.startswith("odollo_"):
        return "odollo"
    return source_type or "unknown"


def reject_forbidden_transcript_fields(value: Any, *, path: str = "transcript") -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            if key in FORBIDDEN_TRANSCRIPT_KEYS:
                raise TranscriptionError(f"Calibration manifest includes forbidden raw transcript field: {path}.{key}")
            reject_forbidden_transcript_fields(child, path=f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            reject_forbidden_transcript_fields(child, path=f"{path}[{index}]")


def provenance_source_from_manifest(payload: dict[str, Any]) -> ProvenanceSource:
    return ProvenanceSource(
        source_type=normalize_string(payload.get("source_type")),
        source_id=normalize_string(payload.get("source_id")),
        label=normalize_string(payload.get("label")),
        uri=normalize_string(payload.get("uri")),
        snippet=normalize_string(payload.get("snippet")),
        metadata=payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {},
    )


def evaluate_manifest(payload: dict[str, Any], *, min_score: int) -> list[EvaluationResult]:
    schema_version = payload.get("schema_version")
    if schema_version != MANIFEST_SCHEMA_VERSION:
        raise TranscriptionError(f"Calibration manifest schema_version must be {MANIFEST_SCHEMA_VERSION}.")
    manifest_id = normalize_string(payload.get("manifest_id")) or "unnamed-manifest"
    cases = payload.get("cases")
    if not isinstance(cases, list) or not cases:
        raise TranscriptionError(f"{manifest_id} must contain at least one calibration case.")

    results: list[EvaluationResult] = []
    for case in cases:
        if not isinstance(case, dict):
            raise TranscriptionError(f"{manifest_id} contains a non-object case.")
        case_id = normalize_string(case.get("case_id")) or "unnamed-case"
        transcript = case.get("transcript") if isinstance(case.get("transcript"), dict) else {}
        readout = case.get("readout") if isinstance(case.get("readout"), dict) else {}
        reject_forbidden_transcript_fields(transcript)
        terms = provenance_quality_terms(transcript, readout)
        decisions = case.get("decisions")
        if not isinstance(decisions, list) or not decisions:
            raise TranscriptionError(f"{manifest_id}/{case_id} must contain at least one source decision.")
        for index, decision in enumerate(decisions, start=1):
            if not isinstance(decision, dict):
                raise TranscriptionError(f"{manifest_id}/{case_id} contains a non-object decision.")
            source_payload = decision.get("source") if isinstance(decision.get("source"), dict) else {}
            source = provenance_source_from_manifest(source_payload)
            if not source.source_type:
                raise TranscriptionError(f"{manifest_id}/{case_id} decision {index} has no source_type.")
            expected = expected_decision(decision.get("expected"))
            status, score, matched_terms, reason = quality_for_source(source, terms=terms, min_score=max(min_score, 0))
            actual = actual_decision(status)
            source_family = normalize_string(decision.get("source_family")) or source_family_for_type(source.source_type)
            outcome = "pass" if expected == actual else ("false_positive" if actual == "include" else "false_negative")
            results.append(
                EvaluationResult(
                    manifest_id=manifest_id,
                    case_id=case_id,
                    decision_id=normalize_string(decision.get("decision_id")) or f"{case_id}-{index}",
                    source_family=source_family,
                    source_type=source.source_type,
                    expected=expected,
                    actual=actual,
                    outcome=outcome,
                    score=score,
                    required_score=source_type_min_score(source, max(min_score, 0)),
                    matched_terms=matched_terms,
                    quality_profile=source_quality_profile(source),
                    reason=reason,
                    rationale=normalize_string(decision.get("rationale")),
                    evidence_label=normalize_string(decision.get("evidence_label")),
                )
            )
    return results


def summarize_bucket(results: list[EvaluationResult], attr: str) -> dict[str, dict[str, int]]:
    buckets: dict[str, Counter[str]] = defaultdict(Counter)
    for result in results:
        buckets[str(getattr(result, attr))][result.outcome] += 1
        buckets[str(getattr(result, attr))]["total"] += 1
        buckets[str(getattr(result, attr))][f"expected_{result.expected}"] += 1
        buckets[str(getattr(result, attr))][f"actual_{result.actual}"] += 1
    normalized = {}
    for key, value in sorted(buckets.items()):
        normalized[key] = {
            "total": value.get("total", 0),
            "pass": value.get("pass", 0),
            "false_positive": value.get("false_positive", 0),
            "false_negative": value.get("false_negative", 0),
            "expected_include": value.get("expected_include", 0),
            "expected_exclude": value.get("expected_exclude", 0),
            "actual_include": value.get("actual_include", 0),
            "actual_exclude": value.get("actual_exclude", 0),
        }
    return normalized


def build_report(
    *,
    manifest_paths: list[Path],
    results: list[EvaluationResult],
    min_score: int,
    include_passed: bool,
) -> dict[str, Any]:
    totals = Counter(result.outcome for result in results)
    totals["decisions"] = len(results)
    totals["manifests"] = len(manifest_paths)
    totals["cases"] = len({(result.manifest_id, result.case_id) for result in results})
    totals["expected_include"] = sum(1 for result in results if result.expected == "include")
    totals["expected_exclude"] = sum(1 for result in results if result.expected == "exclude")
    totals["actual_include"] = sum(1 for result in results if result.actual == "include")
    totals["actual_exclude"] = sum(1 for result in results if result.actual == "exclude")
    for key in ("pass", "false_positive", "false_negative"):
        totals[key] = totals.get(key, 0)
    mismatches = [result for result in results if result.outcome != "pass"]
    report_decisions = results if include_passed else mismatches
    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "generated_at": utc_now_iso(),
        "quality_profile": provenance_quality_profile_summary(min_score=min_score, enabled=True),
        "inputs": [str(path) for path in manifest_paths],
        "totals": dict(totals),
        "source_families": summarize_bucket(results, "source_family"),
        "source_types": summarize_bucket(results, "source_type"),
        "mismatches": [result.to_report_dict() for result in mismatches],
        "decisions": [result.to_report_dict() for result in report_decisions],
    }


def write_report(path: Path, report: dict[str, Any]) -> None:
    expanded = path.expanduser()
    expanded.parent.mkdir(parents=True, exist_ok=True)
    with expanded.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True, ensure_ascii=False)
        handle.write("\n")


def run(args: argparse.Namespace) -> dict[str, Any]:
    inputs = list(args.inputs) + list(args.manifest_dir)
    if not inputs:
        inputs = [DEFAULT_MANIFEST_DIR]
    manifest_paths = iter_manifest_paths(inputs)
    if not manifest_paths:
        raise TranscriptionError("No calibration manifests found.")
    results: list[EvaluationResult] = []
    for path in manifest_paths:
        results.extend(evaluate_manifest(load_json_object(path), min_score=args.min_score))
    report = build_report(
        manifest_paths=manifest_paths,
        results=results,
        min_score=args.min_score,
        include_passed=args.include_passed,
    )
    source_families = {result.source_family for result in results}
    errors = []
    if args.require_decision_count and len(results) < args.require_decision_count:
        errors.append(f"Expected at least {args.require_decision_count} decisions; evaluated {len(results)}.")
    if args.require_source_families and len(source_families) < args.require_source_families:
        errors.append(
            f"Expected at least {args.require_source_families} source families; evaluated {len(source_families)}."
        )
    if args.fail_on_mismatch and report["totals"].get("false_positive", 0) + report["totals"].get("false_negative", 0):
        errors.append("Calibration mismatches were found.")
    report["gate_errors"] = errors
    if args.output:
        write_report(args.output, report)
    if errors:
        raise TranscriptionError("; ".join(errors))
    return report


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = parse_args(argv)
    try:
        report = run(args)
    except TranscriptionError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    if args.output:
        print(f"CALIBRATION_REPORT_JSON={args.output.expanduser()}")
    print(
        "Calibration decisions: "
        f"{report['totals']['decisions']} evaluated, "
        f"{report['totals'].get('false_positive', 0)} false positives, "
        f"{report['totals'].get('false_negative', 0)} false negatives."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
