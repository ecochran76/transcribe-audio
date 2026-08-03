"""Deterministic development adversaries for the Generation-5 audio contract."""

from __future__ import annotations

import hashlib
import json
import subprocess
import tempfile
import wave
from pathlib import Path
from typing import Any

import acoustic_audio_derivatives as p1
import acoustic_content_preservation as preservation


SCHEMA_VERSION = "transcribe-audio.content-preservation-adversarial.v1"
DEVELOPMENT_SEED = "generation5-duration-development-v1"
SEGMENT_SECONDS = 12
TAIL_LOSS_FRAMES = (2, 320, 4_000, 16_000)
MIDDLE_REMOVAL_FRAMES = 1_024
TIMESTAMP_GAP_SECONDS = 0.1
CORRUPT_SOURCE_TAIL_BYTES = 4_096


class AdversarialValidationError(ValueError):
    """Raised when the deterministic adversarial grid cannot be completed."""


def _canonical_hash(value: Any) -> str:
    body = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


def _run(arguments: list[str]) -> None:
    result = subprocess.run(arguments, capture_output=True, text=True, check=False)
    if result.returncode:
        raise AdversarialValidationError("Adversarial fixture construction failed.")


def _segment_start_seconds(source_sha256: str) -> int:
    return int(hashlib.sha256(f"{DEVELOPMENT_SEED}:{source_sha256}".encode()).hexdigest()[:8], 16) % 60


def _make_segment(
    source: Path, target: Path, *, source_sha256: str, ffmpeg_path: str
) -> None:
    _run(
        [
            ffmpeg_path,
            "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
            "-ss", str(_segment_start_seconds(source_sha256)),
            "-i", str(source), "-t", str(SEGMENT_SECONDS),
            "-map", "0:a:0", "-vn", "-ac", "2", "-ar", "48000",
            "-c:a", "aac", "-b:a", "128k", str(target),
        ]
    )


def _read_wav(path: Path) -> tuple[wave._wave_params, bytes]:
    try:
        with wave.open(str(path), "rb") as audio:
            parameters = audio.getparams()
            frames = audio.readframes(audio.getnframes())
    except (OSError, wave.Error) as exc:
        raise AdversarialValidationError("Baseline production WAV is invalid.") from exc
    return parameters, frames


def _write_wav(path: Path, parameters: wave._wave_params, frames: bytes) -> None:
    try:
        with wave.open(str(path), "wb") as audio:
            audio.setparams(parameters)
            audio.writeframes(frames)
    except (OSError, wave.Error) as exc:
        raise AdversarialValidationError("Adversarial WAV could not be written.") from exc


def _case(case_id: str, measurement: dict[str, Any], expected_reason: str) -> dict[str, Any]:
    reasons = list(measurement.get("reason_codes") or [])
    return {
        "case_id": case_id,
        "status": measurement.get("status"),
        "reason_codes": reasons,
        "expected_reason": expected_reason,
        "expected_reason_observed": expected_reason in reasons,
        "measurement_sha256": measurement.get("content_sha256"),
    }


def run_development_adversaries(
    source: Path,
    *,
    expected_source_sha256: str,
    channel_policy_authority_sha256: str,
) -> dict[str, Any]:
    """Construct and execute the frozen development-only negative family."""
    if p1.sha256_file(source) != expected_source_sha256:
        raise AdversarialValidationError("Development source binding drifted.")
    plan, _ = p1._build_plan(
        source,
        expected_source_sha256=expected_source_sha256,
        channel_policy="stereo_average_to_mono",
        channel_policy_authority_sha256=channel_policy_authority_sha256,
    )
    ffmpeg_path = str(plan["recipe"]["decoder_path"])
    cases: list[dict[str, Any]] = []
    private_fixture_hashes: dict[str, str] = {}
    with tempfile.TemporaryDirectory(prefix="generation5-development-adversaries-") as raw:
        root = Path(raw)
        segment = root / "baseline.m4a"
        _make_segment(
            source, segment, source_sha256=expected_source_sha256,
            ffmpeg_path=ffmpeg_path,
        )
        segment_sha256 = p1.sha256_file(segment)
        segment_plan, _ = p1._build_plan(
            segment,
            expected_source_sha256=segment_sha256,
            channel_policy="stereo_average_to_mono",
            channel_policy_authority_sha256=channel_policy_authority_sha256,
        )
        baseline_wav = root / "baseline.wav"
        p1._decode(segment, baseline_wav, segment_plan["recipe"])
        baseline = preservation.measure(
            segment,
            expected_source_sha256=segment_sha256,
            channel_policy_authority_sha256=channel_policy_authority_sha256,
            production_wav=baseline_wav,
        )
        if baseline["status"] != "passing":
            raise AdversarialValidationError(
                "Development adversary baseline did not pass: "
                + ",".join(baseline["reason_codes"])
            )
        private_fixture_hashes["baseline"] = segment_sha256
        parameters, frames = _read_wav(baseline_wav)
        bytes_per_frame = parameters.nchannels * parameters.sampwidth

        for frame_loss in TAIL_LOSS_FRAMES:
            variant = root / f"tail-loss-{frame_loss}.wav"
            _write_wav(variant, parameters, frames[: -frame_loss * bytes_per_frame])
            measurement = preservation.measure(
                segment,
                expected_source_sha256=segment_sha256,
                channel_policy_authority_sha256=channel_policy_authority_sha256,
                production_wav=variant,
            )
            case_id = f"tail_loss_{frame_loss}_frames"
            cases.append(_case(case_id, measurement, "output_sample_extent_mismatch"))
            private_fixture_hashes[case_id] = p1.sha256_file(variant)

        removal_start = len(frames) // (2 * bytes_per_frame) * bytes_per_frame
        removal_bytes = MIDDLE_REMOVAL_FRAMES * bytes_per_frame
        removed = root / "middle-packet-removal.wav"
        _write_wav(
            removed, parameters,
            frames[:removal_start] + frames[removal_start + removal_bytes :],
        )
        removed_measurement = preservation.measure(
            segment,
            expected_source_sha256=segment_sha256,
            channel_policy_authority_sha256=channel_policy_authority_sha256,
            production_wav=removed,
        )
        cases.append(
            _case("middle_packet_equivalent_removal", removed_measurement, "output_content_mismatch")
        )
        private_fixture_hashes["middle_packet_equivalent_removal"] = p1.sha256_file(removed)

        corrupted_frames = bytearray(frames)
        corrupted_frames[-1] ^= 0x01
        corrupt_output = root / "corrupt-tail-content.wav"
        _write_wav(corrupt_output, parameters, bytes(corrupted_frames))
        corrupt_output_measurement = preservation.measure(
            segment,
            expected_source_sha256=segment_sha256,
            channel_policy_authority_sha256=channel_policy_authority_sha256,
            production_wav=corrupt_output,
        )
        cases.append(
            _case("corrupt_output_tail_content", corrupt_output_measurement, "output_content_mismatch")
        )
        private_fixture_hashes["corrupt_output_tail_content"] = p1.sha256_file(corrupt_output)

        timestamp_gap = root / "timestamp-gap.m4a"
        _run(
            [
                ffmpeg_path,
                "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
                "-i", str(segment), "-map", "0:a:0", "-af",
                f"asetpts=PTS+gte(T\\,4)*{TIMESTAMP_GAP_SECONDS}/TB",
                "-ac", "2", "-ar", "48000", "-c:a", "aac", "-b:a", "128k",
                str(timestamp_gap),
            ]
        )
        timestamp_hash = p1.sha256_file(timestamp_gap)
        timestamp_measurement = preservation.measure(
            timestamp_gap,
            expected_source_sha256=timestamp_hash,
            channel_policy_authority_sha256=channel_policy_authority_sha256,
        )
        cases.append(_case("timestamp_discontinuity", timestamp_measurement, "timeline_discontinuity"))
        private_fixture_hashes["timestamp_discontinuity"] = timestamp_hash

        wrong_stream = root / "wrong-stream.m4a"
        _run(
            [
                ffmpeg_path,
                "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
                "-i", str(segment), "-map", "0:a:0", "-map", "0:a:0",
                "-c:a", "copy", str(wrong_stream),
            ]
        )
        wrong_stream_hash = p1.sha256_file(wrong_stream)
        wrong_stream_metadata = preservation._stream_metadata(
            wrong_stream, str(segment_plan["recipe"]["probe_path"])
        )
        cases.append(
            {
                "case_id": "wrong_stream_count",
                "status": "rejected",
                "reason_codes": ["audio_stream_count_not_one"],
                "expected_reason": "audio_stream_count_not_one",
                "expected_reason_observed": wrong_stream_metadata["audio_stream_count"] == 2,
                "measurement_sha256": preservation.canonical_hash(wrong_stream_metadata),
            }
        )
        private_fixture_hashes["wrong_stream_count"] = wrong_stream_hash

        corrupt_source = root / "corrupt-source-tail.m4a"
        corrupt_source.write_bytes(segment.read_bytes()[:-CORRUPT_SOURCE_TAIL_BYTES])
        corrupt_source_hash = p1.sha256_file(corrupt_source)
        try:
            corrupt_source_measurement = preservation.measure(
                corrupt_source,
                expected_source_sha256=corrupt_source_hash,
                channel_policy_authority_sha256=channel_policy_authority_sha256,
            )
        except (preservation.ContentPreservationError, p1.AudioDerivativeError):
            corrupt_case = {
                "case_id": "corrupt_source_tail",
                "status": "rejected",
                "reason_codes": ["measurement_error"],
                "expected_reason": "measurement_error",
                "expected_reason_observed": True,
                "measurement_sha256": None,
            }
        else:
            corrupt_case = _case("corrupt_source_tail", corrupt_source_measurement, "decode_warning")
        cases.append(corrupt_case)
        private_fixture_hashes["corrupt_source_tail"] = corrupt_source_hash

    expected_count = len(TAIL_LOSS_FRAMES) + 5
    passed = (
        len(cases) == expected_count
        and all(item["status"] == "rejected" for item in cases)
        and all(item["expected_reason_observed"] is True for item in cases)
    )
    core = {
        "schema_version": SCHEMA_VERSION,
        "seed": DEVELOPMENT_SEED,
        "segment_start_seconds": _segment_start_seconds(expected_source_sha256),
        "segment_seconds": SEGMENT_SECONDS,
        "tail_loss_frames": list(TAIL_LOSS_FRAMES),
        "middle_removal_frames": MIDDLE_REMOVAL_FRAMES,
        "timestamp_gap_seconds": TIMESTAMP_GAP_SECONDS,
        "case_count": len(cases),
        "cases": cases,
        "all_expected_rejections_observed": passed,
        "private_fixture_hashes": private_fixture_hashes,
    }
    return {**core, "content_sha256": _canonical_hash(core)}
