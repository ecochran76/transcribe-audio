"""Sample-, timeline-, and content-aware validation for deterministic P1 audio."""

from __future__ import annotations

import hashlib
import json
import math
import re
import subprocess
import tempfile
import wave
from collections import Counter
from fractions import Fraction
from pathlib import Path
from typing import Any, Mapping

import acoustic_audio_derivatives as p1


CONTRACT_SCHEMA = "transcribe-audio.content-preservation-contract.v1"
MEASUREMENT_SCHEMA = "transcribe-audio.content-preservation-measurement.v1"
SUPPORTED_CODEC = "aac"
AAC_ACCESS_UNIT_SAMPLES = 1024
MAX_RESAMPLER_ERROR_SAMPLES = 1
MAX_PACKET_INTERVALS_WITHOUT_DISCONTINUITY = 2


class ContentPreservationError(ValueError):
    """Raised when content preservation cannot be measured exactly."""

    def __init__(self, message: str, *, reason_code: str = "measurement_error") -> None:
        super().__init__(message)
        self.reason_code = reason_code


def canonical_hash(value: Any) -> str:
    body = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


def _run_json(arguments: list[str]) -> dict[str, Any]:
    result = subprocess.run(arguments, capture_output=True, text=True, check=False)
    if result.returncode:
        raise ContentPreservationError("Audio metadata command failed.")
    try:
        value = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise ContentPreservationError("Audio metadata output is invalid.") from exc
    if not isinstance(value, dict):
        raise ContentPreservationError("Audio metadata output must be an object.")
    return value


def _stream_metadata(source: Path, ffprobe_path: str) -> dict[str, Any]:
    payload = _run_json(
        [
            ffprobe_path,
            "-v", "error",
            "-select_streams", "a",
            "-count_frames",
            "-show_entries",
            (
                "format=format_name,duration,start_time:"
                "stream=index,codec_type,codec_name,profile,sample_rate,channels,"
                "time_base,start_pts,start_time,duration_ts,duration,nb_frames,"
                "nb_read_frames,initial_padding,trailing_padding"
            ),
            "-of", "json",
            str(source),
        ]
    )
    streams = [
        item
        for item in payload.get("streams") or []
        if isinstance(item, Mapping) and item.get("codec_type") == "audio"
    ]
    if len(streams) != 1:
        return {
            "audio_stream_count": len(streams),
            "format": dict(payload.get("format") or {}),
            "stream": dict(streams[0]) if streams else {},
        }
    return {
        "audio_stream_count": 1,
        "format": dict(payload.get("format") or {}),
        "stream": dict(streams[0]),
    }


def _fraction(value: str) -> Fraction:
    try:
        return Fraction(value)
    except (ValueError, ZeroDivisionError) as exc:
        raise ContentPreservationError("Audio time base is invalid.") from exc


def _packet_metrics(
    source: Path, ffprobe_path: str, *, time_base: Fraction, sample_rate: int
) -> dict[str, Any]:
    process = subprocess.Popen(
        [
            ffprobe_path,
            "-v", "error",
            "-select_streams", "a:0",
            "-show_packets",
            "-show_entries", "packet=pts,duration,flags,side_data_list",
            "-of", "compact=p=0:nk=0",
            str(source),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if process.stdout is None or process.stderr is None:
        raise ContentPreservationError("Packet accounting could not start.")
    packet_count = 0
    first_pts: int | None = None
    last_pts: int | None = None
    previous_pts: int | None = None
    deltas: Counter[int] = Counter()
    durations: Counter[int] = Counter()
    non_monotonic_count = 0
    skip_samples = 0
    discard_padding = 0
    for line in process.stdout:
        pts_match = re.search(r"(?:^|\|)pts=(-?\d+)(?:\||$)", line)
        duration_match = re.search(r"(?:^|\|)duration=(\d+)(?:\||$)", line)
        if pts_match is None or duration_match is None:
            continue
        pts = int(pts_match.group(1))
        duration = int(duration_match.group(1))
        skip_match = re.search(r"(?:^|:)skip_samples=(\d+)(?:\||$)", line)
        discard_match = re.search(r"(?:^|:)discard_padding=(\d+)(?:\||$)", line)
        if skip_match is not None:
            skip_samples += int(skip_match.group(1))
        if discard_match is not None:
            discard_padding += int(discard_match.group(1))
        if first_pts is None:
            first_pts = pts
        if previous_pts is not None:
            delta = pts - previous_pts
            deltas[delta] += 1
            if delta <= 0:
                non_monotonic_count += 1
        durations[duration] += 1
        previous_pts = pts
        last_pts = pts
        packet_count += 1
    stderr = process.stderr.read()
    returncode = process.wait()
    if returncode or stderr.strip():
        raise ContentPreservationError("Packet accounting reported an error.")
    if packet_count <= 0 or first_pts is None or last_pts is None or not deltas:
        raise ContentPreservationError("Packet accounting is incomplete.")
    nominal_ticks = Fraction(AAC_ACCESS_UNIT_SAMPLES, sample_rate) / time_base
    maximum_delta = max(deltas)
    minimum_delta = min(deltas)
    discontinuity_limit = nominal_ticks * MAX_PACKET_INTERVALS_WITHOUT_DISCONTINUITY
    discontinuity_count = sum(
        count for delta, count in deltas.items() if Fraction(delta) >= discontinuity_limit
    )
    return {
        "packet_count": packet_count,
        "first_pts": first_pts,
        "last_pts": last_pts,
        "minimum_pts_delta": minimum_delta,
        "maximum_pts_delta": maximum_delta,
        "distinct_pts_delta_count": len(deltas),
        "non_monotonic_count": non_monotonic_count,
        "discontinuity_count": discontinuity_count,
        "nominal_access_unit_ticks_numerator": nominal_ticks.numerator,
        "nominal_access_unit_ticks_denominator": nominal_ticks.denominator,
        "discontinuity_limit_ticks_numerator": discontinuity_limit.numerator,
        "discontinuity_limit_ticks_denominator": discontinuity_limit.denominator,
        "packet_duration_value_count": len(durations),
        "maximum_packet_duration_ticks": max(durations),
        "skip_samples": skip_samples,
        "discard_padding_samples": discard_padding,
        "packet_pts_extent_seconds": float(Fraction(last_pts - first_pts, 1) * time_base),
    }


def _decode_raw(
    source: Path,
    ffmpeg_path: str,
    *,
    output_channels: int,
    extra_arguments: list[str],
) -> dict[str, Any]:
    process = subprocess.Popen(
        [
            ffmpeg_path,
            "-nostdin", "-hide_banner", "-loglevel", "warning",
            "-i", str(source),
            "-map", "0:a:0", "-vn",
            *extra_arguments,
            "-c:a", "pcm_s16le", "-f", "s16le", "pipe:1",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if process.stdout is None or process.stderr is None:
        raise ContentPreservationError("Reference decode could not start.")
    digest = hashlib.sha256()
    byte_count = 0
    for chunk in iter(lambda: process.stdout.read(1024 * 1024), b""):
        digest.update(chunk)
        byte_count += len(chunk)
    stderr = process.stderr.read().decode("utf-8", "replace")
    returncode = process.wait()
    warning_lines = [line for line in stderr.splitlines() if line.strip()]
    if byte_count % (2 * output_channels):
        raise ContentPreservationError("Reference PCM byte count is misaligned.")
    return {
        "returncode": returncode,
        "warning_line_count": len(warning_lines),
        "sample_count_per_channel": byte_count // (2 * output_channels),
        "pcm_sha256": digest.hexdigest(),
    }


def _wav_pcm_metrics(path: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    try:
        with wave.open(str(path), "rb") as audio:
            channels = audio.getnchannels()
            sample_rate = audio.getframerate()
            sample_width = audio.getsampwidth()
            frame_count = audio.getnframes()
            while True:
                frames = audio.readframes(65_536)
                if not frames:
                    break
                digest.update(frames)
    except (OSError, wave.Error) as exc:
        raise ContentPreservationError("Production WAV is invalid.") from exc
    return {
        "channels": channels,
        "sample_rate": sample_rate,
        "sample_width_bytes": sample_width,
        "frame_count": frame_count,
        "pcm_sha256": digest.hexdigest(),
    }


def contract() -> dict[str, Any]:
    core = {
        "schema_version": CONTRACT_SCHEMA,
        "supported_codec": SUPPORTED_CODEC,
        "aac_access_unit_samples": AAC_ACCESS_UNIT_SAMPLES,
        "maximum_resampler_error_samples": MAX_RESAMPLER_ERROR_SAMPLES,
        "maximum_packet_intervals_without_discontinuity": (
            MAX_PACKET_INTERVALS_WITHOUT_DISCONTINUITY
        ),
        "discontinuity_comparator": "greater_than_or_equal",
        "discontinuity_boundary_meaning": "one_complete_aac_access_unit_interval_missing",
        "reference_paths": [
            "ffprobe_packet_pts_and_duration",
            "native_rate_decode_to_raw_pcm",
            "recipe_rate_decode_to_raw_pcm",
            "production_wav_pcm_frames",
        ],
        "container_duration_is_decision_authority": False,
        "requires_exact_source_hash": True,
        "requires_source_hash_unchanged_after_measurement": True,
        "requires_exact_tool_versions": True,
        "requires_one_audio_stream": True,
        "requires_zero_decode_warnings": True,
        "requires_packet_to_native_sample_equality": True,
        "reconciles_packet_skip_samples_and_discard_padding": True,
        "requires_reference_to_wav_pcm_hash_equality": True,
        "ambiguous_timeline_discontinuity_policy": "reject",
    }
    return {**core, "content_sha256": canonical_hash(core)}


def validate_measurement(value: Mapping[str, Any]) -> dict[str, Any]:
    reasons: list[str] = []
    metadata = value.get("metadata") or {}
    stream = metadata.get("stream") or {}
    packets = value.get("packets") or {}
    native = value.get("native_decode") or {}
    reference = value.get("recipe_reference_decode") or {}
    output = value.get("production_wav") or {}
    tools = value.get("tool_identity") or {}
    if value.get("source_unchanged") is not True:
        reasons.append("source_changed_during_measurement")
    if metadata.get("audio_stream_count") != 1:
        reasons.append("audio_stream_count_not_one")
    if stream.get("codec_name") != SUPPORTED_CODEC:
        reasons.append("unsupported_codec")
    if not all(
        isinstance(tools.get(field), str) and tools.get(field)
        for field in ("decoder_path", "decoder_version", "probe_path", "probe_version")
    ):
        reasons.append("tool_identity_missing")
    if native.get("returncode") != 0 or reference.get("returncode") != 0:
        reasons.append("decode_error")
    if native.get("warning_line_count") != 0 or reference.get("warning_line_count") != 0:
        reasons.append("decode_warning")
    if packets.get("non_monotonic_count") != 0:
        reasons.append("non_monotonic_packet_timeline")
    if packets.get("discontinuity_count") != 0:
        reasons.append("timeline_discontinuity")
    if value.get("packet_expected_native_samples") != native.get("sample_count_per_channel"):
        reasons.append("packet_native_sample_mismatch")
    output_sample_error = value.get("output_sample_error")
    if not isinstance(output_sample_error, int):
        reasons.append("output_sample_error_missing")
    elif abs(output_sample_error) > MAX_RESAMPLER_ERROR_SAMPLES:
        reasons.append("output_sample_extent_mismatch")
    if reference.get("sample_count_per_channel") != output.get("frame_count"):
        reasons.append("reference_output_frame_mismatch")
    if reference.get("pcm_sha256") != output.get("pcm_sha256"):
        reasons.append("output_content_mismatch")
    if output.get("channels") != p1.TARGET_CHANNELS:
        reasons.append("output_channel_mismatch")
    if output.get("sample_rate") != p1.TARGET_SAMPLE_RATE:
        reasons.append("output_sample_rate_mismatch")
    if output.get("sample_width_bytes") != 2:
        reasons.append("output_sample_width_mismatch")
    unique_reasons = list(dict.fromkeys(reasons))
    return {
        "status": "passing" if not unique_reasons else "rejected",
        "reason_codes": unique_reasons,
    }


def measure(
    source: Path,
    *,
    expected_source_sha256: str,
    channel_policy: str = "stereo_average_to_mono",
    channel_policy_authority_sha256: str,
    production_wav: Path | None = None,
) -> dict[str, Any]:
    if p1.sha256_file(source) != expected_source_sha256:
        raise ContentPreservationError(
            "Source hash does not match authority.", reason_code="source_hash_mismatch"
        )
    ffprobe_preflight = p1._tool("ffprobe")
    preflight = _stream_metadata(source, ffprobe_preflight)
    if preflight["audio_stream_count"] != 1:
        raise ContentPreservationError(
            "Source must contain exactly one audio stream.",
            reason_code="audio_stream_count_not_one",
        )
    try:
        plan, _ = p1._build_plan(
            source,
            expected_source_sha256=expected_source_sha256,
            channel_policy=channel_policy,
            channel_policy_authority_sha256=channel_policy_authority_sha256,
        )
    except p1.AudioDerivativeError as exc:
        raise ContentPreservationError(
            "Source probe or deterministic recipe construction failed.",
            reason_code="source_probe_or_recipe_error",
        ) from exc
    ffmpeg_path = str(plan["recipe"]["decoder_path"])
    ffprobe_path = str(plan["recipe"]["probe_path"])
    metadata = preflight
    stream = metadata["stream"]
    try:
        sample_rate = int(stream.get("sample_rate") or 0)
        channels = int(stream.get("channels") or 0)
        time_base = _fraction(str(stream.get("time_base") or ""))
    except (TypeError, ValueError) as exc:
        raise ContentPreservationError("Source stream dimensions are invalid.") from exc
    if sample_rate <= 0 or channels <= 0:
        raise ContentPreservationError("Source stream dimensions are invalid.")
    packets = _packet_metrics(
        source, ffprobe_path, time_base=time_base, sample_rate=sample_rate
    )
    native = _decode_raw(
        source, ffmpeg_path, output_channels=channels, extra_arguments=[]
    )
    channel_arguments = (
        ["-af", "pan=mono|c0=0.5*c0+0.5*c1"]
        if channel_policy == "stereo_average_to_mono" and channels == 2
        else []
    )
    reference = _decode_raw(
        source,
        ffmpeg_path,
        output_channels=p1.TARGET_CHANNELS,
        extra_arguments=[
            *channel_arguments,
            "-ac", str(p1.TARGET_CHANNELS),
            "-ar", str(p1.TARGET_SAMPLE_RATE),
        ],
    )
    if production_wav is None:
        with tempfile.TemporaryDirectory(prefix="content-preservation-") as directory:
            output_path = Path(directory) / "production.wav"
            p1._decode(source, output_path, plan["recipe"])
            output = _wav_pcm_metrics(output_path)
    else:
        output = _wav_pcm_metrics(production_wav)
    native_samples = int(native["sample_count_per_channel"])
    expected_output = round(Fraction(native_samples * p1.TARGET_SAMPLE_RATE, sample_rate))
    format_duration = float((metadata.get("format") or {}).get("duration") or 0)
    native_duration = native_samples / sample_rate
    core = {
        "schema_version": MEASUREMENT_SCHEMA,
        "contract_sha256": contract()["content_sha256"],
        "source_sha256": expected_source_sha256,
        "tool_identity": {
            "decoder_path": plan["recipe"]["decoder_path"],
                "decoder_version": plan["recipe"]["decoder_revision"],
            "probe_path": plan["recipe"]["probe_path"],
                "probe_version": plan["recipe"]["probe_revision"],
        },
        "metadata": metadata,
        "packets": packets,
        "native_decode": native,
        "recipe_reference_decode": reference,
        "production_wav": output,
        "packet_expected_native_samples": (
            int(packets["packet_count"]) * AAC_ACCESS_UNIT_SAMPLES
            - int(packets["skip_samples"])
            - int(packets["discard_padding_samples"])
        ),
        "expected_output_samples": expected_output,
        "output_sample_error": int(output["frame_count"]) - expected_output,
        "container_duration_seconds": format_duration,
        "native_decoded_duration_seconds": native_duration,
        "container_minus_native_seconds": format_duration - native_duration,
        "container_clock_difference_ppm": (
            ((format_duration - native_duration) / native_duration) * 1_000_000
            if native_duration
            else math.inf
        ),
        "source_unchanged": p1.sha256_file(source) == expected_source_sha256,
        "used_injected_production_fixture": production_wav is not None,
    }
    decision = validate_measurement(core)
    return {**core, **decision, "content_sha256": canonical_hash({**core, **decision})}
