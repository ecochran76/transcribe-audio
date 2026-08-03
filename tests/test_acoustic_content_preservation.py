import copy

import acoustic_content_preservation as preservation


def _measurement() -> dict:
    return {
        "metadata": {
            "audio_stream_count": 1,
            "stream": {"codec_name": "aac"},
        },
        "tool_identity": {
            "decoder_path": "/usr/bin/ffmpeg",
            "decoder_version": "ffmpeg test",
            "probe_path": "/usr/bin/ffprobe",
            "probe_version": "ffprobe test",
        },
        "packets": {
            "non_monotonic_count": 0,
            "discontinuity_count": 0,
        },
        "native_decode": {
            "returncode": 0,
            "warning_line_count": 0,
            "sample_count_per_channel": 30_720,
        },
        "recipe_reference_decode": {
            "returncode": 0,
            "warning_line_count": 0,
            "sample_count_per_channel": 10_240,
            "pcm_sha256": "a" * 64,
        },
        "production_wav": {
            "channels": 1,
            "sample_rate": 16_000,
            "sample_width_bytes": 2,
            "frame_count": 10_240,
            "pcm_sha256": "a" * 64,
        },
        "packet_expected_native_samples": 30_720,
        "output_sample_error": 0,
        "source_unchanged": True,
    }


def test_validator_accepts_content_equivalent_decode() -> None:
    assert preservation.validate_measurement(_measurement()) == {
        "status": "passing",
        "reason_codes": [],
    }


def test_validator_rejects_each_non_circular_failure_class() -> None:
    cases = []
    timeline = copy.deepcopy(_measurement())
    timeline["packets"]["discontinuity_count"] = 1
    cases.append((timeline, "timeline_discontinuity"))
    extent = copy.deepcopy(_measurement())
    extent["output_sample_error"] = preservation.MAX_RESAMPLER_ERROR_SAMPLES + 1
    cases.append((extent, "output_sample_extent_mismatch"))
    content = copy.deepcopy(_measurement())
    content["production_wav"]["pcm_sha256"] = "b" * 64
    cases.append((content, "output_content_mismatch"))
    corrupt = copy.deepcopy(_measurement())
    corrupt["native_decode"]["warning_line_count"] = 1
    cases.append((corrupt, "decode_warning"))
    stream = copy.deepcopy(_measurement())
    stream["metadata"]["audio_stream_count"] = 2
    cases.append((stream, "audio_stream_count_not_one"))
    source = copy.deepcopy(_measurement())
    source["source_unchanged"] = False
    cases.append((source, "source_changed_during_measurement"))
    malformed = copy.deepcopy(_measurement())
    del malformed["output_sample_error"]
    cases.append((malformed, "output_sample_error_missing"))

    for value, reason in cases:
        result = preservation.validate_measurement(value)
        assert result["status"] == "rejected"
        assert reason in result["reason_codes"]


def test_contract_does_not_use_container_duration_as_authority() -> None:
    value = preservation.contract()

    assert value["container_duration_is_decision_authority"] is False
    assert value["maximum_resampler_error_samples"] == 1
    assert value["maximum_packet_intervals_without_discontinuity"] == 2
    assert value["ambiguous_timeline_discontinuity_policy"] == "reject"
