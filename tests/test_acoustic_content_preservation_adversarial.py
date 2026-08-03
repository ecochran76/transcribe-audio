import shutil
import subprocess
from pathlib import Path

import acoustic_audio_derivatives as p1
import acoustic_content_preservation_adversarial as adversarial


def _make_source(path: Path) -> None:
    ffmpeg = shutil.which("ffmpeg")
    assert ffmpeg is not None
    result = subprocess.run(
        [
            ffmpeg,
            "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
            "-f", "lavfi", "-i", "sine=frequency=997:sample_rate=48000:duration=75",
            "-ac", "2", "-c:a", "aac", "-b:a", "128k", str(path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_real_adversarial_grid_rejects_every_fault(tmp_path: Path) -> None:
    source = tmp_path / "source.m4a"
    _make_source(source)

    result = adversarial.run_development_adversaries(
        source,
        expected_source_sha256=p1.sha256_file(source),
        channel_policy_authority_sha256="a" * 64,
    )

    assert result["case_count"] == 9
    for item in result["cases"]:
        assert item["status"] == "rejected", item
        assert item["expected_reason_observed"] is True, item
    assert result["all_expected_rejections_observed"] is True
    assert {item["case_id"] for item in result["cases"]} == {
        "tail_loss_2_frames",
        "tail_loss_320_frames",
        "tail_loss_4000_frames",
        "tail_loss_16000_frames",
        "middle_packet_equivalent_removal",
        "corrupt_output_tail_content",
        "timestamp_discontinuity",
        "wrong_stream_count",
        "corrupt_source_tail",
    }
