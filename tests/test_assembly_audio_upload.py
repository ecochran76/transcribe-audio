from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import assembly_transcribe as assembly


def make_video(path: Path, *, audio: bool = True) -> None:
    command = ["ffmpeg", "-nostdin", "-v", "error", "-f", "lavfi",
               "-i", "testsrc2=size=64x64:rate=5"]
    if audio:
        command += ["-f", "lavfi", "-i", "sine=frequency=440:sample_rate=48000",
                    "-c:a", "pcm_s16le"]
    command += ["-t", "0.5", "-c:v", "ffv1", str(path)]
    subprocess.run(command, check=True, capture_output=True, timeout=15)


def probe(path: Path) -> list[dict]:
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_streams", "-of", "json", str(path)],
        check=True, capture_output=True, text=True, timeout=15,
    )
    return json.loads(result.stdout)["streams"]


class ProviderResponse:
    status_code = 200

    def __init__(self, payload):
        self.payload = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self.payload


class LocalProvider:
    def __init__(self, upload: Path):
        self.headers = {}
        self.upload = upload

    def post(self, url, *, data=None, json=None):
        if url.endswith("/upload"):
            with self.upload.open("wb") as output:
                for chunk in data:
                    output.write(chunk)
            return ProviderResponse({"upload_url": "https://example.invalid/audio"})
        return ProviderResponse({"id": "synthetic-transcript"})

    def get(self, url):
        return ProviderResponse({"status": "completed", "audio_duration": 0.5,
                                 "utterances": [{"speaker": "A", "start": 0,
                                                 "end": 500, "text": "Synthetic test."}]})


def test_video_processing_uploads_audio_and_keeps_original_outputs(
    tmp_path, monkeypatch,
):
    source = tmp_path / "meeting.mkv"
    make_video(source)
    before = source.read_bytes()
    os.utime(source, (1700000000, 1700000000))
    work = tmp_path / "work"
    work.mkdir()
    monkeypatch.setattr(assembly.tempfile, "tempdir", str(work))
    uploaded = tmp_path / "uploaded.flac"
    monkeypatch.setattr(
        assembly.requests, "Session", lambda: LocalProvider(uploaded),
    )
    args = assembly.parse_args([str(source), "--text-output"])
    assert assembly.process_audio_file(source, args, "fake", ("en_us", False), None)
    streams = probe(uploaded)
    assert [s["codec_type"] for s in streams] == ["audio"]
    assert streams[0]["codec_name"] == "flac"
    assert streams[0]["sample_rate"] == "48000"
    assert source.read_bytes() == before
    assert list(tmp_path.glob("*Transcript.docx"))
    assert list(tmp_path.glob("*Transcript.txt"))
    artifact = json.loads(next(tmp_path.glob("*.transcript.json")).read_text())
    assert artifact["source_media_path"] == str(source)
    assert artifact["working_media_path"] == str(source)
    recording_end = datetime.fromisoformat(artifact["recording_end"])
    assert recording_end.timestamp() == 1700000000
    assert list(work.iterdir()) == []
    for media in [source, uploaded]:
        decoded = subprocess.run(
            ["ffmpeg", "-v", "error", "-i", str(media), "-map", "0:a:0",
             "-f", "s16le", "-"], capture_output=True, check=True, timeout=15,
        ).stdout
        if media == source:
            expected = decoded
        else:
            assert decoded == expected


def test_audio_only_upload_is_unchanged_without_ffmpeg(tmp_path, monkeypatch):
    import wave

    source = tmp_path / "voice.wav"
    with wave.open(str(source), "wb") as audio:
        audio.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
        audio.writeframes(b"\0\0" * 8000)
    uploaded = tmp_path / "uploaded.wav"
    monkeypatch.setenv("PATH", "")
    monkeypatch.setattr(
        assembly.requests, "Session", lambda: LocalProvider(uploaded),
    )
    assert assembly.main([str(source), "--api-key", "fake", "--text-output"]) == 0
    assert uploaded.read_bytes() == source.read_bytes()


def test_video_without_audio_fails_without_upload(tmp_path, monkeypatch, capsys):
    source = tmp_path / "silent.mkv"
    make_video(source, audio=False)
    work = tmp_path / "work"
    work.mkdir()
    monkeypatch.setattr(assembly.tempfile, "tempdir", str(work))
    uploaded = tmp_path / "uploaded"
    monkeypatch.setattr(
        assembly.requests, "Session", lambda: LocalProvider(uploaded),
    )
    assert assembly.main([str(source), "--api-key", "fake"]) == 1
    assert "extraction failed" in capsys.readouterr().err
    assert not uploaded.exists()
    assert list(work.iterdir()) == []


@pytest.mark.parametrize(
    "failure", ["missing_tool", "timeout", "upload", "invalid_response"],
)
def test_video_failures_clean_up_and_return_backend_failure(
    tmp_path, monkeypatch, capsys, failure,
):
    source = tmp_path / "meeting.mkv"
    make_video(source)
    before = source.read_bytes()
    work = tmp_path / "work"
    work.mkdir()
    monkeypatch.setattr(assembly.tempfile, "tempdir", str(work))
    uploaded = tmp_path / "uploaded"

    class FailingProvider(LocalProvider):
        def post(self, url, *, data=None, json=None):
            if url.endswith("/upload"):
                if failure == "upload":
                    next(data)  # exercise cleanup after opening the derivative
                    raise assembly.requests.ConnectionError(
                        "synthetic upload failure"
                    )
                if failure == "invalid_response":
                    return ProviderResponse({})
            return super().post(url, data=data, json=json)

    monkeypatch.setattr(
        assembly.requests, "Session", lambda: FailingProvider(uploaded),
    )
    if failure == "missing_tool":
        monkeypatch.setenv("PATH", "")
        expected = "ffmpeg is required"
    elif failure == "timeout":
        # A controlled external executable makes the timeout deterministic.
        tools = tmp_path / "tools"
        tools.mkdir()
        executable = tools / "ffmpeg"
        executable.write_text("#!/bin/sh\nexec /bin/sleep 5\n")
        executable.chmod(0o700)
        monkeypatch.setenv("PATH", str(tools))
        monkeypatch.setattr(assembly, "AUDIO_EXTRACTION_TIMEOUT_SECONDS", 0.05)
        expected = "timed out"
    elif failure == "upload":
        expected = "synthetic upload failure"
    else:
        expected = "missing 'upload_url'"
    assert assembly.main([str(source), "--api-key", "fake"]) == 1
    assert expected in capsys.readouterr().err
    assert source.read_bytes() == before
    assert not uploaded.exists()
    assert list(work.iterdir()) == []


def test_calendar_association_and_rename_use_original_recording(tmp_path, monkeypatch):
    from datetime import timedelta

    import transcribe_common

    source = tmp_path / "meeting.mkv"
    make_video(source)
    before = source.read_bytes()
    os.utime(source, (1700000000, 1700000000))
    end = datetime.fromtimestamp(1700000000).astimezone()
    start = end - timedelta(seconds=0.5)
    event = {"id": "synthetic-event", "summary": "Synthetic meeting",
             "start": {"dateTime": start.isoformat()},
             "end": {"dateTime": end.isoformat()}}

    def calendar_lookup(service, calendar_id, recording_start, recording_end, *args):
        assert recording_start == start
        assert recording_end == end
        return [{"event": event, "start": start, "end": end}], None, []

    monkeypatch.setattr(transcribe_common, "find_matching_events", calendar_lookup)
    uploaded = tmp_path / "uploaded.flac"
    monkeypatch.setattr(
        assembly.requests, "Session", lambda: LocalProvider(uploaded),
    )
    args = assembly.parse_args([str(source), "--use-calendar", "--text-output"])
    assert assembly.process_audio_file(
        source, args, "fake", ("en_us", False), object(),
    )
    artifact = json.loads(next(tmp_path.glob("*.transcript.json")).read_text())
    assert artifact["event"]["summary"] == "Synthetic meeting"
    assert artifact["source_media_path"] == str(source)
    working = Path(artifact["working_media_path"])
    assert "Synthetic meeting" in working.name
    assert working.read_bytes() == before
    assert not source.exists()  # existing calendar rename retains video bytes
