from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from transcribe_common import (
    CalendarProvider,
    attach_matching_calendars,
    build_event_base_name,
    build_gog_calendar_list_command,
    build_gog_calendar_events_command,
    build_gws_calendar_env,
    build_gws_calendar_list_command,
    build_gws_calendar_events_command,
    ensure_selected_calendar_context,
    extract_calendars_from_provider_payload,
    parse_calendar_provider_order,
    process_transcription_outputs,
)
from watch_transcriptions import (
    JobState,
    ProcessedRecord,
    WatchJob,
    extract_artifact_paths,
    ingest_store_artifacts,
    load_jobs,
    load_state,
    parse_args,
    save_state,
)


def base_args(tmp_path: Path) -> argparse.Namespace:
    return argparse.Namespace(
        translate_to=None,
        print_key_sources=False,
        use_calendar=False,
        calendar_id="primary",
        calendar_window=24.0,
        output_dir=tmp_path,
        srt_output=False,
        docx_output=False,
        text_output=True,
        embed_subtitles=False,
    )


def test_process_transcription_outputs_writes_artifact_json(tmp_path: Path) -> None:
    audio_path = tmp_path / "meeting.m4a"
    audio_path.write_bytes(b"placeholder")
    utterances = [
        {
            "speaker": "A",
            "start": 0,
            "end": 1250,
            "text": "Hello from the meeting.",
        }
    ]

    ok = process_transcription_outputs(
        audio_path,
        utterances,
        1.25,
        base_args(tmp_path),
        None,
        docx_title="Test Transcript",
        backend_name="test_backend",
    )

    assert ok is True
    artifact_path = tmp_path / "meeting Transcript.transcript.json"
    assert artifact_path.exists()
    payload = json.loads(artifact_path.read_text(encoding="utf-8"))

    assert payload["schema_version"] == 1
    assert payload["source_media_path"] == str(audio_path)
    assert payload["working_media_path"] == str(audio_path)
    assert payload["backend"] == "test_backend"
    assert payload["duration_seconds"] == 1.25
    assert payload["utterance_count"] == 1
    assert payload["utterances"][0]["text"] == "Hello from the meeting."
    assert "Hello from the meeting." in payload["transcript_text"]
    assert payload["transcript_window_start_seconds"] == 0.0
    assert payload["transcript_window_end_seconds"] == 1.25
    assert payload["output_paths"]["artifact"] == str(artifact_path)
    assert payload["output_paths"]["docx"].endswith("meeting Transcript.docx")
    assert payload["output_paths"]["txt"].endswith("meeting Transcript.txt")


def test_event_base_name_does_not_duplicate_existing_calendar_prefix() -> None:
    event_time = datetime(2026, 5, 13, 13, 0).astimezone()

    assert (
        build_event_base_name(
            event_time,
            "Kiddie training and 1 other(s)",
            "2026-05-13 13-00 Kiddie training and 1 other(s) My recording 129",
        )
        == "2026-05-13 13-00 Kiddie training and 1 other(s) My recording 129"
    )
    assert (
        build_event_base_name(
            event_time,
            "Kiddie training",
            "2026-05-13 13-00 Kiddie training and 1 other(s) 2026-05-13 13-00 Kiddie training and 1 other(s) My recording 129",
        )
        == "2026-05-13 13-00 Kiddie training My recording 129"
    )


def test_extract_artifact_paths_from_backend_stdout() -> None:
    stdout = "\n".join(
        [
            "Uploading file...",
            "TRANSCRIPT_ARTIFACT_JSON=/tmp/a.transcript.json",
            "TRANSCRIPT_ARTIFACT_JSON=/tmp/a.transcript.json",
            "TRANSCRIPT_ARTIFACT_JSON=/tmp/b.transcript.json",
            "Completed successfully.",
        ]
    )

    assert extract_artifact_paths(stdout) == [
        "/tmp/a.transcript.json",
        "/tmp/b.transcript.json",
    ]


def test_watcher_state_preserves_artifact_paths(tmp_path: Path) -> None:
    state_path = tmp_path / "state.json"
    job = WatchJob(
        name="downloads",
        watch_dir=tmp_path,
        glob="*.m4a",
        backends=["assembly"],
        recursive=False,
        settle_seconds=120,
        min_age_seconds=20,
        scan_interval=30,
        failure_retry_seconds=900,
        cli_args={"assembly": []},
        notify_on_success=False,
        notify_on_failure=False,
        slack_channel=None,
    )
    artifact_path = str(tmp_path / "meeting Transcript.transcript.json")
    save_state(
        state_path,
        {
            job.name: JobState(
                processed={
                    str(tmp_path / "meeting.m4a"): ProcessedRecord(
                        status="success",
                        completed_at=1.0,
                        size=123,
                        mtime=2.0,
                        fingerprint="abc",
                        command=["python", "assembly_transcribe.py"],
                        returncode=0,
                        backend="assembly",
                        attempted_backends=["assembly"],
                        artifact_paths=[artifact_path],
                    )
                },
                candidates={},
            )
        },
    )

    loaded = load_state(state_path, [job])
    record = next(iter(loaded[job.name].processed.values()))
    assert record.artifact_paths == [artifact_path]


def test_watcher_state_preserves_store_paths(tmp_path: Path) -> None:
    state_path = tmp_path / "state.json"
    job = WatchJob(
        name="downloads",
        watch_dir=tmp_path,
        glob="*.m4a",
        backends=["assembly"],
        recursive=False,
        settle_seconds=120,
        min_age_seconds=20,
        scan_interval=30,
        failure_retry_seconds=900,
        cli_args={"assembly": []},
        notify_on_success=False,
        notify_on_failure=False,
        slack_channel=None,
    )
    store_path = str(tmp_path / "store" / "artifacts" / "meeting.transcript.json")
    save_state(
        state_path,
        {
            job.name: JobState(
                processed={
                    str(tmp_path / "meeting.m4a"): ProcessedRecord(
                        status="success",
                        completed_at=1.0,
                        size=123,
                        mtime=2.0,
                        fingerprint="abc",
                        command=["python", "assembly_transcribe.py"],
                        returncode=0,
                        backend="assembly",
                        attempted_backends=["assembly"],
                        artifact_paths=[str(tmp_path / "meeting Transcript.transcript.json")],
                        store_paths=[store_path],
                    )
                },
                candidates={},
            )
        },
    )

    loaded = load_state(state_path, [job])
    record = next(iter(loaded[job.name].processed.values()))
    assert record.store_paths == [store_path]


def test_watcher_store_config_expands_to_job_settings(tmp_path: Path) -> None:
    config_path = tmp_path / "watch.json"
    config_path.write_text(
        json.dumps(
            {
                "jobs": [
                    {
                        "name": "downloads",
                        "watch_dir": str(tmp_path),
                        "backend": "assembly",
                        "store": {
                            "enabled": True,
                            "store_dir": str(tmp_path / "store"),
                            "embedding_provider": "debug-hash",
                            "embedding_model": "debug-hash",
                        },
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    jobs = load_jobs(config_path, parse_args(["--config", str(config_path)]))

    assert jobs[0].store_enabled is True
    assert jobs[0].store_dir == tmp_path / "store"
    assert jobs[0].store_embedding_provider == "debug-hash"
    assert jobs[0].store_embedding_model == "debug-hash"


def test_watcher_store_ingest_uses_configured_provider(tmp_path: Path, monkeypatch) -> None:
    captured = {}

    class FakeResult:
        kind = "transcript"
        stored_path = str(tmp_path / "store" / "artifact.json")

    def fake_ingest_artifact(path, *, root, embedding_provider, embedding_model):
        captured["path"] = path
        captured["root"] = root
        captured["embedding_provider"] = embedding_provider
        captured["embedding_model"] = embedding_model
        return FakeResult()

    monkeypatch.setattr("watch_transcriptions.ingest_artifact", fake_ingest_artifact)

    job = WatchJob(
        name="downloads",
        watch_dir=tmp_path,
        glob="*.m4a",
        backends=["assembly"],
        recursive=False,
        settle_seconds=120,
        min_age_seconds=20,
        scan_interval=30,
        failure_retry_seconds=900,
        cli_args={"assembly": []},
        notify_on_success=False,
        notify_on_failure=False,
        slack_channel=None,
        store_enabled=True,
        store_dir=tmp_path / "store",
        store_embedding_provider="debug-hash",
        store_embedding_model="debug-hash",
    )

    store_paths = ingest_store_artifacts(job, [str(tmp_path / "meeting.transcript.json")])

    assert store_paths == [str(tmp_path / "store" / "artifact.json")]
    assert captured == {
        "path": tmp_path / "meeting.transcript.json",
        "root": tmp_path / "store",
        "embedding_provider": "debug-hash",
        "embedding_model": "debug-hash",
    }


def test_calendar_provider_order_parsing() -> None:
    assert parse_calendar_provider_order(None) == ["gog", "gws", "google-api"]
    assert parse_calendar_provider_order("google,gog,google-api") == ["google-api", "gog"]


def test_gog_calendar_command_includes_tenant_selectors() -> None:
    command = build_gog_calendar_events_command(
        "primary",
        time_min="2026-05-04T00:00:00Z",
        time_max="2026-05-05T00:00:00Z",
        provider=CalendarProvider(name="gog", account="me@example.com", client="work"),
    )

    assert command[:5] == ["gog", "--account", "me@example.com", "--client", "work"]
    assert command[5:8] == ["calendar", "events", "primary"]
    assert "--json" in command
    assert "--results-only" in command
    assert "--no-input" in command


def test_gog_calendar_list_command_includes_tenant_selectors() -> None:
    command = build_gog_calendar_list_command(
        CalendarProvider(name="gog", account="me@example.com", client="work")
    )

    assert command == [
        "gog",
        "--account",
        "me@example.com",
        "--client",
        "work",
        "calendar",
        "calendars",
        "--json",
        "--results-only",
        "--no-input",
    ]


def test_gws_calendar_command_and_env_include_config_dir(tmp_path: Path) -> None:
    provider = CalendarProvider(name="gws", config_dir=tmp_path / "gws-config")
    command = build_gws_calendar_events_command(
        "primary",
        time_min="2026-05-04T00:00:00Z",
        time_max="2026-05-05T00:00:00Z",
    )
    env = build_gws_calendar_env(provider)

    assert command[:4] == ["gws", "calendar", "events", "list"]
    assert "--params" in command
    assert '"calendarId":"primary"' in command[command.index("--params") + 1]
    assert env["GOOGLE_WORKSPACE_CLI_CONFIG_DIR"] == str(tmp_path / "gws-config")


def test_gws_calendar_list_command() -> None:
    command = build_gws_calendar_list_command()

    assert command[:4] == ["gws", "calendar", "calendarList", "list"]
    assert "--params" in command
    assert '"maxResults":250' in command[command.index("--params") + 1]


def test_extract_calendars_from_provider_payload() -> None:
    payload = {
        "items": [
            {"id": "primary", "summary": "Primary"},
            {"id": "team@example.com", "summaryOverride": "Team"},
        ]
    }

    assert extract_calendars_from_provider_payload(payload) == payload["items"]


def test_attach_matching_calendars_to_event_metadata() -> None:
    event_info = {"summary": "Meeting"}
    matching_calendars = [
        {
            "calendar_id": "primary",
            "calendar_summary": "Primary",
            "event_summary": "Meeting",
            "coverage": 1.0,
        }
    ]

    result = attach_matching_calendars(event_info, matching_calendars)

    assert result["matching_calendars"] == matching_calendars
    assert "matching_calendars" not in event_info


def test_selected_calendar_context_falls_back_to_primary_event() -> None:
    matching_events = [
        {
            "event": {"id": "evt1", "summary": "Meeting"},
            "start": "start",
            "end": "end",
            "overlap_seconds": 120.0,
            "coverage": 1.0,
        }
    ]

    result = ensure_selected_calendar_context(
        calendar_id="primary",
        matching_events=matching_events,
        best_event=None,
        matching_calendars=[],
    )

    assert result == [
        {
            "calendar_id": "primary",
            "calendar_summary": "primary",
            "accessRole": None,
            "event_id": "evt1",
            "event_summary": "Meeting",
            "event_start": "start",
            "event_end": "end",
            "overlap_seconds": 120.0,
            "coverage": 1.0,
        }
    ]


def test_watcher_calendar_config_expands_to_cli_args(tmp_path: Path) -> None:
    config_path = tmp_path / "watch.json"
    config_path.write_text(
        json.dumps(
            {
                "jobs": [
                    {
                        "name": "downloads",
                        "watch_dir": str(tmp_path),
                        "backend": "assembly",
                        "calendar": {
                            "providers": ["gog", "gws", "google-api"],
                            "calendar_id": "primary",
                            "window_hours": 8,
                            "gog": {"account": "me@example.com", "client": "work"},
                            "gws": {"config_dir": "~/.config/gws-work"},
                        },
                        "cli_args": ["--text-output"],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    jobs = load_jobs(config_path, parse_args(["--config", str(config_path)]))
    cli_args = jobs[0].cli_args["assembly"]

    assert cli_args == [
        "--use-calendar",
        "--calendar-providers",
        "gog,gws,google-api",
        "--calendar-id",
        "primary",
        "--calendar-window",
        "8",
        "--calendar-gog-account",
        "me@example.com",
        "--calendar-gog-client",
        "work",
        "--calendar-gws-config-dir",
        "~/.config/gws-work",
        "--text-output",
    ]
