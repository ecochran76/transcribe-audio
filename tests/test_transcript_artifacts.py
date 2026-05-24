from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from transcribe_common import (
    CalendarProvider,
    attach_matching_calendars,
    build_event_base_name,
    describe_matching_calendars,
    find_matching_calendars_for_provider,
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
    CandidateSnapshot,
    JobState,
    ProcessedRecord,
    WatchJob,
    check_watcher_readiness,
    extract_artifact_paths,
    fingerprint_for,
    format_blocked_summary,
    ingest_store_artifacts,
    load_jobs,
    load_state,
    parse_args,
    save_state,
    scan_job,
)
import watch_transcriptions


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


def test_watcher_state_preserves_candidate_blocked_reasons(tmp_path: Path) -> None:
    state_path = tmp_path / "state.json"
    media_path = tmp_path / "meeting.m4a"
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
    save_state(
        state_path,
        {
            job.name: JobState(
                processed={},
                candidates={
                    str(media_path): CandidateSnapshot(
                        size=123,
                        mtime=2.0,
                        seen_at=1.0,
                        blocked_kind="missing_tool",
                        blocked_reason="ffprobe not found on PATH",
                        blocked_since=1.0,
                    )
                },
            )
        },
    )

    loaded = load_state(state_path, [job])
    snapshot = loaded[job.name].candidates[str(media_path)]

    assert snapshot.blocked_kind == "missing_tool"
    assert snapshot.blocked_reason == "ffprobe not found on PATH"
    assert snapshot.blocked_since == 1.0


def test_scan_job_does_not_count_processed_files_as_queued(tmp_path: Path) -> None:
    media_path = tmp_path / "meeting.m4a"
    media_path.write_bytes(b"audio")
    media_stats = media_path.stat()
    size = int(media_stats.st_size)
    mtime = float(media_stats.st_mtime)
    job = WatchJob(
        name="downloads",
        watch_dir=tmp_path,
        glob="*.m4a",
        backends=["assembly"],
        recursive=False,
        settle_seconds=0,
        min_age_seconds=0,
        scan_interval=30,
        failure_retry_seconds=900,
        cli_args={"assembly": []},
        notify_on_success=False,
        notify_on_failure=False,
        slack_channel=None,
    )
    job_state = JobState(
        processed={
            str(media_path.resolve()): ProcessedRecord(
                status="success",
                completed_at=1.0,
                size=size,
                mtime=mtime,
                fingerprint=fingerprint_for(media_path.resolve(), size, mtime),
                command=["python", "assembly_transcribe.py"],
                returncode=0,
                backend="assembly",
                attempted_backends=["assembly"],
            )
        },
        candidates={},
    )

    changed, stats = scan_job(job, job_state, verbose=False)

    assert changed is False
    assert stats.candidate_count == 0
    assert stats.processed_attempts == 0


def test_scan_job_records_minimum_age_blocked_reason(tmp_path: Path) -> None:
    media_path = tmp_path / "meeting.m4a"
    media_path.write_bytes(b"audio")
    job = WatchJob(
        name="downloads",
        watch_dir=tmp_path,
        glob="*.m4a",
        backends=["assembly"],
        recursive=False,
        settle_seconds=0,
        min_age_seconds=3600,
        scan_interval=30,
        failure_retry_seconds=900,
        cli_args={"assembly": []},
        notify_on_success=False,
        notify_on_failure=False,
        slack_channel=None,
    )
    job_state = JobState(processed={}, candidates={})

    changed, stats = scan_job(job, job_state, verbose=False)
    snapshot = job_state.candidates[str(media_path.resolve())]

    assert changed is True
    assert stats.candidate_count == 1
    assert stats.blocked_reasons == {"minimum_age": 1}
    assert snapshot.blocked_kind == "minimum_age"
    assert "minimum age" in snapshot.blocked_reason


def test_scan_job_settling_reason_does_not_create_fake_progress(tmp_path: Path) -> None:
    media_path = tmp_path / "meeting.m4a"
    media_path.write_bytes(b"audio")
    job = WatchJob(
        name="downloads",
        watch_dir=tmp_path,
        glob="*.m4a",
        backends=["assembly"],
        recursive=False,
        settle_seconds=3600,
        min_age_seconds=0,
        scan_interval=30,
        failure_retry_seconds=900,
        cli_args={"assembly": []},
        notify_on_success=False,
        notify_on_failure=False,
        slack_channel=None,
    )
    job_state = JobState(processed={}, candidates={})

    first_changed, first_stats = scan_job(job, job_state, verbose=False)
    second_changed, second_stats = scan_job(job, job_state, verbose=False)

    assert first_changed is True
    assert first_stats.blocked_reasons == {"settling": 1}
    assert second_changed is False
    assert second_stats.blocked_reasons == {"settling": 1}


def test_scan_job_keeps_retry_backoff_visible_as_queued(tmp_path: Path) -> None:
    media_path = tmp_path / "meeting.m4a"
    media_path.write_bytes(b"audio")
    media_stats = media_path.stat()
    size = int(media_stats.st_size)
    mtime = float(media_stats.st_mtime)
    retry_after = time.time() + 3600
    job = WatchJob(
        name="downloads",
        watch_dir=tmp_path,
        glob="*.m4a",
        backends=["assembly"],
        recursive=False,
        settle_seconds=0,
        min_age_seconds=0,
        scan_interval=30,
        failure_retry_seconds=900,
        cli_args={"assembly": []},
        notify_on_success=False,
        notify_on_failure=False,
        slack_channel=None,
    )
    job_state = JobState(
        processed={
            str(media_path.resolve()): ProcessedRecord(
                status="failed",
                completed_at=time.time(),
                size=size,
                mtime=mtime,
                fingerprint=fingerprint_for(media_path.resolve(), size, mtime),
                command=["python", "assembly_transcribe.py"],
                returncode=1,
                backend="assembly",
                attempted_backends=["assembly"],
                next_retry_after=retry_after,
                failure_kind="auth_config_failed",
                failure_reason="missing API key",
            )
        },
        candidates={},
    )

    changed, stats = scan_job(job, job_state, verbose=False)
    snapshot = job_state.candidates[str(media_path.resolve())]

    assert changed is True
    assert stats.candidate_count == 1
    assert stats.processed_attempts == 0
    assert stats.blocked_reasons == {"retry_backoff": 1}
    assert snapshot.blocked_kind == "retry_backoff"
    assert "missing API key" in snapshot.blocked_reason


def test_scan_job_records_media_probe_blocked_reason(tmp_path: Path, monkeypatch) -> None:
    media_path = tmp_path / "meeting.m4a"
    media_path.write_bytes(b"audio")
    media_stats = media_path.stat()
    size = int(media_stats.st_size)
    mtime = float(media_stats.st_mtime)
    key = str(media_path.resolve())
    monkeypatch.setattr(
        watch_transcriptions,
        "probe_media_readiness",
        lambda path: (False, "ffprobe not found on PATH"),
    )
    job = WatchJob(
        name="downloads",
        watch_dir=tmp_path,
        glob="*.m4a",
        backends=["assembly"],
        recursive=False,
        settle_seconds=0,
        min_age_seconds=0,
        scan_interval=30,
        failure_retry_seconds=900,
        cli_args={"assembly": []},
        notify_on_success=False,
        notify_on_failure=False,
        slack_channel=None,
    )
    job_state = JobState(
        processed={},
        candidates={key: CandidateSnapshot(size=size, mtime=mtime, seen_at=0.0)},
    )

    changed, stats = scan_job(job, job_state, verbose=False)
    snapshot = job_state.candidates[key]

    assert changed is True
    assert stats.blocked_reasons == {"missing_tool": 1}
    assert snapshot.blocked_kind == "missing_tool"
    assert snapshot.blocked_reason == "ffprobe not found on PATH"


def test_watcher_readiness_reports_missing_ffprobe(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(
        watch_transcriptions.shutil,
        "which",
        lambda name: None if name == "ffprobe" else f"/usr/bin/{name}",
    )
    job = WatchJob(
        name="downloads",
        watch_dir=tmp_path,
        glob="*.m4a",
        backends=["assembly"],
        recursive=False,
        settle_seconds=0,
        min_age_seconds=0,
        scan_interval=30,
        failure_retry_seconds=900,
        cli_args={"assembly": []},
        notify_on_success=False,
        notify_on_failure=False,
        slack_channel=None,
    )

    issues = check_watcher_readiness([job])

    assert [issue.code for issue in issues if issue.severity == "error"] == ["missing_ffprobe"]


def test_blocked_summary_is_stable_for_heartbeat_logs() -> None:
    assert format_blocked_summary({}) == "none"
    assert format_blocked_summary({"settling": 2, "missing_tool": 1}) == "missing_tool=1,settling=2"


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


def test_matching_calendar_descriptions_include_attendees() -> None:
    matching_events = [
        {
            "event": {
                "id": "evt1",
                "summary": "Shared calendar meeting",
                "attendees": [
                    {"displayName": "Alice Example", "email": "alice@example.com"},
                    {"email": "declined@example.com", "responseStatus": "declined"},
                ],
            },
            "start": "start",
            "end": "end",
            "overlap_seconds": 120.0,
            "coverage": 1.0,
        }
    ]

    result = describe_matching_calendars(
        matching_events,
        {"id": "team@example.com", "summary": "Team", "accessRole": "reader"},
    )

    assert result[0]["attendees"] == ["Alice Example <alice@example.com>"]
    assert result[0]["attendee_emails"] == ["alice@example.com"]


def test_explicit_provenance_calendar_ids_are_scanned(monkeypatch) -> None:
    queried_calendar_ids = []

    def fake_list_calendars_for_provider(provider):
        assert provider.name == "gog"
        return [{"id": "primary", "summary": "Primary", "accessRole": "owner"}]

    def fake_list_events_for_provider(provider, calendar_id, *, time_min, time_max):
        queried_calendar_ids.append(calendar_id)
        if calendar_id == "shared@example.com":
            return [
                {
                    "id": "evt-shared",
                    "summary": "Shared Meeting",
                    "start": {"dateTime": "2026-05-22T15:30:00Z"},
                    "end": {"dateTime": "2026-05-22T16:00:00Z"},
                    "attendees": [{"email": "shared-attendee@example.com"}],
                }
            ]
        return []

    monkeypatch.setattr("transcribe_common.list_calendars_for_provider", fake_list_calendars_for_provider)
    monkeypatch.setattr("transcribe_common.list_events_for_provider", fake_list_events_for_provider)

    result = find_matching_calendars_for_provider(
        CalendarProvider(name="gog"),
        requested_calendar_id="primary",
        provenance_calendar_ids=["shared@example.com"],
        recording_start=datetime.fromisoformat("2026-05-22T15:30:00+00:00"),
        recording_end=datetime.fromisoformat("2026-05-22T16:00:00+00:00"),
        time_min="2026-05-22T15:00:00Z",
        time_max="2026-05-22T17:00:00Z",
    )

    assert queried_calendar_ids == ["primary", "shared@example.com"]
    assert result[0]["calendar_id"] == "shared@example.com"
    assert result[0]["attendee_emails"] == ["shared-attendee@example.com"]


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
                            "provenance_calendar_ids": ["shared@example.com", "team@example.com"],
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
        "--calendar-provenance-calendar-id",
        "shared@example.com",
        "--calendar-provenance-calendar-id",
        "team@example.com",
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
