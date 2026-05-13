from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import transcript_store


def write_json(path: Path, payload: dict) -> Path:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def transcript_payload() -> dict:
    return {
        "transcript_title": "SoyLei Tempo Transcript",
        "backend": "assembly",
        "recording_start": "2026-05-06T13:15:00-05:00",
        "duration_seconds": 120,
        "event": {
            "summary": "SoyLei and Tempo Chemical Technical discussion",
            "participants": ["eric@soylei.com", "paul@tempochem.com"],
            "matching_calendars": [
                {"calendar_summary": "SoyLei", "event_summary": "Tempo Chemical technical discussion"}
            ],
        },
        "transcript_text": "Tempo Chemical discussed concrete sealer and SoyLei samples.",
    }


def timestamped_transcript_payload() -> dict:
    payload = transcript_payload()
    payload["source_media_path"] = "/tmp/source-call.m4a"
    payload["working_media_path"] = "/tmp/trimmed-call.m4a"
    payload["utterances"] = [
        {"speaker": "A", "start": 0, "end": 1500, "text": "Opening notes."},
        {"speaker": "B", "start": 1500, "end": 3200, "text": "Rare catalyst procurement detail."},
        {"speaker": "A", "start": 3200, "end": 5000, "text": "Closing notes."},
    ]
    payload["transcript_text"] = "\n".join(
        transcript_store.formatted_utterance_text(utterance) for utterance in payload["utterances"]
    )
    return payload


def legacy_transcript_payload(title: str = "Legacy Client Call") -> dict:
    return {
        "schema_version": 1,
        "transcript_title": title,
        "backend": "legacy-import",
        "recording_start": "2026-05-06T13:15:00-05:00",
        "duration_seconds": 0,
        "transcript_text": "Legacy transcript text.",
        "legacy_import": {
            "source_path": f"/tmp/{title} Transcript.docx",
            "source_sha256": "abc123",
            "needs_enrichment": True,
        },
    }


def readout_payload() -> dict:
    return {
        "title": "SoyLei Tempo Readout",
        "summary": "Tempo Chemical discussed sample evaluation.",
        "topics": ["concrete sealer", "technical samples"],
        "matter_candidates": [
            {"label": "SoyLei Tempo matter", "confidence": 0.92, "evidence": "calendar and transcript"}
        ],
        "memory_candidates": [
            {"text": "Tempo Chemical is evaluating SoyLei samples.", "kind": "matter_fact", "evidence": "readout"}
        ],
    }


def contextual_readout_payload() -> dict:
    return {
        **readout_payload(),
        "title": "SoyLei Tempo Contextual Readout",
        "provider": {"name": "codex-exec", "mode": "contextual_reread"},
        "contextualization": {
            "supporting_context_sources": [
                {"source_type": "odollo_contact", "source_id": "contact-1", "label": "Tempo Chemical"}
            ]
        },
    }


def test_store_init_creates_database(tmp_path: Path) -> None:
    with transcript_store.connect(tmp_path) as con:
        transcript_store.init_db(con)

    assert (tmp_path / "transcripts.sqlite3").exists()


def test_ollama_embedding_provider_uses_api_embed(monkeypatch) -> None:
    captured = {}

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self):
            return json.dumps({"embeddings": [[3.0, 4.0]]}).encode("utf-8")

    def fake_urlopen(request, timeout):
        captured["url"] = request.full_url
        captured["payload"] = json.loads(request.data.decode("utf-8"))
        captured["timeout"] = timeout
        return FakeResponse()

    monkeypatch.setattr(transcript_store, "urlopen", fake_urlopen)

    vector = transcript_store.embedding_for_text(
        "Tempo Chemical",
        provider="ollama",
        model_name="ollama/nomic-embed-text",
    )

    assert captured["url"] == "http://127.0.0.1:11434/api/embed"
    assert captured["payload"] == {"model": "nomic-embed-text", "input": "search_document: Tempo Chemical"}
    assert vector == [0.6, 0.8]


def test_ollama_nomic_query_embedding_uses_query_prefix(monkeypatch) -> None:
    captured = {}

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self):
            return json.dumps({"embedding": [1.0, 0.0]}).encode("utf-8")

    def fake_urlopen(request, timeout):
        captured["payload"] = json.loads(request.data.decode("utf-8"))
        return FakeResponse()

    monkeypatch.setattr(transcript_store, "urlopen", fake_urlopen)

    transcript_store.embedding_for_text(
        "Tempo Chemical",
        provider="ollama",
        model_name="ollama/nomic-embed-text",
        purpose="query",
    )

    assert captured["payload"] == {"model": "nomic-embed-text", "input": "search_query: Tempo Chemical"}


def test_long_document_embedding_is_chunked(monkeypatch) -> None:
    payloads = []

    class FakeResponse:
        def __init__(self, vector):
            self.vector = vector

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self):
            return json.dumps({"embedding": self.vector}).encode("utf-8")

    def fake_urlopen(request, timeout):
        payload = json.loads(request.data.decode("utf-8"))
        payloads.append(payload)
        return FakeResponse([1.0, 0.0] if len(payloads) == 1 else [0.0, 1.0])

    monkeypatch.setattr(transcript_store, "urlopen", fake_urlopen)

    vector = transcript_store.embedding_for_text(
        "Tempo Chemical. " * 600,
        provider="ollama",
        model_name="ollama/nomic-embed-text",
        purpose="document",
    )

    assert len(payloads) > 1
    assert all(payload["input"].startswith("search_document: ") for payload in payloads)
    assert vector[0] > 0
    assert vector[1] > 0


def test_ingest_artifacts_copies_files_and_indexes_kinds(tmp_path: Path) -> None:
    transcript_path = write_json(tmp_path / "meeting.transcript.json", transcript_payload())
    readout_path = write_json(tmp_path / "meeting.readout.json", readout_payload())
    contextual_path = write_json(tmp_path / "meeting.contextual.readout.json", contextual_readout_payload())
    store_root = tmp_path / "store"

    results = [
        transcript_store.ingest_artifact(transcript_path, root=store_root, embedding_provider="debug-hash"),
        transcript_store.ingest_artifact(readout_path, root=store_root, embedding_provider="debug-hash"),
        transcript_store.ingest_artifact(contextual_path, root=store_root, embedding_provider="debug-hash"),
    ]

    assert [result.kind for result in results] == ["transcript", "readout", "contextual_readout"]
    assert all(Path(result.stored_path).exists() for result in results)

    with sqlite3.connect(store_root / "transcripts.sqlite3") as con:
        rows = con.execute("SELECT kind, title FROM documents ORDER BY kind").fetchall()

    assert ("transcript", "SoyLei Tempo Transcript") in rows
    assert ("readout", "SoyLei Tempo Readout") in rows
    assert ("contextual_readout", "SoyLei Tempo Contextual Readout") in rows


def test_ingest_artifact_persists_chunk_embeddings(tmp_path: Path) -> None:
    payload = transcript_payload()
    payload["transcript_text"] = "Opening notes. " * 150 + "Rare catalyst procurement detail. " + "Closing notes. " * 150
    artifact = write_json(tmp_path / "meeting.transcript.json", payload)
    store_root = tmp_path / "store"

    transcript_store.ingest_artifact(artifact, root=store_root, embedding_provider="debug-hash")

    with sqlite3.connect(store_root / "transcripts.sqlite3") as con:
        chunk_count = con.execute("SELECT COUNT(*) FROM document_chunks").fetchone()[0]

    assert chunk_count > 1


def test_ingest_transcript_chunk_metadata_includes_timestamps_and_speakers(tmp_path: Path) -> None:
    artifact = write_json(tmp_path / "meeting.transcript.json", timestamped_transcript_payload())
    store_root = tmp_path / "store"

    transcript_store.ingest_artifact(artifact, root=store_root, embedding_provider="debug-hash")

    with sqlite3.connect(store_root / "transcripts.sqlite3") as con:
        row = con.execute("SELECT metadata_json FROM document_chunks ORDER BY chunk_index LIMIT 1").fetchone()
    metadata = json.loads(row[0])

    assert metadata["start_seconds"] == 0.0
    assert metadata["end_seconds"] == 5.0
    assert metadata["speakers"] == ["A", "B"]
    assert metadata["utterance_count"] == 3


def test_store_search_finds_lexical_and_semantic_matches(tmp_path: Path) -> None:
    store_root = tmp_path / "store"
    transcript_store.ingest_artifact(
        write_json(tmp_path / "meeting.transcript.json", transcript_payload()),
        root=store_root,
        embedding_provider="debug-hash",
    )
    transcript_store.ingest_artifact(
        write_json(tmp_path / "meeting.contextual.readout.json", contextual_readout_payload()),
        root=store_root,
        embedding_provider="debug-hash",
    )

    lexical = transcript_store.search_store(
        "concrete sealer Tempo", root=store_root, limit=3, embedding_provider="debug-hash"
    )
    semantic = transcript_store.search_store(
        "sample evaluation chemical partner", root=store_root, limit=3, embedding_provider="debug-hash"
    )
    contextual = transcript_store.search_store(
        "Tempo Chemical", root=store_root, kind="contextual_readout", limit=3, embedding_provider="debug-hash"
    )

    assert lexical
    assert lexical[0]["score"] > 0
    assert semantic
    assert contextual
    assert all(item["kind"] == "contextual_readout" for item in contextual)


def test_store_search_returns_best_chunk_match(tmp_path: Path) -> None:
    payload = timestamped_transcript_payload()
    store_root = tmp_path / "store"
    transcript_store.ingest_artifact(
        write_json(tmp_path / "meeting.transcript.json", payload),
        root=store_root,
        embedding_provider="debug-hash",
    )

    results = transcript_store.search_store(
        "rare catalyst procurement",
        root=store_root,
        embedding_provider="debug-hash",
    )

    assert results
    assert results[0]["best_chunk"] is not None
    assert "Rare catalyst procurement" in results[0]["best_chunk"]["snippet"]
    assert results[0]["best_chunk"]["start_seconds"] == 0.0
    assert results[0]["best_chunk"]["end_seconds"] == 5.0
    assert results[0]["best_chunk"]["speakers"] == ["A", "B"]


def test_context_for_document_returns_timestamped_media_context(tmp_path: Path) -> None:
    store_root = tmp_path / "store"
    result = transcript_store.ingest_artifact(
        write_json(tmp_path / "meeting.transcript.json", timestamped_transcript_payload()),
        root=store_root,
        embedding_provider="debug-hash",
    )

    context = transcript_store.context_for_document(
        result.id,
        root=store_root,
        chunk_index=0,
        context_chunks=1,
        embedding_provider="debug-hash",
    )

    assert context["document"]["id"] == result.id
    assert context["chunk"]["start_seconds"] == 0.0
    assert context["chunk"]["end_seconds"] == 5.0
    assert context["chunk"]["speakers"] == ["A", "B"]
    assert context["media"]["path"] == "/tmp/trimmed-call.m4a"
    assert context["media"]["start_timestamp"] == "00:00.00"
    assert context["media"]["seek_hint"] == "ffplay -ss 0.00 '/tmp/trimmed-call.m4a'"
    assert "Rare catalyst procurement" in context["context_chunks"][0]["text"]


def test_cli_ingest_and_search_outputs_json(tmp_path: Path, capsys) -> None:
    store_root = tmp_path / "store"
    artifact = write_json(tmp_path / "meeting.readout.json", readout_payload())

    assert transcript_store.main(
        ["--store-dir", str(store_root), "ingest", str(artifact), "--embedding-provider", "debug-hash"]
    ) == 0
    ingest_stdout = capsys.readouterr().out
    assert transcript_store.STORE_JSON_STDOUT_PREFIX in ingest_stdout

    assert transcript_store.main(
        ["--store-dir", str(store_root), "search", "Tempo samples", "--embedding-provider", "debug-hash"]
    ) == 0
    search_stdout = capsys.readouterr().out
    assert transcript_store.SEARCH_JSON_STDOUT_PREFIX in search_stdout
    assert "SoyLei Tempo Readout" in search_stdout


def test_cli_context_outputs_text_and_json_prefix(tmp_path: Path, capsys) -> None:
    store_root = tmp_path / "store"
    result = transcript_store.ingest_artifact(
        write_json(tmp_path / "meeting.transcript.json", timestamped_transcript_payload()),
        root=store_root,
        embedding_provider="debug-hash",
    )

    assert transcript_store.main(
        [
            "--store-dir",
            str(store_root),
            "context",
            result.id,
            "--chunk-index",
            "0",
            "--embedding-provider",
            "debug-hash",
        ]
    ) == 0

    stdout = capsys.readouterr().out
    assert "Media seek: ffplay -ss 0.00 '/tmp/trimmed-call.m4a'" in stdout
    assert "> chunk 0 [00:00.00 - 00:05.00]" in stdout
    assert "Rare catalyst procurement" in stdout
    assert transcript_store.CONTEXT_JSON_STDOUT_PREFIX in stdout


def test_cli_context_compact_json_outputs_pure_json(tmp_path: Path, capsys) -> None:
    store_root = tmp_path / "store"
    result = transcript_store.ingest_artifact(
        write_json(tmp_path / "meeting.transcript.json", timestamped_transcript_payload()),
        root=store_root,
        embedding_provider="debug-hash",
    )

    assert transcript_store.main(
        [
            "--store-dir",
            str(store_root),
            "context",
            result.id,
            "--chunk-index",
            "0",
            "--format",
            "compact-json",
            "--embedding-provider",
            "debug-hash",
        ]
    ) == 0

    stdout = capsys.readouterr().out
    payload = json.loads(stdout)
    assert payload["document"]["id"] == result.id
    assert payload["media"]["seek_hint"] == "ffplay -ss 0.00 '/tmp/trimmed-call.m4a'"
    assert transcript_store.CONTEXT_JSON_STDOUT_PREFIX not in stdout


def test_cli_search_context_opens_best_chunk(tmp_path: Path, capsys) -> None:
    store_root = tmp_path / "store"
    transcript_store.ingest_artifact(
        write_json(tmp_path / "meeting.transcript.json", timestamped_transcript_payload()),
        root=store_root,
        embedding_provider="debug-hash",
    )

    assert transcript_store.main(
        [
            "--store-dir",
            str(store_root),
            "search",
            "rare catalyst procurement",
            "--context",
            "--embedding-provider",
            "debug-hash",
        ]
    ) == 0

    stdout = capsys.readouterr().out
    assert stdout.startswith("Search hit 1/")
    assert "Media seek: ffplay -ss 0.00 '/tmp/trimmed-call.m4a'" in stdout
    assert "> chunk 0 [00:00.00 - 00:05.00]" in stdout
    assert "Rare catalyst procurement" in stdout
    assert transcript_store.CONTEXT_JSON_STDOUT_PREFIX in stdout
    assert transcript_store.SEARCH_JSON_STDOUT_PREFIX not in stdout


def test_cli_search_context_compact_json_includes_search_metadata(tmp_path: Path, capsys) -> None:
    store_root = tmp_path / "store"
    result = transcript_store.ingest_artifact(
        write_json(tmp_path / "meeting.transcript.json", timestamped_transcript_payload()),
        root=store_root,
        embedding_provider="debug-hash",
    )

    assert transcript_store.main(
        [
            "--store-dir",
            str(store_root),
            "search",
            "rare catalyst procurement",
            "--context",
            "--context-format",
            "compact-json",
            "--embedding-provider",
            "debug-hash",
        ]
    ) == 0

    stdout = capsys.readouterr().out
    payload = json.loads(stdout)
    assert payload["query"] == "rare catalyst procurement"
    assert payload["selected_rank"] == 1
    assert payload["selected_result"]["id"] == result.id
    assert payload["selected_result"]["best_chunk"]["chunk_index"] == 0
    assert payload["context"]["media"]["seek_hint"] == "ffplay -ss 0.00 '/tmp/trimmed-call.m4a'"
    assert transcript_store.CONTEXT_JSON_STDOUT_PREFIX not in stdout


def test_legacy_enrichment_queue_lists_pending_legacy_imports(tmp_path: Path, capsys) -> None:
    store_root = tmp_path / "store"
    transcript_result = transcript_store.ingest_artifact(
        write_json(tmp_path / "legacy.transcript.json", legacy_transcript_payload()),
        root=store_root,
        embedding_provider="debug-hash",
    )
    transcript_store.ingest_artifact(
        write_json(tmp_path / "legacy-copy.transcript.json", legacy_transcript_payload()),
        root=store_root,
        embedding_provider="debug-hash",
    )
    transcript_store.ingest_artifact(
        write_json(
            tmp_path / "done.transcript.json",
            legacy_transcript_payload("Legacy Done Call"),
        ),
        root=store_root,
        embedding_provider="debug-hash",
    )
    transcript_store.ingest_artifact(
        write_json(
            tmp_path / "done.readout.json",
            {
                **readout_payload(),
                "source_artifact_path": str((tmp_path / "done.transcript.json").resolve()),
            },
        ),
        root=store_root,
        embedding_provider="debug-hash",
    )

    queue = transcript_store.legacy_enrichment_queue(
        root=store_root,
        provider="codex-exec",
        model="gpt-5.4",
    )

    assert queue["selected_count"] == 1
    assert queue["duplicate_count"] == 1
    assert queue["items"][0]["id"] == transcript_result.id
    assert queue["items"][0]["pending_first_pass_readout"] is True
    assert queue["items"][0]["command"] == [
        "python",
        "summarize_transcript.py",
        str((tmp_path / "legacy.transcript.json").resolve()),
        "--provider",
        "codex-exec",
        "--model",
        "gpt-5.4",
        "--store",
    ]

    assert transcript_store.main(
        [
            "--store-dir",
            str(store_root),
            "legacy-enrichment-queue",
            "--format",
            "commands",
            "--provider",
            "codex-exec",
            "--limit",
            "1",
        ]
    ) == 0
    stdout = capsys.readouterr().out
    assert "summarize_transcript.py" in stdout
    assert "--provider" in stdout
    assert transcript_store.LEGACY_ENRICHMENT_QUEUE_JSON_STDOUT_PREFIX not in stdout


def test_backfill_dry_run_reports_counts_and_kinds(tmp_path: Path, capsys) -> None:
    artifacts_dir = tmp_path / "artifacts"
    artifacts_dir.mkdir()
    write_json(artifacts_dir / "meeting.transcript.json", transcript_payload())
    write_json(artifacts_dir / "meeting.readout.json", readout_payload())
    write_json(artifacts_dir / "notes.json", {"ignored": True})

    assert transcript_store.main(
        [
            "--store-dir",
            str(tmp_path / "store"),
            "backfill",
            str(artifacts_dir),
            "--dry-run",
            "--embedding-provider",
            "debug-hash",
        ]
    ) == 0

    stdout = capsys.readouterr().out
    payload = json.loads(stdout.split(transcript_store.BACKFILL_JSON_STDOUT_PREFIX)[0])
    assert payload["dry_run"] is True
    assert payload["selected_count"] == 2
    assert payload["by_kind"] == {"readout": 1, "transcript": 1}
    assert payload["by_status"] == {"insert": 2}


def test_backfill_skips_current_artifacts(tmp_path: Path, capsys) -> None:
    artifacts_dir = tmp_path / "artifacts"
    artifacts_dir.mkdir()
    artifact = write_json(artifacts_dir / "meeting.transcript.json", transcript_payload())
    store_root = tmp_path / "store"
    transcript_store.ingest_artifact(artifact, root=store_root, embedding_provider="debug-hash")

    assert transcript_store.main(
        [
            "--store-dir",
            str(store_root),
            "backfill",
            str(artifacts_dir),
            "--embedding-provider",
            "debug-hash",
        ]
    ) == 0

    stdout = capsys.readouterr().out
    payload = json.loads(stdout.split(transcript_store.BACKFILL_JSON_STDOUT_PREFIX)[0])
    assert payload["selected_count"] == 1
    assert payload["by_status"] == {"skip": 1}
    assert payload["items"][0]["result"] == "skipped"


def test_backfill_kind_filter_and_limit(tmp_path: Path) -> None:
    artifacts_dir = tmp_path / "artifacts"
    artifacts_dir.mkdir()
    write_json(artifacts_dir / "a.transcript.json", transcript_payload())
    write_json(artifacts_dir / "b.readout.json", readout_payload())
    write_json(artifacts_dir / "c.contextual.readout.json", contextual_readout_payload())

    planned = transcript_store.plan_backfill(
        [artifacts_dir],
        root=tmp_path / "store",
        kinds={"readout", "contextual_readout"},
        limit=1,
        embedding_provider="debug-hash",
    )

    assert len(planned) == 1
    assert planned[0]["kind"] in {"readout", "contextual_readout"}


def test_backfill_reports_invalid_json_without_aborting(tmp_path: Path, capsys) -> None:
    artifacts_dir = tmp_path / "artifacts"
    artifacts_dir.mkdir()
    (artifacts_dir / "bad.readout.json").write_text("not-json", encoding="utf-8")

    assert transcript_store.main(
        [
            "--store-dir",
            str(tmp_path / "store"),
            "backfill",
            str(artifacts_dir),
            "--dry-run",
            "--embedding-provider",
            "debug-hash",
        ]
    ) == 0

    stdout = capsys.readouterr().out
    payload = json.loads(stdout.split(transcript_store.BACKFILL_JSON_STDOUT_PREFIX)[0])
    assert payload["selected_count"] == 1
    assert payload["by_status"] == {"error": 1}
    assert "not valid JSON" in payload["items"][0]["error"]


def test_backfill_excludes_store_artifacts_by_default(tmp_path: Path) -> None:
    pytest_dir = tmp_path / "pytest-of-ecochran76" / "pytest-1"
    copied_store_dir = tmp_path / "store" / "artifacts" / "aa"
    valid_dir = tmp_path / "valid"
    pytest_dir.mkdir(parents=True)
    copied_store_dir.mkdir(parents=True)
    valid_dir.mkdir()
    write_json(pytest_dir / "fixture.readout.json", readout_payload())
    write_json(copied_store_dir / "copied.readout.json", readout_payload())
    valid = write_json(valid_dir / "meeting.readout.json", readout_payload())

    candidates = transcript_store.iter_backfill_candidates([tmp_path], recursive=True)

    assert candidates == [
        (pytest_dir / "fixture.readout.json").resolve(),
        valid.resolve(),
    ]


def test_backfill_accepts_explicit_exclude_patterns(tmp_path: Path) -> None:
    excluded_dir = tmp_path / "excluded-fixtures"
    copied_store_dir = tmp_path / "store" / "artifacts" / "aa"
    valid_dir = tmp_path / "valid"
    excluded_dir.mkdir(parents=True)
    copied_store_dir.mkdir(parents=True)
    valid_dir.mkdir()
    write_json(excluded_dir / "fixture.readout.json", readout_payload())
    write_json(copied_store_dir / "copied.readout.json", readout_payload())
    valid = write_json(valid_dir / "meeting.readout.json", readout_payload())

    planned = transcript_store.plan_backfill(
        [tmp_path],
        root=tmp_path / "store-db",
        recursive=True,
        exclude_patterns=(*transcript_store.DEFAULT_BACKFILL_EXCLUDE_PATTERNS, "*/excluded-fixtures/*"),
        embedding_provider="debug-hash",
    )

    assert [item["source_path"] for item in planned] == [str(valid.resolve())]
