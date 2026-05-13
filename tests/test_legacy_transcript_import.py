from __future__ import annotations

import json
import sys
from pathlib import Path

from docx import Document

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import legacy_transcript_import
import transcript_api


def write_docx(path: Path, paragraphs: list[str]) -> Path:
    document = Document()
    for paragraph in paragraphs:
        document.add_paragraph(paragraph)
    document.save(path)
    return path


def test_legacy_import_dry_run_plans_txt_and_media_match(tmp_path: Path) -> None:
    transcript = tmp_path / "Weekly Call Transcript.txt"
    transcript.write_text("Speaker A: legacy notes", encoding="utf-8")
    media = tmp_path / "Weekly Call.m4a"
    media.write_bytes(b"audio")

    planned = legacy_transcript_import.plan_legacy_import(
        [tmp_path],
        root=tmp_path / "store",
        media_roots=[tmp_path],
        embedding_provider="debug-hash",
        embedding_model="debug-hash",
    )

    assert len(planned) == 1
    assert planned[0]["status"] == "convert"
    assert planned[0]["title"] == "Weekly Call"
    assert planned[0]["media_path"] == str(media.resolve())
    assert planned[0]["artifact_path"].endswith(".transcript.json")


def test_legacy_import_apply_writes_sidecar_and_ingests_docx(tmp_path: Path) -> None:
    transcript = write_docx(tmp_path / "Board Meeting Transcript.docx", ["First paragraph.", "Second paragraph."])
    media = tmp_path / "Board Meeting.mp3"
    media.write_bytes(b"0123456789")
    store_root = tmp_path / "store"
    planned = legacy_transcript_import.plan_legacy_import(
        [tmp_path],
        root=store_root,
        media_roots=[tmp_path],
        embedding_provider="debug-hash",
        embedding_model="debug-hash",
    )

    results = legacy_transcript_import.apply_legacy_import(
        planned,
        root=store_root,
        embedding_provider="debug-hash",
        embedding_model="debug-hash",
    )
    artifact_payload = json.loads(Path(results[0]["artifact_path"]).read_text(encoding="utf-8"))
    documents = transcript_api.list_documents(root=store_root)
    document = transcript_api.get_document(documents["items"][0]["id"], root=store_root)

    assert results[0]["result"] == "inserted"
    assert artifact_payload["backend"] == "legacy-import"
    assert artifact_payload["legacy_import"]["needs_enrichment"] is True
    assert artifact_payload["source_media_path"] == str(media.resolve())
    assert "First paragraph.\nSecond paragraph." == artifact_payload["transcript_text"]
    assert document["media_blob"]["id"]
    assert transcript.exists()


def test_legacy_import_apply_reports_skip_after_current_ingest(tmp_path: Path) -> None:
    transcript = tmp_path / "Sales Call Transcript.txt"
    transcript.write_text("Legacy sales transcript", encoding="utf-8")
    store_root = tmp_path / "store"
    planned = legacy_transcript_import.plan_legacy_import(
        [tmp_path],
        root=store_root,
        embedding_provider="debug-hash",
        embedding_model="debug-hash",
    )
    legacy_transcript_import.apply_legacy_import(
        planned,
        root=store_root,
        embedding_provider="debug-hash",
        embedding_model="debug-hash",
    )

    planned_again = legacy_transcript_import.plan_legacy_import(
        [tmp_path],
        root=store_root,
        embedding_provider="debug-hash",
        embedding_model="debug-hash",
    )

    assert planned_again[0]["status"] == "duplicate_existing"


def test_legacy_import_uses_media_index_file(tmp_path: Path) -> None:
    transcript = tmp_path / "Indexed Call Transcript.txt"
    transcript.write_text("Indexed transcript", encoding="utf-8")
    media = tmp_path / "Indexed Call.wav"
    media.write_bytes(b"audio")
    media_index = tmp_path / "media-index.txt"
    media_index.write_text(str(media), encoding="utf-8")

    planned = legacy_transcript_import.plan_legacy_import(
        [transcript],
        root=tmp_path / "store",
        media_index_files=[media_index],
        embedding_provider="debug-hash",
        embedding_model="debug-hash",
    )

    assert planned[0]["media_path"] == str(media.resolve())


def test_legacy_import_can_skip_media_matching(tmp_path: Path) -> None:
    transcript = tmp_path / "No Media Match Transcript.txt"
    transcript.write_text("Transcript", encoding="utf-8")
    media = tmp_path / "No Media Match.wav"
    media.write_bytes(b"audio")

    planned = legacy_transcript_import.plan_legacy_import(
        [transcript],
        root=tmp_path / "store",
        no_media_match=True,
        embedding_provider="debug-hash",
        embedding_model="debug-hash",
    )

    assert planned[0]["media_path"] == ""


def test_legacy_import_dedupes_existing_title(tmp_path: Path) -> None:
    store_root = tmp_path / "store"
    first = tmp_path / "Client Call Transcript.txt"
    first.write_text("First transcript", encoding="utf-8")
    second_dir = tmp_path / "copies"
    second_dir.mkdir()
    second = second_dir / "Client Call Transcript.txt"
    second.write_text("Second transcript with different bytes", encoding="utf-8")
    planned = legacy_transcript_import.plan_legacy_import(
        [first],
        root=store_root,
        embedding_provider="debug-hash",
        embedding_model="debug-hash",
    )
    legacy_transcript_import.apply_legacy_import(
        planned,
        root=store_root,
        embedding_provider="debug-hash",
        embedding_model="debug-hash",
    )

    planned_again = legacy_transcript_import.plan_legacy_import(
        [second],
        root=store_root,
        embedding_provider="debug-hash",
        embedding_model="debug-hash",
    )

    assert planned_again[0]["status"] == "duplicate_existing"
    assert "same normalized title" in planned_again[0]["reason"]


def test_legacy_import_dedupes_batch_titles(tmp_path: Path) -> None:
    first = tmp_path / "Team Meeting Transcript.txt"
    first.write_text("First transcript", encoding="utf-8")
    second = tmp_path / "nested"
    second.mkdir()
    duplicate = second / "Team Meeting Transcript.docx"
    write_docx(duplicate, ["Different file type"])

    planned = legacy_transcript_import.plan_legacy_import(
        [tmp_path],
        recursive=True,
        root=tmp_path / "store",
        embedding_provider="debug-hash",
        embedding_model="debug-hash",
    )
    statuses = {item["source_path"]: item["status"] for item in planned}

    assert statuses[str(first.resolve())] == "convert"
    assert statuses[str(duplicate.resolve())] == "duplicate_in_batch"
