from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

import speaker_identity_plan0071_d2_predictions as predictions


class FakeRunner:
    def __init__(self, *, repair_discovery: bool = False) -> None:
        self.repair_discovery = repair_discovery
        self.prepare_evaluation_calls = 0
        self.paths: list[str] = []

    def _post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        self.paths.append(path)
        if path.endswith("prepare-discovery"):
            return {
                "run_id": "discovery-run",
                "prompt_packet": {"route": {"provider": predictions.PRIMARY_PROVIDER}},
            }
        if path.endswith("prepare-evaluation"):
            self.prepare_evaluation_calls += 1
            if self.repair_discovery and self.prepare_evaluation_calls == 1:
                raise ValueError("invalid prepared clue reference")
            return {
                "run_id": "evaluation-run",
                "prompt_packet": {"route": {"provider": predictions.PRIMARY_PROVIDER}},
                "packet": {"people": [{"person_id": "person-1"}]},
            }
        raise AssertionError(path)

    def _execute_prepared(self, prepared: dict[str, Any]) -> dict[str, Any]:
        return {"output_text": "{}"}

    def _captured_json(self, status: dict[str, Any]) -> dict[str, Any]:
        return {}

    def _execute_reference_repair(
        self,
        document_id: str,
        *,
        phase: str,
        original: dict[str, Any],
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        return {"run_id": f"{phase}-repair"}, {}


def test_context_case_validates_locally_and_never_calls_capture(monkeypatch) -> None:
    monkeypatch.setattr(
        predictions.speaker_identity_preprocess,
        "validate_and_score_identity_evaluation",
        lambda packet, readout: {"readout": {"evaluation_id": "evaluation-1"}},
    )
    monkeypatch.setattr(
        predictions.plan0064_p2,
        "_successful_case",
        lambda **kwargs: {
            "schema_version": "prior",
            "status": "context_workflow_complete",
            "document_id": kwargs["document_id"],
            "speaker_slots": [],
            "content_sha256": "discarded",
        },
    )
    runner = FakeRunner()

    result = predictions.execute_context_case(
        runner, document_id="document-1", speaker_labels=["A", "B", "C"]
    )

    assert result["status"] == "context_workflow_complete"
    assert result["phase_turn_attempts"] == {
        "clue_discovery": 1,
        "identity_evaluation": 1,
    }
    assert result["capture_evaluation_call_count"] == 0
    assert not any("capture-evaluation" in path for path in runner.paths)
    assert result["mutation_effect_counts"] == predictions.MUTATION_EFFECT_COUNTS


def test_context_case_allows_one_discovery_reference_repair(monkeypatch) -> None:
    monkeypatch.setattr(
        predictions.speaker_identity_preprocess,
        "validate_and_score_identity_evaluation",
        lambda packet, readout: {"readout": {"evaluation_id": "evaluation-1"}},
    )
    monkeypatch.setattr(
        predictions.plan0064_p2,
        "_successful_case",
        lambda **kwargs: {
            "status": "context_workflow_complete",
            "document_id": kwargs["document_id"],
            "speaker_slots": [],
        },
    )
    runner = FakeRunner(repair_discovery=True)

    result = predictions.execute_context_case(
        runner, document_id="document-1", speaker_labels=["A", "B", "C"]
    )

    assert result["phase_turn_attempts"]["clue_discovery"] == 2
    assert result["reference_repair_counts"]["clue_discovery"] == 1
    assert result["fallback_model_turn_count"] == 0
    assert result["capture_evaluation_call_count"] == 0


def test_primary_route_is_required() -> None:
    runner = FakeRunner()
    original = runner._post

    def wrong_route(path: str, payload: dict[str, Any]) -> dict[str, Any]:
        value = original(path, payload)
        value["prompt_packet"]["route"]["provider"] = "other"
        return value

    runner._post = wrong_route  # type: ignore[method-assign]

    result = predictions.execute_context_case(
        runner, document_id="document-1", speaker_labels=["A", "B", "C"]
    )

    assert result["status"] == "context_workflow_unavailable"
    assert result["phase_turn_attempts"] == {
        "clue_discovery": 0,
        "identity_evaluation": 0,
    }
    assert result["capture_evaluation_call_count"] == 0


def test_private_store_redirects_normalization_away_from_live_sources(
    tmp_path: Path,
) -> None:
    tmp_path.chmod(0o700)
    source_root = tmp_path / "source"
    source_root.mkdir(mode=0o700)
    source_database = source_root / "transcripts.sqlite3"
    transcript = source_root / "source.transcript.json"
    media = source_root / "source.m4a"
    transcript.write_text(json.dumps({"utterances": []}), encoding="utf-8")
    media.write_bytes(b"media")
    transcript.chmod(0o600)
    media.chmod(0o600)
    with sqlite3.connect(source_database) as connection:
        connection.executescript(
            """
            CREATE TABLE documents (
                id TEXT PRIMARY KEY,
                source_path TEXT NOT NULL,
                stored_path TEXT NOT NULL
            );
            CREATE TABLE blobs (sha256 TEXT PRIMARY KEY, stored_path TEXT NOT NULL);
            """
        )
        connection.execute(
            "INSERT INTO documents VALUES (?, ?, ?)",
            ("document-1", str(transcript), str(transcript)),
        )
        connection.execute(
            "INSERT INTO blobs VALUES (?, ?)", ("media-hash", str(media))
        )
    private_root = tmp_path / "private"
    private_root.mkdir(mode=0o700)
    destination = private_root / "store" / "transcripts.sqlite3"
    selected = [
        {
            "document_id": "document-1",
            "transcript_artifact": {"path": str(transcript)},
            "source_media_artifact": {"path": str(media)},
            "transcript_sha256": predictions.sha256_file(transcript),
            "source_media_sha256": "media-hash",
        }
    ]

    predictions._prepare_private_store(source_database, destination, selected)

    with sqlite3.connect(destination) as connection:
        private_transcript = Path(
            connection.execute(
                "SELECT stored_path FROM documents WHERE id = 'document-1'"
            ).fetchone()[0]
        )
        private_source = Path(
            connection.execute(
                "SELECT source_path FROM documents WHERE id = 'document-1'"
            ).fetchone()[0]
        )
        private_media = Path(
            connection.execute(
                "SELECT stored_path FROM blobs WHERE sha256 = 'media-hash'"
            ).fetchone()[0]
        )
    assert private_transcript != transcript
    assert private_source == private_transcript
    assert private_media != media
    assert private_transcript.read_bytes() == transcript.read_bytes()
    assert private_media.read_bytes() == media.read_bytes()
    private_transcript.write_text("normalized-private-copy", encoding="utf-8")
    assert json.loads(transcript.read_text(encoding="utf-8")) == {"utterances": []}
