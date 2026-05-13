from __future__ import annotations

import importlib.util
import io
import json
import subprocess
import sys
from pathlib import Path


def load_apply_module():
    scripts_dir = Path(__file__).resolve().parents[1] / "scripts"
    sys.path.insert(0, str(scripts_dir))
    try:
        module_path = scripts_dir / "context_packet_apply.py"
        spec = importlib.util.spec_from_file_location("context_packet_apply", module_path)
        assert spec is not None
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.remove(str(scripts_dir))


def context_packet() -> dict:
    return {
        "query": "hamburg sample",
        "selected_rank": 1,
        "context": {
            "document": {
                "id": "doc-1",
                "title": "Meeting Transcript",
                "source_path": "/tmp/meeting.transcript.json",
            },
            "chunk": {"chunk_index": 5},
            "media": {"start_timestamp": "08:02.96"},
        },
    }


def test_context_packet_apply_previews_without_running(capsys) -> None:
    apply = load_apply_module()
    calls = []

    def fake_runner(*args, **kwargs):
        calls.append(args)
        return subprocess.CompletedProcess(args[0], 0, "", "")

    assert apply.main(["-", "--store", "--with-provenance"], stdin=io.StringIO(json.dumps(context_packet())), runner=fake_runner) == 0

    stdout = capsys.readouterr().out
    assert "Mode: preview" in stdout
    assert "summarize_transcript.py /tmp/meeting.transcript.json --provider codex-exec --store" in stdout
    assert "--gws-provenance --graphiti-provenance --odollo-provenance" in stdout
    assert calls == []


def test_context_packet_apply_json_preview_uses_existing_paths(capsys) -> None:
    apply = load_apply_module()

    assert apply.main(
        [
            "-",
            "--readout",
            "/tmp/meeting.readout.json",
            "--route",
            "/tmp/meeting.route.json",
            "--provider-timeout",
            "600",
            "--format",
            "json",
        ],
        stdin=io.StringIO(json.dumps(context_packet())),
    ) == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload["mode"] == "preview"
    assert payload["steps"][0]["status"] == "skipped_existing"
    assert payload["steps"][0]["path"] == "/tmp/meeting.readout.json"
    assert payload["steps"][1]["status"] == "skipped_existing"
    assert payload["steps"][1]["path"] == "/tmp/meeting.route.json"
    assert payload["steps"][2]["argv"][-2:] == ["--timeout", "600.0"]


def test_context_packet_apply_executes_steps_and_writes_manifest_with_fake_runner(tmp_path: Path, capsys) -> None:
    apply = load_apply_module()
    calls = []
    manifest_dir = tmp_path / "manifests"

    def fake_runner(argv, **kwargs):
        calls.append(argv)
        if "summarize_transcript.py" in argv:
            return subprocess.CompletedProcess(argv, 0, "READOUT_JSON=/tmp/generated.readout.json\n", "")
        if "route_transcript.py" in argv:
            return subprocess.CompletedProcess(argv, 0, "ROUTE_DECISION_JSON=/tmp/generated.route.json\n", "")
        if "contextual_reread.py" in argv:
            return subprocess.CompletedProcess(argv, 0, "CONTEXTUAL_READOUT_JSON=/tmp/generated.contextual.readout.json\n", "")
        return subprocess.CompletedProcess(argv, 1, "", "unexpected")

    assert apply.main(
        ["-", "--apply", "--format", "json", "--manifest-dir", str(manifest_dir)],
        stdin=io.StringIO(json.dumps(context_packet())),
        runner=fake_runner,
    ) == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "completed"
    assert payload["readout"] == "/tmp/generated.readout.json"
    assert payload["route"] == "/tmp/generated.route.json"
    assert payload["contextual_readout"] == "/tmp/generated.contextual.readout.json"
    assert payload["steps"][2]["path"] == "/tmp/generated.contextual.readout.json"
    assert calls[1][3] == "/tmp/generated.readout.json"
    assert calls[2][3] == "/tmp/generated.readout.json"
    assert calls[2][4] == "/tmp/generated.route.json"
    manifest_path = Path(payload["manifest_path"])
    assert manifest_path.parent == manifest_dir
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["schema_version"] == 1
    assert manifest["artifacts"]["readout"] == "/tmp/generated.readout.json"
    assert manifest["artifacts"]["route"] == "/tmp/generated.route.json"
    assert manifest["artifacts"]["contextual_readout"] == "/tmp/generated.contextual.readout.json"
    assert "stdout" not in manifest["steps"][0]


def test_context_packet_apply_lists_recent_manifests(tmp_path: Path, capsys) -> None:
    apply = load_apply_module()
    manifest_dir = tmp_path / "manifests"
    manifest_dir.mkdir()
    older = manifest_dir / "2026-05-12T10-00-00Z-old.json"
    newer = manifest_dir / "2026-05-12T11-00-00Z-new.json"
    older.write_text(
        json.dumps(
            {
                "run_id": "old",
                "created_at": "2026-05-12T10:00:00Z",
                "status": "completed",
                "transcript": "/tmp/old.transcript.json",
                "artifacts": {"contextual_readout": "/tmp/old.contextual.readout.json"},
            }
        ),
        encoding="utf-8",
    )
    newer.write_text(
        json.dumps(
            {
                "run_id": "new",
                "created_at": "2026-05-12T11:00:00Z",
                "status": "completed",
                "transcript": "/tmp/new.transcript.json",
                "query": "new query",
                "document_id": "doc-new",
                "chunk_index": 7,
                "artifacts": {
                    "readout": "/tmp/new.readout.json",
                    "route": "/tmp/new.route.json",
                    "contextual_readout": "/tmp/new.contextual.readout.json",
                },
            }
        ),
        encoding="utf-8",
    )

    assert apply.main(
        ["--list-manifests", "--manifest-dir", str(manifest_dir), "--limit", "1", "--format", "json"]
    ) == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload["count"] == 1
    assert payload["manifests"][0]["run_id"] == "new"
    assert payload["manifests"][0]["contextual_readout"] == "/tmp/new.contextual.readout.json"
