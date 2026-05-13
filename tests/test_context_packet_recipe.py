from __future__ import annotations

import importlib.util
import io
import json
from pathlib import Path


def load_recipe_module():
    module_path = Path(__file__).resolve().parents[1] / "scripts" / "context_packet_recipe.py"
    spec = importlib.util.spec_from_file_location("context_packet_recipe", module_path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def context_packet() -> dict:
    return {
        "query": "rare catalyst procurement",
        "result_count": 2,
        "selected_rank": 1,
        "selected_result": {"id": "doc-1", "best_chunk": {"chunk_index": 4}},
        "context": {
            "document": {
                "id": "doc-1",
                "title": "Meeting Transcript",
                "source_path": "/tmp/meeting transcript.transcript.json",
            },
            "chunk": {"chunk_index": 4},
            "media": {
                "start_timestamp": "08:02.96",
                "end_timestamp": "10:04.86",
                "seek_hint": "ffplay -ss 482.96 '/tmp/meeting.m4a'",
            },
        },
    }


def test_context_packet_recipe_reads_stdin_and_prints_commands(capsys) -> None:
    recipe = load_recipe_module()
    stdin = io.StringIO(json.dumps(context_packet()))

    assert recipe.main(["-", "--provider", "codex-exec", "--store", "--with-provenance"], stdin=stdin) == 0

    stdout = capsys.readouterr().out
    assert "# Query: rare catalyst procurement" in stdout
    assert "python summarize_transcript.py '/tmp/meeting transcript.transcript.json' --provider codex-exec --store" in stdout
    assert "python route_transcript.py '/tmp/meeting transcript.transcript.json' '<READOUT_JSON_FROM_PREVIOUS_COMMAND>' --gws-provenance --graphiti-provenance --odollo-provenance" in stdout
    assert "python contextual_reread.py '/tmp/meeting transcript.transcript.json' '<READOUT_JSON_FROM_PREVIOUS_COMMAND>' '<ROUTE_DECISION_JSON_FROM_PREVIOUS_COMMAND>' --provider codex-exec --store" in stdout


def test_context_packet_recipe_accepts_explicit_readout_and_route(tmp_path: Path, capsys) -> None:
    recipe = load_recipe_module()
    packet_path = tmp_path / "context.json"
    packet_path.write_text(json.dumps(context_packet()), encoding="utf-8")

    assert recipe.main(
        [
            str(packet_path),
            "--readout",
            "/tmp/meeting.readout.json",
            "--route",
            "/tmp/meeting.route.json",
            "--model",
            "gpt-test",
        ]
    ) == 0

    stdout = capsys.readouterr().out
    assert "READOUT=/tmp/meeting.readout.json" in stdout
    assert "ROUTE=/tmp/meeting.route.json" in stdout
    assert "--model gpt-test" in stdout
