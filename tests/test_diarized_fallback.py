import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import assembly_transcribe as assembly
from parakeet_transcribe import diarized_words
from watch_transcriptions import backend_script_path, parse_backends


def test_explicit_key_cannot_fall_through_to_other_account(monkeypatch):
    args = assembly.parse_args(["test.wav", "--api-key-env", "GROUP_KEY"])
    monkeypatch.setenv("ASSEMBLYAI_API_KEY", "wrong-account")
    monkeypatch.delenv("GROUP_KEY", raising=False)
    with pytest.raises(assembly.TranscriptionError, match="GROUP_KEY"):
        assembly.resolve_api_key(args)
    monkeypatch.setenv("GROUP_KEY", "selected-account")
    assert assembly.resolve_api_key(args) == ("selected-account", "GROUP_KEY")


def test_speaker_overlap_and_gaps():
    words = [
        {"word": "Hello", "start": 0.1, "end": 0.5},
        {"word": "there.", "start": 0.5, "end": 1},
        {"word": "Reply.", "start": 1.1, "end": 2},
        {"word": "Gap.", "start": 4, "end": 5},
    ]
    turns = [
        {"start": 0, "end": 1, "speaker": "A"},
        {"start": 1, "end": 2.5, "speaker": "B"},
    ]
    result = diarized_words(words, turns)
    assert [(u["speaker"], u["text"]) for u in result] == [
        ("A", "Hello there."),
        ("B", "Reply."),
        ("UNKNOWN", "Gap."),
    ]
    assert result[1]["start"] == 1100


def test_no_diarization_fails_instead_of_publishing_plain_text():
    with pytest.raises(ValueError):
        diarized_words([{"word": "hello", "start": 0, "end": 1}], [])


def test_watcher_routes_to_parakeet():
    assert parse_backends({"backends": ["assembly", "parakeet"]}) == [
        "assembly",
        "parakeet",
    ]
    assert backend_script_path("parakeet").name == "parakeet_transcribe.py"
